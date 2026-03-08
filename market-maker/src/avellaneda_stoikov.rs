//! Avellaneda-Stoikov optimal market making model.
//!
//! Implements the seminal framework from "High-frequency trading in a limit
//! order book" (Avellaneda & Stoikov, 2008). The model computes:
//!
//! 1. **Reservation price**: the fair value adjusted for inventory risk
//!    `r(s, q, t) = s - q * gamma * sigma^2 * (T - t)`
//!
//! 2. **Optimal spread**: balancing profit capture vs adverse selection
//!    `delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)`
//!
//! 3. **Order arrival intensity**: Poisson process with exponential decay
//!    `lambda(delta) = A * exp(-kappa * delta)`

use orderbook_core::types::{Price, Quantity, Side};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for the Avellaneda-Stoikov model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASConfig {
    /// Risk aversion parameter (gamma). Higher = more conservative.
    /// Typical range: 0.01 - 1.0
    pub gamma: f64,

    /// Order arrival intensity decay (kappa). Estimated from fill data.
    /// Typical range: 1.0 - 100.0
    pub kappa: f64,

    /// Order arrival baseline intensity (A).
    /// lambda(delta) = A * exp(-kappa * delta)
    pub intensity_a: f64,

    /// Trading session duration in seconds.
    pub session_duration_secs: f64,

    /// Volatility estimation lookback window (number of ticks).
    pub vol_lookback: usize,

    /// EWMA decay factor for volatility estimation (0 < alpha < 1).
    /// Higher alpha = more reactive to recent data.
    pub vol_alpha: f64,

    /// Minimum allowed spread (in price units, as f64).
    pub min_spread: f64,

    /// Maximum allowed spread.
    pub max_spread: f64,

    /// Tick size for price rounding.
    pub tick_size: f64,
}

impl Default for ASConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            kappa: 1.5,
            intensity_a: 140.0,
            session_duration_secs: 86400.0, // 24 hours for crypto
            vol_lookback: 100,
            vol_alpha: 0.06,
            min_spread: 0.0001,
            max_spread: 0.05,
            tick_size: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Volatility estimator
// ---------------------------------------------------------------------------

/// Exponentially-weighted realized volatility estimator.
///
/// Computes variance as EWMA of squared returns:
/// `var(t) = alpha * r(t)^2 + (1 - alpha) * var(t-1)`
///
/// This is more responsive to regime changes than simple window variance.
pub struct VolatilityEstimator {
    alpha: f64,
    ewma_variance: f64,
    last_price: Option<f64>,
    /// Number of observations seen.
    count: u64,
}

impl VolatilityEstimator {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ewma_variance: 0.0,
            last_price: None,
            count: 0,
        }
    }

    /// Update with a new mid-price observation.
    pub fn update(&mut self, mid_price: f64) {
        if let Some(last) = self.last_price {
            if last > 0.0 {
                let log_return = (mid_price / last).ln();
                let sq_return = log_return * log_return;

                if self.count < 2 {
                    self.ewma_variance = sq_return;
                } else {
                    self.ewma_variance = self.alpha * sq_return
                        + (1.0 - self.alpha) * self.ewma_variance;
                }
            }
        }

        self.last_price = Some(mid_price);
        self.count += 1;
    }

    /// Get the current volatility estimate (sigma).
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.ewma_variance.sqrt()
    }

    /// Get the current variance estimate.
    #[inline]
    pub fn variance(&self) -> f64 {
        self.ewma_variance
    }

    /// Check if we have enough data for a meaningful estimate.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.count >= 10
    }
}

// ---------------------------------------------------------------------------
// Kappa calibration
// ---------------------------------------------------------------------------

/// Calibrate kappa from historical fill data.
///
/// The order arrival intensity follows:
///   lambda(delta) = A * exp(-kappa * delta)
///
/// Given a set of (spread, fill_count) observations, we fit kappa via
/// maximum likelihood estimation (log-linear regression).
pub struct KappaCalibrator {
    /// (spread, number_of_fills) observations.
    observations: Vec<(f64, f64)>,
}

impl KappaCalibrator {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }

    /// Record an observation: at spread `delta`, we observed `n_fills` fills
    /// over a time window.
    pub fn add_observation(&mut self, delta: f64, n_fills: f64) {
        if delta > 0.0 && n_fills > 0.0 {
            self.observations.push((delta, n_fills));
        }
    }

    /// Fit the model and return (A, kappa).
    ///
    /// Uses log-linear regression:
    ///   ln(lambda) = ln(A) - kappa * delta
    ///
    /// Returns None if insufficient data.
    pub fn calibrate(&self) -> Option<(f64, f64)> {
        let n = self.observations.len();
        if n < 5 {
            return None;
        }

        // Log-linear regression: y = ln(fills), x = spread
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for &(delta, fills) in &self.observations {
            let y = fills.ln();
            sum_x += delta;
            sum_y += y;
            sum_xy += delta * y;
            sum_xx += delta * delta;
        }

        let n_f = n as f64;
        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return None;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        let kappa = -slope; // kappa is positive (slope is negative)
        let a = intercept.exp();

        if kappa > 0.0 && a > 0.0 {
            Some((a, kappa))
        } else {
            None
        }
    }

    /// Clear observations for fresh calibration.
    pub fn reset(&mut self) {
        self.observations.clear();
    }
}

// ---------------------------------------------------------------------------
// Core Avellaneda-Stoikov model
// ---------------------------------------------------------------------------

/// The Avellaneda-Stoikov market making model.
pub struct AvellanedaStoikov {
    pub config: ASConfig,
    vol_estimator: VolatilityEstimator,
    calibrator: KappaCalibrator,

    /// Current elapsed time in the session [0, T].
    current_time: f64,

    /// Current inventory position (positive = long, negative = short).
    inventory: f64,
}

impl AvellanedaStoikov {
    pub fn new(config: ASConfig) -> Self {
        let alpha = config.vol_alpha;
        Self {
            config,
            vol_estimator: VolatilityEstimator::new(alpha),
            calibrator: KappaCalibrator::new(),
            current_time: 0.0,
            inventory: 0.0,
        }
    }

    /// Update the model with a new mid-price tick.
    pub fn on_tick(&mut self, mid_price: f64, elapsed_secs: f64) {
        self.vol_estimator.update(mid_price);
        self.current_time = elapsed_secs;
    }

    /// Update inventory position.
    pub fn set_inventory(&mut self, qty: f64) {
        self.inventory = qty;
    }

    /// Time remaining in the session.
    #[inline]
    fn tau(&self) -> f64 {
        (self.config.session_duration_secs - self.current_time).max(0.001)
    }

    /// Compute the reservation (indifference) price.
    ///
    /// `r = s - q * gamma * sigma^2 * tau`
    ///
    /// When inventory is positive (long), the reservation price is below
    /// the mid -- we're willing to sell cheaper to reduce risk.
    /// When inventory is negative (short), it's above the mid.
    #[inline]
    pub fn reservation_price(&self, mid: f64) -> f64 {
        let sigma = self.vol_estimator.sigma();
        let gamma = self.config.gamma;
        let tau = self.tau();
        let q = self.inventory;

        mid - q * gamma * sigma * sigma * tau
    }

    /// Compute the optimal spread around the reservation price.
    ///
    /// `delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)`
    ///
    /// The first term is the inventory risk premium.
    /// The second term is the profit from adverse selection.
    #[inline]
    pub fn optimal_spread(&self) -> f64 {
        let sigma = self.vol_estimator.sigma();
        let gamma = self.config.gamma;
        let kappa = self.config.kappa;
        let tau = self.tau();

        let risk_premium = gamma * sigma * sigma * tau;
        let adverse_selection = (2.0 / gamma) * (1.0 + gamma / kappa).ln();

        let spread = risk_premium + adverse_selection;

        // Clamp to configured bounds
        spread.clamp(self.config.min_spread, self.config.max_spread)
    }

    /// Compute optimal bid and ask quotes.
    ///
    /// Returns `(bid_price, ask_price)` as f64.
    pub fn compute_quotes(&self, mid: f64) -> (f64, f64) {
        let reservation = self.reservation_price(mid);
        let half_spread = self.optimal_spread() / 2.0;

        let bid = reservation - half_spread;
        let ask = reservation + half_spread;

        // Round to tick size
        let tick = self.config.tick_size;
        let bid_rounded = (bid / tick).floor() * tick;
        let ask_rounded = (ask / tick).ceil() * tick;

        (bid_rounded, ask_rounded)
    }

    /// Compute quotes as fixed-point Prices.
    pub fn compute_price_quotes(&self, mid: f64) -> (Price, Price) {
        let (bid, ask) = self.compute_quotes(mid);
        (Price::from_f64(bid), Price::from_f64(ask))
    }

    /// Compute order arrival intensity at a given spread.
    ///
    /// `lambda(delta) = A * exp(-kappa * delta)`
    #[inline]
    pub fn arrival_intensity(&self, spread: f64) -> f64 {
        self.config.intensity_a * (-self.config.kappa * spread).exp()
    }

    /// Expected fill probability for a quote at distance `delta` from mid,
    /// over time window `dt` seconds.
    ///
    /// `P(fill) = 1 - exp(-lambda(delta) * dt)`
    #[inline]
    pub fn fill_probability(&self, delta: f64, dt: f64) -> f64 {
        let lambda = self.arrival_intensity(delta);
        1.0 - (-lambda * dt).exp()
    }

    /// Get current volatility estimate.
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.vol_estimator.sigma()
    }

    /// Check if the model is warmed up (enough volatility data).
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.vol_estimator.is_ready()
    }

    /// Re-calibrate kappa from accumulated fill data.
    pub fn recalibrate_kappa(&mut self) {
        if let Some((a, kappa)) = self.calibrator.calibrate() {
            tracing::info!(
                "Kappa recalibrated: A={:.4}, kappa={:.4} (was {:.4})",
                a,
                kappa,
                self.config.kappa
            );
            self.config.kappa = kappa;
            self.config.intensity_a = a;
            self.calibrator.reset();
        }
    }

    /// Record a fill observation for kappa calibration.
    pub fn record_fill(&mut self, spread_at_fill: f64, fills_in_window: f64) {
        self.calibrator
            .add_observation(spread_at_fill, fills_in_window);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> AvellanedaStoikov {
        let config = ASConfig {
            gamma: 0.1,
            kappa: 1.5,
            intensity_a: 140.0,
            session_duration_secs: 3600.0,
            vol_lookback: 100,
            vol_alpha: 0.1,
            min_spread: 0.01,
            max_spread: 100.0,
            tick_size: 0.01,
        };
        AvellanedaStoikov::new(config)
    }

    fn warm_up(model: &mut AvellanedaStoikov) {
        // Feed enough ticks for volatility estimation
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.01).sin() * 0.5;
            model.on_tick(price, i as f64 * 0.1);
        }
    }

    #[test]
    fn test_reservation_price_neutral_inventory() {
        let mut model = make_model();
        warm_up(&mut model);
        model.set_inventory(0.0);

        let mid = 100.0;
        let reservation = model.reservation_price(mid);

        // With zero inventory, reservation == mid
        assert!((reservation - mid).abs() < 1e-6);
    }

    #[test]
    fn test_reservation_price_long_inventory() {
        let mut model = make_model();
        warm_up(&mut model);
        model.set_inventory(10.0); // Long 10 units

        let mid = 100.0;
        let reservation = model.reservation_price(mid);

        // Long inventory -> reservation below mid (willing to sell cheaper)
        assert!(reservation < mid);
    }

    #[test]
    fn test_reservation_price_short_inventory() {
        let mut model = make_model();
        warm_up(&mut model);
        model.set_inventory(-10.0); // Short 10 units

        let mid = 100.0;
        let reservation = model.reservation_price(mid);

        // Short inventory -> reservation above mid (willing to buy higher)
        assert!(reservation > mid);
    }

    #[test]
    fn test_optimal_spread_positive() {
        let mut model = make_model();
        warm_up(&mut model);

        let spread = model.optimal_spread();
        assert!(spread > 0.0);
    }

    #[test]
    fn test_quotes_bracket_reservation() {
        let mut model = make_model();
        warm_up(&mut model);
        model.set_inventory(0.0);

        let mid = 100.0;
        let (bid, ask) = model.compute_quotes(mid);

        assert!(bid < mid);
        assert!(ask > mid);
        assert!(ask > bid);
    }

    #[test]
    fn test_arrival_intensity_decreases_with_spread() {
        let model = make_model();

        let lambda_tight = model.arrival_intensity(0.01);
        let lambda_wide = model.arrival_intensity(0.10);

        assert!(lambda_tight > lambda_wide);
    }

    #[test]
    fn test_kappa_calibration() {
        let mut cal = KappaCalibrator::new();

        // Simulate exponential decay: lambda = 100 * exp(-2 * delta)
        let true_a = 100.0;
        let true_kappa = 2.0;

        for i in 1..20 {
            let delta = i as f64 * 0.05;
            let fills = true_a * (-true_kappa * delta).exp();
            // Add some noise
            cal.add_observation(delta, fills * (1.0 + 0.01 * (i as f64)));
        }

        let (a, kappa) = cal.calibrate().unwrap();
        // Should be approximately correct
        assert!((kappa - true_kappa).abs() < 1.0);
        assert!(a > 0.0);
    }

    #[test]
    fn test_volatility_estimator() {
        let mut est = VolatilityEstimator::new(0.1);

        // Feed constant prices -> zero volatility
        for _ in 0..20 {
            est.update(100.0);
        }
        assert!(est.sigma() < 1e-10);

        // Feed oscillating prices -> nonzero volatility
        let mut est2 = VolatilityEstimator::new(0.1);
        for i in 0..50 {
            let price = 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 };
            est2.update(price);
        }
        assert!(est2.sigma() > 0.0);
    }
}
