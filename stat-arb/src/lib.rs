//! # stat-arb
//!
//! Statistical arbitrage engine with Kalman filter and cointegration analysis.
//!
//! ## Components
//!
//! - **`KalmanSpreadTracker`** -- Online Kalman filter for dynamic hedge ratio estimation
//! - **`CointegrationAnalyzer`** -- Engle-Granger cointegration test
//! - **`StatArbStrategy`** -- Mean-reversion signal generation with z-score thresholds
//! - **`PairState`** -- Tracks a single pair's spread, z-score, and position

use nalgebra::{Matrix2, Vector2};
use orderbook_core::types::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Kalman Filter for dynamic hedge ratio
// ---------------------------------------------------------------------------

/// Online Kalman filter for estimating the time-varying hedge ratio
/// between two cointegrated assets.
///
/// State: [alpha, beta] where spread = price_y - beta * price_x - alpha
/// Observation: price_y = alpha + beta * price_x + noise
pub struct KalmanSpreadTracker {
    /// State estimate: [intercept, slope/hedge_ratio]
    state: Vector2<f64>,
    /// State covariance matrix (2x2)
    covariance: Matrix2<f64>,
    /// Process noise covariance (controls adaptation speed)
    process_noise: Matrix2<f64>,
    /// Observation noise variance
    obs_noise: f64,
    /// Number of observations processed
    n_obs: u64,
}

impl KalmanSpreadTracker {
    /// Create a new Kalman tracker.
    ///
    /// - `delta`: Process noise scaling (higher = faster adaptation, noisier).
    ///   Typical values: 1e-5 to 1e-3.
    /// - `obs_noise`: Observation noise variance. Typical: 1.0.
    pub fn new(delta: f64, obs_noise: f64) -> Self {
        Self {
            state: Vector2::new(0.0, 1.0), // Initial: intercept=0, hedge_ratio=1
            covariance: Matrix2::identity() * 1.0, // High initial uncertainty
            process_noise: Matrix2::identity() * delta,
            obs_noise,
            n_obs: 0,
        }
    }

    /// Default parameters for crypto pairs.
    pub fn crypto_default() -> Self {
        Self::new(1e-4, 1.0)
    }

    /// Update the filter with a new observation pair.
    ///
    /// Returns (spread, spread_variance) for z-score computation.
    pub fn update(&mut self, price_x: f64, price_y: f64) -> (f64, f64) {
        self.n_obs += 1;

        // Observation vector: [1, price_x]
        let h = Vector2::new(1.0, price_x);

        // Predict step: add process noise to covariance
        let p_pred = self.covariance + self.process_noise;

        // Innovation (prediction error)
        let y_pred = h.dot(&self.state);
        let spread = price_y - y_pred;

        // Innovation variance
        let s = h.dot(&(p_pred * h)) + self.obs_noise;

        // Kalman gain
        let k = (p_pred * h) / s;

        // Update state
        self.state = self.state + k * spread;

        // Update covariance (Joseph form for numerical stability)
        let i_kh = Matrix2::identity() - k * h.transpose();
        self.covariance = i_kh * p_pred;

        (spread, s)
    }

    /// Get current hedge ratio (beta).
    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        self.state[1]
    }

    /// Get current intercept (alpha).
    #[inline]
    pub fn intercept(&self) -> f64 {
        self.state[0]
    }

    /// Get number of observations processed.
    #[inline]
    pub fn observation_count(&self) -> u64 {
        self.n_obs
    }
}

// ---------------------------------------------------------------------------
// Cointegration Analyzer (Engle-Granger)
// ---------------------------------------------------------------------------

/// Performs Engle-Granger cointegration analysis on a pair of price series.
///
/// Two series are cointegrated if their linear combination is stationary.
/// This is tested via OLS regression followed by an ADF test on residuals.
pub struct CointegrationAnalyzer {
    /// Minimum observations before testing.
    pub min_observations: usize,
    /// ADF critical value (5% significance, approx for large N).
    pub adf_critical_value: f64,
    /// Rolling window size for online analysis.
    pub window_size: usize,
    /// Stored price history for X.
    prices_x: Vec<f64>,
    /// Stored price history for Y.
    prices_y: Vec<f64>,
}

impl CointegrationAnalyzer {
    pub fn new(window_size: usize, adf_critical_value: f64) -> Self {
        Self {
            min_observations: 30,
            adf_critical_value,
            window_size,
            prices_x: Vec::with_capacity(window_size),
            prices_y: Vec::with_capacity(window_size),
        }
    }

    /// Default for crypto pair analysis.
    pub fn crypto_default() -> Self {
        Self::new(500, -2.86) // 5% critical value
    }

    /// Add a new price observation.
    pub fn add_observation(&mut self, price_x: f64, price_y: f64) {
        self.prices_x.push(price_x);
        self.prices_y.push(price_y);

        // Maintain rolling window
        if self.prices_x.len() > self.window_size {
            self.prices_x.remove(0);
            self.prices_y.remove(0);
        }
    }

    /// Run OLS regression: Y = alpha + beta * X + epsilon
    /// Returns (alpha, beta, residuals)
    pub fn ols_regression(&self) -> Option<(f64, f64, Vec<f64>)> {
        let n = self.prices_x.len();
        if n < self.min_observations {
            return None;
        }

        let n_f = n as f64;
        let sum_x: f64 = self.prices_x.iter().sum();
        let sum_y: f64 = self.prices_y.iter().sum();
        let sum_xy: f64 = self.prices_x.iter()
            .zip(self.prices_y.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_x2: f64 = self.prices_x.iter().map(|x| x * x).sum();

        let mean_x = sum_x / n_f;
        let mean_y = sum_y / n_f;

        let denom = sum_x2 - n_f * mean_x * mean_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        let beta = (sum_xy - n_f * mean_x * mean_y) / denom;
        let alpha = mean_y - beta * mean_x;

        let residuals: Vec<f64> = self.prices_x.iter()
            .zip(self.prices_y.iter())
            .map(|(x, y)| y - alpha - beta * x)
            .collect();

        Some((alpha, beta, residuals))
    }

    /// Simplified ADF test statistic on residuals.
    ///
    /// Tests H0: residuals have a unit root (not stationary)
    /// Returns the t-statistic. Compare against critical value.
    pub fn adf_test_statistic(residuals: &[f64]) -> Option<f64> {
        let n = residuals.len();
        if n < 10 {
            return None;
        }

        // Compute first differences and lagged values
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_e2 = 0.0;

        for i in 1..n {
            let dy = residuals[i] - residuals[i - 1];
            let y_lag = residuals[i - 1];
            sum_xy += y_lag * dy;
            sum_x2 += y_lag * y_lag;
        }

        if sum_x2.abs() < 1e-12 {
            return None;
        }

        let gamma = sum_xy / sum_x2;

        // Residual variance
        for i in 1..n {
            let dy = residuals[i] - residuals[i - 1];
            let y_lag = residuals[i - 1];
            let e = dy - gamma * y_lag;
            sum_e2 += e * e;
        }

        let sigma2 = sum_e2 / (n as f64 - 2.0);
        let se_gamma = (sigma2 / sum_x2).sqrt();

        if se_gamma.abs() < 1e-12 {
            return None;
        }

        Some(gamma / se_gamma)
    }

    /// Test if the pair is cointegrated.
    /// Returns Some((beta, adf_stat)) if enough data, None otherwise.
    pub fn test_cointegration(&self) -> Option<(f64, f64, bool)> {
        let (alpha, beta, residuals) = self.ols_regression()?;
        let adf_stat = Self::adf_test_statistic(&residuals)?;
        let is_cointegrated = adf_stat < self.adf_critical_value;

        debug!(
            "Cointegration test: alpha={:.4}, beta={:.4}, ADF={:.4}, cointegrated={}",
            alpha, beta, adf_stat, is_cointegrated
        );

        Some((beta, adf_stat, is_cointegrated))
    }

    /// Number of stored observations.
    pub fn observation_count(&self) -> usize {
        self.prices_x.len()
    }
}

// ---------------------------------------------------------------------------
// Pair State
// ---------------------------------------------------------------------------

/// Position direction for a stat-arb pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairPosition {
    /// No position.
    Flat,
    /// Long spread (long Y, short X).
    LongSpread,
    /// Short spread (short Y, long X).
    ShortSpread,
}

/// Tracks the state of a single stat-arb pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairState {
    /// Symbol for asset X.
    pub symbol_x: String,
    /// Symbol for asset Y.
    pub symbol_y: String,
    /// Current spread value.
    pub spread: f64,
    /// Current z-score.
    pub z_score: f64,
    /// Current hedge ratio.
    pub hedge_ratio: f64,
    /// Current position.
    pub position: PairPosition,
    /// Cumulative PnL.
    pub pnl: f64,
    /// Number of trades executed.
    pub trade_count: u64,
    /// Entry spread (when position was opened).
    pub entry_spread: f64,
}

impl PairState {
    pub fn new(symbol_x: &str, symbol_y: &str) -> Self {
        Self {
            symbol_x: symbol_x.to_string(),
            symbol_y: symbol_y.to_string(),
            spread: 0.0,
            z_score: 0.0,
            hedge_ratio: 1.0,
            position: PairPosition::Flat,
            pnl: 0.0,
            trade_count: 0,
            entry_spread: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Stat-Arb Strategy
// ---------------------------------------------------------------------------

/// Configuration for the stat-arb strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatArbConfig {
    /// Z-score threshold to enter a position.
    pub entry_z: f64,
    /// Z-score threshold to exit a position (mean reversion target).
    pub exit_z: f64,
    /// Z-score threshold for stop-loss.
    pub stop_z: f64,
    /// Kalman filter process noise (delta).
    pub kalman_delta: f64,
    /// Kalman filter observation noise.
    pub kalman_obs_noise: f64,
    /// Minimum observations before trading.
    pub warmup_period: u64,
    /// Rolling z-score window size.
    pub z_score_window: usize,
}

impl Default for StatArbConfig {
    fn default() -> Self {
        Self {
            entry_z: 2.0,
            exit_z: 0.5,
            stop_z: 4.0,
            kalman_delta: 1e-4,
            kalman_obs_noise: 1.0,
            warmup_period: 100,
            z_score_window: 100,
        }
    }
}

/// Statistical arbitrage strategy.
///
/// Monitors a pair of assets, estimates the dynamic hedge ratio via Kalman
/// filter, computes z-scores of the spread, and generates entry/exit signals.
pub struct StatArbStrategy {
    pub config: StatArbConfig,
    pub state: PairState,
    kalman: KalmanSpreadTracker,
    /// Rolling spread history for z-score computation.
    spread_history: Vec<f64>,
    /// Rolling mean of spreads.
    spread_mean: f64,
    /// Rolling std of spreads.
    spread_std: f64,
}

/// Signals emitted by the strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// No action.
    Hold,
    /// Enter long spread (long Y, short X).
    EnterLong,
    /// Enter short spread (short Y, long X).
    EnterShort,
    /// Exit current position.
    Exit,
    /// Stop-loss triggered.
    StopLoss,
}

impl StatArbStrategy {
    /// Create a new stat-arb strategy.
    pub fn new(config: StatArbConfig, symbol_x: &str, symbol_y: &str) -> Self {
        let kalman = KalmanSpreadTracker::new(
            config.kalman_delta,
            config.kalman_obs_noise,
        );

        Self {
            config: config.clone(),
            state: PairState::new(symbol_x, symbol_y),
            kalman,
            spread_history: Vec::with_capacity(config.z_score_window),
            spread_mean: 0.0,
            spread_std: 1.0,
        }
    }

    /// Process a new price tick and generate a signal.
    pub fn on_tick(&mut self, price_x: f64, price_y: f64) -> Signal {
        // Update Kalman filter
        let (spread, _variance) = self.kalman.update(price_x, price_y);

        // Update rolling spread statistics
        self.spread_history.push(spread);
        if self.spread_history.len() > self.config.z_score_window {
            self.spread_history.remove(0);
        }

        self.update_rolling_stats();

        // Compute z-score
        let z_score = if self.spread_std > 1e-10 {
            (spread - self.spread_mean) / self.spread_std
        } else {
            0.0
        };

        // Update state
        self.state.spread = spread;
        self.state.z_score = z_score;
        self.state.hedge_ratio = self.kalman.hedge_ratio();

        // Don't trade during warmup
        if self.kalman.observation_count() < self.config.warmup_period {
            return Signal::Hold;
        }

        // Generate signal based on z-score and current position
        self.generate_signal(z_score, spread)
    }

    /// Update rolling mean and std of spreads.
    fn update_rolling_stats(&mut self) {
        let n = self.spread_history.len();
        if n < 2 {
            return;
        }

        let n_f = n as f64;
        let mean: f64 = self.spread_history.iter().sum::<f64>() / n_f;
        let var: f64 = self.spread_history.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / (n_f - 1.0);

        self.spread_mean = mean;
        self.spread_std = var.sqrt();
    }

    /// Generate trading signal based on z-score.
    fn generate_signal(&mut self, z_score: f64, spread: f64) -> Signal {
        match self.state.position {
            PairPosition::Flat => {
                // Entry signals
                if z_score > self.config.entry_z {
                    // Spread is abnormally high -> expect reversion down
                    // Short spread: short Y, long X
                    self.state.position = PairPosition::ShortSpread;
                    self.state.entry_spread = spread;
                    self.state.trade_count += 1;
                    info!(
                        "[StatArb] ENTER SHORT spread: z={:.2}, spread={:.6}, hedge={:.4}",
                        z_score, spread, self.state.hedge_ratio
                    );
                    Signal::EnterShort
                } else if z_score < -self.config.entry_z {
                    // Spread is abnormally low -> expect reversion up
                    // Long spread: long Y, short X
                    self.state.position = PairPosition::LongSpread;
                    self.state.entry_spread = spread;
                    self.state.trade_count += 1;
                    info!(
                        "[StatArb] ENTER LONG spread: z={:.2}, spread={:.6}, hedge={:.4}",
                        z_score, spread, self.state.hedge_ratio
                    );
                    Signal::EnterLong
                } else {
                    Signal::Hold
                }
            }
            PairPosition::LongSpread => {
                // Stop loss
                if z_score < -self.config.stop_z {
                    let pnl = spread - self.state.entry_spread;
                    self.state.pnl += pnl;
                    self.state.position = PairPosition::Flat;
                    warn!(
                        "[StatArb] STOP LOSS (long): z={:.2}, pnl={:.6}",
                        z_score, pnl
                    );
                    Signal::StopLoss
                }
                // Exit on mean reversion
                else if z_score > -self.config.exit_z {
                    let pnl = spread - self.state.entry_spread;
                    self.state.pnl += pnl;
                    self.state.position = PairPosition::Flat;
                    info!(
                        "[StatArb] EXIT (long): z={:.2}, pnl={:.6}, total_pnl={:.6}",
                        z_score, pnl, self.state.pnl
                    );
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
            PairPosition::ShortSpread => {
                // Stop loss
                if z_score > self.config.stop_z {
                    let pnl = self.state.entry_spread - spread;
                    self.state.pnl += pnl;
                    self.state.position = PairPosition::Flat;
                    warn!(
                        "[StatArb] STOP LOSS (short): z={:.2}, pnl={:.6}",
                        z_score, pnl
                    );
                    Signal::StopLoss
                }
                // Exit on mean reversion
                else if z_score < self.config.exit_z {
                    let pnl = self.state.entry_spread - spread;
                    self.state.pnl += pnl;
                    self.state.position = PairPosition::Flat;
                    info!(
                        "[StatArb] EXIT (short): z={:.2}, pnl={:.6}, total_pnl={:.6}",
                        z_score, pnl, self.state.pnl
                    );
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
        }
    }

    /// Get current strategy state.
    pub fn get_state(&self) -> &PairState {
        &self.state
    }

    /// Get the Kalman tracker.
    pub fn kalman(&self) -> &KalmanSpreadTracker {
        &self.kalman
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kalman_tracker_converges() {
        let mut tracker = KalmanSpreadTracker::new(1e-3, 1.0);

        // Simulate: Y = 2.0 * X + 10.0 + noise
        let true_beta = 2.0;
        let true_alpha = 10.0;

        let mut rng = StdRng::seed_from_u64(42);
        use rand_distr::{Distribution, Normal};
        let noise = Normal::new(0.0, 0.1).unwrap();

        for i in 0..500 {
            let x = 100.0 + (i as f64) * 0.1;
            let y = true_alpha + true_beta * x + noise.sample(&mut rng);
            tracker.update(x, y);
        }

        // After enough observations, hedge ratio should converge near 2.0
        let estimated_beta = tracker.hedge_ratio();
        assert!(
            (estimated_beta - true_beta).abs() < 0.1,
            "Hedge ratio didn't converge: got {}, expected ~{}",
            estimated_beta,
            true_beta
        );
    }

    #[test]
    fn test_kalman_initial_state() {
        let tracker = KalmanSpreadTracker::crypto_default();
        assert_eq!(tracker.observation_count(), 0);
        assert!((tracker.hedge_ratio() - 1.0).abs() < 1e-10);
        assert!(tracker.intercept().abs() < 1e-10);
    }

    #[test]
    fn test_cointegration_analyzer_ols() {
        let mut analyzer = CointegrationAnalyzer::new(1000, -2.86);

        // Perfect linear relationship: Y = 0.5 * X + 5
        for i in 0..100 {
            let x = 50.0 + (i as f64) * 0.5;
            let y = 0.5 * x + 5.0;
            analyzer.add_observation(x, y);
        }

        let (alpha, beta, residuals) = analyzer.ols_regression().unwrap();
        assert!((beta - 0.5).abs() < 1e-6, "Beta: {}", beta);
        assert!((alpha - 5.0).abs() < 1e-4, "Alpha: {}", alpha);

        // Residuals should be near zero for perfect relationship
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
        assert!(max_residual < 1e-6, "Max residual: {}", max_residual);
    }

    #[test]
    fn test_cointegration_not_enough_data() {
        let analyzer = CointegrationAnalyzer::new(1000, -2.86);
        assert!(analyzer.ols_regression().is_none());
        assert!(analyzer.test_cointegration().is_none());
    }

    #[test]
    fn test_stat_arb_strategy_warmup() {
        let config = StatArbConfig {
            warmup_period: 50,
            ..Default::default()
        };
        let mut strategy = StatArbStrategy::new(config, "BTC", "ETH");

        // During warmup, should always return Hold
        for i in 0..49 {
            let signal = strategy.on_tick(100.0 + i as f64, 200.0 + i as f64 * 2.0);
            assert_eq!(signal, Signal::Hold);
        }
    }

    #[test]
    fn test_stat_arb_signal_generation() {
        let config = StatArbConfig {
            entry_z: 2.0,
            exit_z: 0.5,
            stop_z: 4.0,
            warmup_period: 10,
            z_score_window: 20,
            ..Default::default()
        };
        let mut strategy = StatArbStrategy::new(config, "BTC", "ETH");

        // Warmup with stable prices
        for i in 0..20 {
            let x = 100.0 + (i as f64) * 0.01;
            let y = 200.0 + (i as f64) * 0.02;
            strategy.on_tick(x, y);
        }

        // State should be flat after warmup with stable prices
        assert_eq!(strategy.state.position, PairPosition::Flat);
    }

    #[test]
    fn test_pair_state_creation() {
        let state = PairState::new("BTC-USD", "ETH-USD");
        assert_eq!(state.symbol_x, "BTC-USD");
        assert_eq!(state.symbol_y, "ETH-USD");
        assert_eq!(state.position, PairPosition::Flat);
        assert_eq!(state.trade_count, 0);
        assert!((state.pnl).abs() < 1e-10);
    }

    #[test]
    fn test_adf_stationary_series() {
        // A stationary series (mean-reverting) should have a negative ADF statistic
        let mut rng = StdRng::seed_from_u64(123);
        use rand_distr::{Distribution, Normal};
        let noise = Normal::new(0.0, 1.0).unwrap();

        let mut residuals = Vec::new();
        let mut val = 0.0;
        for _ in 0..200 {
            val = 0.5 * val + noise.sample(&mut rng); // AR(1) with phi=0.5
            residuals.push(val);
        }

        let adf = CointegrationAnalyzer::adf_test_statistic(&residuals);
        assert!(adf.is_some());
        // Stationary series should have a significantly negative ADF stat
        assert!(adf.unwrap() < -1.0, "ADF stat: {}", adf.unwrap());
    }

    #[test]
    fn test_rolling_window_maintenance() {
        let mut analyzer = CointegrationAnalyzer::new(50, -2.86);

        for i in 0..100 {
            analyzer.add_observation(i as f64, (i * 2) as f64);
        }

        // Should only keep last 50 observations
        assert_eq!(analyzer.observation_count(), 50);
    }
}
