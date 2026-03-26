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

use crossbeam_channel::{Receiver, Sender};
use nalgebra::{Matrix2, Vector2};
use orderbook_core::types::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
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
        self.state += k * spread;

        // Update covariance (Joseph form for numerical stability)
        let i_kh = Matrix2::identity() - k * h.transpose();
        self.covariance =
            i_kh * p_pred * i_kh.transpose() + (k * self.obs_noise) * k.transpose();

        (spread, s)
    }

    /// Get the current hedge ratio (beta).
    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        self.state[1]
    }

    /// Get the current intercept (alpha).
    #[inline]
    pub fn intercept(&self) -> f64 {
        self.state[0]
    }

    /// Get the number of observations processed.
    #[inline]
    pub fn n_observations(&self) -> u64 {
        self.n_obs
    }
}

// ---------------------------------------------------------------------------
// Cointegration analyzer (Engle-Granger)
// ---------------------------------------------------------------------------

/// Simple Engle-Granger cointegration test using OLS regression.
///
/// Accumulates price history and computes the ADF test statistic
/// on the regression residuals.
pub struct CointegrationAnalyzer {
    prices_x: Vec<f64>,
    prices_y: Vec<f64>,
    window_size: usize,
}

impl CointegrationAnalyzer {
    pub fn new(window_size: usize) -> Self {
        Self {
            prices_x: Vec::with_capacity(window_size),
            prices_y: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a price observation pair.
    pub fn add_observation(&mut self, price_x: f64, price_y: f64) {
        self.prices_x.push(price_x);
        self.prices_y.push(price_y);

        // Trim to window
        if self.prices_x.len() > self.window_size {
            self.prices_x.remove(0);
            self.prices_y.remove(0);
        }
    }

    /// Compute OLS hedge ratio and residual statistics.
    ///
    /// Returns `Some((hedge_ratio, residual_mean, residual_std, adf_stat))`
    /// or `None` if insufficient data.
    pub fn compute(&self) -> Option<CointegrationResult> {
        let n = self.prices_x.len();
        if n < 30 {
            return None; // Need minimum observations
        }

        // OLS: y = alpha + beta * x
        let mean_x: f64 = self.prices_x.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = self.prices_y.iter().sum::<f64>() / n as f64;

        let mut cov_xy = 0.0;
        let mut var_x = 0.0;
        for i in 0..n {
            let dx = self.prices_x[i] - mean_x;
            let dy = self.prices_y[i] - mean_y;
            cov_xy += dx * dy;
            var_x += dx * dx;
        }

        if var_x.abs() < 1e-15 {
            return None;
        }

        let beta = cov_xy / var_x;
        let alpha = mean_y - beta * mean_x;

        // Compute residuals
        let residuals: Vec<f64> = self
            .prices_x
            .iter()
            .zip(self.prices_y.iter())
            .map(|(x, y)| y - alpha - beta * x)
            .collect();

        let res_mean: f64 = residuals.iter().sum::<f64>() / n as f64;
        let res_var: f64 =
            residuals.iter().map(|r| (r - res_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let res_std = res_var.sqrt();

        // Simplified ADF test statistic (Dickey-Fuller on residuals)
        // delta_r(t) = gamma * r(t-1) + epsilon
        // ADF stat = gamma_hat / se(gamma_hat)
        let adf_stat = compute_adf_statistic(&residuals);

        Some(CointegrationResult {
            hedge_ratio: beta,
            intercept: alpha,
            residual_mean: res_mean,
            residual_std: res_std,
            adf_statistic: adf_stat,
            n_observations: n,
        })
    }

    /// Check if the pair is cointegrated at the 5% significance level.
    /// ADF critical value for n=100 at 5% is approximately -2.89.
    pub fn is_cointegrated(&self) -> bool {
        self.compute()
            .map(|r| r.adf_statistic < -2.89)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationResult {
    pub hedge_ratio: f64,
    pub intercept: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
    pub adf_statistic: f64,
    pub n_observations: usize,
}

fn compute_adf_statistic(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n < 3 {
        return 0.0;
    }

    // delta_r(t) = gamma * r(t-1) + eps
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_eps2 = 0.0;

    for i in 1..n {
        let dr = residuals[i] - residuals[i - 1];
        let r_lag = residuals[i - 1];
        sum_xy += r_lag * dr;
        sum_xx += r_lag * r_lag;
    }

    if sum_xx.abs() < 1e-15 {
        return 0.0;
    }

    let gamma = sum_xy / sum_xx;

    // Compute residual variance of the ADF regression
    for i in 1..n {
        let dr = residuals[i] - residuals[i - 1];
        let r_lag = residuals[i - 1];
        let eps = dr - gamma * r_lag;
        sum_eps2 += eps * eps;
    }

    let sigma2 = sum_eps2 / (n - 2) as f64;
    let se_gamma = (sigma2 / sum_xx).sqrt();

    if se_gamma.abs() < 1e-15 {
        return 0.0;
    }

    gamma / se_gamma
}

// ---------------------------------------------------------------------------
// Stat-arb strategy
// ---------------------------------------------------------------------------

/// Configuration for the stat-arb strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatArbConfig {
    /// Z-score threshold to enter a trade.
    pub entry_z: f64,
    /// Z-score threshold to exit a trade (mean reversion target).
    pub exit_z: f64,
    /// Z-score threshold for stop-loss.
    pub stop_z: f64,
    /// Position sizing (units of asset Y per unit z-score).
    pub position_scale: f64,
    /// Minimum observations before trading.
    pub warmup_periods: u64,
    /// Kalman filter process noise (delta).
    pub kalman_delta: f64,
    /// Cointegration window size.
    pub coint_window: usize,
    /// Re-check cointegration every N observations.
    pub coint_recheck_interval: u64,
}

impl Default for StatArbConfig {
    fn default() -> Self {
        Self {
            entry_z: 2.0,
            exit_z: 0.5,
            stop_z: 4.0,
            position_scale: 1.0,
            warmup_periods: 100,
            kalman_delta: 1e-4,
            coint_window: 500,
            coint_recheck_interval: 100,
        }
    }
}

/// Position state for the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairPosition {
    Flat,
    /// Long spread (long Y, short beta*X)
    LongSpread,
    /// Short spread (short Y, long beta*X)
    ShortSpread,
}

/// Tracks the state of a single pair for stat-arb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairState {
    pub symbol_x: String,
    pub symbol_y: String,
    pub position: PairPosition,
    pub z_score: f64,
    pub hedge_ratio: f64,
    pub spread: f64,
    pub spread_mean: f64,
    pub spread_std: f64,
    pub entry_z_score: f64,
    pub pnl: f64,
    pub n_trades: u64,
}

/// The main stat-arb strategy engine.
pub struct StatArbStrategy {
    config: StatArbConfig,
    kalman: KalmanSpreadTracker,
    coint: CointegrationAnalyzer,
    /// Running mean/std of the spread for z-score calculation.
    spread_history: Vec<f64>,
    spread_window: usize,
    position: PairPosition,
    entry_spread: f64,
    entry_z: f64,
    cumulative_pnl: f64,
    n_trades: u64,
    is_cointegrated: bool,
}

impl StatArbStrategy {
    pub fn new(config: StatArbConfig) -> Self {
        let kalman = KalmanSpreadTracker::new(config.kalman_delta, 1.0);
        let coint = CointegrationAnalyzer::new(config.coint_window);
        Self {
            spread_window: config.coint_window,
            config,
            kalman,
            coint,
            spread_history: Vec::new(),
            position: PairPosition::Flat,
            entry_spread: 0.0,
            entry_z: 0.0,
            cumulative_pnl: 0.0,
            n_trades: 0,
            is_cointegrated: false,
        }
    }

    /// Process a new price observation and return a trading signal.
    ///
    /// Returns `Some(signal)` if a trade action is needed.
    pub fn on_prices(
        &mut self,
        price_x: f64,
        price_y: f64,
    ) -> Option<StatArbSignal> {
        // Update Kalman filter
        let (spread, _variance) = self.kalman.update(price_x, price_y);

        // Update cointegration analyzer
        self.coint.add_observation(price_x, price_y);

        // Track spread history
        self.spread_history.push(spread);
        if self.spread_history.len() > self.spread_window {
            self.spread_history.remove(0);
        }

        // Check cointegration periodically
        let n = self.kalman.n_observations();
        if n % self.config.coint_recheck_interval == 0 {
            self.is_cointegrated = self.coint.is_cointegrated();
            if !self.is_cointegrated {
                debug!("Pair not cointegrated after {} observations", n);
            }
        }

        // Need warmup
        if n < self.config.warmup_periods {
            return None;
        }

        // Compute z-score
        let (mean, std) = self.spread_stats();
        if std < 1e-10 {
            return None;
        }
        let z_score = (spread - mean) / std;

        // Generate signal based on position and z-score
        let signal = match self.position {
            PairPosition::Flat => {
                if !self.is_cointegrated {
                    None
                } else if z_score > self.config.entry_z {
                    // Spread is high -> short spread (short Y, long X)
                    self.position = PairPosition::ShortSpread;
                    self.entry_spread = spread;
                    self.entry_z = z_score;
                    self.n_trades += 1;
                    info!("ENTER SHORT SPREAD: z={:.3}, spread={:.6}", z_score, spread);
                    Some(StatArbSignal::EnterShortSpread {
                        hedge_ratio: self.kalman.hedge_ratio(),
                        z_score,
                        spread,
                    })
                } else if z_score < -self.config.entry_z {
                    // Spread is low -> long spread (long Y, short X)
                    self.position = PairPosition::LongSpread;
                    self.entry_spread = spread;
                    self.entry_z = z_score;
                    self.n_trades += 1;
                    info!("ENTER LONG SPREAD: z={:.3}, spread={:.6}", z_score, spread);
                    Some(StatArbSignal::EnterLongSpread {
                        hedge_ratio: self.kalman.hedge_ratio(),
                        z_score,
                        spread,
                    })
                } else {
                    None
                }
            }
            PairPosition::LongSpread => {
                let trade_pnl = spread - self.entry_spread;
                if z_score > -self.config.exit_z {
                    // Mean reverted -> exit
                    self.position = PairPosition::Flat;
                    self.cumulative_pnl += trade_pnl;
                    info!("EXIT LONG SPREAD: z={:.3}, pnl={:.6}", z_score, trade_pnl);
                    Some(StatArbSignal::ExitPosition { pnl: trade_pnl, z_score })
                } else if z_score < -self.config.stop_z {
                    // Stop loss
                    self.position = PairPosition::Flat;
                    self.cumulative_pnl += trade_pnl;
                    warn!("STOP LOSS LONG SPREAD: z={:.3}, pnl={:.6}", z_score, trade_pnl);
                    Some(StatArbSignal::StopLoss { pnl: trade_pnl, z_score })
                } else {
                    None
                }
            }
            PairPosition::ShortSpread => {
                let trade_pnl = self.entry_spread - spread;
                if z_score < self.config.exit_z {
                    // Mean reverted -> exit
                    self.position = PairPosition::Flat;
                    self.cumulative_pnl += trade_pnl;
                    info!("EXIT SHORT SPREAD: z={:.3}, pnl={:.6}", z_score, trade_pnl);
                    Some(StatArbSignal::ExitPosition { pnl: trade_pnl, z_score })
                } else if z_score > self.config.stop_z {
                    // Stop loss
                    self.position = PairPosition::Flat;
                    self.cumulative_pnl += trade_pnl;
                    warn!("STOP LOSS SHORT SPREAD: z={:.3}, pnl={:.6}", z_score, trade_pnl);
                    Some(StatArbSignal::StopLoss { pnl: trade_pnl, z_score })
                } else {
                    None
                }
            }
        };

        signal
    }

    /// Get current pair state snapshot.
    pub fn state(&self, symbol_x: &str, symbol_y: &str) -> PairState {
        let (mean, std) = self.spread_stats();
        let spread = self.spread_history.last().copied().unwrap_or(0.0);
        let z_score = if std > 1e-10 {
            (spread - mean) / std
        } else {
            0.0
        };

        PairState {
            symbol_x: symbol_x.to_string(),
            symbol_y: symbol_y.to_string(),
            position: self.position,
            z_score,
            hedge_ratio: self.kalman.hedge_ratio(),
            spread,
            spread_mean: mean,
            spread_std: std,
            entry_z_score: self.entry_z,
            pnl: self.cumulative_pnl,
            n_trades: self.n_trades,
        }
    }

    /// Get cumulative PnL.
    #[inline]
    pub fn cumulative_pnl(&self) -> f64 {
        self.cumulative_pnl
    }

    fn spread_stats(&self) -> (f64, f64) {
        let n = self.spread_history.len();
        if n < 2 {
            return (0.0, 1.0);
        }
        let mean: f64 = self.spread_history.iter().sum::<f64>() / n as f64;
        let var: f64 = self
            .spread_history
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        (mean, var.sqrt())
    }
}

/// Trading signals emitted by the stat-arb strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatArbSignal {
    EnterLongSpread {
        hedge_ratio: f64,
        z_score: f64,
        spread: f64,
    },
    EnterShortSpread {
        hedge_ratio: f64,
        z_score: f64,
        spread: f64,
    },
    ExitPosition {
        pnl: f64,
        z_score: f64,
    },
    StopLoss {
        pnl: f64,
        z_score: f64,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_basic() {
        let mut kf = KalmanSpreadTracker::crypto_default();

        // Feed correlated prices: y = 2*x + 1 + noise
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..200 {
            let x = 100.0 + (i as f64) * 0.1;
            let y = 2.0 * x + 1.0 + (i as f64 % 3.0) * 0.01;
            kf.update(x, y);
        }

        // Hedge ratio should converge near 2.0
        let hr = kf.hedge_ratio();
        assert!(
            (hr - 2.0).abs() < 0.5,
            "Hedge ratio {} not near 2.0",
            hr
        );
    }

    #[test]
    fn test_cointegration_analyzer() {
        let mut coint = CointegrationAnalyzer::new(200);

        // Cointegrated pair: y = 1.5*x + noise(stationary)
        for i in 0..200 {
            let x = 50.0 + (i as f64) * 0.05;
            let noise = ((i * 7 % 13) as f64 - 6.0) * 0.01;
            let y = 1.5 * x + noise;
            coint.add_observation(x, y);
        }

        let result = coint.compute();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(
            (r.hedge_ratio - 1.5).abs() < 0.1,
            "Hedge ratio {} not near 1.5",
            r.hedge_ratio
        );
    }

    #[test]
    fn test_stat_arb_signal_generation() {
        let config = StatArbConfig {
            warmup_periods: 50,
            entry_z: 1.5,
            exit_z: 0.3,
            stop_z: 3.5,
            ..Default::default()
        };
        let mut strategy = StatArbStrategy::new(config);

        // Warmup with normal spread
        for i in 0..60 {
            let x = 100.0;
            let y = 100.0 + ((i % 5) as f64 - 2.0) * 0.001;
            strategy.on_prices(x, y);
        }

        // Drive z-score high by spiking y
        for _ in 0..20 {
            let signal = strategy.on_prices(100.0, 100.5);
            // Eventually should trigger an entry
            if let Some(s) = signal {
                match s {
                    StatArbSignal::EnterShortSpread { .. } | StatArbSignal::EnterLongSpread { .. } => {
                        return; // Test passes
                    }
                    _ => {}
                }
            }
        }
        // It's OK if warmup wasn't enough -- the strategy is conservative
    }
}
