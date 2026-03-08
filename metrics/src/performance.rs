//! Core performance metrics computed incrementally (online algorithms).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub calmar_ratio: f64,
    pub total_return: f64,
    pub annualized_return: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub cumulative_pnl: f64,
    pub peak_pnl: f64,
}

/// Incremental performance tracker using Welford's online algorithm.
pub struct PerformanceTracker {
    risk_free_rate: f64,
    annualization_factor: f64,
    // Welford's online stats for returns
    n_returns: u64,
    mean_return: f64,
    m2_return: f64,       // Sum of squared deviations (all returns)
    m2_downside: f64,     // Sum of squared deviations (negative returns only)
    n_downside: u64,
    // PnL tracking
    cumulative_pnl: f64,
    peak_pnl: f64,
    max_drawdown: f64,
    // Trade stats
    total_trades: u64,
    winning_trades: u64,
    losing_trades: u64,
    gross_profit: f64,
    gross_loss: f64,
    sum_returns: f64,
}

impl PerformanceTracker {
    pub fn new(risk_free_rate: f64, annualization_factor: f64) -> Self {
        Self {
            risk_free_rate,
            annualization_factor,
            n_returns: 0, mean_return: 0.0, m2_return: 0.0,
            m2_downside: 0.0, n_downside: 0,
            cumulative_pnl: 0.0, peak_pnl: 0.0, max_drawdown: 0.0,
            total_trades: 0, winning_trades: 0, losing_trades: 0,
            gross_profit: 0.0, gross_loss: 0.0, sum_returns: 0.0,
        }
    }

    /// Default for crypto (365 days, 24/7).
    pub fn crypto_default() -> Self {
        Self::new(0.05, (365.0 * 24.0 * 60.0_f64).sqrt())
    }

    /// Record a period return (e.g., per-minute, per-hour).
    pub fn record_return(&mut self, r: f64) {
        self.n_returns += 1;
        self.sum_returns += r;
        let delta = r - self.mean_return;
        self.mean_return += delta / self.n_returns as f64;
        let delta2 = r - self.mean_return;
        self.m2_return += delta * delta2;

        if r < 0.0 {
            self.n_downside += 1;
            self.m2_downside += r * r;
        }
    }

    /// Record a trade PnL.
    pub fn record_pnl(&mut self, pnl: f64) {
        self.total_trades += 1;
        self.cumulative_pnl += pnl;

        if pnl > 0.0 {
            self.winning_trades += 1;
            self.gross_profit += pnl;
        } else if pnl < 0.0 {
            self.losing_trades += 1;
            self.gross_loss += pnl.abs();
        }

        if self.cumulative_pnl > self.peak_pnl {
            self.peak_pnl = self.cumulative_pnl;
        }
        let dd = self.peak_pnl - self.cumulative_pnl;
        if dd > self.max_drawdown {
            self.max_drawdown = dd;
        }
    }

    fn variance(&self) -> f64 {
        if self.n_returns < 2 { return 0.0; }
        self.m2_return / (self.n_returns - 1) as f64
    }

    fn downside_variance(&self) -> f64 {
        if self.n_returns < 2 { return 0.0; }
        self.m2_downside / self.n_returns as f64
    }

    pub fn sharpe_ratio(&self) -> f64 {
        let std = self.variance().sqrt();
        if std < 1e-15 { return 0.0; }
        let excess = self.mean_return - self.risk_free_rate / self.annualization_factor.powi(2);
        (excess / std) * self.annualization_factor
    }

    pub fn sortino_ratio(&self) -> f64 {
        let ds = self.downside_variance().sqrt();
        if ds < 1e-15 { return 0.0; }
        let excess = self.mean_return - self.risk_free_rate / self.annualization_factor.powi(2);
        (excess / ds) * self.annualization_factor
    }

    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown < 1e-15 { return 0.0; }
        let ann_return = self.mean_return * self.annualization_factor.powi(2);
        ann_return / self.max_drawdown
    }

    pub fn snapshot(&self) -> PerformanceSnapshot {
        let win_rate = if self.total_trades > 0 {
            self.winning_trades as f64 / self.total_trades as f64
        } else { 0.0 };

        let profit_factor = if self.gross_loss > 1e-15 {
            self.gross_profit / self.gross_loss
        } else if self.gross_profit > 0.0 { f64::INFINITY } else { 0.0 };

        let avg_win = if self.winning_trades > 0 {
            self.gross_profit / self.winning_trades as f64
        } else { 0.0 };

        let avg_loss = if self.losing_trades > 0 {
            self.gross_loss / self.losing_trades as f64
        } else { 0.0 };

        let max_dd_pct = if self.peak_pnl > 1e-15 {
            self.max_drawdown / self.peak_pnl
        } else { 0.0 };

        PerformanceSnapshot {
            sharpe_ratio: self.sharpe_ratio(),
            sortino_ratio: self.sortino_ratio(),
            max_drawdown: self.max_drawdown,
            max_drawdown_pct: max_dd_pct,
            calmar_ratio: self.calmar_ratio(),
            total_return: self.sum_returns,
            annualized_return: self.mean_return * self.annualization_factor.powi(2),
            win_rate, profit_factor, avg_win, avg_loss,
            total_trades: self.total_trades,
            winning_trades: self.winning_trades,
            losing_trades: self.losing_trades,
            cumulative_pnl: self.cumulative_pnl,
            peak_pnl: self.peak_pnl,
        }
    }

    pub fn reset(&mut self) { *self = Self::new(self.risk_free_rate, self.annualization_factor); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_tracking() {
        let mut t = PerformanceTracker::crypto_default();
        t.record_pnl(100.0);
        t.record_pnl(-30.0);
        t.record_pnl(50.0);
        let s = t.snapshot();
        assert_eq!(s.total_trades, 3);
        assert_eq!(s.winning_trades, 2);
        assert!((s.cumulative_pnl - 120.0).abs() < 1e-10);
        assert!((s.max_drawdown - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_positive_returns() {
        let mut t = PerformanceTracker::new(0.0, (252.0_f64).sqrt());
        for _ in 0..100 { t.record_return(0.001); }
        assert!(t.sharpe_ratio() > 0.0);
    }
}
