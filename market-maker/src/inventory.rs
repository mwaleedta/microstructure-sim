//! Inventory risk management for market making.
//!
//! Tracks position, PnL, and enforces risk limits. Provides inventory-aware
//! adjustments to quote prices and sizes.
//!
//! Key features:
//! - Hard position limits with emergency flatten
//! - Inventory skew scaling (linear and exponential)
//! - Quote size adjustment based on current exposure
//! - Real-time PnL tracking (realized + unrealized)

use orderbook_core::types::{Price, Quantity, Side, Fill};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryConfig {
    /// Maximum allowed absolute position.
    pub max_position: f64,
    /// Target (ideal) position. Usually 0 for delta-neutral.
    pub target_position: f64,
    /// Linear skew factor: higher values -> more aggressive skew.
    pub skew_factor: f64,
    /// Position level at which emergency flatten is triggered.
    /// Expressed as fraction of max_position (e.g., 0.9 = 90%).
    pub flatten_threshold: f64,
    /// Base quote size.
    pub base_qty: f64,
    /// Minimum quote size.
    pub min_qty: f64,
    /// Maximum quote size.
    pub max_qty: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            max_position: 100.0,
            target_position: 0.0,
            skew_factor: 0.5,
            flatten_threshold: 0.9,
            base_qty: 1.0,
            min_qty: 0.01,
            max_qty: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PnL tracking
// ---------------------------------------------------------------------------

/// Tracks realized and unrealized PnL.
#[derive(Debug, Clone, Default)]
pub struct PnLTracker {
    /// Cumulative realized PnL from closed trades.
    pub realized_pnl: f64,
    /// Cost basis of current position (VWAP entry price * position).
    pub cost_basis: f64,
    /// Volume-weighted average entry price.
    pub avg_entry_price: f64,
    /// Total traded volume (for statistics).
    pub total_volume: f64,
    /// Number of trades.
    pub trade_count: u64,
}

impl PnLTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fill and update PnL.
    pub fn on_fill(&mut self, side: Side, price: f64, qty: f64, position: f64) {
        self.total_volume += qty * price;
        self.trade_count += 1;

        let signed_qty = match side {
            Side::Bid => qty,   // Bought
            Side::Ask => -qty,  // Sold
        };

        let prev_position = position - signed_qty;

        // Check if this is reducing or reversing position
        if prev_position.signum() != 0.0 && prev_position.signum() != signed_qty.signum() {
            // Reducing position -- realize PnL
            let reducing_qty = qty.min(prev_position.abs());
            let pnl = match side {
                Side::Ask => (price - self.avg_entry_price) * reducing_qty,  // Selling long
                Side::Bid => (self.avg_entry_price - price) * reducing_qty,  // Covering short
            };
            self.realized_pnl += pnl;
        }

        // Update average entry price
        if position.abs() > 1e-12 {
            if prev_position.signum() == signed_qty.signum() || prev_position.abs() < 1e-12 {
                // Adding to position -- update VWAP
                let old_cost = self.avg_entry_price * prev_position.abs();
                let new_cost = price * qty;
                self.avg_entry_price =
                    (old_cost + new_cost) / position.abs();
            } else if position.signum() != prev_position.signum() {
                // Position reversed -- new entry price
                self.avg_entry_price = price;
            }
        } else {
            self.avg_entry_price = 0.0;
        }

        self.cost_basis = self.avg_entry_price * position.abs();
    }

    /// Compute unrealized PnL at current market price.
    pub fn unrealized_pnl(&self, position: f64, current_price: f64) -> f64 {
        if position.abs() < 1e-12 {
            return 0.0;
        }
        (current_price - self.avg_entry_price) * position
    }

    /// Total PnL (realized + unrealized).
    pub fn total_pnl(&self, position: f64, current_price: f64) -> f64 {
        self.realized_pnl + self.unrealized_pnl(position, current_price)
    }
}

// ---------------------------------------------------------------------------
// Inventory skew
// ---------------------------------------------------------------------------

/// Skew scaling method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SkewMode {
    /// Linear: skew proportional to inventory.
    Linear,
    /// Exponential: skew grows exponentially near limits.
    Exponential,
}

/// Compute inventory skew adjustment.
///
/// Returns a value in [-1, 1] that shifts quotes:
/// - Positive skew (long inventory): shifts quotes down (sell more aggressively)
/// - Negative skew (short inventory): shifts quotes up (buy more aggressively)
pub fn compute_skew(
    position: f64,
    target: f64,
    max_position: f64,
    skew_factor: f64,
    mode: SkewMode,
) -> f64 {
    if max_position <= 0.0 {
        return 0.0;
    }

    let deviation = (position - target) / max_position;

    match mode {
        SkewMode::Linear => {
            (deviation * skew_factor).clamp(-1.0, 1.0)
        }
        SkewMode::Exponential => {
            let sign = deviation.signum();
            let magnitude = deviation.abs();
            let exp_skew = (magnitude * skew_factor).exp() - 1.0;
            (sign * exp_skew).clamp(-1.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// InventoryManager
// ---------------------------------------------------------------------------

/// Central inventory and risk management.
pub struct InventoryManager {
    pub config: InventoryConfig,
    pub pnl: PnLTracker,

    /// Current net position (positive = long, negative = short).
    position: f64,

    /// Skew mode.
    skew_mode: SkewMode,

    /// Whether emergency flatten is active.
    flatten_active: bool,
}

impl InventoryManager {
    pub fn new(config: InventoryConfig) -> Self {
        Self {
            config,
            pnl: PnLTracker::new(),
            position: 0.0,
            skew_mode: SkewMode::Linear,
            flatten_active: false,
        }
    }

    /// Get current position.
    #[inline]
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Check if we're in emergency flatten mode.
    #[inline]
    pub fn is_flattening(&self) -> bool {
        self.flatten_active
    }

    /// Update position after a fill.
    pub fn on_fill(&mut self, side: Side, price: f64, qty: f64) {
        let signed_qty = match side {
            Side::Bid => qty,
            Side::Ask => -qty,
        };

        self.position += signed_qty;
        self.pnl.on_fill(side, price, qty, self.position);

        // Check flatten trigger
        let utilization = self.position.abs() / self.config.max_position;
        if utilization >= self.config.flatten_threshold {
            if !self.flatten_active {
                warn!(
                    "FLATTEN triggered: position={:.4}, utilization={:.1}%",
                    self.position,
                    utilization * 100.0
                );
            }
            self.flatten_active = true;
        } else if utilization < self.config.flatten_threshold * 0.8 {
            // Hysteresis: only deactivate at 80% of threshold
            if self.flatten_active {
                info!("Flatten deactivated: position={:.4}", self.position);
            }
            self.flatten_active = false;
        }
    }

    /// Get the current inventory skew.
    pub fn skew(&self) -> f64 {
        compute_skew(
            self.position,
            self.config.target_position,
            self.config.max_position,
            self.config.skew_factor,
            self.skew_mode,
        )
    }

    /// Adjust quote sizes based on inventory.
    ///
    /// When long, increase ask size and decrease bid size (want to sell).
    /// When short, increase bid size and decrease ask size (want to buy).
    ///
    /// Returns (bid_qty, ask_qty).
    pub fn adjusted_quantities(&self) -> (f64, f64) {
        let skew = self.skew();
        let base = self.config.base_qty;

        // Positive skew = long, reduce bids, increase asks
        let bid_adj = 1.0 - skew.max(0.0);
        let ask_adj = 1.0 + skew.max(0.0);

        let bid_qty = (base * bid_adj).clamp(
            self.config.min_qty,
            self.config.max_qty,
        );
        let ask_qty = (base * ask_adj).clamp(
            self.config.min_qty,
            self.config.max_qty,
        );

        (bid_qty, ask_qty)
    }

    /// Check if a new order on the given side would breach limits.
    pub fn can_trade(&self, side: Side, qty: f64) -> bool {
        let new_position = match side {
            Side::Bid => self.position + qty,
            Side::Ask => self.position - qty,
        };
        new_position.abs() <= self.config.max_position
    }

    /// Get the flatten direction (which side to aggressively quote).
    pub fn flatten_side(&self) -> Option<Side> {
        if !self.flatten_active {
            return None;
        }
        if self.position > 0.0 {
            Some(Side::Ask) // Need to sell
        } else {
            Some(Side::Bid) // Need to buy
        }
    }

    /// Set the skew mode.
    pub fn set_skew_mode(&mut self, mode: SkewMode) {
        self.skew_mode = mode;
    }

    /// Reset position and PnL (for new session).
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.pnl = PnLTracker::new();
        self.flatten_active = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> InventoryManager {
        InventoryManager::new(InventoryConfig {
            max_position: 10.0,
            target_position: 0.0,
            skew_factor: 0.5,
            flatten_threshold: 0.9,
            base_qty: 1.0,
            min_qty: 0.1,
            max_qty: 5.0,
        })
    }

    #[test]
    fn test_position_tracking() {
        let mut mgr = make_manager();

        mgr.on_fill(Side::Bid, 100.0, 5.0);
        assert!((mgr.position() - 5.0).abs() < 1e-10);

        mgr.on_fill(Side::Ask, 101.0, 3.0);
        assert!((mgr.position() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_flatten_trigger() {
        let mut mgr = make_manager();

        // Buy up to 90% of max
        mgr.on_fill(Side::Bid, 100.0, 9.0);
        assert!(mgr.is_flattening());
        assert_eq!(mgr.flatten_side(), Some(Side::Ask));
    }

    #[test]
    fn test_skew_neutral() {
        let mgr = make_manager();
        assert!((mgr.skew()).abs() < 1e-10);
    }

    #[test]
    fn test_skew_long() {
        let mut mgr = make_manager();
        mgr.on_fill(Side::Bid, 100.0, 5.0);
        let skew = mgr.skew();
        assert!(skew > 0.0); // Long -> positive skew
    }

    #[test]
    fn test_adjusted_quantities() {
        let mut mgr = make_manager();
        mgr.on_fill(Side::Bid, 100.0, 5.0);

        let (bid_qty, ask_qty) = mgr.adjusted_quantities();
        // Long position: bid reduced, ask increased
        assert!(bid_qty < ask_qty);
    }

    #[test]
    fn test_can_trade_limits() {
        let mut mgr = make_manager();
        mgr.on_fill(Side::Bid, 100.0, 9.0);

        assert!(mgr.can_trade(Side::Bid, 1.0)); // 9+1 = 10 = max
        assert!(!mgr.can_trade(Side::Bid, 2.0)); // 9+2 = 11 > max
        assert!(mgr.can_trade(Side::Ask, 5.0)); // 9-5 = 4, ok
    }

    #[test]
    fn test_pnl_tracking() {
        let mut mgr = make_manager();

        // Buy 5 @ 100
        mgr.on_fill(Side::Bid, 100.0, 5.0);
        assert!((mgr.pnl.avg_entry_price - 100.0).abs() < 1e-10);

        // Sell 5 @ 102 -- realize profit
        mgr.on_fill(Side::Ask, 102.0, 5.0);
        assert!((mgr.pnl.realized_pnl - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_skew() {
        let skew_lin = compute_skew(5.0, 0.0, 10.0, 0.5, SkewMode::Linear);
        let skew_exp = compute_skew(5.0, 0.0, 10.0, 0.5, SkewMode::Exponential);

        // Exponential should be more aggressive
        assert!(skew_exp.abs() > skew_lin.abs());
    }
}
