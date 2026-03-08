//! Queue position estimation engine.
//!
//! Estimates where our resting orders sit in the queue at each price level,
//! using only L2 (aggregate depth) data. This is crucial for realistic
//! fill simulation -- without it, backtests assume fills happen immediately
//! when price touches our level, which massively overstates performance.
//!
//! ## Model
//!
//! When depth at our price level decreases, the reduction could be:
//! 1. Orders ahead of us being cancelled (helps us)
//! 2. Orders behind us being cancelled (no effect)
//! 3. Trades executing orders ahead of us (helps us)
//!
//! We use a parametric cancellation model:
//! `V_hat(n+1) = max(V_hat(n) - p(n) * |delta_Q(n)|, 0)`
//!
//! Where `p(x) = x^(1+k)` with bias parameter `k in (-1, 0]`.
//! - `k = 0`: cancellations uniformly distributed in queue (naive)
//! - `k < 0`: cancellations biased toward back of queue (more realistic)

use dashmap::DashMap;
use orderbook_core::types::{OrderId, Price, Quantity, Side};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEstimatorConfig {
    /// Cancellation bias parameter. k=0 is uniform, k=-0.5 is back-biased.
    pub cancellation_bias_k: f64,
    /// Initial queue position assumption as fraction of total depth.
    /// 1.0 = assume we're at the back of the queue.
    pub initial_position_fraction: f64,
    /// Minimum trade size to count as a trade (filter noise).
    pub min_trade_size: f64,
}

impl Default for QueueEstimatorConfig {
    fn default() -> Self {
        Self {
            cancellation_bias_k: -0.3,
            initial_position_fraction: 1.0,
            min_trade_size: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tracked order state
// ---------------------------------------------------------------------------

/// State tracked for each of our resting orders.
#[derive(Debug, Clone)]
struct TrackedOrder {
    order_id: OrderId,
    side: Side,
    price: Price,
    quantity: Quantity,
    /// Estimated volume ahead of us in the queue.
    estimated_ahead: f64,
    /// Total depth at our level when we joined.
    initial_depth: f64,
    /// Depth at our level on last update.
    last_known_depth: f64,
    /// Number of trades through our level observed.
    trades_through: u64,
    /// Timestamp when we started tracking.
    created_ns: u64,
    /// Actual fill timestamp (set when filled, for accuracy tracking).
    actual_fill_ns: Option<u64>,
    /// Estimated fill timestamp (based on queue model).
    estimated_fill_ns: Option<u64>,
}

// ---------------------------------------------------------------------------
// Accuracy metrics
// ---------------------------------------------------------------------------

/// Tracks estimation accuracy for calibration.
#[derive(Debug, Clone, Default)]
pub struct QueueAccuracyMetrics {
    /// Number of orders where we predicted fill and got fill.
    pub true_positive: u64,
    /// Number of orders where we predicted fill but didn't get fill.
    pub false_positive: u64,
    /// Number of orders where we didn't predict fill but got fill.
    pub false_negative: u64,
    /// Number of orders where we correctly predicted no fill.
    pub true_negative: u64,
    /// Sum of squared estimation errors (for RMSE).
    pub sum_sq_error: f64,
    /// Total comparisons.
    pub total_comparisons: u64,
}

impl QueueAccuracyMetrics {
    /// Fill prediction accuracy.
    pub fn accuracy(&self) -> f64 {
        let total = self.true_positive + self.false_positive
            + self.false_negative + self.true_negative;
        if total == 0 {
            return 0.0;
        }
        (self.true_positive + self.true_negative) as f64 / total as f64
    }

    /// Root mean squared error of position estimates.
    pub fn rmse(&self) -> f64 {
        if self.total_comparisons == 0 {
            return 0.0;
        }
        (self.sum_sq_error / self.total_comparisons as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// QueueEstimator
// ---------------------------------------------------------------------------

pub struct QueueEstimator {
    config: QueueEstimatorConfig,
    /// Tracked orders by ID.
    tracked: HashMap<OrderId, TrackedOrder>,
    /// Accuracy metrics.
    pub metrics: QueueAccuracyMetrics,
}

impl QueueEstimator {
    pub fn new(config: QueueEstimatorConfig) -> Self {
        Self {
            config,
            tracked: HashMap::new(),
            metrics: QueueAccuracyMetrics::default(),
        }
    }

    /// Start tracking a new resting order.
    pub fn track_order(
        &mut self,
        order_id: OrderId,
        side: Side,
        price: Price,
        quantity: Quantity,
        current_depth: f64,
        timestamp_ns: u64,
    ) {
        let estimated_ahead =
            current_depth * self.config.initial_position_fraction;

        self.tracked.insert(
            order_id,
            TrackedOrder {
                order_id,
                side,
                price,
                quantity,
                estimated_ahead,
                initial_depth: current_depth,
                last_known_depth: current_depth,
                trades_through: 0,
                created_ns: timestamp_ns,
                actual_fill_ns: None,
                estimated_fill_ns: None,
            },
        );
    }

    /// Stop tracking an order (cancelled or filled).
    pub fn untrack_order(&mut self, order_id: OrderId) {
        self.tracked.remove(&order_id);
    }

    /// Update on a depth change at a price level.
    ///
    /// When depth decreases, we estimate how much of the reduction
    /// was ahead of us using the parametric cancellation model.
    pub fn on_depth_update(
        &mut self,
        price: Price,
        old_qty: f64,
        new_qty: f64,
    ) {
        let delta_q = new_qty - old_qty; // Negative when depth decreases

        for order in self.tracked.values_mut() {
            if order.price != price {
                continue;
            }

            if delta_q < 0.0 {
                // Depth decreased -- estimate how much was ahead of us
                let reduction = delta_q.abs();
                let p_ahead = self.cancellation_probability(
                    order.estimated_ahead,
                    order.last_known_depth,
                );

                let removed_ahead = p_ahead * reduction;
                order.estimated_ahead =
                    (order.estimated_ahead - removed_ahead).max(0.0);
            } else {
                // Depth increased -- new orders go behind us (no effect)
                // estimated_ahead stays the same
            }

            order.last_known_depth = new_qty;
        }
    }

    /// Update on a trade at a price level.
    ///
    /// Trades definitively remove volume from the front of the queue.
    pub fn on_trade(&mut self, price: Price, qty: f64) {
        if qty < self.config.min_trade_size {
            return;
        }

        for order in self.tracked.values_mut() {
            if order.price != price {
                continue;
            }

            // Trades always remove from front of queue
            order.estimated_ahead =
                (order.estimated_ahead - qty).max(0.0);
            order.trades_through += 1;
            order.last_known_depth =
                (order.last_known_depth - qty).max(0.0);
        }
    }

    /// Get estimated queue position for an order.
    ///
    /// Returns the estimated volume ahead of the order.
    pub fn estimated_position(&self, order_id: OrderId) -> Option<f64> {
        self.tracked.get(&order_id).map(|o| o.estimated_ahead)
    }

    /// Get estimated fill probability.
    ///
    /// Returns a value in [0, 1] -- higher means closer to being filled.
    pub fn fill_probability(&self, order_id: OrderId) -> Option<f64> {
        self.tracked.get(&order_id).map(|order| {
            if order.initial_depth <= 0.0 {
                return 1.0;
            }
            let progress =
                1.0 - (order.estimated_ahead / order.initial_depth);
            progress.clamp(0.0, 1.0)
        })
    }

    /// Check if an order is estimated to be at the front of the queue.
    pub fn is_at_front(&self, order_id: OrderId) -> bool {
        self.estimated_position(order_id)
            .map(|pos| pos <= 0.0)
            .unwrap_or(false)
    }

    /// Record actual fill for accuracy tracking.
    pub fn record_actual_fill(
        &mut self,
        order_id: OrderId,
        timestamp_ns: u64,
    ) {
        if let Some(order) = self.tracked.get_mut(&order_id) {
            order.actual_fill_ns = Some(timestamp_ns);

            // Compare with prediction
            if order.estimated_ahead <= 0.0 {
                self.metrics.true_positive += 1;
            } else {
                self.metrics.false_negative += 1;
            }

            let error = order.estimated_ahead;
            self.metrics.sum_sq_error += error * error;
            self.metrics.total_comparisons += 1;
        }
    }

    /// Parametric cancellation probability.
    ///
    /// Given our estimated position `v` in a queue of total depth `d`,
    /// what fraction of a random cancellation is likely to be ahead of us?
    ///
    /// `p(v, d) = (v/d)^(1+k)` where k is the cancellation bias.
    fn cancellation_probability(&self, v: f64, d: f64) -> f64 {
        if d <= 0.0 || v <= 0.0 {
            return 0.0;
        }

        let normalized = (v / d).clamp(0.0, 1.0);
        let k = self.config.cancellation_bias_k;

        normalized.powf(1.0 + k)
    }

    /// Get number of tracked orders.
    pub fn tracked_count(&self) -> usize {
        self.tracked.len()
    }

    /// Get all tracked order IDs.
    pub fn tracked_orders(&self) -> Vec<OrderId> {
        self.tracked.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> QueueEstimator {
        QueueEstimator::new(QueueEstimatorConfig {
            cancellation_bias_k: -0.3,
            initial_position_fraction: 1.0,
            min_trade_size: 0.0,
        })
    }

    #[test]
    fn test_track_order() {
        let mut est = setup();
        est.track_order(
            OrderId(1),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            10.0,
            0,
        );

        assert_eq!(est.tracked_count(), 1);
        // Initially at back of queue
        assert!((est.estimated_position(OrderId(1)).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_trade_reduces_position() {
        let mut est = setup();
        est.track_order(
            OrderId(1),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            10.0,
            0,
        );

        // Trade of 5.0 removes from front
        est.on_trade(Price::from_f64(100.0), 5.0);
        assert!((est.estimated_position(OrderId(1)).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_depth_decrease_partial_reduction() {
        let mut est = setup();
        est.track_order(
            OrderId(1),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            10.0,
            0,
        );

        // Depth decreases by 3 (cancellation)
        est.on_depth_update(Price::from_f64(100.0), 10.0, 7.0);

        // Position should decrease but not by the full 3
        // (back-biased cancellations are more likely behind us)
        let pos = est.estimated_position(OrderId(1)).unwrap();
        assert!(pos < 10.0);
        assert!(pos > 7.0); // Not all removals from ahead
    }

    #[test]
    fn test_fill_probability_increases() {
        let mut est = setup();
        est.track_order(
            OrderId(1),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            10.0,
            0,
        );

        let prob_before = est.fill_probability(OrderId(1)).unwrap();

        // Trade removes 8 from front
        est.on_trade(Price::from_f64(100.0), 8.0);

        let prob_after = est.fill_probability(OrderId(1)).unwrap();
        assert!(prob_after > prob_before);
    }

    #[test]
    fn test_is_at_front() {
        let mut est = setup();
        est.track_order(
            OrderId(1),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            5.0,
            0,
        );

        assert!(!est.is_at_front(OrderId(1)));

        // Clear the queue
        est.on_trade(Price::from_f64(100.0), 5.0);
        assert!(est.is_at_front(OrderId(1)));
    }
}
