//! Market microstructure quality metrics.

use orderbook_core::types::{Price, Quantity, Side, OrderId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureSnapshot {
    pub fill_rate: f64,
    pub effective_spread: f64,
    pub quoted_spread: f64,
    pub price_improvement_rate: f64,
    pub queue_accuracy_rmse: f64,
    pub markout_1s: f64,
    pub markout_5s: f64,
    pub markout_30s: f64,
    pub markout_5min: f64,
    pub pnl_spread_capture: f64,
    pub pnl_inventory_appreciation: f64,
    pub pnl_adverse_selection: f64,
}

struct FillRecord {
    timestamp_ns: u64,
    price: f64,
    quantity: f64,
    side: Side,
    mid_at_fill: f64,
    quoted_spread_at_fill: f64,
}

struct MarkoutRecord {
    fill_price: f64,
    fill_side: Side,
    fill_qty: f64,
    mid_at_fill: f64,
}

pub struct MicrostructureTracker {
    quotes_placed: u64,
    fills_received: u64,
    effective_spread_sum: f64,
    quoted_spread_sum: f64,
    price_improvements: u64,
    queue_est_errors: Vec<f64>,
    // Markout tracking
    pending_markouts: VecDeque<(u64, MarkoutRecord)>,
    markout_1s: Vec<f64>,
    markout_5s: Vec<f64>,
    markout_30s: Vec<f64>,
    markout_5min: Vec<f64>,
    // PnL attribution
    spread_capture_pnl: f64,
    inventory_pnl: f64,
    adverse_selection_pnl: f64,
    // Recent fills for markout resolution
    recent_fills: VecDeque<FillRecord>,
}

impl MicrostructureTracker {
    pub fn new() -> Self {
        Self {
            quotes_placed: 0, fills_received: 0,
            effective_spread_sum: 0.0, quoted_spread_sum: 0.0,
            price_improvements: 0, queue_est_errors: Vec::new(),
            pending_markouts: VecDeque::new(),
            markout_1s: Vec::new(), markout_5s: Vec::new(),
            markout_30s: Vec::new(), markout_5min: Vec::new(),
            spread_capture_pnl: 0.0, inventory_pnl: 0.0,
            adverse_selection_pnl: 0.0, recent_fills: VecDeque::new(),
        }
    }

    pub fn on_quote_placed(&mut self) { self.quotes_placed += 1; }

    pub fn on_fill(
        &mut self, timestamp_ns: u64, price: f64, quantity: f64,
        side: Side, mid_price: f64, quoted_spread: f64,
    ) {
        self.fills_received += 1;

        let effective_half_spread = match side {
            Side::Bid => mid_price - price,
            Side::Ask => price - mid_price,
        };
        self.effective_spread_sum += effective_half_spread * 2.0;
        self.quoted_spread_sum += quoted_spread;

        let expected_half = quoted_spread / 2.0;
        if effective_half_spread < expected_half { self.price_improvements += 1; }

        // Spread capture = half_spread * qty (maker's edge)
        self.spread_capture_pnl += effective_half_spread * quantity;

        // Queue for markout computation
        self.pending_markouts.push_back((timestamp_ns, MarkoutRecord {
            fill_price: price, fill_side: side, fill_qty: quantity, mid_at_fill: mid_price,
        }));

        self.recent_fills.push_back(FillRecord {
            timestamp_ns, price, quantity, side,
            mid_at_fill: mid_price, quoted_spread_at_fill: quoted_spread,
        });
        while self.recent_fills.len() > 10000 { self.recent_fills.pop_front(); }
    }

    /// Call periodically with current mid to resolve markouts.
    pub fn resolve_markouts(&mut self, current_time_ns: u64, current_mid: f64) {
        while let Some(&(ts, _)) = self.pending_markouts.front() {
            let age_ns = current_time_ns.saturating_sub(ts);
            let age_secs = age_ns as f64 / 1e9;

            if age_secs < 1.0 { break; }

            let (_, record) = self.pending_markouts.pop_front().unwrap();
            let mid_move = current_mid - record.mid_at_fill;
            let signed_move = match record.fill_side {
                Side::Bid => -mid_move,  // Bought: adverse if price drops
                Side::Ask => mid_move,   // Sold: adverse if price rises
            };

            if age_secs >= 1.0 { self.markout_1s.push(signed_move * record.fill_qty); }
            if age_secs >= 5.0 { self.markout_5s.push(signed_move * record.fill_qty); }
            if age_secs >= 30.0 { self.markout_30s.push(signed_move * record.fill_qty); }
            if age_secs >= 300.0 {
                self.markout_5min.push(signed_move * record.fill_qty);
                let adverse = signed_move * record.fill_qty;
                self.adverse_selection_pnl += adverse;
            }
        }
    }

    pub fn on_queue_estimate(&mut self, estimated: f64, actual: f64) {
        self.queue_est_errors.push((estimated - actual).powi(2));
    }

    pub fn snapshot(&self) -> MicrostructureSnapshot {
        let fill_rate = if self.quotes_placed > 0 {
            self.fills_received as f64 / self.quotes_placed as f64
        } else { 0.0 };

        let n = self.fills_received.max(1) as f64;
        let avg = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        let rmse = if self.queue_est_errors.is_empty() { 0.0 }
            else { (avg(&self.queue_est_errors)).sqrt() };

        MicrostructureSnapshot {
            fill_rate,
            effective_spread: self.effective_spread_sum / n,
            quoted_spread: self.quoted_spread_sum / n,
            price_improvement_rate: self.price_improvements as f64 / n,
            queue_accuracy_rmse: rmse,
            markout_1s: avg(&self.markout_1s),
            markout_5s: avg(&self.markout_5s),
            markout_30s: avg(&self.markout_30s),
            markout_5min: avg(&self.markout_5min),
            pnl_spread_capture: self.spread_capture_pnl,
            pnl_inventory_appreciation: self.inventory_pnl,
            pnl_adverse_selection: self.adverse_selection_pnl,
        }
    }
}
