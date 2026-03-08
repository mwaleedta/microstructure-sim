//! Quote management engine.
//!
//! Combines Avellaneda-Stoikov model signals with inventory constraints
//! to produce actionable quote updates. Handles:
//!
//! - Requoting triggers (price move, time refresh, inventory change)
//! - Flickering prevention (minimum quote lifetime)
//! - Spread floor enforcement
//! - Tick-size rounding
//! - Emergency flatten quoting

use crate::avellaneda_stoikov::AvellanedaStoikov;
use crate::inventory::InventoryManager;
use orderbook_core::types::{OrderId, Price, Quantity, Side};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotingConfig {
    /// Minimum time (ms) a quote must live before being replaced.
    pub min_quote_lifetime_ms: u64,
    /// Price move threshold (as fraction of spread) to trigger requote.
    pub requote_price_threshold: f64,
    /// Time-based refresh interval (ms). Requote even if price hasn't moved.
    pub refresh_interval_ms: u64,
    /// Minimum spread floor (absolute price units).
    pub min_spread: f64,
    /// Tick size for price rounding.
    pub tick_size: f64,
    /// Number of quote levels (1 = single level, >1 = layered).
    pub num_levels: usize,
    /// Spread increment between layers (fraction of base spread).
    pub layer_spread_increment: f64,
    /// Size multiplier per layer (geometric growth).
    pub layer_size_multiplier: f64,
}

impl Default for QuotingConfig {
    fn default() -> Self {
        Self {
            min_quote_lifetime_ms: 100,
            requote_price_threshold: 0.3,
            refresh_interval_ms: 1000,
            min_spread: 0.01,
            tick_size: 0.01,
            num_levels: 1,
            layer_spread_increment: 0.5,
            layer_size_multiplier: 1.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Quote state
// ---------------------------------------------------------------------------

/// A single outstanding quote.
#[derive(Debug, Clone, Copy)]
pub struct ActiveQuote {
    pub order_id: OrderId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub placed_at_ms: u64,
    pub level: usize,
}

/// Desired update to the quote set.
#[derive(Debug, Clone)]
pub struct QuoteUpdate {
    /// New bid quotes to place (price, qty) per level.
    pub bids: Vec<(Price, Quantity)>,
    /// New ask quotes to place (price, qty) per level.
    pub asks: Vec<(Price, Quantity)>,
    /// Order IDs to cancel before placing new quotes.
    pub cancel_ids: Vec<OrderId>,
    /// Reason for the update.
    pub reason: RequoteReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequoteReason {
    Initial,
    PriceMove,
    TimeRefresh,
    InventoryChange,
    EmergencyFlatten,
    Fill,
}

// ---------------------------------------------------------------------------
// QuotingEngine
// ---------------------------------------------------------------------------

pub struct QuotingEngine {
    pub config: QuotingConfig,

    /// Currently active bid quotes.
    active_bids: Vec<ActiveQuote>,
    /// Currently active ask quotes.
    active_asks: Vec<ActiveQuote>,

    /// Last mid price when quotes were placed.
    last_quote_mid: f64,
    /// Timestamp (ms) of last quote update.
    last_quote_time_ms: u64,
    /// Last known inventory when quotes were placed.
    last_quote_inventory: f64,

    /// Counter for generating local order IDs.
    next_order_id: u64,

    /// Recent quote update history (for analysis).
    update_history: VecDeque<(u64, RequoteReason)>,
}

impl QuotingEngine {
    pub fn new(config: QuotingConfig, start_order_id: u64) -> Self {
        Self {
            config,
            active_bids: Vec::new(),
            active_asks: Vec::new(),
            last_quote_mid: 0.0,
            last_quote_time_ms: 0,
            last_quote_inventory: 0.0,
            next_order_id: start_order_id,
            update_history: VecDeque::with_capacity(1000),
        }
    }

    /// Generate the next order ID.
    fn gen_order_id(&mut self) -> OrderId {
        let id = self.next_order_id;
        self.next_order_id += 1;
        OrderId(id)
    }

    /// Check if requoting is needed.
    pub fn should_requote(
        &self,
        current_mid: f64,
        current_time_ms: u64,
        current_inventory: f64,
    ) -> Option<RequoteReason> {
        // No active quotes -> always requote
        if self.active_bids.is_empty() && self.active_asks.is_empty() {
            return Some(RequoteReason::Initial);
        }

        // Check minimum lifetime
        let age_ms = current_time_ms.saturating_sub(self.last_quote_time_ms);
        if age_ms < self.config.min_quote_lifetime_ms {
            return None; // Too soon -- prevent flickering
        }

        // Price move check
        if self.last_quote_mid > 0.0 {
            let move_frac =
                (current_mid - self.last_quote_mid).abs() / self.last_quote_mid;
            let spread_frac = self.config.min_spread / current_mid;
            if move_frac > spread_frac * self.config.requote_price_threshold {
                return Some(RequoteReason::PriceMove);
            }
        }

        // Time refresh
        if age_ms >= self.config.refresh_interval_ms {
            return Some(RequoteReason::TimeRefresh);
        }

        // Inventory change
        let inv_change =
            (current_inventory - self.last_quote_inventory).abs();
        if inv_change > 0.5 {
            return Some(RequoteReason::InventoryChange);
        }

        None
    }

    /// Generate a quote update using the A-S model and inventory manager.
    pub fn generate_quotes(
        &mut self,
        model: &AvellanedaStoikov,
        inventory: &InventoryManager,
        current_mid: f64,
        current_time_ms: u64,
    ) -> Option<QuoteUpdate> {
        // Check if model is warmed up
        if !model.is_ready() {
            return None;
        }

        // Check for emergency flatten
        if inventory.is_flattening() {
            return Some(self.generate_flatten_quotes(
                inventory,
                current_mid,
                current_time_ms,
            ));
        }

        // Check if we need to requote
        let reason = self.should_requote(
            current_mid,
            current_time_ms,
            inventory.position(),
        )?;

        // Get model quotes
        let (base_bid, base_ask) = model.compute_quotes(current_mid);

        // Get inventory-adjusted quantities
        let (bid_qty, ask_qty) = inventory.adjusted_quantities();

        // Build cancellation list
        let cancel_ids: Vec<OrderId> = self
            .active_bids
            .iter()
            .chain(self.active_asks.iter())
            .map(|q| q.order_id)
            .collect();

        // Generate multi-level quotes
        let mut bids = Vec::with_capacity(self.config.num_levels);
        let mut asks = Vec::with_capacity(self.config.num_levels);

        for level in 0..self.config.num_levels {
            let spread_offset = level as f64
                * self.config.layer_spread_increment
                * model.optimal_spread();
            let size_mult = self.config.layer_size_multiplier.powi(level as i32);

            // Bid: move further from mid at each level
            let bid_price = base_bid - spread_offset;
            let bid_rounded = self.round_tick(bid_price, Side::Bid);
            let bid_size = bid_qty * size_mult;

            // Ask: move further from mid at each level
            let ask_price = base_ask + spread_offset;
            let ask_rounded = self.round_tick(ask_price, Side::Ask);
            let ask_size = ask_qty * size_mult;

            // Enforce spread floor
            if ask_rounded - bid_rounded >= self.config.min_spread {
                if inventory.can_trade(Side::Bid, bid_size) {
                    bids.push((
                        Price::from_f64(bid_rounded),
                        Quantity::from_f64(bid_size),
                    ));
                }
                if inventory.can_trade(Side::Ask, ask_size) {
                    asks.push((
                        Price::from_f64(ask_rounded),
                        Quantity::from_f64(ask_size),
                    ));
                }
            }
        }

        // Update state
        self.last_quote_mid = current_mid;
        self.last_quote_time_ms = current_time_ms;
        self.last_quote_inventory = inventory.position();

        // Track active quotes
        self.active_bids.clear();
        self.active_asks.clear();

        for (i, (price, qty)) in bids.iter().enumerate() {
            self.active_bids.push(ActiveQuote {
                order_id: self.gen_order_id(),
                side: Side::Bid,
                price: *price,
                quantity: *qty,
                placed_at_ms: current_time_ms,
                level: i,
            });
        }
        for (i, (price, qty)) in asks.iter().enumerate() {
            self.active_asks.push(ActiveQuote {
                order_id: self.gen_order_id(),
                side: Side::Ask,
                price: *price,
                quantity: *qty,
                placed_at_ms: current_time_ms,
                level: i,
            });
        }

        // Record history
        self.update_history
            .push_back((current_time_ms, reason));
        if self.update_history.len() > 1000 {
            self.update_history.pop_front();
        }

        Some(QuoteUpdate {
            bids,
            asks,
            cancel_ids,
            reason,
        })
    }

    /// Generate aggressive flatten quotes.
    fn generate_flatten_quotes(
        &mut self,
        inventory: &InventoryManager,
        current_mid: f64,
        current_time_ms: u64,
    ) -> QuoteUpdate {
        let cancel_ids: Vec<OrderId> = self
            .active_bids
            .iter()
            .chain(self.active_asks.iter())
            .map(|q| q.order_id)
            .collect();

        let position = inventory.position();
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        if position > 0.0 {
            // Long -- aggressively sell at mid or below
            let ask_price = self.round_tick(current_mid, Side::Ask);
            asks.push((
                Price::from_f64(ask_price),
                Quantity::from_f64(position.abs()),
            ));
        } else if position < 0.0 {
            // Short -- aggressively buy at mid or above
            let bid_price = self.round_tick(current_mid, Side::Bid);
            bids.push((
                Price::from_f64(bid_price),
                Quantity::from_f64(position.abs()),
            ));
        }

        self.active_bids.clear();
        self.active_asks.clear();
        self.last_quote_time_ms = current_time_ms;

        QuoteUpdate {
            bids,
            asks,
            cancel_ids,
            reason: RequoteReason::EmergencyFlatten,
        }
    }

    /// Notify that one of our quotes was filled.
    pub fn on_fill(&mut self, order_id: OrderId, filled_qty: Quantity) {
        self.active_bids.retain(|q| q.order_id != order_id);
        self.active_asks.retain(|q| q.order_id != order_id);
    }

    /// Round a price to tick size.
    fn round_tick(&self, price: f64, side: Side) -> f64 {
        let tick = self.config.tick_size;
        if tick <= 0.0 {
            return price;
        }
        match side {
            Side::Bid => (price / tick).floor() * tick,
            Side::Ask => (price / tick).ceil() * tick,
        }
    }

    /// Get the current active bid quotes.
    pub fn active_bids(&self) -> &[ActiveQuote] {
        &self.active_bids
    }

    /// Get the current active ask quotes.
    pub fn active_asks(&self) -> &[ActiveQuote] {
        &self.active_asks
    }

    /// Get requote frequency over the last N milliseconds.
    pub fn requote_rate(&self, window_ms: u64, current_time_ms: u64) -> f64 {
        let cutoff = current_time_ms.saturating_sub(window_ms);
        let count = self
            .update_history
            .iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .count();
        if window_ms > 0 {
            count as f64 / (window_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avellaneda_stoikov::{ASConfig, AvellanedaStoikov};
    use crate::inventory::{InventoryConfig, InventoryManager};

    fn setup() -> (QuotingEngine, AvellanedaStoikov, InventoryManager) {
        let qe = QuotingEngine::new(QuotingConfig::default(), 1000);
        let mut model = AvellanedaStoikov::new(ASConfig {
            tick_size: 0.01,
            min_spread: 0.01,
            max_spread: 100.0,
            ..Default::default()
        });

        // Warm up model
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            model.on_tick(price, i as f64 * 0.1);
        }

        let inv = InventoryManager::new(InventoryConfig::default());
        (qe, model, inv)
    }

    #[test]
    fn test_initial_quote_generation() {
        let (mut qe, model, inv) = setup();

        let update = qe.generate_quotes(&model, &inv, 100.0, 0);
        assert!(update.is_some());

        let update = update.unwrap();
        assert_eq!(update.reason, RequoteReason::Initial);
        assert!(!update.bids.is_empty());
        assert!(!update.asks.is_empty());
    }

    #[test]
    fn test_flickering_prevention() {
        let (mut qe, model, inv) = setup();

        // Place initial quotes
        qe.generate_quotes(&model, &inv, 100.0, 0);

        // Try to requote immediately -- should be blocked
        let should_requote = qe.should_requote(100.001, 50, 0.0);
        assert!(should_requote.is_none());
    }

    #[test]
    fn test_time_refresh_requote() {
        let (mut qe, model, inv) = setup();

        qe.generate_quotes(&model, &inv, 100.0, 0);

        // After refresh interval
        let should_requote = qe.should_requote(100.0, 2000, 0.0);
        assert_eq!(should_requote, Some(RequoteReason::TimeRefresh));
    }

    #[test]
    fn test_spread_floor_enforcement() {
        let (mut qe, model, inv) = setup();

        let update = qe.generate_quotes(&model, &inv, 100.0, 0).unwrap();

        if !update.bids.is_empty() && !update.asks.is_empty() {
            let bid = update.bids[0].0.to_f64();
            let ask = update.asks[0].0.to_f64();
            assert!(ask - bid >= qe.config.min_spread);
        }
    }
}
