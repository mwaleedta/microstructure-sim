//! Main OrderBook implementation.
//!
//! Uses a `crossbeam_skiplist::SkipMap` for each side (bids, asks) giving
//! O(log n) insertion with lock-free concurrent access. A `DashMap` provides
//! O(1) order lookup by OrderId for fast cancellation.
//!
//! The book does NOT perform matching directly -- it delegates to the
//! `matching` module. This separation keeps the book as a pure data structure.

use crate::matching::MatchingEngine;
use crate::price_level::PriceLevel;
use crate::types::*;
use crossbeam_epoch as epoch;
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// OrderRef: location of an order in the book
// ---------------------------------------------------------------------------

/// Reference to an order's location in the book for O(1) cancel.
#[derive(Clone)]
pub struct OrderRef {
    pub price: Price,
    pub side: Side,
    pub level: Arc<PriceLevel>,
}

// ---------------------------------------------------------------------------
// BookSide wrapper
// ---------------------------------------------------------------------------

/// One side of the order book (bids or asks).
///
/// For bids, we want to iterate from highest to lowest price.
/// For asks, we want to iterate from lowest to highest price.
///
/// SkipMap iterates in ascending key order, so:
/// - Asks: store `Price` as key (ascending = best ask first)
/// - Bids: store `Reverse<Price>` as key (ascending Reverse = highest price first)
pub struct BookSide {
    /// Maps price -> PriceLevel. Using i64 directly for SkipMap compatibility.
    levels: SkipMap<i64, Arc<PriceLevel>>,
    side: Side,
}

impl BookSide {
    pub fn new(side: Side) -> Self {
        Self {
            levels: SkipMap::new(),
            side,
        }
    }

    /// Convert a Price to the internal key representation.
    /// Bids: negate so that highest price sorts first in ascending order.
    /// Asks: use as-is so that lowest price sorts first.
    #[inline]
    fn price_to_key(&self, price: Price) -> i64 {
        match self.side {
            Side::Bid => -price.0, // Negate for descending order
            Side::Ask => price.0,  // Ascending order
        }
    }

    /// Convert internal key back to Price.
    #[inline]
    fn key_to_price(&self, key: i64) -> Price {
        match self.side {
            Side::Bid => Price(-key),
            Side::Ask => Price(key),
        }
    }

    /// Get or create the PriceLevel at the given price.
    pub fn get_or_create_level(&self, price: Price) -> Arc<PriceLevel> {
        let key = self.price_to_key(price);

        // Fast path: level exists
        if let Some(entry) = self.levels.get(&key) {
            return Arc::clone(entry.value());
        }

        // Slow path: create new level
        let level = Arc::new(PriceLevel::new(price));
        // insert_or returns existing if race condition
        self.levels.get_or_insert(key, Arc::clone(&level));

        // Re-fetch to handle race
        if let Some(entry) = self.levels.get(&key) {
            Arc::clone(entry.value())
        } else {
            level
        }
    }

    /// Get the best price level (lowest ask or highest bid).
    pub fn best_level(&self) -> Option<(Price, Arc<PriceLevel>)> {
        self.levels.front().map(|entry| {
            let price = self.key_to_price(*entry.key());
            (price, Arc::clone(entry.value()))
        })
    }

    /// Get the best price.
    pub fn best_price(&self) -> Option<Price> {
        self.best_level().map(|(p, _)| p)
    }

    /// Remove an empty price level.
    pub fn remove_level_if_empty(&self, price: Price) {
        let key = self.price_to_key(price);
        if let Some(entry) = self.levels.get(&key) {
            if entry.value().is_empty() {
                entry.remove();
            }
        }
    }

    /// Iterate levels from best to worst price (front of SkipMap).
    /// Returns up to `depth` levels.
    pub fn depth(&self, max_levels: usize) -> Vec<(Price, Quantity, u32)> {
        let mut result = Vec::with_capacity(max_levels);
        for entry in self.levels.iter() {
            if result.len() >= max_levels {
                break;
            }
            let price = self.key_to_price(*entry.key());
            let level = entry.value();
            let qty = level.get_total_quantity();
            let count = level.get_order_count();
            if !qty.is_zero() {
                result.push((price, qty, count));
            }
        }
        result
    }

    /// Total number of price levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

// ---------------------------------------------------------------------------
// OrderBook
// ---------------------------------------------------------------------------

/// Configuration for the order book.
#[derive(Debug, Clone)]
pub struct OrderBookConfig {
    /// Symbol identifier (e.g., "BTC-USD").
    pub symbol: String,
    /// Minimum tick size.
    pub tick_size: Price,
    /// Minimum lot size.
    pub lot_size: Quantity,
    /// Enable self-trade prevention.
    pub self_trade_prevention: bool,
}

impl Default for OrderBookConfig {
    fn default() -> Self {
        Self {
            symbol: "UNKNOWN".to_string(),
            tick_size: Price(1), // 1 raw unit = 0.00000001
            lot_size: Quantity(1),
            self_trade_prevention: true,
        }
    }
}

/// The main order book.
///
/// Thread-safe by construction: SkipMap and DashMap provide lock-free
/// concurrent access. The matching engine runs single-threaded on the
/// hot path but readers (depth queries, BBO) can run concurrently.
pub struct OrderBook {
    pub config: OrderBookConfig,
    pub bids: BookSide,
    pub asks: BookSide,
    /// O(1) order lookup by ID.
    pub orders: DashMap<OrderId, OrderRef>,
    /// Monotonically increasing sequence number for events.
    sequence: AtomicU64,
    /// The matching engine.
    matching_engine: MatchingEngine,
}

impl OrderBook {
    /// Create a new empty order book.
    pub fn new(config: OrderBookConfig) -> Self {
        Self {
            config,
            bids: BookSide::new(Side::Bid),
            asks: BookSide::new(Side::Ask),
            orders: DashMap::new(),
            sequence: AtomicU64::new(1),
            matching_engine: MatchingEngine::new(),
        }
    }

    /// Create with default config for a symbol.
    pub fn with_symbol(symbol: &str) -> Self {
        Self::new(OrderBookConfig {
            symbol: symbol.to_string(),
            ..Default::default()
        })
    }

    /// Get the next sequence number.
    #[inline]
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Submit a new order. Returns all events generated (ack, fills, cancels).
    pub fn submit_order(&self, order: Order) -> Vec<MatchEvent> {
        let guard = &epoch::pin();
        let mut events = Vec::new();
        let seq = self.next_sequence();

        debug!(
            "[{}] Submit: {:?} {} {} @ {} qty={}",
            self.config.symbol,
            order.order_type,
            order.side,
            order.id,
            order.price,
            order.remaining_qty
        );

        // Delegate to the matching engine
        self.matching_engine.process_order(
            order,
            &self.bids,
            &self.asks,
            &self.orders,
            &self.config,
            seq,
            guard,
            &mut events,
        );

        events
    }

    /// Cancel an existing order by ID.
    pub fn cancel_order(&self, order_id: OrderId) -> Option<CancelAck> {
        let guard = &epoch::pin();

        // Look up the order's location
        let order_ref = self.orders.remove(&order_id)?;
        let (_, oref) = order_ref;

        // Remove from the price level
        let remaining = oref.level.cancel_order(order_id, guard)?;

        // Clean up empty levels
        let book_side = match oref.side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        book_side.remove_level_if_empty(oref.price);

        let ack = CancelAck {
            order_id,
            remaining_qty: remaining,
            timestamp_ns: now_ns(),
            reason: CancelReason::UserRequested,
        };

        debug!(
            "[{}] Cancel: {} remaining={}",
            self.config.symbol, order_id, remaining
        );

        Some(ack)
    }

    /// Get the best bid price.
    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.best_price()
    }

    /// Get the best ask price.
    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.best_price()
    }

    /// Get the mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((bid.to_f64() + ask.to_f64()) / 2.0)
            }
            _ => None,
        }
    }

    /// Get the spread in price units.
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the top N levels of depth for both sides.
    pub fn get_depth(
        &self,
        levels: usize,
    ) -> (Vec<(Price, Quantity, u32)>, Vec<(Price, Quantity, u32)>) {
        (self.bids.depth(levels), self.asks.depth(levels))
    }

    /// Total number of resting orders.
    pub fn total_orders(&self) -> usize {
        self.orders.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_book() -> OrderBook {
        OrderBook::with_symbol("BTC-USD")
    }

    #[test]
    fn test_submit_limit_order_rests() {
        let book = setup_book();

        let order = Order::new_limit(
            OrderId(1),
            Side::Bid,
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            now_ns(),
            1,
        );

        let events = book.submit_order(order);
        // Should be accepted (no opposing liquidity to match)
        assert!(events.iter().any(|e| matches!(e, MatchEvent::Accepted(_))));
        assert_eq!(book.best_bid(), Some(Price::from_f64(50000.0)));
        assert_eq!(book.total_orders(), 1);
    }

    #[test]
    fn test_crossing_limit_order_fills() {
        let book = setup_book();

        // Place a resting ask
        let ask = Order::new_limit(
            OrderId(1),
            Side::Ask,
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            now_ns(),
            1,
        );
        book.submit_order(ask);

        // Place a crossing bid
        let bid = Order::new_limit(
            OrderId(2),
            Side::Bid,
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            now_ns() + 1,
            2,
        );
        let events = book.submit_order(bid);

        // Should have a fill
        let fills: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, MatchEvent::Filled(_) | MatchEvent::PartialFill(_)))
            .collect();
        assert!(!fills.is_empty());
    }

    #[test]
    fn test_cancel_order() {
        let book = setup_book();

        let order = Order::new_limit(
            OrderId(1),
            Side::Bid,
            Price::from_f64(50000.0),
            Quantity::from_f64(1.0),
            now_ns(),
            1,
        );
        book.submit_order(order);

        let ack = book.cancel_order(OrderId(1));
        assert!(ack.is_some());
        assert_eq!(book.total_orders(), 0);
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_depth_ordering() {
        let book = setup_book();

        // Add bids at different prices
        for (i, price) in [49000.0, 50000.0, 48000.0].iter().enumerate() {
            let order = Order::new_limit(
                OrderId(i as u64 + 1),
                Side::Bid,
                Price::from_f64(*price),
                Quantity::from_f64(1.0),
                now_ns() + i as u64,
                1,
            );
            book.submit_order(order);
        }

        let (bids, _) = book.get_depth(10);
        assert_eq!(bids.len(), 3);
        // Highest bid first
        assert_eq!(bids[0].0, Price::from_f64(50000.0));
        assert_eq!(bids[1].0, Price::from_f64(49000.0));
        assert_eq!(bids[2].0, Price::from_f64(48000.0));
    }

    #[test]
    fn test_mid_price_and_spread() {
        let book = setup_book();

        let bid = Order::new_limit(
            OrderId(1),
            Side::Bid,
            Price::from_f64(49999.0),
            Quantity::from_f64(1.0),
            now_ns(),
            1,
        );
        let ask = Order::new_limit(
            OrderId(2),
            Side::Ask,
            Price::from_f64(50001.0),
            Quantity::from_f64(1.0),
            now_ns(),
            2,
        );

        book.submit_order(bid);
        book.submit_order(ask);

        let mid = book.mid_price().unwrap();
        assert!((mid - 50000.0).abs() < 1e-6);

        let spread = book.spread().unwrap();
        assert_eq!(spread, Price::from_f64(2.0));
    }
}
