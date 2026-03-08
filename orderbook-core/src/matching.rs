//! Price-time priority matching engine.
//!
//! The matching engine is the heart of the order book. It processes incoming
//! orders against resting liquidity, enforcing price-time priority, and
//! handles all order type semantics (Limit, Market, IOC, FOK, PostOnly).
//!
//! Design: single-threaded on the hot path (one match sweep at a time),
//! but operates on lock-free data structures that allow concurrent reads.

use crate::book::{BookSide, OrderBookConfig, OrderRef};
use crate::price_level::PriceLevel;
use crate::types::*;
use crossbeam_epoch::{self as epoch, Guard};
use dashmap::DashMap;
use std::sync::Arc;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// MatchingEngine
// ---------------------------------------------------------------------------

pub struct MatchingEngine {
    // Stateless for now -- all state lives in the OrderBook.
    // Future: event channel sender, STP config, etc.
}

impl MatchingEngine {
    pub fn new() -> Self {
        Self {}
    }

    /// Process an incoming order against the book.
    ///
    /// This is the main entry point. It handles:
    /// 1. PostOnly rejection if it would cross
    /// 2. FOK feasibility check
    /// 3. Aggressive matching against the opposite side
    /// 4. Resting the remainder (for Limit orders)
    /// 5. Cancelling unfilled IOC remainder
    pub fn process_order(
        &self,
        mut order: Order,
        bids: &BookSide,
        asks: &BookSide,
        orders: &DashMap<OrderId, OrderRef>,
        config: &OrderBookConfig,
        sequence: u64,
        guard: &Guard,
        events: &mut Vec<MatchEvent>,
    ) {
        let opposite_side = match order.side {
            Side::Bid => asks,
            Side::Ask => bids,
        };

        let same_side = match order.side {
            Side::Bid => bids,
            Side::Ask => asks,
        };

        // --- PostOnly check ---
        if order.order_type == OrderType::PostOnly {
            if self.would_cross(&order, opposite_side) {
                events.push(MatchEvent::Cancelled(CancelAck {
                    order_id: order.id,
                    remaining_qty: order.remaining_qty,
                    timestamp_ns: now_ns(),
                    reason: CancelReason::PostOnlyWouldMatch,
                }));
                return;
            }
        }

        // --- FOK feasibility check ---
        if order.order_type == OrderType::FillOrKill {
            if !self.can_fill_completely(&order, opposite_side, guard) {
                events.push(MatchEvent::Cancelled(CancelAck {
                    order_id: order.id,
                    remaining_qty: order.remaining_qty,
                    timestamp_ns: now_ns(),
                    reason: CancelReason::FillOrKill,
                }));
                return;
            }
        }

        // --- Aggressive matching ---
        let fills = self.match_against(
            &mut order,
            opposite_side,
            orders,
            config,
            guard,
        );

        for fill in &fills {
            if fill.quantity == order.original_qty {
                events.push(MatchEvent::Filled(*fill));
            } else {
                events.push(MatchEvent::PartialFill(*fill));
            }
        }

        // --- Handle remainder ---
        if !order.remaining_qty.is_zero() {
            match order.order_type {
                OrderType::Market | OrderType::ImmediateOrCancel => {
                    // Cancel unfilled remainder
                    events.push(MatchEvent::Cancelled(CancelAck {
                        order_id: order.id,
                        remaining_qty: order.remaining_qty,
                        timestamp_ns: now_ns(),
                        reason: CancelReason::ImmediateOrCancel,
                    }));
                }
                OrderType::FillOrKill => {
                    // Should not reach here (checked above), but safety net
                    events.push(MatchEvent::Cancelled(CancelAck {
                        order_id: order.id,
                        remaining_qty: order.remaining_qty,
                        timestamp_ns: now_ns(),
                        reason: CancelReason::FillOrKill,
                    }));
                }
                OrderType::Limit | OrderType::PostOnly => {
                    // Rest on the book
                    let level = same_side.get_or_create_level(order.price);
                    level.add_order(order, guard);

                    orders.insert(
                        order.id,
                        OrderRef {
                            price: order.price,
                            side: order.side,
                            level: Arc::clone(&level),
                        },
                    );

                    events.push(MatchEvent::Accepted(OrderAck {
                        order_id: order.id,
                        timestamp_ns: now_ns(),
                        sequence,
                    }));

                    trace!(
                        "Order {} resting at {} qty={}",
                        order.id,
                        order.price,
                        order.remaining_qty
                    );
                }
            }
        } else if fills.is_empty() {
            // Edge case: zero-qty order
            events.push(MatchEvent::Rejected {
                order_id: order.id,
                reason: "Zero quantity".to_string(),
            });
        }
    }

    /// Match an incoming order against the opposite book side.
    ///
    /// Walks price levels from best to worst, consuming liquidity at each
    /// level until the order is filled or no more matchable levels exist.
    fn match_against(
        &self,
        order: &mut Order,
        opposite: &BookSide,
        orders: &DashMap<OrderId, OrderRef>,
        config: &OrderBookConfig,
        guard: &Guard,
    ) -> Vec<Fill> {
        let mut fills = Vec::new();
        let ts = now_ns();

        // Walk the opposite side from best to worst
        loop {
            if order.remaining_qty.is_zero() {
                break;
            }

            // Get best level on opposite side
            let best = match opposite.best_level() {
                Some(bl) => bl,
                None => break,
            };

            let (level_price, level) = best;

            // Check if prices cross
            let crosses = match order.side {
                Side::Bid => order.price.0 >= level_price.0,
                Side::Ask => order.price.0 <= level_price.0,
            };

            if !crosses && order.order_type != OrderType::Market {
                break; // No more matchable prices
            }

            // Match at this level
            let (level_fills, total_filled) =
                level.match_quantity(order.remaining_qty, guard);

            for (maker_id, fill_qty) in level_fills {
                // Self-trade prevention
                if config.self_trade_prevention {
                    if let Some(maker_ref) = orders.get(&maker_id) {
                        // Check if same trader -- would need to look up
                        // trader_id from the order. For now, skip STP
                        // as it requires order storage in the level.
                    }
                }

                let fill = Fill {
                    maker_order_id: maker_id,
                    taker_order_id: order.id,
                    price: level_price,
                    quantity: fill_qty,
                    timestamp_ns: ts,
                    maker_side: order.side.opposite(),
                };
                fills.push(fill);

                // Remove fully filled maker from order map
                // (partially filled makers remain)
                orders.remove(&maker_id);
            }

            order.remaining_qty = Quantity(
                order.remaining_qty.0.saturating_sub(total_filled.0),
            );

            // Clean up empty levels
            if level.is_empty() {
                opposite.remove_level_if_empty(level_price);
            }
        }

        fills
    }

    /// Check if a PostOnly order would immediately cross.
    fn would_cross(&self, order: &Order, opposite: &BookSide) -> bool {
        if let Some(best_price) = opposite.best_price() {
            match order.side {
                Side::Bid => order.price.0 >= best_price.0,
                Side::Ask => order.price.0 <= best_price.0,
            }
        } else {
            false // No opposing liquidity
        }
    }

    /// Check if a FOK order can be completely filled.
    fn can_fill_completely(
        &self,
        order: &Order,
        opposite: &BookSide,
        guard: &Guard,
    ) -> bool {
        let mut remaining = order.remaining_qty.0;

        for entry in opposite.depth(1000) {
            let (price, qty, _) = entry;

            // Check price compatibility
            let crosses = match order.side {
                Side::Bid => order.price.0 >= price.0,
                Side::Ask => order.price.0 <= price.0,
            };

            if !crosses {
                break;
            }

            remaining = remaining.saturating_sub(qty.0);
            if remaining == 0 {
                return true;
            }
        }

        remaining == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_engine_basic_fill() {
        let engine = MatchingEngine::new();
        let bids = BookSide::new(Side::Bid);
        let asks = BookSide::new(Side::Ask);
        let orders = DashMap::new();
        let config = OrderBookConfig::default();
        let guard = &epoch::pin();

        // Place resting ask
        let ask = Order::new_limit(
            OrderId(1),
            Side::Ask,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            1000,
            1,
        );

        let mut events = Vec::new();
        engine.process_order(
            ask, &bids, &asks, &orders, &config, 1, guard, &mut events,
        );
        assert!(events.iter().any(|e| matches!(e, MatchEvent::Accepted(_))));

        // Place crossing bid
        let bid = Order::new_limit(
            OrderId(2),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(3.0),
            2000,
            2,
        );

        let mut events = Vec::new();
        engine.process_order(
            bid, &bids, &asks, &orders, &config, 2, guard, &mut events,
        );

        let fill_count = events
            .iter()
            .filter(|e| {
                matches!(e, MatchEvent::Filled(_) | MatchEvent::PartialFill(_))
            })
            .count();
        assert!(fill_count > 0);
    }

    #[test]
    fn test_post_only_rejection() {
        let engine = MatchingEngine::new();
        let bids = BookSide::new(Side::Bid);
        let asks = BookSide::new(Side::Ask);
        let orders = DashMap::new();
        let config = OrderBookConfig::default();
        let guard = &epoch::pin();

        // Place resting ask at 100
        let ask = Order::new_limit(
            OrderId(1),
            Side::Ask,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            1000,
            1,
        );
        let mut events = Vec::new();
        engine.process_order(
            ask, &bids, &asks, &orders, &config, 1, guard, &mut events,
        );

        // PostOnly bid at 100 should be rejected
        let mut po_bid = Order::new_limit(
            OrderId(2),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(1.0),
            2000,
            2,
        );
        po_bid.order_type = OrderType::PostOnly;

        let mut events = Vec::new();
        engine.process_order(
            po_bid, &bids, &asks, &orders, &config, 2, guard, &mut events,
        );

        assert!(events.iter().any(|e| {
            if let MatchEvent::Cancelled(ack) = e {
                ack.reason == CancelReason::PostOnlyWouldMatch
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_ioc_partial_fill_cancels_remainder() {
        let engine = MatchingEngine::new();
        let bids = BookSide::new(Side::Bid);
        let asks = BookSide::new(Side::Ask);
        let orders = DashMap::new();
        let config = OrderBookConfig::default();
        let guard = &epoch::pin();

        // Resting ask: 2.0 @ 100
        let ask = Order::new_limit(
            OrderId(1),
            Side::Ask,
            Price::from_f64(100.0),
            Quantity::from_f64(2.0),
            1000,
            1,
        );
        let mut events = Vec::new();
        engine.process_order(
            ask, &bids, &asks, &orders, &config, 1, guard, &mut events,
        );

        // IOC bid for 5.0 @ 100 -- should fill 2.0, cancel 3.0
        let mut ioc_bid = Order::new_limit(
            OrderId(2),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(5.0),
            2000,
            2,
        );
        ioc_bid.order_type = OrderType::ImmediateOrCancel;

        let mut events = Vec::new();
        engine.process_order(
            ioc_bid, &bids, &asks, &orders, &config, 2, guard, &mut events,
        );

        // Should have fill + cancel
        let has_fill = events
            .iter()
            .any(|e| matches!(e, MatchEvent::Filled(_) | MatchEvent::PartialFill(_)));
        let has_cancel = events
            .iter()
            .any(|e| matches!(e, MatchEvent::Cancelled(_)));
        assert!(has_fill);
        assert!(has_cancel);
    }
}
