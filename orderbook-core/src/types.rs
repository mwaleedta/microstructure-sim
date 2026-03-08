//! Core types for the order book engine.
//!
//! All price/quantity types use fixed-point integer arithmetic to avoid
//! floating-point non-determinism. `Price` is stored as `i64` with a
//! configurable decimal shift (default 10^8), giving sub-satoshi precision
//! for crypto and sub-cent precision for equities.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Decimal shift configuration
// ---------------------------------------------------------------------------

/// Number of decimal places encoded in fixed-point prices.
/// Price = raw_value / 10^PRICE_DECIMALS
pub const PRICE_DECIMALS: u32 = 8;

/// Multiplier for converting f64 -> fixed-point Price.
pub const PRICE_MULTIPLIER: i64 = 10_i64.pow(PRICE_DECIMALS);

/// Multiplier for converting f64 -> fixed-point Quantity.
pub const QTY_DECIMALS: u32 = 8;
pub const QTY_MULTIPLIER: u64 = 10_u64.pow(QTY_DECIMALS);

// ---------------------------------------------------------------------------
// OrderId
// ---------------------------------------------------------------------------

/// Unique order identifier. Monotonically increasing within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OID-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Price (fixed-point i64)
// ---------------------------------------------------------------------------

/// Fixed-point price representation. Stored as `raw_value` where
/// `true_price = raw_value / 10^PRICE_DECIMALS`.
///
/// Using i64 instead of f64 eliminates floating-point non-determinism
/// in the matching engine -- critical for reproducible simulations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct Price(pub i64);

impl Price {
    pub const ZERO: Price = Price(0);
    pub const MAX: Price = Price(i64::MAX);
    pub const MIN: Price = Price(i64::MIN);

    /// Create a `Price` from a floating-point value.
    #[inline]
    pub fn from_f64(val: f64) -> Self {
        Price((val * PRICE_MULTIPLIER as f64).round() as i64)
    }

    /// Convert back to f64 for display / analytics.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / PRICE_MULTIPLIER as f64
    }

    /// Tick-size aware rounding (round down for bids, up for asks).
    #[inline]
    pub fn round_to_tick(self, tick_size: Price, side: Side) -> Price {
        if tick_size.0 == 0 {
            return self;
        }
        match side {
            Side::Bid => Price((self.0 / tick_size.0) * tick_size.0),
            Side::Ask => {
                let rem = self.0 % tick_size.0;
                if rem == 0 {
                    self
                } else {
                    Price((self.0 / tick_size.0 + 1) * tick_size.0)
                }
            }
        }
    }
}

impl Ord for Price {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Price {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price({:.8})", self.to_f64())
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl std::ops::Sub for Price {
    type Output = Price;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Price(self.0 - rhs.0)
    }
}

impl std::ops::Add for Price {
    type Output = Price;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Price(self.0 + rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Quantity (fixed-point u64)
// ---------------------------------------------------------------------------

/// Fixed-point quantity. `true_qty = raw_value / 10^QTY_DECIMALS`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct Quantity(pub u64);

impl Quantity {
    pub const ZERO: Quantity = Quantity(0);

    #[inline]
    pub fn from_f64(val: f64) -> Self {
        Quantity((val * QTY_MULTIPLIER as f64).round() as u64)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / QTY_MULTIPLIER as f64
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Saturating subtraction -- never underflows.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Quantity(self.0.saturating_sub(rhs.0))
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qty({:.8})", self.to_f64())
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl std::ops::Sub for Quantity {
    type Output = Quantity;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Quantity(self.0 - rhs.0)
    }
}

impl std::ops::Add for Quantity {
    type Output = Quantity;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Quantity(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for Quantity {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for Quantity {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

// ---------------------------------------------------------------------------
// Side
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Side {
    Bid = 0,
    Ask = 1,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Bid => write!(f, "BID"),
            Side::Ask => write!(f, "ASK"),
        }
    }
}

// ---------------------------------------------------------------------------
// OrderType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderType {
    /// Standard limit order -- rests on the book if not immediately matched.
    Limit = 0,
    /// Market order -- executes at best available price, no resting.
    Market = 1,
    /// Immediate-or-Cancel -- fill what you can, cancel the rest.
    ImmediateOrCancel = 2,
    /// Fill-or-Kill -- fill entirely or cancel entirely.
    FillOrKill = 3,
    /// Post-Only -- rejected if it would immediately match (maker only).
    PostOnly = 4,
}

// ---------------------------------------------------------------------------
// TimeInForce
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeInForce {
    /// Good-til-cancelled (default for limit orders).
    GoodTilCancel = 0,
    /// Good-til-date (expires at a specific timestamp).
    GoodTilDate = 1,
    /// Day order (expires at session close).
    Day = 2,
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

/// A single order in the book. Designed for cache efficiency:
/// - `#[repr(C)]` for predictable layout
/// - All fields are `Copy` (no heap allocations on the hot path)
/// - Total size: 56 bytes (fits in one cache line with padding)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct Order {
    /// Unique identifier.
    pub id: OrderId,
    /// Buy or sell.
    pub side: Side,
    /// Limit price (ignored for Market orders).
    pub price: Price,
    /// Remaining quantity (decremented on partial fills).
    pub remaining_qty: Quantity,
    /// Original quantity (immutable after creation).
    pub original_qty: Quantity,
    /// Timestamp in nanoseconds since epoch (for time priority).
    pub timestamp_ns: u64,
    /// Order type governs matching semantics.
    pub order_type: OrderType,
    /// Time-in-force policy.
    pub time_in_force: TimeInForce,
    /// Optional expiry timestamp (for GoodTilDate).
    pub expire_ns: u64,
    /// Originator identifier for self-trade prevention.
    pub trader_id: u64,
}

impl Order {
    /// Create a new limit order.
    pub fn new_limit(
        id: OrderId,
        side: Side,
        price: Price,
        qty: Quantity,
        timestamp_ns: u64,
        trader_id: u64,
    ) -> Self {
        Self {
            id,
            side,
            price,
            remaining_qty: qty,
            original_qty: qty,
            timestamp_ns,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GoodTilCancel,
            expire_ns: 0,
            trader_id,
        }
    }

    /// Create a new market order.
    pub fn new_market(
        id: OrderId,
        side: Side,
        qty: Quantity,
        timestamp_ns: u64,
        trader_id: u64,
    ) -> Self {
        Self {
            id,
            side,
            price: match side {
                Side::Bid => Price::MAX,
                Side::Ask => Price::MIN,
            },
            remaining_qty: qty,
            original_qty: qty,
            timestamp_ns,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::GoodTilCancel,
            expire_ns: 0,
            trader_id,
        }
    }

    /// Check if order is fully filled.
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.remaining_qty.is_zero()
    }

    /// Check if order can rest on the book.
    #[inline]
    pub fn is_resting_type(&self) -> bool {
        matches!(self.order_type, OrderType::Limit | OrderType::PostOnly)
    }
}

// ---------------------------------------------------------------------------
// Events emitted by the matching engine
// ---------------------------------------------------------------------------

/// A single fill (trade execution).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Fill {
    pub maker_order_id: OrderId,
    pub taker_order_id: OrderId,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp_ns: u64,
    pub maker_side: Side,
}

/// Acknowledgement that an order was accepted into the book.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrderAck {
    pub order_id: OrderId,
    pub timestamp_ns: u64,
    pub sequence: u64,
}

/// Acknowledgement that an order was cancelled.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CancelAck {
    pub order_id: OrderId,
    pub remaining_qty: Quantity,
    pub timestamp_ns: u64,
    pub reason: CancelReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CancelReason {
    UserRequested = 0,
    ImmediateOrCancel = 1,
    FillOrKill = 2,
    PostOnlyWouldMatch = 3,
    Expired = 4,
    SelfTradePrevent = 5,
}

/// Union of all events the matching engine can emit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchEvent {
    Accepted(OrderAck),
    Filled(Fill),
    PartialFill(Fill),
    Cancelled(CancelAck),
    Rejected {
        order_id: OrderId,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Utility: nanosecond timestamp
// ---------------------------------------------------------------------------

/// Get current time as nanoseconds since UNIX epoch.
#[inline]
pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_roundtrip() {
        let p = Price::from_f64(123.45678901);
        let back = p.to_f64();
        assert!((back - 123.45678901).abs() < 1e-8);
    }

    #[test]
    fn test_price_ordering() {
        let a = Price::from_f64(100.0);
        let b = Price::from_f64(200.0);
        assert!(a < b);
    }

    #[test]
    fn test_tick_rounding() {
        let tick = Price::from_f64(0.01);
        let bid = Price::from_f64(100.015);
        let ask = Price::from_f64(100.015);
        // Bid rounds down, ask rounds up
        assert!(bid.round_to_tick(tick, Side::Bid) <= bid);
        assert!(ask.round_to_tick(tick, Side::Ask) >= ask);
    }

    #[test]
    fn test_quantity_saturating_sub() {
        let a = Quantity::from_f64(1.0);
        let b = Quantity::from_f64(2.0);
        assert_eq!(a.saturating_sub(b), Quantity::ZERO);
    }

    #[test]
    fn test_order_creation() {
        let order = Order::new_limit(
            OrderId(1),
            Side::Bid,
            Price::from_f64(50000.0),
            Quantity::from_f64(0.5),
            now_ns(),
            42,
        );
        assert!(!order.is_filled());
        assert!(order.is_resting_type());
    }
}
