//! # orderbook-core
//!
//! High-performance, lock-free limit order book with price-time priority.
//!
//! ## Architecture
//!
//! - **`types`** -- Fixed-point prices/quantities, order structs, event types
//! - **`price_level`** -- Cache-line-aligned, lock-free price level with FIFO queue
//! - **`book`** -- Main `OrderBook` backed by concurrent skip-list + DashMap
//! - **`matching`** -- Price-time priority matching engine with IOC/FOK/PostOnly
//! - **`simd`** -- SIMD-accelerated hot paths (depth aggregation, VWAP, price scan)

pub mod types;
pub mod price_level;
pub mod book;
pub mod matching;
pub mod simd;

pub use types::*;
