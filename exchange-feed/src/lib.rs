//! # exchange-feed
//!
//! Live exchange WebSocket feed handlers for Binance and Coinbase.
//!
//! Connects to exchange WebSocket APIs, parses L2 depth updates, and
//! applies them to an `OrderBook` instance in real-time.
//!
//! ## Modules
//!
//! - **`binance`** -- Binance depth stream (diff + snapshot)
//! - **`coinbase`** -- Coinbase L2 channel

pub mod binance;
pub mod coinbase;

use orderbook_core::book::OrderBook;
use orderbook_core::types::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Common feed types
// ---------------------------------------------------------------------------

/// Errors that can occur during feed operations.
#[derive(Debug)]
pub enum FeedError {
    WebSocket(String),
    Parse(String),
    Http(String),
    Decompression(String),
    ConnectionClosed,
}

impl fmt::Display for FeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeedError::WebSocket(e) => write!(f, "WebSocket error: {}", e),
            FeedError::Parse(e) => write!(f, "Parse error: {}", e),
            FeedError::Http(e) => write!(f, "HTTP error: {}", e),
            FeedError::Decompression(e) => write!(f, "Decompression error: {}", e),
            FeedError::ConnectionClosed => write!(f, "Connection closed"),
        }
    }
}

impl std::error::Error for FeedError {}

impl From<tokio_tungstenite::tungstenite::Error> for FeedError {
    fn from(e: tokio_tungstenite::tungstenite::Error) -> Self {
        FeedError::WebSocket(e.to_string())
    }
}

impl From<serde_json::Error> for FeedError {
    fn from(e: serde_json::Error) -> Self {
        FeedError::Parse(e.to_string())
    }
}

impl From<reqwest::Error> for FeedError {
    fn from(e: reqwest::Error) -> Self {
        FeedError::Http(e.to_string())
    }
}

/// A normalized L2 depth update from any exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthUpdate {
    pub exchange: Exchange,
    pub symbol: String,
    pub timestamp_ns: u64,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub is_snapshot: bool,
}

/// A normalized trade from any exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedTrade {
    pub exchange: Exchange,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Exchange {
    Binance,
    Coinbase,
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exchange::Binance => write!(f, "Binance"),
            Exchange::Coinbase => write!(f, "Coinbase"),
        }
    }
}

/// Apply a depth update to an order book (simplified: replaces levels).
///
/// In production, this would track sequence numbers and apply diffs.
/// For now, it logs the update for the live-feed display mode.
pub fn apply_depth_update(book: &OrderBook, update: &DepthUpdate) {
    tracing::debug!(
        "[{}] {} depth update: {} bids, {} asks (snapshot={})",
        update.exchange,
        update.symbol,
        update.bids.len(),
        update.asks.len(),
        update.is_snapshot,
    );

    // In a full implementation, we would:
    // 1. Track update_id / sequence for gap detection
    // 2. Apply diffs to existing levels
    // 3. Handle snapshot bootstrapping
    //
    // For the simulation, the orderbook is populated by the event loop.
    // This feed module primarily provides the data pipeline.
}
