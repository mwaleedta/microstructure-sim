//! Binance WebSocket depth stream handler.
//!
//! Connects to `wss://stream.binance.com:9443/ws/{symbol}@depth{levels}@100ms`
//! and applies partial book depth updates. Bootstraps with a REST snapshot
//! from `https://api.binance.com/api/v3/depth`.

use crate::{DepthUpdate, Exchange, FeedError, apply_depth_update};
use crypto_orderbook::Market;
use crypto_orderbook::orderbook::sync::{Live, live};
use futures_util::{SinkExt, StreamExt};
use orderbook_core::book::OrderBook;
use orderbook_core::types::*;
use serde::Deserialize;
use tokio_tungstenite::connect_async;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Binance message types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BinanceDepthUpdate {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "U")]
    first_update_id: u64,
    #[serde(rename = "u")]
    final_update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<(String, String)>,
    #[serde(rename = "a")]
    asks: Vec<(String, String)>,
}

#[derive(Debug, Deserialize)]
struct BinanceSnapshot {
    #[serde(rename = "lastUpdateId")]
    last_update_id: u64,
    bids: Vec<(String, String)>,
    asks: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Stream order book updates from Binance.
///
/// 1. Connects to the WebSocket depth stream
/// 2. Fetches an initial REST snapshot
/// 3. Applies buffered + live updates to the book
pub async fn stream_orderbook(
    symbol: &str,
    depth: usize,
    book: OrderBook,
) -> Result<(), FeedError> {
    let ws_symbol = symbol.to_lowercase().replace('-', "");
    let ws_url = format!(
        "wss://stream.binance.com:9443/ws/{}@depth{}@100ms",
        ws_symbol, depth
    );

    info!("Connecting to Binance WebSocket: {}", ws_url);

    let (ws_stream, _) = connect_async(&ws_url).await?;
    let (mut _write, mut read) = ws_stream.split();

    // Fetch REST snapshot for initial book state
    let snapshot = fetch_snapshot(&ws_symbol, depth).await?;
    info!(
        "Binance snapshot: {} bids, {} asks (lastUpdateId={})",
        snapshot.bids.len(),
        snapshot.asks.len(),
        snapshot.last_update_id
    );

    let initial_update = DepthUpdate {
        exchange: Exchange::Binance,
        symbol: symbol.to_string(),
        timestamp_ns: now_ns(),
        bids: parse_levels(&snapshot.bids),
        asks: parse_levels(&snapshot.asks),
        is_snapshot: true,
    };
    apply_depth_update(&book, &initial_update);

    let mut last_update_id = snapshot.last_update_id;
    let mut update_count: u64 = 0;

    info!("Streaming Binance depth updates for {}...", symbol);

    while let Some(msg) = read.next().await {
        let msg = msg?;
        let text = match msg {
            tokio_tungstenite::tungstenite::Message::Text(t) => t,
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(_) => {
                warn!("Binance WebSocket closed");
                return Err(FeedError::ConnectionClosed);
            }
            _ => continue,
        };

        let update: BinanceDepthUpdate = serde_json::from_str(&text)?;

        match sequence(last_update_id, &update) {
            // Drop stale updates (before our snapshot)
            Live::Ignore => continue,
            Live::Gap { expected, got } => {
                warn!(
                    "Binance gap detected: expected {}, got {}. Re-snapshotting.",
                    expected, got
                );
                let snapshot = fetch_snapshot(&ws_symbol, depth).await?;
                apply_depth_update(
                    &book,
                    &DepthUpdate {
                        exchange: Exchange::Binance,
                        symbol: symbol.to_string(),
                        timestamp_ns: now_ns(),
                        bids: parse_levels(&snapshot.bids),
                        asks: parse_levels(&snapshot.asks),
                        is_snapshot: true,
                    },
                );
                last_update_id = snapshot.last_update_id;
                continue;
            }
            Live::Apply => {}
        }

        last_update_id = update.final_update_id;

        let depth_update = DepthUpdate {
            exchange: Exchange::Binance,
            symbol: update.symbol.clone(),
            timestamp_ns: update.event_time * 1_000_000, // ms -> ns
            bids: parse_levels(&update.bids),
            asks: parse_levels(&update.asks),
            is_snapshot: false,
        };

        apply_depth_update(&book, &depth_update);

        update_count += 1;
        if update_count % 100 == 0 {
            let (bids, asks) = book.get_depth(5);
            if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
                info!(
                    "[Binance] Updates: {} | BBO: {} / {} | Spread: {:.8}",
                    update_count,
                    bb.0,
                    ba.0,
                    ba.0.to_f64() - bb.0.to_f64()
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn fetch_snapshot(
    symbol: &str,
    limit: usize,
) -> Result<BinanceSnapshot, FeedError> {
    let url = format!(
        "https://api.binance.com/api/v3/depth?symbol={}&limit={}",
        symbol.to_uppercase(),
        limit.min(5000)
    );
    let resp = reqwest::get(&url).await?.json::<BinanceSnapshot>().await?;
    Ok(resp)
}

/// Binance spot sequencing, decided by `crypto-orderbook`: an event must
/// start at or before `last_update_id + 1`, otherwise the book has a hole.
fn sequence(last_update_id: u64, update: &BinanceDepthUpdate) -> Live {
    let ids = crypto_orderbook::DepthUpdate {
        first_update_id: update.first_update_id,
        last_update_id: update.final_update_id,
        prev_last_update_id: None,
        bids: Vec::new(),
        asks: Vec::new(),
    };
    live(Market::Spot, last_update_id, &ids)
}

fn parse_levels(raw: &[(String, String)]) -> Vec<(f64, f64)> {
    raw.iter()
        .filter_map(|(price, qty)| {
            let p = price.parse::<f64>().ok()?;
            let q = qty.parse::<f64>().ok()?;
            Some((p, q))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stub types referenced by main.rs
// ---------------------------------------------------------------------------

/// Configuration for the Binance feed handler.
#[derive(Debug, Clone)]
pub struct BinanceConfig {
    pub symbol: String,
    pub depth: usize,
}

impl Default for BinanceConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            depth: 20,
        }
    }
}

/// Feed handler that manages a Binance WebSocket connection.
pub struct BinanceFeedHandler {
    config: BinanceConfig,
    tx: tokio::sync::mpsc::UnboundedSender<crate::normalizer::BookUpdate>,
}

impl BinanceFeedHandler {
    pub fn new(
        config: BinanceConfig,
        tx: tokio::sync::mpsc::UnboundedSender<crate::normalizer::BookUpdate>,
    ) -> Self {
        Self { config, tx }
    }

    pub fn ws_config(&self) -> String {
        format!(
            "wss://stream.binance.com:9443/ws/{}@depth{}@100ms",
            self.config.symbol.to_lowercase(),
            self.config.depth
        )
    }

    pub async fn fetch_snapshot(&mut self) -> Result<(), FeedError> {
        let snapshot = fetch_snapshot(&self.config.symbol, self.config.depth).await?;
        let update = DepthUpdate {
            exchange: Exchange::Binance,
            symbol: self.config.symbol.clone(),
            timestamp_ns: now_ns(),
            bids: parse_levels(&snapshot.bids),
            asks: parse_levels(&snapshot.asks),
            is_snapshot: true,
        };
        let _ = self.tx.send(crate::normalizer::BookUpdate::L2(update));
        Ok(())
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(first: u64, last: u64) -> BinanceDepthUpdate {
        BinanceDepthUpdate {
            event_type: "depthUpdate".into(),
            event_time: 0,
            symbol: "BTCUSDT".into(),
            first_update_id: first,
            final_update_id: last,
            bids: Vec::new(),
            asks: Vec::new(),
        }
    }

    #[test]
    fn gap_forces_resnapshot_instead_of_applying() {
        assert_eq!(sequence(102, &event(101, 102)), Live::Ignore);
        assert_eq!(sequence(100, &event(101, 102)), Live::Apply);
        // 103..=104 was dropped; 105 must not be applied on top.
        assert_eq!(
            sequence(102, &event(105, 105)),
            Live::Gap {
                expected: 103,
                got: 105
            }
        );
    }
}
