//! Coinbase WebSocket L2 channel handler.
//!
//! Connects to `wss://ws-feed.exchange.coinbase.com` and subscribes
//! to the `level2_batch` channel for order book updates.

use crate::{DepthUpdate, Exchange, FeedError, apply_depth_update};
use futures_util::{SinkExt, StreamExt};
use orderbook_core::book::OrderBook;
use orderbook_core::types::*;
use serde::{Deserialize, Serialize};
use tokio_tungstenite::connect_async;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Coinbase message types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct SubscribeMessage {
    #[serde(rename = "type")]
    msg_type: String,
    product_ids: Vec<String>,
    channels: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum CoinbaseMessage {
    #[serde(rename = "snapshot")]
    Snapshot {
        product_id: String,
        bids: Vec<(String, String)>,
        asks: Vec<(String, String)>,
    },
    #[serde(rename = "l2update")]
    L2Update {
        product_id: String,
        time: String,
        changes: Vec<(String, String, String)>, // [side, price, qty]
    },
    #[serde(rename = "subscriptions")]
    Subscriptions {
        channels: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Stream order book updates from Coinbase.
///
/// 1. Connects to the Coinbase WebSocket feed
/// 2. Subscribes to the level2_batch channel
/// 3. Receives snapshot + incremental updates
pub async fn stream_orderbook(
    symbol: &str,
    depth: usize,
    book: OrderBook,
) -> Result<(), FeedError> {
    let ws_url = "wss://ws-feed.exchange.coinbase.com";

    info!("Connecting to Coinbase WebSocket: {}", ws_url);

    let (ws_stream, _) = connect_async(ws_url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to L2 channel
    let subscribe = SubscribeMessage {
        msg_type: "subscribe".to_string(),
        product_ids: vec![symbol.to_uppercase()],
        channels: vec!["level2_batch".to_string()],
    };

    let sub_json = serde_json::to_string(&subscribe)?;
    write
        .send(tokio_tungstenite::tungstenite::Message::Text(sub_json))
        .await?;

    info!("Subscribed to Coinbase L2 for {}", symbol);

    let mut update_count: u64 = 0;

    while let Some(msg) = read.next().await {
        let msg = msg?;
        let text = match msg {
            tokio_tungstenite::tungstenite::Message::Text(t) => t,
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(_) => {
                warn!("Coinbase WebSocket closed");
                return Err(FeedError::ConnectionClosed);
            }
            _ => continue,
        };

        let cb_msg: CoinbaseMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                debug!("Failed to parse Coinbase message: {}", e);
                continue;
            }
        };

        match cb_msg {
            CoinbaseMessage::Snapshot { product_id, bids, asks } => {
                info!(
                    "Coinbase snapshot for {}: {} bids, {} asks",
                    product_id,
                    bids.len(),
                    asks.len()
                );

                let depth_update = DepthUpdate {
                    exchange: Exchange::Coinbase,
                    symbol: product_id,
                    timestamp_ns: now_ns(),
                    bids: parse_levels(&bids),
                    asks: parse_levels(&asks),
                    is_snapshot: true,
                };
                apply_depth_update(&book, &depth_update);
            }
            CoinbaseMessage::L2Update { product_id, time: _, changes } => {
                let mut bid_updates = Vec::new();
                let mut ask_updates = Vec::new();

                for (side, price_str, qty_str) in &changes {
                    let price: f64 = match price_str.parse() {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    let qty: f64 = match qty_str.parse() {
                        Ok(q) => q,
                        Err(_) => continue,
                    };

                    match side.as_str() {
                        "buy" => bid_updates.push((Price::from_f64(price), Quantity::from_f64(qty))),
                        "sell" => ask_updates.push((Price::from_f64(price), Quantity::from_f64(qty))),
                        _ => {
                            debug!("Unknown side in Coinbase L2 update: {}", side);
                        }
                    }
                }

                let depth_update = DepthUpdate {
                    exchange: Exchange::Coinbase,
                    symbol: product_id,
                    timestamp_ns: now_ns(),
                    bids: bid_updates,
                    asks: ask_updates,
                    is_snapshot: false,
                };
                apply_depth_update(&book, &depth_update);

                update_count += 1;
                if update_count % 100 == 0 {
                    let mid = book.mid_price().unwrap_or(0.0);
                    let spread = book.spread().map(|s| s.to_f64()).unwrap_or(0.0);
                    info!(
                        "[Coinbase/{}] updates={}, mid={:.2}, spread={:.8}, orders={}",
                        symbol,
                        update_count,
                        mid,
                        spread,
                        book.total_orders()
                    );
                }
            }
            CoinbaseMessage::Subscriptions { channels } => {
                info!("Coinbase subscriptions confirmed: {:?}", channels);
            }
            CoinbaseMessage::Other => {
                debug!("Received unknown Coinbase message type");
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse string price/quantity pairs into (Price, Quantity) tuples.
fn parse_levels(raw: &[(String, String)]) -> Vec<(Price, Quantity)> {
    raw.iter()
        .filter_map(|(price_str, qty_str)| {
            let price: f64 = price_str.parse().ok()?;
            let qty: f64 = qty_str.parse().ok()?;
            Some((Price::from_f64(price), Quantity::from_f64(qty)))
        })
        .collect()
}
