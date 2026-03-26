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
use tracing::{debug, error, info, warn};

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

        let message: CoinbaseMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                debug!("Skipping unparseable Coinbase message: {}", e);
                continue;
            }
        };

        match message {
            CoinbaseMessage::Snapshot {
                product_id,
                bids,
                asks,
            } => {
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
            CoinbaseMessage::L2Update {
                product_id,
                time,
                changes,
            } => {
                let mut bid_updates = Vec::new();
                let mut ask_updates = Vec::new();

                for (side, price, qty) in &changes {
                    let p = price.parse::<f64>().unwrap_or(0.0);
                    let q = qty.parse::<f64>().unwrap_or(0.0);
                    match side.as_str() {
                        "buy" => bid_updates.push((p, q)),
                        "sell" => ask_updates.push((p, q)),
                        _ => {}
                    }
                }

                let timestamp_ns = chrono::DateTime::parse_from_rfc3339(&time)
                    .map(|dt| dt.timestamp_nanos_opt().unwrap_or(0) as u64)
                    .unwrap_or_else(|_| now_ns());

                let depth_update = DepthUpdate {
                    exchange: Exchange::Coinbase,
                    symbol: product_id,
                    timestamp_ns,
                    bids: bid_updates,
                    asks: ask_updates,
                    is_snapshot: false,
                };
                apply_depth_update(&book, &depth_update);

                update_count += 1;
                if update_count % 100 == 0 {
                    let (bids, asks) = book.get_depth(5);
                    if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
                        info!(
                            "[Coinbase] Updates: {} | BBO: {} / {} | Spread: {:.8}",
                            update_count,
                            bb.0,
                            ba.0,
                            ba.0.to_f64() - bb.0.to_f64()
                        );
                    }
                }
            }
            CoinbaseMessage::Subscriptions { channels } => {
                info!("Coinbase subscription confirmed: {:?}", channels);
            }
            CoinbaseMessage::Other => {}
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_levels(raw: &[(String, String)]) -> Vec<(f64, f64)> {
    raw.iter()
        .filter_map(|(price, qty)| {
            let p = price.parse::<f64>().ok()?;
            let q = qty.parse::<f64>().ok()?;
            Some((p, q))
        })
        .collect()
}
