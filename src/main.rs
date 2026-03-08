//! microstructure-sim: High-Performance Limit Order Book + Market Microstructure Simulator
//!
//! Usage:
//!   microstructure-sim backtest --symbol BTC-USD --strategy market-maker
//!   microstructure-sim live-feed --exchange binance --symbol btcusdt
//!   microstructure-sim paper --exchange coinbase --symbol BTC-USD --strategy stat-arb

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Parser)]
#[command(name = "microstructure-sim")]
#[command(about = "High-Performance Limit Order Book + Market Microstructure Simulator")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a backtest simulation
    Backtest {
        #[arg(long, default_value = "BTC-USD")]
        symbol: String,
        #[arg(long, default_value = "market-maker")]
        strategy: StrategyType,
        #[arg(long)]
        data_file: Option<String>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 0.1)]
        gamma: f64,
        #[arg(long, default_value_t = 1.5)]
        kappa: f64,
    },
    /// Connect to live exchange feeds (read-only)
    LiveFeed {
        #[arg(long, default_value = "binance")]
        exchange: ExchangeChoice,
        #[arg(long, default_value = "btcusdt")]
        symbol: String,
        #[arg(long, default_value_t = 20)]
        depth: usize,
    },
    /// Paper trading with live data
    Paper {
        #[arg(long, default_value = "binance")]
        exchange: ExchangeChoice,
        #[arg(long, default_value = "btcusdt")]
        symbol: String,
        #[arg(long, default_value = "market-maker")]
        strategy: StrategyType,
    },
}

#[derive(Clone, ValueEnum)]
enum ExchangeChoice { Binance, Coinbase }

#[derive(Clone, ValueEnum)]
enum StrategyType { MarketMaker, StatArb }

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = EnvFilter::try_new(&cli.log_level).unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();

    tracing::info!("microstructure-sim v0.1.0 starting...");

    match cli.command {
        Commands::Backtest { symbol, strategy, data_file, seed, gamma, kappa } => {
            tracing::info!(
                "Backtest mode: symbol={}, strategy={:?}, seed={}",
                symbol, strategy as u8, seed
            );

            let book = orderbook_core::book::OrderBook::with_symbol(&symbol);
            let latency = simulator::latency::LatencySimulator::typical_crypto();
            let sim_config = simulator::event_loop::SimulationConfig {
                seed,
                timer_interval_ns: 100_000_000, // 100ms ticks
                max_events: 10_000_000,
            };

            let mut engine = simulator::event_loop::SimulationEngine::new(sim_config, latency);

            tracing::info!("Order book initialized for {}", symbol);
            tracing::info!("Simulation engine ready. Load market data to begin.");

            // TODO: Load data from file and run simulation
            // For now, print configuration
            println!("Backtest configured:");
            println!("  Symbol: {}", symbol);
            println!("  Gamma: {}", gamma);
            println!("  Kappa: {}", kappa);
            println!("  Seed: {}", seed);
            if let Some(f) = data_file {
                println!("  Data file: {}", f);
            }
        }

        Commands::LiveFeed { exchange, symbol, depth } => {
            tracing::info!("Live feed mode: {:?} {}", exchange as u8, symbol);

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

            match exchange {
                ExchangeChoice::Binance => {
                    let config = exchange_feed::binance::BinanceConfig {
                        symbol: symbol.clone(),
                        ..Default::default()
                    };
                    let mut handler = exchange_feed::binance::BinanceFeedHandler::new(config, tx);
                    let ws_config = handler.ws_config();

                    // Spawn snapshot fetcher
                    let snapshot_handle = tokio::spawn(async move {
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                        if let Err(e) = handler.fetch_snapshot().await {
                            tracing::error!("Snapshot fetch failed: {}", e);
                        }
                    });

                    tracing::info!("Connecting to Binance {} feed...", symbol);
                }
                ExchangeChoice::Coinbase => {
                    tracing::info!("Connecting to Coinbase {} feed...", symbol);
                }
            }

            // Print incoming updates
            tracing::info!("Listening for book updates...");
            while let Some(update) = rx.recv().await {
                match update {
                    exchange_feed::normalizer::BookUpdate::L2(u) => {
                        if let (Some(bb), Some(ba)) = (u.bids.first(), u.asks.first()) {
                            let latency_us = u.local_ts_ns.saturating_sub(u.exchange_ts_ns) / 1000;
                            println!(
                                "[{}] {} bid={} ask={} spread={:.8} latency={}us",
                                u.exchange, u.symbol, bb.0, ba.0,
                                ba.0.to_f64() - bb.0.to_f64(), latency_us
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        Commands::Paper { exchange, symbol, strategy } => {
            tracing::info!("Paper trading mode");
            println!("Paper trading not yet implemented. Use backtest mode.");
        }
    }

    // Graceful shutdown
    tracing::info!("Shutting down...");
}
