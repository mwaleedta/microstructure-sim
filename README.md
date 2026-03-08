# microstructure-sim

**High-Performance Limit Order Book + Market Microstructure Simulator in Rust**

A full market microstructure research platform — not just another matching engine.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           microstructure-sim         │
                    │          (binary / CLI entry)        │
                    └────────────────┬────────────────────┘
                                     │
         ┌───────────┬───────────────┼───────────────┬──────────┐
         │           │               │               │          │
    ┌────▼────┐ ┌────▼─────┐  ┌─────▼──────┐  ┌────▼────┐ ┌───▼────┐
    │orderbook│ │ market-  │  │  simulator  │  │exchange │ │  stat  │
    │  -core  │ │  maker   │  │             │  │  -feed  │ │  -arb  │
    └─────────┘ └──────────┘  └─────────────┘  └─────────┘ └────────┘
         │           │               │               │          │
    ┌────▼───────────▼───────────────▼───────────────▼──────────▼────┐
    │                         metrics-engine                         │
    └────────────────────────────────────────────────────────────────┘
```

## Modules

| Crate | Description |
|-------|-------------|
| **orderbook-core** | Lock-free limit order book with price-time priority, SIMD-optimized hot paths, cache-line-aligned structures |
| **market-maker** | Avellaneda-Stoikov optimal market making with inventory risk management and quote lifecycle |
| **simulator** | Latency-aware discrete-event simulation, queue position estimation, deterministic replay |
| **exchange-feed** | Live L2/L3 order book reconstruction from Binance and Coinbase WebSocket feeds |
| **stat-arb** | Cointegration analysis (ADF, Engle-Granger), Kalman filter hedge ratios, pairs trading signals |
| **metrics-engine** | Sharpe/Sortino/drawdown, fill rate, adverse selection markouts, PnL attribution, terminal dashboard |

## Build

```bash
cargo build --release
```

## Usage

### Backtest Mode
```bash
cargo run --release -- backtest \
  --symbol BTC-USD \
  --strategy market-maker \
  --gamma 0.1 \
  --kappa 1.5 \
  --seed 42
```

### Live Feed (Read-Only)
```bash
cargo run --release -- live-feed \
  --exchange binance \
  --symbol btcusdt \
  --depth 20
```

### Paper Trading
```bash
cargo run --release -- paper \
  --exchange binance \
  --symbol btcusdt \
  --strategy market-maker
```

## Parameter Tuning

### Avellaneda-Stoikov (Market Making)

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| `gamma` | Risk aversion | 0.01 - 1.0 | Higher = wider spreads, lower inventory risk |
| `kappa` | Order arrival decay | 1.0 - 100.0 | Calibrate from fill data; higher = steeper fill decay |
| `sigma` | Volatility | Auto-estimated | EWMA of realized variance |
| `T` | Session duration | 3600 - 86400s | Crypto: 86400 (24h), TradFi: 23400 (6.5h) |

**Tuning strategy:**
1. Start with `gamma=0.1` (moderate risk aversion)
2. Calibrate `kappa` from 1 week of fill data using the built-in MLE calibrator
3. Adjust `gamma` based on max drawdown tolerance
4. Monitor fill rate — if < 10%, reduce gamma; if > 50%, increase gamma

### Statistical Arbitrage

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `R` (obs_variance) | Observation noise | 0.01 - 10.0 |
| `Q` (state_variance) | State transition noise | 1e-6 - 1e-3 |
| `entry_z` | Entry z-score threshold | 1.5 - 3.0 |
| `exit_z` | Exit z-score threshold | 0.0 - 1.0 |
| `stop_loss_z` | Stop-loss z-score | 3.0 - 5.0 |

**Tuning strategy:**
1. Higher `Q/R` ratio = faster adaptation to regime changes, more noise
2. Start with `entry_z=2.0`, `exit_z=0.5`
3. Verify cointegration half-life is 5-200 periods
4. Use Kelly fraction cap of 0.25 (quarter-Kelly for safety)

## Key Design Decisions

- **Fixed-point arithmetic**: All prices/quantities use `i64`/`u64` with 10^8 decimal shift — eliminates float non-determinism in the matching engine
- **Lock-free data structures**: `crossbeam_skiplist::SkipMap` for price levels, `DashMap` for order lookup, `crossbeam-epoch` for memory reclamation
- **Cache-line alignment**: `#[repr(align(64))]` on `PriceLevel` prevents false sharing
- **Parametric queue model**: Queue position estimation using `p(x) = x^(1+k)` cancellation bias — far more realistic than naive models
- **Latency modeling**: Separate feed/order/cancel latency with Gaussian, log-normal, or empirical distributions

## Performance Targets

| Operation | Target Latency |
|-----------|---------------|
| Order insert | < 500ns |
| Order cancel | < 300ns |
| Match (per level) | < 200ns |
| VWAP (20 levels) | < 100ns |
| Depth snapshot | < 1us |

## References

1. Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.
2. Gueant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2012). "Dealing with the Inventory Risk: A Solution to the Market Making Problem." *Mathematics and Financial Economics*.
3. Cont, R., Stoikov, S., & Talreja, R. (2010). "A Stochastic Model for Order Book Dynamics." *Operations Research*.
4. Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. (Cointegration, ADF test)
5. Harvey, A.C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

## License

MIT
