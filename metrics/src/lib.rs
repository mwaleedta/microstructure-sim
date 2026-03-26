//! # metrics-engine
//!
//! Real-time performance and market microstructure metrics.
//!
//! ## Modules
//!
//! - **`performance`** -- Sharpe, Sortino, drawdown, PnL tracking (Welford's online algorithm)
//! - **`microstructure`** -- Fill rate, effective spread, markouts, PnL attribution
//! - **`reporting`** -- Terminal dashboard + JSON/CSV export

pub mod performance;
pub mod microstructure;
pub mod reporting;

pub use performance::{PerformanceSnapshot, PerformanceTracker};
pub use microstructure::{MicrostructureSnapshot, MicrostructureTracker};
pub use reporting::{MetricsReport, MetricsReporter, ReportingConfig};
