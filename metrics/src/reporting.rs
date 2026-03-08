//! Real-time metrics reporting: terminal dashboard + JSON/CSV export.

use crate::microstructure::MicrostructureSnapshot;
use crate::performance::PerformanceSnapshot;
use serde::{Deserialize, Serialize};
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub timestamp: String,
    pub performance: PerformanceSnapshot,
    pub microstructure: MicrostructureSnapshot,
    pub inventory_position: f64,
    pub current_bid: f64,
    pub current_ask: f64,
    pub mid_price: f64,
}

#[derive(Debug, Clone)]
pub struct ReportingConfig {
    pub terminal_enabled: bool,
    pub json_export_path: Option<String>,
    pub csv_export_path: Option<String>,
    pub report_interval_ms: u64,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            terminal_enabled: true,
            json_export_path: None,
            csv_export_path: None,
            report_interval_ms: 1000,
        }
    }
}

pub struct MetricsReporter {
    config: ReportingConfig,
    reports: Vec<MetricsReport>,
    last_report_ms: u64,
}

impl MetricsReporter {
    pub fn new(config: ReportingConfig) -> Self {
        Self { config, reports: Vec::new(), last_report_ms: 0 }
    }

    pub fn should_report(&self, current_ms: u64) -> bool {
        current_ms.saturating_sub(self.last_report_ms) >= self.config.report_interval_ms
    }

    pub fn report(&mut self, report: MetricsReport, current_ms: u64) {
        self.last_report_ms = current_ms;

        if self.config.terminal_enabled {
            self.print_terminal(&report);
        }

        self.reports.push(report);
    }

    fn print_terminal(&self, r: &MetricsReport) {
        print!("\x1B[2J\x1B[H"); // Clear screen
        println!("=== Microstructure Simulator Dashboard ===");
        println!("Time: {}", r.timestamp);
        println!();
        println!("--- Position ---");
        println!("  Inventory:  {:>12.4}", r.inventory_position);
        println!("  Bid/Ask:    {:>12.4} / {:.4}", r.current_bid, r.current_ask);
        println!("  Mid:        {:>12.4}", r.mid_price);
        println!();
        println!("--- Performance ---");
        println!("  PnL:        {:>12.2}", r.performance.cumulative_pnl);
        println!("  Sharpe:     {:>12.4}", r.performance.sharpe_ratio);
        println!("  Sortino:    {:>12.4}", r.performance.sortino_ratio);
        println!("  Max DD:     {:>12.2} ({:.1}%)", r.performance.max_drawdown, r.performance.max_drawdown_pct * 100.0);
        println!("  Win Rate:   {:>12.1}%", r.performance.win_rate * 100.0);
        println!("  Trades:     {:>12}", r.performance.total_trades);
        println!();
        println!("--- Microstructure ---");
        println!("  Fill Rate:  {:>12.1}%", r.microstructure.fill_rate * 100.0);
        println!("  Eff Spread: {:>12.6}", r.microstructure.effective_spread);
        println!("  Queue RMSE: {:>12.4}", r.microstructure.queue_accuracy_rmse);
        println!("  Markout 1s: {:>12.4}", r.microstructure.markout_1s);
        println!("  Markout 5m: {:>12.4}", r.microstructure.markout_5min);
        println!();
        println!("--- PnL Attribution ---");
        println!("  Spread:     {:>12.2}", r.microstructure.pnl_spread_capture);
        println!("  Adverse:    {:>12.2}", r.microstructure.pnl_adverse_selection);
    }

    pub fn export_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.reports)?;
        std::fs::write(path, json)
    }

    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "timestamp,pnl,sharpe,sortino,max_dd,win_rate,fill_rate,eff_spread,markout_1s")?;
        for r in &self.reports {
            writeln!(f, "{},{:.2},{:.4},{:.4},{:.2},{:.4},{:.4},{:.6},{:.4}",
                r.timestamp, r.performance.cumulative_pnl, r.performance.sharpe_ratio,
                r.performance.sortino_ratio, r.performance.max_drawdown,
                r.performance.win_rate, r.microstructure.fill_rate,
                r.microstructure.effective_spread, r.microstructure.markout_1s)?;
        }
        Ok(())
    }
}
