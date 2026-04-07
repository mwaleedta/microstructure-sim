//! Latency simulation models.
//!
//! Models realistic network and processing delays for:
//! - Feed latency: time for market data to reach our system
//! - Order latency: time for order submission to reach exchange
//! - Cancel latency: time for cancel request to reach exchange
//!
//! Three models provided:
//! - Fixed: constant delay (for testing)
//! - Gaussian: normal distribution with mean and std
//! - Empirical: samples from a recorded latency distribution

use rand::{Rng, RngCore};
use rand_distr::{Distribution, LogNormal, Normal};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// LatencyModel trait
// ---------------------------------------------------------------------------

/// Trait for generating latency samples.
pub trait LatencyModel: Send + Sync {
    /// Sample a latency value in nanoseconds.
    fn sample_ns(&self, rng: &mut dyn RngCore) -> u64;

    /// Get the expected (mean) latency in nanoseconds.
    fn mean_ns(&self) -> u64;

    /// Get a human-readable description.
    fn description(&self) -> String;
}

// ---------------------------------------------------------------------------
// FixedLatency
// ---------------------------------------------------------------------------

/// Constant latency -- useful for deterministic testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedLatency {
    pub latency_ns: u64,
}

impl FixedLatency {
    pub fn new(latency_ns: u64) -> Self {
        Self { latency_ns }
    }

    pub fn from_micros(us: u64) -> Self {
        Self {
            latency_ns: us * 1_000,
        }
    }

    pub fn from_millis(ms: u64) -> Self {
        Self {
            latency_ns: ms * 1_000_000,
        }
    }
}

impl LatencyModel for FixedLatency {
    fn sample_ns(&self, _rng: &mut dyn RngCore) -> u64 {
        self.latency_ns
    }

    fn mean_ns(&self) -> u64 {
        self.latency_ns
    }

    fn description(&self) -> String {
        format!("Fixed({} us)", self.latency_ns / 1_000)
    }
}

// ---------------------------------------------------------------------------
// GaussianLatency
// ---------------------------------------------------------------------------

/// Gaussian (normal) latency distribution.
/// Samples are clamped to be non-negative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianLatency {
    pub mean_ns: u64,
    pub std_ns: u64,
}

impl GaussianLatency {
    pub fn new(mean_ns: u64, std_ns: u64) -> Self {
        Self { mean_ns, std_ns }
    }

    pub fn from_micros(mean_us: u64, std_us: u64) -> Self {
        Self {
            mean_ns: mean_us * 1_000,
            std_ns: std_us * 1_000,
        }
    }
}

impl LatencyModel for GaussianLatency {
    fn sample_ns(&self, rng: &mut dyn RngCore) -> u64 {
        let dist =
            Normal::new(self.mean_ns as f64, self.std_ns as f64).unwrap();
        let sample = dist.sample(rng);
        sample.max(0.0) as u64
    }

    fn mean_ns(&self) -> u64 {
        self.mean_ns
    }

    fn description(&self) -> String {
        format!(
            "Gaussian(mean={} us, std={} us)",
            self.mean_ns / 1_000,
            self.std_ns / 1_000
        )
    }
}

// ---------------------------------------------------------------------------
// LogNormalLatency
// ---------------------------------------------------------------------------

/// Log-normal latency -- more realistic for network latencies
/// (right-skewed with occasional spikes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogNormalLatency {
    pub mu: f64,    // log-space mean
    pub sigma: f64, // log-space std
}

impl LogNormalLatency {
    /// Create from desired real-space mean and std (in nanoseconds).
    pub fn from_real_params(mean_ns: f64, std_ns: f64) -> Self {
        let variance = std_ns * std_ns;
        let mu = (mean_ns * mean_ns / (mean_ns * mean_ns + variance).sqrt()).ln();
        let sigma = (1.0 + variance / (mean_ns * mean_ns)).ln().sqrt();
        Self { mu, sigma }
    }
}

impl LatencyModel for LogNormalLatency {
    fn sample_ns(&self, rng: &mut dyn RngCore) -> u64 {
        let dist = LogNormal::new(self.mu, self.sigma).unwrap();
        dist.sample(rng) as u64
    }

    fn mean_ns(&self) -> u64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp() as u64
    }

    fn description(&self) -> String {
        format!(
            "LogNormal(mean={} us)",
            self.mean_ns() / 1_000
        )
    }
}

// ---------------------------------------------------------------------------
// EmpiricalLatency
// ---------------------------------------------------------------------------

/// Samples from a recorded latency distribution.
/// Feed it real latency measurements and it will replay them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalLatency {
    /// Sorted latency samples in nanoseconds.
    samples: Vec<u64>,
}

impl EmpiricalLatency {
    pub fn new(mut samples: Vec<u64>) -> Self {
        samples.sort_unstable();
        Self { samples }
    }

    /// Get a percentile value (0.0 - 1.0).
    pub fn percentile(&self, p: f64) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let idx = (p * self.samples.len() as f64).ceil() as usize;
        let idx = if idx == 0 { 0 } else { idx - 1 };
        self.samples[idx.min(self.samples.len() - 1)]
    }

    /// P50 latency.
    pub fn p50(&self) -> u64 {
        self.percentile(0.5)
    }

    /// P99 latency.
    pub fn p99(&self) -> u64 {
        self.percentile(0.99)
    }
}

impl LatencyModel for EmpiricalLatency {
    fn sample_ns(&self, rng: &mut dyn RngCore) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let idx = rng.gen_range(0..self.samples.len());
        self.samples[idx]
    }

    fn mean_ns(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum: u64 = self.samples.iter().sum();
        sum / self.samples.len() as u64
    }

    fn description(&self) -> String {
        format!(
            "Empirical(n={}, p50={} us, p99={} us)",
            self.samples.len(),
            self.p50() / 1_000,
            self.p99() / 1_000
        )
    }
}

// ---------------------------------------------------------------------------
// LatencySimulator: composite model for full system
// ---------------------------------------------------------------------------

/// Composite latency configuration for the full simulation.
pub struct LatencySimulator {
    pub feed_latency: Box<dyn LatencyModel>,
    pub order_latency: Box<dyn LatencyModel>,
    pub cancel_latency: Box<dyn LatencyModel>,
}

impl LatencySimulator {
    pub fn new(
        feed: Box<dyn LatencyModel>,
        order: Box<dyn LatencyModel>,
        cancel: Box<dyn LatencyModel>,
    ) -> Self {
        Self {
            feed_latency: feed,
            order_latency: order,
            cancel_latency: cancel,
        }
    }

    /// Zero-latency simulator for testing.
    pub fn zero_latency() -> Self {
        Self {
            feed_latency: Box::new(FixedLatency::new(0)),
            order_latency: Box::new(FixedLatency::new(0)),
            cancel_latency: Box::new(FixedLatency::new(0)),
        }
    }

    /// Typical crypto exchange latencies.
    pub fn typical_crypto() -> Self {
        Self {
            feed_latency: Box::new(GaussianLatency::from_micros(500, 100)),
            order_latency: Box::new(GaussianLatency::from_micros(1000, 200)),
            cancel_latency: Box::new(GaussianLatency::from_micros(800, 150)),
        }
    }

    /// Sample feed latency.
    pub fn sample_feed(&self, rng: &mut dyn RngCore) -> u64 {
        self.feed_latency.sample_ns(rng)
    }

    /// Sample order latency.
    pub fn sample_order(&self, rng: &mut dyn RngCore) -> u64 {
        self.order_latency.sample_ns(rng)
    }

    /// Sample cancel latency.
    pub fn sample_cancel(&self, rng: &mut dyn RngCore) -> u64 {
        self.cancel_latency.sample_ns(rng)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_fixed_latency() {
        let model = FixedLatency::from_micros(100);
        let mut rng = StdRng::seed_from_u64(42);

        assert_eq!(model.sample_ns(&mut rng), 100_000);
        assert_eq!(model.mean_ns(), 100_000);
    }

    #[test]
    fn test_gaussian_latency() {
        let model = GaussianLatency::from_micros(1000, 100);
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<u64> = (0..1000)
            .map(|_| model.sample_ns(&mut rng))
            .collect();

        let mean = samples.iter().sum::<u64>() as f64 / 1000.0;
        // Should be roughly 1000 us
        assert!((mean / 1_000.0 - 1000.0).abs() < 50.0);
    }

    #[test]
    fn test_empirical_latency() {
        let samples = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let model = EmpiricalLatency::new(samples);

        assert_eq!(model.p50(), 500);
        assert_eq!(model.p99(), 1000);
        assert_eq!(model.mean_ns(), 550);
    }

    #[test]
    fn test_latency_simulator() {
        let sim = LatencySimulator::typical_crypto();
        let mut rng = StdRng::seed_from_u64(42);

        let feed = sim.sample_feed(&mut rng);
        let order = sim.sample_order(&mut rng);
        let cancel = sim.sample_cancel(&mut rng);

        // All should be positive
        assert!(feed > 0 || order > 0 || cancel > 0);
    }
}
