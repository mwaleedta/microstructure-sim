//! # simulator
//!
//! Latency-aware discrete-event simulation engine.
//!
//! ## Modules
//!
//! - **`latency`** -- Latency models (fixed, Gaussian, empirical)
//! - **`queue_position`** -- Queue position estimation with parametric cancellation
//! - **`event_loop`** -- Discrete-event simulation engine with priority queue

pub mod latency;
pub mod queue_position;
pub mod event_loop;
