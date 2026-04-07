//! Normalized book update types.

use crate::DepthUpdate;

/// Normalized book update enum.
#[derive(Debug)]
pub enum BookUpdate {
    L2(DepthUpdate),
    Trade,
    Heartbeat,
}
