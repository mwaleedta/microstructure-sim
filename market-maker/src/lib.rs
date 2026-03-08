//! # market-maker
//!
//! Avellaneda-Stoikov market making engine with inventory risk management.
//!
//! ## Modules
//!
//! - **`avellaneda_stoikov`** -- Core A-S model: reservation price, optimal spread
//! - **`inventory`** -- Position tracking, risk limits, skew scaling
//! - **`quoting`** -- Quote management, flickering prevention, tick rounding

pub mod avellaneda_stoikov;
pub mod inventory;
pub mod quoting;
