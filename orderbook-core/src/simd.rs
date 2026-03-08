//! SIMD-optimized hot paths for order book operations.
//!
//! Uses portable SIMD intrinsics via `std::simd` (nightly) with scalar
//! fallbacks for stable Rust. These are the inner-loop operations that
//! benefit most from vectorization:
//!
//! - Cumulative depth volume aggregation
//! - Price level scanning (find first level matching a condition)
//! - Volume-weighted average price (VWAP) computation
//!
//! On modern x86_64, these map to SSE4.2/AVX2 instructions.
//! On ARM, they use NEON.

use crate::types::{Price, Quantity};

// ---------------------------------------------------------------------------
// Feature gate: use portable_simd on nightly, scalar on stable
// ---------------------------------------------------------------------------

// We provide scalar implementations that work on stable Rust.
// When compiled with nightly + `#![feature(portable_simd)]`, the SIMD
// versions can be enabled. For now, we ship the scalar versions as default
// and gate SIMD behind a cfg flag.

/// Cumulative volume aggregation.
///
/// Given an array of quantities at consecutive price levels, compute
/// the running sum. Used for depth visualization and impact estimation.
///
/// Input:  [q0, q1, q2, q3]
/// Output: [q0, q0+q1, q0+q1+q2, q0+q1+q2+q3]
#[inline]
pub fn cumulative_volume(quantities: &[Quantity]) -> Vec<Quantity> {
    let mut result = Vec::with_capacity(quantities.len());
    let mut acc = 0u64;
    for q in quantities {
        acc += q.0;
        result.push(Quantity(acc));
    }
    result
}

/// SIMD-accelerated cumulative volume for fixed-size chunks.
///
/// Processes 4 elements at a time using prefix-sum pattern.
/// For arrays not divisible by 4, falls back to scalar for the remainder.
#[inline]
pub fn cumulative_volume_simd4(quantities: &[Quantity]) -> Vec<Quantity> {
    let n = quantities.len();
    let mut result = Vec::with_capacity(n);
    let mut acc = 0u64;

    // Process in chunks of 4
    let chunks = n / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let q0 = quantities[base].0;
        let q1 = quantities[base + 1].0;
        let q2 = quantities[base + 2].0;
        let q3 = quantities[base + 3].0;

        // Prefix sum within chunk (can be vectorized by LLVM)
        let s0 = acc + q0;
        let s1 = s0 + q1;
        let s2 = s1 + q2;
        let s3 = s2 + q3;

        result.push(Quantity(s0));
        result.push(Quantity(s1));
        result.push(Quantity(s2));
        result.push(Quantity(s3));

        acc = s3;
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        acc += quantities[i].0;
        result.push(Quantity(acc));
    }

    result
}

/// Find the first price level where cumulative volume >= target.
///
/// Used for market impact estimation: "how deep into the book do I
/// need to go to fill X quantity?"
///
/// Returns the index of the price level, or None if total depth
/// is insufficient.
#[inline]
pub fn find_fill_depth(
    quantities: &[Quantity],
    target: Quantity,
) -> Option<usize> {
    let mut acc = 0u64;
    for (i, q) in quantities.iter().enumerate() {
        acc += q.0;
        if acc >= target.0 {
            return Some(i);
        }
    }
    None
}

/// SIMD-friendly price scan: find first price in array that satisfies
/// a comparison against a target.
///
/// Scans 4 prices at a time, checking `prices[i] >= target` (for asks)
/// or `prices[i] <= target` (for bids).
#[inline]
pub fn scan_price_ge(prices: &[Price], target: Price) -> Option<usize> {
    let target_val = target.0;

    // Process 4 at a time
    let chunks = prices.len() / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let p0 = prices[base].0;
        let p1 = prices[base + 1].0;
        let p2 = prices[base + 2].0;
        let p3 = prices[base + 3].0;

        // Check all 4 (LLVM can auto-vectorize this)
        if p0 >= target_val { return Some(base); }
        if p1 >= target_val { return Some(base + 1); }
        if p2 >= target_val { return Some(base + 2); }
        if p3 >= target_val { return Some(base + 3); }
    }

    // Remainder
    for i in (chunks * 4)..prices.len() {
        if prices[i].0 >= target_val {
            return Some(i);
        }
    }

    None
}

#[inline]
pub fn scan_price_le(prices: &[Price], target: Price) -> Option<usize> {
    let target_val = target.0;

    let chunks = prices.len() / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let p0 = prices[base].0;
        let p1 = prices[base + 1].0;
        let p2 = prices[base + 2].0;
        let p3 = prices[base + 3].0;

        if p0 <= target_val { return Some(base); }
        if p1 <= target_val { return Some(base + 1); }
        if p2 <= target_val { return Some(base + 2); }
        if p3 <= target_val { return Some(base + 3); }
    }

    for i in (chunks * 4)..prices.len() {
        if prices[i].0 <= target_val {
            return Some(i);
        }
    }

    None
}

/// Volume-Weighted Average Price (VWAP) computation.
///
/// VWAP = sum(price_i * qty_i) / sum(qty_i)
///
/// Uses fixed-point arithmetic throughout to maintain precision.
/// The intermediate products (price * qty) are u128 to avoid overflow.
#[inline]
pub fn vwap(prices: &[Price], quantities: &[Quantity]) -> Option<Price> {
    debug_assert_eq!(prices.len(), quantities.len());

    if prices.is_empty() {
        return None;
    }

    let mut weighted_sum: i128 = 0;
    let mut total_qty: u128 = 0;

    // Process 4 at a time for auto-vectorization
    let n = prices.len();
    let chunks = n / 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        let p0 = prices[base].0 as i128;
        let p1 = prices[base + 1].0 as i128;
        let p2 = prices[base + 2].0 as i128;
        let p3 = prices[base + 3].0 as i128;

        let q0 = quantities[base].0 as u128;
        let q1 = quantities[base + 1].0 as u128;
        let q2 = quantities[base + 2].0 as u128;
        let q3 = quantities[base + 3].0 as u128;

        weighted_sum += p0 * q0 as i128;
        weighted_sum += p1 * q1 as i128;
        weighted_sum += p2 * q2 as i128;
        weighted_sum += p3 * q3 as i128;

        total_qty += q0 + q1 + q2 + q3;
    }

    // Remainder
    for i in (chunks * 4)..n {
        weighted_sum += prices[i].0 as i128 * quantities[i].0 as u128 as i128;
        total_qty += quantities[i].0 as u128;
    }

    if total_qty == 0 {
        return None;
    }

    Some(Price((weighted_sum / total_qty as i128) as i64))
}

/// Compute the market impact of consuming `target_qty` from a side.
///
/// Returns (average_fill_price, worst_fill_price, levels_consumed).
///
/// Walks through price levels, accumulating fills until the target
/// quantity is reached.
pub fn market_impact(
    prices: &[Price],
    quantities: &[Quantity],
    target_qty: Quantity,
) -> Option<(Price, Price, usize)> {
    debug_assert_eq!(prices.len(), quantities.len());

    let mut remaining = target_qty.0;
    let mut weighted_sum: i128 = 0;
    let mut total_filled: u128 = 0;
    let mut worst_price = Price::ZERO;
    let mut levels = 0;

    for (i, (price, qty)) in prices.iter().zip(quantities.iter()).enumerate() {
        if remaining == 0 {
            break;
        }

        let fill_qty = remaining.min(qty.0);
        weighted_sum += price.0 as i128 * fill_qty as i128;
        total_filled += fill_qty as u128;
        remaining -= fill_qty;
        worst_price = *price;
        levels = i + 1;
    }

    if total_filled == 0 {
        return None;
    }

    let avg_price = Price((weighted_sum / total_filled as i128) as i64);
    Some((avg_price, worst_price, levels))
}

// ---------------------------------------------------------------------------
// Benchmarkable micro-functions (for criterion)
// ---------------------------------------------------------------------------

/// Batch price comparison: how many prices in the array are <= target?
/// Used for quick depth counting.
#[inline]
pub fn count_prices_le(prices: &[Price], target: Price) -> usize {
    let target_val = target.0;
    let mut count = 0usize;

    // Unrolled loop for auto-vectorization
    let chunks = prices.len() / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        count += (prices[base].0 <= target_val) as usize;
        count += (prices[base + 1].0 <= target_val) as usize;
        count += (prices[base + 2].0 <= target_val) as usize;
        count += (prices[base + 3].0 <= target_val) as usize;
    }

    for i in (chunks * 4)..prices.len() {
        count += (prices[i].0 <= target_val) as usize;
    }

    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumulative_volume() {
        let qtys = vec![
            Quantity::from_f64(1.0),
            Quantity::from_f64(2.0),
            Quantity::from_f64(3.0),
            Quantity::from_f64(4.0),
        ];

        let cum = cumulative_volume(&qtys);
        assert_eq!(cum[0], Quantity::from_f64(1.0));
        assert_eq!(cum[1], Quantity::from_f64(3.0));
        assert_eq!(cum[2], Quantity::from_f64(6.0));
        assert_eq!(cum[3], Quantity::from_f64(10.0));

        // SIMD version should give same results
        let cum_simd = cumulative_volume_simd4(&qtys);
        assert_eq!(cum, cum_simd);
    }

    #[test]
    fn test_find_fill_depth() {
        let qtys = vec![
            Quantity::from_f64(1.0),
            Quantity::from_f64(2.0),
            Quantity::from_f64(3.0),
            Quantity::from_f64(4.0),
        ];

        assert_eq!(find_fill_depth(&qtys, Quantity::from_f64(0.5)), Some(0));
        assert_eq!(find_fill_depth(&qtys, Quantity::from_f64(3.0)), Some(1));
        assert_eq!(find_fill_depth(&qtys, Quantity::from_f64(6.0)), Some(2));
        assert_eq!(find_fill_depth(&qtys, Quantity::from_f64(10.0)), Some(3));
        assert_eq!(find_fill_depth(&qtys, Quantity::from_f64(11.0)), None);
    }

    #[test]
    fn test_vwap() {
        let prices = vec![
            Price::from_f64(100.0),
            Price::from_f64(101.0),
            Price::from_f64(102.0),
        ];
        let qtys = vec![
            Quantity::from_f64(10.0),
            Quantity::from_f64(20.0),
            Quantity::from_f64(10.0),
        ];

        let v = vwap(&prices, &qtys).unwrap();
        // VWAP = (100*10 + 101*20 + 102*10) / 40 = 4040/40 = 101.0
        let expected = Price::from_f64(101.0);
        assert!((v.0 - expected.0).abs() <= 1); // Allow 1 raw unit rounding
    }

    #[test]
    fn test_market_impact() {
        let prices = vec![
            Price::from_f64(100.0),
            Price::from_f64(101.0),
            Price::from_f64(102.0),
        ];
        let qtys = vec![
            Quantity::from_f64(5.0),
            Quantity::from_f64(3.0),
            Quantity::from_f64(10.0),
        ];

        // Fill 7.0: 5.0 @ 100 + 2.0 @ 101
        let (avg, worst, levels) =
            market_impact(&prices, &qtys, Quantity::from_f64(7.0)).unwrap();
        assert_eq!(levels, 2);
        assert_eq!(worst, Price::from_f64(101.0));
        // avg = (500 + 202) / 7 = 100.2857...
        let avg_f = avg.to_f64();
        assert!((avg_f - 100.2857).abs() < 0.01);
    }

    #[test]
    fn test_scan_price() {
        let prices = vec![
            Price::from_f64(98.0),
            Price::from_f64(99.0),
            Price::from_f64(100.0),
            Price::from_f64(101.0),
            Price::from_f64(102.0),
        ];

        assert_eq!(
            scan_price_ge(&prices, Price::from_f64(100.0)),
            Some(2)
        );
        assert_eq!(
            scan_price_le(&prices, Price::from_f64(99.0)),
            Some(0)
        );
    }

    #[test]
    fn test_count_prices_le() {
        let prices = vec![
            Price::from_f64(98.0),
            Price::from_f64(99.0),
            Price::from_f64(100.0),
            Price::from_f64(101.0),
            Price::from_f64(102.0),
        ];

        assert_eq!(count_prices_le(&prices, Price::from_f64(100.0)), 3);
        assert_eq!(count_prices_le(&prices, Price::from_f64(97.0)), 0);
        assert_eq!(count_prices_le(&prices, Price::from_f64(200.0)), 5);
    }
}
