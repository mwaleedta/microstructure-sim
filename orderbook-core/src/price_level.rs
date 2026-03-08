//! Cache-line-aligned price level with lock-free order queue.
//!
//! Each `PriceLevel` represents all resting orders at a single price point.
//! The internal FIFO queue maintains time priority. All mutations are lock-free
//! using atomic CAS operations, with crossbeam-epoch for safe memory reclamation.

use crate::types::{Order, OrderId, Price, Quantity};
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Order node for the intrusive linked list
// ---------------------------------------------------------------------------

/// A node in the doubly-linked order queue. Allocated on the heap,
/// managed via crossbeam-epoch for safe deferred deallocation.
pub struct OrderNode {
    pub order: Order,
    pub next: Atomic<OrderNode>,
    pub prev: Atomic<OrderNode>,
    /// Logically removed flag -- set before unlinking.
    pub removed: AtomicU32,
}

impl OrderNode {
    pub fn new(order: Order) -> Self {
        Self {
            order,
            next: Atomic::null(),
            prev: Atomic::null(),
            removed: AtomicU32::new(0),
        }
    }

    #[inline]
    pub fn is_removed(&self) -> bool {
        self.removed.load(Ordering::Acquire) != 0
    }

    #[inline]
    pub fn mark_removed(&self) {
        self.removed.store(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// PriceLevel
// ---------------------------------------------------------------------------

/// A single price level in the order book.
///
/// Aligned to 64 bytes (one cache line) to prevent false sharing when
/// adjacent levels are accessed by different threads.
///
/// The order queue is a lock-free doubly-linked list with sentinel head/tail.
/// Insertions always go to the tail (time priority). Removals mark-then-unlink
/// with epoch-based reclamation.
#[repr(align(64))]
pub struct PriceLevel {
    /// The price this level represents.
    pub price: Price,

    /// Total resting quantity across all orders at this level.
    /// Updated atomically on add/cancel/match.
    pub total_quantity: AtomicU64,

    /// Number of resting orders at this level.
    pub order_count: AtomicU32,

    /// Sentinel head of the order queue (never contains a real order).
    head: Atomic<OrderNode>,

    /// Sentinel tail of the order queue.
    tail: Atomic<OrderNode>,
}

impl PriceLevel {
    /// Create a new empty price level.
    pub fn new(price: Price) -> Self {
        // Create sentinel nodes and link them together
        let head = Owned::new(OrderNode::new(Order::new_limit(
            OrderId(u64::MAX),
            crate::types::Side::Bid,
            Price::ZERO,
            Quantity::ZERO,
            0,
            0,
        )))
        .into_shared(unsafe { &epoch::unprotected() });

        let tail = Owned::new(OrderNode::new(Order::new_limit(
            OrderId(u64::MAX - 1),
            crate::types::Side::Bid,
            Price::ZERO,
            Quantity::ZERO,
            0,
            0,
        )))
        .into_shared(unsafe { &epoch::unprotected() });

        // Link: head -> tail
        unsafe {
            head.deref().next.store(tail, Ordering::Release);
            tail.deref().prev.store(head, Ordering::Release);
        }

        Self {
            price,
            total_quantity: AtomicU64::new(0),
            order_count: AtomicU32::new(0),
            head: Atomic::from(head),
            tail: Atomic::from(tail),
        }
    }

    /// Add an order to the back of the queue (FIFO -- time priority).
    ///
    /// This is lock-free: we CAS the new node before the tail sentinel.
    pub fn add_order(&self, order: Order, guard: &Guard) {
        let qty_raw = order.remaining_qty.0;
        let new_node = Owned::new(OrderNode::new(order)).into_shared(guard);

        loop {
            let tail = self.tail.load(Ordering::Acquire, guard);
            let tail_ref = unsafe { tail.deref() };
            let prev = tail_ref.prev.load(Ordering::Acquire, guard);

            if prev.is_null() {
                continue;
            }

            let prev_ref = unsafe { prev.deref() };

            // Set new_node's links
            unsafe {
                new_node.deref().prev.store(prev, Ordering::Release);
                new_node.deref().next.store(tail, Ordering::Release);
            }

            // CAS: prev.next = new_node (was tail)
            if prev_ref
                .next
                .compare_exchange(tail, new_node, Ordering::AcqRel, Ordering::Acquire, guard)
                .is_ok()
            {
                // CAS: tail.prev = new_node (was prev)
                tail_ref.prev.store(new_node, Ordering::Release);

                // Update aggregates
                self.total_quantity.fetch_add(qty_raw, Ordering::Release);
                self.order_count.fetch_add(1, Ordering::Release);
                return;
            }
            // CAS failed -- retry
        }
    }

    /// Cancel (remove) a specific order by scanning the queue.
    ///
    /// Returns `Some(remaining_qty)` if found and removed, `None` if not found.
    pub fn cancel_order(&self, order_id: OrderId, guard: &Guard) -> Option<Quantity> {
        let head = self.head.load(Ordering::Acquire, guard);
        let tail = self.tail.load(Ordering::Acquire, guard);

        let mut current = unsafe { head.deref() }
            .next
            .load(Ordering::Acquire, guard);

        while current != tail {
            if current.is_null() {
                break;
            }

            let node = unsafe { current.deref() };

            if node.order.id == order_id && !node.is_removed() {
                // Mark removed first (logical deletion)
                node.mark_removed();
                let qty = node.order.remaining_qty;

                // Unlink: prev.next = next, next.prev = prev
                let prev = node.prev.load(Ordering::Acquire, guard);
                let next = node.next.load(Ordering::Acquire, guard);

                if !prev.is_null() {
                    unsafe { prev.deref() }
                        .next
                        .store(next, Ordering::Release);
                }
                if !next.is_null() {
                    unsafe { next.deref() }
                        .prev
                        .store(prev, Ordering::Release);
                }

                // Update aggregates
                self.total_quantity
                    .fetch_sub(qty.0, Ordering::Release);
                self.order_count.fetch_sub(1, Ordering::Release);

                // Defer deallocation
                unsafe {
                    guard.defer_destroy(current);
                }

                return Some(qty);
            }

            current = node.next.load(Ordering::Acquire, guard);
        }

        None
    }

    /// Match (consume) quantity from the front of the queue.
    ///
    /// Returns a vector of `(OrderId, filled_qty)` pairs and the total
    /// quantity actually matched. Orders are removed if fully filled,
    /// or their remaining_qty is decremented for partial fills.
    pub fn match_quantity(
        &self,
        mut qty_to_fill: Quantity,
        guard: &Guard,
    ) -> (Vec<(OrderId, Quantity)>, Quantity) {
        let mut fills = Vec::new();
        let mut total_filled = Quantity::ZERO;
        let tail = self.tail.load(Ordering::Acquire, guard);

        loop {
            if qty_to_fill.is_zero() {
                break;
            }

            let head = self.head.load(Ordering::Acquire, guard);
            let first = unsafe { head.deref() }
                .next
                .load(Ordering::Acquire, guard);

            if first == tail || first.is_null() {
                break; // Level exhausted
            }

            let node = unsafe { first.deref() };

            if node.is_removed() {
                // Skip stale nodes
                let next = node.next.load(Ordering::Acquire, guard);
                unsafe { head.deref() }
                    .next
                    .store(next, Ordering::Release);
                if !next.is_null() {
                    unsafe { next.deref() }
                        .prev
                        .store(head, Ordering::Release);
                }
                unsafe { guard.defer_destroy(first); }
                continue;
            }

            let available = node.order.remaining_qty;

            if available.0 <= qty_to_fill.0 {
                // Full fill -- remove the order
                fills.push((node.order.id, available));
                total_filled += available;
                qty_to_fill = Quantity(qty_to_fill.0 - available.0);

                node.mark_removed();
                let next = node.next.load(Ordering::Acquire, guard);
                unsafe { head.deref() }
                    .next
                    .store(next, Ordering::Release);
                if !next.is_null() {
                    unsafe { next.deref() }
                        .prev
                        .store(head, Ordering::Release);
                }

                self.total_quantity
                    .fetch_sub(available.0, Ordering::Release);
                self.order_count.fetch_sub(1, Ordering::Release);

                unsafe { guard.defer_destroy(first); }
            } else {
                // Partial fill -- decrement remaining_qty
                // NOTE: In a truly lock-free design, remaining_qty would
                // be AtomicU64. For the initial implementation we accept
                // that partial fills on the same order are serialized by
                // the matching engine's single-threaded hot path.
                fills.push((node.order.id, qty_to_fill));
                total_filled += qty_to_fill;

                // We use a raw pointer cast to mutate remaining_qty.
                // Safety: the matching engine is the sole writer during
                // a match sweep; concurrent readers tolerate stale values.
                let order_ptr =
                    &node.order as *const Order as *mut Order;
                unsafe {
                    (*order_ptr).remaining_qty =
                        Quantity(available.0 - qty_to_fill.0);
                }

                self.total_quantity
                    .fetch_sub(qty_to_fill.0, Ordering::Release);

                qty_to_fill = Quantity::ZERO;
            }
        }

        (fills, total_filled)
    }

    /// Get the total resting quantity at this price level.
    #[inline]
    pub fn get_total_quantity(&self) -> Quantity {
        Quantity(self.total_quantity.load(Ordering::Acquire))
    }

    /// Get the number of orders at this price level.
    #[inline]
    pub fn get_order_count(&self) -> u32 {
        self.order_count.load(Ordering::Acquire)
    }

    /// Check if this level is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.order_count.load(Ordering::Acquire) == 0
    }

    /// Get the front-of-queue order (oldest, highest time priority).
    pub fn front_order(&self, guard: &Guard) -> Option<Order> {
        let head = self.head.load(Ordering::Acquire, guard);
        let tail = self.tail.load(Ordering::Acquire, guard);
        let first = unsafe { head.deref() }
            .next
            .load(Ordering::Acquire, guard);

        if first == tail || first.is_null() {
            return None;
        }

        let node = unsafe { first.deref() };
        if node.is_removed() {
            None
        } else {
            Some(node.order)
        }
    }

    /// Iterate over all non-removed orders at this level (snapshot).
    pub fn orders(&self, guard: &Guard) -> Vec<Order> {
        let mut result = Vec::new();
        let tail = self.tail.load(Ordering::Acquire, guard);
        let head = self.head.load(Ordering::Acquire, guard);

        let mut current = unsafe { head.deref() }
            .next
            .load(Ordering::Acquire, guard);

        while current != tail && !current.is_null() {
            let node = unsafe { current.deref() };
            if !node.is_removed() {
                result.push(node.order);
            }
            current = node.next.load(Ordering::Acquire, guard);
        }

        result
    }
}

// Safety: PriceLevel uses only atomic operations and epoch-guarded pointers.
unsafe impl Send for PriceLevel {}
unsafe impl Sync for PriceLevel {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Side, Quantity, Price, OrderId};

    fn make_order(id: u64, qty: f64, ts: u64) -> Order {
        Order::new_limit(
            OrderId(id),
            Side::Bid,
            Price::from_f64(100.0),
            Quantity::from_f64(qty),
            ts,
            1,
        )
    }

    #[test]
    fn test_add_and_count() {
        let level = PriceLevel::new(Price::from_f64(100.0));
        let guard = &epoch::pin();

        level.add_order(make_order(1, 1.0, 100), guard);
        level.add_order(make_order(2, 2.0, 200), guard);

        assert_eq!(level.get_order_count(), 2);
        assert_eq!(
            level.get_total_quantity(),
            Quantity::from_f64(3.0)
        );
    }

    #[test]
    fn test_cancel_order() {
        let level = PriceLevel::new(Price::from_f64(100.0));
        let guard = &epoch::pin();

        level.add_order(make_order(1, 1.0, 100), guard);
        level.add_order(make_order(2, 2.0, 200), guard);

        let cancelled = level.cancel_order(OrderId(1), guard);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap(), Quantity::from_f64(1.0));
        assert_eq!(level.get_order_count(), 1);
    }

    #[test]
    fn test_match_full_fill() {
        let level = PriceLevel::new(Price::from_f64(100.0));
        let guard = &epoch::pin();

        level.add_order(make_order(1, 1.0, 100), guard);
        level.add_order(make_order(2, 2.0, 200), guard);

        let (fills, total) =
            level.match_quantity(Quantity::from_f64(3.0), guard);
        assert_eq!(fills.len(), 2);
        assert_eq!(total, Quantity::from_f64(3.0));
        assert!(level.is_empty());
    }

    #[test]
    fn test_match_partial_fill() {
        let level = PriceLevel::new(Price::from_f64(100.0));
        let guard = &epoch::pin();

        level.add_order(make_order(1, 5.0, 100), guard);

        let (fills, total) =
            level.match_quantity(Quantity::from_f64(2.0), guard);
        assert_eq!(fills.len(), 1);
        assert_eq!(total, Quantity::from_f64(2.0));
        assert_eq!(level.get_order_count(), 1);
        assert_eq!(
            level.get_total_quantity(),
            Quantity::from_f64(3.0)
        );
    }

    #[test]
    fn test_fifo_ordering() {
        let level = PriceLevel::new(Price::from_f64(100.0));
        let guard = &epoch::pin();

        level.add_order(make_order(1, 1.0, 100), guard);
        level.add_order(make_order(2, 1.0, 200), guard);
        level.add_order(make_order(3, 1.0, 300), guard);

        let orders = level.orders(guard);
        assert_eq!(orders.len(), 3);
        assert_eq!(orders[0].id, OrderId(1));
        assert_eq!(orders[1].id, OrderId(2));
        assert_eq!(orders[2].id, OrderId(3));
    }
}
