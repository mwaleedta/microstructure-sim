//! Discrete-event simulation engine.
//!
//! Processes events in strict timestamp order using a priority queue.
//! Integrates latency models to schedule delayed events realistically,
//! and supports deterministic replay via seeded RNG.

use crate::latency::LatencySimulator;
use crate::queue_position::QueueEstimator;
use orderbook_core::types::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrd;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimEventType {
    /// Market data update (L2 depth change).
    MarketDataUpdate {
        bids: Vec<(Price, Quantity)>,
        asks: Vec<(Price, Quantity)>,
    },
    /// Trade occurred on the exchange.
    Trade {
        price: Price,
        quantity: Quantity,
        side: Side,
    },
    /// Strategy wants to submit an order.
    OrderSubmit(Order),
    /// Exchange acknowledges an order (after order_latency).
    OrderAck {
        order_id: OrderId,
    },
    /// A fill occurred.
    FillEvent(Fill),
    /// Strategy wants to cancel an order.
    CancelRequest {
        order_id: OrderId,
    },
    /// Exchange acknowledges a cancel (after cancel_latency).
    CancelAck {
        order_id: OrderId,
    },
    /// Periodic timer tick for strategy evaluation.
    TimerTick {
        tick_id: u64,
    },
    /// Simulation complete.
    EndOfData,
}

/// A single simulation event with a timestamp.
#[derive(Debug, Clone)]
pub struct SimEvent {
    pub timestamp_ns: u64,
    pub event_type: SimEventType,
    /// Sequence number for stable sorting of simultaneous events.
    pub sequence: u64,
}

// BinaryHeap is a max-heap, but we want min-heap (earliest first).
impl Eq for SimEvent {}
impl PartialEq for SimEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_ns == other.timestamp_ns
            && self.sequence == other.sequence
    }
}

impl Ord for SimEvent {
    fn cmp(&self, other: &Self) -> CmpOrd {
        // Reverse ordering for min-heap behavior
        other
            .timestamp_ns
            .cmp(&self.timestamp_ns)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for SimEvent {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrd> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

/// Trait that strategies implement to participate in the simulation.
pub trait SimStrategy {
    /// Called on each market data update. Returns orders to submit.
    fn on_market_data(
        &mut self,
        timestamp_ns: u64,
        bids: &[(Price, Quantity)],
        asks: &[(Price, Quantity)],
    ) -> Vec<Order>;

    /// Called on each fill of one of our orders.
    fn on_fill(&mut self, timestamp_ns: u64, fill: &Fill);

    /// Called on periodic timer tick.
    fn on_timer(&mut self, timestamp_ns: u64, tick_id: u64) -> Vec<Order>;

    /// Called when an order is acknowledged.
    fn on_order_ack(&mut self, timestamp_ns: u64, order_id: OrderId);

    /// Called when a cancel is acknowledged.
    fn on_cancel_ack(&mut self, timestamp_ns: u64, order_id: OrderId);
}

// ---------------------------------------------------------------------------
// SimulationConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// RNG seed for deterministic replay.
    pub seed: u64,
    /// Timer tick interval in nanoseconds (0 = no timer).
    pub timer_interval_ns: u64,
    /// Maximum number of events to process (safety limit).
    pub max_events: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            timer_interval_ns: 1_000_000_000, // 1 second
            max_events: 10_000_000,
        }
    }
}

// ---------------------------------------------------------------------------
// SimulationEngine
// ---------------------------------------------------------------------------

/// The core discrete-event simulation engine.
pub struct SimulationEngine {
    config: SimulationConfig,
    /// Priority queue of pending events (min-heap by timestamp).
    event_queue: BinaryHeap<SimEvent>,
    /// Monotonic sequence counter for stable event ordering.
    sequence_counter: u64,
    /// Current simulation time.
    current_time_ns: u64,
    /// Deterministic RNG.
    rng: StdRng,
    /// Latency models.
    latency: LatencySimulator,
    /// Total events processed.
    events_processed: u64,
}

impl SimulationEngine {
    pub fn new(
        config: SimulationConfig,
        latency: LatencySimulator,
    ) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            event_queue: BinaryHeap::with_capacity(100_000),
            sequence_counter: 0,
            current_time_ns: 0,
            rng,
            latency,
            events_processed: 0,
        }
    }

    /// Schedule an event at a specific timestamp.
    pub fn schedule(&mut self, timestamp_ns: u64, event_type: SimEventType) {
        self.sequence_counter += 1;
        self.event_queue.push(SimEvent {
            timestamp_ns,
            event_type,
            sequence: self.sequence_counter,
        });
    }

    /// Schedule an event with latency applied.
    pub fn schedule_with_feed_latency(
        &mut self,
        base_timestamp_ns: u64,
        event_type: SimEventType,
    ) {
        let delay = self.latency.sample_feed(&mut self.rng);
        self.schedule(base_timestamp_ns + delay, event_type);
    }

    /// Load market data events from a pre-recorded dataset.
    pub fn load_market_data(
        &mut self,
        data: Vec<(u64, Vec<(Price, Quantity)>, Vec<(Price, Quantity)>)>,
    ) {
        for (ts, bids, asks) in data {
            self.schedule_with_feed_latency(
                ts,
                SimEventType::MarketDataUpdate { bids, asks },
            );
        }
    }

    /// Load trade events.
    pub fn load_trades(
        &mut self,
        trades: Vec<(u64, Price, Quantity, Side)>,
    ) {
        for (ts, price, qty, side) in trades {
            self.schedule_with_feed_latency(
                ts,
                SimEventType::Trade {
                    price,
                    quantity: qty,
                    side,
                },
            );
        }
    }

    /// Set up periodic timer ticks.
    pub fn setup_timer(&mut self, start_ns: u64, end_ns: u64) {
        if self.config.timer_interval_ns == 0 {
            return;
        }
        let mut ts = start_ns;
        let mut tick_id = 0u64;
        while ts <= end_ns {
            self.schedule(
                ts,
                SimEventType::TimerTick { tick_id },
            );
            ts += self.config.timer_interval_ns;
            tick_id += 1;
        }
    }

    /// Run the simulation to completion.
    ///
    /// Processes events in timestamp order, dispatching to the strategy
    /// and the order book, with latency-aware scheduling of responses.
    pub fn run(
        &mut self,
        book: &orderbook_core::book::OrderBook,
        strategy: &mut dyn SimStrategy,
        queue_estimator: &mut QueueEstimator,
    ) -> SimulationResult {
        let mut result = SimulationResult::default();

        while let Some(event) = self.event_queue.pop() {
            if self.events_processed >= self.config.max_events {
                tracing::warn!("Max events reached: {}", self.config.max_events);
                break;
            }

            self.current_time_ns = event.timestamp_ns;
            self.events_processed += 1;

            match event.event_type {
                SimEventType::MarketDataUpdate { ref bids, ref asks } => {
                    result.market_data_count += 1;

                    // Feed to strategy
                    let orders = strategy.on_market_data(
                        event.timestamp_ns,
                        bids,
                        asks,
                    );

                    // Schedule order submissions with latency
                    for order in orders {
                        let delay =
                            self.latency.sample_order(&mut self.rng);
                        self.schedule(
                            event.timestamp_ns + delay,
                            SimEventType::OrderSubmit(order),
                        );
                    }
                }

                SimEventType::Trade { price, quantity, .. } => {
                    result.trade_count += 1;
                    // Update queue position estimates
                    queue_estimator
                        .on_trade(price, quantity.to_f64());
                }

                SimEventType::OrderSubmit(order) => {
                    // Submit to the book
                    let events = book.submit_order(order);
                    result.orders_submitted += 1;

                    // Process matching engine events
                    for me_event in events {
                        match me_event {
                            MatchEvent::Accepted(ack) => {
                                self.schedule(
                                    event.timestamp_ns,
                                    SimEventType::OrderAck {
                                        order_id: ack.order_id,
                                    },
                                );
                            }
                            MatchEvent::Filled(fill)
                            | MatchEvent::PartialFill(fill) => {
                                self.schedule(
                                    event.timestamp_ns,
                                    SimEventType::FillEvent(fill),
                                );
                            }
                            MatchEvent::Cancelled(cancel) => {
                                self.schedule(
                                    event.timestamp_ns,
                                    SimEventType::CancelAck {
                                        order_id: cancel.order_id,
                                    },
                                );
                            }
                            _ => {}
                        }
                    }
                }

                SimEventType::OrderAck { order_id } => {
                    strategy.on_order_ack(event.timestamp_ns, order_id);
                }

                SimEventType::FillEvent(ref fill) => {
                    result.fills += 1;
                    strategy.on_fill(event.timestamp_ns, fill);
                    queue_estimator.record_actual_fill(
                        fill.maker_order_id,
                        event.timestamp_ns,
                    );
                }

                SimEventType::CancelRequest { order_id } => {
                    let _ = book.cancel_order(order_id);
                    let delay =
                        self.latency.sample_cancel(&mut self.rng);
                    self.schedule(
                        event.timestamp_ns + delay,
                        SimEventType::CancelAck { order_id },
                    );
                }

                SimEventType::CancelAck { order_id } => {
                    result.cancels += 1;
                    strategy.on_cancel_ack(event.timestamp_ns, order_id);
                }

                SimEventType::TimerTick { tick_id } => {
                    let orders = strategy.on_timer(
                        event.timestamp_ns,
                        tick_id,
                    );
                    for order in orders {
                        let delay =
                            self.latency.sample_order(&mut self.rng);
                        self.schedule(
                            event.timestamp_ns + delay,
                            SimEventType::OrderSubmit(order),
                        );
                    }
                }

                SimEventType::EndOfData => {
                    break;
                }
            }
        }

        result.events_processed = self.events_processed;
        result.simulation_duration_ns =
            self.current_time_ns.saturating_sub(0);
        result
    }

    /// Get current simulation time.
    pub fn current_time_ns(&self) -> u64 {
        self.current_time_ns
    }

    /// Get number of pending events.
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }
}

// ---------------------------------------------------------------------------
// SimulationResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SimulationResult {
    pub events_processed: u64,
    pub market_data_count: u64,
    pub trade_count: u64,
    pub orders_submitted: u64,
    pub fills: u64,
    pub cancels: u64,
    pub simulation_duration_ns: u64,
}

impl SimulationResult {
    pub fn summary(&self) -> String {
        format!(
            "Simulation complete: {} events, {} MD updates, {} trades, \
             {} orders, {} fills, {} cancels, {:.2}s duration",
            self.events_processed,
            self.market_data_count,
            self.trade_count,
            self.orders_submitted,
            self.fills,
            self.cancels,
            self.simulation_duration_ns as f64 / 1e9,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latency::FixedLatency;
    use crate::queue_position::{QueueEstimator, QueueEstimatorConfig};
    use orderbook_core::book::{OrderBook, OrderBookConfig};

    struct DummyStrategy {
        data_count: u64,
    }

    impl SimStrategy for DummyStrategy {
        fn on_market_data(
            &mut self,
            _ts: u64,
            _bids: &[(Price, Quantity)],
            _asks: &[(Price, Quantity)],
        ) -> Vec<Order> {
            self.data_count += 1;
            Vec::new()
        }

        fn on_fill(&mut self, _ts: u64, _fill: &Fill) {}
        fn on_timer(&mut self, _ts: u64, _tick_id: u64) -> Vec<Order> {
            Vec::new()
        }
        fn on_order_ack(&mut self, _ts: u64, _oid: OrderId) {}
        fn on_cancel_ack(&mut self, _ts: u64, _oid: OrderId) {}
    }

    #[test]
    fn test_event_ordering() {
        let mut engine = SimulationEngine::new(
            SimulationConfig::default(),
            LatencySimulator::zero_latency(),
        );

        engine.schedule(
            300,
            SimEventType::TimerTick { tick_id: 3 },
        );
        engine.schedule(
            100,
            SimEventType::TimerTick { tick_id: 1 },
        );
        engine.schedule(
            200,
            SimEventType::TimerTick { tick_id: 2 },
        );

        // Should come out in order: 100, 200, 300
        let e1 = engine.event_queue.pop().unwrap();
        assert_eq!(e1.timestamp_ns, 100);
        let e2 = engine.event_queue.pop().unwrap();
        assert_eq!(e2.timestamp_ns, 200);
    }

    #[test]
    fn test_basic_simulation() {
        let mut engine = SimulationEngine::new(
            SimulationConfig {
                timer_interval_ns: 0,
                ..Default::default()
            },
            LatencySimulator::zero_latency(),
        );

        // Load some market data
        engine.load_market_data(vec![
            (
                1_000_000,
                vec![(Price::from_f64(99.0), Quantity::from_f64(10.0))],
                vec![(Price::from_f64(101.0), Quantity::from_f64(10.0))],
            ),
            (
                2_000_000,
                vec![(Price::from_f64(99.5), Quantity::from_f64(10.0))],
                vec![(Price::from_f64(100.5), Quantity::from_f64(10.0))],
            ),
        ]);

        engine.schedule(3_000_000, SimEventType::EndOfData);

        let book = OrderBook::with_symbol("TEST");
        let mut strategy = DummyStrategy { data_count: 0 };
        let mut queue_est =
            QueueEstimator::new(QueueEstimatorConfig::default());

        let result =
            engine.run(&book, &mut strategy, &mut queue_est);

        assert_eq!(result.market_data_count, 2);
        assert_eq!(strategy.data_count, 2);
    }
}
