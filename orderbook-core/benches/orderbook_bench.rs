use criterion::{criterion_group, criterion_main, Criterion};

fn orderbook_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, orderbook_benchmark);
criterion_main!(benches);
