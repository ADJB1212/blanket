use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_composite::composite::image_alpha_composite;
use blanket_core::PixelMode;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn alpha_composite(c: &mut Criterion) {
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &depth in blanket_bench_support::depths() {
                let image = fixture(width, height, PixelMode::Rgba, depth, 12345, true);
                let second = fixture(width, height, PixelMode::Rgba, depth, 67890, true);
                let mut group = c.benchmark_group(format!("alpha-composite/RGBA-{depth}/{width}x{height}"));
                group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                group.bench_function("alpha-composite", |b| {
                    b.iter(|| image_alpha_composite(py, black_box(&image), black_box(&second)).unwrap());
                });
                group.finish();
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = alpha_composite }
criterion_main!(benches);
