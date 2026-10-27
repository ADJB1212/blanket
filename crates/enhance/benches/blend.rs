use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_enhance::enhance::enhance_blend;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn blend(c: &mut Criterion) {
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in blanket_bench_support::depths() {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let second = fixture(width, height, mode, depth, 67890, true);
                    let mut group = c.benchmark_group(format!("blend/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    group.bench_function("blend-0.4", |b| {
                        b.iter(|| enhance_blend(py, black_box(&image), black_box(&second), black_box(0.4)).unwrap());
                    });
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = blend }
criterion_main!(benches);
