use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_filter::filter::filter_blur;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn blur(c: &mut Criterion) {
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in blanket_bench_support::depths() {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let mut group = c.benchmark_group(format!("blur/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    group.bench_function("gaussian-blur-radius-2", |b| {
                        b.iter(|| filter_blur(py, black_box(&image), black_box((2.0, 2.0)), true).unwrap());
                    });
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = blur }
criterion_main!(benches);
