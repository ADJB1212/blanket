use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_filter::filter::{filter_blur, filter_rank, filter_unsharp};
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

fn high_depth_filters(c: &mut Criterion) {
    Python::attach(|py| {
        let image = fixture(512, 512, PixelMode::Rgb, 16, 12345, true);
        let mut group = c.benchmark_group("high-depth-filters");
        group.throughput(Throughput::Elements(512 * 512));
        for size in [9, 31] {
            group.bench_function(format!("maximum-{size}"), |b| {
                b.iter(|| filter_rank(py, black_box(&image), size, size * size - 1).unwrap());
            });
            group.bench_function(format!("median-{size}"), |b| {
                b.iter(|| filter_rank(py, black_box(&image), size, size * size / 2).unwrap());
            });
        }
        for (name, percent, threshold) in [("zero-percent", 0, 3), ("high-threshold", 150, 65535), ("normal", 150, 3)] {
            group.bench_function(format!("unsharp-{name}"), |b| {
                b.iter(|| filter_unsharp(py, black_box(&image), 2.0, percent, threshold).unwrap());
            });
        }
        group.finish();
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = blur, high_depth_filters }
criterion_main!(benches);
