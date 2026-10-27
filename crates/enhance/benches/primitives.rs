use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_enhance::enhance::{enhance_brightness, enhance_color, enhance_contrast, enhance_sharpness};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn primitives(c: &mut Criterion) {
    Python::attach(|py| {
        for &(width, height) in sizes() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in blanket_bench_support::depths() {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let mut group = c.benchmark_group(format!("enhance/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    group.bench_function("color", |b| b.iter(|| enhance_color(py, black_box(&image)).unwrap()));
                    group.bench_function("contrast", |b| b.iter(|| enhance_contrast(py, black_box(&image)).unwrap()));
                    group.bench_function("brightness", |b| b.iter(|| enhance_brightness(py, black_box(&image)).unwrap()));
                    group.bench_function("sharpness", |b| b.iter(|| enhance_sharpness(py, black_box(&image)).unwrap()));
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = primitives }
criterion_main!(benches);
