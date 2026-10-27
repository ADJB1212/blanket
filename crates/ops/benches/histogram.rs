use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::{Image, PixelMode};
use blanket_ops::ops::ops_histogram;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn histogram(c: &mut Criterion) {
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                let image = fixture(width, height, mode, 8, 12345, true);
                let flat = Image::from_pixels(width, height, mode, vec![71; image.raw_data().unwrap().len()], None).unwrap();
                let mask =
                    Image::from_pixels(width, height, PixelMode::L, (0..width * height).map(|i| if i % 3 == 0 { 0 } else { 255 }).collect(), None)
                        .unwrap();
                let mut group = c.benchmark_group(format!("histogram/{}-8/{width}x{height}", mode.as_str()));
                group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                group.bench_function("histogram", |b| {
                    b.iter(|| ops_histogram(py, black_box(&image), None).unwrap());
                });
                group.bench_function("histogram-flat", |b| {
                    b.iter(|| ops_histogram(py, black_box(&flat), None).unwrap());
                });
                group.bench_function("histogram-masked", |b| {
                    b.iter(|| ops_histogram(py, black_box(&image), Some(black_box(&mask))).unwrap());
                });
                group.finish();
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = histogram }
criterion_main!(benches);
