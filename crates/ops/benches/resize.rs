use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_ops::ops::ops_resize;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn resize(c: &mut Criterion) {
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in blanket_bench_support::depths() {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let mut group = c.benchmark_group(format!("resize/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    for (name, method) in [("nearest", 0), ("bilinear", 2), ("lanczos", 1)] {
                        for (case, target, bounds) in [
                            ("resize-half", (width / 2, height / 2), (0.0, 0.0, f64::from(width), f64::from(height))),
                            ("resize-upscale", (width * 3 / 2, height * 3 / 2), (0.0, 0.0, f64::from(width), f64::from(height))),
                            ("resize-crop", (width / 2, height / 2), (1.25, 2.5, f64::from(width) - 0.75, f64::from(height) - 1.5)),
                        ] {
                            group.bench_with_input(BenchmarkId::new(case, name), &method, |b, &method| {
                                b.iter(|| ops_resize(py, black_box(&image), black_box(target), black_box(method), black_box(bounds), None).unwrap());
                            });
                        }
                    }
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = resize }
criterion_main!(benches);
