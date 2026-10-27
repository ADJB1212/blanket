use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn conversion(c: &mut Criterion) {
    let depths: &[u8] = if blanket_bench_support::full_matrix() { &[8, 10, 12, 16] } else { &[8, 16] };
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in depths {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let mut group = c.benchmark_group(format!("conversion/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    for target in ["L", "RGB", "RGBA"] {
                        if target != mode.as_str() {
                            group.bench_function(format!("convert-{target}"), |b| {
                                b.iter(|| black_box(&image).convert(py, black_box(target), Some(depth)).unwrap());
                            });
                        }
                        if blanket_bench_support::full_matrix() {
                            let target_depth = if depth == 8 { 16 } else { 8 };
                            group.bench_function(format!("convert-{target}-{target_depth}"), |b| {
                                b.iter(|| black_box(&image).convert(py, black_box(target), Some(target_depth)).unwrap());
                            });
                        }
                    }
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = conversion }
criterion_main!(benches);
