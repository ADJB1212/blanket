use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_core::PixelMode;
use blanket_ops::ops::{ops_affine, transpose};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;

fn transform(c: &mut Criterion) {
    Python::attach(|py| {
        for &(width, height) in sizes() {
            for &mode in blanket_bench_support::modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba]) {
                for &depth in blanket_bench_support::depths() {
                    let image = fixture(width, height, mode, depth, 12345, true);
                    let source = image.raw_data().unwrap();
                    let mut output = vec![0; source.len()];
                    let mut group = c.benchmark_group(format!("transform/{}-{depth}/{width}x{height}", mode.as_str()));
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    for (name, orientation) in [("transpose", 5), ("rotate-270", 6), ("transverse", 7), ("rotate-90", 8)] {
                        group.bench_function(name, |b| {
                            b.iter(|| match image.raw_bytes_per_pixel() {
                                1 => transpose::<1>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                2 => transpose::<2>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                3 => transpose::<3>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                4 => transpose::<4>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                6 => transpose::<6>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                8 => transpose::<8>(black_box(source), black_box(&mut output), width as usize, height as usize, orientation),
                                _ => unreachable!(),
                            })
                        });
                    }
                    if depth == 8 {
                        let angle = 0.37_f64;
                        let (sin, cos) = angle.sin_cos();
                        let rotation = [cos, -sin, f64::from(width) * 0.17, sin, cos, -f64::from(height) * 0.19];
                        let affine = [0.91, 0.13, -17.25, -0.07, 1.03, 11.5];
                        for (name, matrix) in [("rotate-arbitrary", rotation), ("affine", affine)] {
                            for (filter, method) in [("nearest", 0), ("bilinear", 2), ("bicubic", 3)] {
                                group.bench_with_input(BenchmarkId::new(name, filter), &method, |b, &method| {
                                    b.iter(|| {
                                        ops_affine(py, black_box(&image), (width, height), black_box(matrix), method, vec![0; mode.channels()], false)
                                            .unwrap()
                                    })
                                });
                            }
                        }
                    }
                    group.finish();
                }
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = transform }
criterion_main!(benches);
