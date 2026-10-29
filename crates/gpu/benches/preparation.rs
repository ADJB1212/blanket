use std::hint::black_box;

use blanket_gpu::protocol::resampling;
use blanket_gpu::{blur_parameters, lut_controls, resize_weights};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn preparation(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("gpu/preparation/resize_weights");
        for (input, output) in [(64, 128), (1920, 1024), (4096, 256)] {
            group.throughput(Throughput::Elements(u64::from(output)));
            for (name, method) in [
                ("nearest", resampling::NEAREST),
                ("bilinear", resampling::BILINEAR),
                ("bicubic", resampling::BICUBIC),
                ("lanczos", resampling::LANCZOS),
            ] {
                group.bench_function(BenchmarkId::new(name, format!("{input}-{output}")), |b| {
                    b.iter(|| resize_weights(black_box(input), black_box(output), black_box(method)).unwrap());
                });
            }
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("gpu/preparation/lut");
        for size in [17, 33, 65] {
            let table: Vec<f32> = (0..size * size * size * 3).map(|i| (i % 257) as f32 / 256.0).collect();
            group.throughput(Throughput::Elements(table.len() as u64));
            for (name, floating) in [("u8", false), ("f32", true)] {
                group.bench_function(BenchmarkId::new(name, size), |b| {
                    b.iter(|| lut_controls(black_box([size; 3]), 3, black_box(&table), floating).unwrap());
                });
            }
        }
        group.finish();
    }

    let mut group = c.benchmark_group("gpu/preparation/blur");
    for (name, gaussian) in [("box", false), ("gaussian", true)] {
        group.bench_function(name, |b| b.iter(|| blur_parameters(black_box(2.5), gaussian).unwrap()));
    }
    group.finish();
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = preparation }
criterion_main!(benches);
