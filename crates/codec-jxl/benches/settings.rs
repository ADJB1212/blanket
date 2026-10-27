use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_codec_common::SaveOptions;
use blanket_codec_jxl::{JxlThreads, PreparedJxl};
use blanket_core::PixelMode;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn jxl_settings(c: &mut Criterion) {
    for (width, height) in sizes().iter().copied() {
        for &depth in blanket_bench_support::depths() {
            let image = fixture(width, height, PixelMode::Rgb, depth, 12345, false);
            let mut group = c.benchmark_group(format!("jxl-settings/RGB-{depth}/{width}x{height}"));
            group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
            for effort in [1, 3, 7] {
                for lossless in [false, true] {
                    let options = SaveOptions { quality: 85, effort, lossless, ..SaveOptions::default() };
                    let coding = if lossless { "lossless" } else { "lossy-q85" };
                    for (name, threads) in [("single", JxlThreads::Fixed(1)), ("pool", JxlThreads::Pool)] {
                        group.bench_with_input(BenchmarkId::new(format!("{coding}-e{effort}"), name), &threads, |b, &threads| {
                            b.iter(|| PreparedJxl::new(black_box(&image)).unwrap().encode_with_threads(black_box(options), threads).unwrap());
                        });
                    }
                }
            }
            group.finish();
        }
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = jxl_settings }
criterion_main!(benches);
