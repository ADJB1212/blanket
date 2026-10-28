use std::hint::black_box;

use blanket_core::pixels;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn pixels(c: &mut Criterion) {
    for count in [4096, 480_000] {
        let source: Vec<_> = (0..count).map(|i| (i * 71) as u8).collect();
        let mut output = vec![0; count / 8];
        let mut group = c.benchmark_group(format!("pixels/{count}"));
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function("pack-bilevel", |b| b.iter(|| pixels::pack_bilevel::<false>(black_box(&source), black_box(&mut output))));
        group.finish();
    }
}

fn bilevel(c: &mut Criterion) {
    for (width, height) in [(800, 600), (801, 600), (1920, 1080)] {
        let source: Vec<_> = (0..width * height).map(|i| if (i * 37 + i / 7) % 3 == 0 { 0 } else { 255 }).collect();
        let packed = blanket_core::raster::pack_bilevel(&source, width, height);
        let mut group = c.benchmark_group(format!("bilevel/{width}x{height}"));
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function("pack", |b| b.iter(|| blanket_core::raster::pack_bilevel(black_box(&source), width, height)));
        group.bench_function("unpack", |b| b.iter(|| blanket_core::raster::unpack_bilevel(black_box(&packed), width, height)));
        group.bench_function("unpack-inverted", |b| b.iter(|| blanket_core::raster::unpack_bilevel_inverted(black_box(&packed), width, height)));
        group.finish();
    }
}

fn swaps(c: &mut Criterion) {
    fn run<const I: usize, const O: usize>(c: &mut Criterion) {
        for count in [17, 4096, 480_000] {
            let source: Vec<u8> = (0..count * I).map(|i| (i * 71) as u8).collect();
            let mut output = vec![0; count * O];
            let mut group = c.benchmark_group(format!("swap/{I}-{O}/{count}"));
            group.bench_function("scalar", |b| {
                b.iter(|| {
                    for (src, dst) in black_box(&source).as_chunks::<I>().0.iter().zip(black_box(&mut output).as_chunks_mut::<O>().0) {
                        dst[..3].copy_from_slice(&[src[2], src[1], src[0]]);
                        if O == 4 {
                            dst[3] = if I == 4 { src[3] } else { 0 };
                        }
                    }
                })
            });
            group.bench_function("simd", |b| b.iter(|| pixels::swap_red_blue::<I, O>(black_box(&source), black_box(&mut output))));
            group.finish();
        }
    }
    run::<3, 3>(c);
    run::<4, 4>(c);
    run::<3, 4>(c);
    run::<4, 3>(c);
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = pixels, bilevel, swaps }
criterion_main!(benches);
