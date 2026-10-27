use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_codec_common::SaveOptions;
use blanket_codec_jpeg::stream::JpegStreamDecoder;
use blanket_codec_jpeg::{JpegCoding, encode_jpeg_variant};
use blanket_core::{Image, PixelMode};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn decode(encoded: &[u8], chunk_size: usize) -> Image {
    let mut decoder = JpegStreamDecoder::new().unwrap();
    for chunk in encoded.chunks(chunk_size) {
        decoder.feed(chunk).unwrap();
    }
    assert!(decoder.complete);
    decoder.image().unwrap().unwrap()
}

fn streaming(c: &mut Criterion) {
    for (width, height) in sizes().iter().copied() {
        let image = fixture(width, height, PixelMode::Rgb, 8, 12345, false);
        let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
        for (name, progressive) in [("jpeg-baseline", false), ("jpeg-progressive", true)] {
            let encoded = encode_jpeg_variant(&image, options, JpegCoding { progressive, ..JpegCoding::default() }).unwrap();
            let mut group = c.benchmark_group(format!("streaming/{name}/{width}x{height}"));
            group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
            for chunk_size in [1024, 16 * 1024, 64 * 1024] {
                group.bench_with_input(BenchmarkId::new("decode-chunk-bytes", chunk_size), &chunk_size, |b, &chunk_size| {
                    assert_eq!(decode(&encoded, chunk_size).size(), image.size());
                    b.iter(|| decode(black_box(&encoded), black_box(chunk_size)));
                });
            }
            group.finish();
        }
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = streaming }
criterion_main!(benches);
