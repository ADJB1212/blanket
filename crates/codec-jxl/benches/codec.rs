use std::hint::black_box;

use blanket_bench_support::{SIZES, codec_fixtures, fixture, measure_codec, sizes};
use blanket_codec_common::SaveOptions;
use blanket_codec_jxl::{PreparedJxl, decode_jxl};
use blanket_core::PixelMode;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use jpegxl_rs::encode::{ColorEncoding, EncoderSpeed};
use jpegxl_rs::encoder_builder;

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba], true, &SIZES, |image, content| {
        for (name, settings) in [("jxl-lossy-q85-e3", options), ("jxl-lossless-e3", SaveOptions { lossless: true, ..options })] {
            measure_codec(
                c,
                name,
                image,
                content,
                |image| PreparedJxl::new(image).unwrap().encode(settings).unwrap(),
                |data| decode_jxl(data).unwrap(),
            );
        }
    });
}

fn float_decode(c: &mut Criterion) {
    for &(width, height) in sizes() {
        let image = fixture(width, height, PixelMode::Rgb, 8, 12345, false);
        let samples: Vec<f32> = image.pixel_data().unwrap().iter().map(|&v| f32::from(v) / 255.0).collect();
        let encoded = encoder_builder().build().unwrap().encode(&samples, width, height).unwrap();
        let mut group = c.benchmark_group(format!("codecs/jxl-float/RGB/{width}x{height}"));
        group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
        group.bench_function("decode", |b| b.iter(|| decode_jxl(black_box(&encoded)).unwrap()));
        group.finish();
    }
}

fn small_images(c: &mut Criterion) {
    for side in [32, 64] {
        let image = fixture(side, side, PixelMode::Rgb, 8, 12345, false);
        let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
        let mut group = c.benchmark_group(format!("jxl-context/RGB-8/{side}x{side}"));
        group.throughput(Throughput::Elements(u64::from(side) * u64::from(side)));
        group.bench_function("encode", |b| b.iter(|| PreparedJxl::new(black_box(&image)).unwrap().encode(options).unwrap()));
        group.bench_function("encode-fresh-reference", |b| {
            b.iter(|| {
                let mut encoder = encoder_builder()
                    .init_buffer_size((image.pixel_data().unwrap().len() + 1024).clamp(4096, 512 * 1024))
                    .lossless(false)
                    .speed(EncoderSpeed::Falcon)
                    .decoding_speed(0)
                    .use_container(false)
                    .jpeg_quality(85.0)
                    .uses_original_profile(false)
                    .color_encoding(ColorEncoding::Srgb)
                    .build()
                    .unwrap();
                encoder.encode(black_box(image.pixel_data().unwrap()), side, side).unwrap()
            })
        });
        group.finish();
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs, float_decode, small_images }
criterion_main!(benches);
