use std::hint::black_box;

use blanket_bench_support::{SIZES, codec_fixtures, fixture, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_jpeg::{decode_jpeg, encode_jpeg};
use blanket_core::{Image, PixelMode};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use turbojpeg::{Compressor, Decompressor, PixelFormat, Subsamp};

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::L, PixelMode::Rgb], false, &SIZES, |image, content| {
        measure_codec(
            c,
            "jpeg-quality-85",
            image,
            content,
            |image| encode_jpeg(image, image.pixel_data().unwrap(), options.quality).unwrap(),
            |data| decode_jpeg(data).unwrap(),
        );
    });
}

fn small_images(c: &mut Criterion) {
    for side in [32, 64] {
        let image = fixture(side, side, PixelMode::Rgb, 8, 12345, false);
        let encoded = encode_jpeg(&image, image.pixel_data().unwrap(), 85).unwrap();
        let mut group = c.benchmark_group(format!("jpeg-context/RGB-8/{side}x{side}"));
        group.throughput(Throughput::Elements(u64::from(side) * u64::from(side)));
        group.bench_function("encode", |b| b.iter(|| encode_jpeg(black_box(&image), image.pixel_data().unwrap(), 85).unwrap()));
        group.bench_function("encode-fresh-reference", |b| {
            b.iter(|| {
                let mut encoder = Compressor::new().unwrap();
                encoder.set_quality(85).unwrap();
                encoder.set_subsamp(Subsamp::Sub2x2).unwrap();
                encoder
                    .compress_to_vec(turbojpeg::Image {
                        pixels: black_box(image.pixel_data().unwrap()),
                        width: side as usize,
                        height: side as usize,
                        pitch: side as usize * 3,
                        format: PixelFormat::RGB,
                    })
                    .unwrap()
            })
        });
        group.bench_function("decode", |b| b.iter(|| decode_jpeg(black_box(&encoded)).unwrap()));
        group.bench_function("decode-fresh-reference", |b| {
            b.iter(|| {
                let data = black_box(&encoded);
                let mut decoder = Decompressor::new().unwrap();
                let header = decoder.read_header(data).unwrap();
                let mut pixels = vec![0; header.width * header.height * 3];
                decoder
                    .decompress(
                        data,
                        turbojpeg::Image {
                            pixels: &mut pixels[..],
                            width: header.width,
                            height: header.height,
                            pitch: header.width * 3,
                            format: PixelFormat::RGB,
                        },
                    )
                    .unwrap();
                Image::from_pixels(side, side, PixelMode::Rgb, pixels, Some("JPEG".into())).unwrap()
            })
        });
        group.finish();
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs, small_images }
criterion_main!(benches);
