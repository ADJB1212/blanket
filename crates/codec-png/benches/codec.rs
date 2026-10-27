use std::hint::black_box;

use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_png::{decode_png, encode_png, encode_wide_png};
use blanket_core::PixelMode;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba], true, &SIZES, |image, content| {
        for (name, settings) in [("png-level-1", SaveOptions { compress_level: 1, ..options }), ("png-level-6", options)] {
            measure_codec(
                c,
                name,
                image,
                content,
                |image| {
                    if image.bit_depth > 8 {
                        encode_wide_png(image, image.raw_data().unwrap(), settings).unwrap()
                    } else {
                        encode_png(image, image.pixel_data().unwrap(), settings.compress_level).unwrap()
                    }
                },
                |data| decode_png(data).unwrap(),
            );
        }
    });
}

fn metadata_decode(c: &mut Criterion) {
    use std::borrow::Cow;

    for kind in ["text", "icc"] {
        let mut encoded = Vec::new();
        {
            let mut info = png::Info::with_size(64, 64);
            if kind == "icc" {
                info.icc_profile = Some(Cow::Owned(vec![91; 1024 * 1024]));
            }
            let mut encoder = png::Encoder::with_info(&mut encoded, info).unwrap();
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            if kind == "text" {
                encoder.add_ztxt_chunk("comment".into(), "metadata".repeat(128 * 1024)).unwrap();
            }
            encoder.write_header().unwrap().write_image_data(&vec![91; 64 * 64 * 3]).unwrap();
        }
        let mut group = c.benchmark_group(format!("codecs/png-{kind}/RGB-8/64x64"));
        group.throughput(Throughput::Elements(64 * 64));
        group.bench_function("decode", |b| b.iter(|| decode_png(black_box(&encoded)).unwrap()));
        group.finish();
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs, metadata_decode }
criterion_main!(benches);
