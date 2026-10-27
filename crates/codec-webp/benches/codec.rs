use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_webp::{decode_webp, encode_webp};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba], false, &SIZES, |image, content| {
        for (name, settings) in [("webp-lossy-q85", options), ("webp-lossless", SaveOptions { lossless: true, ..options })] {
            measure_codec(
                c,
                name,
                image,
                content,
                |image| encode_webp(image, image.pixel_data().unwrap(), settings).unwrap(),
                |data| decode_webp(data).unwrap(),
            );
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
