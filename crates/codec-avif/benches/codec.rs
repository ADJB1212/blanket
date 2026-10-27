use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_avif::{decode_avif, encode_avif};
use blanket_codec_common::SaveOptions;
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::Rgb], false, &SIZES[..1], |image, content| {
        measure_codec(
            c,
            "avif-q85-e3",
            image,
            content,
            |image| encode_avif(image, image.pixel_data().unwrap(), options).unwrap(),
            |data| decode_avif(data).unwrap(),
        );
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
