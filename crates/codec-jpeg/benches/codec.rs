use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_jpeg::{decode_jpeg, encode_jpeg};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

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

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
