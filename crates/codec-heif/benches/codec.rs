use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_heif::{decode_heif, encode_heif};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

fn codecs(c: &mut Criterion) {
    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
    codec_fixtures(&[PixelMode::Rgb], false, &SIZES[..1], |image, content| {
        measure_codec(c, "heif-q85", image, content, |image| encode_heif(image, options).unwrap(), |data| decode_heif(data).unwrap());
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
