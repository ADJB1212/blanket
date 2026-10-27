use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_jxl::{PreparedJxl, decode_jxl};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

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

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
