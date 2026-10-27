use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_common::SaveOptions;
use blanket_codec_png::{decode_png, encode_png, encode_wide_png};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

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

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
