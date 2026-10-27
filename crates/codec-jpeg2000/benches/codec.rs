use blanket_bench_support::{SIZES, codec_fixtures, measure_codec};
use blanket_codec_jpeg2000::{decode_jpeg2000, encode_jpeg2000};
use blanket_core::PixelMode;
use criterion::{Criterion, criterion_group, criterion_main};

fn codecs(c: &mut Criterion) {
    codec_fixtures(&[PixelMode::Rgb], false, &SIZES, |image, content| {
        measure_codec(c, "jpeg2000-lossless", image, content, |image| encode_jpeg2000(image).unwrap(), |data| decode_jpeg2000(data).unwrap());
    });
    for depth in [10, 12, 16].into_iter().filter(|_| blanket_bench_support::full_matrix()) {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            let image = blanket_bench_support::fixture(512, 512, mode, depth, 12345, false);
            measure_codec(
                c,
                "jpeg2000-lossless",
                &image,
                "patterned",
                |image| encode_jpeg2000(image).unwrap(),
                |data| decode_jpeg2000(data).unwrap(),
            );
        }
    }
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = codecs }
criterion_main!(benches);
