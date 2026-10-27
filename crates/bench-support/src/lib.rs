use std::hint::black_box;
use std::time::Duration;

use blanket_core::{Image, PixelMode};
use criterion::{Criterion, Throughput};

pub const SIZES: [(u32, u32); 2] = [(512, 512), (2048, 1536)];

pub fn full_matrix() -> bool {
    std::env::var_os("BLANKET_BENCH_FULL").is_some_and(|value| value == "1")
}

pub fn criterion() -> Criterion {
    if full_matrix() {
        Criterion::default()
    } else {
        Criterion::default().sample_size(50).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_millis(1000)).nresamples(10_000)
    }
}

fn select<T>(values: &[T], full: bool) -> &[T] {
    if full { values } else { &values[..values.len().min(1)] }
}

pub fn sizes() -> &'static [(u32, u32)] {
    select(&SIZES, full_matrix())
}

pub fn depths() -> &'static [u8] {
    select(&[8, 16], full_matrix())
}

pub fn contents() -> &'static [(&'static str, bool)] {
    select(&[("patterned", false), ("noise", true)], full_matrix())
}

fn select_modes(modes: &[PixelMode], full: bool) -> &[PixelMode] {
    if full {
        modes
    } else if let Some(index) = modes.iter().position(|&mode| mode == PixelMode::Rgb) {
        &modes[index..=index]
    } else {
        select(modes, false)
    }
}

pub fn modes(available: &[PixelMode]) -> &[PixelMode] {
    select_modes(available, full_matrix())
}

pub fn fixture(width: u32, height: u32, mode: PixelMode, depth: u8, seed: u32, noise: bool) -> Image {
    let mut state = seed;
    let channels = mode.channels();
    let samples = (0..width as usize * height as usize * channels).map(|i| {
        if noise {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> (32 - depth)) as u16
        } else {
            let x = i / channels % width as usize;
            let y = i / channels / width as usize;
            let value = (x * 255 / width as usize + y * 127 / height as usize + i % channels * 53 + (x / 32 + y / 32) % 2 * 32) % 256;
            (value as u32 * ((1_u32 << depth) - 1) / 255) as u16
        }
    });
    if depth == 8 {
        Image::from_pixels(width, height, mode, samples.map(|value| value as u8).collect(), None).unwrap()
    } else {
        Image::from_samples(width, height, mode, samples.collect(), depth, None).unwrap()
    }
}

pub fn codec_fixtures(modes: &[PixelMode], wide: bool, sizes: &[(u32, u32)], mut bench: impl FnMut(&Image, &str)) {
    for &(width, height) in select(sizes, full_matrix()) {
        for &(content, noise) in contents() {
            for &mode in select_modes(modes, full_matrix()) {
                bench(&fixture(width, height, mode, 8, 12345, noise), content);
            }
        }
    }
    if wide && full_matrix() {
        for depth in [10, 12, 16] {
            for &mode in modes {
                bench(&fixture(512, 512, mode, depth, 12345, false), "patterned");
            }
        }
    }
}

pub fn measure_codec(
    c: &mut Criterion, name: &str, image: &Image, content: &str, encode: impl Fn(&Image) -> Vec<u8>, decode: impl Fn(&[u8]) -> Image,
) {
    let mut group =
        c.benchmark_group(format!("codecs/{name}/{}-{}/{content}/{}x{}", image.mode.as_str(), image.bit_depth, image.width, image.height));
    group.throughput(Throughput::Elements(u64::from(image.width) * u64::from(image.height)));
    group.bench_function("encode", |b| {
        b.iter(|| encode(black_box(image)));
    });
    group.bench_function("decode", |b| {
        let encoded = encode(image);
        assert_eq!(decode(&encoded).size(), image.size());
        b.iter(|| decode(black_box(&encoded)));
    });
    group.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_matrix_keeps_one_size_depth_and_rgb_mode() {
        assert_eq!(select(&SIZES, false), &[(512, 512)]);
        assert_eq!(select(&[8, 16], false), &[8]);
        assert_eq!(select_modes(&[PixelMode::L, PixelMode::Rgb, PixelMode::Rgba], false), &[PixelMode::Rgb]);
        assert_eq!(select_modes(&[PixelMode::Rgba], false), &[PixelMode::Rgba]);
        assert_eq!(select(&[("patterned", false), ("noise", true)], false), &[("patterned", false)]);
    }

    #[test]
    fn full_matrix_preserves_all_inputs() {
        assert_eq!(select(&SIZES, true), &SIZES);
        assert_eq!(select(&[8, 16], true), &[8, 16]);
        let modes = [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba];
        assert_eq!(select_modes(&modes, true), &modes);
        let contents = [("patterned", false), ("noise", true)];
        assert_eq!(select(&contents, true), &contents);
    }
}
