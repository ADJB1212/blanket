use crate::raster::PixelMode;
use garb::bytes;

pub(crate) fn convert(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    debug_assert_ne!(from, to, "caller should short-circuit identity conversion");

    match (from, to) {
        (PixelMode::Rgba, PixelMode::Rgb) => convert_layout(source, 4, 3, bytes::rgba_to_rgb),
        (PixelMode::Rgb, PixelMode::Rgba) => convert_layout(source, 3, 4, bytes::rgb_to_rgba),
        (PixelMode::L, PixelMode::Rgb) => gray_to_rgb(source),
        (PixelMode::L, PixelMode::Rgba) => convert_layout(source, 1, 4, bytes::gray_to_rgba),
        (PixelMode::Rgb, PixelMode::L) => color_to_gray::<3>(source),
        (PixelMode::Rgba, PixelMode::L) => color_to_gray::<4>(source),
        _ => unreachable!("all mode pairs are covered"),
    }
}

fn convert_layout(
    source: &[u8],
    source_channels: usize,
    destination_channels: usize,
    conversion: fn(&[u8], &mut [u8]) -> Result<(), garb::SizeError>,
) -> Vec<u8> {
    if source.is_empty() {
        return Vec::new();
    }

    let pixel_count = source.len() / source_channels;
    let mut output = vec![0; pixel_count * destination_channels];
    conversion(source, &mut output).expect("validated image buffers have matching pixel counts");
    output
}

fn gray_to_rgb(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; source.len() * 3];
    let (pixels, remainder) = output.as_chunks_mut::<3>();
    debug_assert!(remainder.is_empty());
    for (&gray, rgb) in source.iter().zip(pixels) {
        rgb.fill(gray);
    }
    output
}

fn color_to_gray<const SOURCE_CHANNELS: usize>(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / SOURCE_CHANNELS;
    let mut output = vec![0u8; pixel_count];
    let (colors, remainder) = source.as_chunks::<SOURCE_CHANNELS>();
    debug_assert!(remainder.is_empty());
    for (color, gray) in colors.iter().zip(output.iter_mut()) {
        *gray = pillow_luma(color[0], color[1], color[2]);
    }
    output
}

#[inline(always)]
fn pillow_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 19_595 + u32::from(g) * 38_470 + u32::from(b) * 7_471 + 0x8000) >> 16) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_rgb_basic() {
        let rgba: Vec<u8> = (0..256)
            .flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8, 0xAA])
            .collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        for i in 0..256 {
            assert_eq!(result[i * 3], rgba[i * 4], "R mismatch at pixel {i}");
            assert_eq!(
                result[i * 3 + 1],
                rgba[i * 4 + 1],
                "G mismatch at pixel {i}"
            );
            assert_eq!(
                result[i * 3 + 2],
                rgba[i * 4 + 2],
                "B mismatch at pixel {i}"
            );
        }
    }

    #[test]
    fn rgb_to_rgba_basic() {
        let rgb: Vec<u8> = (0..256)
            .flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8])
            .collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::Rgba);
        for i in 0..256 {
            assert_eq!(result[i * 4], rgb[i * 3], "R mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 1], rgb[i * 3 + 1], "G mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 2], rgb[i * 3 + 2], "B mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 3], 255, "A mismatch at pixel {i}");
        }
    }

    #[test]
    fn l_to_rgb_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgb);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 3], l);
            assert_eq!(result[i * 3 + 1], l);
            assert_eq!(result[i * 3 + 2], l);
        }
    }

    #[test]
    fn l_to_rgba_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgba);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 4], l);
            assert_eq!(result[i * 4 + 1], l);
            assert_eq!(result[i * 4 + 2], l);
            assert_eq!(result[i * 4 + 3], 255);
        }
    }

    #[test]
    fn rgb_to_l_matches_pillow_luminance() {
        let rgb: Vec<u8> = (0u8..=u8::MAX)
            .flat_map(|value| [value, value.wrapping_mul(37), value.wrapping_add(113)])
            .collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::L);
        let expected = scalar_luminance_rgb(&rgb);
        assert_eq!(result, expected);
    }

    #[test]
    fn rgba_to_l_matches_pillow_luminance() {
        let rgba: Vec<u8> = (0u8..=u8::MAX)
            .flat_map(|value| {
                [
                    value,
                    value.wrapping_mul(37),
                    value.wrapping_add(113),
                    value.wrapping_mul(19),
                ]
            })
            .collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::L);
        let expected = scalar_luminance_rgba(&rgba);
        assert_eq!(result, expected);
    }

    #[test]
    fn handles_non_simd_aligned_sizes() {
        let rgba: Vec<u8> = (0..68).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        assert_eq!(result.len(), 51);
        for i in 0..17 {
            assert_eq!(result[i * 3], rgba[i * 4]);
            assert_eq!(result[i * 3 + 1], rgba[i * 4 + 1]);
            assert_eq!(result[i * 3 + 2], rgba[i * 4 + 2]);
        }
    }

    #[test]
    fn handles_empty_input() {
        assert!(convert(&[], PixelMode::Rgba, PixelMode::Rgb).is_empty());
        assert!(convert(&[], PixelMode::Rgb, PixelMode::L).is_empty());
        assert!(convert(&[], PixelMode::L, PixelMode::Rgba).is_empty());
    }

    #[test]
    fn handles_single_pixel() {
        assert_eq!(
            convert(&[10, 20, 30, 40], PixelMode::Rgba, PixelMode::Rgb),
            [10, 20, 30]
        );
        assert_eq!(
            convert(&[10, 20, 30], PixelMode::Rgb, PixelMode::Rgba),
            [10, 20, 30, 255]
        );
        assert_eq!(convert(&[42], PixelMode::L, PixelMode::Rgb), [42, 42, 42]);
    }

    fn scalar_luminance_rgb(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| {
                ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000)
                    >> 16) as u8
            })
            .collect()
    }

    fn scalar_luminance_rgba(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| {
                ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000)
                    >> 16) as u8
            })
            .collect()
    }
}
