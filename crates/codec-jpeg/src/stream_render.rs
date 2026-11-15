use std::simd::Simd;
use std::simd::cmp::{SimdOrd, SimdPartialOrd};
use std::simd::num::{SimdInt, SimdUint};

use blanket_core::{Image, PixelMode, pixels};
use libjpeg_turbo_rs::JpegCoefficients;
use libjpeg_turbo_rs::decode::color::ycbcr_to_rgb_row;
use libjpeg_turbo_rs::decode::upsample;
use rayon::prelude::*;

use super::entropy::ZIGZAG;

type I32x8 = Simd<i32, 8>;

fn pass(s: [I32x8; 8]) -> [I32x8; 8] {
    let c = I32x8::splat;
    let z = (s[2] + s[6]) * c(4433);
    let even2 = z - s[6] * c(15137);
    let even3 = z + s[2] * c(6270);
    let even0 = (s[0] + s[4]) << 13;
    let even1 = (s[0] - s[4]) << 13;
    let t0 = even0 + even3;
    let t3 = even0 - even3;
    let t1 = even1 + even2;
    let t2 = even1 - even2;
    let z1 = (s[7] + s[1]) * c(-7373);
    let z2 = (s[5] + s[3]) * c(-20995);
    let z3 = s[7] + s[3];
    let z4 = s[5] + s[1];
    let z5 = (z3 + z4) * c(9633);
    let z3 = z3 * c(-16069) + z5;
    let z4 = z4 * c(-3196) + z5;
    let o0 = s[7] * c(2446) + z1 + z3;
    let o1 = s[5] * c(16819) + z2 + z4;
    let o2 = s[3] * c(25172) + z2 + z3;
    let o3 = s[1] * c(12299) + z1 + z4;
    [t0 + o3, t1 + o2, t2 + o1, t3 + o0, t3 - o0, t2 - o1, t1 - o2, t0 - o3]
}

fn transpose(rows: [I32x8; 8]) -> [I32x8; 8] {
    let rows = rows.map(Simd::to_array);
    std::array::from_fn(|column| I32x8::from_array(std::array::from_fn(|row| rows[row][column])))
}

fn idct(block: &[i16; 64], quant: &[u16; 64]) -> Option<[u8; 64]> {
    let mut natural = [0; 64];
    for (k, &index) in ZIGZAG.iter().enumerate() {
        natural[index] = i32::from(block[k]) * i32::from(quant[index]);
    }
    let rows = std::array::from_fn(|row| I32x8::from_slice(&natural[row * 8..][..8]));
    if rows.iter().any(|row| (row.simd_lt(I32x8::splat(-32768)) | row.simd_gt(I32x8::splat(32767))).any()) {
        return None;
    }
    // Match libjpeg's 13-bit constants and two retained pass-one bits.
    let columns = pass(rows).map(|value| (value + I32x8::splat(1024)) >> 11);
    if columns.iter().any(|row| (row.simd_lt(I32x8::splat(-32768)) | row.simd_gt(I32x8::splat(32767))).any()) {
        return None;
    }
    let columns = pass(transpose(columns)).map(|value| (value + I32x8::splat(131072)) >> 18);
    let mut output = [0; 64];
    for (row, values) in transpose(columns).iter().enumerate() {
        let values = (*values + I32x8::splat(128)).simd_clamp(I32x8::splat(0), I32x8::splat(255)).cast::<u8>();
        output[row * 8..][..8].copy_from_slice(&values.to_array());
    }
    Some(output)
}

fn rgb_row(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8], width: usize) {
    let end = width / 16 * 16;
    let c = Simd::<i32, 16>::splat;
    for x in (0..end).step_by(16) {
        let y = pixels::Bytes::from_slice(&y[x..]).cast::<i32>();
        let cb = pixels::Bytes::from_slice(&cb[x..]).cast::<i32>() - c(128);
        let cr = pixels::Bytes::from_slice(&cr[x..]).cast::<i32>() - c(128);
        let r = y + ((cr * c(91881) + c(32768)) >> 16);
        let g = y - ((cb * c(22554) + cr * c(46802) + c(32768)) >> 16);
        let b = y + ((cb * c(116130) + c(32768)) >> 16);
        pixels::store::<3>(&mut output[x * 3..], [r, g, b].map(|value| value.simd_clamp(c(0), c(255)).cast::<u8>()));
    }
    ycbcr_to_rgb_row(&y[end..], &cb[end..], &cr[end..], &mut output[end * 3..], width - end);
}

pub(super) fn render(coefficients: &JpegCoefficients, adobe: Option<u8>) -> Result<Option<Image>, String> {
    let components = &coefficients.components;
    let count = components.len();
    let (width, height) = (usize::from(coefficients.width), usize::from(coefficients.height));
    if !matches!(count, 1 | 3)
        || width < 8
        || height < 8
        || adobe == Some(0)
        || adobe.is_some_and(|value| value > 1)
        || components.iter().map(|c| c.component_id).eq(*b"RGB")
        || coefficients.quant_tables.iter().flatten().any(|&value| value > 255)
    {
        return Ok(None);
    }
    let max_h = components.iter().map(|c| usize::from(c.h_sampling)).max().unwrap();
    let max_v = components.iter().map(|c| usize::from(c.v_sampling)).max().unwrap();
    if components[0].h_sampling as usize != max_h
        || components[0].v_sampling as usize != max_v
        || components.iter().any(|c| {
            !max_h.is_multiple_of(c.h_sampling as usize)
                || !max_v.is_multiple_of(c.v_sampling as usize)
                || !matches!((max_h / c.h_sampling as usize, max_v / c.v_sampling as usize), (1, 1) | (2, 1) | (2, 2))
        })
    {
        return Ok(None);
    }
    let plane = |component: &libjpeg_turbo_rs::ComponentCoefficients| {
        let stride = component.blocks_x * 8;
        let mut plane = vec![0; stride * component.blocks_y * 8];
        let quant = &coefficients.quant_tables[component.quant_table_index as usize];
        for (index, block) in component.blocks.iter().enumerate() {
            let pixels = idct(block, quant)?;
            let start = (index / component.blocks_x * stride + index % component.blocks_x) * 8;
            for row in 0..8 {
                for column in 0..8 {
                    plane[start + row * stride + column] = pixels[row * 8 + column];
                }
            }
        }
        Some((plane, stride))
    };
    let parallel = blanket_core::parallel::should_parallel(width * height, 256 * 1024, 1024 * 1024);
    let planes: Option<Vec<_>> = if parallel { components.par_iter().map(plane).collect() } else { components.iter().map(plane).collect() };
    let Some(planes) = planes else {
        return Ok(None);
    };
    if count == 1 {
        let mut output = Vec::with_capacity(width * height);
        for row in planes[0].0.chunks_exact(planes[0].1).take(height) {
            output.extend_from_slice(&row[..width]);
        }
        return Image::from_pixels(width as u32, height as u32, PixelMode::L, output, Some("JPEG".into()))
            .map(Some)
            .map_err(|error| error.to_string());
    }
    let planes: Vec<_> = planes
        .into_iter()
        .zip(components)
        .map(|((plane, stride), component)| {
            let h = max_h / component.h_sampling as usize;
            let v = max_v / component.v_sampling as usize;
            if h == 1 {
                return (plane, stride);
            }
            let columns = width.div_ceil(2);
            let rows = height.div_ceil(v);
            let out_stride = columns * 2;
            let mut output = vec![0; out_stride * rows * v];
            if v == 2 {
                upsample::fancy_h2v2_strided(&plane, columns, stride, rows, &mut output, out_stride);
            } else {
                for row in 0..rows {
                    upsample::fancy_h2v1(&plane[row * stride..], columns, &mut output[row * out_stride..], out_stride);
                }
            }
            (output, out_stride)
        })
        .collect();
    let mut output = vec![0; width * height * 3];
    let convert = |(row, pixels): (usize, &mut [u8])| {
        rgb_row(&planes[0].0[row * planes[0].1..], &planes[1].0[row * planes[1].1..], &planes[2].0[row * planes[2].1..], pixels, width);
    };
    if parallel {
        output.par_chunks_exact_mut(width * 3).enumerate().for_each(convert);
    } else {
        output.chunks_exact_mut(width * 3).enumerate().for_each(convert);
    }
    Image::from_pixels(width as u32, height as u32, PixelMode::Rgb, output, Some("JPEG".into())).map(Some).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use blanket_codec_common::SaveOptions;
    use libjpeg_turbo_rs::read_coefficients;

    use super::*;

    #[test]
    fn direct_render_matches_eager_for_sampling_edges_and_quantization() {
        for mode in [PixelMode::L, PixelMode::Rgb] {
            for (width, height) in [(8, 8), (17, 9), (33, 19), (65, 47)] {
                let pixels = (0..width * height * mode.channels()).map(|i| (i * 71 + i / 7) as u8).collect();
                let image = Image::from_pixels(width as u32, height as u32, mode, pixels, None).unwrap();
                for jpeg_subsampling in [0, 1, 2] {
                    for quality in [10, 75, 100] {
                        let encoded =
                            crate::encode_jpeg_options(&image, image.pixel_data().unwrap(), SaveOptions { jpeg_subsampling, quality, .. }).unwrap();
                        let coefficients = read_coefficients(&encoded).unwrap();
                        let actual = render(&coefficients, None).unwrap().expect("supported JPEG should render directly");
                        let expected = crate::decode_jpeg(&encoded).unwrap();
                        assert_eq!(
                            actual.pixel_data().unwrap(),
                            expected.pixel_data().unwrap(),
                            "{mode:?}, {width}x{height}, sampling={jpeg_subsampling}, quality={quality}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_rows_preserve_unaligned_vectors_tails_and_guards() {
        let y: Vec<_> = (0..82).map(|i| (i * 71) as u8).collect();
        let cb: Vec<_> = (0..82).map(|i| (i * 37) as u8).collect();
        let cr: Vec<_> = (0..82).map(|i| (i * 17) as u8).collect();
        for width in 0..=65 {
            let mut expected = vec![0; width * 3];
            ycbcr_to_rgb_row(&y[1..], &cb[2..], &cr[3..], &mut expected, width);
            let mut actual = vec![73; width * 3 + 2];
            rgb_row(&y[1..], &cb[2..], &cr[3..], &mut actual[1..width * 3 + 1], width);
            assert_eq!(&actual[1..width * 3 + 1], expected);
            assert_eq!((actual[0], actual[width * 3 + 1]), (73, 73));
        }
    }

    #[test]
    fn idct_overflow_uses_the_established_renderer() {
        let mut block = [0; 64];
        block[0] = 1023;
        assert!(idct(&block, &[255; 64]).is_none());
        block[0] = 8192;
        assert!(idct(&block, &[1; 64]).is_none());
    }
}
