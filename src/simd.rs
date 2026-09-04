use crate::raster::PixelMode;

pub(crate) fn convert(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    debug_assert_ne!(from, to, "caller should short-circuit identity conversion");

    match (from, to) {
        (PixelMode::Rgba, PixelMode::Rgb) => rgba_to_rgb(source),
        (PixelMode::Rgb, PixelMode::Rgba) => rgb_to_rgba(source),
        (PixelMode::L, PixelMode::Rgb) => l_to_rgb(source),
        (PixelMode::L, PixelMode::Rgba) => l_to_rgba(source),
        (PixelMode::Rgb, PixelMode::L) => rgb_to_l(source),
        (PixelMode::Rgba, PixelMode::L) => rgba_to_l(source),
        _ => unreachable!("all mode pairs are covered"),
    }
}

#[inline(always)]
fn simd_prefix<F>(source: &[u8], output: &mut [u8], f: F) -> (usize, usize)
where
    F: FnOnce(&[u8], &mut [u8]) -> (usize, usize),
{
    f(source, output)
}

fn rgba_to_rgb(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / 4;
    let mut output = vec![0u8; pixel_count * 3];

    #[cfg(target_arch = "aarch64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::rgba_to_rgb(s, o);
        ((di / 3) * 4, di)
    });

    #[cfg(target_arch = "x86_64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = ssse3::rgba_to_rgb(s, o);
        ((di / 3) * 4, di)
    });

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    while si + 3 < source.len() {
        output[di] = source[si];
        output[di + 1] = source[si + 1];
        output[di + 2] = source[si + 2];
        si += 4;
        di += 3;
    }
    output
}

fn rgb_to_rgba(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / 3;
    let mut output = vec![0u8; pixel_count * 4];

    #[cfg(target_arch = "aarch64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::rgb_to_rgba(s, o);
        ((di / 4) * 3, di)
    });

    #[cfg(target_arch = "x86_64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = ssse3::rgb_to_rgba(s, o);
        ((di / 4) * 3, di)
    });

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    while si + 2 < source.len() {
        output[di] = source[si];
        output[di + 1] = source[si + 1];
        output[di + 2] = source[si + 2];
        output[di + 3] = 255;
        si += 3;
        di += 4;
    }
    output
}

fn l_to_rgb(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; source.len() * 3];

    #[cfg(target_arch = "aarch64")]
    let (si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::l_to_rgb(s, o);
        (di / 3, di)
    });

    #[cfg(not(target_arch = "aarch64"))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    for &luma in &source[si..] {
        output[di] = luma;
        output[di + 1] = luma;
        output[di + 2] = luma;
        di += 3;
    }
    let _ = si;
    output
}

fn l_to_rgba(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; source.len() * 4];

    #[cfg(target_arch = "aarch64")]
    let (si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::l_to_rgba(s, o);
        (di / 4, di)
    });

    #[cfg(not(target_arch = "aarch64"))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    for &luma in &source[si..] {
        output[di] = luma;
        output[di + 1] = luma;
        output[di + 2] = luma;
        output[di + 3] = 255;
        di += 4;
    }
    let _ = si;
    output
}

fn rgb_to_l(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / 3;
    let mut output = vec![0u8; pixel_count];

    #[cfg(target_arch = "aarch64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::rgb_to_l(s, o);
        (di * 3, di)
    });

    #[cfg(not(target_arch = "aarch64"))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    while si + 2 < source.len() {
        let r = source[si] as u32;
        let g = source[si + 1] as u32;
        let b = source[si + 2] as u32;
        output[di] = ((r * 19_595 + g * 38_470 + b * 7_471 + 0x8000) >> 16) as u8;
        si += 3;
        di += 1;
    }
    output
}

fn rgba_to_l(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / 4;
    let mut output = vec![0u8; pixel_count];

    #[cfg(target_arch = "aarch64")]
    let (mut si, mut di) = simd_prefix(source, &mut output, |s, o| {
        let di = neon::rgba_to_l(s, o);
        (di * 4, di)
    });

    #[cfg(not(target_arch = "aarch64"))]
    let (mut si, mut di): (usize, usize) = (0, 0);

    while si + 3 < source.len() {
        let r = source[si] as u32;
        let g = source[si + 1] as u32;
        let b = source[si + 2] as u32;
        output[di] = ((r * 19_595 + g * 38_470 + b * 7_471 + 0x8000) >> 16) as u8;
        si += 4;
        di += 1;
    }
    output
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub(super) fn rgba_to_rgb(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = source.len() / 64;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 64;
            unsafe {
                let rgba = vld4q_u8(source.as_ptr().add(si));
                let rgb = uint8x16x3_t(rgba.0, rgba.1, rgba.2);
                vst3q_u8(output.as_mut_ptr().add(di), rgb);
            }
            di += 48;
        }
        di
    }

    pub(super) fn rgb_to_rgba(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = source.len() / 48;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 48;
            unsafe {
                let rgb = vld3q_u8(source.as_ptr().add(si));
                let alpha = vdupq_n_u8(255);
                let rgba = uint8x16x4_t(rgb.0, rgb.1, rgb.2, alpha);
                vst4q_u8(output.as_mut_ptr().add(di), rgba);
            }
            di += 64;
        }
        di
    }

    pub(super) fn l_to_rgb(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = source.len() / 16;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 16;
            unsafe {
                let luma = vld1q_u8(source.as_ptr().add(si));
                let rgb = uint8x16x3_t(luma, luma, luma);
                vst3q_u8(output.as_mut_ptr().add(di), rgb);
            }
            di += 48;
        }
        di
    }

    pub(super) fn l_to_rgba(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = source.len() / 16;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 16;
            unsafe {
                let luma = vld1q_u8(source.as_ptr().add(si));
                let alpha = vdupq_n_u8(255);
                let rgba = uint8x16x4_t(luma, luma, luma, alpha);
                vst4q_u8(output.as_mut_ptr().add(di), rgba);
            }
            di += 64;
        }
        di
    }

    pub(super) fn rgb_to_l(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = (source.len() / 3) / 8;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 24;
            unsafe {
                let rgb = vld3_u8(source.as_ptr().add(si));
                let r16 = vmovl_u8(rgb.0);
                let g16 = vmovl_u8(rgb.1);
                let b16 = vmovl_u8(rgb.2);

                let coeff_r = vdupq_n_u16(77);
                let coeff_g = vdupq_n_u16(150);
                let coeff_b = vdupq_n_u16(29);

                let acc = vmulq_u16(r16, coeff_r);
                let acc = vmlaq_u16(acc, g16, coeff_g);
                let acc = vmlaq_u16(acc, b16, coeff_b);
                let acc = vaddq_u16(acc, vdupq_n_u16(128));

                let result = vshrn_n_u16(acc, 8);
                vst1_u8(output.as_mut_ptr().add(di), result);
            }
            di += 8;
        }
        di
    }

    pub(super) fn rgba_to_l(source: &[u8], output: &mut [u8]) -> usize {
        let chunks = source.len() / 64;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 64;
            unsafe {
                let rgba = vld4q_u8(source.as_ptr().add(si));

                let coeff_r = vdupq_n_u16(77);
                let coeff_g = vdupq_n_u16(150);
                let coeff_b = vdupq_n_u16(29);
                let bias = vdupq_n_u16(128);

                let r_lo = vmovl_u8(vget_low_u8(rgba.0));
                let g_lo = vmovl_u8(vget_low_u8(rgba.1));
                let b_lo = vmovl_u8(vget_low_u8(rgba.2));
                let acc_lo = vmulq_u16(r_lo, coeff_r);
                let acc_lo = vmlaq_u16(acc_lo, g_lo, coeff_g);
                let acc_lo = vmlaq_u16(acc_lo, b_lo, coeff_b);
                let acc_lo = vaddq_u16(acc_lo, bias);
                let res_lo = vshrn_n_u16(acc_lo, 8);

                let r_hi = vmovl_u8(vget_high_u8(rgba.0));
                let g_hi = vmovl_u8(vget_high_u8(rgba.1));
                let b_hi = vmovl_u8(vget_high_u8(rgba.2));
                let acc_hi = vmulq_u16(r_hi, coeff_r);
                let acc_hi = vmlaq_u16(acc_hi, g_hi, coeff_g);
                let acc_hi = vmlaq_u16(acc_hi, b_hi, coeff_b);
                let acc_hi = vaddq_u16(acc_hi, bias);
                let res_hi = vshrn_n_u16(acc_hi, 8);

                let result = vcombine_u8(res_lo, res_hi);
                vst1q_u8(output.as_mut_ptr().add(di), result);
            }
            di += 16;
        }
        di
    }
}

#[cfg(target_arch = "x86_64")]
mod ssse3 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "ssse3")]
    unsafe fn rgba_to_rgb_inner(source: &[u8], output: &mut [u8]) -> usize {
        let shuf = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
        let chunks = source.len() / 16;
        let mut di = 0;
        for i in 0..chunks {
            let si = i * 16;
            let rgba = _mm_loadu_si128(source.as_ptr().add(si) as *const __m128i);
            let rgb = _mm_shuffle_epi8(rgba, shuf);

            let ptr = output.as_mut_ptr().add(di);
            _mm_storel_epi64(ptr as *mut __m128i, rgb);
            let upper = _mm_srli_si128(rgb, 8);
            std::ptr::copy_nonoverlapping(&upper as *const __m128i as *const u8, ptr.add(8), 4);
            di += 12;
        }
        di
    }

    pub(super) fn rgba_to_rgb(source: &[u8], output: &mut [u8]) -> usize {
        if is_x86_feature_detected!("ssse3") {
            unsafe { rgba_to_rgb_inner(source, output) }
        } else {
            0
        }
    }

    #[target_feature(enable = "ssse3")]
    unsafe fn rgb_to_rgba_inner(source: &[u8], output: &mut [u8]) -> usize {
        let shuf = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        let alpha_mask = _mm_setr_epi8(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);

        let safe_chunks = if source.len() >= 16 {
            (source.len() - 16) / 12 + 1
        } else {
            0
        };

        let mut di = 0;
        for i in 0..safe_chunks {
            let si = i * 12;
            let rgb = _mm_loadu_si128(source.as_ptr().add(si) as *const __m128i);
            let shuffled = _mm_shuffle_epi8(rgb, shuf);
            let result = _mm_or_si128(shuffled, alpha_mask);
            _mm_storeu_si128(output.as_mut_ptr().add(di) as *mut __m128i, result);
            di += 16;
        }
        di
    }

    pub(super) fn rgb_to_rgba(source: &[u8], output: &mut [u8]) -> usize {
        if is_x86_feature_detected!("ssse3") {
            unsafe { rgb_to_rgba_inner(source, output) }
        } else {
            0
        }
    }
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
    fn rgb_to_l_matches_scalar() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 12, 34, 56];
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::L);
        let expected = scalar_luminance_rgb(&rgb);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got as i16 - exp as i16).unsigned_abs() <= 1,
                "pixel {i}: SIMD={got}, scalar={exp}"
            );
        }
    }

    #[test]
    fn rgba_to_l_matches_scalar() {
        let rgba = [
            255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 12, 34, 56, 200,
        ];
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::L);
        let expected = scalar_luminance_rgba(&rgba);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got as i16 - exp as i16).unsigned_abs() <= 1,
                "pixel {i}: SIMD={got}, scalar={exp}"
            );
        }
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
