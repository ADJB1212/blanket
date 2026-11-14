use blanket_kernels::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_kernels::pixels::alpha_scale;

/// Copy cache-sized chunks, then scale them in place while they are hot.
pub fn convert<const C: usize>(source: &[u8], output: &mut [u8], premultiply: bool) {
    assert_eq!(source.len(), output.len());
    chunks_mut(output, CHUNK_PIXELS * C, |chunk, dst| {
        dst.copy_from_slice(&source[chunk * CHUNK_PIXELS * C..][..dst.len()]);
        if premultiply {
            alpha_scale::<C, false>(dst);
        } else {
            alpha_scale::<C, true>(dst);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const C: usize>() {
        for premultiply in [false, true] {
            for offset in 0..4 {
                for count in [0, 1, 15, 16, 17, 31, 32, 33, 65536] {
                    let mut source = vec![93; offset + count * C + 1];
                    for (i, pixel) in source[offset..offset + count * C].as_chunks_mut::<C>().0.iter_mut().enumerate() {
                        pixel.fill((i % 256) as u8);
                        pixel[C - 1] = (i / 256) as u8;
                    }
                    let mut output = vec![77; source.len()];
                    convert::<C>(&source[offset..offset + count * C], &mut output[offset..offset + count * C], premultiply);
                    assert!(output[..offset].iter().all(|&v| v == 77));
                    assert_eq!(output[offset + count * C], 77);
                    for (src, dst) in source[offset..offset + count * C].as_chunks::<C>().0.iter().zip(output[offset..].as_chunks::<C>().0) {
                        let alpha = u32::from(src[C - 1]);
                        for c in 0..C - 1 {
                            let sample = u32::from(src[c]);
                            let expected =
                                if premultiply { (sample * alpha + 127) / 255 } else { (sample * 255).checked_div(alpha).unwrap_or(sample).min(255) };
                            assert_eq!(u32::from(dst[c]), expected);
                        }
                        assert_eq!(dst[C - 1], src[C - 1]);
                    }
                    assert_eq!(source[0], if offset == 0 && count > 0 { 0 } else { 93 });
                    assert_eq!(source[offset + count * C], 93);
                }
            }
        }
    }

    #[test]
    fn alpha_arithmetic_matches_scalar_with_unaligned_buffers_and_guards() {
        check::<2>();
        check::<4>();
    }
}
