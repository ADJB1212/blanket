use std::simd::cmp::SimdOrd;

use blanket_core::pixels::{load, store};

pub(crate) fn max_assign(output: &mut [u8], source: &[u8]) {
    assert_eq!(output.len(), source.len());
    let end = output.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let [a] = load::<1>(&output[i..]);
        let [b] = load::<1>(&source[i..]);
        store(&mut output[i..], [a.simd_max(b)]);
    }
    for (dst, &src) in output[end..].iter_mut().zip(&source[end..]) {
        *dst = (*dst).max(src);
    }
}

pub(crate) fn horizontal_max(source: &[u8], radius: usize, output: &mut [u8]) {
    assert_eq!(output.len(), source.len() + 2 * radius);
    output[..source.len()].copy_from_slice(source);
    output[source.len()..].fill(0);
    let window = 2 * radius + 1;
    let mut covered = 1;
    while covered < window {
        let shift = covered.min(window - covered);
        let mut end = output.len();
        // Traverse backwards so each pass reads the previous pass's values.
        while end >= shift + 16 {
            let start = end - 16;
            let [a] = load::<1>(&output[start..]);
            let [b] = load::<1>(&output[start - shift..]);
            store(&mut output[start..], [a.simd_max(b)]);
            end = start;
        }
        for i in (shift..end).rev() {
            output[i] = output[i].max(output[i - shift]);
        }
        covered += shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_handles_unaligned_slices_tails_and_guards() {
        for length in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
            for offset in 0..16 {
                let source: Vec<u8> = (0..length + 16).map(|i| (i * 73) as u8).collect();
                let mut output = vec![91; length + 32];
                let expected: Vec<u8> = source[offset..offset + length].iter().map(|&value| value.max(91)).collect();
                max_assign(&mut output[offset..offset + length], &source[offset..offset + length]);
                assert_eq!(&output[offset..offset + length], expected);
                assert!(output[..offset].iter().all(|&value| value == 91));
                assert!(output[offset + length..].iter().all(|&value| value == 91));
            }
        }
    }

    #[test]
    fn horizontal_windows_match_scalar_with_unaligned_output_and_guards() {
        for width in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
            for radius in [0, 1, 2, 4, 5, 8, 17, 32, 1024] {
                let input: Vec<u8> = (0..width + 3).map(|i| (i * 173) as u8).collect();
                let source = &input[3..];
                for offset in 0..16 {
                    let length = width + 2 * radius;
                    let mut output = vec![87; length + 32];
                    horizontal_max(source, radius, &mut output[offset..offset + length]);
                    for x in 0..length {
                        let expected = source[x.saturating_sub(2 * radius).min(width)..(x + 1).min(width)].iter().copied().max().unwrap_or(0);
                        assert_eq!(output[offset + x], expected, "width {width}, radius {radius}, x {x}");
                    }
                    assert!(output[..offset].iter().all(|&value| value == 87));
                    assert!(output[offset + length..].iter().all(|&value| value == 87));
                }
            }
        }
    }
}
