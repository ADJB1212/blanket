use std::simd::Simd;
use std::simd::cmp::SimdOrd;

type Bytes = Simd<u8, 16>;

fn median(mut values: [Bytes; 9]) -> Bytes {
    // A selection network leaves the median at index four of the window.
    for (a, b) in [
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 1),
        (3, 4),
        (6, 7),
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 3),
        (5, 8),
        (4, 7),
        (3, 6),
        (1, 4),
        (2, 5),
        (4, 7),
        (4, 2),
        (6, 4),
        (4, 2),
    ] {
        let left = values[a];
        let right = values[b];
        values[a] = left.simd_min(right);
        values[b] = left.simd_max(right);
    }
    values[4]
}

pub(crate) fn rank_row<const K: usize, const R: usize>(source: &[u8], row: &mut [u8], channels: usize, y: usize) {
    let stride = row.len();
    let width = stride / channels;
    let height = source.len() / stride;
    let radius = K / 2;
    let first = (radius * channels).min(stride);
    let interior = width.saturating_sub(2 * radius) * channels;
    let end = first + interior / 16 * 16;
    let sample = |x: usize, ky: usize, kx: usize| {
        let yy = (y + ky).saturating_sub(radius).min(height - 1);
        let xx = (x / channels + kx).saturating_sub(radius).min(width - 1);
        source[yy * stride + xx * channels + x % channels]
    };
    for x in (first..end).step_by(16) {
        let load = |ky: usize, kx: usize| {
            let yy = (y + ky).saturating_sub(radius).min(height - 1);
            let start = yy * stride + x + kx * channels - radius * channels;
            Bytes::from_slice(&source[start..start + 16])
        };
        let result = if R == 4 {
            median(std::array::from_fn(|i| load(i / 3, i % 3)))
        } else {
            let mut result = Bytes::splat(if R == 0 { 255 } else { 0 });
            for ky in 0..K {
                for kx in 0..K {
                    let value = load(ky, kx);
                    result = if R == 0 { result.simd_min(value) } else { result.simd_max(value) };
                }
            }
            result
        };
        result.copy_to_slice(&mut row[x..x + 16]);
    }
    for x in (0..first).chain(end..stride) {
        row[x] = if R == 4 {
            let mut values: [u8; 9] = std::array::from_fn(|i| sample(x, i / 3, i % 3));
            values.sort_unstable();
            values[4]
        } else {
            let mut result = if R == 0 { 255 } else { 0 };
            for ky in 0..K {
                for kx in 0..K {
                    let value = sample(x, ky, kx);
                    result = if R == 0 { result.min(value) } else { result.max(value) };
                }
            }
            result
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_binary_windows() {
        for bits in 0_u16..512 {
            let values = std::array::from_fn(|i| Bytes::splat(((bits >> i) & 1) as u8));
            assert_eq!(median(values).to_array(), [u8::from(bits.count_ones() >= 5); 16]);
        }
    }

    fn check_rows<const K: usize, const R: usize>() {
        for channels in 1..=4 {
            for width in 1..=40 {
                for height in [1, 2, 7, 17] {
                    let stride = width * channels;
                    let mut storage = vec![0; stride * height + 3];
                    for (i, value) in storage.iter_mut().enumerate() {
                        *value = ((i * 71 + i / 7 * 113) % 256) as u8;
                    }
                    let source = &storage[3..];
                    for y in 0..height {
                        let mut output = vec![93; stride + 5];
                        rank_row::<K, R>(source, &mut output[1..stride + 1], channels, y);
                        assert_eq!(output[0], 93);
                        assert_eq!(&output[stride + 1..], &[93; 4]);
                        for x in 0..stride {
                            let mut values = Vec::new();
                            for ky in 0..K {
                                for kx in 0..K {
                                    let yy = (y + ky).saturating_sub(K / 2).min(height - 1);
                                    let xx = (x / channels + kx).saturating_sub(K / 2).min(width - 1);
                                    values.push(source[yy * stride + xx * channels + x % channels]);
                                }
                            }
                            values.sort_unstable();
                            assert_eq!(output[x + 1], values[R], "K={K} R={R} channels={channels} size={width}x{height} x={x} y={y}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rank_rows_match_scalar_with_unaligned_buffers_and_guards() {
        check_rows::<3, 0>();
        check_rows::<3, 4>();
        check_rows::<3, 8>();
        check_rows::<5, 0>();
        check_rows::<5, 24>();
    }
}
