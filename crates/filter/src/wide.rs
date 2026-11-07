use std::borrow::Cow;
use std::collections::VecDeque;
use std::simd::Simd;
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdInt};

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;

pub(crate) fn kernel(image: &Image, size: usize, scale: f32, offset: f32, weights: &[f32]) -> PyResult<Image> {
    let source = integer_view(image.raw_data()?);
    let mut values = source.to_vec();
    let (width, height) = (image.width as usize, image.height as usize);
    convolve(&source, &mut values, size, (width, height), scale, offset, weights);
    drop(source);
    Image::from_integer_bytes(image.width, image.height, image.mode, values.into_iter().map(i32::to_le_bytes).collect::<Vec<_>>().into_flattened())
}

fn integer_view(bytes: &[u8]) -> Cow<'_, [i32]> {
    if cfg!(target_endian = "little") {
        // All i32 bit patterns are valid; align_to checks alignment and bounds.
        let (prefix, values, suffix) = unsafe { bytes.align_to::<i32>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Cow::Borrowed(values);
        }
    }
    Cow::Owned(bytes.as_chunks::<4>().0.iter().map(|&value| i32::from_le_bytes(value)).collect())
}

fn float_view(bytes: &[u8]) -> Cow<'_, [f32]> {
    if cfg!(target_endian = "little") {
        // All f32 bit patterns are valid; align_to checks alignment and bounds.
        let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Cow::Borrowed(values);
        }
    }
    Cow::Owned(bytes.as_chunks::<4>().0.iter().map(|&value| f32::from_le_bytes(value)).collect())
}

fn convolve(source: &[i32], values: &mut [i32], size: usize, dimensions: (usize, usize), scale: f32, offset: f32, weights: &[f32]) {
    let (width, height) = dimensions;
    if width >= size && height >= size {
        let radius = size / 2;
        let weights: Vec<f32> = weights.iter().map(|weight| weight / scale).collect();
        blanket_core::parallel::chunks_mut_above(&mut values[radius * width..(height - radius) * width], width * 16, 256 * 1024, |band, rows| {
            for (row_index, output) in rows.chunks_exact_mut(width).enumerate() {
                let y = radius + band * 16 + row_index;
                let first = y * width + radius;
                let last = (y + 1) * width - radius;
                let end = first + (last - first) / 16 * 16;
                for i in (first..end).step_by(16) {
                    let mut sum = Simd::splat(offset + 0.5);
                    for ky in 0..size {
                        let start = i + radius * width - ky * width - radius;
                        let at = |kx| Simd::<i32, 16>::from_slice(&source[start + kx..]).cast::<f32>();
                        let mut row =
                            super::filter::kernel_add_simd(at(0), Simd::splat(weights[ky * size]), at(1) * Simd::splat(weights[ky * size + 1]));
                        for kx in 2..size {
                            row = super::filter::kernel_add_simd(at(kx), Simd::splat(weights[ky * size + kx]), row);
                        }
                        sum += row;
                    }
                    sum.cast::<i32>().simd_max(Simd::splat(0)).copy_to_slice(&mut output[i - y * width..i - y * width + 16]);
                }
                for (relative, value) in output[end - y * width..last - y * width].iter_mut().enumerate() {
                    let i = end + relative;
                    let mut sum = offset + 0.5;
                    for ky in 0..size {
                        let start = i + radius * width - ky * width - radius;
                        let mut row =
                            super::filter::kernel_add(source[start] as f32, weights[ky * size], source[start + 1] as f32 * weights[ky * size + 1]);
                        for kx in 2..size {
                            row = super::filter::kernel_add(source[start + kx] as f32, weights[ky * size + kx], row);
                        }
                        sum += row;
                    }
                    *value = (sum as i32).max(0);
                }
            }
        });
    }
}

pub(crate) fn rank(image: &Image, size: usize, rank: usize) -> PyResult<Image> {
    let bytes = image.raw_data()?;
    if size == 1 {
        return blanket_core::buffers::output(image, image.size(), blanket_core::buffers::copy(bytes)?);
    }
    let pixels = if image.mode == PixelMode::I {
        let source = integer_view(bytes);
        integer_rank(source, image.width as usize, image.height as usize, size, rank)?
            .into_iter()
            .map(i32::to_le_bytes)
            .collect::<Vec<_>>()
            .into_flattened()
    } else {
        let source = float_view(bytes);
        let values = rank_values(&source, image.width as usize, image.height as usize, size, rank)?;
        drop(source);
        values.into_iter().map(f32::to_le_bytes).collect::<Vec<_>>().into_flattened()
    };
    blanket_core::buffers::output(image, (image.width, image.height), pixels)
}

fn integer_rank(source: Cow<'_, [i32]>, width: usize, height: usize, size: usize, rank: usize) -> PyResult<Vec<i32>> {
    if source.is_empty() || size == 1 {
        return Ok(source.into_owned());
    }
    if rank != 0 && rank != size * size - 1 {
        if size >= 9 && (source.is_sorted() || source.windows(2).all(|pair| pair[0] >= pair[1])) {
            return Ok(monotonic_integer_rank(&source, width, height, size, rank));
        }
        if size >= 31 || size * size > source.len() * 4 {
            return Ok(sliding_integer_rank(&source, width, height, size / 2, rank));
        }
        return rank_values(&source, width, height, size, rank);
    }
    let mut source = source.into_owned();
    let radius = size / 2;
    let maximum = rank != 0;
    let mut horizontal = vec![0; source.len()];
    blanket_core::parallel::chunks_mut_above(&mut horizontal, width * 16, 256 * 1024, |band, rows| {
        let mut queue = VecDeque::new();
        for (i, row) in rows.chunks_exact_mut(width).enumerate() {
            let input = &source[(band * 16 + i) * width..][..width];
            super::filter::extrema_line(input, row, 1, 0, radius, maximum, &mut queue);
        }
    });
    let band_rows = 64.max(radius.min(height));
    blanket_core::parallel::chunks_mut_above(&mut source, width * band_rows, 256 * 1024, |band, rows| {
        let mut queue = VecDeque::new();
        for column in 0..width {
            super::filter::extrema_line(&horizontal[column..], &mut rows[column..], width, band * band_rows, radius, maximum, &mut queue);
        }
    });
    Ok(source)
}

fn monotonic_integer_rank(source: &[i32], width: usize, height: usize, size: usize, rank: usize) -> Vec<i32> {
    let rank = if source[0] > source[source.len() - 1] { size * size - 1 - rank } else { rank };
    let radius = size / 2;
    let mut output = Vec::with_capacity(source.len());
    for y in 0..height {
        let yy = (y + rank / size).saturating_sub(radius).min(height - 1);
        let first = if yy == 0 { 0 } else { (yy + radius).saturating_sub(y) };
        let weight =
            1 + if yy == 0 { radius.saturating_sub(y) } else { 0 } + if yy + 1 == height { (y + radius).saturating_sub(height - 1) } else { 0 };
        let column = (rank - first * size) / weight;
        for x in 0..width {
            let xx = (x + column).saturating_sub(radius).min(width - 1);
            output.push(source[yy * width + xx]);
        }
    }
    output
}

fn sliding_integer_rank(source: &[i32], width: usize, height: usize, radius: usize, rank: usize) -> Vec<i32> {
    let mut output = vec![0; source.len()];
    if source.is_empty() {
        return output;
    }
    let band_rows = 64.max(radius.min(height));
    blanket_core::parallel::chunks_mut_above(&mut output, width * band_rows, 256 * 1024, |band, rows| {
        sliding_integer_rank_rows(source, rows, (width, height), radius, rank, band * band_rows);
    });
    output
}

fn sliding_integer_rank_rows(source: &[i32], output: &mut [i32], dimensions: (usize, usize), radius: usize, rank: usize, first: usize) {
    let (width, height) = dimensions;
    let last = first + output.len() / width;
    let top = first.saturating_sub(radius);
    let bottom = (last + radius).min(height);
    let mut keys = source[top * width..bottom * width].to_vec();
    keys.sort_unstable();
    keys.dedup();
    if keys.len() == 1 {
        output.fill(keys[0]);
        return;
    }
    let indices: Vec<usize> = source[top * width..bottom * width].iter().map(|value| keys.binary_search(value).unwrap()).collect();
    let mut counts = super::high_depth::RankCounts { tree: vec![0; keys.len() + 1] };
    let weight = |position: usize, center: usize, length: usize| {
        1 + if position == 0 { radius.saturating_sub(center) } else { 0 }
            + if position + 1 == length { (center + radius).saturating_sub(length - 1) } else { 0 }
    };
    for y in first.saturating_sub(radius)..=(first + radius).min(height - 1) {
        for x in 0..=radius.min(width - 1) {
            let index = indices[(y - top) * width + x];
            counts.update(index, weight(x, 0, width) as u64 * weight(y, first, height) as u64, true);
        }
    }
    let column = |counts: &mut super::high_depth::RankCounts, x: usize, y: usize, add: bool| {
        for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
            let index = indices[(yy - top) * width + x];
            counts.update(index, weight(yy, y, height) as u64, add);
        }
    };
    let row = |counts: &mut super::high_depth::RankCounts, y: usize, x: usize, add: bool| {
        for xx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
            let index = indices[(y - top) * width + xx];
            counts.update(index, weight(xx, x, width) as u64, add);
        }
    };
    let mut x = 0;
    for y in first..last {
        for position in 0..width {
            output[(y - first) * width + x] = keys[counts.select(rank as u64)];
            if position + 1 < width {
                if (y - first).is_multiple_of(2) {
                    column(&mut counts, x.saturating_sub(radius), y, false);
                    column(&mut counts, (x + radius + 1).min(width - 1), y, true);
                    x += 1;
                } else {
                    column(&mut counts, (x + radius).min(width - 1), y, false);
                    column(&mut counts, x.saturating_sub(radius + 1), y, true);
                    x -= 1;
                }
            }
        }
        if y + 1 < last {
            row(&mut counts, y.saturating_sub(radius), x, false);
            row(&mut counts, (y + radius + 1).min(height - 1), x, true);
        }
    }
}

fn rank_values<T: Copy + PartialOrd>(source: &[T], width: usize, height: usize, size: usize, rank: usize) -> PyResult<Vec<T>> {
    let mut output = source.to_vec();
    if output.is_empty() || size == 1 {
        return Ok(output);
    }
    let mut window = Vec::new();
    window.try_reserve_exact(size * size).map_err(|_| PyMemoryError::new_err("cannot allocate rank window"))?;
    let radius = size / 2;
    for y in 0..height {
        for x in 0..width {
            window.clear();
            for ky in 0..size {
                let yy = (y + ky).saturating_sub(radius).min(height - 1);
                for kx in 0..size {
                    let xx = (x + kx).saturating_sub(radius).min(width - 1);
                    window.push(source[yy * width + xx]);
                }
            }
            output[y * width + x] = select(&mut window, rank);
        }
    }
    Ok(output)
}

fn select<T: Copy + PartialOrd>(values: &mut [T], rank: usize) -> T {
    let (mut left, mut right) = (0_isize, values.len() as isize - 1);
    while left < right {
        let pivot = values[rank];
        let (mut i, mut j) = (left, right);
        loop {
            while values[i as usize] < pivot {
                i += 1;
            }
            while pivot < values[j as usize] {
                j -= 1;
            }
            if i <= j {
                values.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
            }
            if i > j {
                break;
            }
        }
        if j < rank as isize {
            left = i;
        }
        if (rank as isize) < i {
            right = j;
        }
    }
    values[rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_views_preserve_unaligned_bits_and_borrow_native_storage() {
        for count in [0, 1, 15, 16, 17, 4097] {
            for offset in 0..4 {
                let values: Vec<u32> = (0..count).map(|i| [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, u32::MAX][i % 6]).collect();
                let mut storage = vec![173; offset];
                storage.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                let bytes = &storage[offset..];
                let integers = integer_view(bytes);
                let floats = float_view(bytes);
                assert_eq!(integers.iter().map(|&v| v as u32).collect::<Vec<_>>(), values);
                assert_eq!(floats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), values);
                if count != 0 {
                    let borrowed = cfg!(target_endian = "little") && bytes.as_ptr().align_offset(align_of::<i32>()) == 0;
                    assert_eq!(matches!(integers, Cow::Borrowed(_)), borrowed);
                    assert_eq!(matches!(floats, Cow::Borrowed(_)), borrowed);
                }
            }
        }
    }

    #[test]
    fn sliding_integer_ranks_match_square_selection() {
        for (width, height) in [(1, 1), (1, 7), (7, 1), (2, 7), (19, 5), (65, 17), (1, 131), (19, 137)] {
            for pattern in 0..3 {
                let source: Vec<i32> = (0..width * height)
                    .map(|i| match pattern {
                        0 => 17,
                        1 => [i32::MIN, 0, i32::MAX][i % 3],
                        _ => (i * 7919 % 4093) as i32 - 2046,
                    })
                    .collect();
                for size in [9, 11, 31] {
                    for rank in [1, size * size / 2, size * size - 2] {
                        let expected = rank_values(&source, width, height, size, rank).unwrap();
                        assert_eq!(
                            integer_rank(Cow::Borrowed(&source), width, height, size, rank).unwrap(),
                            expected,
                            "width={width}, height={height}, size={size}, rank={rank}, pattern={pattern}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn monotonic_integer_ranks_preserve_weighted_rows_and_columns() {
        for (width, height) in [(1, 1), (1, 9), (9, 1), (3, 7), (19, 13)] {
            for repeated in [false, true] {
                for descending in [false, true] {
                    let mut source: Vec<i32> = (0..width * height).map(|i| (if repeated { i / 3 } else { i }) as i32 - 17).collect();
                    if descending {
                        source.reverse();
                    }
                    for size in [3, 9, 31] {
                        for rank in [0, 1, size, size * size / 2, size * size - 2, size * size - 1] {
                            assert_eq!(
                                monotonic_integer_rank(&source, width, height, size, rank),
                                rank_values(&source, width, height, size, rank).unwrap(),
                                "width={width}, height={height}, size={size}, rank={rank}, repeated={repeated}, descending={descending}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sliding_rank_aggregates_large_clamped_borders() {
        let source = vec![i32::MIN, 17, i32::MAX];
        let size = 23169;
        for (rank, expected) in [(1, vec![i32::MIN; 3]), (size * size / 2, source.clone()), (size * size - 2, vec![i32::MAX; 3])] {
            assert_eq!(integer_rank(Cow::Borrowed(&source), 3, 1, size, rank).unwrap(), expected);
            assert_eq!(integer_rank(Cow::Borrowed(&source), 1, 3, size, rank).unwrap(), expected);
            assert_eq!(sliding_integer_rank(&source, 3, 1, size / 2, rank), expected);
            assert_eq!(sliding_integer_rank(&source, 1, 3, size / 2, rank), expected);
        }
    }

    #[test]
    fn integer_extrema_match_square_windows() {
        for (width, height) in [(0, 1), (1, 0), (1, 1), (2, 7), (19, 5), (65, 67), (513, 521)] {
            for size in [3, 5, 9, 31, 129] {
                if width > 100 && size > 9 {
                    continue;
                }
                let storage: Vec<i32> = (0..width * height + 3)
                    .map(|i| match i % 7 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        _ => (i * 7919 % 4093) as i32 - 2046,
                    })
                    .collect();
                let source = &storage[3..];
                for rank in [0, size * size - 1] {
                    let actual = integer_rank(Cow::Borrowed(source), width, height, size, rank).unwrap();
                    let radius = size / 2;
                    for y in 0..height {
                        for x in 0..width {
                            let window = (y.saturating_sub(radius)..=(y + radius).min(height - 1))
                                .flat_map(|yy| (x.saturating_sub(radius)..=(x + radius).min(width - 1)).map(move |xx| source[yy * width + xx]));
                            let expected = if rank == 0 { window.min() } else { window.max() }.unwrap();
                            assert_eq!(actual[y * width + x], expected, "size={size}, x={x}, y={y}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_convolution_covers_unaligned_vectors_tails_borders_and_guards() {
        for size in [3, 5] {
            for width in 0..40 {
                for height in [0, 1, 2, 3, 5, 7] {
                    for offset in 0..4 {
                        let count = width * height;
                        let storage: Vec<i32> = (0..offset + count).map(|i| (i * 37 % 511) as i32 - 255).collect();
                        let source = &storage[offset..];
                        for nonfinite in [false, true] {
                            let weights: Vec<f32> = (0..size * size).map(|i| if nonfinite { f32::NAN } else { (i % 5) as f32 - 2.0 }).collect();
                            let mut expected = source.to_vec();
                            if width >= size && height >= size {
                                let radius = size / 2;
                                for y in radius..height - radius {
                                    for x in radius..width - radius {
                                        let mut sum = 3.0;
                                        for ky in 0..size {
                                            for kx in 0..size {
                                                sum += source[(y + radius - ky) * width + x + kx - radius] as f32 * weights[ky * size + kx];
                                            }
                                        }
                                        expected[y * width + x] = (sum as i32).max(0);
                                    }
                                }
                            }
                            let mut output = vec![0x5a5a; offset + count + 13];
                            output[offset..offset + count].copy_from_slice(source);
                            convolve(source, &mut output[offset..offset + count], size, (width, height), 1.0, 2.5, &weights);
                            assert_eq!(&output[offset..offset + count], expected, "width={width}, height={height}, size={size}");
                            assert!(output[..offset].iter().chain(&output[offset + count..]).all(|&value| value == 0x5a5a));
                        }
                    }
                }
            }
        }
    }
}
