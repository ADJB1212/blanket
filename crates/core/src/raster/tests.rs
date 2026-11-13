use pyo3::prelude::*;

use super::packing::lab_raw_bytes_mut;
use super::queries::{extrema, sample_extrema};

#[test]
fn sample_extrema_match_scalar_across_unaligned_vectors_and_tails() {
    fn check<const C: usize>() {
        for count in [0, 1, 15, 16, 17, 255, 256, 257, 1025] {
            for offset in 0..4 {
                let mut data = vec![91; offset];
                let mut expected = [(u16::MAX, 0); C];
                for i in 0..count {
                    for (c, range) in expected.iter_mut().enumerate() {
                        let value = ((i * 7919 + c * 137) % 65536) as u16;
                        data.extend_from_slice(&value.to_le_bytes());
                        range.0 = range.0.min(value);
                        range.1 = range.1.max(value);
                    }
                }
                data.extend_from_slice(&[93; 4]);
                assert_eq!(sample_extrema::<C>(&data[offset..offset + count * C * 2], u16::MAX), expected);
                assert!(data[..offset].iter().all(|&v| v == 91));
                assert_eq!(&data[offset + count * C * 2..], &[93; 4]);
            }
        }
    }
    check::<1>();
    check::<3>();
    check::<4>();
}

#[test]
fn mutable_high_depth_samples_are_revalidated_and_cloned_independently() {
    use super::{Image, PixelMode};

    for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
        for depth in [10, 12] {
            let mut image = Image::from_samples(1, 1, mode, vec![0; mode.channels()], depth, None).unwrap();
            image.pixels_mut().unwrap()[..2].copy_from_slice(&u16::MAX.to_le_bytes());
            let mut cloned = image.clone();
            assert!(image.raw_data().is_err());
            assert!(cloned.raw_data().is_err());
            cloned.pixels_mut().unwrap()[..2].copy_from_slice(&1_u16.to_le_bytes());
            assert!(cloned.raw_data().is_ok());
            assert!(image.raw_data().is_err());
            image.pixels_mut().unwrap()[..2].copy_from_slice(&0_u16.to_le_bytes());
            assert!(image.raw_data().is_ok());
            image.pixels_mut().unwrap()[..2].copy_from_slice(&(1_u16 << depth).to_le_bytes());
            assert!(image.raw_data().is_err());
        }
    }
}

struct Storage(Vec<u8>);

impl super::SharedPixels for Storage {
    fn bytes(&self) -> &[u8] {
        &self.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn shared_storage_validates_layout_without_copying() {
    Python::initialize();
    for (mode, depth) in [
        (PixelMode::One, 8),
        (PixelMode::L, 8),
        (PixelMode::Rgb, 10),
        (PixelMode::Rgb, 12),
        (PixelMode::Rgba, 16),
        (PixelMode::I, 32),
        (PixelMode::I16, 16),
        (PixelMode::I16B, 16),
        (PixelMode::F, 32),
        (PixelMode::RgbF, 32),
        (PixelMode::RgbaF, 32),
    ] {
        let length = 6 * mode.channels() * usize::from(depth).div_ceil(8);
        let storage = std::sync::Arc::new(Storage(vec![0; length]));
        let pointer = storage.0.as_ptr();
        let image = Image::from_shared(3, 2, mode, depth, storage).unwrap();
        assert_eq!(image.raw_data().unwrap().as_ptr(), pointer);
        assert_eq!(image.raw_bytes_per_pixel(), length / 6);
        assert_eq!(image.into_pixels().unwrap(), vec![0; length]);
        for invalid in [length - 1, length + 1] {
            assert!(Image::from_shared(3, 2, mode, depth, std::sync::Arc::new(Storage(vec![0; invalid]))).is_err());
        }
    }
    for (mode, depth) in [(PixelMode::Rgb, 32), (PixelMode::F, 8), (PixelMode::I, 16), (PixelMode::La, 16), (PixelMode::L, 9)] {
        assert!(Image::from_shared(1, 1, mode, depth, std::sync::Arc::new(Storage(vec![0; 4]))).is_err());
        assert!(Image::from_raw_bytes(1, 1, mode, vec![0; 4], depth).is_err());
    }
    for depth in [10, 12] {
        let data = (1_u16 << depth).to_le_bytes().to_vec();
        assert!(Image::from_shared(1, 1, PixelMode::L, depth, std::sync::Arc::new(Storage(data))).is_err());
    }
    assert!(Image::from_shared(u32::MAX, u32::MAX, PixelMode::RgbaF, 32, std::sync::Arc::new(Storage(Vec::new()))).is_err());
}

#[test]
fn storage_replacement_and_reinterpretation_preserve_invariants() {
    Python::initialize();
    let mut image = Image::from_pixels(2, 1, PixelMode::L, vec![1, 2], Some("PNG".into())).unwrap();
    let replacement = Image::from_pixels(1, 2, PixelMode::L, vec![3, 4], None).unwrap();
    assert!(image.replace_storage(replacement).is_err());
    assert_eq!(image.raw_data().unwrap(), [1, 2]);
    assert!(image.reinterpret_mode(PixelMode::Rgb).is_err());
    assert_eq!(image.pixel_mode(), PixelMode::L);
    let replacement = Image::from_raw_bytes(2, 1, PixelMode::L, vec![0, 1, 0, 2], 16).unwrap();
    image.replace_storage(replacement).unwrap();
    image.reinterpret_mode(PixelMode::I16).unwrap();
    assert_eq!(image.bit_depth(), 16);
    assert_eq!(image.format(), Some("PNG"));
    assert_eq!(image.raw_data().unwrap(), [0, 1, 0, 2]);
}

#[test]
fn changing_shared_storage_cannot_bypass_validation_on_detach() {
    struct ChangingStorage(std::sync::atomic::AtomicBool);

    impl SharedPixels for ChangingStorage {
        fn bytes(&self) -> &[u8] {
            if self.0.load(std::sync::atomic::Ordering::Relaxed) { &[1] } else { &[1, 2] }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    Python::initialize();
    let storage = std::sync::Arc::new(ChangingStorage(std::sync::atomic::AtomicBool::new(false)));
    let shared = std::sync::Arc::clone(&storage);
    let mut image = Image::from_shared(2, 1, PixelMode::L, 8, shared).unwrap();
    storage.0.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(image.raw_data().is_err());
    assert!(image.ensure_owned().is_err());
    assert!(image.pixels_mut().is_none());
    storage.0.store(false, std::sync::atomic::Ordering::Relaxed);
    image.ensure_owned().unwrap();
    assert_eq!(image.raw_data().unwrap(), [1, 2]);
}

#[test]
fn shared_pixels_detach_on_write_and_close_independently() {
    let mut image = Image::from_shared(3, 1, PixelMode::L, 8, std::sync::Arc::new(Storage(vec![1, 2, 3]))).unwrap();
    let mut sibling = image.clone();
    assert_eq!(image.raw_data().unwrap(), [1, 2, 3]);
    image.ensure_owned().unwrap();
    image.pixels_mut().unwrap()[0] = 9;
    assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
    assert_eq!(sibling.raw_data().unwrap(), [1, 2, 3]);
    assert!(image.shared_pixels.is_none());
    sibling.close();
    assert!(sibling.shared_pixels.is_none());
    assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
}

#[test]
fn owned_raster_copy_shares_storage_until_mutated() {
    Python::initialize();
    let mut image = Image::from_pixels(3, 1, PixelMode::L, vec![1, 2, 3], Some("PNG".into())).unwrap();
    let mut sibling = Python::attach(|py| image.copy(py)).unwrap();
    assert!(std::ptr::eq(image.raw_data().unwrap(), sibling.raw_data().unwrap()));
    assert_eq!(sibling.format(), None);
    image.pixels_mut().unwrap()[0] = 9;
    assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
    assert_eq!(sibling.raw_data().unwrap(), [1, 2, 3]);
    sibling.close();
    assert_eq!(image.into_pixels().unwrap(), [9, 2, 3]);
}

#[test]
fn zeroed_pixel_allocations_support_vec_mutation_and_growth() {
    for len in [0, 1, 15, 16, 17, 4095, 4096, 4097, 1024 * 1024] {
        let mut pixels = super::alloc_pixels(len).unwrap();
        assert_eq!(pixels.len(), len);
        assert!(pixels.iter().all(|&value| value == 0));
        pixels.fill(173);
        pixels.push(91);
        assert_eq!(pixels[len], 91);
        assert!(pixels[..len].iter().all(|&value| value == 173));
    }
    assert!(super::alloc_pixels(usize::MAX).is_err());
}

use super::*;

#[test]
fn lab_conversion_bounds_capacity_and_reuses_owned_storage() {
    for count in [0, 1, 15, 16, 17, 1048577] {
        let source: Vec<u8> = (0..count * 3).map(|i| (i * 71) as u8).collect();
        let output = lab_raw_bytes(&source);
        assert_eq!(output.len(), source.len());
        assert!(output.capacity() <= source.len() + 47);
        let mut owned = output;
        let pointer = owned.as_ptr();
        lab_raw_bytes_mut(&mut owned);
        assert_eq!(owned, source);
        assert_eq!(owned.as_ptr(), pointer);
    }
}

#[test]
fn bilevel_rows_preserve_partial_bytes_and_parallel_partitions() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            for width in [0_usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 800, 1023, 4097] {
                for height in [0, 1, 3, 129, 1025] {
                    let row_bytes = width.div_ceil(8);
                    let packed: Vec<_> = (0..row_bytes * height).map(|i| (i * 37 + 0x95) as u8).collect();
                    let pixels = unpack_bilevel(&packed, width, height);
                    let inverted = unpack_bilevel_inverted(&packed, width, height);
                    assert_eq!(pixels.len(), width * height);
                    for y in 0..height {
                        for x in 0..width {
                            let value = if packed[y * row_bytes + x / 8] & (128 >> (x % 8)) != 0 { 255 } else { 0 };
                            assert_eq!(pixels[y * width + x], value);
                            assert_eq!(inverted[y * width + x], !value);
                        }
                    }
                    let mut expected = packed;
                    if width % 8 != 0 {
                        for row in expected.chunks_mut(row_bytes) {
                            row[row_bytes - 1] &= 0xff << (8 - width % 8);
                        }
                    }
                    assert_eq!(pack_bilevel(&pixels, width, height), expected);
                }
            }
        });
    }
}

#[test]
fn extrema_matches_scalar_for_unaligned_buffers_and_tails() {
    fn check<const C: usize>() {
        for offset in 0..16 {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 521] {
                for fill in [None, Some(0), Some(255)] {
                    let mut data: Vec<u8> = (0..offset + count * C).map(|i| fill.unwrap_or(((i * 37 + i / 11) % 192 + 32) as u8)).collect();
                    if fill.is_none() && count >= 2 {
                        for c in 0..C {
                            data[offset + (count - 1) * C + c] = c as u8;
                            data[offset + (count - 2) * C + c] = 255 - c as u8;
                        }
                    }
                    let data = &data[offset..];
                    let expected: Vec<_> = (0..C)
                        .map(|c| {
                            data.as_chunks::<C>()
                                .0
                                .iter()
                                .fold((255_u16, 0_u16), |(low, high), p| (low.min(u16::from(p[c])), high.max(u16::from(p[c]))))
                        })
                        .collect();
                    assert_eq!(extrema::<C>(data), expected, "C={C}, offset={offset}, count={count}");
                }
            }
        }
    }
    check::<1>();
    check::<2>();
    check::<3>();
    check::<4>();
}

#[test]
pub(super) fn validates_buffer_length() {
    Python::initialize();
    Python::attach(|_| {
        let error = Image::from_pixels(2, 2, PixelMode::Rgb, vec![0; 11], None).expect_err("invalid buffer should fail");
        assert!(error.to_string().contains("expected 12 bytes"));
    });
}
