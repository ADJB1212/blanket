#![feature(default_field_values, portable_simd)]

mod decode;
mod entropy;
pub mod stream;

use std::cell::RefCell;

use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use turbojpeg::{Colorspace, Compressor, Decompressor, PixelFormat, Subsamp};

thread_local! {
    static JPEG_ENCODER: RefCell<Option<Compressor>> = const { RefCell::new(None) };
    static JPEG_DECODER: RefCell<Option<decode::Decoder>> = const { RefCell::new(None) };
}

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    decode::Decoder::new()?.header(data)
}

fn with_codec<T, R, E>(cache: &RefCell<Option<T>>, create: impl FnOnce() -> Result<T, E>, run: impl FnOnce(&mut T) -> Result<R, E>) -> Result<R, E> {
    let mut cached = cache.borrow_mut();
    if cached.is_none() {
        *cached = Some(create()?);
    }
    let result = run(cached.as_mut().expect("codec initialized above"));
    if result.is_err() {
        *cached = None;
    }
    result
}

pub fn transform_jpeg(
    data: &[u8], method: Option<u8>, crop: Option<(usize, usize, usize, usize)>, trim: bool, preserve_metadata: bool,
) -> PyResult<Vec<u8>> {
    use turbojpeg::{Transform, TransformCrop, TransformOp, Transformer};

    let op = match method {
        None => TransformOp::None,
        Some(0) => TransformOp::Hflip,
        Some(1) => TransformOp::Vflip,
        Some(2) => TransformOp::Rot270,
        Some(3) => TransformOp::Rot180,
        Some(4) => TransformOp::Rot90,
        Some(5) => TransformOp::Transpose,
        Some(6) => TransformOp::Transverse,
        _ => return Err(PyValueError::new_err("invalid JPEG transform method")),
    };
    let header = Decompressor::new().map_err(codec_error)?.read_header(data).map_err(codec_error)?;
    validate_dimensions(u32::try_from(header.width).map_err(codec_error)?, u32::try_from(header.height).map_err(codec_error)?)
        .map_err(codec_error)?;
    let mut transformer = Transformer::new().map_err(codec_error)?;
    let mut transform = Transform::op(op);
    transform.perfect = !trim;
    transform.trim = trim;
    transform.copy_none = !preserve_metadata;
    transform.progressive = header.is_progressive;
    transform.arithmetic = header.is_arithmetic;
    let cropped;
    let input = if let Some((left, top, right, bottom)) = crop {
        if left >= right || top >= bottom || right > header.width || bottom > header.height {
            return Err(PyValueError::new_err("crop must be a nonempty box inside the JPEG"));
        }
        let (mcu_width, mcu_height) = (header.subsamp.mcu_width(), header.subsamp.mcu_height());
        if !left.is_multiple_of(mcu_width) || !top.is_multiple_of(mcu_height) {
            return Err(PyValueError::new_err(format!("crop origin must align to JPEG MCU boundaries ({mcu_width} x {mcu_height})")));
        }
        let mut crop_transform = transform.clone();
        crop_transform.op = TransformOp::None;
        crop_transform.crop = Some(TransformCrop { x: left, y: top, width: Some(right - left), height: Some(bottom - top) });
        cropped = transformer.transform_to_vec(&crop_transform, data).map_err(codec_error)?;
        if method.is_none() {
            return Ok(cropped);
        }
        cropped.as_slice()
    } else {
        data
    };
    transformer.transform_to_vec(&transform, input).map_err(codec_error)
}

pub fn decode_jpeg(data: &[u8]) -> Result<Image, String> {
    JPEG_DECODER.with(|cache| with_codec(cache, decode::Decoder::new, |decoder| decoder.decode(data)))
}

#[cfg(test)]
fn decode_jpeg_with(decoder: &mut Decompressor, data: &[u8]) -> Result<Image, String> {
    let header = decoder.read_header(data).map_err(|error| error.to_string())?;
    let width = u32::try_from(header.width).map_err(|error| error.to_string())?;
    let height = u32::try_from(header.height).map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;

    let (mode, format) = match header.colorspace {
        Colorspace::Gray => (PixelMode::L, PixelFormat::GRAY),
        Colorspace::CMYK | Colorspace::YCCK => (PixelMode::Cmyk, PixelFormat::CMYK),
        Colorspace::RGB | Colorspace::YCbCr => (PixelMode::Rgb, PixelFormat::RGB),
    };
    let pitch = header.width.checked_mul(format.size()).ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let length = header.height.checked_mul(pitch).ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let mut pixels = vec![0; length];
    decoder
        .decompress(data, turbojpeg::Image { pixels: pixels.as_mut_slice(), width: header.width, pitch, height: header.height, format })
        .map_err(|error| error.to_string())?;
    if format == PixelFormat::CMYK {
        pixels.iter_mut().for_each(|value| *value = 255 - *value);
    }
    Image::from_pixels(width, height, mode, pixels, Some("JPEG".to_owned())).map_err(|error| error.to_string())
}

/// Entropy coding choices for one JPEG encoding of the same DCT coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JpegCoding {
    pub optimize: bool = false,
    pub progressive: bool = false,
}

impl JpegCoding {
    pub const BASELINE: Self = Self { .. };
}

pub fn encode_jpeg(image: &Image, pixels: &[u8], quality: u8) -> PyResult<Vec<u8>> {
    encode_jpeg_coded(image, pixels, quality, JpegCoding::BASELINE)
}

/// Encode with the normal save's validation, quality, and subsampling, but
/// with the given entropy coding. Quantized coefficients do not depend on the
/// entropy coder, so this equals losslessly transforming the normal save while
/// avoiding a second entropy decode and re-encode of the whole image.
pub fn encode_jpeg_variant(image: &Image, options: SaveOptions, coding: JpegCoding) -> PyResult<Vec<u8>> {
    if image.bit_depth > 8 {
        image.raw_data()?;
        return Err(PyValueError::new_err(
            "high-bit-depth saving supports HEIF, PNG, TIFF, or JXL; convert to bit_depth=8 explicitly for this format",
        ));
    }
    encode_jpeg_coded(image, image.pixel_data()?, options.quality, coding)
}

fn encode_jpeg_coded(image: &Image, pixels: &[u8], quality: u8, coding: JpegCoding) -> PyResult<Vec<u8>> {
    JPEG_ENCODER.with(|cache| {
        with_codec(cache, || Compressor::new().map_err(codec_error), |encoder| encode_jpeg_with(encoder, image, pixels, quality, coding))
    })
}

fn encode_jpeg_with(encoder: &mut Compressor, image: &Image, pixels: &[u8], quality: u8, coding: JpegCoding) -> PyResult<Vec<u8>> {
    if image.mode == PixelMode::Rgba {
        return Err(PyOSError::new_err("cannot write mode RGBA as JPEG"));
    }

    let (format, subsampling) = match image.mode {
        PixelMode::L => (PixelFormat::GRAY, Subsamp::Gray),
        // Match the default used by Pillow/libjpeg for RGB JPEG output.
        PixelMode::Rgb | PixelMode::YCbCr => (PixelFormat::RGB, Subsamp::Sub2x2),
        PixelMode::Cmyk => (PixelFormat::CMYK, Subsamp::None),
        PixelMode::Rgba => unreachable!("RGBA is rejected above"),
        _ => return Err(PyOSError::new_err(format!("cannot write mode {} as JPEG", image.mode.as_str()))),
    };
    encoder.set_quality(i32::from(quality)).map_err(codec_error)?;
    encoder.set_subsamp(subsampling).map_err(codec_error)?;
    encoder.set_optimize(coding.optimize).map_err(codec_error)?;
    encoder.set_progressive(coding.progressive).map_err(codec_error)?;
    encoder
        .set_colorspace(match image.mode {
            PixelMode::L => Colorspace::Gray,
            PixelMode::Cmyk => Colorspace::CMYK,
            _ => Colorspace::YCbCr,
        })
        .map_err(codec_error)?;
    if image.mode == PixelMode::YCbCr {
        let (width, height) = (image.width as usize, image.height as usize);
        let planes = ycbcr_planes(pixels, width, height);
        let chroma = width.div_ceil(2);
        return encoder
            .compress_yuv_planes_to_vec(&turbojpeg::YuvPlanesImage {
                y_plane: &planes.0,
                u_plane: &planes.1,
                v_plane: &planes.2,
                width,
                height,
                y_stride: width.next_multiple_of(2),
                u_stride: chroma,
                v_stride: chroma,
                subsamp: Subsamp::Sub2x2,
            })
            .map_err(codec_error);
    }
    let inverted;
    let pixels = if image.mode == PixelMode::Cmyk {
        // Like Pillow, store inverted Adobe CMYK without a YCCK transform.
        encoder.set_colorspace(Colorspace::CMYK).map_err(codec_error)?;
        inverted = invert_samples(pixels);
        inverted.as_slice()
    } else {
        pixels
    };
    encoder
        .compress_to_vec(turbojpeg::Image {
            pixels,
            width: image.width as usize,
            pitch: image.width as usize * format.size(),
            height: image.height as usize,
            format,
        })
        .map_err(codec_error)
}

fn invert_samples(pixels: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;

    if blanket_core::parallel::should_parallel(pixels.len(), 256 * 1024, 4 * 1024 * 1024) {
        pixels.par_iter().with_min_len(256 * 1024).map(|&sample| 255 - sample).collect()
    } else {
        pixels.iter().map(|&sample| 255 - sample).collect()
    }
}

/// Planar Y plus 2x2 chroma averaged like libjpeg's h2v2 downsampler, with
/// replicated right and bottom edges. Y is padded to even dimensions.
fn ycbcr_planes(pixels: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    let y_stride = width.next_multiple_of(2);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let mut y = vec![0; y_stride * ch * 2];
    let mut cb = vec![0; cw * ch];
    let mut cr = vec![0; cw * ch];
    if width == 0 || height == 0 {
        return (y, cb, cr);
    }
    let row = |r: usize| &pixels[r.min(height - 1) * width * 3..(r.min(height - 1) + 1) * width * 3];
    let work = |cy: usize, luma: &mut [u8], cb: &mut [u8], cr: &mut [u8]| {
        let (top, bottom) = (row(cy * 2), row(cy * 2 + 1));
        let (luma0, luma1) = luma.split_at_mut(y_stride);
        let mut x = 0;
        use std::simd::num::SimdUint;
        use std::simd::{Simd, simd_swizzle};

        use blanket_core::pixels::load;
        let bias = Simd::<u16, 8>::from_array([1, 2, 1, 2, 1, 2, 1, 2]);
        while x + 16 <= width {
            let a = load::<3>(&top[x * 3..]);
            let b = load::<3>(&bottom[x * 3..]);
            a[0].copy_to_slice(&mut luma0[x..x + 16]);
            b[0].copy_to_slice(&mut luma1[x..x + 16]);
            let average = |p: Simd<u8, 16>, q: Simd<u8, 16>| {
                let even_p = simd_swizzle!(p, [0, 2, 4, 6, 8, 10, 12, 14]);
                let odd_p = simd_swizzle!(p, [1, 3, 5, 7, 9, 11, 13, 15]);
                let even_q = simd_swizzle!(q, [0, 2, 4, 6, 8, 10, 12, 14]);
                let odd_q = simd_swizzle!(q, [1, 3, 5, 7, 9, 11, 13, 15]);
                ((even_p.cast::<u16>() + odd_p.cast::<u16>() + even_q.cast::<u16>() + odd_q.cast::<u16>() + bias) >> 2).cast::<u8>()
            };
            average(a[1], b[1]).copy_to_slice(&mut cb[x / 2..x / 2 + 8]);
            average(a[2], b[2]).copy_to_slice(&mut cr[x / 2..x / 2 + 8]);
            x += 16;
        }
        for i in x..width {
            luma0[i] = top[i * 3];
            luma1[i] = bottom[i * 3];
        }
        if width < y_stride {
            luma0[width] = luma0[width - 1];
            luma1[width] = luma1[width - 1];
        }
        for cx in x / 2..cw {
            let (l, r) = (cx * 6, (cx * 2 + 1).min(width - 1) * 3);
            let bias = 1 + (cx as u16 & 1);
            let sum = |c: usize| (u16::from(top[l + c]) + u16::from(top[r + c]) + u16::from(bottom[l + c]) + u16::from(bottom[r + c]) + bias) >> 2;
            cb[cx] = sum(1) as u8;
            cr[cx] = sum(2) as u8;
        }
    };
    if blanket_core::parallel::should_parallel(pixels.len(), 64 * 1024, 1024 * 1024) {
        y.par_chunks_mut(y_stride * 2)
            .zip(cb.par_chunks_mut(cw))
            .zip(cr.par_chunks_mut(cw))
            .enumerate()
            .for_each(|(cy, ((luma, cb), cr))| work(cy, luma, cb, cr));
    } else {
        for (cy, ((luma, cb), cr)) in y.chunks_mut(y_stride * 2).zip(cb.chunks_mut(cw)).zip(cr.chunks_mut(cw)).enumerate() {
            work(cy, luma, cb, cr);
        }
    }
    (y, cb, cr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_codecs_match_fresh_codecs_after_mode_quality_and_error_changes() {
        for mode in [PixelMode::Cmyk, PixelMode::Rgb, PixelMode::L, PixelMode::YCbCr, PixelMode::Rgb] {
            for size in [(1, 1), (17, 9), (257, 65)] {
                let pixels = (0..size.0 * size.1 * mode.channels() as u32).map(|i| (i * 137 + i / 7) as u8).collect();
                let image = Image::from_pixels(size.0, size.1, mode, pixels, None).unwrap();
                for quality in [25, 100, 85] {
                    for coding in [JpegCoding { progressive: true, optimize: true }, JpegCoding::BASELINE] {
                        let actual = encode_jpeg_coded(&image, image.pixel_data().unwrap(), quality, coding).unwrap();
                        let expected =
                            encode_jpeg_with(&mut Compressor::new().unwrap(), &image, image.pixel_data().unwrap(), quality, coding).unwrap();
                        assert_eq!(actual, expected);
                        let expected = decode_jpeg_with(&mut Decompressor::new().unwrap(), &actual).unwrap();
                        let decoded = decode_jpeg(&actual).unwrap();
                        assert_eq!(decoded.size(), expected.size());
                        assert_eq!(decoded.mode, expected.mode);
                        assert_eq!(decoded.raw_data().unwrap(), expected.raw_data().unwrap());
                        assert!(decode_jpeg(&actual[..actual.len() / 2]).is_err());
                        assert_eq!(decode_jpeg(&actual).unwrap().raw_data().unwrap(), expected.raw_data().unwrap());
                    }
                }
                assert!(encode_jpeg(&image, image.pixel_data().unwrap(), 0).is_err());
            }
        }
    }

    #[test]
    fn inversion_preserves_length_and_matches_scalar_across_parallel_threshold() {
        for len in [0, 1, 15, 16, 17, (4 << 20) - 1, (4 << 20) + 3] {
            let source: Vec<u8> = (0..len).map(|i| (i * 73 + i / 7) as u8).collect();
            let output = invert_samples(&source);
            assert_eq!(output.len(), len);
            assert!(output.iter().zip(&source).all(|(&value, &sample)| value == 255 - sample));
        }
    }
}
