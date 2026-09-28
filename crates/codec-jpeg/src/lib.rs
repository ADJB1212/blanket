use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use turbojpeg::{Colorspace, Compressor, Decompressor, PixelFormat, Subsamp};

pub fn decode_jpeg(data: &[u8]) -> Result<Image, String> {
    let mut decoder = Decompressor::new().map_err(|error| error.to_string())?;
    let header = decoder.read_header(data).map_err(|error| error.to_string())?;
    let width = u32::try_from(header.width).map_err(|error| error.to_string())?;
    let height = u32::try_from(header.height).map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;

    let (mode, format) = match header.colorspace {
        Colorspace::Gray => (PixelMode::L, PixelFormat::GRAY),
        Colorspace::CMYK | Colorspace::YCCK => (PixelMode::Cmyk, PixelFormat::CMYK),
        Colorspace::RGB | Colorspace::YCbCr => (PixelMode::Rgb, PixelFormat::RGB),
    };
    let pitch = header
        .width
        .checked_mul(format.size())
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let length = header
        .height
        .checked_mul(pitch)
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let mut pixels = vec![0; length];
    decoder
        .decompress(
            data,
            turbojpeg::Image {
                pixels: pixels.as_mut_slice(),
                width: header.width,
                pitch,
                height: header.height,
                format,
            },
        )
        .map_err(|error| error.to_string())?;
    if format == PixelFormat::CMYK {
        pixels.iter_mut().for_each(|value| *value = 255 - *value);
    }
    Image::from_pixels(width, height, mode, pixels, Some("JPEG".to_owned())).map_err(|error| error.to_string())
}

/// Entropy coding choices for one JPEG encoding of the same DCT coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegCoding {
    pub optimize: bool,
    pub progressive: bool,
}

impl JpegCoding {
    pub const BASELINE: Self = Self {
        optimize: false,
        progressive: false,
    };
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
    let mut encoder = Compressor::new().map_err(codec_error)?;
    encoder.set_quality(i32::from(quality)).map_err(codec_error)?;
    encoder.set_subsamp(subsampling).map_err(codec_error)?;
    if coding.optimize {
        encoder.set_optimize(true).map_err(codec_error)?;
    }
    if coding.progressive {
        encoder.set_progressive(true).map_err(codec_error)?;
    }
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
    let mut output = Vec::with_capacity(pixels.len());
    blanket_core::parallel::chunks_mut_above(&mut output.spare_capacity_mut()[..pixels.len()], 256 * 1024, 4 * 1024 * 1024, |i, dst| {
        for (value, &sample) in dst.iter_mut().zip(&pixels[i * 256 * 1024..]) {
            value.write(255 - sample);
        }
    });
    // Every reserved byte is written by the disjoint chunks above.
    unsafe { output.set_len(pixels.len()) };
    output
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
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let bias = vld1q_u16([1, 2, 1, 2, 1, 2, 1, 2].as_ptr());
            while x + 16 <= width {
                let a = vld3q_u8(top.as_ptr().add(x * 3));
                let b = vld3q_u8(bottom.as_ptr().add(x * 3));
                vst1q_u8(luma0.as_mut_ptr().add(x), a.0);
                vst1q_u8(luma1.as_mut_ptr().add(x), b.0);
                let average = |p: uint8x16_t, q: uint8x16_t| vshrn_n_u16::<2>(vaddq_u16(vaddq_u16(vpaddlq_u8(p), vpaddlq_u8(q)), bias));
                vst1_u8(cb.as_mut_ptr().add(x / 2), average(a.1, b.1));
                vst1_u8(cr.as_mut_ptr().add(x / 2), average(a.2, b.2));
                x += 16;
            }
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
