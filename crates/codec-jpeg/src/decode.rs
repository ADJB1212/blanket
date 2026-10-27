use std::ffi::CStr;

use blanket_codec_common::validate_dimensions;
use blanket_core::{Image, PixelMode};
use turbojpeg::raw;

pub(super) struct Decoder(raw::tjhandle);

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: this guard exclusively owns a non-null decoder.
        unsafe { raw::tj3Destroy(self.0) };
    }
}

impl Decoder {
    pub(super) fn new() -> Result<Self, String> {
        // SAFETY: initialization does not borrow any caller storage.
        let handle = unsafe { raw::tj3Init(raw::TJINIT_TJINIT_DECOMPRESS as i32) };
        if handle.is_null() {
            return Err("cannot allocate JPEG decoder".into());
        }
        Ok(Self(handle))
    }

    fn error(&self) -> String {
        // SAFETY: the live decoder owns a NUL-terminated error message.
        unsafe { CStr::from_ptr(raw::tj3GetErrorStr(self.0)) }.to_string_lossy().into_owned()
    }

    pub(super) fn header(&mut self, data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
        // SAFETY: the decoder is live and input remains borrowed through decoding.
        if unsafe { raw::tj3DecompressHeader(self.0, data.as_ptr(), data.len() as raw::size_t) } != 0 {
            return Err(self.error());
        }
        // SAFETY: the successful header call initialized these parameters.
        let (width, height, color, depth) = unsafe {
            (
                raw::tj3Get(self.0, raw::TJPARAM_TJPARAM_JPEGWIDTH as i32),
                raw::tj3Get(self.0, raw::TJPARAM_TJPARAM_JPEGHEIGHT as i32),
                raw::tj3Get(self.0, raw::TJPARAM_TJPARAM_COLORSPACE as i32),
                raw::tj3Get(self.0, raw::TJPARAM_TJPARAM_PRECISION as i32),
            )
        };
        let width = u32::try_from(width).map_err(|e| e.to_string())?;
        let height = u32::try_from(height).map_err(|e| e.to_string())?;
        let mode = match color as u32 {
            raw::TJCS_TJCS_GRAY => "L",
            raw::TJCS_TJCS_CMYK | raw::TJCS_TJCS_YCCK => "CMYK",
            raw::TJCS_TJCS_RGB | raw::TJCS_TJCS_YCbCr => "RGB",
            _ => return Err("unsupported JPEG colorspace".into()),
        };
        blanket_codec_common::ImageHeader::new(width, height, mode, depth.try_into().map_err(|_| "invalid JPEG precision")?)
    }

    pub(super) fn decode(&mut self, data: &[u8]) -> Result<Image, String> {
        let header = self.header(data)?;
        let (width, height) = (header.width, header.height);
        validate_dimensions(width, height)?;
        let (mode, format) = match header.mode {
            "L" => (PixelMode::L, raw::TJPF_TJPF_GRAY),
            "CMYK" => (PixelMode::Cmyk, raw::TJPF_TJPF_CMYK),
            _ => (PixelMode::Rgb, raw::TJPF_TJPF_RGB),
        };
        let pitch = (width as usize).checked_mul(mode.channels()).ok_or("JPEG row size overflow")?;
        let length = pitch.checked_mul(height as usize).ok_or("JPEG buffer size overflow")?;
        let pitch = i32::try_from(pitch).map_err(|e| e.to_string())?;
        let mut pixels = Vec::<u8>::new();
        pixels.try_reserve_exact(length).map_err(|e| e.to_string())?;
        // SAFETY: capacity covers every tightly packed output row for the validated
        // dimensions and format. TurboJPEG writes all samples on success; errors
        // leave the Vec empty so partially initialized storage is never read.
        if unsafe { raw::tj3Decompress8(self.0, data.as_ptr(), data.len() as raw::size_t, pixels.as_mut_ptr(), pitch, format) } != 0 {
            return Err(self.error());
        }
        // SAFETY: successful decompression initialized all length bytes above.
        unsafe { pixels.set_len(length) };
        if mode == PixelMode::Cmyk {
            pixels.iter_mut().for_each(|value| *value = 255 - *value);
        }
        Image::from_pixels(width, height, mode, pixels, Some("JPEG".into())).map_err(|e| e.to_string())
    }
}
