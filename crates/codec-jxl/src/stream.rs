use std::mem::MaybeUninit;
use std::ptr;

use blanket_codec_common::validate_dimensions;
use blanket_core::{Image, PixelMode};
use jpegxl_sys::common::types::{JxlBool, JxlDataType, JxlEndianness, JxlPixelFormat};
use jpegxl_sys::decode::*;
use jpegxl_sys::metadata::codestream_header::JxlBasicInfo;

pub struct JxlStreamDecoder {
    decoder: *mut JxlDecoder,
    input: Vec<u8> = Vec::new(),
    pixels: Vec<u8> = Vec::new(),
    format: JxlPixelFormat = JxlPixelFormat { num_channels: 3, data_type: JxlDataType::Uint8, endianness: JxlEndianness::Little, align: 0 },
    width: u32 = 0,
    height: u32 = 0,
    mode: PixelMode = PixelMode::Rgb,
    depth: u8 = 8,
    available: bool = false,
    failed: bool = false,
    pub complete: bool = false,
}

impl Drop for JxlStreamDecoder {
    fn drop(&mut self) {
        // Destroy the decoder before releasing its registered output buffer.
        unsafe { JxlDecoderDestroy(self.decoder) };
    }
}

fn check(status: JxlDecoderStatus) -> Result<(), String> {
    if status == JxlDecoderStatus::Success { Ok(()) } else { Err("JPEG XL streaming decoder error".into()) }
}

impl JxlStreamDecoder {
    pub fn new() -> Result<Self, String> {
        let decoder = unsafe { JxlDecoderCreate(ptr::null()) };
        if decoder.is_null() {
            return Err("cannot allocate JPEG XL decoder".into());
        }
        let stream = Self { decoder, .. };
        unsafe {
            check(JxlDecoderSubscribeEvents(
                decoder,
                JxlDecoderStatus::BasicInfo as i32 | JxlDecoderStatus::FullImage as i32 | JxlDecoderStatus::FrameProgression as i32,
            ))?;
            check(JxlDecoderSetProgressiveDetail(decoder, JxlProgressiveDetail::Passes))?;
        }
        Ok(stream)
    }

    pub fn feed(&mut self, data: &[u8]) -> Result<(), String> {
        if self.failed {
            return Err("failed JPEG XL decoder".into());
        }
        let result = self.process_input(data);
        self.failed = result.is_err();
        result
    }

    fn process_input(&mut self, data: &[u8]) -> Result<(), String> {
        if self.complete {
            return Ok(());
        }
        self.input.extend_from_slice(data);
        // libjxl retains input only until ReleaseInput; output remains allocated
        // until decoder destruction and is never resized after registration.
        unsafe {
            check(JxlDecoderSetInput(self.decoder, self.input.as_ptr(), self.input.len()))?;
            let result = self.process();
            let remaining = JxlDecoderReleaseInput(self.decoder);
            if self.complete {
                self.input = Vec::new();
            } else {
                self.input.drain(..self.input.len() - remaining);
            }
            result
        }
    }

    unsafe fn process(&mut self) -> Result<(), String> {
        loop {
            let status = unsafe { JxlDecoderProcessInput(self.decoder) };
            match status {
                JxlDecoderStatus::NeedMoreInput => return Ok(()),
                JxlDecoderStatus::Success => {
                    self.complete = true;
                    return Ok(());
                }
                JxlDecoderStatus::BasicInfo => unsafe {
                    let mut info = MaybeUninit::<JxlBasicInfo>::uninit();
                    check(JxlDecoderGetBasicInfo(self.decoder, info.as_mut_ptr()))?;
                    let info = info.assume_init();
                    validate_dimensions(info.xsize, info.ysize)?;
                    if info.have_animation == JxlBool::True {
                        return Err("streaming JPEG XL animation is not supported".into());
                    }
                    self.width = info.xsize;
                    self.height = info.ysize;
                    self.mode = match (info.num_color_channels, info.alpha_bits > 0) {
                        (1, false) => PixelMode::L,
                        (1 | 3, true) => PixelMode::Rgba,
                        (3, false) => PixelMode::Rgb,
                        _ => return Err("unsupported JPEG XL channel layout".into()),
                    };
                    self.depth = if info.bits_per_sample > 8 && info.exponent_bits_per_sample == 0 { 16 } else { 8 };
                    self.format.num_channels = self.mode.channels() as u32;
                    self.format.data_type = if self.depth == 16 { JxlDataType::Uint16 } else { JxlDataType::Uint8 };
                },
                JxlDecoderStatus::NeedImageOutBuffer => unsafe {
                    let mut size = 0;
                    check(JxlDecoderImageOutBufferSize(self.decoder, &raw const self.format, &raw mut size))?;
                    self.pixels = vec![0; size];
                    check(JxlDecoderSetImageOutBuffer(self.decoder, &raw const self.format, self.pixels.as_mut_ptr().cast(), size))?;
                },
                JxlDecoderStatus::FrameProgression => unsafe {
                    check(JxlDecoderFlushImage(self.decoder))?;
                    self.available = true;
                },
                JxlDecoderStatus::FullImage => self.available = true,
                _ => return Err("invalid JPEG XL stream".into()),
            }
        }
    }

    pub fn image(&self) -> Result<Option<Image>, String> {
        if !self.available {
            return Ok(None);
        }
        let image = if self.depth == 8 {
            Image::from_pixels(self.width, self.height, self.mode, self.pixels.clone(), Some("JXL".into()))
        } else {
            Image::from_sample_bytes(self.width, self.height, self.mode, self.pixels.clone(), self.depth, Some("JXL".into()))
        };
        image.map(Some).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use jpegxl_sys::color::color_encoding::JxlColorEncoding;
    use jpegxl_sys::encoder::encode::*;

    use super::*;

    #[test]
    fn progressive_passes_discard_input_and_match_eager_pixels() {
        struct Encoder(*mut JxlEncoder);
        impl Drop for Encoder {
            fn drop(&mut self) {
                unsafe { JxlEncoderDestroy(self.0) };
            }
        }

        let mut data = vec![0; 256 * 1024];
        // All encoder buffers outlive processing; the owner frees libjxl state.
        unsafe {
            let encoder = Encoder(JxlEncoderCreate(ptr::null()));
            assert!(!encoder.0.is_null());
            let mut info = MaybeUninit::<JxlBasicInfo>::uninit();
            JxlEncoderInitBasicInfo(info.as_mut_ptr());
            let mut info = info.assume_init();
            info.xsize = 257;
            info.ysize = 193;
            assert_eq!(JxlEncoderSetBasicInfo(encoder.0, &raw const info), JxlEncoderStatus::Success);
            let mut color = MaybeUninit::<JxlColorEncoding>::uninit();
            JxlColorEncodingSetToSRGB(color.as_mut_ptr(), JxlBool::False);
            assert_eq!(JxlEncoderSetColorEncoding(encoder.0, color.as_ptr()), JxlEncoderStatus::Success);
            let settings = JxlEncoderFrameSettingsCreate(encoder.0, ptr::null());
            assert!(!settings.is_null());
            for (option, value) in
                [(JxlEncoderFrameSettingId::Effort, 3), (JxlEncoderFrameSettingId::ProgressiveAc, 1), (JxlEncoderFrameSettingId::ProgressiveDc, 1)]
            {
                assert_eq!(JxlEncoderFrameSettingsSetOption(settings, option, value), JxlEncoderStatus::Success);
            }
            assert_eq!(JxlEncoderSetFrameDistance(settings, 1.0), JxlEncoderStatus::Success);
            let pixels: Vec<u8> = (0..257 * 193 * 3).map(|i| (i * 71 + i / 7) as u8).collect();
            let format = JxlPixelFormat { num_channels: 3, data_type: JxlDataType::Uint8, endianness: JxlEndianness::Native, align: 0 };
            assert_eq!(JxlEncoderAddImageFrame(settings, &raw const format, pixels.as_ptr().cast(), pixels.len()), JxlEncoderStatus::Success);
            JxlEncoderCloseInput(encoder.0);
            let mut next = data.as_mut_ptr();
            let mut available = data.len();
            assert_eq!(JxlEncoderProcessOutput(encoder.0, &raw mut next, &raw mut available), JxlEncoderStatus::Success);
            data.truncate(data.len() - available);
        }
        let mut decoder = JxlStreamDecoder::new().unwrap();
        let mut preview = None;
        for chunk in data.chunks(17) {
            decoder.feed(chunk).unwrap();
            assert!(decoder.input.len() < 32);
            if decoder.available && !decoder.complete && preview.is_none() {
                preview = decoder.image().unwrap();
            }
        }
        assert!(decoder.complete);
        assert!(preview.is_some());
        let result = decoder.image().unwrap().unwrap();
        assert_eq!(result.pixel_data().unwrap(), crate::decode_jxl(&data).unwrap().pixel_data().unwrap());
    }

    #[test]
    fn failed_decoder_cannot_resume() {
        let mut decoder = JxlStreamDecoder::new().unwrap();
        assert!(decoder.feed(b"invalid JPEG XL stream").is_err());
        assert!(decoder.feed(&[]).unwrap_err().contains("failed JPEG XL decoder"));
    }
}
