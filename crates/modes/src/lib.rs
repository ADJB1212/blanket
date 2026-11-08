use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelMode {
    One,
    L,
    I,
    F,
    RgbF,
    RgbaF,
    I16,
    I16L,
    I16B,
    I16N,
    La,
    PremulLa,
    Pa,
    Rgb,
    Rgba,
    PremulRgba,
    Rgbx,
    Hsv,
    Cmyk,
    YCbCr,
    Lab,
}

impl PixelMode {
    pub fn parse(value: &str) -> PyResult<Self> {
        match value {
            "1" => Ok(Self::One),
            "L" => Ok(Self::L),
            "I" => Ok(Self::I),
            "F" => Ok(Self::F),
            "RGB;F" => Ok(Self::RgbF),
            "RGBA;F" => Ok(Self::RgbaF),
            "I;16" => Ok(Self::I16),
            "I;16L" => Ok(Self::I16L),
            "I;16B" => Ok(Self::I16B),
            "I;16N" => Ok(Self::I16N),
            "LA" => Ok(Self::La),
            "La" => Ok(Self::PremulLa),
            "PA" => Ok(Self::Pa),
            "RGB" => Ok(Self::Rgb),
            "RGBA" => Ok(Self::Rgba),
            "RGBa" => Ok(Self::PremulRgba),
            "RGBX" => Ok(Self::Rgbx),
            "HSV" => Ok(Self::Hsv),
            "CMYK" => Ok(Self::Cmyk),
            "YCbCr" => Ok(Self::YCbCr),
            "LAB" => Ok(Self::Lab),
            _ => Err(PyValueError::new_err(format!("unsupported image mode {value:?}"))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::L => "L",
            Self::I => "I",
            Self::F => "F",
            Self::RgbF => "RGB;F",
            Self::RgbaF => "RGBA;F",
            Self::I16 => "I;16",
            Self::I16L => "I;16L",
            Self::I16B => "I;16B",
            Self::I16N => "I;16N",
            Self::La => "LA",
            Self::PremulLa => "La",
            Self::Pa => "PA",
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
            Self::PremulRgba => "RGBa",
            Self::Rgbx => "RGBX",
            Self::Hsv => "HSV",
            Self::Cmyk => "CMYK",
            Self::YCbCr => "YCbCr",
            Self::Lab => "LAB",
        }
    }

    pub const fn channels(self) -> usize {
        match self {
            Self::One | Self::L | Self::I | Self::F | Self::I16 | Self::I16L | Self::I16B | Self::I16N => 1,
            Self::La | Self::PremulLa | Self::Pa => 2,
            Self::Rgb | Self::RgbF | Self::Hsv | Self::YCbCr | Self::Lab => 3,
            Self::Rgba | Self::PremulRgba | Self::Rgbx | Self::RgbaF | Self::Cmyk => 4,
        }
    }

    pub const fn sample_bytes(self) -> usize {
        match self {
            Self::I | Self::F | Self::RgbF | Self::RgbaF => 4,
            Self::I16 | Self::I16L | Self::I16B | Self::I16N => 2,
            _ => 1,
        }
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, Self::I | Self::I16 | Self::I16L | Self::I16B | Self::I16N)
    }

    pub const fn is_big_endian(self) -> bool {
        matches!(self, Self::I16B) || matches!(self, Self::I16N) && cfg!(target_endian = "big")
    }

    pub const fn is_wide_scalar(self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F | Self::RgbF | Self::RgbaF)
    }
}
