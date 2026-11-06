mod safety;

use blanket_codec_common::{CodecErrorKind, ImageHeader};
use blanket_core::{Image, PixelMode};
pub use safety::{Document, Error, Limits, RenderOptions, detect, parse};

impl Error {
    pub fn kind(&self) -> CodecErrorKind {
        match self {
            Self::Corrupt(_) => CodecErrorKind::Corrupt,
            Self::Unsupported(_) => CodecErrorKind::Unsupported,
            Self::Limit(_) => CodecErrorKind::ResourceLimit,
        }
    }
}

pub fn decode(data: &[u8]) -> Result<Image, Error> {
    decode_options(data, RenderOptions::default())
}

pub fn decode_options(data: &[u8], options: RenderOptions) -> Result<Image, Error> {
    let document = safety::parse_options(data, Limits::default(), options, true)?;
    let (width, height) = document.size();
    Image::from_pixels(width, height, PixelMode::Rgba, document.render()?, Some("SVG".into())).map_err(|_| Error::Corrupt("invalid rendered image"))
}

pub fn inspect(data: &[u8]) -> Result<ImageHeader, Error> {
    let document = safety::parse_options(data, Limits::default(), RenderOptions::default(), false)?;
    let (width, height) = document.size();
    ImageHeader::new(width, height, "RGBA", 8).map_err(|_| Error::Corrupt("invalid SVG header"))
}
