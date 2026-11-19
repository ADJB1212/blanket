mod safety;
mod trace;
mod writer;

use blanket_codec_common::{CodecErrorKind, ImageHeader};
use blanket_core::{Image, PixelMode};
pub use safety::{Document, Error, Limits, RenderOptions, detect, parse};
pub use trace::encode;

impl Error {
    #[must_use]
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
    prepare(data, options)?.into_image()
}

/// Parse an SVG and resolve its raster dimensions before allocating the output.
pub fn prepare(data: &[u8], options: RenderOptions) -> Result<Document, Error> {
    safety::parse_options(data, Limits::default(), options, safety::SampleValidation::Decode)
}

impl Document {
    /// Render this document as a straight-alpha RGBA image.
    pub fn into_image(self) -> Result<Image, Error> {
        let (width, height) = self.size();
        Image::from_pixels(width, height, PixelMode::Rgba, self.render()?, Some("SVG".into())).map_err(|_| Error::Corrupt("invalid rendered image"))
    }
}

pub fn inspect(data: &[u8]) -> Result<ImageHeader, Error> {
    let document = safety::parse_options(data, Limits::default(), RenderOptions::default(), safety::SampleValidation::Header)?;
    let (width, height) = document.size();
    ImageHeader::new(width, height, "RGBA", 8).map_err(|_| Error::Corrupt("invalid SVG header"))
}
