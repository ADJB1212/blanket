use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

enum Stream<'a> {
    Bytes(&'a [u8]),
    Rgb(&'a [[u8; 4]]),
    Alpha(&'a [[u8; 4]]),
}

impl Stream<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Rgb(pixels) => pixels.len() * 3,
            Self::Alpha(pixels) => pixels.len(),
        }
    }

    fn append_to(self, output: &mut Vec<u8>) {
        match self {
            Self::Bytes(bytes) => output.extend_from_slice(bytes),
            Self::Rgb(pixels) => append_rgba_channels::<3>(output, pixels, 0),
            Self::Alpha(pixels) => append_rgba_channels::<1>(output, pixels, 3),
        }
    }
}

fn append_rgba_channels<const C: usize>(output: &mut Vec<u8>, pixels: &[[u8; 4]], first: usize) {
    output.reserve(pixels.len() * C);
    for pixel in pixels {
        output.extend_from_slice(&pixel[first..first + C]);
    }
}

// PDF 1.4 image XObjects with an optional grayscale soft mask. Raw streams
// retain exact samples without adding a runtime dependency or a lossy codec.
pub fn encode_pdf(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    encode_pages(&[(image, pixels)])
}

pub fn encode_pages(pages: &[(&Image, &[u8])]) -> PyResult<Vec<u8>> {
    if pages.is_empty() || pages.iter().any(|(image, _)| image.width == 0 || image.height == 0) {
        return Err(PyValueError::new_err("cannot encode empty PDF image"));
    }
    let mut output = Vec::new();
    // Color and alpha streams fit in pixels.len(); leave room for PDF objects.
    output
        .try_reserve_exact(
            pages
                .iter()
                .try_fold(2048_usize, |n, (_, pixels)| n.checked_add(pixels.len()).and_then(|v| v.checked_add(2048)))
                .ok_or_else(|| PyMemoryError::new_err("PDF is too large"))?,
        )
        .map_err(|_| PyMemoryError::new_err("cannot allocate PDF"))?;
    output.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");
    let mut offsets = Vec::with_capacity(6);
    let mut object = |dictionary: &str, stream: Option<Stream<'_>>| {
        offsets.push(output.len());
        output.extend_from_slice(format!("{} 0 obj\n<< {dictionary}", offsets.len()).as_bytes());
        if let Some(data) = stream {
            output.extend_from_slice(format!(" /Length {} >>\nstream\n", data.len()).as_bytes());
            data.append_to(&mut output);
            output.extend_from_slice(b"\nendstream\nendobj\n");
        } else {
            output.extend_from_slice(b" >>\nendobj\n");
        }
    };
    object("/Type /Catalog /Pages 2 0 R", None);
    let mut page_id = 3;
    let mut kids = String::new();
    for (image, _) in pages {
        use std::fmt::Write;
        write!(kids, "{page_id} 0 R ").unwrap();
        page_id += if image.mode == PixelMode::Rgba { 4 } else { 3 };
    }
    object(&format!("/Type /Pages /Kids [{kids}] /Count {}", pages.len()), None);
    page_id = 3;
    for &(image, pixels) in pages {
        let (width, height) = (image.width, image.height);
        let content_id = page_id + 1;
        let image_id = page_id + 2;
        let mask_id = page_id + 3;
        object(
            &format!(
                "/Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R"
            ),
            None,
        );
        let content = format!("q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n");
        object("", Some(Stream::Bytes(content.as_bytes())));
        let color = match image.mode {
            PixelMode::L => "DeviceGray",
            PixelMode::Cmyk => "DeviceCMYK",
            _ => "DeviceRGB",
        };
        let (samples, mask) = if image.mode == PixelMode::Rgba {
            (Stream::Rgb(pixels.as_chunks::<4>().0), format!(" /SMask {mask_id} 0 R"))
        } else {
            (Stream::Bytes(pixels), String::new())
        };
        object(
            &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /{color} /BitsPerComponent 8{mask}"),
            Some(samples),
        );
        if image.mode == PixelMode::Rgba {
            object(
                &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8"),
                Some(Stream::Alpha(pixels.as_chunks::<4>().0)),
            );
        }
        page_id += if image.mode == PixelMode::Rgba { 4 } else { 3 };
    }
    let xref = output.len();
    let size = offsets.len() + 1;
    output.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for offset in offsets {
        output.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    output.extend_from_slice(format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_streams_preserve_prefixes_and_pixel_tails() {
        for count in [0, 1, 15, 16, 17, 4097] {
            let source: Vec<u8> = (0..count * 4 + 3).map(|i| (i * 73 + i / 7) as u8).collect();
            let pixels = source[3..].as_chunks::<4>().0;
            for prefix in [0, 1, 17] {
                let mut rgb = vec![93; prefix];
                Stream::Rgb(pixels).append_to(&mut rgb);
                let mut expected = vec![93; prefix];
                expected.extend(source[3..].iter().enumerate().filter_map(|(i, &value)| (i % 4 != 3).then_some(value)));
                assert_eq!(rgb, expected);
                Stream::Alpha(pixels).append_to(&mut rgb);
                expected.extend(source[3..].iter().skip(3).step_by(4));
                assert_eq!(rgb, expected);
            }
        }
    }
}
