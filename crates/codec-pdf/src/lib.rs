use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// PDF 1.4 image XObjects with an optional grayscale soft mask. Raw streams
// retain exact samples without adding a runtime dependency or a lossy codec.
pub fn encode_pdf(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode empty PDF image"));
    }
    let (width, height) = (image.width, image.height);
    let mut output = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::new();
    let mut object = |dictionary: &str, stream: Option<&[u8]>| {
        offsets.push(output.len());
        output.extend_from_slice(format!("{} 0 obj\n<< {dictionary}", offsets.len()).as_bytes());
        if let Some(data) = stream {
            output.extend_from_slice(format!(" /Length {} >>\nstream\n", data.len()).as_bytes());
            output.extend_from_slice(data);
            output.extend_from_slice(b"\nendstream\nendobj\n");
        } else {
            output.extend_from_slice(b" >>\nendobj\n");
        }
    };
    object("/Type /Catalog /Pages 2 0 R", None);
    object("/Type /Pages /Kids [3 0 R] /Count 1", None);
    object(
        &format!("/Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R"),
        None,
    );
    let content = format!("q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n");
    object("", Some(content.as_bytes()));
    let rgb;
    let color = match image.mode {
        PixelMode::L => "DeviceGray",
        PixelMode::Cmyk => "DeviceCMYK",
        _ => "DeviceRGB",
    };
    let (samples, mask) = if image.mode == PixelMode::Rgba {
        rgb = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| pixel[..3].iter().copied())
            .collect::<Vec<_>>();
        (rgb.as_slice(), " /SMask 6 0 R")
    } else {
        (pixels, "")
    };
    object(
        &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /{color} /BitsPerComponent 8{mask}"),
        Some(samples),
    );
    if image.mode == PixelMode::Rgba {
        let alpha: Vec<u8> = pixels.as_chunks::<4>().0.iter().map(|pixel| pixel[3]).collect();
        object(
            &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8"),
            Some(&alpha),
        );
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
