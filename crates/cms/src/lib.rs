use std::sync::Mutex;

use blanket_core::raster::{Image, PixelMode, alloc_pixels};
use lcms2::{
    CIExyY, CIExyYTRIPLE, Flags, InfoType, Intent, Locale, MLU, PixelFormat, Profile, Tag, TagSignature, ThreadContext, ToneCurve, Transform,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub fn littlecms_version() -> String {
    let version = lcms2::version();
    format!("{}.{}", version / 1_000, version % 1_000 / 10)
}

fn error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn intent(value: u32) -> PyResult<Intent> {
    match value {
        0 => Ok(Intent::Perceptual),
        1 => Ok(Intent::RelativeColorimetric),
        2 => Ok(Intent::Saturation),
        3 => Ok(Intent::AbsoluteColorimetric),
        _ => Err(error("rendering intent must be between 0 and 3")),
    }
}

fn pixel_format(mode: PixelMode) -> PyResult<PixelFormat> {
    match mode {
        PixelMode::L => Ok(PixelFormat::GRAY_8),
        PixelMode::Rgb => Ok(PixelFormat::RGB_8),
        PixelMode::Rgba => Ok(PixelFormat::RGBA_8),
        PixelMode::Cmyk => Ok(PixelFormat::CMYK_8),
        PixelMode::Lab => Ok(PixelFormat::Lab_8),
        _ => Err(error("unsupported ImageCms mode")),
    }
}

#[pyfunction]
fn _cms_create_profile<'py>(py: Python<'py>, space: &str, temperature: f64) -> PyResult<Bound<'py, PyBytes>> {
    let context = ThreadContext::new();
    let mut profile = match space {
        "sRGB" => Profile::new_srgb_context(&context),
        "LAB" => {
            let white = if temperature == 0.0 {
                *CIExyY::d50()
            } else {
                lcms2::white_point_from_temp(temperature).ok_or_else(|| error("invalid color temperature"))?
            };
            Profile::new_lab4_context(&context, &white).map_err(error)?
        }
        "XYZ" => Profile::new_xyz_context(&context),
        "Adobe RGB" | "Display P3" => {
            let p3 = space == "Display P3";
            let xy = |x, y| CIExyY { x, y, Y: 1.0 };
            let primaries = CIExyYTRIPLE {
                Red: xy(if p3 { 0.68 } else { 0.64 }, if p3 { 0.32 } else { 0.33 }),
                Green: xy(if p3 { 0.265 } else { 0.21 }, if p3 { 0.69 } else { 0.71 }),
                Blue: xy(0.15, 0.06),
            };
            let curve = if p3 {
                ToneCurve::new_parametric(4, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045]).map_err(error)?
            } else {
                ToneCurve::new(563.0 / 256.0)
            };
            Profile::new_rgb_context(&context, &xy(0.3127, 0.3290), &primaries, &[&curve, &curve, &curve]).map_err(error)?
        }
        _ => return Err(error("unsupported color space")),
    };
    if matches!(space, "Adobe RGB" | "Display P3") {
        let mut description = MLU::new(1);
        description.set_text(space, Locale::new("en_US"));
        profile.write_tag(TagSignature::ProfileDescriptionTag, Tag::MLU(&description));
    }
    Ok(PyBytes::new(py, &profile.icc().map_err(error)?))
}

#[pyfunction]
fn _cms_profile_info(data: &[u8]) -> PyResult<(String, String, String, String)> {
    let context = ThreadContext::new();
    let profile = Profile::new_icc_context(&context, data).map_err(error)?;
    let info = |kind| profile.info(kind, Locale::new("en_US")).unwrap_or_default();
    Ok((info(InfoType::Description), info(InfoType::Manufacturer), info(InfoType::Model), info(InfoType::Copyright)))
}

#[pyclass(name = "_CmsTransform", module = "blanket._blanket")]
struct CmsTransform {
    transform: Mutex<(Transform<u8, u8, ThreadContext>, ThreadContext)>,
    input: PixelMode,
    output: PixelMode,
}

#[pymethods]
impl CmsTransform {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: &[u8], output: &[u8], input_mode: &str, output_mode: &str, rendering_intent: u32, flags: u32, proof: Option<&[u8]>, proof_intent: u32,
    ) -> PyResult<Self> {
        let context = ThreadContext::new();
        let input_profile = Profile::new_icc_context(&context, input).map_err(error)?;
        let output_profile = Profile::new_icc_context(&context, output).map_err(error)?;
        let input = PixelMode::parse(input_mode)?;
        let output = PixelMode::parse(output_mode)?;
        let mut options = Flags::default();
        options.0 = flags;
        let transform = if let Some(proof) = proof {
            let proof = Profile::new_icc_context(&context, proof).map_err(error)?;
            Transform::new_proofing_context(
                &context,
                &input_profile,
                pixel_format(input)?,
                &output_profile,
                pixel_format(output)?,
                &proof,
                intent(rendering_intent)?,
                intent(proof_intent)?,
                options,
            )
        } else {
            Transform::new_flags_context(
                &context,
                &input_profile,
                pixel_format(input)?,
                &output_profile,
                pixel_format(output)?,
                intent(rendering_intent)?,
                options,
            )
        }
        .map_err(error)?;
        Ok(Self { transform: Mutex::new((transform, context)), input, output })
    }

    fn apply(&self, py: Python<'_>, image: &Image) -> PyResult<Image> {
        if image.mode != self.input {
            return Err(error("image mode does not match transform input mode"));
        }
        let source = image.pixel_data()?;
        let mut pixels = alloc_pixels(image.width as usize * image.height as usize * self.output.channels())?;
        py.detach(|| {
            self.transform.lock().map_err(|_| error("color transform lock poisoned"))?.0.transform_pixels(source, &mut pixels);
            Ok::<_, PyErr>(())
        })?;
        if self.output == PixelMode::Rgba {
            if self.input == PixelMode::Rgba {
                blanket_core::pixels::restore_alpha::<true>(&mut pixels, source);
            } else {
                blanket_core::pixels::restore_alpha::<false>(&mut pixels, source);
            }
        }
        Image::from_pixels(image.width, image.height, self.output, pixels, None)
    }

    fn apply_in_place(&self, py: Python<'_>, image: &mut Image) -> PyResult<()> {
        if self.input != self.output {
            return Err(error("in-place transforms require identical input and output modes"));
        }
        let result = self.apply(py, image)?;
        image.pixels = result.pixels;
        Ok(())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CmsTransform>()?;
    module.add_function(wrap_pyfunction!(_cms_create_profile, module)?)?;
    module.add_function(wrap_pyfunction!(_cms_profile_info, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_profile_transform() {
        Python::initialize();
        Python::attach(|py| {
            let context = ThreadContext::new();
            let gray = Profile::new_gray_context(&context, CIExyY::d50(), &ToneCurve::new(1.0)).unwrap().icc().unwrap();
            let rgb = Profile::new_srgb_context(&context).icc().unwrap();
            let transform = CmsTransform::new(&gray, &rgb, "L", "RGB", 1, 0, None, 3).unwrap();
            let image = Image::from_pixels(3, 1, PixelMode::L, vec![0, 128, 255], None).unwrap();
            let result = transform.apply(py, &image).unwrap();
            let pixels = result.pixel_data().unwrap();
            assert_eq!(&pixels[..3], &[0, 0, 0]);
            assert_eq!(&pixels[6..], &[255, 255, 255]);
            assert!(pixels[3..6].iter().all(|&v| (187..=189).contains(&v)));
        });
    }
}
