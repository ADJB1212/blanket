use std::borrow::Cow;
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

fn mode_depth(mode: PixelMode, source_depth: u8) -> u8 {
    if mode.is_float() {
        32
    } else if mode.is_integer() {
        16
    } else if source_depth == 32 {
        8
    } else {
        source_depth
    }
}

fn pixel_format(mode: PixelMode, depth: u8) -> PyResult<PixelFormat> {
    let format = match (mode, depth) {
        (PixelMode::L, 8) => PixelFormat::GRAY_8,
        (PixelMode::Rgb, 8) => PixelFormat::RGB_8,
        (PixelMode::Rgba, 8) => PixelFormat::RGBA_8,
        (PixelMode::Cmyk, 8) => PixelFormat::CMYK_8,
        (PixelMode::Lab, 8) => PixelFormat::Lab_8,
        (PixelMode::L | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::I16N, 16) => PixelFormat::GRAY_16,
        (PixelMode::Rgb, 16) => PixelFormat::RGB_16,
        (PixelMode::Rgba, 16) => PixelFormat::RGBA_16,
        (PixelMode::F, 32) => PixelFormat::GRAY_FLT,
        (PixelMode::RgbF, 32) => PixelFormat::RGB_FLT,
        (PixelMode::RgbaF, 32) => PixelFormat::RGBA_FLT,
        _ => return Err(error("unsupported ImageCms mode or sample depth")),
    };
    Ok(format)
}

fn swap_samples(pixels: &mut [u8], depth: u8) {
    for sample in pixels.chunks_exact_mut(usize::from(depth / 8)) {
        sample.reverse();
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
    state: Mutex<TransformState>,
    input: PixelMode,
    output: PixelMode,
}

struct TransformState {
    transforms: Vec<(u8, Transform<u8, u8, ThreadContext>)>,
    input_profile: Profile<ThreadContext>,
    output_profile: Profile<ThreadContext>,
    proof_profile: Option<Profile<ThreadContext>>,
    rendering_intent: Intent,
    proof_intent: Intent,
    flags: u32,
    context: ThreadContext,
}

impl TransformState {
    fn build(&self, input: PixelMode, output: PixelMode, depth: u8) -> PyResult<Transform<u8, u8, ThreadContext>> {
        let mut options = Flags::default();
        // Alpha is restored separately, including when channel counts differ.
        options.0 = self.flags & !Flags::COPY_ALPHA.0;
        let input_format = pixel_format(input, depth)?;
        let output_format = pixel_format(output, mode_depth(output, depth))?;
        if let Some(proof) = &self.proof_profile {
            Transform::new_proofing_context(
                &self.context,
                &self.input_profile,
                input_format,
                &self.output_profile,
                output_format,
                proof,
                self.rendering_intent,
                self.proof_intent,
                options,
            )
        } else {
            Transform::new_flags_context(
                &self.context,
                &self.input_profile,
                input_format,
                &self.output_profile,
                output_format,
                self.rendering_intent,
                options,
            )
        }
        .map_err(error)
    }
}

fn restore_alpha(pixels: &mut [u8], source: &[u8], input: PixelMode, depth: u8, output_depth: u8) {
    let bytes = usize::from(depth / 8);
    let output_bytes = usize::from(output_depth / 8);
    let opaque = if output_depth == 32 { 1.0_f32.to_le_bytes() } else { [255; 4] };
    for (output, input_pixel) in pixels.chunks_exact_mut(4 * output_bytes).zip(source.chunks_exact(input.channels() * bytes)) {
        let alpha = &mut output[3 * output_bytes..];
        if matches!(input, PixelMode::Rgba | PixelMode::RgbaF) {
            let original = &input_pixel[3 * bytes..];
            if depth == output_depth {
                alpha.copy_from_slice(original);
            } else {
                let value = match depth {
                    32 => f64::from(f32::from_le_bytes(original.try_into().unwrap())),
                    16 => f64::from(u16::from_le_bytes(original.try_into().unwrap())) / 65535.0,
                    _ => f64::from(original[0]) / 255.0,
                };
                match output_depth {
                    32 => alpha.copy_from_slice(&(value as f32).to_le_bytes()),
                    16 => alpha.copy_from_slice(&((value * 65535.0).round() as u16).to_le_bytes()),
                    _ => alpha[0] = (value * 255.0).round() as u8,
                }
            }
        } else {
            alpha.copy_from_slice(&opaque[..output_bytes]);
        }
    }
}

#[pymethods]
impl CmsTransform {
    #[new]
    fn new(
        py: Python<'_>, input: &[u8], output: &[u8], input_mode: &str, output_mode: &str, rendering_intent: u32, flags: u32, proof: Option<&[u8]>,
        proof_intent: u32,
    ) -> PyResult<Self> {
        py.detach(|| {
            let context = ThreadContext::new();
            let input_profile = Profile::new_icc_context(&context, input).map_err(|e| error(format!("invalid input ICC profile: {e}")))?;
            let output_profile = Profile::new_icc_context(&context, output).map_err(|e| error(format!("invalid output ICC profile: {e}")))?;
            let input = PixelMode::parse(input_mode)?;
            let output = PixelMode::parse(output_mode)?;
            let proof_profile = proof
                .map(|data| Profile::new_icc_context(&context, data).map_err(|e| error(format!("invalid proof ICC profile: {e}"))))
                .transpose()?;
            let mut state = TransformState {
                transforms: Vec::new(),
                input_profile,
                output_profile,
                proof_profile,
                rendering_intent: intent(rendering_intent)?,
                proof_intent: intent(proof_intent)?,
                flags,
                context,
            };
            let depth = mode_depth(input, 8);
            state.transforms.push((depth, state.build(input, output, depth)?));
            Ok(Self { state: Mutex::new(state), input, output })
        })
    }

    fn apply(&self, py: Python<'_>, image: &Image) -> PyResult<Image> {
        if image.pixel_mode() != self.input {
            return Err(error("image mode does not match transform input mode"));
        }
        let depth = image.bit_depth();
        pixel_format(self.input, depth)?;
        let output_depth = mode_depth(self.output, depth);
        pixel_format(self.output, output_depth)?;
        let source = image.raw_data()?;
        let count = source.len() / image.raw_bytes_per_pixel();
        let len = count.checked_mul(self.output.channels() * usize::from(output_depth / 8)).ok_or_else(|| error("image dimensions are too large"))?;
        let mut pixels = alloc_pixels(len)?;
        py.detach(|| {
            // LittleCMS float pipelines do not support swapped 16-bit formatters.
            let source = if depth > 8 && self.input.is_big_endian() != cfg!(target_endian = "big") {
                let mut native = alloc_pixels(source.len())?;
                native.copy_from_slice(source);
                swap_samples(&mut native, depth);
                Cow::Owned(native)
            } else {
                Cow::Borrowed(source)
            };
            let mut state = self.state.lock().map_err(|_| error("color transform lock poisoned"))?;
            if !state.transforms.iter().any(|(cached_depth, _)| *cached_depth == depth) {
                let transform = state.build(self.input, self.output, depth)?;
                state.transforms.push((depth, transform));
            }
            let transform = &state.transforms.iter().find(|(cached_depth, _)| *cached_depth == depth).unwrap().1;
            let input_stride = image.raw_bytes_per_pixel();
            let output_stride = self.output.channels() * usize::from(output_depth / 8);
            for (src, dst) in source.chunks(input_stride * 65536).zip(pixels.chunks_mut(output_stride * 65536)) {
                transform.transform_pixels(src, dst);
            }
            drop(state);
            if output_depth > 8 && self.output.is_big_endian() != cfg!(target_endian = "big") {
                swap_samples(&mut pixels, output_depth);
            }
            if matches!(self.output, PixelMode::Rgba | PixelMode::RgbaF) {
                restore_alpha(&mut pixels, image.raw_data()?, self.input, depth, output_depth);
            }
            Ok::<_, PyErr>(())
        })?;
        Image::from_raw_bytes(image.width(), image.height(), self.output, pixels, output_depth)
    }

    fn apply_in_place(&self, py: Python<'_>, image: &mut Image) -> PyResult<()> {
        if self.input != self.output {
            return Err(error("in-place transforms require identical input and output modes"));
        }
        let result = self.apply(py, image)?;
        image.replace_storage(result)
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
    fn invalid_transform_profiles_identify_the_argument() {
        Python::initialize();
        Python::attach(|py| {
            let context = ThreadContext::new();
            let valid = Profile::new_srgb_context(&context).icc().unwrap();
            let invalid = b"invalid ICC".as_slice();
            for (input, output, proof, role) in [
                (invalid, valid.as_slice(), None, "input"),
                (valid.as_slice(), invalid, None, "output"),
                (valid.as_slice(), valid.as_slice(), Some(invalid), "proof"),
            ] {
                let err = CmsTransform::new(py, input, output, "RGB", "RGB", 0, 0, proof, 0).err().expect("invalid profile must fail");
                assert!(err.is_instance_of::<PyValueError>(py));
                let message = err.value(py).to_string();
                assert!(message.starts_with(&format!("invalid {role} ICC profile: ")), "{message}");
                assert!(message.contains("Could not create the object"), "{message}");
            }
        });
    }

    #[test]
    fn grayscale_profile_transform() {
        Python::initialize();
        Python::attach(|py| {
            let context = ThreadContext::new();
            let gray = Profile::new_gray_context(&context, CIExyY::d50(), &ToneCurve::new(1.0)).unwrap().icc().unwrap();
            let rgb = Profile::new_srgb_context(&context).icc().unwrap();
            let transform = CmsTransform::new(py, &gray, &rgb, "L", "RGB", 1, 0, None, 3).unwrap();
            let image = Image::from_pixels(3, 1, PixelMode::L, vec![0, 128, 255], None).unwrap();
            let result = transform.apply(py, &image).unwrap();
            let pixels = result.pixel_data().unwrap();
            assert_eq!(&pixels[..3], &[0, 0, 0]);
            assert_eq!(&pixels[6..], &[255, 255, 255]);
            assert!(pixels[3..6].iter().all(|&v| (187..=189).contains(&v)));
        });
    }

    #[test]
    fn wide_grayscale_modes_and_byte_order() {
        Python::initialize();
        Python::attach(|py| {
            let context = ThreadContext::new();
            let gray = Profile::new_gray_context(&context, CIExyY::d50(), &ToneCurve::new(1.0)).unwrap().icc().unwrap();
            let samples = [0_u16, 1, 12345, 32768, 65534, 65535];
            for mode in [PixelMode::L, PixelMode::I16, PixelMode::I16L, PixelMode::I16B, PixelMode::I16N] {
                let raw = samples.iter().flat_map(|v| if mode.is_big_endian() { v.to_be_bytes() } else { v.to_le_bytes() }).collect();
                let image = Image::from_raw_bytes(6, 1, mode, raw, 16).unwrap();
                let transform =
                    CmsTransform::new(py, &gray, &gray, mode.as_str(), "F", 1, 0, None, 3).unwrap_or_else(|err| panic!("{mode:?}: {err}"));
                let result = transform.apply(py, &image).unwrap();
                for (bytes, sample) in result.raw_data().unwrap().as_chunks::<4>().0.iter().zip(samples) {
                    let value = f32::from_le_bytes(*bytes);
                    assert!((value - f32::from(sample) / 65535.0).abs() < 0.00001);
                }
                let output_mode = if mode == PixelMode::L { "I;16L" } else { mode.as_str() };
                let inverse = CmsTransform::new(py, &gray, &gray, "F", output_mode, 1, 0, None, 3).unwrap();
                assert_eq!(inverse.apply(py, &result).unwrap().raw_data().unwrap(), image.raw_data().unwrap());
            }
        });
    }

    #[test]
    fn float_pq_profile_uses_hdr_curve_without_eight_bit_quantization() {
        Python::initialize();
        Python::attach(|py| {
            let context = ThreadContext::new();
            let pq = |value: f64| {
                let power = value.powf(32.0 / 2523.0);
                ((power - 3424.0 / 4096.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * power)).powf(16384.0 / 2610.0)
            };
            let table: Vec<u16> = (0..=4096).map(|i| (pq(f64::from(i) / 4096.0) * 65535.0).round() as u16).collect();
            let input = Profile::new_gray_context(&context, CIExyY::d50(), &ToneCurve::new_tabulated(&table)).unwrap().icc().unwrap();
            let output = Profile::new_gray_context(&context, CIExyY::d50(), &ToneCurve::new(1.0)).unwrap().icc().unwrap();
            let values = [0.1_f32, 0.5, 0.501, 0.75, 0.9, 1.0];
            let image = Image::from_float_bytes(6, 1, values.iter().flat_map(|v| v.to_le_bytes()).collect()).unwrap();
            let transform = CmsTransform::new(py, &input, &output, "F", "F", 1, 0, None, 3).unwrap();
            let result = transform.apply(py, &image).unwrap();
            let actual: Vec<f32> = result.raw_data().unwrap().as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect();
            for (&value, expected) in actual.iter().zip(values.map(|v| pq(f64::from(v)))) {
                assert!((f64::from(value) - expected).abs() < 0.00006, "{value} != {expected}");
            }
            assert_ne!(actual[1], actual[2]);
        });
    }
}
