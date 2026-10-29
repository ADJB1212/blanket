use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn _gpu_unsigned_bytes<'py>(py: Python<'py>, data: &[u8], depth: u8, encode: bool) -> PyResult<Bound<'py, PyBytes>> {
    if !matches!(depth, 10 | 12 | 16) || !data.len().is_multiple_of(if encode { 4 } else { 2 }) {
        return Err(PyValueError::new_err("invalid unsigned GPU transfer"));
    }
    let maximum = f32::from(u16::MAX >> (16 - depth));
    let bytes = if encode {
        data.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|chunk| {
                let value = f32::from_le_bytes(*chunk);
                let value = if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) };
                ((value * maximum).round() as u16).to_le_bytes()
            })
            .collect::<Vec<_>>()
    } else {
        data.as_chunks::<2>()
            .0
            .iter()
            .flat_map(|chunk| {
                let value = u16::from_le_bytes(*chunk);
                (f32::from(value).min(maximum) / maximum).to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn _gpu_weights(input: u32, output: u32, method: u8) -> PyResult<(u32, Vec<f32>)> {
    blanket_gpu::resize_weights(input, output, method).map_err(PyValueError::new_err)
}

#[pyfunction]
fn _gpu_blur_parameters(radius: f32, gaussian: bool) -> PyResult<(u32, [f32; 2])> {
    blanket_gpu::blur_parameters(radius, gaussian).map_err(PyValueError::new_err)
}

#[pyfunction]
fn _gpu_lut_controls(size: [u32; 3], channels: u32, table: Vec<f32>, floating: bool) -> PyResult<Vec<f32>> {
    blanket_gpu::lut_controls(size, channels, &table, floating).map_err(PyValueError::new_err)
}

#[cfg(target_os = "macos")]
mod native {
    use std::sync::OnceLock;

    use blanket_gpu::protocol::{
        BINARY_CONTROL_LEN, BINARY_SUBTRACT, COLOR_ALPHA_CHANNELS, CONVOLUTION_HEADER_LEN, GRAY_ALPHA_CHANNELS, HORIZONTAL, LARGE_KERNEL_SIZE,
        LUT_ENTRIES, MAX_RANK_SIZE, PARAMETER_COUNT, SMALL_KERNEL_SIZE, VERTICAL, enhancement, format, lut, operation, parameter, reduction,
        transpose, weights,
    };
    use blanket_gpu::{Backend, Descriptor, Dispatch};
    use blanket_gpu_metal::{Context, Image};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyBytes;

    use super::*;

    static CONTEXT: OnceLock<Result<Context, String>> = OnceLock::new();
    fn context() -> PyResult<&'static Context> {
        CONTEXT.get_or_init(Context::new).as_ref().map_err(|error| PyRuntimeError::new_err(error.clone()))
    }
    #[pyclass(unsendable)]
    pub(super) struct MetalImage {
        context: &'static Context,
        image: Image,
    }
    #[pymethods]
    impl MetalImage {
        #[new]
        fn new(width: u32, height: u32, channels: u32, float: bool, data: &[u8]) -> PyResult<Self> {
            let context = context()?;
            let image = context.upload(Descriptor { width, height, channels, float }, data).map_err(PyValueError::new_err)?;
            Ok(Self { context, image })
        }
        fn run(&self, other: &Self, parameters: [u32; PARAMETER_COUNT], values: Vec<f32>) -> PyResult<Self> {
            let d = self.image.descriptor;
            let mut output = Descriptor { width: parameters[parameter::OUTPUT_WIDTH], height: parameters[parameter::OUTPUT_HEIGHT], ..d };
            let source_shape = [parameters[parameter::SOURCE_WIDTH], parameters[parameter::SOURCE_HEIGHT], parameters[parameter::CHANNELS]];
            if source_shape != [d.width, d.height, d.channels] || !std::ptr::eq(self.context, other.context) {
                return Err(PyValueError::new_err("incompatible GPU images"));
            }
            let op = parameters[parameter::OPERATION];
            let k = parameters[parameter::KERNEL_OR_X] as usize;
            if matches!(op, operation::CONVERT | operation::COLOR_LUT) {
                let (channels, float) =
                    format::layout(parameters[if op == operation::CONVERT { parameter::KERNEL_OR_X } else { parameter::AXIS_OR_Y }])
                        .ok_or_else(|| PyValueError::new_err("invalid GPU output format"))?;
                output.channels = channels;
                output.float = float;
            } else if op == operation::CHANNEL {
                output.channels = 1;
            } else if op == operation::LUMINANCE_REDUCE {
                output.channels = COLOR_ALPHA_CHANNELS;
                output.float = true;
            } else if op == operation::STATISTICS {
                output.channels = 1;
                output.float = true;
            }
            let same_size = output.width == d.width && output.height == d.height;
            let valid = match op {
                operation::RESIZE => {
                    let axis = parameters[parameter::AXIS_OR_Y];
                    let count = if axis == HORIZONTAL { output.width } else { output.height };
                    let input = if axis == HORIZONTAL { d.width } else { d.height };
                    axis <= VERTICAL
                        && k > weights::HEADER_LEN
                        && values.len() == count as usize * k
                        && (if axis == HORIZONTAL { output.height == d.height } else { output.width == d.width })
                        && values.chunks_exact(k).all(|row| {
                            row[weights::START] >= 0.0
                                && row[weights::COUNT] >= 1.0
                                && row[weights::COUNT] as usize <= k - weights::HEADER_LEN
                                && row[weights::START] + row[weights::COUNT] <= input as f32
                        })
                }
                operation::CONVOLVE => {
                    matches!(k, SMALL_KERNEL_SIZE | LARGE_KERNEL_SIZE) && values.len() == k * k + CONVOLUTION_HEADER_LEN && output == d
                }
                operation::BLEND => values.len() == 1 && output == d && other.image.descriptor == d,
                operation::ALPHA_COMPOSITE => {
                    matches!(d.channels, GRAY_ALPHA_CHANNELS | COLOR_ALPHA_CHANNELS) && output == d && other.image.descriptor == d
                }
                operation::PREMULTIPLY | operation::UNPREMULTIPLY => matches!(d.channels, GRAY_ALPHA_CHANNELS | COLOR_ALPHA_CHANNELS) && output == d,
                operation::COPY => output == d,
                operation::CROP => true,
                operation::TRANSPOSE => {
                    k <= transpose::TRANSVERSE
                        && if matches!(k, transpose::ROTATE_90 | transpose::ROTATE_270 | transpose::TRANSPOSE | transpose::TRANSVERSE) {
                            output.width == d.height && output.height == d.width
                        } else {
                            output == d
                        }
                }
                operation::OFFSET => output == d && k < d.width as usize && parameters[parameter::AXIS_OR_Y] < d.height,
                operation::BRIGHTNESS => output == d && values.len() == 1 && values[0].is_finite(),
                operation::INVERT => output == d,
                operation::POINT => {
                    output == d
                        && !d.float
                        && values.len() == LUT_ENTRIES * d.channels as usize
                        && values.iter().all(|v| v.is_finite() && (0.0..=f32::from(u8::MAX)).contains(v))
                }
                operation::BINARY => {
                    output == d
                        && other.image.descriptor == d
                        && k <= BINARY_SUBTRACT
                        && values.len() == BINARY_CONTROL_LEN
                        && values.iter().all(|v| v.is_finite())
                        && values[0] != 0.0
                }
                operation::CONVERT => same_size,
                operation::QUANTIZE => output == d && d.float && values.len() == 1 && matches!(values[0], 1023.0 | 4095.0 | 65535.0),
                operation::STATISTICS => {
                    k <= 1
                        && output.height == d.channels
                        && output.width
                            == if k == 0 { blanket_gpu::protocol::statistics::FIELDS } else { blanket_gpu::protocol::statistics::HISTOGRAM_BINS }
                        && (k == 0 || !d.float)
                }
                operation::MASK => {
                    output == d
                        && d.float
                        && values.len() == 1
                        && matches!(values[0], 1.0 | 255.0)
                        && other.image.descriptor == Descriptor { channels: 1, ..d }
                }
                operation::WARP => {
                    k <= 2 && values.len() == blanket_gpu::protocol::WARP_COEFFICIENTS + d.channels as usize && values.iter().all(|v| v.is_finite())
                }
                operation::REMAP => {
                    k <= 2
                        && values.len() == blanket_gpu::protocol::WARP_COEFFICIENTS + d.channels as usize
                        && values.iter().all(|v| v.is_finite())
                        && other.image.descriptor == Descriptor { width: output.width, height: output.height, channels: 3, float: true }
                }
                operation::CHANNEL => same_size && k < d.channels as usize,
                operation::REPLACE_CHANNEL => output == d && k < d.channels as usize && other.image.descriptor == Descriptor { channels: 1, ..d },
                operation::SET_ALPHA => {
                    output == d && matches!(d.channels, GRAY_ALPHA_CHANNELS | COLOR_ALPHA_CHANNELS) && values.len() == 1 && values[0].is_finite()
                }
                operation::BOX_BLUR => {
                    output == d
                        && parameters[parameter::AXIS_OR_Y] <= VERTICAL
                        && k < i32::MAX as usize
                        && values.len() == 2
                        && values.iter().all(|v| v.is_finite() && (0.0..=blanket_gpu::protocol::BOX_WEIGHT_SCALE as f32).contains(v))
                }
                operation::RANK => output == d && (1..=MAX_RANK_SIZE).contains(&k) && k % 2 == 1 && parameters[parameter::AXIS_OR_Y] < (k * k) as u32,
                operation::UNSHARP => output == d && other.image.descriptor == d && values.len() == 1 && values[0].is_finite(),
                operation::LUMINANCE_REDUCE => {
                    let valid_input = match k {
                        reduction::BYTE_INPUT => !d.float,
                        reduction::FLOAT_INPUT => d.float,
                        reduction::BYTE_PARTIALS | reduction::FLOAT_PARTIALS => d.float && d.channels == COLOR_ALPHA_CHANNELS,
                        _ => false,
                    };
                    valid_input
                        && output.height == 1
                        && u64::from(output.width) == (u64::from(d.width) * u64::from(d.height)).div_ceil(u64::from(reduction::BLOCK_SIZE))
                }
                operation::ENHANCE => {
                    output == d
                        && k <= enhancement::SHARPNESS
                        && values.len() == 1
                        && values[0].is_finite()
                        && (k != enhancement::CONTRAST
                            || other.image.descriptor == Descriptor { width: 1, height: 1, channels: COLOR_ALPHA_CHANNELS, float: true })
                        && (k != enhancement::CONTRAST || u64::from(parameters[parameter::AXIS_OR_Y]) == u64::from(d.width) * u64::from(d.height))
                }
                operation::TRANSFER => output == d && d.float && matches!(d.channels, 3 | COLOR_ALPHA_CHANNELS) && k <= 1,
                operation::CLAMP_ALPHA => output == d && d.float && d.channels == COLOR_ALPHA_CHANNELS,
                operation::COLOR_LUT => {
                    let valid_header = values.len() >= lut::HEADER_LEN
                        && values[..3].iter().all(|v| v.is_finite() && v.fract() == 0.0 && (lut::MIN_SIZE as f32..=lut::MAX_SIZE as f32).contains(v));
                    same_size
                        && d.channels >= 3
                        && matches!(k, 3 | 4)
                        && output.float == d.float
                        && output.channels >= k as u32
                        && (output.channels == k as u32 || d.channels >= output.channels)
                        && valid_header
                        && values.len() == lut::HEADER_LEN + k * values[..3].iter().map(|v| *v as usize).product::<usize>()
                        && values[3..lut::HEADER_LEN]
                            .iter()
                            .all(|v| v.is_finite() && (d.float || (0.0..=(1 << lut::POSITION_BITS) as f32).contains(v)))
                        && values[lut::HEADER_LEN..]
                            .iter()
                            .all(|v| v.is_finite() && (d.float || (f32::from(i16::MIN)..=f32::from(i16::MAX)).contains(v)))
                }
                _ => false,
            };
            if !valid {
                return Err(PyValueError::new_err("invalid GPU dispatch"));
            }
            // Validation above checks shader bounds, layouts, and the shared context.
            let image = unsafe { self.context.dispatch(&self.image, &other.image, &Dispatch { output, parameters, values }) }
                .map_err(PyRuntimeError::new_err)?;
            Ok(Self { context: self.context, image })
        }
        fn download<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let data = self.context.download_bytes(&self.image).map_err(PyRuntimeError::new_err)?;
            Ok(PyBytes::new(py, data))
        }
        fn synchronize(&self) -> PyResult<()> {
            self.context.synchronize(&self.image).map_err(PyRuntimeError::new_err)
        }
    }
    #[pyfunction]
    pub(super) fn _gpu_available() -> bool {
        context().is_ok()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_gpu_unsigned_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(_gpu_weights, module)?)?;
    module.add_function(wrap_pyfunction!(_gpu_blur_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(_gpu_lut_controls, module)?)?;
    module.add("_gpu_shader_source", blanket_gpu_metal::SHADERS)?;
    #[cfg(target_os = "macos")]
    {
        module.add_class::<native::MetalImage>()?;
        module.add_function(wrap_pyfunction!(native::_gpu_available, module)?)?;
    }
    Ok(())
}
