use blanket_gpu::protocol::operation;
use blanket_gpu::{Backend, Descriptor, Dispatch};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

// Metal device discovery requires CoreGraphics to be linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// Pipeline indices distinguish uint8 and float32 sample storage.
const SAMPLE_FORMAT_COUNT: usize = 2;
// Buffer slots match the [[buffer(n)]] arguments in kernels.metal.
const SOURCE_SLOT: usize = 0;
const OUTPUT_SLOT: usize = 1;
const OTHER_SLOT: usize = 2;
const PARAMETERS_SLOT: usize = 3;
const VALUES_SLOT: usize = 4;
// Metal setBytes accepts at most 4 KiB per binding.
const INLINE_CONTROL_BYTES: usize = 4096;

pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    max_buffer_length: usize,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; SAMPLE_FORMAT_COUNT],
    conversions: [[Retained<ProtocolObject<dyn MTLComputePipelineState>>; SAMPLE_FORMAT_COUNT]; SAMPLE_FORMAT_COUNT],
    reductions: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; SAMPLE_FORMAT_COUNT],
}
pub struct Image {
    pub descriptor: Descriptor,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    completion: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl Context {
    /// Borrow completed pixel storage until the image is released.
    pub fn download_bytes<'a>(&self, image: &'a Image) -> Result<&'a [u8], String> {
        self.synchronize(image)?;
        // Images are immutable; completion precedes reads and image owns the buffer.
        Ok(unsafe { std::slice::from_raw_parts(image.buffer.contents().cast().as_ptr(), image.buffer.length()) })
    }

    pub fn new() -> Result<Self, String> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
            let options = MTLCompileOptions::new();
            // Keep the selector available on macOS versions predating mathMode.
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let library =
                device.newLibraryWithSource_options_error(&NSString::from_str(super::SHADERS), Some(&options)).map_err(|error| error.to_string())?;
            let pipeline = |name| {
                let function = library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("Metal kernel unavailable: {name}"))?;
                device.newComputePipelineStateWithFunction_error(&function).map_err(|error| error.to_string())
            };
            let pipelines = [pipeline("image_u8")?, pipeline("image_f32")?];
            let conversions =
                [[pipeline("convert_u8_u8")?, pipeline("convert_u8_f32")?], [pipeline("convert_f32_u8")?, pipeline("convert_f32_f32")?]];
            let reductions = [pipeline("reduce_u8")?, pipeline("reduce_f32")?];
            let queue = device.newCommandQueue().ok_or("cannot allocate Metal command queue")?;
            let max_buffer_length = device.maxBufferLength();
            Ok(Self { device, max_buffer_length, queue, pipelines, conversions, reductions })
        })
    }
    fn allocate_bytes(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        if len > self.max_buffer_length {
            return Err("image exceeds Metal buffer limit".into());
        }
        self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared).ok_or_else(|| "cannot allocate Metal buffer".into())
    }
}

impl Backend for Context {
    type Image = Image;
    fn upload(&self, descriptor: Descriptor, bytes: &[u8]) -> Result<Image, String> {
        if descriptor.byte_len()? != bytes.len() {
            return Err("incorrect upload length".into());
        }
        let buffer = self.allocate_bytes(bytes.len())?;
        // The new allocation is exclusively owned and has exactly bytes.len() writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast().as_ptr(), bytes.len());
        }
        Ok(Image { descriptor, buffer, completion: None })
    }
    unsafe fn dispatch(&self, source: &Image, other: &Image, request: &Dispatch) -> Result<Image, String> {
        autoreleasepool(|_| {
            let output_bytes = request.output.byte_len()?;
            let buffer = self.allocate_bytes(output_bytes)?;
            let values = if request.values.is_empty() { &[0.0][..] } else { &request.values };
            let values_bytes = std::mem::size_of_val(values);
            let weights = if values_bytes > INLINE_CONTROL_BYTES {
                let weights = self.allocate_bytes(values_bytes)?;
                // The new control buffer is exclusively owned and fits the entire slice.
                unsafe {
                    std::ptr::copy_nonoverlapping(values.as_ptr(), weights.contents().cast().as_ptr(), values.len());
                }
                Some(weights)
            } else {
                None
            };
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal command buffer")?;
            let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal compute encoder")?;
            let pipeline = if request.parameters[blanket_gpu::protocol::parameter::OPERATION] == operation::CONVERT {
                &self.conversions[usize::from(source.descriptor.float)][usize::from(request.output.float)]
            } else if request.parameters[blanket_gpu::protocol::parameter::OPERATION] == operation::LUMINANCE_REDUCE {
                &self.reductions[usize::from(source.descriptor.float)]
            } else if request.parameters[blanket_gpu::protocol::parameter::OPERATION] == operation::STATISTICS {
                &self.pipelines[usize::from(source.descriptor.float)]
            } else {
                &self.pipelines[usize::from(request.output.float)]
            };
            encoder.setComputePipelineState(pipeline);
            // The caller validates shader bounds and layouts; the command retains all buffers.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.buffer), 0, SOURCE_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&buffer), 0, OUTPUT_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&other.buffer), 0, OTHER_SLOT);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&request.parameters).cast(),
                    std::mem::size_of_val(&request.parameters),
                    PARAMETERS_SLOT,
                );
                if let Some(weights) = &weights {
                    encoder.setBuffer_offset_atIndex(Some(weights), 0, VALUES_SLOT);
                } else {
                    // setBytes copies these controls before the borrowed slice expires.
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(values).cast(), values_bytes, VALUES_SLOT);
                }
            }
            let n = output_bytes / if request.output.float { size_of::<f32>() } else { size_of::<u8>() };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: pipeline.threadExecutionWidth(), height: 1, depth: 1 },
            );
            encoder.endEncoding();
            command.commit();
            Ok(Image { descriptor: request.output, buffer, completion: Some(command) })
        })
    }
    fn synchronize(&self, image: &Image) -> Result<(), String> {
        if let Some(command) = &image.completion {
            command.waitUntilCompleted();
            if command.status() == MTLCommandBufferStatus::Error {
                return Err(command.error().map_or_else(|| "Metal command execution failed".into(), |error| error.to_string()));
            }
        }
        Ok(())
    }
    fn download(&self, image: &Image) -> Result<Vec<u8>, String> {
        self.download_bytes(image).map(<[u8]>::to_vec)
    }
}
