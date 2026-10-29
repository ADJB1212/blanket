use std::sync::{Arc, Mutex, OnceLock};

use blanket_gpu::protocol::operation;
use blanket_gpu::{Backend, Descriptor, Dispatch};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLLibraryOptimizationLevel, MTLMathMode, MTLResourceOptions, MTLSize,
};

// Metal device discovery requires CoreGraphics to be linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// Pipeline indices distinguish uint8 and float32 sample storage.
const SAMPLE_FORMAT_COUNT: usize = 2;
// Scalar bytes, packed bytes, and scalar float kernels have separate pipelines.
const PIPELINE_VARIANTS: usize = 3;
// Buffer slots match the [[buffer(n)]] arguments in kernels.metal.
const SOURCE_SLOT: usize = 0;
const OUTPUT_SLOT: usize = 1;
const OTHER_SLOT: usize = 2;
const PARAMETERS_SLOT: usize = 3;
const VALUES_SLOT: usize = 4;
// Metal setBytes accepts at most 4 KiB per binding.
const INLINE_CONTROL_BYTES: usize = 4096;
// Eight SIMD groups amortize threadgroup scheduling for streaming image kernels.
const THREADGROUP_THREADS: usize = 256;
// Operation IDs are contiguous through unsigned quantization.
const OPERATION_COUNT: usize = operation::QUANTIZE as usize + 1;
// Matches SPECIALIZED_OPERATION in kernels.metal.
const OPERATION_CONSTANT: usize = 0;
// Packed byte kernels process one aligned uchar4 per thread, with a scalar tail.
const PACKED_SAMPLES: usize = 4;
// Bound retained allocations, including buffers whose GPU work is still pending.
const POOL_MAX_BYTES: usize = 128 * 1024 * 1024;
const POOL_MAX_BUFFERS: usize = 16;

type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn packed_operation(op: u32, floating: bool, channels: u32) -> bool {
    !floating
        && (matches!(op, operation::COPY | operation::BLEND | operation::BRIGHTNESS | operation::INVERT)
            || (channels == blanket_gpu::protocol::COLOR_ALPHA_CHANNELS
                && matches!(op, operation::RESIZE | operation::BOX_BLUR | operation::CONVOLVE)))
}

fn finished(command: &Command) -> bool {
    matches!(command.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
}

#[derive(Default)]
struct BufferPool {
    entries: Vec<PoolEntry>,
    bytes: usize,
}

struct PoolEntry {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    uses: Vec<Command>,
}

// Entries own closed images and only committed commands. Moving them transfers
// ownership; CPU access and re-encoding wait until every GPU use has completed.
unsafe impl Send for PoolEntry {}

impl BufferPool {
    fn take(&mut self, len: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let index = self.entries.iter().position(|entry| entry.buffer.length() == len && entry.uses.iter().all(finished))?;
        self.bytes -= len;
        Some(self.entries.swap_remove(index).buffer)
    }

    fn put(&mut self, entry: PoolEntry) {
        let len = entry.buffer.length();
        if len > POOL_MAX_BYTES {
            return;
        }
        while self.entries.len() >= POOL_MAX_BUFFERS || len > POOL_MAX_BYTES - self.bytes {
            self.bytes -= self.entries.remove(0).buffer.length();
        }
        self.bytes += len;
        self.entries.push(entry);
    }
}

pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    max_buffer_length: usize,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: [[OnceLock<Result<Pipeline, String>>; PIPELINE_VARIANTS]; OPERATION_COUNT],
    pool: Arc<Mutex<BufferPool>>,
    conversions: [[Retained<ProtocolObject<dyn MTLComputePipelineState>>; SAMPLE_FORMAT_COUNT]; SAMPLE_FORMAT_COUNT],
    reductions: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; SAMPLE_FORMAT_COUNT],
}
pub struct Image {
    pub descriptor: Descriptor,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    completion: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pool: Arc<Mutex<BufferPool>>,
    uses: Mutex<Vec<Command>>,
}

// Published buffers are immutable. Reader tracking and pool access are locked;
// command status and completion waits may be queried from any CPU thread.
unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    fn track_read(&self, command: &Command) {
        let mut uses = self.uses.lock().unwrap_or_else(|error| error.into_inner());
        uses.retain(|command| !finished(command));
        uses.push(command.clone());
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let mut uses = std::mem::take(self.uses.get_mut().unwrap_or_else(|error| error.into_inner()));
        uses.extend(self.completion.take());
        self.pool.lock().unwrap_or_else(|error| error.into_inner()).put(PoolEntry { buffer: self.buffer.clone(), uses });
    }
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

            options.setMathMode(MTLMathMode::Relaxed);
            options.setOptimizationLevel(MTLLibraryOptimizationLevel::Default);
            let source = NSString::from_str(&format!("#define BLANKET_METAL_SPECIALIZE\n{}", super::SHADERS));
            let library = device.newLibraryWithSource_options_error(&source, Some(&options)).map_err(|error| error.to_string())?;
            let pipeline = |name| {
                let function = library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("Metal kernel unavailable: {name}"))?;
                device.newComputePipelineStateWithFunction_error(&function).map_err(|error| error.to_string())
            };
            let pipelines = std::array::from_fn(|_| std::array::from_fn(|_| OnceLock::new()));
            let conversions =
                [[pipeline("convert_u8_u8")?, pipeline("convert_u8_f32")?], [pipeline("convert_f32_u8")?, pipeline("convert_f32_f32")?]];
            let reductions = [pipeline("reduce_u8")?, pipeline("reduce_f32")?];
            let queue = device.newCommandQueue().ok_or("cannot allocate Metal command queue")?;
            let max_buffer_length = device.maxBufferLength();
            Ok(Self { device, max_buffer_length, queue, library, pipelines, conversions, reductions, pool: Arc::default() })
        })
    }
    fn operation_pipeline(&self, operation: u32, floating: bool, channels: u32) -> Result<&ProtocolObject<dyn MTLComputePipelineState>, String> {
        let slots = self.pipelines.get(operation as usize).ok_or("invalid GPU operation")?;
        let packed = packed_operation(operation, floating, channels);
        let variant = if floating { PIPELINE_VARIANTS - 1 } else { usize::from(packed) };
        slots[variant]
            .get_or_init(|| {
                let constants = MTLFunctionConstantValues::new();
                // Metal copies one uint into the shader's operation constant at index zero.
                unsafe {
                    constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(&operation).cast(), MTLDataType::UInt, OPERATION_CONSTANT);
                }
                let name = if floating {
                    "image_f32"
                } else if packed {
                    "packed_u8"
                } else {
                    "image_u8"
                };
                let function = self
                    .library
                    .newFunctionWithName_constantValues_error(&NSString::from_str(name), &constants)
                    .map_err(|error| error.to_string())?;
                self.device.newComputePipelineStateWithFunction_error(&function).map_err(|error| error.to_string())
            })
            .as_ref()
            .map(|pipeline| &**pipeline)
            .map_err(Clone::clone)
    }
    fn allocate_bytes(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        if len > self.max_buffer_length {
            return Err("image exceeds Metal buffer limit".into());
        }
        self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared).ok_or_else(|| "cannot allocate Metal buffer".into())
    }
    fn allocate_image(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let cached = self.pool.lock().unwrap_or_else(|error| error.into_inner()).take(len);
        cached.map_or_else(|| self.allocate_bytes(len), Ok)
    }
}

impl Backend for Context {
    type Image = Image;
    fn upload(&self, descriptor: Descriptor, bytes: &[u8]) -> Result<Image, String> {
        if descriptor.byte_len()? != bytes.len() {
            return Err("incorrect upload length".into());
        }
        let buffer = self.allocate_image(bytes.len())?;
        // The new allocation is exclusively owned and has exactly bytes.len() writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast().as_ptr(), bytes.len());
        }
        Ok(Image { descriptor, buffer, completion: None, pool: Arc::clone(&self.pool), uses: Mutex::default() })
    }
    unsafe fn dispatch(&self, source: &Image, other: &Image, request: &Dispatch) -> Result<Image, String> {
        autoreleasepool(|_| {
            let output_bytes = request.output.byte_len()?;
            let buffer = self.allocate_image(output_bytes)?;
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
            let op = request.parameters[blanket_gpu::protocol::parameter::OPERATION];
            let pipeline = if op == operation::CONVERT {
                &*self.conversions[usize::from(source.descriptor.float)][usize::from(request.output.float)]
            } else if op == operation::LUMINANCE_REDUCE {
                &*self.reductions[usize::from(source.descriptor.float)]
            } else {
                self.operation_pipeline(
                    op,
                    if op == operation::STATISTICS { source.descriptor.float } else { request.output.float },
                    source.descriptor.channels,
                )?
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
            let n = if packed_operation(op, request.output.float, request.output.channels) { n.div_ceil(PACKED_SAMPLES) } else { n };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: THREADGROUP_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
            );
            encoder.endEncoding();
            // A source may be closed before its consumers finish; pool reuse waits for every reader.
            source.track_read(&command);
            if !std::ptr::eq(source, other) {
                other.track_read(&command);
            }
            command.commit();
            Ok(Image { descriptor: request.output, buffer, completion: Some(command), pool: Arc::clone(&self.pool), uses: Mutex::default() })
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
