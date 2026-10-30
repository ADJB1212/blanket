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
const SLIDING_BLUR_RADIUS: u32 = 32;
// Bound retained allocations, including buffers whose GPU work is still pending.
const POOL_MAX_BYTES: usize = 128 * 1024 * 1024;
const POOL_MAX_BUFFERS: usize = 16;
const CONTROL_MAX_BYTES: usize = 16 * 1024 * 1024;

type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ControlEntry = (Vec<u32>, Retained<ProtocolObject<dyn MTLBuffer>>);

fn packed_operation(op: u32, floating: bool, channels: u32) -> bool {
    !floating
        && (matches!(op, operation::COPY | operation::BLEND | operation::BRIGHTNESS | operation::INVERT)
            || (channels == blanket_gpu::protocol::COLOR_ALPHA_CHANNELS
                && matches!(op, operation::RESIZE | operation::BOX_BLUR | operation::CONVOLVE)))
}

fn dispatch_samples(parameters: &[u32; 8], floating: bool, samples: usize, simd_width: usize) -> usize {
    let [width, height, channels, _, _, op, radius, axis] = *parameters;
    if op == operation::BOX_BLUR && !floating && radius >= SLIDING_BLUR_RADIUS {
        let (length, lines) = if axis == 0 { (width, height) } else { (height, width) };
        let lanes = if channels == 4 { 1 } else { channels as usize };
        (length as usize).div_ceil(simd_width) * lines as usize * lanes * simd_width
    } else if packed_operation(op, floating, channels) {
        samples.div_ceil(PACKED_SAMPLES)
    } else {
        samples
    }
}

fn finished(command: &Command) -> bool {
    matches!(command.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
}

struct BufferPool {
    entries: Vec<PoolEntry>,
    bytes: usize,
    limit: usize,
}
impl Default for BufferPool {
    fn default() -> Self {
        Self { entries: Vec::new(), bytes: 0, limit: POOL_MAX_BYTES }
    }
}

struct PoolEntry {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    uses: Vec<Command>,
}

// Entries own closed images and only committed commands. Moving them transfers
// ownership; CPU access waits for completion and GPU writes follow queue order.
unsafe impl Send for PoolEntry {}

impl BufferPool {
    fn take(&mut self, len: usize, cpu_access: bool) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let index = self.entries.iter().position(|entry| entry.buffer.length() == len && (!cpu_access || entry.uses.iter().all(finished)))?;
        self.bytes -= len;
        Some(self.entries.swap_remove(index).buffer)
    }

    fn put(&mut self, entry: PoolEntry) {
        let len = entry.buffer.length();
        if len > self.limit {
            return;
        }
        while self.entries.len() >= POOL_MAX_BUFFERS || len > self.limit - self.bytes {
            self.bytes -= self.entries.remove(0).buffer.length();
        }
        self.bytes += len;
        self.entries.push(entry);
    }
}

struct ControlCache {
    entries: Vec<ControlEntry>,
    bytes: usize,
    limit: usize,
}
unsafe impl Send for ControlCache {}
impl Default for ControlCache {
    fn default() -> Self {
        Self { entries: Vec::new(), bytes: 0, limit: CONTROL_MAX_BYTES }
    }
}

pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    max_buffer_length: usize,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: [[OnceLock<Result<Pipeline, String>>; PIPELINE_VARIANTS]; OPERATION_COUNT],
    pool: Arc<Mutex<BufferPool>>,
    controls: Mutex<ControlCache>,
    tile_pipelines: [Pipeline; SAMPLE_FORMAT_COUNT],
    blur_pipeline: Pipeline,
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
pub struct Plan {
    geometry: blanket_gpu::plan::Plan,
    controls: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}
// Control buffers are immutable after preparation.
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

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
    pub fn upload_with(&self, descriptor: Descriptor, initialize: impl FnOnce(&mut [u8])) -> Result<Image, String> {
        let len = descriptor.byte_len()?;
        let buffer = self.pool.lock().unwrap_or_else(|error| error.into_inner()).take(len, true);
        let buffer = buffer.map_or_else(|| self.allocate_bytes(len), Ok)?;
        // CPU reuse requires completed GPU readers and exclusive buffer ownership.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast().as_ptr(), len) };
        initialize(bytes);
        Ok(Image { descriptor, buffer, completion: None, pool: Arc::clone(&self.pool), uses: Mutex::default() })
    }

    pub fn prepare_plan(&self, geometry: blanket_gpu::plan::Plan) -> Result<Plan, String> {
        let controls = geometry
            .steps()
            .iter()
            .map(|step| {
                self.operation_pipeline(step.operation, geometry.descriptor().float, geometry.descriptor().channels)?;
                self.control_buffer(if step.values.is_empty() { &[0.0] } else { &step.values })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Plan { geometry, controls })
    }
    pub fn plan_info(plan: &Plan) -> (usize, [u32; 2], [u32; 2], usize) {
        (plan.geometry.steps().len(), plan.geometry.tile(), plan.geometry.halo(), plan.geometry.scratch_bytes())
    }
    fn encode_tile(
        &self, command: &Command, src: &ProtocolObject<dyn MTLBuffer>, dst: &ProtocolObject<dyn MTLBuffer>, parameters: [u32; 11], floating: bool,
    ) -> Result<(), String> {
        let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal tile encoder")?;
        let pipeline = &self.tile_pipelines[usize::from(floating)];
        encoder.setComputePipelineState(pipeline);
        // The plan bounds source/destination rectangles and retains every buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src), 0, SOURCE_SLOT);
            encoder.setBuffer_offset_atIndex(Some(dst), 0, OUTPUT_SLOT);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), PARAMETERS_SLOT);
        }
        let samples = parameters[9] as usize * parameters[10] as usize * parameters[2] as usize;
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: samples, height: 1, depth: 1 },
            MTLSize { width: THREADGROUP_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
        );
        encoder.endEncoding();
        Ok(())
    }
    pub fn execute_plan(&self, plan: &Plan, sources: &[&Image]) -> Result<Vec<Image>, String> {
        autoreleasepool(|_| {
            let geometry = &plan.geometry;
            let d = geometry.descriptor();
            let tile = geometry.tile();
            let halo = geometry.halo();
            if sources.len() > 1024 || sources.iter().any(|source| source.descriptor != d || !Arc::ptr_eq(&source.pool, &self.pool)) {
                return Err("GPU plan inputs must match its descriptor and execution context (at most 1024 images)".into());
            }
            if sources.is_empty() {
                return Ok(Vec::new());
            }
            let scratch = [self.allocate_image(geometry.scratch_bytes() / 2)?, self.allocate_image(geometry.scratch_bytes() / 2)?];
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal plan command buffer")?;
            let mut outputs = Vec::with_capacity(sources.len());
            for source in sources {
                let output = self.allocate_image(d.byte_len()?)?;
                for y in (0..d.height).step_by(tile[1] as usize) {
                    for x in (0..d.width).step_by(tile[0] as usize) {
                        let width = tile[0].min(d.width - x);
                        let height = tile[1].min(d.height - y);
                        let left = x.saturating_sub(halo[0]);
                        let top = y.saturating_sub(halo[1]);
                        let right = (u64::from(x) + u64::from(width) + u64::from(halo[0])).min(u64::from(d.width)) as u32;
                        let bottom = (u64::from(y) + u64::from(height) + u64::from(halo[1])).min(u64::from(d.height)) as u32;
                        let tw = right - left;
                        let th = bottom - top;
                        self.encode_tile(
                            &command,
                            &source.buffer,
                            &scratch[0],
                            [d.width, d.height, d.channels, tw, th, left, top, 0, 0, tw, th],
                            d.float,
                        )?;
                        for (index, step) in geometry.steps().iter().enumerate() {
                            let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal plan encoder")?;
                            let pipeline = if step.operation == operation::BOX_BLUR && !d.float && step.kernel >= SLIDING_BLUR_RADIUS {
                                &*self.blur_pipeline
                            } else {
                                self.operation_pipeline(step.operation, d.float, d.channels)?
                            };
                            encoder.setComputePipelineState(pipeline);
                            let parameters = [tw, th, d.channels, tw, th, step.operation, step.kernel, step.axis];
                            // Both scratch buffers fit the expanded tile and are used in command order.
                            unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&scratch[index % 2]), 0, SOURCE_SLOT);
                                encoder.setBuffer_offset_atIndex(Some(&scratch[(index + 1) % 2]), 0, OUTPUT_SLOT);
                                encoder.setBuffer_offset_atIndex(Some(&scratch[index % 2]), 0, OTHER_SLOT);
                                encoder.setBytes_length_atIndex(
                                    std::ptr::NonNull::from(&parameters).cast(),
                                    std::mem::size_of_val(&parameters),
                                    PARAMETERS_SLOT,
                                );
                                encoder.setBuffer_offset_atIndex(Some(&plan.controls[index]), 0, VALUES_SLOT);
                            }
                            let samples = tw as usize * th as usize * d.channels as usize;
                            let samples = dispatch_samples(&parameters, d.float, samples, pipeline.threadExecutionWidth());
                            encoder.dispatchThreads_threadsPerThreadgroup(
                                MTLSize { width: samples, height: 1, depth: 1 },
                                MTLSize { width: THREADGROUP_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
                            );
                            encoder.endEncoding();
                        }
                        self.encode_tile(
                            &command,
                            &scratch[geometry.steps().len() % 2],
                            &output,
                            [tw, th, d.channels, d.width, d.height, x - left, y - top, x, y, width, height],
                            d.float,
                        )?;
                    }
                }
                outputs.push(Image {
                    descriptor: d,
                    buffer: output,
                    completion: Some(command.clone()),
                    pool: Arc::clone(&self.pool),
                    uses: Mutex::default(),
                });
            }
            for source in sources {
                source.track_read(&command);
            }
            command.commit();
            for buffer in scratch {
                self.pool.lock().unwrap_or_else(|error| error.into_inner()).put(PoolEntry { buffer, uses: vec![command.clone()] });
            }
            Ok(outputs)
        })
    }
    pub fn is_complete(&self, image: &Image) -> bool {
        image.completion.as_ref().is_none_or(finished)
    }

    pub fn wait_idle(&self) -> Result<(), String> {
        autoreleasepool(|_| {
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal command buffer")?;
            command.commit();
            command.waitUntilCompleted();
            if command.status() == MTLCommandBufferStatus::Error {
                return Err(command.error().map_or_else(|| "Metal synchronization failed".into(), |error| error.to_string()));
            }
            Ok(())
        })
    }
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
            let source = NSString::from_str(&format!(
                "#define BLANKET_METAL_SPECIALIZE\n{}\n{}\n{}",
                super::SHADERS,
                include_str!("tiles.metal"),
                include_str!("blur.metal")
            ));
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
            let tile_pipelines = [pipeline("tile_u8")?, pipeline("tile_f32")?];
            let blur_pipeline = pipeline("scan_blur_u8")?;
            Ok(Self {
                device,
                max_buffer_length,
                queue,
                library,
                pipelines,
                conversions,
                reductions,
                pool: Arc::default(),
                controls: Mutex::default(),
                tile_pipelines,
                blur_pipeline,
            })
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
        let cached = self.pool.lock().unwrap_or_else(|error| error.into_inner()).take(len, false);
        cached.map_or_else(|| self.allocate_bytes(len), Ok)
    }
    pub fn configure_cache(&self, image_bytes: usize, control_bytes: usize) {
        let mut pool = self.pool.lock().unwrap_or_else(|error| error.into_inner());
        pool.limit = image_bytes;
        pool.entries.clear();
        pool.bytes = 0;
        drop(pool);
        let mut controls = self.controls.lock().unwrap_or_else(|error| error.into_inner());
        controls.limit = control_bytes;
        controls.entries.clear();
        controls.bytes = 0;
    }
    pub fn cache_info(&self) -> (usize, usize, usize, usize) {
        let pool = self.pool.lock().unwrap_or_else(|error| error.into_inner());
        let controls = self.controls.lock().unwrap_or_else(|error| error.into_inner());
        (pool.bytes, pool.limit, controls.bytes, controls.limit)
    }
    fn control_buffer(&self, values: &[f32]) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let key: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        let mut cache = self.controls.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(index) = cache.entries.iter().position(|entry| entry.0 == key) {
            let entry = cache.entries.remove(index);
            let buffer = entry.1.clone();
            cache.entries.push(entry);
            return Ok(buffer);
        }
        let len = std::mem::size_of_val(values);
        let buffer = self.allocate_bytes(len)?;
        // New storage is exclusively owned until published to the cache.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast().as_ptr(), values.len());
        }
        let retained_bytes = len.saturating_mul(2);
        if retained_bytes <= cache.limit {
            while cache.entries.len() >= 64 || retained_bytes > cache.limit - cache.bytes {
                cache.bytes -= cache.entries.remove(0).0.len() * 8;
            }
            cache.bytes += retained_bytes;
            cache.entries.push((key, buffer.clone()));
        }
        drop(cache);
        Ok(buffer)
    }
}

impl Backend for Context {
    type Image = Image;
    fn upload(&self, descriptor: Descriptor, bytes: &[u8]) -> Result<Image, String> {
        if descriptor.byte_len()? != bytes.len() {
            return Err("incorrect upload length".into());
        }
        self.upload_with(descriptor, |output| output.copy_from_slice(bytes))
    }
    unsafe fn dispatch(&self, source: &Image, other: &Image, request: &Dispatch) -> Result<Image, String> {
        autoreleasepool(|_| {
            let output_bytes = request.output.byte_len()?;
            let buffer = self.allocate_image(output_bytes)?;
            let values = if request.values.is_empty() { &[0.0][..] } else { &request.values };
            let values_bytes = std::mem::size_of_val(values);
            let weights = if values_bytes > INLINE_CONTROL_BYTES {
                let weights = self.control_buffer(values)?;
                Some(weights)
            } else {
                None
            };
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal command buffer")?;
            let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal compute encoder")?;
            let op = request.parameters[blanket_gpu::protocol::parameter::OPERATION];
            let pipeline = if op == operation::CONVERT {
                &*self.conversions[usize::from(source.descriptor.float)][usize::from(request.output.float)]
            } else if op == operation::BOX_BLUR && !request.output.float && request.parameters[6] >= SLIDING_BLUR_RADIUS {
                &*self.blur_pipeline
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
            let n = dispatch_samples(&request.parameters, request.output.float, n, pipeline.threadExecutionWidth());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_buffers_require_completion_only_for_cpu_access() {
        let context = Context::new().unwrap();
        let buffer = context.allocate_bytes(1024).unwrap();
        let address = buffer.contents();
        let command = context.queue.commandBuffer().unwrap();
        let mut pool = BufferPool::default();
        pool.put(PoolEntry { buffer, uses: vec![command.clone()] });
        assert!(pool.take(1024, true).is_none());
        assert_eq!(pool.take(1024, false).unwrap().contents(), address);
        assert_eq!(pool.bytes, 0);
        command.commit();
        command.waitUntilCompleted();
    }

    #[test]
    fn queued_operations_reuse_storage_without_overwriting_live_inputs() {
        let context = Context::new().unwrap();
        let descriptor = Descriptor { width: 257, height: 129, channels: 4, float: false };
        let data: Vec<u8> = (0..descriptor.byte_len().unwrap()).map(|i| (i * 37) as u8).collect();
        let mut image = context.upload(descriptor, &data).unwrap();
        let mut addresses = std::collections::HashSet::new();
        let request = Dispatch { output: descriptor, parameters: [257, 129, 4, 257, 129, operation::INVERT, 0, 0], values: vec![] };
        for _ in 0..64 {
            addresses.insert(image.buffer.contents());
            // Matching immutable inputs and output dimensions satisfy shader bounds.
            image = unsafe { context.dispatch(&image, &image, &request) }.unwrap();
        }
        assert_eq!(addresses.len(), 2);
        assert_eq!(context.download(&image).unwrap(), data);
    }
}
