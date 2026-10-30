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
// Scalar bytes, packed bytes, and floats specialize each channel count.
const PIPELINE_VARIANTS: usize = 12;
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
const CHANNELS_CONSTANT: usize = 1;
// Packed byte kernels process one aligned uchar4 per thread, with a scalar tail.
const PACKED_SAMPLES: usize = 4;
const SLIDING_BLUR_RADIUS: u32 = 32;
// Matches MAX_SIMDGROUPS in statistics.metal.
const STATISTICS_SIMDGROUPS: usize = 32;
// Bound retained allocations, including buffers whose GPU work is still pending.
const POOL_MAX_BYTES: usize = 128 * 1024 * 1024;
const POOL_MAX_BUFFERS: usize = 16;
const CONTROL_MAX_BYTES: usize = 16 * 1024 * 1024;

type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ControlEntry = (Vec<u32>, Retained<ProtocolObject<dyn MTLBuffer>>);

fn packed_operation(op: u32, floating: bool, channels: u32) -> bool {
    !floating
        && (matches!(
            op,
            operation::COPY
                | operation::BLEND
                | operation::BRIGHTNESS
                | operation::INVERT
                | operation::BINARY
                | operation::POINT
                | operation::RANK
                | operation::BOX_BLUR
        ) || (channels >= blanket_gpu::protocol::GRAY_ALPHA_CHANNELS && matches!(op, operation::RESIZE | operation::CONVOLVE)))
}

fn dispatch_samples(parameters: &[u32; 8], floating: bool, samples: usize, simd_width: usize) -> usize {
    let [width, height, channels, _, _, op, radius, axis] = *parameters;
    if op == operation::BOX_BLUR && !floating && radius >= SLIDING_BLUR_RADIUS {
        let (length, lines) = if axis == 0 { (width, height) } else { (height, width) };
        let lanes = if channels == 4 { 1 } else { channels as usize };
        (length as usize).div_ceil(simd_width) * lines as usize * lanes * simd_width
    } else if op == operation::LUMINANCE_REDUCE {
        samples / 4 * simd_width
    } else if !floating && matches!(op, operation::RESIZE | operation::CONVOLVE | operation::BOX_BLUR) && matches!(channels, 2 | 3) {
        samples / channels as usize
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
    statistics: [Pipeline; SAMPLE_FORMAT_COUNT],
    finish_statistics: Pipeline,
    paste: [Pipeline; SAMPLE_FORMAT_COUNT],
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
        &self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, src: &ProtocolObject<dyn MTLBuffer>, dst: &ProtocolObject<dyn MTLBuffer>,
        parameters: [u32; 11], floating: bool,
    ) {
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
            let buffers = sources.iter().map(|_| self.allocate_image(d.byte_len()?)).collect::<Result<Vec<_>, String>>()?;
            let pipelines = geometry
                .steps()
                .iter()
                .map(|step| {
                    if step.operation == operation::BOX_BLUR && !d.float && step.kernel >= SLIDING_BLUR_RADIUS {
                        Ok(&*self.blur_pipeline)
                    } else {
                        self.operation_pipeline(step.operation, d.float, d.channels)
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal plan command buffer")?;
            // Serial compute encoders complete each dispatch before the next begins.
            let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal plan encoder")?;
            let mut outputs = Vec::with_capacity(sources.len());
            for (source, output) in sources.iter().zip(buffers) {
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
                            &encoder,
                            &source.buffer,
                            &scratch[0],
                            [d.width, d.height, d.channels, tw, th, left, top, 0, 0, tw, th],
                            d.float,
                        );
                        for (index, (step, pipeline)) in geometry.steps().iter().zip(&pipelines).enumerate() {
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
                        }
                        self.encode_tile(
                            &encoder,
                            &scratch[geometry.steps().len() % 2],
                            &output,
                            [tw, th, d.channels, d.width, d.height, x - left, y - top, x, y, width, height],
                            d.float,
                        );
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
            encoder.endEncoding();
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

    pub fn paste(&self, background: &Image, source: &Image, mask: Option<&Image>, position: [i32; 2]) -> Result<Image, String> {
        let d = background.descriptor;
        let s = source.descriptor;
        if d.channels != s.channels
            || d.float != s.float
            || !Arc::ptr_eq(&background.pool, &self.pool)
            || !Arc::ptr_eq(&source.pool, &self.pool)
            || mask.is_some_and(|mask| {
                mask.descriptor.width != s.width
                    || mask.descriptor.height != s.height
                    || mask.descriptor.channels != 1
                    || !Arc::ptr_eq(&mask.pool, &self.pool)
            })
        {
            return Err("incompatible GPU paste images".into());
        }
        autoreleasepool(|_| {
            let buffer = self.allocate_image(d.byte_len()?)?;
            let pipeline = &self.paste[usize::from(d.float)];
            let command = self.queue.commandBuffer().ok_or("cannot allocate Metal command buffer")?;
            let encoder = command.computeCommandEncoder().ok_or("cannot allocate Metal compute encoder")?;
            let parameters = [
                d.width,
                d.height,
                d.channels,
                s.width,
                s.height,
                position[0] as u32,
                position[1] as u32,
                mask.map_or(0, |mask| if mask.descriptor.float { 2 } else { 1 }),
            ];
            encoder.setComputePipelineState(pipeline);
            // Shapes and contexts are checked above; commands retain every input.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&background.buffer), 0, SOURCE_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&buffer), 0, OUTPUT_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&source.buffer), 0, OTHER_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&mask.unwrap_or(source).buffer), 0, 5);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), PARAMETERS_SLOT);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: d.width as usize * d.height as usize * if d.float { d.channels as usize } else { 1 }, height: 1, depth: 1 },
                MTLSize { width: THREADGROUP_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
            );
            encoder.endEncoding();
            background.track_read(&command);
            source.track_read(&command);
            if let Some(mask) = mask {
                mask.track_read(&command);
            }
            command.commit();
            Ok(Image { descriptor: d, buffer, completion: Some(command), pool: Arc::clone(&self.pool), uses: Mutex::default() })
        })
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
                "#define BLANKET_METAL_SPECIALIZE\n{}\n{}\n{}\n{}\n{}",
                super::SHADERS,
                include_str!("tiles.metal"),
                include_str!("blur.metal"),
                include_str!("statistics.metal"),
                include_str!("paste.metal")
            ));
            let library = device.newLibraryWithSource_options_error(&source, Some(&options)).map_err(|error| error.to_string())?;
            let pipeline = |name| {
                let function = library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("Metal kernel unavailable: {name}"))?;
                device.newComputePipelineStateWithFunction_error(&function).map_err(|error| error.to_string())
            };
            let pipelines = std::array::from_fn(|_| std::array::from_fn(|_| OnceLock::new()));
            let conversions =
                [[pipeline("convert_u8_u8")?, pipeline("convert_u8_f32")?], [pipeline("convert_f32_u8")?, pipeline("convert_f32_f32")?]];
            let reductions = [pipeline("luminance_u8")?, pipeline("luminance_f32")?];
            let statistics = [pipeline("statistics_u8")?, pipeline("statistics_f32")?];
            let finish_statistics = pipeline("finish_statistics")?;
            let paste = [pipeline("paste_u8")?, pipeline("paste_f32")?];
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
                statistics,
                finish_statistics,
                paste,
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
        let variant = (if floating {
            8
        } else if packed {
            4
        } else {
            0
        }) + channels as usize
            - 1;
        slots[variant]
            .get_or_init(|| {
                let constants = MTLFunctionConstantValues::new();
                // Metal copies one uint into the shader's operation constant at index zero.
                unsafe {
                    constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(&operation).cast(), MTLDataType::UInt, OPERATION_CONSTANT);
                    constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(&channels).cast(), MTLDataType::UInt, CHANNELS_CONSTANT);
                }
                let name = if floating {
                    "image_f32"
                } else if matches!(operation, operation::RESIZE | operation::CONVOLVE | operation::BOX_BLUR) && matches!(channels, 2 | 3) {
                    "color_u8"
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
            } else if op == operation::STATISTICS {
                &*self.statistics[usize::from(source.descriptor.float)]
            } else {
                self.operation_pipeline(op, request.output.float, source.descriptor.channels)?
            };
            let blocks = (source.descriptor.width as usize * source.descriptor.height as usize).div_ceil(4096).min(256);
            let partials = if op == operation::STATISTICS && blocks > 1 { Some(self.allocate_image(output_bytes * blocks)?) } else { None };
            let mut parameters = request.parameters;
            if op == operation::STATISTICS {
                parameters[7] = blocks as u32;
            }
            encoder.setComputePipelineState(pipeline);
            // The caller validates shader bounds and layouts; the command retains all buffers.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.buffer), 0, SOURCE_SLOT);
                encoder.setBuffer_offset_atIndex(Some(partials.as_ref().unwrap_or(&buffer)), 0, OUTPUT_SLOT);
                encoder.setBuffer_offset_atIndex(Some(&other.buffer), 0, OTHER_SLOT);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), PARAMETERS_SLOT);
                if let Some(weights) = &weights {
                    encoder.setBuffer_offset_atIndex(Some(weights), 0, VALUES_SLOT);
                } else {
                    // setBytes copies these controls before the borrowed slice expires.
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(values).cast(), values_bytes, VALUES_SLOT);
                }
            }
            if op == operation::STATISTICS {
                let width = pipeline.threadExecutionWidth();
                let threads = pipeline.maxTotalThreadsPerThreadgroup().min(THREADGROUP_THREADS).min(STATISTICS_SIMDGROUPS * width) / width * width;
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: source.descriptor.channels as usize * blocks, height: 1, depth: 1 },
                    MTLSize { width: threads, height: 1, depth: 1 },
                );
                if let Some(partials) = &partials {
                    encoder.setComputePipelineState(&self.finish_statistics);
                    // The serial encoder completes partial writes before the merge.
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(partials), 0, SOURCE_SLOT);
                        encoder.setBuffer_offset_atIndex(Some(&buffer), 0, OUTPUT_SLOT);
                    }
                    let samples = if parameters[6] == 1 { output_bytes / size_of::<f32>() } else { source.descriptor.channels as usize };
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: samples, height: 1, depth: 1 },
                        MTLSize { width: THREADGROUP_THREADS.min(self.finish_statistics.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
                    );
                }
            } else {
                let n = output_bytes / if request.output.float { size_of::<f32>() } else { size_of::<u8>() };
                let n = dispatch_samples(&request.parameters, request.output.float, n, pipeline.threadExecutionWidth());
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: n, height: 1, depth: 1 },
                    MTLSize { width: THREADGROUP_THREADS.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
                );
            }
            encoder.endEncoding();
            // A source may be closed before its consumers finish; pool reuse waits for every reader.
            source.track_read(&command);
            if !std::ptr::eq(source, other) {
                other.track_read(&command);
            }
            command.commit();
            if let Some(buffer) = partials {
                self.pool.lock().unwrap_or_else(|error| error.into_inner()).put(PoolEntry { buffer, uses: vec![command.clone()] });
            }
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

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes.as_chunks::<4>().0.iter().map(|&chunk| f32::from_ne_bytes(chunk)).collect()
    }

    #[test]
    fn luminance_partials_preserve_carries_and_short_simd_groups() {
        let context = Context::new().unwrap();
        for width in [1, 31, 32, 33, 255, 256, 257] {
            let descriptor = Descriptor { width, height: 1, channels: 4, float: true };
            let values: Vec<u64> = (0..width).map(|i| u64::from(u32::MAX) + u64::from(i) * 17).collect();
            let data: Vec<u8> =
                values.iter().flat_map(|&value| [value as u32, (value >> 32) as u32, 0, 0].into_iter().flat_map(u32::to_ne_bytes)).collect();
            let source = context.upload(descriptor, &data).unwrap();
            let output = Descriptor { width: width.div_ceil(256), ..descriptor };
            let request = Dispatch { output, parameters: [width, 1, 4, output.width, 1, operation::LUMINANCE_REDUCE, 2, 0], values: vec![] };
            let result = unsafe { context.dispatch(&source, &source, &request) }.unwrap();
            let actual = floats(&context.download(&result).unwrap());
            for (index, block) in values.chunks(256).enumerate() {
                let sum = u64::from(actual[index * 4].to_bits()) | (u64::from(actual[index * 4 + 1].to_bits()) << 32);
                assert_eq!(sum, block.iter().sum());
            }
        }
    }

    #[test]
    fn parallel_statistics_match_serial_reductions() {
        use blanket_gpu::protocol::statistics;
        let context = Context::new().unwrap();
        // Odd sizes leave partial SIMD groups and idle threads in each threadgroup.
        for (width, height) in [(1, 1), (37, 3), (1031, 17)] {
            let descriptor = Descriptor { width, height, channels: 3, float: false };
            let data: Vec<u8> = (0..descriptor.byte_len().unwrap()).map(|i| ((i * 37) % 251) as u8).collect();
            let image = context.upload(descriptor, &data).unwrap();
            let parameters = |output: Descriptor, kind| [width, height, 3, output.width, output.height, operation::STATISTICS, kind, 0];
            let output = Descriptor { width: statistics::HISTOGRAM_BINS, height: 3, channels: 1, float: true };
            let request = Dispatch { output, parameters: parameters(output, 1), values: vec![] };
            let counts = floats(&context.download(&unsafe { context.dispatch(&image, &image, &request) }.unwrap()).unwrap());
            let output = Descriptor { width: statistics::FIELDS, height: 3, channels: 1, float: true };
            let request = Dispatch { output, parameters: parameters(output, 0), values: vec![] };
            let fields = floats(&context.download(&unsafe { context.dispatch(&image, &image, &request) }.unwrap()).unwrap());
            for band in 0..3 {
                let values: Vec<u32> = data.iter().skip(band).step_by(3).map(|&value| u32::from(value)).collect();
                let mut expected = [0_u32; 256];
                values.iter().for_each(|&value| expected[value as usize] += 1);
                let actual: Vec<u32> = counts[band * 256..][..256].iter().map(|count| count.to_bits()).collect();
                assert_eq!(actual, expected);
                let field = &fields[band * statistics::FIELDS as usize..][..statistics::FIELDS as usize];
                assert_eq!(field[0], *values.iter().min().unwrap() as f32);
                assert_eq!(field[1], *values.iter().max().unwrap() as f32);
                assert_eq!(field[2], values.iter().sum::<u32>() as f32);
                let fourth: f64 = values.iter().map(|&value| f64::from(value).powi(4)).sum();
                assert!((f64::from(field[5]) - fourth).abs() <= fourth * 1e-5);
                assert_eq!(field[6].to_bits(), values.len() as u32);
            }
        }
        let descriptor = Descriptor { width: 999, height: 1, channels: 1, float: true };
        let samples: Vec<f32> = (0..999).map(|i| if i == 700 { f32::NAN } else { i as f32 }).collect();
        let image = context.upload(descriptor, &samples.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>()).unwrap();
        let output = Descriptor { width: statistics::FIELDS, height: 1, channels: 1, float: true };
        let request = Dispatch { output, parameters: [999, 1, 1, output.width, 1, operation::STATISTICS, 0, 0], values: vec![] };
        let fields = floats(&context.download(&unsafe { context.dispatch(&image, &image, &request) }.unwrap()).unwrap());
        assert!(fields[0].is_nan() && fields[1].is_nan());
    }

    #[test]
    fn byte_resize_matches_signed_fixed_point() {
        let context = Context::new().unwrap();
        for channels in [1, 2, 3, 4] {
            let descriptor = Descriptor { width: 9, height: 2, channels, float: false };
            let data: Vec<u8> = (0..descriptor.byte_len().unwrap()).map(|i| (i * 97) as u8).collect();
            let image = context.upload(descriptor, &data).unwrap();
            let weights = [-0.1_f32, 0.6, 0.7, -0.2];
            let values: Vec<f32> = (0..6).flat_map(|x| [x as f32, 4.0].into_iter().chain(weights)).collect();
            let output = Descriptor { width: 6, ..descriptor };
            let request = Dispatch { output, parameters: [9, 2, channels, 6, 2, operation::RESIZE, 6, 0], values };
            let actual = context.download(&unsafe { context.dispatch(&image, &image, &request) }.unwrap()).unwrap();
            let fixed = weights.map(|weight| (weight * (1 << 22) as f32).round() as i64);
            let expected: Vec<u8> = (0..actual.len())
                .map(|i| {
                    let (band, x, y) = (i % channels as usize, i / channels as usize % 6, i / channels as usize / 6);
                    let sum: i64 = (0..4).map(|t| i64::from(data[(y * 9 + x + t) * channels as usize + band]) * fixed[t]).sum();
                    ((sum + (1 << 21)) >> 22).clamp(0, 255) as u8
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }
}
