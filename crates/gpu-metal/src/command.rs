use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, Weak};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue,
    MTL4CommitFeedback, MTL4CommitOptions, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer, MTLCommandBufferStatus,
    MTLComputePipelineState, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSize, MTLStages,
};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(super) type Command = Arc<CommandBuffer>;

struct Resources {
    buffers: HashMap<usize, Buffer>,
    pipelines: HashMap<usize, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
}

struct Completion {
    status: MTLCommandBufferStatus,
    error: Option<String>,
}

pub(super) struct Queue {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pending: Arc<Mutex<Vec<Weak<CommandBuffer>>>>,
}

pub(super) struct CommandBuffer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    _allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    resources: Mutex<Resources>,
    completion: Mutex<Completion>,
    completed: Condvar,
    pending: Arc<Mutex<Vec<Weak<CommandBuffer>>>>,
}

// Encoding is private to one caller; submitted resources survive through feedback.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

pub(super) struct Encoder {
    encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    command: Command,
    bindings: RefCell<[u64; 6]>,
}

impl Queue {
    pub(super) fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self, String> {
        let queue = device.newMTL4CommandQueue().ok_or("cannot allocate Metal 4 command queue")?;
        Ok(Self { device: device.clone(), queue, pending: Arc::default() })
    }

    pub(super) fn command_buffer(&self) -> Option<Command> {
        let allocator = self.device.newCommandAllocator()?;
        let buffer = self.device.newCommandBuffer()?;
        let residency = self.device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok()?;
        buffer.beginCommandBufferWithAllocator(&allocator);
        buffer.useResidencySet(&residency);
        Some(Arc::new(CommandBuffer {
            device: self.device.clone(),
            queue: self.queue.clone(),
            buffer,
            _allocator: allocator,
            residency,
            resources: Mutex::new(Resources { buffers: HashMap::new(), pipelines: HashMap::new(), tables: Vec::new() }),
            completion: Mutex::new(Completion { status: MTLCommandBufferStatus::NotEnqueued, error: None }),
            completed: Condvar::new(),
            pending: Arc::clone(&self.pending),
        }))
    }

    pub(super) fn wait_idle(&self) -> Result<(), String> {
        let commands: Vec<_> = self.pending.lock().unwrap_or_else(|error| error.into_inner()).iter().filter_map(Weak::upgrade).collect();
        let mut failure = None;
        for command in commands {
            command.wait_until_completed();
            if let Some(error) = command.error() {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl CommandBuffer {
    pub(super) fn compute_encoder(self: &Arc<Self>) -> Option<Encoder> {
        let encoder = self.buffer.computeCommandEncoder()?;
        encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
            MTLStages::All,
            MTLStages::Dispatch | MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );
        Some(Encoder { encoder, command: Arc::clone(self), bindings: RefCell::new([0; 6]) })
    }

    pub(super) fn status(&self) -> MTLCommandBufferStatus {
        self.completion.lock().unwrap_or_else(|error| error.into_inner()).status
    }

    pub(super) fn error(&self) -> Option<String> {
        self.completion.lock().unwrap_or_else(|error| error.into_inner()).error.clone()
    }

    pub(super) fn commit(self: &Arc<Self>) {
        let mut completion = self.completion.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(completion.status, MTLCommandBufferStatus::NotEnqueued);
        completion.status = MTLCommandBufferStatus::Committed;
        drop(completion);
        let mut pending = self.pending.lock().unwrap_or_else(|error| error.into_inner());
        pending.retain(|command| command.upgrade().is_some_and(|command| command.status() == MTLCommandBufferStatus::Committed));
        pending.push(Arc::downgrade(self));
        drop(pending);
        self.residency.commit();
        self.buffer.endCommandBuffer();
        if let Some(error) = self.error() {
            self.complete(Some(error));
            return;
        }
        let command = Arc::clone(self);
        let feedback = RcBlock::new(move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            // Metal owns the feedback object for the duration of this callback.
            let error = unsafe { feedback.as_ref() }.error().map(|error| error.to_string());
            command.complete(error);
        });
        let options = MTL4CommitOptions::new();
        let mut buffer = NonNull::from(&*self.buffer);
        // Both pointers remain live during commit; Metal copies the feedback block.
        unsafe {
            options.addFeedbackHandler((&raw const *feedback).cast_mut());
            self.queue.commit_count_options(NonNull::from(&mut buffer), 1, &options);
        }
    }

    fn complete(&self, error: Option<String>) {
        self.residency.removeAllAllocations();
        self.residency.commit();
        let mut resources = self.resources.lock().unwrap_or_else(|error| error.into_inner());
        resources.tables.clear();
        resources.pipelines.clear();
        resources.buffers.clear();
        drop(resources);
        let mut completion = self.completion.lock().unwrap_or_else(|error| error.into_inner());
        completion.status = if error.is_some() { MTLCommandBufferStatus::Error } else { MTLCommandBufferStatus::Completed };
        completion.error = error;
        drop(completion);
        self.completed.notify_all();
    }

    fn encoding_error(&self, error: String) {
        self.completion.lock().unwrap_or_else(|error| error.into_inner()).error.get_or_insert(error);
    }

    pub(super) fn wait_until_completed(&self) {
        let mut completion = self.completion.lock().unwrap_or_else(|error| error.into_inner());
        while !matches!(completion.status, MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error) {
            completion = self.completed.wait(completion).unwrap_or_else(|error| error.into_inner());
        }
        drop(completion);
    }

    fn retain_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        self.residency.addAllocation(ProtocolObject::from_ref(buffer));
        let mut resources = self.resources.lock().unwrap_or_else(|error| error.into_inner());
        resources.buffers.entry(std::ptr::from_ref(buffer) as usize).or_insert_with(|| {
            // Binding callers provide a live buffer; retain it through GPU completion.
            unsafe { Retained::retain(std::ptr::from_ref(buffer).cast_mut()) }.unwrap()
        });
        drop(resources);
    }
}

impl Encoder {
    pub(super) fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.encoder.setComputePipelineState(pipeline);
        let mut resources = self.command.resources.lock().unwrap_or_else(|error| error.into_inner());
        resources.pipelines.entry(std::ptr::from_ref(pipeline) as usize).or_insert_with(|| {
            // The pipeline belongs to the context and remains live during encoding.
            unsafe { Retained::retain(std::ptr::from_ref(pipeline).cast_mut()) }.unwrap()
        });
        drop(resources);
    }

    pub(super) unsafe fn set_buffer(&self, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: usize, slot: usize) {
        self.bindings.borrow_mut()[slot] = buffer.map_or(0, |buffer| {
            self.command.retain_buffer(buffer);
            buffer.gpuAddress() + offset as u64
        });
    }

    pub(super) unsafe fn set_bytes(&self, bytes: NonNull<std::ffi::c_void>, len: usize, slot: usize) {
        // The encoding caller supplies a readable control slice of the stated length.
        if let Some(buffer) = unsafe { self.command.device.newBufferWithBytes_length_options(bytes, len, MTLResourceOptions::StorageModeShared) } {
            unsafe { self.set_buffer(Some(&buffer), 0, slot) };
        } else {
            self.command.encoding_error("cannot allocate Metal 4 control buffer".into());
        }
    }

    fn bind_arguments(&self) -> bool {
        if self.command.error().is_some() {
            return false;
        }
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(6);
        descriptor.setInitializeBindings(true);
        let table = match self.command.device.newArgumentTableWithDescriptor_error(&descriptor) {
            Ok(table) => table,
            Err(error) => {
                self.command.encoding_error(error.to_string());
                return false;
            }
        };
        for (slot, &address) in self.bindings.borrow().iter().enumerate() {
            unsafe { table.setAddress_atIndex(address, slot) };
        }
        self.encoder.setArgumentTable(Some(&table));
        self.command.resources.lock().unwrap_or_else(|error| error.into_inner()).tables.push(table);
        true
    }

    fn barrier(&self) {
        let stages = MTLStages::Dispatch | MTLStages::Blit;
        self.encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(stages, stages, MTL4VisibilityOptions::Device);
    }

    pub(super) fn dispatch_threads(&self, grid: MTLSize, group: MTLSize) {
        if self.bind_arguments() {
            self.encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
            self.barrier();
        }
    }

    pub(super) fn dispatch_threadgroups(&self, grid: MTLSize, group: MTLSize) {
        if self.bind_arguments() {
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            self.barrier();
        }
    }

    pub(super) unsafe fn copy_buffer(
        &self, source: &ProtocolObject<dyn MTLBuffer>, source_offset: usize, destination: &ProtocolObject<dyn MTLBuffer>, destination_offset: usize,
        len: usize,
    ) {
        self.command.retain_buffer(source);
        self.command.retain_buffer(destination);
        unsafe {
            self.encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(source, source_offset, destination, destination_offset, len)
        };
        self.barrier();
    }

    pub(super) fn end_encoding(&self) {
        self.encoder.endEncoding();
    }
}
