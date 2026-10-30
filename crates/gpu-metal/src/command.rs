use std::collections::{HashMap, HashSet};
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

const CONTROL_PAGE_BYTES: usize = 4096;
const CONTROL_ALIGNMENT: usize = 16;
const MAX_CACHED_SUBMISSIONS: usize = 16;
const MAX_CACHED_SUBMISSION_BYTES: u64 = 1024 * 1024;
const PENDING_PRUNE_INTERVAL: usize = 64;

// GPU-visible storage is reused only after commit feedback reports completion.
struct Submission {
    buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    controls: Vec<Buffer>,
    control_page: usize,
    control_offset: usize,
    buffers: HashMap<usize, Buffer>,
    pipelines: HashMap<usize, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    resident: HashSet<usize>,
}

impl Submission {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(6);
        descriptor.setInitializeBindings(true);
        Some(Self {
            buffer: device.newCommandBuffer()?,
            allocator: device.newCommandAllocator()?,
            residency: device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok()?,
            table: device.newArgumentTableWithDescriptor_error(&descriptor).ok()?,
            controls: Vec::new(),
            control_page: 0,
            control_offset: 0,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            resident: HashSet::new(),
        })
    }

    fn make_resident(&mut self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        if self.resident.insert(std::ptr::from_ref(buffer) as usize) {
            self.residency.addAllocation(ProtocolObject::from_ref(buffer));
        }
    }

    fn retain_buffer(&mut self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        self.make_resident(buffer);
        self.buffers.entry(std::ptr::from_ref(buffer) as usize).or_insert_with(|| {
            // Encoding callers provide a live buffer.
            unsafe { Retained::retain(std::ptr::from_ref(buffer).cast_mut()) }.unwrap()
        });
    }

    unsafe fn copy_controls(&mut self, device: &ProtocolObject<dyn MTLDevice>, bytes: NonNull<std::ffi::c_void>, len: usize) -> Option<u64> {
        let size = len.checked_add(CONTROL_ALIGNMENT - 1)? / CONTROL_ALIGNMENT * CONTROL_ALIGNMENT;
        while self.controls.get(self.control_page).is_some_and(|page| size > page.length().saturating_sub(self.control_offset)) {
            self.control_page += 1;
            self.control_offset = 0;
        }
        if self.control_page == self.controls.len() {
            self.controls.push(device.newBufferWithLength_options(CONTROL_PAGE_BYTES.max(size), MTLResourceOptions::StorageModeShared)?);
        }
        let page = self.controls[self.control_page].clone();
        let offset = self.control_offset;
        if size > page.length() {
            return None;
        }
        // Each control slice owns a distinct aligned range until GPU completion.
        unsafe { std::ptr::copy_nonoverlapping(bytes.cast::<u8>().as_ptr(), page.contents().cast::<u8>().as_ptr().add(offset), len) };
        self.control_offset += size;
        self.make_resident(&page);
        Some(page.gpuAddress() + offset as u64)
    }

    fn recycle(&mut self) {
        self.residency.removeAllAllocations();
        self.residency.commit();
        self.buffers.clear();
        self.pipelines.clear();
        self.resident.clear();
        self.control_page = 0;
        self.control_offset = 0;
        self.allocator.reset();
    }

    fn cached_bytes(&self) -> u64 {
        self.controls.iter().fold(self.allocator.allocatedSize(), |bytes, page| bytes.saturating_add(page.length() as u64))
    }
}

// Completion records remain independent when their Metal storage is recycled.
struct Completion {
    status: MTLCommandBufferStatus,
    error: Option<String>,
}

#[derive(Default)]
struct QueueState {
    pending: Mutex<Vec<Weak<CommandBuffer>>>,
    storage: Mutex<Vec<Submission>>,
}

// Submission storage is exclusively owned while encoding or cached under a lock.
unsafe impl Send for Submission {}

pub(super) struct Queue {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    state: Arc<QueueState>,
}

pub(super) struct CommandBuffer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    submission: Mutex<Option<Submission>>,
    completion: Mutex<Completion>,
    completed: Condvar,
    state: Arc<QueueState>,
}

// Encoding is private to one caller; submitted resources survive through feedback.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

pub(super) struct Encoder {
    encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    command: Command,
}

impl Queue {
    pub(super) fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self, String> {
        let queue = device.newMTL4CommandQueue().ok_or("cannot allocate Metal 4 command queue")?;
        Ok(Self { device: device.clone(), queue, state: Arc::default() })
    }

    pub(super) fn command_buffer(&self) -> Option<Command> {
        let cached = self.state.storage.lock().unwrap_or_else(|error| error.into_inner()).pop();
        let submission = cached.or_else(|| Submission::new(&self.device))?;
        submission.buffer.beginCommandBufferWithAllocator(&submission.allocator);
        submission.buffer.useResidencySet(&submission.residency);
        for slot in 0..6 {
            unsafe { submission.table.setAddress_atIndex(0, slot) };
        }
        Some(Arc::new(CommandBuffer {
            device: self.device.clone(),
            queue: self.queue.clone(),
            submission: Mutex::new(Some(submission)),
            completion: Mutex::new(Completion { status: MTLCommandBufferStatus::NotEnqueued, error: None }),
            completed: Condvar::new(),
            state: Arc::clone(&self.state),
        }))
    }

    pub(super) fn wait_idle(&self) -> Result<(), String> {
        let commands: Vec<_> = self.state.pending.lock().unwrap_or_else(|error| error.into_inner()).iter().filter_map(Weak::upgrade).collect();
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
        let submission = self.submission.lock().unwrap_or_else(|error| error.into_inner());
        let storage = submission.as_ref()?;
        let encoder = storage.buffer.computeCommandEncoder()?;
        encoder.setArgumentTable(Some(&storage.table));
        encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
            MTLStages::Dispatch | MTLStages::Blit,
            MTLStages::Dispatch | MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );
        drop(submission);
        Some(Encoder { encoder, command: Arc::clone(self) })
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
        let mut pending = self.state.pending.lock().unwrap_or_else(|error| error.into_inner());
        if pending.len().is_multiple_of(PENDING_PRUNE_INTERVAL) {
            pending.retain(|command| command.upgrade().is_some_and(|command| command.status() == MTLCommandBufferStatus::Committed));
        }
        pending.push(Arc::downgrade(self));
        drop(pending);
        let submission = self.submission.lock().unwrap_or_else(|error| error.into_inner());
        let storage = submission.as_ref().expect("command storage must exist until completion");
        storage.residency.commit();
        storage.buffer.endCommandBuffer();
        if let Some(error) = self.error() {
            drop(submission);
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
        let mut buffer = NonNull::from(&*storage.buffer);
        // Keep storage locked until commit returns, even if feedback arrives earlier.
        unsafe {
            options.addFeedbackHandler((&raw const *feedback).cast_mut());
            self.queue.commit_count_options(NonNull::from(&mut buffer), 1, &options);
        }
        drop(submission);
    }

    fn complete(&self, error: Option<String>) {
        let storage = self.submission.lock().unwrap_or_else(|error| error.into_inner()).take();
        if let Some(mut storage) = storage {
            storage.recycle();
            if storage.cached_bytes() <= MAX_CACHED_SUBMISSION_BYTES {
                let mut cache = self.state.storage.lock().unwrap_or_else(|error| error.into_inner());
                if cache.len() < MAX_CACHED_SUBMISSIONS {
                    cache.push(storage);
                }
                drop(cache);
            }
        }
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
}

impl Encoder {
    pub(super) fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.encoder.setComputePipelineState(pipeline);
        let mut submission = self.command.submission.lock().unwrap_or_else(|error| error.into_inner());
        submission.as_mut().unwrap().pipelines.entry(std::ptr::from_ref(pipeline) as usize).or_insert_with(|| {
            // The pipeline belongs to the context and remains live during encoding.
            unsafe { Retained::retain(std::ptr::from_ref(pipeline).cast_mut()) }.unwrap()
        });
        drop(submission);
    }

    pub(super) unsafe fn set_buffer(&self, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: usize, slot: usize) {
        let mut submission = self.command.submission.lock().unwrap_or_else(|error| error.into_inner());
        let storage = submission.as_mut().unwrap();
        let address = buffer.map_or(0, |buffer| {
            storage.retain_buffer(buffer);
            buffer.gpuAddress() + offset as u64
        });
        unsafe { storage.table.setAddress_atIndex(address, slot) };
        drop(submission);
    }

    pub(super) unsafe fn set_bytes(&self, bytes: NonNull<std::ffi::c_void>, len: usize, slot: usize) {
        let mut submission = self.command.submission.lock().unwrap_or_else(|error| error.into_inner());
        let storage = submission.as_mut().unwrap();
        // The encoding caller supplies a readable slice of the stated length.
        if let Some(address) = unsafe { storage.copy_controls(&self.command.device, bytes, len) } {
            unsafe { storage.table.setAddress_atIndex(address, slot) };
        } else {
            self.command.encoding_error("cannot allocate Metal 4 control storage".into());
        }
        drop(submission);
    }

    fn barrier(&self) {
        let stages = MTLStages::Dispatch | MTLStages::Blit;
        self.encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(stages, stages, MTL4VisibilityOptions::Device);
    }

    pub(super) fn dispatch_threads(&self, grid: MTLSize, group: MTLSize) {
        if self.command.error().is_none() {
            self.encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
            self.barrier();
        }
    }

    pub(super) fn dispatch_threadgroups(&self, grid: MTLSize, group: MTLSize) {
        if self.command.error().is_none() {
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            self.barrier();
        }
    }

    pub(super) unsafe fn copy_buffer(
        &self, source: &ProtocolObject<dyn MTLBuffer>, source_offset: usize, destination: &ProtocolObject<dyn MTLBuffer>, destination_offset: usize,
        len: usize,
    ) {
        let mut submission = self.command.submission.lock().unwrap_or_else(|error| error.into_inner());
        let storage = submission.as_mut().unwrap();
        storage.retain_buffer(source);
        storage.retain_buffer(destination);
        drop(submission);
        unsafe {
            self.encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(source, source_offset, destination, destination_offset, len)
        };
        self.barrier();
    }

    pub(super) fn end_encoding(&self) {
        self.encoder.endEncoding();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_submission_storage_is_reused_without_changing_completion_records() {
        let Some(context) = super::super::test_context() else { return };
        let command = context.queue.command_buffer().unwrap();
        let identity = {
            let submission = command.submission.lock().unwrap();
            let storage = submission.as_ref().unwrap();
            let identity = (Retained::as_ptr(&storage.buffer), Retained::as_ptr(&storage.allocator), Retained::as_ptr(&storage.table));
            drop(submission);
            identity
        };
        command.commit();
        command.wait_until_completed();
        let next = context.queue.command_buffer().unwrap();
        {
            let submission = next.submission.lock().unwrap();
            let storage = submission.as_ref().unwrap();
            assert_eq!(identity, (Retained::as_ptr(&storage.buffer), Retained::as_ptr(&storage.allocator), Retained::as_ptr(&storage.table)));
            drop(submission);
        }
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(next.status(), MTLCommandBufferStatus::NotEnqueued);
        next.commit();
        next.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }

    #[test]
    fn submission_cache_is_bounded_and_drops_oversized_control_pages() {
        let Some(context) = super::super::test_context() else { return };
        let commands: Vec<_> = (0..MAX_CACHED_SUBMISSIONS + 3).map(|_| context.queue.command_buffer().unwrap()).collect();
        for command in &commands {
            command.commit();
        }
        for command in commands {
            command.wait_until_completed();
        }
        assert_eq!(context.queue.state.storage.lock().unwrap().len(), MAX_CACHED_SUBMISSIONS);
        context.queue.state.storage.lock().unwrap().clear();
        let command = context.queue.command_buffer().unwrap();
        let encoder = command.compute_encoder().unwrap();
        let controls = vec![0u8; MAX_CACHED_SUBMISSION_BYTES as usize + CONTROL_ALIGNMENT];
        unsafe { encoder.set_bytes(NonNull::from(controls.as_slice()).cast(), controls.len(), 3) };
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(context.queue.state.storage.lock().unwrap().is_empty());
    }
}
