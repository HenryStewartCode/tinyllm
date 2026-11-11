//! The parts of Apple's Metal API that tinyllm uses, called through the Objective-C runtime.
//!
//! Metal is an Objective-C API: everything is a message sent to an object with `objc_msgSend`,
//! which has to be called through a function pointer of the right type for each message. That's
//! all done here, so the rest of the backend sees plain Rust. Metal is part of every Mac, so
//! nothing needs installing, and Apple's GPUs share memory with the CPU, so buffers are plain
//! memory that both sides read and write.

use anyhow::{Result, bail};
use std::ffi::{CStr, c_char, c_void};

#[repr(C)]
pub struct Object {
    _private: [u8; 0],
}
type Id = *mut Object;

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Sel(*const c_void);

#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    width: usize,
    height: usize,
    depth: usize,
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_release(object: Id);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

fn sel(name: &CStr) -> Sel {
    // SAFETY: a valid C string.
    unsafe { sel_registerName(name.as_ptr()) }
}

/// Send a message: `msg!(object, c"selector:", arg => Type, ...; ReturnType)`.
macro_rules! msg {
    ($object:expr, $selector:expr $(, $arg:expr => $ty:ty)* ; $ret:ty) => {{
        // Each place that sends a message looks its selector up once.
        static SELECTOR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let selector = Sel(*SELECTOR.get_or_init(|| sel($selector).0 as usize) as *const c_void);
        // SAFETY: objc_msgSend takes the receiver and selector, then the message's own arguments,
        // with the C calling convention of a function of exactly that type.
        let send: unsafe extern "C" fn(Id, Sel $(, $ty)*) -> $ret =
            unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
        unsafe { send($object, selector $(, $arg)*) }
    }};
}

/// An NSString, released when dropped.
struct NsString(Id);

impl NsString {
    fn new(text: &str) -> Self {
        // SAFETY: NSString is a class in Foundation, which is linked.
        let class = unsafe { objc_getClass(c"NSString".as_ptr()) };
        let blank = msg!(class, c"alloc"; Id);
        let ptr = text.as_ptr().cast::<c_void>();
        // NSUTF8StringEncoding is 4.
        Self(
            msg!(blank, c"initWithBytes:length:encoding:", ptr => *const c_void, text.len() => usize, 4usize => usize; Id),
        )
    }
}

impl Drop for NsString {
    fn drop(&mut self) {
        // SAFETY: made by alloc/init, so this owns one reference.
        unsafe { objc_release(self.0) }
    }
}

/// The text of an NSString or nil.
fn string(object: Id) -> String {
    if object.is_null() {
        return String::new();
    }
    let text = msg!(object, c"UTF8String"; *const c_char);
    if text.is_null() {
        return String::new();
    }
    // SAFETY: UTF8String gives a NUL-terminated string that lives as long as the object.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// What an NSError says.
fn describe(error: Id) -> String {
    match error.is_null() {
        true => "unknown error".to_string(),
        false => string(msg!(error, c"localizedDescription"; Id)),
    }
}

/// A Metal object we own one reference to, released when dropped.
struct Owned(Id);

// SAFETY: Metal's devices, queues, pipelines and buffers can be used from any thread.
unsafe impl Send for Owned {}
unsafe impl Sync for Owned {}

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: we own one reference.
            unsafe { objc_release(self.0) }
        }
    }
}

/// The GPU, and the queue its work goes into.
pub struct Device {
    device: Owned,
    queue: Owned,
}

/// Compiled kernels.
pub struct Library(Owned);

/// One kernel, ready to run.
pub struct Pipeline {
    state: Owned,
    /// The most threads a group of this kernel can have.
    pub max_threads: usize,
}

/// GPU memory that the CPU can read and write too.
pub struct Buffer {
    buffer: Owned,
    contents: *mut u8,
    pub bytes: usize,
}

// SAFETY: the memory belongs to the buffer, which can be used from any thread; callers keep the
// CPU and GPU from using it at the same time.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// The memory, as elements of `T`.
    ///
    /// # Safety
    /// The GPU mustn't be using the buffer, and `T` must be plain data.
    // GPU memory is written through a shared handle; the caller vouches that nothing else is
    // using it, as with any memory the GPU and CPU share.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut<T>(&self) -> &mut [T] {
        // SAFETY: the caller promises nothing else is using the memory.
        unsafe { std::slice::from_raw_parts_mut(self.contents.cast(), self.bytes / size_of::<T>()) }
    }

    /// The memory, as elements of `T`, for reading.
    ///
    /// # Safety
    /// The GPU mustn't be writing to the buffer, and `T` must be plain data.
    pub unsafe fn slice<T>(&self) -> &[T] {
        // SAFETY: the caller promises nothing is writing to the memory.
        unsafe { std::slice::from_raw_parts(self.contents.cast(), self.bytes / size_of::<T>()) }
    }
}

impl Device {
    /// The Mac's GPU, if it has one Metal can use.
    pub fn open() -> Result<Self> {
        // SAFETY: Metal is linked; this returns nil if there's no device.
        let device = unsafe { MTLCreateSystemDefaultDevice() };
        if device.is_null() {
            bail!("no Metal GPU");
        }
        let queue = msg!(device, c"newCommandQueue"; Id);
        if queue.is_null() {
            bail!("couldn't make a Metal command queue");
        }
        Ok(Self {
            device: Owned(device),
            queue: Owned(queue),
        })
    }

    pub fn name(&self) -> String {
        string(msg!(self.device.0, c"name"; Id))
    }

    /// How much memory the GPU should use, by Metal's reckoning.
    /// The largest buffer the GPU can have, in bytes.
    pub fn max_buffer_bytes(&self) -> usize {
        msg!(self.device.0, c"maxBufferLength"; usize)
    }

    pub fn recommended_memory(&self) -> usize {
        msg!(self.device.0, c"recommendedMaxWorkingSetSize"; u64) as usize
    }

    /// Compile Metal Shading Language source.
    pub fn compile(&self, source: &str) -> Result<Library> {
        let source = NsString::new(source);
        // SAFETY: MTLCompileOptions is a class in Metal, which is linked.
        let class = unsafe { objc_getClass(c"MTLCompileOptions".as_ptr()) };
        let options = Owned(msg!(class, c"new"; Id));
        // Fast math would let the compiler reorder arithmetic and approximate division and
        // square roots, so results would drift from the CPU's.
        msg!(options.0, c"setFastMathEnabled:", false => bool; ());
        let mut error: Id = std::ptr::null_mut();
        let library = msg!(self.device.0, c"newLibraryWithSource:options:error:",
            source.0 => Id, options.0 => Id, &mut error => *mut Id; Id);
        if library.is_null() {
            bail!("couldn't compile the Metal kernels: {}", describe(error));
        }
        Ok(Library(Owned(library)))
    }

    pub fn pipeline(&self, library: &Library, name: &str) -> Result<Pipeline> {
        let ns_name = NsString::new(name);
        let function = msg!(library.0.0, c"newFunctionWithName:", ns_name.0 => Id; Id);
        if function.is_null() {
            bail!("no Metal kernel called {name}");
        }
        let function = Owned(function);
        let mut error: Id = std::ptr::null_mut();
        let state = msg!(self.device.0, c"newComputePipelineStateWithFunction:error:",
            function.0 => Id, &mut error => *mut Id; Id);
        if state.is_null() {
            bail!(
                "couldn't set up the Metal kernel {name}: {}",
                describe(error)
            );
        }
        let max_threads = msg!(state, c"maxTotalThreadsPerThreadgroup"; usize);
        Ok(Pipeline {
            state: Owned(state),
            max_threads,
        })
    }

    /// A zeroed buffer of `bytes` bytes, shared between the CPU and GPU.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(16);
        // MTLResourceStorageModeShared is 0.
        let buffer = msg!(self.device.0, c"newBufferWithLength:options:", bytes => usize, 0usize => usize; Id);
        if buffer.is_null() {
            bail!("couldn't allocate {} MiB of GPU memory", bytes >> 20);
        }
        let contents = msg!(buffer, c"contents"; *mut c_void).cast::<u8>();
        Ok(Buffer {
            buffer: Owned(buffer),
            contents,
            bytes,
        })
    }

    /// Record kernels with `record`, run them, and wait for them to finish. Returns how long the
    /// GPU took, in seconds. Each kernel sees what the ones before it wrote.
    pub fn run(&self, record: impl FnOnce(&Encoder) -> Result<()>) -> Result<f64> {
        // SAFETY: matched by the pop below; the command buffer and encoder are autoreleased.
        let pool = unsafe { objc_autoreleasePoolPush() };
        let result = (|| {
            let commands = msg!(self.queue.0, c"commandBuffer"; Id);
            let encoder = msg!(commands, c"computeCommandEncoder"; Id);
            if commands.is_null() || encoder.is_null() {
                bail!("couldn't start Metal work");
            }
            let recorded = record(&Encoder(encoder));
            msg!(encoder, c"endEncoding"; ());
            msg!(commands, c"commit"; ());
            msg!(commands, c"waitUntilCompleted"; ());
            recorded?;
            let error = msg!(commands, c"error"; Id);
            if !error.is_null() {
                bail!("the Metal GPU failed: {}", describe(error));
            }
            let start = msg!(commands, c"GPUStartTime"; f64);
            let end = msg!(commands, c"GPUEndTime"; f64);
            Ok(end - start)
        })();
        // SAFETY: the pool pushed above.
        unsafe { objc_autoreleasePoolPop(pool) };
        result
    }
}

/// Where kernels get queued, one after another; each sees what the ones before it wrote.
pub struct Encoder(Id);

impl Encoder {
    pub fn set_pipeline(&self, pipeline: &Pipeline) {
        msg!(self.0, c"setComputePipelineState:", pipeline.state.0 => Id; ());
    }

    pub fn set_buffer(&self, index: usize, buffer: &Buffer, offset_bytes: usize) {
        msg!(self.0, c"setBuffer:offset:atIndex:", buffer.buffer.0 => Id, offset_bytes => usize, index => usize; ());
    }

    /// Pass a small value (up to 4 KB) directly.
    pub fn set_bytes<T: Copy>(&self, index: usize, value: &T) {
        let ptr = (value as *const T).cast::<c_void>();
        msg!(self.0, c"setBytes:length:atIndex:", ptr => *const c_void, size_of::<T>() => usize, index => usize; ());
    }

    /// Threadgroup memory for argument `index`.
    pub fn set_threadgroup_memory(&self, index: usize, bytes: usize) {
        msg!(self.0, c"setThreadgroupMemoryLength:atIndex:", bytes.next_multiple_of(16) => usize, index => usize; ());
    }

    /// Run `groups` threadgroups of `threads` threads each.
    pub fn dispatch(&self, groups: [usize; 2], threads: usize) {
        self.dispatch_3d([groups[0], groups[1], 1], threads);
    }

    /// `dispatch`, over a grid of threadgroups in three dimensions.
    pub fn dispatch_3d(&self, groups: [usize; 3], threads: usize) {
        let groups = Size {
            width: groups[0],
            height: groups[1],
            depth: groups[2],
        };
        let threads = Size {
            width: threads,
            height: 1,
            depth: 1,
        };
        msg!(self.0, c"dispatchThreadgroups:threadsPerThreadgroup:", groups => Size, threads => Size; ());
    }
}
