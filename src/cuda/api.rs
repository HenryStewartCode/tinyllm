//! The parts of NVIDIA's CUDA driver API that tinyllm uses.
//!
//! The library is loaded when the program runs rather than linked at build time, so the same
//! binary works on machines with no NVIDIA GPU, and building it needs no CUDA toolkit.

use anyhow::{Context as _, Result, anyhow, bail};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_ulonglong, c_void};
use std::marker::PhantomData;

type Status = c_int;
type Handle = *mut c_void;

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> Handle;
    fn dlsym(library: Handle, name: *const c_char) -> Handle;
}

/// `RTLD_NOW`
#[cfg(unix)]
const LOAD_NOW: c_int = 2;

macro_rules! driver_api {
    ($($name:ident($($arg:ty),*);)*) => {
        #[allow(non_snake_case)]
        struct Driver {
            $($name: unsafe extern "C" fn($($arg),*) -> Status,)*
            cuGetErrorName: unsafe extern "C" fn(Status, *mut *const c_char) -> Status,
        }

        impl Driver {
            #[cfg(unix)]
            fn load() -> Result<Self> {
                // SAFETY: loading the driver's library runs its initialization, and these are
                // its documented entry points with the signatures it documents.
                unsafe {
                    let library = dlopen(c"libcuda.so.1".as_ptr(), LOAD_NOW);
                    if library.is_null() {
                        bail!("the NVIDIA driver's libcuda.so.1 isn't available");
                    }
                    let symbol = |name: &CStr| -> Result<Handle> {
                        let found = dlsym(library, name.as_ptr());
                        if found.is_null() {
                            bail!("the NVIDIA driver has no {}", name.to_string_lossy());
                        }
                        Ok(found)
                    };
                    Ok(Self {
                        $($name: std::mem::transmute::<
                            Handle,
                            unsafe extern "C" fn($($arg),*) -> Status,
                        >(symbol(CString::new(stringify!($name))?.as_c_str())?),)*
                        cuGetErrorName: std::mem::transmute::<
                            Handle,
                            unsafe extern "C" fn(Status, *mut *const c_char) -> Status,
                        >(symbol(c"cuGetErrorName")?),
                    })
                }
            }

            #[cfg(not(unix))]
            fn load() -> Result<Self> {
                bail!("the CUDA backend only supports Linux so far");
            }
        }
    };
}

driver_api! {
    cuInit(c_uint);
    cuDeviceGet(*mut c_int, c_int);
    cuDeviceGetName(*mut c_char, c_int, c_int);
    cuDeviceTotalMem_v2(*mut usize, c_int);
    cuCtxCreate_v2(*mut Handle, c_uint, c_int);
    cuCtxSetCurrent(Handle);
    cuCtxDestroy_v2(Handle);
    cuMemGetInfo_v2(*mut usize, *mut usize);
    cuMemAlloc_v2(*mut u64, usize);
    cuMemFree_v2(u64);
    cuMemsetD8_v2(u64, u8, usize);
    cuMemcpyHtoD_v2(u64, *const c_void, usize);
    cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle);
    cuMemcpyDtoHAsync_v2(*mut c_void, u64, usize, Handle);
    cuStreamCreate(*mut Handle, c_uint);
    cuStreamSynchronize(Handle);
    cuModuleLoadDataEx(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void);
    cuModuleGetFunction(*mut Handle, Handle, *const c_char);
    cuLaunchKernel(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut c_void);
    cuStreamBeginCapture_v2(Handle, c_int);
    cuStreamEndCapture(Handle, *mut Handle);
    cuGraphInstantiateWithFlags(*mut Handle, Handle, c_ulonglong);
    cuGraphLaunch(Handle, Handle);
    cuGraphDestroy(Handle);
    cuGraphExecDestroy(Handle);
}

/// An address in GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Pointer(pub u64);

impl Pointer {
    pub const NULL: Pointer = Pointer(0);

    /// The address `elements` floats (or other 4-byte values) further along.
    pub fn offset(self, elements: usize) -> Pointer {
        Pointer(self.0 + (elements * 4) as u64)
    }
}

/// A kernel in a loaded module.
#[derive(Clone, Copy)]
pub struct Kernel(Handle);

// SAFETY: a kernel handle is just a reference to something owned by the GPU context.
unsafe impl Send for Kernel {}

/// A recorded sequence of kernels, ready to be replayed.
pub struct Graph {
    graph: Handle,
    exec: Handle,
}

// SAFETY: like a kernel, a graph is a reference to something owned by the GPU context.
unsafe impl Send for Graph {}

/// One argument to a kernel.
#[derive(Clone, Copy)]
pub enum Arg {
    Ptr(Pointer),
    Int(usize),
    Float(f32),
}

/// A GPU, with a context and a queue of work.
///
/// CUDA contexts belong to whichever thread made them current, so all use of this after moving
/// it to another thread has to follow a call to [`attach`](Self::attach).
pub struct Gpu {
    driver: Driver,
    context: Handle,
    stream: Handle,
    device: c_int,
    /// A GPU context can only be used from one thread at a time.
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: a CUDA context can be moved between threads, as long as it's made current on the new
// one, which `attach` does. `Gpu` isn't `Sync`, so it's only ever used from one thread at a time.
unsafe impl Send for Gpu {}

impl Gpu {
    /// Open the first GPU that `CUDA_VISIBLE_DEVICES` allows.
    pub fn open() -> Result<Self> {
        let driver = Driver::load()?;
        let mut gpu = Self {
            driver,
            context: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            device: 0,
            _not_sync: PhantomData,
        };
        // SAFETY: each call follows its documented contract, with pointers to live variables.
        unsafe {
            gpu.check((gpu.driver.cuInit)(0))
                .context("no usable NVIDIA GPU")?;
            let mut device = 0;
            gpu.check((gpu.driver.cuDeviceGet)(&mut device, 0))
                .context("no usable NVIDIA GPU")?;
            gpu.device = device;
            let mut context = std::ptr::null_mut();
            gpu.check((gpu.driver.cuCtxCreate_v2)(&mut context, 0, device))?;
            gpu.context = context;
            let mut stream = std::ptr::null_mut();
            gpu.check((gpu.driver.cuStreamCreate)(&mut stream, 0))?;
            gpu.stream = stream;
        }
        Ok(gpu)
    }

    fn check(&self, status: Status) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let mut name = std::ptr::null();
        // SAFETY: on success `name` points to a static string.
        unsafe {
            if (self.driver.cuGetErrorName)(status, &mut name) == 0 && !name.is_null() {
                return Err(anyhow!(
                    "CUDA error: {}",
                    CStr::from_ptr(name).to_string_lossy()
                ));
            }
        }
        Err(anyhow!("CUDA error {status}"))
    }

    /// Make this thread the one using the GPU.
    pub fn attach(&self) -> Result<()> {
        // SAFETY: `context` is a live context.
        self.check(unsafe { (self.driver.cuCtxSetCurrent)(self.context) })
    }

    pub fn name(&self) -> String {
        let mut buffer = [0 as c_char; 256];
        // SAFETY: the driver writes a NUL-terminated string that fits the given length.
        unsafe {
            if (self.driver.cuDeviceGetName)(buffer.as_mut_ptr(), 256, self.device) != 0 {
                return "unknown NVIDIA GPU".to_string();
            }
            CStr::from_ptr(buffer.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Bytes of GPU memory that are free right now.
    pub fn free_memory(&self) -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: both pointers are to live variables.
        self.check(unsafe { (self.driver.cuMemGetInfo_v2)(&mut free, &mut total) })?;
        Ok(free)
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut total = 0;
        // SAFETY: the pointer is to a live variable, and `device` is a valid ordinal.
        self.check(unsafe { (self.driver.cuDeviceTotalMem_v2)(&mut total, self.device) })?;
        Ok(total)
    }

    /// Allocate GPU memory for `elements` floats.
    pub fn alloc(&self, elements: usize) -> Result<Pointer> {
        let mut pointer = 0;
        let bytes = (elements * 4).max(4);
        // SAFETY: the pointer is to a live variable.
        self.check(unsafe { (self.driver.cuMemAlloc_v2)(&mut pointer, bytes) })
            .with_context(|| format!("couldn't allocate {} MiB of GPU memory", bytes >> 20))?;
        Ok(Pointer(pointer))
    }

    pub fn free(&self, pointer: Pointer) -> Result<()> {
        // SAFETY: only pointers from `alloc` are passed, and the stream is idle when it's used.
        self.check(unsafe { (self.driver.cuMemFree_v2)(pointer.0) })
    }

    /// Fill `elements` floats with zeros.
    pub fn zero(&self, pointer: Pointer, elements: usize) -> Result<()> {
        // SAFETY: the caller's pointer is to at least `elements` floats.
        self.check(unsafe { (self.driver.cuMemsetD8_v2)(pointer.0, 0, elements * 4) })
    }

    /// Copy to the GPU, returning when it's done.
    pub fn write_now<T: Copy>(&self, to: Pointer, data: &[T]) -> Result<()> {
        // SAFETY: `data` is a valid slice, and the caller's pointer has room for it.
        self.check(unsafe {
            (self.driver.cuMemcpyHtoD_v2)(to.0, data.as_ptr().cast(), size_of_val(data))
        })
    }

    /// Queue a copy to the GPU. `data` has to stay put until the queue has been waited for.
    pub fn write<T: Copy>(&self, to: Pointer, data: &[T]) -> Result<()> {
        // SAFETY: the caller keeps `data` alive and unmoved until `wait` returns.
        self.check(unsafe {
            (self.driver.cuMemcpyHtoDAsync_v2)(
                to.0,
                data.as_ptr().cast(),
                size_of_val(data),
                self.stream,
            )
        })
    }

    /// Queue a copy from the GPU. `data` isn't valid until the queue has been waited for.
    pub fn read<T: Copy>(&self, from: Pointer, data: &mut [T]) -> Result<()> {
        // SAFETY: the caller keeps `data` alive and unmoved until `wait` returns.
        self.check(unsafe {
            (self.driver.cuMemcpyDtoHAsync_v2)(
                data.as_mut_ptr().cast(),
                from.0,
                size_of_val(data),
                self.stream,
            )
        })
    }

    /// Wait for everything queued so far to finish.
    pub fn wait(&self) -> Result<()> {
        // SAFETY: `stream` is a live stream.
        self.check(unsafe { (self.driver.cuStreamSynchronize)(self.stream) })
    }

    /// Compile a module of kernels from PTX source.
    pub fn load_kernels(&self, ptx: &str) -> Result<Module> {
        let source = CString::new(ptx)?;
        let mut module = std::ptr::null_mut();
        // SAFETY: `source` is NUL-terminated PTX, and no JIT options are passed.
        self.check(unsafe {
            (self.driver.cuModuleLoadDataEx)(
                &mut module,
                source.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        })
        .context("couldn't compile the GPU kernels")?;
        Ok(Module(module))
    }

    pub fn kernel(&self, module: &Module, name: &str) -> Result<Kernel> {
        let name = CString::new(name)?;
        let mut function = std::ptr::null_mut();
        // SAFETY: `module` is a live module, and `name` is NUL-terminated.
        self.check(unsafe {
            (self.driver.cuModuleGetFunction)(&mut function, module.0, name.as_ptr())
        })
        .with_context(|| format!("no GPU kernel called {name:?}"))?;
        Ok(Kernel(function))
    }

    /// Queue a kernel, with `grid` blocks of 32 threads. `args` must match the kernel's
    /// parameters exactly. Pointers in it have to stay allocated until the queue is waited for.
    pub fn launch(&self, kernel: Kernel, grid: [usize; 2], args: &[Arg]) -> Result<()> {
        self.launch_with(kernel, grid, 32, 0, args)
    }

    /// Start recording. Kernels launched from here on go into the graph instead of running.
    pub fn begin_recording(&self) -> Result<()> {
        // SAFETY: `stream` is a live stream that isn't being captured already; mode 0 is the
        // default capture mode.
        self.check(unsafe { (self.driver.cuStreamBeginCapture_v2)(self.stream, 0) })
    }

    /// Stop recording, and get a graph that can be replayed.
    pub fn end_recording(&self) -> Result<Graph> {
        let (mut graph, mut exec) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: capture was begun on `stream` by this thread; the out-pointers are live.
        unsafe {
            self.check((self.driver.cuStreamEndCapture)(self.stream, &mut graph))?;
            self.check((self.driver.cuGraphInstantiateWithFlags)(
                &mut exec, graph, 0,
            ))?;
        }
        Ok(Graph { graph, exec })
    }

    /// Queue a replay of a recorded graph.
    pub fn launch_graph(&self, graph: &Graph) -> Result<()> {
        // SAFETY: `graph` holds a live executable graph, and `stream` is a live stream.
        self.check(unsafe { (self.driver.cuGraphLaunch)(graph.exec, self.stream) })
    }

    /// Dispose of a graph. It must not be in flight.
    pub fn destroy_graph(&self, graph: Graph) -> Result<()> {
        // SAFETY: the caller has waited for the stream, so nothing is using the graph.
        unsafe {
            self.check((self.driver.cuGraphExecDestroy)(graph.exec))?;
            self.check((self.driver.cuGraphDestroy)(graph.graph))
        }
    }

    /// `launch`, with a different number of threads per block.
    pub fn launch_blocks(
        &self,
        kernel: Kernel,
        grid: [usize; 2],
        threads: usize,
        args: &[Arg],
    ) -> Result<()> {
        self.launch_with(kernel, grid, threads, 0, args)
    }

    /// `launch`, giving each block `shared_bytes` of on-chip memory.
    pub fn launch_shared(
        &self,
        kernel: Kernel,
        grid: [usize; 2],
        shared_bytes: usize,
        args: &[Arg],
    ) -> Result<()> {
        self.launch_with(kernel, grid, 32, shared_bytes, args)
    }

    /// `launch`, with `threads` threads per block and `shared_bytes` of on-chip memory.
    pub fn launch_with_shared(
        &self,
        kernel: Kernel,
        grid: [usize; 2],
        threads: usize,
        shared_bytes: usize,
        args: &[Arg],
    ) -> Result<()> {
        self.launch_with(kernel, grid, threads, shared_bytes, args)
    }

    fn launch_with(
        &self,
        kernel: Kernel,
        grid: [usize; 2],
        threads: usize,
        shared_bytes: usize,
        args: &[Arg],
    ) -> Result<()> {
        self.launch_grid(kernel, [grid[0], grid[1], 1], threads, shared_bytes, args)
    }

    /// Launch `kernel` over a grid of blocks in three dimensions, with `threads` threads per
    /// block and `shared_bytes` of on-chip memory.
    pub fn launch_grid(
        &self,
        kernel: Kernel,
        grid: [usize; 3],
        threads: usize,
        shared_bytes: usize,
        args: &[Arg],
    ) -> Result<()> {
        const MAX_ARGS: usize = 20;
        assert!(args.len() <= MAX_ARGS, "too many kernel arguments");
        let mut values = [0u64; MAX_ARGS];
        let mut pointers = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (i, arg) in args.iter().enumerate() {
            values[i] = match *arg {
                Arg::Ptr(pointer) => pointer.0,
                Arg::Int(value) => value as u64,
                Arg::Float(value) => value.to_bits() as u64,
            };
            pointers[i] = (&raw mut values[i]).cast();
        }
        // SAFETY: the kernel handle is live, and `args` matches its parameters as documented
        // above. The driver copies the argument values before this returns.
        self.check(unsafe {
            (self.driver.cuLaunchKernel)(
                kernel.0,
                grid[0] as c_uint,
                grid[1] as c_uint,
                grid[2] as c_uint,
                threads as c_uint,
                1,
                1,
                shared_bytes as c_uint,
                self.stream,
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
}

/// A set of compiled kernels. They last as long as the GPU context.
pub struct Module(Handle);

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: nothing can be using the context, since `self` is going away.
        unsafe {
            (self.driver.cuCtxDestroy_v2)(self.context);
        }
    }
}
