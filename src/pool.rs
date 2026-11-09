//! A fork-join thread pool for very short parallel sections.
//!
//! One forward pass is hundreds of parallel sections, each lasting only microseconds. A general
//! purpose pool puts its workers to sleep between sections, and waking them takes longer than
//! the work itself. Here the workers spin, watching a counter, so a section starts within a
//! fraction of a microsecond. They only go to sleep after a few milliseconds with nothing to do.

use std::cell::{Cell, UnsafeCell};
use std::hint::spin_loop;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// How many times a worker checks for new work before going to sleep (a few milliseconds).
const SPINS_BEFORE_SLEEP: u32 = 200_000;

/// Keeps a value on its own cache line (128 bytes on Apple CPUs).
#[repr(align(128))]
struct CacheLine<T>(T);

#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

struct Shared {
    /// Bumped to announce a new job in `job`.
    epoch: CacheLine<AtomicU32>,
    /// How many workers have finished the current job.
    done: CacheLine<AtomicUsize>,
    sleepers: CacheLine<AtomicUsize>,
    job: UnsafeCell<Job>,
    panicked: AtomicBool,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` is written only while no worker is running (between jobs) and read by workers
// only after they see the new `epoch`, which is published with Release ordering. The pointer
// inside stays valid because `broadcast` doesn't return until every worker is done with it.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Not `Sync`: one section at a time, and no sections started from inside a section.
    _not_sync: PhantomData<Cell<()>>,
}

impl Pool {
    /// A pool of `threads` threads, counting the caller of `for_chunks`/`par_for`.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            epoch: CacheLine(AtomicU32::new(0)),
            done: CacheLine(AtomicUsize::new(0)),
            sleepers: CacheLine(AtomicUsize::new(0)),
            job: UnsafeCell::new(Job {
                data: std::ptr::null(),
                call: |_, _| {},
            }),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|index| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("tinyllm-worker-{index}"))
                    .spawn(move || worker(&shared, index))
                    .expect("couldn't start a worker thread")
            })
            .collect();
        Self {
            shared,
            workers,
            _not_sync: PhantomData,
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(thread_index)` once on every thread, returning when they have all finished.
    pub fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        if self.workers.is_empty() {
            f(0);
            return;
        }
        unsafe fn call<F: Fn(usize)>(data: *const (), index: usize) {
            // SAFETY: `data` points to the `F` that `broadcast` keeps alive until workers finish.
            unsafe { (*data.cast::<F>())(index) }
        }
        let s = &*self.shared;
        s.done.0.store(0, Relaxed);
        // SAFETY: every worker has finished the previous job, so none is reading this.
        unsafe {
            *s.job.get() = Job {
                data: (&raw const f).cast(),
                call: call::<F>,
            };
        }
        s.epoch.0.fetch_add(1, SeqCst);
        if s.sleepers.0.load(SeqCst) > 0 {
            let _guard = s.lock.lock().unwrap();
            s.wake.notify_all();
        }
        // Whatever happens to this thread's share, workers must be finished before `f` goes away.
        let result = panic::catch_unwind(AssertUnwindSafe(|| f(0)));
        while s.done.0.load(Acquire) != self.workers.len() {
            spin_loop();
        }
        if let Err(payload) = result {
            panic::resume_unwind(payload);
        }
        assert!(!s.panicked.swap(false, Relaxed), "a worker thread panicked");
    }

    /// Run `f(i)` for every `i` in `0..n`. Threads claim `grain` items at a time, so a slow
    /// thread doesn't hold everyone up. Below `grain` items, it all happens on this thread.
    pub fn par_for<F: Fn(usize) + Sync>(&self, n: usize, grain: usize, f: F) {
        if n <= grain.max(1) || self.workers.is_empty() {
            (0..n).for_each(f);
            return;
        }
        // Claiming bigger pieces when there's plenty to go round keeps each thread reading from
        // one continuous stretch of memory.
        let claim = grain.max(n / (self.threads() * 8)).max(1);
        let next = CacheLine(AtomicUsize::new(0));
        self.broadcast(|_| {
            loop {
                let start = next.0.fetch_add(claim, Relaxed);
                if start >= n {
                    break;
                }
                (start..(start + claim).min(n)).for_each(&f);
            }
        });
    }

    /// Split `data` into chunks of `size` and run `f(index, chunk)` on each, in parallel.
    pub fn for_chunks<T: Send, F>(&self, data: &mut [T], size: usize, grain: usize, f: F)
    where
        F: Fn(usize, &mut [T]) + Sync,
    {
        struct Base<T>(*mut T);
        // SAFETY: threads touch disjoint chunks, and only through `chunk` below.
        unsafe impl<T: Send> Sync for Base<T> {}
        impl<T> Base<T> {
            /// SAFETY: the chunk must be in bounds and handed out only once.
            #[allow(clippy::mut_from_ref)]
            unsafe fn chunk(&self, index: usize, size: usize) -> &mut [T] {
                unsafe { std::slice::from_raw_parts_mut(self.0.add(index * size), size) }
            }
        }
        let n = data.len() / size;
        let base = Base(data.as_mut_ptr());
        // SAFETY: `i < n`, and `par_for` runs each index exactly once.
        self.par_for(n, grain, |i| f(i, unsafe { base.chunk(i, size) }));
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let s = &*self.shared;
        s.shutdown.store(true, SeqCst);
        s.epoch.0.fetch_add(1, SeqCst);
        {
            let _guard = s.lock.lock().unwrap();
            s.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(s: &Shared, index: usize) {
    let mut seen = 0;
    loop {
        let mut spins = 0;
        loop {
            let epoch = s.epoch.0.load(Acquire);
            if epoch != seen {
                seen = epoch;
                break;
            }
            spins += 1;
            if spins < SPINS_BEFORE_SLEEP {
                spin_loop();
                continue;
            }
            // Nothing for a while, so sleep. `broadcast` checks `sleepers` after bumping `epoch`
            // and this checks `epoch` after bumping `sleepers`, so one of them sees the other.
            let mut guard = s.lock.lock().unwrap();
            s.sleepers.0.fetch_add(1, SeqCst);
            while s.epoch.0.load(SeqCst) == seen {
                guard = s.wake.wait(guard).unwrap();
            }
            s.sleepers.0.fetch_sub(1, SeqCst);
            drop(guard);
            spins = 0;
        }
        if s.shutdown.load(SeqCst) {
            return;
        }
        // SAFETY: the new epoch was published after the job was written (Acquire above).
        let job = unsafe { *s.job.get() };
        // SAFETY: `broadcast` keeps the job's data alive until `done` is incremented.
        let result =
            panic::catch_unwind(AssertUnwindSafe(|| unsafe { (job.call)(job.data, index) }));
        if result.is_err() {
            s.panicked.store(true, Relaxed);
        }
        s.done.0.fetch_add(1, Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_for_visits_every_index_once() {
        for threads in [1, 2, 5] {
            let pool = Pool::new(threads);
            for n in [0, 1, 3, 100, 10_000] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                pool.par_for(n, 3, |i| {
                    hits[i].fetch_add(1, Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
            }
        }
    }

    #[test]
    fn for_chunks_gives_each_chunk_its_index() {
        let pool = Pool::new(4);
        let mut data = vec![0.0f32; 300];
        pool.for_chunks(&mut data, 3, 1, |i, chunk| chunk.fill(i as f32));
        for (i, chunk) in data.chunks(3).enumerate() {
            assert_eq!(chunk, [i as f32; 3]);
        }
    }

    #[test]
    fn workers_wake_up_after_sleeping() {
        let pool = Pool::new(3);
        std::thread::sleep(std::time::Duration::from_millis(100));
        let count = AtomicUsize::new(0);
        pool.broadcast(|_| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 3);
    }

    #[test]
    #[should_panic(expected = "a worker thread panicked")]
    fn worker_panics_reach_the_caller() {
        let pool = Pool::new(3);
        pool.broadcast(|i| assert_eq!(i, 0));
    }
}
