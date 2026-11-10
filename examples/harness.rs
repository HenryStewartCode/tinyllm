//! Verification helpers that drive the tinyllm library directly.
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tinyllm::backend::{Backend, CpuBackend, LoadedModel, Logits};
use tinyllm::config::Config;
use tinyllm::cuda::{self, CudaBackend};
use tinyllm::model::{LogitUse, Row};
use tinyllm::tokenizer::Tokenizer;

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A backend chosen by the DEVICE environment variable (cpu or cuda).
fn open(dir: &Path, cache_tokens: usize, max_batch: usize) -> (Box<dyn Backend>, Config) {
    let threads = env_usize("THREADS", 4);
    let config = Config::load(dir).unwrap();
    let kv_dim = config.n_kv_heads() * config.head_dim();
    let bytes = cache_tokens * 2 * config.num_hidden_layers * kv_dim * 4;
    let device = std::env::var("DEVICE").unwrap_or("cpu".into());
    let backend: Box<dyn Backend> = if device == "cuda" {
        let precision = if std::env::var("PRECISION").unwrap_or_default() == "f32" {
            cuda::Precision::F32
        } else {
            cuda::Precision::Bf16
        };
        let quantization = match std::env::var("QUANTIZATION").as_deref() {
            Ok("int8") => Some(cuda::Quantization::Int8),
            _ => None,
        };
        let options = cuda::Options {
            precision,
            quantization,
            max_batch,
            cache_bytes: Some(bytes),
            host_threads: threads,
        };
        let LoadedModel { backend, .. } = CudaBackend::load(dir, &options).unwrap();
        Box::new(backend)
    } else {
        let LoadedModel { backend, .. } = CpuBackend::load(dir, threads, bytes).unwrap();
        Box::new(backend)
    };
    (backend, config)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "tokenize" => {
            let tok = Tokenizer::load(Path::new(&args[2])).unwrap();
            let strings: Vec<String> =
                serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
            let special: Vec<Vec<u32>> = strings.iter().map(|s| tok.encode(s, true)).collect();
            let plain: Vec<Vec<u32>> = strings.iter().map(|s| tok.encode(s, false)).collect();
            let decoded: Vec<String> = plain.iter().map(|ids| tok.decode(ids)).collect();
            println!(
                "{}",
                serde_json::json!({"special": special, "plain": plain, "decoded": decoded})
            );
        }
        // logits for every position, the whole prompt in ONE batch, using scattered KV slots.
        "logits" => {
            let dir = Path::new(&args[2]);
            let ids: Vec<u32> = args[4..].iter().map(|s| s.parse().unwrap()).collect();
            let n = ids.len();
            // Scattered pages, to prove that they needn't be contiguous. Within a page, slots stay in order.
            let pages = n.div_ceil(16) + 8;
            let (mut b, _) = open(dir, pages * 3 * 16 + 16, n.max(8));
            let slots: Vec<u32> = (0..n)
                .map(|i| {
                    let page = (i / 16 * 3 + 5) % (pages * 3);
                    (page * 16 + i % 16) as u32
                })
                .collect();
            let rows: Vec<Row> = (0..n)
                .map(|i| Row {
                    token: ids[i],
                    slots: &slots[..=i],
                    want_logits: true,
                    sampling: LogitUse::All,
                })
                .collect();
            b.attach();
            b.forward(&rows);
            let mut f = std::fs::File::create(&args[3]).unwrap();
            for row in 0..n {
                let Logits::All(l) = b.logits(row) else {
                    panic!()
                };
                for v in l {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
        // Greedy generation one token at a time.
        "greedy" => {
            let dir = Path::new(&args[2]);
            let n: usize = args[3].parse().unwrap();
            let mut ids: Vec<u32> = args[4..].iter().map(|s| s.parse().unwrap()).collect();
            let (mut b, config) = open(dir, ids.len() + n + 1, ids.len().max(8));
            b.attach();
            let vocab = config.vocab_size;
            let slots: Vec<u32> = (0..(ids.len() + n + 1) as u32).collect();
            let plen = ids.len();
            let rows: Vec<Row> = (0..plen)
                .map(|i| Row {
                    token: ids[i],
                    slots: &slots[..=i],
                    want_logits: i + 1 == plen,
                    sampling: LogitUse::All,
                })
                .collect();
            b.forward(&rows);
            let mut out = Vec::new();
            for _ in 0..n {
                let Logits::All(l) = b.logits(0) else {
                    panic!()
                };
                let _ = vocab;
                let t = argmax(l);
                out.push(t);
                ids.push(t);
                let rows = [Row {
                    token: t,
                    slots: &slots[..ids.len()],
                    want_logits: true,
                    sampling: LogitUse::All,
                }];
                b.forward(&rows);
            }
            println!("{}", serde_json::json!(out));
        }
        // Do batched sequences get exactly the results they'd get alone?
        "invariance" => {
            let dir = Path::new(&args[2]);
            let tok = Tokenizer::load(dir).unwrap();
            let prompts = [
                "The capital of France is",
                "def add(a, b):\n    return",
                "Once upon a time, in a land far away, there lived a",
                "2 + 2 =",
                "Rust is a language that",
                "Hello",
                "In 1492,",
                "The quick brown fox jumps over",
                "Water is made of",
                "My favourite colour is",
            ];
            let ids: Vec<Vec<u32>> = prompts.iter().map(|p| tok.encode(p, true)).collect();
            let steps = 12;
            let total: usize = ids.iter().map(|p| p.len() + steps + 1).sum();
            let (mut b, config) = open(dir, total * 3 + 512, 64);
            b.attach();
            let vocab = config.vocab_size;
            let _ = vocab;
            let row = |b: &dyn Backend, i: usize| -> Vec<f32> {
                match b.logits(i) {
                    Logits::All(l) => l.to_vec(),
                    _ => panic!(),
                }
            };
            // Alone, token by token.
            let mut alone: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut base = 0u32;
            for p in &ids {
                let slots: Vec<u32> = (base..base + (p.len() + steps + 1) as u32).collect();
                let mut toks = p.clone();
                let mut got = Vec::new();
                for i in 0..p.len() {
                    b.forward(&[Row {
                        token: p[i],
                        slots: &slots[..=i],
                        want_logits: i + 1 == p.len(),
                        sampling: LogitUse::All,
                    }]);
                }
                for _ in 0..steps {
                    got.push(row(&*b, 0));
                    let t = argmax(&row(&*b, 0));
                    toks.push(t);
                    b.forward(&[Row {
                        token: t,
                        slots: &slots[..toks.len()],
                        want_logits: true,
                        sampling: LogitUse::All,
                    }]);
                }
                alone.push(got);
                base += (slots.len() as u32).next_multiple_of(16);
            }
            // All together: prompts in one batch, then every generation step in one batch.
            let mut next = 0u32;
            let seq_slots: Vec<Vec<u32>> = ids
                .iter()
                .map(|p| {
                    let v: Vec<u32> = (next..next + (p.len() + steps + 1) as u32).collect();
                    next += (v.len() as u32).next_multiple_of(16);
                    v
                })
                .collect();
            let mut toks = ids.clone();
            let rows: Vec<Row> = ids
                .iter()
                .zip(&seq_slots)
                .flat_map(|(p, s)| {
                    (0..p.len()).map(move |i| Row {
                        token: p[i],
                        slots: &s[..=i],
                        want_logits: i + 1 == p.len(),
                        sampling: LogitUse::All,
                    })
                })
                .collect();
            b.forward(&rows);
            let mut mismatches = 0;
            let mut compared = 0;
            let mut worst = 0f32;
            for step in 0..steps {
                for (q, a) in alone.iter().enumerate() {
                    let got = row(&*b, q);
                    for (x, y) in got.iter().zip(&a[step]) {
                        compared += 1;
                        if x.to_bits() != y.to_bits() {
                            mismatches += 1;
                            worst = worst.max((x - y).abs());
                        }
                    }
                    toks[q].push(argmax(&got));
                }
                let rows: Vec<Row> = toks
                    .iter()
                    .zip(&seq_slots)
                    .map(|(t, s)| Row {
                        token: *t.last().unwrap(),
                        slots: &s[..t.len()],
                        want_logits: true,
                        sampling: LogitUse::All,
                    })
                    .collect();
                b.forward(&rows);
            }
            println!(
                "compared {compared} logits: {mismatches} not bit-identical, largest difference {worst:e}"
            );
        }
        // Tokens per second for B sequences generating in lock-step.
        "speed" => {
            let dir = Path::new(&args[2]);
            let batches: Vec<usize> = args[3..].iter().map(|s| s.parse().unwrap()).collect();
            let steps = 48;
            let ctx = (16 + steps + 1usize).next_multiple_of(16);
            let max_b = *batches.iter().max().unwrap();
            let (mut b, config) = open(dir, max_b * ctx, max_b * 16);
            b.attach();
            let vocab = config.vocab_size;
            for &bs in &batches {
                let slots: Vec<Vec<u32>> = (0..bs)
                    .map(|s| ((s * ctx) as u32..((s + 1) * ctx) as u32).collect())
                    .collect();
                // prefill 16 tokens each
                let rows: Vec<Row> = (0..bs)
                    .flat_map(|s| {
                        let sl = &slots[s];
                        (0..16).map(move |i| Row {
                            token: (i as u32 * 7 + s as u32) % 1000 + 100,
                            slots: &sl[..=i],
                            want_logits: i == 15,
                            sampling: LogitUse::All,
                        })
                    })
                    .collect();
                let t0 = Instant::now();
                b.forward(&rows);
                let prefill = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let mut pos = 16;
                for _ in 0..steps {
                    let _ = vocab;
                    let toks: Vec<u32> = (0..bs)
                        .map(|i| match b.logits(i) {
                            Logits::All(l) => argmax(l),
                            Logits::Candidates(c) => {
                                c.iter()
                                    .fold((f32::MIN, 0), |m, x| if x.0 > m.0 { *x } else { m })
                                    .1
                            }
                        })
                        .collect();
                    pos += 1;
                    let sampling = if std::env::var("CANDIDATES").is_ok() {
                        LogitUse::Candidates {
                            gap: 8.0,
                            limit: 40,
                        }
                    } else if std::env::var("DRAW").is_ok() {
                        LogitUse::Draw {
                            temperature: 0.8,
                            key: pos as u64 * 1234567,
                        }
                    } else {
                        LogitUse::All
                    };
                    let rows: Vec<Row> = (0..bs)
                        .map(|s| Row {
                            token: toks[s],
                            slots: &slots[s][..pos],
                            want_logits: true,
                            sampling,
                        })
                        .collect();
                    b.forward(&rows);
                }
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{} batch {bs:4}: {:9.1} tok/s total  {:7.1} tok/s/stream  step {:7.2} ms | prefill {:9.1} tok/s",
                    b.describe(),
                    (bs * steps) as f64 / dt,
                    steps as f64 / dt,
                    dt / steps as f64 * 1e3,
                    (bs * 16) as f64 / prefill
                );
            }
        }
        "long" => {
            let dir = Path::new(&args[2]);
            let n: usize = args[3].parse().unwrap();
            let (mut b, _config) = open(dir, n.next_multiple_of(16) + 16, 512);
            b.attach();
            let slots: Vec<u32> = (0..n as u32).collect();
            let toks: Vec<u32> = (0..n).map(|i| (i as u32 * 7) % 1000 + 100).collect();
            for round in 0..4 {
                let t0 = Instant::now();
                let mut pos = 0;
                while pos < n {
                    let take = (n - pos).min(512);
                    let rows: Vec<Row> = (pos..pos + take)
                        .map(|i| Row {
                            token: toks[i],
                            slots: &slots[..=i],
                            want_logits: i + 1 == n,
                            sampling: LogitUse::All,
                        })
                        .collect();
                    b.forward(&rows);
                    pos += take;
                }
                println!(
                    "round {round}: {n} tokens in {:.1} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        // Do repeat runs of one sequence give identical logits, whatever pages it lands on?
        "repeat" => {
            let dir = Path::new(&args[2]);
            let tok = Tokenizer::load(dir).unwrap();
            let prompt = "Once upon a time, in a land far away, there lived a young dragon who loved to read books about the stars and the sea.";
            let p = tok.encode(prompt, true);
            let steps = 40;
            let total = p.len() + steps + 1;
            let n_pages = total.div_ceil(16);
            let (mut b, _) = open(dir, 16 * n_pages * 40, 64);
            b.attach();
            let run = |b: &mut Box<dyn Backend>,
                       start_page: usize,
                       stride: usize,
                       chunk: usize|
             -> Vec<Vec<f32>> {
                let slots: Vec<u32> = (0..total)
                    .map(|i| (((i / 16) * stride + start_page) * 16 + i % 16) as u32)
                    .collect();
                let rows: Vec<Row> = (0..p.len())
                    .map(|i| Row {
                        token: p[i],
                        slots: &slots[..=i],
                        want_logits: i + 1 == p.len(),
                        sampling: LogitUse::All,
                    })
                    .collect();
                for c in rows.chunks(chunk) {
                    b.forward(c);
                }
                let mut toks = p.clone();
                let mut out = Vec::new();
                for _ in 0..steps {
                    let Logits::All(l) = b.logits(0) else {
                        panic!()
                    };
                    out.push(l.to_vec());
                    let t = argmax(l);
                    toks.push(t);
                    b.forward(&[Row {
                        token: t,
                        slots: &slots[..toks.len()],
                        want_logits: true,
                        sampling: LogitUse::All,
                    }]);
                }
                out
            };
            let reference = run(&mut b, 0, 1, 512);
            // A different sequence dirties the same pages, then the original runs again.
            {
                let other = tok.encode("Totally unrelated words that fill the very same pages with quite different keys and values, on purpose, for a test of stale data.", true);
                let slots: Vec<u32> = (0..total).map(|i| i as u32).collect();
                let mut toks = other.clone();
                let rows: Vec<Row> = (0..other.len())
                    .map(|i| Row {
                        token: other[i],
                        slots: &slots[..=i],
                        want_logits: i + 1 == other.len(),
                        sampling: LogitUse::All,
                    })
                    .collect();
                b.forward(&rows);
                for _ in 0..(total - other.len() - 1) {
                    let Logits::All(l) = b.logits(0) else {
                        panic!()
                    };
                    let t = argmax(l);
                    toks.push(t);
                    b.forward(&[Row {
                        token: t,
                        slots: &slots[..toks.len()],
                        want_logits: true,
                        sampling: LogitUse::All,
                    }]);
                }
            }
            for (name, start, stride, chunk) in [
                ("same pages, now dirty", 0usize, 1usize, 512usize),
                ("same pages", 0usize, 1usize, 512usize),
                ("other pages", 7, 1, 512),
                ("scattered pages", 3, 5, 512),
                ("one token at a time", 0, 1, 1),
                ("chunks of 7", 11, 3, 7),
            ] {
                let got = run(&mut b, start, stride, chunk);
                let (mut diff, mut compared, mut worst) = (0u64, 0u64, 0f32);
                let mut first_bad = None;
                for (i, (x, y)) in got.iter().zip(&reference).enumerate() {
                    for (a, c) in x.iter().zip(y) {
                        compared += 1;
                        if a.to_bits() != c.to_bits() {
                            diff += 1;
                            worst = worst.max((a - c).abs());
                            first_bad.get_or_insert(i);
                        }
                    }
                }
                println!(
                    "{name:22}: {diff}/{compared} logits differ, largest {worst:e}, first at step {first_bad:?}"
                );
            }
        }
        // Raw GPU matrix-kernel speed.
        "gpu-kernel" => {
            use tinyllm::cuda::{Arg, Gpu};
            let gpu = Gpu::open().unwrap();
            gpu.attach().unwrap();
            let module = gpu.load_kernels(cuda::KERNELS_PTX).unwrap();
            for (m, n) in [
                (896usize, 896usize),
                (4864, 896),
                (896, 4864),
                (151936, 896),
            ] {
                for batch in [1usize, 8, 64, 256] {
                    let (name, group) = match batch {
                        1 => ("matmul_1", 1),
                        2 | 3 => ("matmul_2", 2),
                        4..=7 => ("matmul_4", 4),
                        _ => ("matmul_8", 8),
                    };
                    let kernel = gpu.kernel(&module, name).unwrap();
                    let padded = m.div_ceil(8) * 8;
                    let w_host: Vec<f32> = (0..padded * n)
                        .map(|i| {
                            (((i as u64 * 2654435761u64) >> 7) as u32 % 2000) as f32 / 1000.0 - 1.0
                        })
                        .collect();
                    let x_host: Vec<f32> = (0..(batch + 8) * n)
                        .map(|i| (((i as u64 * 40503u64) >> 3) as u32 % 2000) as f32 / 1000.0 - 1.0)
                        .collect();
                    let w = gpu.alloc(w_host.len()).unwrap();
                    let x = gpu.alloc(x_host.len()).unwrap();
                    let out = gpu.alloc(batch * m).unwrap();
                    gpu.write_now(w, &w_host).unwrap();
                    gpu.write_now(x, &x_host).unwrap();
                    let reps = if m > 100000 { 20 } else { 200 };
                    let grid = [batch.div_ceil(group), m.div_ceil(8)];
                    let args = [
                        Arg::Ptr(w),
                        Arg::Int(m),
                        Arg::Int(n),
                        Arg::Ptr(x),
                        Arg::Int(batch),
                        Arg::Ptr(out),
                        Arg::Ptr(cuda::Pointer::NULL),
                        Arg::Int(0),
                    ];
                    for _ in 0..5 {
                        gpu.launch(kernel, grid, &args).unwrap();
                    }
                    gpu.wait().unwrap();
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        gpu.launch(kernel, grid, &args).unwrap();
                    }
                    gpu.wait().unwrap();
                    let dt = t0.elapsed().as_secs_f64();
                    println!(
                        "  W {m}x{n}, batch {batch:4}: {:6.2} T mul-adds/s, {:8.1} us per call",
                        (reps * m * n * batch) as f64 / dt / 1e12,
                        dt / reps as f64 * 1e6
                    );
                    gpu.free(w).unwrap();
                    gpu.free(x).unwrap();
                    gpu.free(out).unwrap();
                }
            }
        }
        // Compute-ceiling micro-tests: every block works on the same small data.
        "ceiling" => {
            use tinyllm::cuda::{Arg, Gpu};
            let gpu = Gpu::open().unwrap();
            gpu.attach().unwrap();
            let ptx = std::fs::read_to_string(&args[2]).unwrap();
            let module = gpu.load_kernels(&ptx).unwrap();
            for spec in &args[3..] {
                let kernel = gpu.kernel(&module, spec).unwrap();
                let data = gpu.alloc(1 << 20).unwrap();
                let host: Vec<f32> = (0..1usize << 20).map(|i| (i % 100) as f32 * 0.01).collect();
                gpu.write_now(data, &host).unwrap();
                let out = gpu.alloc(1 << 20).unwrap();
                let (iters, blocks) = (4096usize, 20000usize);
                let call = [
                    Arg::Ptr(data),
                    Arg::Int(512),
                    Arg::Int(iters),
                    Arg::Ptr(data),
                    Arg::Int(1),
                    Arg::Ptr(out),
                ];
                gpu.launch(kernel, [blocks, 1], &call).unwrap();
                gpu.wait().unwrap();
                let t0 = Instant::now();
                gpu.launch(kernel, [blocks, 1], &call).unwrap();
                gpu.wait().unwrap();
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{spec:8}: {:5.2} T mul-adds/s  ({:.2} ms)",
                    (blocks * 32 * 64 * iters) as f64 / dt / 1e12,
                    dt * 1e3
                );
            }
        }
        // Experimental 16-bit tensor-core kernel: mma <ptx> <name>
        "mma" => {
            use tinyllm::cuda::{Arg, Gpu};
            let gpu = Gpu::open().unwrap();
            gpu.attach().unwrap();
            let ptx = std::fs::read_to_string(&args[2]).unwrap();
            let module = gpu.load_kernels(&ptx).unwrap();
            let kernel = gpu.kernel(&module, &args[3]).unwrap();
            for (m, n) in [(4864usize, 896usize), (896, 4864), (151936, 896)] {
                for batch in [64usize, 256] {
                    let padded = m.div_ceil(64) * 64 + 16;
                    let w_host: Vec<u16> = (0..padded * n)
                        .map(|i| {
                            ((((i as u64 * 2654435761u64) >> 7) as u32 % 2000) as f32 / 1000.0
                                - 1.0)
                                .to_bits()
                                .wrapping_shr(16) as u16
                        })
                        .collect();
                    let x_host: Vec<u16> = (0..(batch + 128) * n)
                        .map(|i| {
                            ((((i as u64 * 40503u64) >> 3) as u32 % 2000) as f32 / 1000.0 - 1.0)
                                .to_bits()
                                .wrapping_shr(16) as u16
                        })
                        .collect();
                    let w = gpu.alloc(w_host.len() / 2 + 1).unwrap();
                    let x = gpu.alloc(x_host.len() / 2 + 1).unwrap();
                    let out = gpu.alloc((batch + 128) * (m + 128)).unwrap();
                    gpu.write_now(w, &w_host).unwrap();
                    gpu.write_now(x, &x_host).unwrap();
                    let reps = if m > 100000 { 20 } else { 200 };
                    let parts: Vec<&str> = args[3].split('_').collect();
                    let (tw, tx): (usize, usize) = if parts[0] == "mt" || parts[0] == "mc" {
                        (parts[1].parse().unwrap(), parts[2].parse().unwrap())
                    } else {
                        (1, 1)
                    };
                    let grid = [batch.div_ceil(16 * tx), m.div_ceil(16 * tw)];
                    let threads = 32 * tw * tx;
                    let call = [
                        Arg::Ptr(w),
                        Arg::Int(m),
                        Arg::Int(n),
                        Arg::Ptr(x),
                        Arg::Int(batch),
                        Arg::Ptr(out),
                    ];
                    for _ in 0..5 {
                        gpu.launch_blocks(kernel, grid, threads, &call).unwrap();
                    }
                    gpu.wait().unwrap();
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        gpu.launch_blocks(kernel, grid, threads, &call).unwrap();
                    }
                    gpu.wait().unwrap();
                    let dt = t0.elapsed().as_secs_f64();
                    println!(
                        "  W {m}x{n}, batch {batch:4}: {:6.2} T mul-adds/s, {:8.1} us per call",
                        (reps * m * n * batch) as f64 / dt / 1e12,
                        dt / reps as f64 * 1e6
                    );
                    gpu.free(w).unwrap();
                    gpu.free(x).unwrap();
                    gpu.free(out).unwrap();
                }
            }
        }
        // Experimental kernels from a PTX file: exp <ptx> <name> <R> <C> ...
        "exp" => {
            use tinyllm::cuda::{Arg, Gpu};
            let gpu = Gpu::open().unwrap();
            gpu.attach().unwrap();
            let ptx = std::fs::read_to_string(&args[2]).unwrap();
            let module = gpu.load_kernels(&ptx).unwrap();
            for spec in &args[3..] {
                let parts: Vec<&str> = spec.split('_').collect();
                let tiled = parts[0] == "t";
                let (a1, a2): (usize, usize) =
                    (parts[1].parse().unwrap(), parts[2].parse().unwrap());
                let (r, c, threads) = if tiled {
                    (8 * a1, 8 * a2, 32 * a1 * a2)
                } else {
                    (a1, a2, 32)
                };
                let kernel = gpu.kernel(&module, spec).unwrap();
                print!("{spec:12}");
                for (m, n, batch) in [
                    (4864usize, 896usize, 64usize),
                    (4864, 896, 256),
                    (896, 4864, 256),
                    (896, 896, 64),
                    (151936, 896, 256),
                ] {
                    let padded = m.div_ceil(r) * r + 8;
                    let w_host: Vec<f32> = (0..padded * n)
                        .map(|i| {
                            (((i as u64 * 2654435761u64) >> 7) as u32 % 2000) as f32 / 1000.0 - 1.0
                        })
                        .collect();
                    let x_host: Vec<f32> = (0..(batch + c + 8) * n)
                        .map(|i| (((i as u64 * 40503u64) >> 3) as u32 % 2000) as f32 / 1000.0 - 1.0)
                        .collect();
                    let w = gpu.alloc(w_host.len()).unwrap();
                    let x = gpu.alloc(x_host.len()).unwrap();
                    let out = gpu.alloc(batch * m).unwrap();
                    gpu.write_now(w, &w_host).unwrap();
                    gpu.write_now(x, &x_host).unwrap();
                    let reps = if m > 100000 { 10 } else { 60 };
                    let grid = [batch.div_ceil(c), m.div_ceil(r)];
                    let call = [
                        Arg::Ptr(w),
                        Arg::Int(m),
                        Arg::Int(n),
                        Arg::Ptr(x),
                        Arg::Int(batch),
                        Arg::Ptr(out),
                    ];
                    for _ in 0..3 {
                        gpu.launch_blocks(kernel, grid, threads, &call).unwrap();
                    }
                    gpu.wait().unwrap();
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        gpu.launch_blocks(kernel, grid, threads, &call).unwrap();
                    }
                    gpu.wait().unwrap();
                    let dt = t0.elapsed().as_secs_f64();
                    print!("  {:5.2}T", (reps * m * n * batch) as f64 / dt / 1e12);
                    gpu.free(w).unwrap();
                    gpu.free(x).unwrap();
                    gpu.free(out).unwrap();
                }
                println!();
            }
        }
        // Render conversations with the model's template. convs.json: [{"messages": [...], "tools": [...]?}]
        "chat" => {
            let dir = Path::new(&args[2]);
            let tok = Tokenizer::load(dir).unwrap();
            let template = tinyllm::chat::ChatTemplate::load(dir)
                .unwrap()
                .expect("no template");
            let convs: Vec<serde_json::Value> =
                serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
            let out: Vec<serde_json::Value> = convs
                .iter()
                .map(|c| {
                    let tools = c.get("tools");
                    let mut messages = c["messages"].as_array().unwrap().clone();
                    if let Err(e) = tinyllm::tools::prepare_messages(&mut messages) {
                        return serde_json::json!({"error": e});
                    }
                    match template.render(&serde_json::Value::Array(messages), tools, None, true) {
                        Ok(text) => {
                            let ids = tinyllm::chat::encode_rendered(&tok, &text);
                            serde_json::json!({"text": text, "ids": ids})
                        }
                        Err(e) => serde_json::json!({"error": e.to_string()}),
                    }
                })
                .collect();
            println!("{}", serde_json::json!(out));
        }
        // Random walks through a grammar's allowed tokens. Prints each accepted output as a JSON string.
        "fuzz" => {
            use tinyllm::grammar::{Grammar, Vocabulary};
            let dir = Path::new(&args[2]);
            let tok = Tokenizer::load(dir).unwrap();
            let vocab_size = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(dir.join("config.json")).unwrap_or("{}".into()),
            )
            .unwrap()["vocab_size"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(32768);
            let vocabulary = Arc::new(Vocabulary::new(&tok, vocab_size));
            let grammar = Grammar::compile(&args[3], vocabulary).unwrap();
            let n: usize = args[4].parse().unwrap();
            let strip = tok.leading_spaces_to_strip() > 0;
            let mut rng = 0x1234_5678_9abc_def1u64;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let stop_id = (0..vocab_size as u32)
                .find(|&t| tok.is_special(t))
                .expect("a special token");
            let mut out = Vec::new();
            let t0 = Instant::now();
            let mut masks = 0u64;
            for _ in 0..n {
                let mut c = grammar.start(strip, vec![stop_id]);
                let mut ids: Vec<u32> = Vec::new();
                let mut closed = false;
                for _ in 0..600 {
                    let mask = c.allowed_tokens();
                    masks += 1;
                    let allowed: Vec<u32> = (0..vocab_size as u32)
                        .filter(|&t| mask.contains(t))
                        .collect();
                    assert!(!allowed.is_empty());
                    let can_stop = mask.contains(stop_id);
                    if can_stop && next() % 3 != 0 {
                        closed = true;
                        break;
                    }
                    let t = allowed[(next() % allowed.len() as u64) as usize];
                    if t == stop_id {
                        closed = true;
                        break;
                    }
                    ids.push(t);
                    c.advance(t);
                }
                let mut bytes = Vec::new();
                for &id in &ids {
                    bytes.extend(tok.token_bytes(id));
                }
                let text = String::from_utf8_lossy(&bytes).into_owned();
                let text = if strip {
                    text.strip_prefix(' ').map(String::from).unwrap_or(text)
                } else {
                    text
                };
                out.push(serde_json::json!({"text": text, "closed": closed, "tokens": ids.len(), "valid_utf8": std::str::from_utf8(&bytes).is_ok()}));
            }
            eprintln!(
                "{} walks, {} masks in {:.2}s ({:.1} us per mask)",
                n,
                masks,
                t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64() * 1e6 / masks as f64
            );
            println!("{}", serde_json::json!(out));
        }
        // Random walks through tool-call grammars, read back by the tool parser.
        "fuzz_tools" => {
            use tinyllm::grammar::{Grammar, Vocabulary};
            use tinyllm::tools::{self, ToolFormat, ToolOptions, ToolOutput};
            let dir = Path::new(&args[2]);
            let tok = Tokenizer::load(dir).unwrap();
            let vocab_size = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(dir.join("config.json")).unwrap_or("{}".into()),
            )
            .unwrap()["vocab_size"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(32768);
            let request: serde_json::Value = serde_json::from_str(&args[3]).unwrap();
            let format = match args[4].as_str() {
                "hermes" => Some(ToolFormat::Hermes),
                "mistral" => Some(ToolFormat::Mistral),
                "compact" => Some(ToolFormat::MistralCompact),
                "llama3" => Some(ToolFormat::Llama3),
                "none" => None,
                other => panic!("{other}"),
            };
            let n: usize = args[5].parse().unwrap();
            let plan = tools::plan(ToolOptions {
                tools: request.get("tools"),
                tool_choice: request.get("tool_choice"),
                parallel_tool_calls: request
                    .get("parallel")
                    .and_then(|p| p.as_bool())
                    .unwrap_or(true),
                format,
                output_constrained: false,
            })
            .unwrap();
            let reading = plan.parser.clone().unwrap();
            let markers = reading.format.marker_tokens(&tok);
            let vocabulary = Arc::new(Vocabulary::with_markers(&tok, vocab_size, &markers));
            let spec = format!(
                "calls:{}",
                serde_json::to_string(plan.grammar.as_ref().unwrap()).unwrap()
            );
            let grammar = Grammar::compile(&spec, vocabulary).unwrap();
            let strip = tok.leading_spaces_to_strip() > 0;
            let mut rng = 0x1234_5678_9abc_def1u64;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let stop_id = (0..vocab_size as u32)
                .find(|&t| tok.is_special(t) && !markers.contains(&t))
                .expect("a special token");
            let mut out = Vec::new();
            for _ in 0..n {
                let mut c = grammar.start(strip, vec![stop_id]);
                let mut parser = reading.parser();
                let mut outputs = Vec::new();
                let mut text = String::new();
                let mut pending: Vec<u8> = Vec::new();
                let mut first = true;
                let mut closed = false;
                let mut tokens = 0;
                let mut used_markers = 0;
                for _ in 0..400 {
                    let mask = c.allowed_tokens();
                    let allowed: Vec<u32> = (0..vocab_size as u32)
                        .filter(|&t| mask.contains(t))
                        .collect();
                    assert!(!allowed.is_empty());
                    if mask.contains(stop_id) && next() % 3 != 0 {
                        closed = true;
                        break;
                    }
                    // Markers would be the model's choice nearly every time, so favour them too.
                    let wanted: Vec<u32> = allowed
                        .iter()
                        .copied()
                        .filter(|t| markers.contains(t))
                        .collect();
                    let pool = if !wanted.is_empty() && next() % 4 != 0 {
                        &wanted
                    } else {
                        &allowed
                    };
                    let t = pool[(next() % pool.len() as u64) as usize];
                    if t == stop_id {
                        closed = true;
                        break;
                    }
                    used_markers += markers.contains(&t) as usize;
                    tokens += 1;
                    c.advance(t);
                    let mut bytes = tok.token_bytes(t);
                    if first {
                        first = false;
                        if strip && bytes.first() == Some(&b' ') {
                            bytes.remove(0);
                        }
                    }
                    pending.extend(bytes);
                    // Only whole characters go on, as in the engine.
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    let piece = String::from_utf8(pending.drain(..valid).collect()).unwrap();
                    text.push_str(&piece);
                    outputs.extend(parser.push(&piece));
                }
                outputs.extend(parser.finish());
                let mut content = String::new();
                let mut calls: Vec<(String, String)> = Vec::new();
                for o in outputs {
                    match o {
                        ToolOutput::Content(t) => content.push_str(&t),
                        ToolOutput::Call { name, .. } => calls.push((name, String::new())),
                        ToolOutput::Arguments { index, text } => calls[index].1.push_str(&text),
                    }
                }
                out.push(serde_json::json!({"text": text, "closed": closed, "tokens": tokens, "calls": calls, "content": content, "markers": used_markers}));
            }
            println!("{}", serde_json::json!(out));
        }
        _ => panic!("unknown command"),
    }
}
