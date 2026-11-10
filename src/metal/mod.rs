//! Running the model on an Apple GPU with Metal.
//!
//! This does the same work as the CPU forward pass in [`crate::model`], one kernel per step. The
//! kernels are in `kernels.metal`, compiled when the model is loaded. Apple's GPUs share memory
//! with the CPU, so the weights, the KV cache and each step's inputs and outputs are buffers that
//! both can use: nothing is copied back and forth, and the CPU reads the logits where the GPU
//! left them. Weights stay as they're stored (bfloat16 or 32-bit) and the arithmetic is 32-bit
//! throughout, like the CPU's, so the two agree to within rounding.

mod api;

use crate::backend::{Backend, LoadedModel, Logits};
use crate::config::{Activation, Architecture, Config};
use crate::lora::{LoraAdapter, LoraPair, Module};
use crate::model::{LogitUse, Row};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Result, ensure};
use api::{Buffer, Device, Encoder, Library, Pipeline};
use std::cell::Cell;
use std::path::Path;

const KERNELS: &str = include_str!("kernels.metal");

/// Threads that run in lockstep on Apple GPUs.
const SIMD: usize = 32;
/// Weight rows each simdgroup of the small-batch matrix kernel takes (`VEC_ROWS`).
const VEC_ROWS: usize = 4;
/// Rows of the batch it takes at once (`VEC_BATCH`).
const VEC_BATCH: usize = 8;
/// Threads per group for it.
const VEC_THREADS: usize = 256;
/// Rows of the batch and weight rows in each tile of the big-batch matrix kernel (`MM_BN` and
/// `MM_BM`), and its threads per group (`MM_THREADS`).
const MM_BN: usize = 32;
const MM_BM: usize = 64;
const MM_THREADS: usize = 128;
/// The big-batch matrix kernel is used for batches at least this big, when it has at least
/// `MMA_MIN_TILES` tiles to share out (it's slower when it can't keep the GPU busy). Both
/// kernels give exactly the same results, so this only affects speed.
const MMA_MIN_BATCH: usize = 16;
const MMA_MIN_TILES: usize = 32;
/// Threads per group for the kernels that go through the elements of a row.
const ROW_THREADS: usize = 256;
/// Threads per group for the RMS norms (`NORM_THREADS` in `kernels.metal`).
const NORM_THREADS: usize = 256;
/// The largest head `kernels.metal` has room for (`MAX_HEAD_DIM`).
const MAX_HEAD_DIM: usize = 512;
/// Simdgroups per attention threadgroup (`ATT_SIMDS` in `kernels.metal`).
const ATT_SIMDS: usize = 8;
/// Memory to leave for everything else when the KV cache would take more than Metal suggests.
const RESERVED_BYTES: usize = 1 << 30;

#[derive(Clone, Debug)]
pub struct Options {
    /// Memory for the KV cache, in bytes.
    pub cache_bytes: usize,
    /// CPU threads for work between GPU steps (choosing the next tokens).
    pub host_threads: usize,
}

// The kernels' arguments, laid out as in `kernels.metal`.

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedArgs {
    dim: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormArgs {
    dim: u32,
    eps: f32,
    select: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeadNormArgs {
    stride: u32,
    n_q: u32,
    head_dim: u32,
    k_offset: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulArgs {
    m: u32,
    n: u32,
    batch: u32,
    scale: f32,
    accumulate: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeArgs {
    stride: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    q_dim: u32,
    kv_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnArgs {
    stride: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    q_dim: u32,
    scale: f32,
    softcap: f32,
    window: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GatedArgs {
    hidden: u32,
    gelu: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LoraArgs {
    in_dim: u32,
    out_dim: u32,
    rank: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LogitArgs {
    count: u32,
    divisor: f32,
    softcap: f32,
}

struct Kernels {
    embed_bf16: Pipeline,
    embed_f32: Pipeline,
    rmsnorm: Pipeline,
    rmsnorm_add: Pipeline,
    head_norm: Pipeline,
    matmul_vec_bf16: Pipeline,
    matmul_vec_f32: Pipeline,
    matmul_mma_bf16: Pipeline,
    matmul_mma_f32: Pipeline,
    rope_store: Pipeline,
    attention: Pipeline,
    gated: Pipeline,
    finish_logits: Pipeline,
    lora_shrink: Pipeline,
    lora_expand: Pipeline,
}

impl Kernels {
    fn find(device: &Device, library: &Library) -> Result<Self> {
        let get = |name| device.pipeline(library, name);
        Ok(Self {
            embed_bf16: get("embed_bf16")?,
            embed_f32: get("embed_f32")?,
            rmsnorm: get("rmsnorm")?,
            rmsnorm_add: get("rmsnorm_add")?,
            head_norm: get("head_norm")?,
            matmul_vec_bf16: get("matmul_vec_bf16")?,
            matmul_vec_f32: get("matmul_vec_f32")?,
            matmul_mma_bf16: get("matmul_mma_bf16")?,
            matmul_mma_f32: get("matmul_mma_f32")?,
            rope_store: get("rope_store")?,
            attention: get("attention")?,
            gated: get("gated")?,
            finish_logits: get("finish_logits")?,
            lora_shrink: get("lora_shrink")?,
            lora_expand: get("lora_expand")?,
        })
    }
}

/// A weight matrix `[rows, cols]`, stored by rows, and what goes with it.
struct Matrix {
    data: Buffer,
    /// The weights are bfloat16 rather than 32-bit.
    bf16: bool,
    rows: usize,
    cols: usize,
    bias: Option<Buffer>,
    /// What every output gets multiplied by.
    scale: f32,
}

struct Layer {
    attn_norm: Buffer,
    /// q_proj, k_proj and v_proj as one matrix, since they all take the same input.
    qkv: Matrix,
    /// For normalizing each head of the queries and of the keys, if the model does that.
    head_norms: Option<(Buffer, Buffer)>,
    o_proj: Matrix,
    /// Normalizes what attention adds to the residual stream, if the model does that.
    norm_after_attn: Option<Buffer>,
    mlp_norm: Buffer,
    /// gate_proj and up_proj as one matrix, for the same reason as `qkv`.
    gate_up: Matrix,
    down: Matrix,
    /// The same as `norm_after_attn`, for the MLP.
    norm_after_mlp: Option<Buffer>,
    k_pool: Buffer,
    v_pool: Buffer,
    rope_table: usize,
    /// How many tokens it can see.
    window: usize,
}

/// The sizes the working memory depends on.
#[derive(Clone, Copy)]
struct Shape {
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
    /// Half a head, the length of a row of RoPE rotations.
    half: usize,
    rope_tables: usize,
}

/// Working memory for a batch of up to `rows` rows. A bigger batch gets a bigger one.
struct Workspace {
    rows: usize,
    /// The residual stream `[rows, dim]`.
    x: Buffer,
    /// A block's normalized input `[rows, dim]`.
    xb: Buffer,
    /// What a block adds to the stream, before it's normalized `[rows, dim]`.
    added: Buffer,
    /// `[rows, q_dim + 2 * kv_dim]`
    qkv: Buffer,
    /// `[rows, q_dim]`
    attn_out: Buffer,
    /// `[rows, 2 * hidden_dim]`
    gate_up: Buffer,
    /// `[rows, hidden_dim]`
    gated: Buffer,
    // What describes the batch, one entry per row:
    tokens: Buffer,
    write_slots: Buffer,
    /// Where the row's visible KV slots start in the step's slot lists.
    row_offset: Buffer,
    /// How many there are.
    row_len: Buffer,
    /// The rows that want logits.
    logit_rows: Buffer,
    /// Each row's LoRA adapter, 0 for none.
    adapters: Buffer,
    /// RoPE rotations for each row's position `[tables, batch, half]`.
    cos: Buffer,
    sin: Buffer,
}

impl Workspace {
    fn new(device: &Device, shape: Shape, rows: usize) -> Result<Self> {
        let floats = |count: usize| device.buffer(rows * count * 4);
        Ok(Self {
            rows,
            x: floats(shape.dim)?,
            xb: floats(shape.dim)?,
            added: floats(shape.dim)?,
            qkv: floats(shape.q_dim + 2 * shape.kv_dim)?,
            attn_out: floats(shape.q_dim)?,
            gate_up: floats(2 * shape.hidden_dim)?,
            gated: floats(shape.hidden_dim)?,
            tokens: floats(1)?,
            write_slots: floats(1)?,
            row_offset: floats(1)?,
            row_len: floats(1)?,
            logit_rows: floats(1)?,
            adapters: floats(1)?,
            cos: floats(shape.rope_tables * shape.half)?,
            sin: floats(shape.rope_tables * shape.half)?,
        })
    }
}

/// What a step's batch asks for, once its description is in place.
struct Step {
    batch: usize,
    /// How many rows want logits.
    n_logits: usize,
    /// Some of them want scores.
    scores: bool,
    /// Some of them want their final hidden state.
    hidden: bool,
    /// Some row uses a LoRA adapter.
    any_adapter: bool,
}

/// Reads weights out of a checkpoint into GPU buffers, counting them.
struct Loader<'a> {
    device: &'a Device,
    st: &'a SafeTensors,
    norm_offset: f32,
    params: Cell<usize>,
}

impl Loader<'_> {
    fn upload<T: Copy>(&self, values: &[T]) -> Result<Buffer> {
        let buffer = self.device.buffer(size_of_val(values))?;
        // SAFETY: the GPU hasn't seen this buffer yet, and T is plain data.
        let memory = unsafe { buffer.slice_mut::<T>() };
        memory[..values.len()].copy_from_slice(values);
        Ok(buffer)
    }

    /// A norm's weights. If `optional` ones are missing, they do nothing but normalize.
    fn norm(&self, name: &str, len: usize, optional: bool) -> Result<Buffer> {
        let values: Vec<f32> = match optional && !self.st.contains(name) {
            true => vec![1.0; len],
            // Gemma's norm weights are stored as an offset from 1.
            false => (self.st.load(name, &[len])?.into_iter())
                .map(|w| w + self.norm_offset)
                .collect(),
        };
        self.params.set(self.params.get() + len);
        self.upload(&values)
    }

    /// Matrices with `cols` columns, `(name, rows)` each, stacked into one, with every output
    /// multiplied by `scale`. They stay bfloat16 if they all are.
    fn matrix(&self, parts: &[(String, usize)], cols: usize, scale: f32) -> Result<Matrix> {
        let rows: usize = parts.iter().map(|(_, rows)| rows).sum();
        let bf16 = parts
            .iter()
            .all(|(name, _)| self.st.dtype(&format!("{name}.weight")) == Some("BF16"));
        let data = self.device.buffer(rows * cols * if bf16 { 2 } else { 4 })?;
        let mut at = 0;
        for (name, part_rows) in parts {
            let (name, shape) = (format!("{name}.weight"), [*part_rows, cols]);
            let range = at * cols..(at + part_rows) * cols;
            // SAFETY: the GPU hasn't seen this buffer yet.
            match bf16 {
                true => {
                    let memory = unsafe { data.slice_mut::<u16>() };
                    memory[range].copy_from_slice(&self.st.load_bf16(&name, &shape)?);
                }
                false => {
                    let memory = unsafe { data.slice_mut::<f32>() };
                    memory[range].copy_from_slice(&self.st.load(&name, &shape)?);
                }
            }
            at += part_rows;
        }
        let biased = parts
            .iter()
            .any(|(name, _)| self.st.contains(&format!("{name}.bias")));
        let bias = match biased {
            false => None,
            true => {
                let mut values = vec![0.0; rows];
                let mut at = 0;
                for (name, part_rows) in parts {
                    let name = format!("{name}.bias");
                    if self.st.contains(&name) {
                        // The kernel scales the product, then adds the bias.
                        let bias = self.st.load(&name, &[*part_rows])?;
                        for (to, b) in values[at..at + part_rows].iter_mut().zip(bias) {
                            *to = b * scale;
                        }
                    }
                    at += part_rows;
                }
                Some(self.upload(&values)?)
            }
        };
        self.params
            .set(self.params.get() + rows * cols + if biased { rows } else { 0 });
        Ok(Matrix {
            data,
            bf16,
            rows,
            cols,
            bias,
            scale,
        })
    }
}

pub struct MetalBackend {
    device: Device,
    kernels: Kernels,
    embed: Matrix,
    /// The output projection, unless it's the embeddings ("tied embeddings").
    lm_head: Option<Matrix>,
    norm: Buffer,
    layers: Vec<Layer>,
    architecture: Architecture,
    shape: Shape,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    eps: f32,
    capacity: usize,
    ws: Workspace,
    /// The KV slots each sequence in the batch can see, one list after another.
    slot_lists: Buffer,
    /// `[rows that want logits, vocab]`
    logits: Buffer,
    /// Final hidden states `[rows that want logits, dim]`, when any want them.
    hidden: Buffer,
    pool: Pool,
    gpu_name: String,
    /// LoRA adapters' corrections as given, for each layer and module, the first being adapter 1.
    adapters: Vec<Vec<[Option<LoraPair>; 7]>>,
    /// The same on the GPU, for each layer and each of `LORA_TARGETS`.
    lora: Vec<[Option<LoraTarget>; 4]>,
    /// Working space for `A x`, `[rows, largest rank]`.
    lora_hidden: Buffer,
    /// The largest rank of any adapter's correction.
    lora_rank: usize,
}

/// The matrices LoRA corrections go to, as stacked here, and the modules each stacks.
const LORA_TARGETS: [&[Module]; 4] = [
    &[Module::Q, Module::K, Module::V],
    &[Module::O],
    &[Module::Gate, Module::Up],
    &[Module::Down],
];

/// Every adapter's correction to one stacked matrix: `A` as `[adapters, rank, in]` and `B` as
/// `[adapters, out, rank]`, padded with zeros to the largest rank.
struct LoraTarget {
    a: Buffer,
    b: Buffer,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
}

impl MetalBackend {
    /// Whether there's a GPU Metal can use.
    pub fn available() -> bool {
        Device::open().is_ok()
    }

    /// Load `config.json` and the weights.
    pub fn load(dir: &Path, options: &Options) -> Result<LoadedModel<Self>> {
        let config = Config::load(dir)?;
        let architecture = config.architecture()?;
        ensure!(
            architecture.experts.is_none(),
            "mixtures of experts don't run on Metal yet"
        );
        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.n_kv_heads();
        let head_dim = config.head_dim();
        let hidden_dim = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        ensure!(
            head_dim <= MAX_HEAD_DIM && head_dim.is_multiple_of(4),
            "the Metal kernels need heads of at most {MAX_HEAD_DIM} elements, in fours"
        );
        ensure!(
            [dim, q_dim, hidden_dim]
                .iter()
                .all(|width| width.is_multiple_of(8)),
            "the Metal kernels need layer widths that are multiples of 8"
        );

        let device = Device::open()?;
        let gpu_name = device.name();
        let kernels = Kernels::find(&device, &device.compile(KERNELS)?)?;
        ensure!(
            [&kernels.matmul_mma_bf16, &kernels.matmul_mma_f32]
                .iter()
                .all(|kernel| kernel.max_threads >= MM_THREADS),
            "this GPU can't run the Metal matrix kernel"
        );
        ensure!(
            kernels.attention.max_threads >= ATT_SIMDS * SIMD
                && kernels.rmsnorm.max_threads >= NORM_THREADS
                && kernels.rmsnorm_add.max_threads >= NORM_THREADS,
            "this GPU can't run the Metal attention and norm kernels"
        );
        let st = SafeTensors::open(dir)?;
        let loader = Loader {
            device: &device,
            st: &st,
            norm_offset: architecture.norm_offset,
            params: Cell::new(0),
        };
        let residual = architecture.residual_scale;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (l, shape) in architecture.layers.iter().enumerate() {
            let p = format!("model.layers.{l}");
            let attn = format!("{p}.self_attn");
            let head_norms = match (architecture.head_norms, architecture.full_qk_norms) {
                (true, _) => Some((
                    loader.norm(&format!("{attn}.q_norm.weight"), head_dim, true)?,
                    loader.norm(&format!("{attn}.k_norm.weight"), head_dim, true)?,
                )),
                // A whole token's queries, and keys, normalized together (OLMoE).
                (_, true) => Some((
                    loader.norm(&format!("{attn}.q_norm.weight"), q_dim, true)?,
                    loader.norm(&format!("{attn}.k_norm.weight"), kv_dim, true)?,
                )),
                _ => None,
            };
            let norm = |name: &str| loader.norm(&format!("{p}.{name}.weight"), dim, false);
            let (norm_after_attn, mlp_norm, norm_after_mlp) = match architecture.norms_after_blocks
            {
                true => (
                    Some(norm("post_attention_layernorm")?),
                    norm("pre_feedforward_layernorm")?,
                    Some(norm("post_feedforward_layernorm")?),
                ),
                // Elsewhere this name means the norm in front of the MLP.
                false => (None, norm("post_attention_layernorm")?, None),
            };
            let (qkv_names, gate_up_names) = match architecture.combined_projections {
                true => (
                    vec![(format!("{attn}.qkv_proj"), q_dim + 2 * kv_dim)],
                    vec![(format!("{p}.mlp.gate_up_proj"), 2 * hidden_dim)],
                ),
                false => (
                    vec![
                        (format!("{attn}.q_proj"), q_dim),
                        (format!("{attn}.k_proj"), kv_dim),
                        (format!("{attn}.v_proj"), kv_dim),
                    ],
                    vec![
                        (format!("{p}.mlp.gate_proj"), hidden_dim),
                        (format!("{p}.mlp.up_proj"), hidden_dim),
                    ],
                ),
            };
            layers.push(Layer {
                attn_norm: norm("input_layernorm")?,
                qkv: loader.matrix(&qkv_names, dim, 1.0)?,
                head_norms,
                o_proj: loader.matrix(&[(format!("{attn}.o_proj"), dim)], q_dim, residual)?,
                norm_after_attn,
                mlp_norm,
                gate_up: loader.matrix(&gate_up_names, dim, 1.0)?,
                down: loader.matrix(
                    &[(format!("{p}.mlp.down_proj"), dim)],
                    hidden_dim,
                    residual,
                )?,
                norm_after_mlp,
                // Allocated below, once the weights' size is known.
                k_pool: device.buffer(0)?,
                v_pool: device.buffer(0)?,
                rope_table: shape.rope_table,
                window: shape.window.unwrap_or(u32::MAX as usize),
            });
        }
        let embed = loader.matrix(&[("model.embed_tokens".into(), vocab_size)], dim, 1.0)?;
        let lm_head = match st.contains("lm_head.weight") {
            true => Some(loader.matrix(&[("lm_head".into(), vocab_size)], dim, 1.0)?),
            false => None,
        };
        let norm = loader.norm("model.norm.weight", dim, false)?;
        let num_params = loader.params.get();

        // The KV cache gets what was asked for, as long as Metal thinks it'll fit.
        let weight_bytes = |m: &Matrix| m.data.bytes;
        let weights = weight_bytes(&embed)
            + lm_head.as_ref().map_or(0, weight_bytes)
            + (layers.iter())
                .map(|l| {
                    [&l.qkv, &l.o_proj, &l.gate_up, &l.down]
                        .map(weight_bytes)
                        .iter()
                        .sum::<usize>()
                })
                .sum::<usize>();
        let room = device
            .recommended_memory()
            .saturating_sub(weights + RESERVED_BYTES);
        let cache_bytes = match options.cache_bytes > room {
            true => {
                eprintln!(
                    "Only {} MiB of the GPU's memory is left for the KV cache.",
                    room >> 20
                );
                room
            }
            false => options.cache_bytes,
        };
        let kv_bytes_per_token = 2 * layers.len() * kv_dim * 4;
        let capacity = (cache_bytes / kv_bytes_per_token).clamp(16, u32::MAX as usize);
        for layer in &mut layers {
            layer.k_pool = device.buffer(capacity * kv_dim * 4)?;
            layer.v_pool = device.buffer(capacity * kv_dim * 4)?;
        }

        let shape = Shape {
            dim,
            q_dim,
            kv_dim,
            hidden_dim,
            half: head_dim / 2,
            rope_tables: architecture.rope_tables.len(),
        };
        let ws = Workspace::new(&device, shape, 16)?;
        Ok(LoadedModel {
            num_params,
            backend: Self {
                slot_lists: device.buffer(1 << 16)?,
                lora_hidden: device.buffer(16)?,
                logits: device.buffer(vocab_size * 4)?,
                hidden: device.buffer(dim * 4)?,
                ws,
                device,
                kernels,
                embed,
                lm_head,
                norm,
                layers,
                architecture,
                shape,
                n_heads,
                n_kv_heads,
                head_dim,
                vocab_size,
                eps: config.rms_norm_eps,
                capacity,
                pool: Pool::new(options.host_threads),
                gpu_name,
                adapters: Vec::new(),
                lora: Vec::new(),
                lora_rank: 0,
            },
            config,
        })
    }

    /// Put the description of the batch where the kernels will look for it.
    fn write_step(&mut self, rows: &[Row]) -> Result<Step> {
        let batch = rows.len();
        if batch > self.ws.rows {
            self.ws = Workspace::new(&self.device, self.shape, batch.next_power_of_two())?;
        }
        // Rows from one sequence are next to each other and share a list of slots: the longest
        // one, belonging to its last row in this batch.
        let mut groups = Vec::new();
        let mut r = 0;
        while r < batch {
            let start = rows[r].slots.as_ptr();
            let group = (rows[r..].iter())
                .take_while(|row| row.slots.as_ptr() == start)
                .count();
            groups.push(r..r + group);
            r += group;
        }
        let total: usize = groups.iter().map(|g| rows[g.end - 1].slots.len()).sum();
        reserve(&self.device, &mut self.slot_lists, total * 4)?;

        let ws = &self.ws;
        // SAFETY: the GPU finished the last step before `forward` returned, and these are all
        // different buffers.
        let (tokens, write_slots, row_offset, row_len, logit_rows, cos, sin, lists) = unsafe {
            (
                ws.tokens.slice_mut::<u32>(),
                ws.write_slots.slice_mut::<u32>(),
                ws.row_offset.slice_mut::<u32>(),
                ws.row_len.slice_mut::<u32>(),
                ws.logit_rows.slice_mut::<u32>(),
                ws.cos.slice_mut::<f32>(),
                ws.sin.slice_mut::<f32>(),
                self.slot_lists.slice_mut::<u32>(),
            )
        };
        // SAFETY: as above.
        let adapters = unsafe { ws.adapters.slice_mut::<u32>() };
        let half = self.shape.half;
        let mut step = Step {
            batch,
            n_logits: 0,
            scores: false,
            hidden: false,
            any_adapter: false,
        };
        let mut offset = 0;
        for group in groups {
            let visible = rows[group.end - 1].slots;
            lists[offset..offset + visible.len()].copy_from_slice(visible);
            for index in group {
                let row = &rows[index];
                assert!(
                    (row.token as usize) < self.vocab_size,
                    "token id {} is outside the vocabulary",
                    row.token
                );
                tokens[index] = row.token;
                assert!(
                    (row.adapter as usize) <= self.adapters.len(),
                    "there's no adapter {}",
                    row.adapter
                );
                adapters[index] = row.adapter as u32;
                step.any_adapter |= row.adapter != 0;
                write_slots[index] = *row.slots.last().expect("a row needs a KV slot");
                row_offset[index] = offset as u32;
                row_len[index] = row.slots.len() as u32;
                if row.want_logits {
                    logit_rows[step.n_logits] = index as u32;
                    step.n_logits += 1;
                    match row.sampling {
                        LogitUse::Hidden => step.hidden = true,
                        _ => step.scores = true,
                    }
                }
                let position = (row.slots.len() - 1) as f32;
                for (table, frequencies) in self.architecture.rope_tables.iter().enumerate() {
                    let scale = self.architecture.rope_scales[table];
                    let start = (table * batch + index) * half;
                    for (j, &freq) in frequencies.iter().enumerate() {
                        let angle = position * freq;
                        cos[start + j] = angle.cos() * scale;
                        sin[start + j] = angle.sin() * scale;
                    }
                }
            }
            offset += visible.len();
        }
        Ok(step)
    }

    fn try_forward(&mut self, rows: &[Row]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let step = self.write_step(rows)?;
        if step.scores {
            reserve(
                &self.device,
                &mut self.logits,
                step.n_logits * self.vocab_size * 4,
            )?;
        }
        if step.hidden {
            reserve(
                &self.device,
                &mut self.hidden,
                step.n_logits * self.shape.dim * 4,
            )?;
        }
        if step.any_adapter {
            let bytes = rows.len() * self.lora_rank * 4;
            reserve(&self.device, &mut self.lora_hidden, bytes)?;
        }
        self.device.run(|e| {
            self.encode(e, &step, None);
            Ok(())
        })?;
        Ok(())
    }

    /// How long the GPU spends on each kind of kernel in a forward pass of `rows`, and on the
    /// whole pass, in seconds (the best of a few runs). The batch's keys and values are written
    /// as usual.
    #[doc(hidden)]
    pub fn profile(&mut self, rows: &[Row]) -> Result<Vec<(String, f64)>> {
        self.try_forward(rows)?;
        let step = self.write_step(rows)?;
        let mut times = Vec::new();
        for part in [
            None,
            Some(Part::Matmuls),
            Some(Part::Attention),
            Some(Part::Norms),
            Some(Part::Rope),
            Some(Part::Other),
            Some(Part::Logits),
        ] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let seconds = self.device.run(|e| {
                    self.encode(e, &step, part);
                    Ok(())
                })?;
                best = best.min(seconds);
            }
            let name = part.map_or("whole step".to_string(), |part| format!("{part:?}"));
            times.push((name, best));
        }
        Ok(times)
    }

    /// Queue the whole forward pass, or (for profiling) only one kind of kernel in it.
    fn encode(&self, e: &Encoder, step: &Step, only: Option<Part>) {
        let (ws, k, shape) = (&self.ws, &self.kernels, self.shape);
        let on = |part: Part| only.is_none_or(|only| only == part);
        let batch = step.batch;
        let embed = &self.embed;
        let pipeline = if embed.bf16 {
            &k.embed_bf16
        } else {
            &k.embed_f32
        };
        e.set_pipeline(pipeline);
        e.set_buffer(0, &embed.data, 0);
        e.set_buffer(1, &ws.tokens, 0);
        e.set_buffer(2, &ws.x, 0);
        e.set_bytes(
            3,
            &EmbedArgs {
                dim: shape.dim as u32,
                scale: self.architecture.embedding_scale,
            },
        );
        if on(Part::Other) {
            e.dispatch([batch, 1], threads(pipeline, ROW_THREADS));
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block: mix in information from earlier tokens.
            if on(Part::Norms) {
                self.rmsnorm(e, &ws.x, &ws.xb, &layer.attn_norm, batch, false);
            }
            if on(Part::Matmuls) {
                self.matmul(e, &layer.qkv, &ws.xb, &ws.qkv, batch, false);
                self.lora(e, step, l, 0, &ws.xb, &ws.qkv);
            }
            if let Some((q_weight, k_weight)) = &layer.head_norms
                && on(Part::Norms)
            {
                self.head_norms(e, q_weight, k_weight, batch);
            }
            if on(Part::Rope) {
                self.rope_and_store(e, layer, batch);
            }
            if on(Part::Attention) {
                self.attention(e, layer, batch);
            }
            if on(Part::Matmuls) {
                self.add_block_output(
                    e,
                    &layer.o_proj,
                    &ws.attn_out,
                    batch,
                    layer.norm_after_attn.as_ref(),
                    (step, l, 1),
                );
            }

            // MLP block: process each token on its own.
            if on(Part::Norms) {
                self.rmsnorm(e, &ws.x, &ws.xb, &layer.mlp_norm, batch, false);
            }
            if on(Part::Matmuls) {
                self.matmul(e, &layer.gate_up, &ws.xb, &ws.gate_up, batch, false);
                self.lora(e, step, l, 2, &ws.xb, &ws.gate_up);
            }
            if !on(Part::Other) {
                if on(Part::Matmuls) {
                    self.add_block_output(
                        e,
                        &layer.down,
                        &ws.gated,
                        batch,
                        layer.norm_after_mlp.as_ref(),
                        (step, l, 3),
                    );
                }
                continue;
            }
            e.set_pipeline(&k.gated);
            e.set_buffer(0, &ws.gate_up, 0);
            e.set_buffer(1, &ws.gated, 0);
            e.set_bytes(
                2,
                &GatedArgs {
                    hidden: shape.hidden_dim as u32,
                    gelu: (self.architecture.activation == Activation::Gelu) as u32,
                },
            );
            let threads = threads(&k.gated, ROW_THREADS);
            e.dispatch([shape.hidden_dim.div_ceil(threads), batch], threads);
            if on(Part::Matmuls) {
                self.add_block_output(
                    e,
                    &layer.down,
                    &ws.gated,
                    batch,
                    layer.norm_after_mlp.as_ref(),
                    (step, l, 3),
                );
            }
        }

        // Only the rows that want logits go through the last norm and the output projection.
        if !on(Part::Logits) {
            return;
        }
        let n_logits = step.n_logits;
        if step.hidden {
            self.rmsnorm(e, &ws.x, &self.hidden, &self.norm, n_logits, true);
        }
        if step.scores {
            self.rmsnorm(e, &ws.x, &ws.xb, &self.norm, n_logits, true);
            let lm_head = self.lm_head.as_ref().unwrap_or(&self.embed);
            self.matmul(e, lm_head, &ws.xb, &self.logits, n_logits, false);
            let (divisor, softcap) = (
                self.architecture.logit_divisor,
                self.architecture.logit_softcap,
            );
            if divisor != 1.0 || softcap != 0.0 {
                let count = n_logits * self.vocab_size;
                e.set_pipeline(&k.finish_logits);
                e.set_buffer(0, &self.logits, 0);
                e.set_bytes(
                    1,
                    &LogitArgs {
                        count: count as u32,
                        divisor,
                        softcap,
                    },
                );
                let threads = threads(&k.finish_logits, ROW_THREADS);
                e.dispatch([count.div_ceil(threads), 1], threads);
            }
        }
    }

    /// Queue `out = W x` for `batch` rows of `x`, or `out += W x` if `accumulate`.
    fn matmul(
        &self,
        e: &Encoder,
        w: &Matrix,
        x: &Buffer,
        out: &Buffer,
        batch: usize,
        accumulate: bool,
    ) {
        let tiles = w.rows.div_ceil(MM_BM) * batch.div_ceil(MM_BN);
        let kernel = match batch >= MMA_MIN_BATCH && tiles >= MMA_MIN_TILES {
            true => MatmulKernel::Mma,
            false => MatmulKernel::Vec,
        };
        encode_matmul(e, &self.kernels, w, x, out, batch, accumulate, kernel);
    }

    /// Queue `out = rmsnorm(x) * weight` for `rows` rows, which are the rows that want logits
    /// if `select`.
    fn rmsnorm(
        &self,
        e: &Encoder,
        x: &Buffer,
        out: &Buffer,
        weight: &Buffer,
        rows: usize,
        select: bool,
    ) {
        e.set_pipeline(&self.kernels.rmsnorm);
        e.set_buffer(0, x, 0);
        e.set_buffer(1, out, 0);
        e.set_buffer(2, weight, 0);
        e.set_buffer(3, &self.ws.logit_rows, 0);
        e.set_bytes(
            4,
            &NormArgs {
                dim: self.shape.dim as u32,
                eps: self.eps,
                select: select as u32,
            },
        );
        e.dispatch([rows, 1], NORM_THREADS);
    }

    /// Queue `x += W input`, normalizing what's added first if the model does that.
    fn add_block_output(
        &self,
        e: &Encoder,
        w: &Matrix,
        input: &Buffer,
        batch: usize,
        norm: Option<&Buffer>,
        (step, layer, target): (&Step, usize, usize),
    ) {
        let ws = &self.ws;
        let Some(norm) = norm else {
            self.matmul(e, w, input, &ws.x, batch, true);
            return self.lora(e, step, layer, target, input, &ws.x);
        };
        self.matmul(e, w, input, &ws.added, batch, false);
        self.lora(e, step, layer, target, input, &ws.added);
        e.set_pipeline(&self.kernels.rmsnorm_add);
        e.set_buffer(0, &ws.added, 0);
        e.set_buffer(1, &ws.x, 0);
        e.set_buffer(2, norm, 0);
        e.set_bytes(
            3,
            &NormArgs {
                dim: self.shape.dim as u32,
                eps: self.eps,
                select: 0,
            },
        );
        e.dispatch([batch, 1], NORM_THREADS);
    }

    /// Queue the LoRA corrections to stacked matrix `target` (one of `LORA_TARGETS`) of `layer`:
    /// `out += B (A input)` for each row using an adapter.
    fn lora(
        &self,
        e: &Encoder,
        step: &Step,
        layer: usize,
        target: usize,
        input: &Buffer,
        out: &Buffer,
    ) {
        if !step.any_adapter {
            return;
        }
        let Some(Some(t)) = self.lora.get(layer).map(|targets| &targets[target]) else {
            return;
        };
        let k = &self.kernels;
        let args = LoraArgs {
            in_dim: t.in_dim as u32,
            out_dim: t.out_dim as u32,
            rank: t.rank as u32,
        };
        e.set_pipeline(&k.lora_shrink);
        e.set_buffer(0, input, 0);
        e.set_buffer(1, &t.a, 0);
        e.set_buffer(2, &self.ws.adapters, 0);
        e.set_buffer(3, &self.lora_hidden, 0);
        e.set_bytes(4, &args);
        let shrink_threads = threads(&k.lora_shrink, ROW_THREADS);
        e.dispatch(
            [t.rank.div_ceil(shrink_threads / SIMD), step.batch],
            shrink_threads,
        );
        e.set_pipeline(&k.lora_expand);
        e.set_buffer(0, &self.lora_hidden, 0);
        e.set_buffer(1, &t.b, 0);
        e.set_buffer(2, &self.ws.adapters, 0);
        e.set_buffer(3, out, 0);
        e.set_bytes(4, &args);
        let threads = threads(&k.lora_expand, ROW_THREADS);
        e.dispatch([t.out_dim.div_ceil(threads), step.batch], threads);
    }

    /// `(in, out)` for a module.
    fn module_shape(&self, module: Module) -> (usize, usize) {
        let Shape {
            dim,
            q_dim,
            kv_dim,
            hidden_dim,
            ..
        } = self.shape;
        match module {
            Module::Q => (dim, q_dim),
            Module::K | Module::V => (dim, kv_dim),
            Module::O => (q_dim, dim),
            Module::Gate | Module::Up => (dim, hidden_dim),
            Module::Down => (hidden_dim, dim),
        }
    }

    /// Put every adapter's corrections in GPU buffers, replacing what was there.
    fn upload_adapters(&mut self) -> Result<()> {
        let residual = self.architecture.residual_scale;
        let mut largest = 0;
        let mut lora = Vec::with_capacity(self.layers.len());
        for l in 0..self.layers.len() {
            let mut targets: [Option<LoraTarget>; 4] = Default::default();
            for (t, modules) in LORA_TARGETS.iter().enumerate() {
                let in_dim = self.module_shape(modules[0]).0;
                let outs: Vec<usize> = modules.iter().map(|&m| self.module_shape(m).1).collect();
                let out_dim: usize = outs.iter().sum();
                let rank_of = |adapter: &Vec<[Option<LoraPair>; 7]>| -> usize {
                    modules
                        .iter()
                        .map(|&m| adapter[l][m as usize].as_ref().map_or(0, |p| p.rank))
                        .sum()
                };
                let rank = self.adapters.iter().map(rank_of).max().unwrap_or(0);
                if rank == 0 {
                    continue;
                }
                let n = self.adapters.len();
                let (mut a, mut b) = (
                    vec![0.0f32; n * rank * in_dim],
                    vec![0.0f32; n * out_dim * rank],
                );
                for (i, adapter) in self.adapters.iter().enumerate() {
                    let (mut first_rank, mut first_out) = (0, 0);
                    for (&module, &out) in modules.iter().zip(&outs) {
                        if let Some(pair) = &adapter[l][module as usize] {
                            // The output projections' outputs are scaled, so their corrections are.
                            let scale = match module {
                                Module::O | Module::Down => residual,
                                _ => 1.0,
                            };
                            let at = (i * rank + first_rank) * in_dim;
                            a[at..at + pair.rank * in_dim].copy_from_slice(&pair.a);
                            for o in 0..out {
                                for j in 0..pair.rank {
                                    b[(i * out_dim + first_out + o) * rank + first_rank + j] =
                                        pair.b[o * pair.rank + j] * scale;
                                }
                            }
                            first_rank += pair.rank;
                        }
                        first_out += out;
                    }
                }
                let upload = |values: &[f32]| -> Result<Buffer> {
                    let buffer = self.device.buffer(values.len() * 4)?;
                    // SAFETY: the GPU hasn't seen this buffer yet.
                    let memory = unsafe { buffer.slice_mut::<f32>() };
                    memory[..values.len()].copy_from_slice(values);
                    Ok(buffer)
                };
                targets[t] = Some(LoraTarget {
                    a: upload(&a)?,
                    b: upload(&b)?,
                    rank,
                    in_dim,
                    out_dim,
                });
                largest = largest.max(rank);
            }
            lora.push(targets);
        }
        self.lora = lora;
        self.lora_rank = largest;
        Ok(())
    }

    /// Queue the normalization of the queries and keys.
    fn head_norms(&self, e: &Encoder, q_weight: &Buffer, k_weight: &Buffer, batch: usize) {
        let Shape { q_dim, kv_dim, .. } = self.shape;
        let (stride, eps) = ((q_dim + 2 * kv_dim) as u32, self.eps);
        e.set_pipeline(&self.kernels.head_norm);
        e.set_buffer(0, &self.ws.qkv, 0);
        e.set_buffer(1, q_weight, 0);
        e.set_buffer(2, k_weight, 0);
        let args = |n_q: usize, head_dim: usize| HeadNormArgs {
            stride,
            n_q: n_q as u32,
            head_dim: head_dim as u32,
            k_offset: q_dim as u32,
            eps,
        };
        if self.architecture.full_qk_norms {
            // All of a token's queries are normalized together, and all its keys (OLMoE).
            e.set_bytes(3, &args(1, q_dim));
            e.dispatch([1, batch], SIMD);
            e.set_bytes(3, &args(0, kv_dim));
            e.dispatch([1, batch], SIMD);
        } else {
            e.set_bytes(3, &args(self.n_heads, self.head_dim));
            e.dispatch([self.n_heads + self.n_kv_heads, batch], SIMD);
        }
    }

    /// Queue the rotation of the queries and keys, and the storing of keys and values.
    fn rope_and_store(&self, e: &Encoder, layer: &Layer, batch: usize) {
        let (ws, shape) = (&self.ws, self.shape);
        let table = layer.rope_table * batch * shape.half * 4;
        e.set_pipeline(&self.kernels.rope_store);
        e.set_buffer(0, &ws.qkv, 0);
        e.set_buffer(1, &ws.cos, table);
        e.set_buffer(2, &ws.sin, table);
        e.set_buffer(3, &ws.write_slots, 0);
        e.set_buffer(4, &layer.k_pool, 0);
        e.set_buffer(5, &layer.v_pool, 0);
        e.set_bytes(
            6,
            &RopeArgs {
                stride: (shape.q_dim + 2 * shape.kv_dim) as u32,
                n_heads: self.n_heads as u32,
                n_kv_heads: self.n_kv_heads as u32,
                head_dim: self.head_dim as u32,
                q_dim: shape.q_dim as u32,
                kv_dim: shape.kv_dim as u32,
            },
        );
        e.dispatch([batch, 1], threads(&self.kernels.rope_store, ROW_THREADS));
    }

    /// Queue the attention step, which leaves its result in `attn_out`.
    fn attention(&self, e: &Encoder, layer: &Layer, batch: usize) {
        let (ws, shape) = (&self.ws, self.shape);
        e.set_pipeline(&self.kernels.attention);
        e.set_buffer(0, &ws.qkv, 0);
        e.set_buffer(1, &layer.k_pool, 0);
        e.set_buffer(2, &layer.v_pool, 0);
        e.set_buffer(3, &self.slot_lists, 0);
        e.set_buffer(4, &ws.row_offset, 0);
        e.set_buffer(5, &ws.row_len, 0);
        e.set_buffer(6, &ws.attn_out, 0);
        e.set_bytes(
            7,
            &AttnArgs {
                stride: (shape.q_dim + 2 * shape.kv_dim) as u32,
                n_heads: self.n_heads as u32,
                n_kv_heads: self.n_kv_heads as u32,
                head_dim: self.head_dim as u32,
                kv_dim: shape.kv_dim as u32,
                q_dim: shape.q_dim as u32,
                scale: self.architecture.attention_scale,
                softcap: self.architecture.attention_softcap,
                window: layer.window.min(u32::MAX as usize) as u32,
            },
        );
        e.set_threadgroup_memory(0, ATT_SIMDS * (self.head_dim + 2) * 4);
        e.dispatch([self.n_heads, batch], ATT_SIMDS * SIMD);
    }
}

/// Kinds of kernel, for profiling.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Part {
    /// The matrix multiplications inside the layers.
    Matmuls,
    Attention,
    /// RMS norms, including the queries' and keys'.
    Norms,
    /// Rotating queries and keys, and storing keys and values.
    Rope,
    /// The embeddings and the MLP's activation.
    Other,
    /// The final norm, output projection and adjustments to the scores.
    Logits,
}

/// The two matrix kernels, which give exactly the same results.
#[derive(Clone, Copy, Debug, PartialEq)]
enum MatmulKernel {
    /// Simple sums, lane by lane, for small batches.
    Vec,
    /// Simdgroup matrix instructions on tiles, for big ones.
    Mma,
}

/// Queue `out = W x` for `batch` rows of `x`, or `out += W x` if `accumulate`, with `kernel`.
#[allow(clippy::too_many_arguments)]
fn encode_matmul(
    e: &Encoder,
    kernels: &Kernels,
    w: &Matrix,
    x: &Buffer,
    out: &Buffer,
    batch: usize,
    accumulate: bool,
    kernel: MatmulKernel,
) {
    let (pipeline, groups, threads) = match (kernel, w.bf16) {
        (MatmulKernel::Vec, bf16) => {
            let pipeline = match bf16 {
                true => &kernels.matmul_vec_bf16,
                false => &kernels.matmul_vec_f32,
            };
            let threads = threads(pipeline, VEC_THREADS);
            let rows_per_group = threads / SIMD * VEC_ROWS;
            let groups = [w.rows.div_ceil(rows_per_group), batch.div_ceil(VEC_BATCH)];
            (pipeline, groups, threads)
        }
        (MatmulKernel::Mma, bf16) => {
            let pipeline = match bf16 {
                true => &kernels.matmul_mma_bf16,
                false => &kernels.matmul_mma_f32,
            };
            let groups = [w.rows.div_ceil(MM_BM), batch.div_ceil(MM_BN)];
            (pipeline, groups, MM_THREADS)
        }
    };
    e.set_pipeline(pipeline);
    e.set_buffer(0, &w.data, 0);
    e.set_buffer(1, x, 0);
    e.set_buffer(2, out, 0);
    e.set_buffer(3, w.bias.as_ref().unwrap_or(&w.data), 0);
    e.set_bytes(
        4,
        &MatmulArgs {
            m: w.rows as u32,
            n: w.cols as u32,
            batch: batch as u32,
            scale: w.scale,
            accumulate: accumulate as u32,
            has_bias: w.bias.is_some() as u32,
        },
    );
    e.dispatch(groups, threads);
}

/// Threads per group for `pipeline`: `wanted`, or fewer if it can't have that many, in whole
/// simdgroups.
fn threads(pipeline: &Pipeline, wanted: usize) -> usize {
    (wanted.min(pipeline.max_threads) / SIMD * SIMD).max(SIMD)
}

/// Make sure `buffer` has room for `bytes`, replacing it with a bigger one if not. Only for
/// buffers the GPU isn't using, whose contents don't need keeping.
fn reserve(device: &Device, buffer: &mut Buffer, bytes: usize) -> Result<()> {
    if buffer.bytes < bytes {
        *buffer = device.buffer(bytes.next_power_of_two())?;
    }
    Ok(())
}

impl Backend for MetalBackend {
    fn describe(&self) -> String {
        let weights = match self.layers.first().is_none_or(|l| l.qkv.bf16) {
            true => "16-bit",
            false => "32-bit",
        };
        format!("{} (Metal, {weights} weights)", self.gpu_name)
    }

    fn forward(&mut self, rows: &[Row]) {
        self.try_forward(rows).expect("the GPU failed");
    }

    fn logits(&self, row: usize) -> Logits<'_> {
        // SAFETY: the GPU finished writing them before `forward` returned.
        let all = unsafe { self.logits.slice::<f32>() };
        Logits::All(&all[row * self.vocab_size..(row + 1) * self.vocab_size])
    }

    fn hidden(&self, row: usize) -> &[f32] {
        // SAFETY: the same as for `logits`.
        let all = unsafe { self.hidden.slice::<f32>() };
        &all[row * self.shape.dim..(row + 1) * self.shape.dim]
    }

    fn pool(&self) -> &Pool {
        &self.pool
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn add_adapter(&mut self, adapter: &LoraAdapter) -> Result<()> {
        ensure!(
            adapter.layers.len() == self.layers.len(),
            "{}: made for a model with {} layers, not {}",
            adapter.name,
            adapter.layers.len(),
            self.layers.len()
        );
        self.adapters.push(adapter.layers.clone());
        self.upload_adapters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::f32_to_bf16;

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Both matrix kernels give `scale * W x + bias` to within rounding, and each row of the
    /// batch gets exactly the same result from either, whatever else is in the batch.
    #[test]
    fn matmul_kernels_match_arithmetic_and_each_other() {
        let Ok(device) = Device::open() else {
            eprintln!("no Metal GPU, so skipping");
            return;
        };
        let kernels = Kernels::find(&device, &device.compile(KERNELS).unwrap()).unwrap();
        // 37 rows and 13 groups of 8 columns leave every tile and run partly full.
        let (m, n, batch) = (37, 104, 45);
        let weights = pseudo_random(m * n, 1);
        let bias = pseudo_random(m, 2);
        let x = pseudo_random(batch * n, 3);
        let upload = |values: &[f32]| {
            let buffer = device.buffer(values.len() * 4).unwrap();
            (unsafe { buffer.slice_mut::<f32>() })[..values.len()].copy_from_slice(values);
            buffer
        };
        for bf16 in [false, true] {
            let data = match bf16 {
                true => {
                    let half: Vec<u16> = weights.iter().map(|&w| f32_to_bf16(w)).collect();
                    let buffer = device.buffer(half.len() * 2).unwrap();
                    (unsafe { buffer.slice_mut::<u16>() })[..half.len()].copy_from_slice(&half);
                    buffer
                }
                false => upload(&weights),
            };
            let widened: Vec<f32> = match bf16 {
                true => (weights.iter())
                    .map(|&w| f32::from_bits((f32_to_bf16(w) as u32) << 16))
                    .collect(),
                false => weights.clone(),
            };
            let w = Matrix {
                data,
                bf16,
                rows: m,
                cols: n,
                bias: Some(upload(&bias)),
                scale: 0.5,
            };
            let run = |rows: usize, kernel: MatmulKernel| {
                let (input, out) = (upload(&x[..rows * n]), device.buffer(rows * m * 4).unwrap());
                device
                    .run(|e| {
                        encode_matmul(e, &kernels, &w, &input, &out, rows, false, kernel);
                        Ok(())
                    })
                    .unwrap();
                (unsafe { out.slice::<f32>() })[..rows * m].to_vec()
            };
            let full = run(batch, MatmulKernel::Mma);
            for b in 0..batch {
                for r in 0..m {
                    let exact: f64 = (0..n)
                        .map(|c| widened[r * n + c] as f64 * x[b * n + c] as f64)
                        .sum::<f64>()
                        * 0.5
                        + bias[r] as f64;
                    let got = full[b * m + r] as f64;
                    assert!(
                        (got - exact).abs() < 1e-5,
                        "row {b}, output {r}: {got} vs {exact}"
                    );
                }
            }
            for rows in [1, 2, 7, 8, 9, 16, 45] {
                for kernel in [MatmulKernel::Vec, MatmulKernel::Mma] {
                    let got = run(rows, kernel);
                    assert_eq!(
                        got,
                        full[..rows * m],
                        "{kernel:?} with {rows} rows, bf16 {bf16}"
                    );
                }
            }
        }
    }

    /// How fast the matrix kernels are on a few shapes:
    /// `cargo test --release --lib metal_matmul_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn metal_matmul_speed() {
        let device = Device::open().unwrap();
        let kernels = Kernels::find(&device, &device.compile(KERNELS).unwrap()).unwrap();
        for (m, n) in [
            (151936, 1024),
            (4096, 1024),
            (6144, 1024),
            (1024, 2048),
            (1024, 3072),
        ] {
            let w = Matrix {
                data: device.buffer(m * n * 2).unwrap(),
                bf16: true,
                rows: m,
                cols: n,
                bias: None,
                scale: 1.0,
            };
            let x = device.buffer(512 * n * 4).unwrap();
            let out = device.buffer(512 * m * 4).unwrap();
            let batches: Vec<usize> = std::env::var("BATCHES")
                .map_or(vec![1, 1, 2, 4, 8, 16, 32, 64, 128, 512], |b| {
                    b.split(',').map(|b| b.parse().unwrap()).collect()
                });
            for batch in batches {
                for kernel in [MatmulKernel::Vec, MatmulKernel::Mma] {
                    if (kernel == MatmulKernel::Vec && batch > 64)
                        || (kernel == MatmulKernel::Mma && batch < 8)
                    {
                        continue;
                    }
                    let repeats = 20;
                    let seconds = device
                        .run(|e| {
                            for _ in 0..repeats {
                                encode_matmul(e, &kernels, &w, &x, &out, batch, false, kernel);
                            }
                            Ok(())
                        })
                        .unwrap()
                        / repeats as f64;
                    println!(
                        "{m:6} x {n:4}, batch {batch:3}, {kernel:?}: {:7.3} ms, {:5.0} GB/s of weights, {:6.0} GFLOP/s",
                        seconds * 1e3,
                        (m * n * 2) as f64 / seconds / 1e9,
                        (2 * m * n * batch) as f64 / seconds / 1e9
                    );
                }
            }
        }
    }
}
