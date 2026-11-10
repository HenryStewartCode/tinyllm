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
use crate::config::{
    Activation, Architecture, Config, ExpertNaming, Experts as ConfigExperts, LatentAttention,
    Routing,
};
use crate::lora::{LoraAdapter, LoraPair, Module};
use crate::model::{LogitUse, Row};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Result, bail, ensure};
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
    /// Another checkpoint to take the embeddings from if this one has none (an EAGLE head
    /// uses its model's).
    pub weights_from: Option<std::path::PathBuf>,
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
    /// How far apart the input's rows are.
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyArgs {
    dim: u32,
    stride: u32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScatterArgs {
    draft_vocab: u32,
    vocab: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeadArgs {
    w_stride: u32,
    x_stride: u32,
    out_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LatentArgs {
    n_heads: u32,
    nope: u32,
    rope: u32,
    rank: u32,
    stride: u32,
    rows: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LatentAttnArgs {
    n_heads: u32,
    rank: u32,
    rope: u32,
    rows: u32,
    scale: f32,
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
    /// How far apart the output's rows are (usually `m`).
    out_stride: u32,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct RouteArgs {
    experts: u32,
    top_k: u32,
    renormalize: u32,
    deepseek: u32,
    sigmoid: u32,
    groups: u32,
    top_groups: u32,
    scale: f32,
    has_correction: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SortArgs {
    rows: u32,
    top_k: u32,
    experts: u32,
    group_rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GatherArgs {
    dim: u32,
    group_rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CombineArgs {
    dim: u32,
    top_k: u32,
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
    moe_route: Pipeline,
    moe_sort: Pipeline,
    moe_gather: Pipeline,
    moe_matmul_vec_bf16: Pipeline,
    moe_matmul_vec_f32: Pipeline,
    moe_matmul_mma_bf16: Pipeline,
    moe_matmul_mma_f32: Pipeline,
    moe_combine: Pipeline,
    add_gated: Pipeline,
    matmul_heads_vec_bf16: Pipeline,
    matmul_heads_vec_f32: Pipeline,
    matmul_heads_mma_bf16: Pipeline,
    matmul_heads_mma_f32: Pipeline,
    latent_queries: Pipeline,
    latent_store: Pipeline,
    attention_latent: Pipeline,
    copy_rows: Pipeline,
    eagle_input: Pipeline,
    scatter_logits: Pipeline,
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
            moe_route: get("moe_route")?,
            moe_sort: get("moe_sort")?,
            moe_gather: get("moe_gather")?,
            moe_matmul_vec_bf16: get("moe_matmul_vec_bf16")?,
            moe_matmul_vec_f32: get("moe_matmul_vec_f32")?,
            moe_matmul_mma_bf16: get("moe_matmul_mma_bf16")?,
            moe_matmul_mma_f32: get("moe_matmul_mma_f32")?,
            moe_combine: get("moe_combine")?,
            add_gated: get("add_gated")?,
            matmul_heads_vec_bf16: get("matmul_heads_vec_bf16")?,
            matmul_heads_vec_f32: get("matmul_heads_vec_f32")?,
            matmul_heads_mma_bf16: get("matmul_heads_mma_bf16")?,
            matmul_heads_mma_f32: get("matmul_heads_mma_f32")?,
            latent_queries: get("latent_queries")?,
            latent_store: get("latent_store")?,
            attention_latent: get("attention_latent")?,
            copy_rows: get("copy_rows")?,
            eagle_input: get("eagle_input")?,
            scatter_logits: get("scatter_logits")?,
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
    feed_forward: FeedForward,
    /// The same as `norm_after_attn`, for the MLP.
    norm_after_mlp: Option<Buffer>,
    /// With latent attention, `k_pool` is the cache of latents and `v_pool` is unused.
    k_pool: Buffer,
    v_pool: Buffer,
    rope_table: usize,
    /// How many tokens it can see.
    window: usize,
    /// DeepSeek's latent attention, which uses `qkv` for the projections of the layer's input.
    latent: Option<LatentLayer>,
}

/// DeepSeek's latent attention for a layer (see `kernels.metal`). The layer's `qkv` stacks
/// `q_a_proj` (or `q_proj`, when queries have no latent of their own) and `kv_a_proj_with_mqa`.
struct LatentLayer {
    /// `q_a_layernorm` and `q_b_proj`, when queries go through a latent of their own.
    q_low: Option<(Buffer, Matrix)>,
    kv_a_norm: Buffer,
    /// Each head's key matrix from `kv_b_proj`, transposed, `rows` being one head's:
    /// `[heads][rank, nope]`.
    absorb_k: Matrix,
    /// Each head's value matrix: `[heads][v, rank]`.
    absorb_v: Matrix,
}

/// What an EAGLE-3 draft head has that a model doesn't (see [`crate::config::Eagle3`]), and its
/// working memory. Its one layer's `attn_norm` normalizes each token's embedding.
struct EagleParts {
    /// Folds the model's kept features (`3 * dim`) into one hidden state.
    fc: Matrix,
    hidden_norm: Buffer,
    /// The model's token id for each of the head's output rows.
    targets: Buffer,
    draft_vocab: usize,
    /// Each row's hidden state for the next forward, `width` a row.
    hidden_in: Buffer,
    width: usize,
    /// The layer's input: each row's normalized embedding, then its normalized hidden state,
    /// `[rows, 2 * dim]`.
    wide: Buffer,
    /// Scores for the head's own vocabulary, `[rows, draft_vocab]`.
    draft_logits: Buffer,
}

/// Working memory for latent attention, sized for the workspace's rows.
struct LatentBuffers {
    /// `q_a_proj`'s outputs, normalized, `[rows, q_lora_rank]` (when queries have a latent).
    q_low: Buffer,
    /// The queries from `q_b_proj`, `[rows, heads * (nope + rope)]` (likewise).
    q: Buffer,
    /// Each head's unrotated query part, `[heads][rows, nope]`.
    q_nope: Buffer,
    /// Attention's queries, `[rows][heads][rank + rope]`.
    query: Buffer,
    /// Each head's blend of latents, `[heads][rows, rank]`.
    blended: Buffer,
}

enum FeedForward {
    /// gate_proj and up_proj as one matrix, for the same reason as `qkv`, then down_proj.
    Dense { gate_up: Matrix, down: Matrix },
    Sparse(Box<Experts>),
}

/// A layer's mixture of experts (see "Mixtures of experts" in `kernels.metal`).
struct Experts {
    router: Matrix,
    /// Every expert's gate_proj and up_proj stacked, one expert after another, `rows` being one
    /// expert's: `[experts][2 * hidden, dim]`.
    gate_up: Matrix,
    /// Every expert's down_proj: `[experts][dim, hidden]`.
    down: Matrix,
    /// What's added to the router's scores to choose experts by (DeepSeek V3).
    correction: Option<Buffer>,
    /// An expert every token goes through as well, like a dense MLP: gate_up, down, and its own
    /// gate (`[1, dim]`, Qwen2 MoE) if it has one.
    shared: Option<(Matrix, Matrix, Option<Matrix>)>,
}

/// The sizes the working memory depends on.
#[derive(Clone, Copy)]
struct Shape {
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
    /// The widest MLP: a dense layer's, or a shared expert's.
    mlp_width: usize,
    /// Half a head, the length of a row of RoPE rotations.
    half: usize,
    rope_tables: usize,
    /// For mixtures of experts: how many, how many per token, and their hidden width.
    experts: Option<(usize, usize, usize)>,
    heads: usize,
    /// The projections of a layer's input, stacked, and attention's output.
    qkv_width: usize,
    attn_width: usize,
    latent: Option<LatentAttention>,
}

/// Rows a mixture of experts' grouped matmuls go through, at most, for `rows` rows: each expert's
/// rows, padded to whole groups of `MM_BN` (the larger of the two kernels' groups).
fn moe_slots(rows: usize, (count, top_k, _): (usize, usize, usize)) -> usize {
    (rows * top_k + count * (MM_BN - 1)).next_multiple_of(MM_BN)
}

/// Working memory for mixtures of experts.
struct MoeBuffers {
    /// The router's scores `[rows, experts]`.
    scores: Buffer,
    /// Each row's experts and their weights `[rows, top_k]`.
    topk_ids: Buffer,
    topk_weights: Buffer,
    /// Where each (row, expert) pair went in expert order.
    slot_of: Buffer,
    /// The row in each slot, or -1 for padding.
    sorted_rows: Buffer,
    /// Each group's expert, and how many groups there are.
    group_expert: Buffer,
    num_groups: Buffer,
    /// The rows in expert order, and what the experts make of them.
    x: Buffer,
    gate_up: Buffer,
    gated: Buffer,
    down: Buffer,
    /// A shared expert's gate for each row.
    shared_gate: Buffer,
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
    moe: Option<MoeBuffers>,
    latent: Option<LatentBuffers>,
}

impl Workspace {
    fn new(device: &Device, shape: Shape, rows: usize) -> Result<Self> {
        let floats = |count: usize| device.buffer(rows * count * 4);
        let moe = match shape.experts {
            None => None,
            Some(spec @ (count, top_k, hidden)) => {
                let slots = moe_slots(rows, spec);
                let buffer = |count: usize| device.buffer(count * 4);
                Some(MoeBuffers {
                    scores: floats(count)?,
                    topk_ids: floats(top_k)?,
                    topk_weights: floats(top_k)?,
                    slot_of: floats(top_k)?,
                    sorted_rows: buffer(slots)?,
                    group_expert: buffer(slots / VEC_BATCH)?,
                    num_groups: buffer(1)?,
                    x: buffer(slots * shape.dim)?,
                    gate_up: buffer(slots * 2 * hidden)?,
                    gated: buffer(slots * hidden)?,
                    down: buffer(slots * shape.dim)?,
                    shared_gate: floats(1)?,
                })
            }
        };
        let latent = match shape.latent {
            None => None,
            Some(l) => {
                let n = shape.heads;
                Some(LatentBuffers {
                    q_low: floats(l.q_lora_rank.unwrap_or(1))?,
                    q: floats(l.q_lora_rank.map_or(1, |_| n * (l.nope + l.rope)))?,
                    q_nope: floats(n * l.nope)?,
                    query: floats(n * (l.kv_lora_rank + l.rope))?,
                    blended: floats(n * l.kv_lora_rank)?,
                })
            }
        };
        Ok(Self {
            moe,
            latent,
            rows,
            x: floats(shape.dim)?,
            xb: floats(shape.dim)?,
            added: floats(shape.dim)?,
            qkv: floats(shape.qkv_width)?,
            attn_out: floats(shape.attn_width)?,
            gate_up: floats(2 * shape.mlp_width)?,
            gated: floats(shape.mlp_width)?,
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

/// Split `kv_b_proj` (`[heads][nope + v, rank]`) into each head's key matrix, transposed
/// (`[heads][rank, nope]`), and value matrix (`[heads][v, rank]`).
fn split_absorbed<T: Copy>(kv_b: &[T], l: LatentAttention) -> (Vec<T>, Vec<T>) {
    let rank = l.kv_lora_rank;
    let (mut keys, mut values) = (Vec::with_capacity(kv_b.len()), Vec::with_capacity(kv_b.len()));
    for head in kv_b.chunks_exact((l.nope + l.v) * rank) {
        for c in 0..rank {
            keys.extend((0..l.nope).map(|i| head[i * rank + c]));
        }
        values.extend_from_slice(&head[l.nope * rank..]);
    }
    (keys, values)
}

impl Loader<'_> {
    /// A layer's `kv_b_proj`, as each head's key and value matrices (see [`LatentLayer`]).
    fn absorbed(&self, attn: &str, l: LatentAttention, n_heads: usize) -> Result<(Matrix, Matrix)> {
        let name = format!("{attn}.kv_b_proj.weight");
        let shape = [n_heads * (l.nope + l.v), l.kv_lora_rank];
        let bf16 = self.st.dtype(&name) == Some("BF16");
        let (keys, values) = match bf16 {
            true => {
                let (keys, values) = split_absorbed(&self.st.load_bf16(&name, &shape)?, l);
                (self.upload(&keys)?, self.upload(&values)?)
            }
            false => {
                let (keys, values) = split_absorbed(&self.st.load(&name, &shape)?, l);
                (self.upload(&keys)?, self.upload(&values)?)
            }
        };
        self.params.set(self.params.get() + shape[0] * shape[1]);
        let matrix = |data, rows, cols| Matrix {
            data,
            bf16,
            rows,
            cols,
            bias: None,
            scale: 1.0,
        };
        Ok((
            matrix(keys, l.kv_lora_rank, l.nope),
            matrix(values, l.v, l.kv_lora_rank),
        ))
    }

    /// A layer's mixture of experts, with layer prefix `p` (see [`Experts`]).
    fn experts(&self, p: &str, spec: &ConfigExperts, dim: usize, scale: f32) -> Result<Experts> {
        let (count, h) = (spec.count, spec.hidden);
        let router_name = match spec.naming {
            ExpertNaming::Qwen | ExpertNaming::DeepSeek => format!("{p}.mlp.gate"),
            ExpertNaming::Mixtral => format!("{p}.block_sparse_moe.gate"),
            ExpertNaming::Granite => format!("{p}.block_sparse_moe.router.layer"),
            ExpertNaming::GptOss => bail!("GPT-OSS's experts don't run on Metal yet"),
        };
        let router = self.matrix(&[(router_name.clone(), count)], dim, 1.0)?;
        // Every expert's gate and up projections in one tensor, and every down: Granite's way,
        // and how transformers 5 saves the others. Otherwise each expert's are named.
        let stacked = match spec.naming {
            ExpertNaming::Granite => Some((
                format!("{p}.block_sparse_moe.input_linear.weight"),
                format!("{p}.block_sparse_moe.output_linear.weight"),
            )),
            ExpertNaming::Mixtral => None,
            _ => Some((
                format!("{p}.mlp.experts.gate_up_proj"),
                format!("{p}.mlp.experts.down_proj"),
            ))
            .filter(|(input, _)| self.st.contains(input)),
        };
        let (prefix, [gate, up, down]) = match spec.naming {
            ExpertNaming::Mixtral => (format!("{p}.block_sparse_moe.experts"), ["w1", "w3", "w2"]),
            _ => (
                format!("{p}.mlp.experts"),
                ["gate_proj", "up_proj", "down_proj"],
            ),
        };
        // (tensor, rows, cols) in the order they're stacked.
        let (gate_up_parts, down_parts): (Vec<(String, usize, usize)>, Vec<_>) = match &stacked {
            Some((input, output)) => (
                vec![(input.clone(), count * 2 * h, dim)],
                vec![(output.clone(), count * dim, h)],
            ),
            None => (
                (0..count)
                    .flat_map(|e| {
                        [gate, up].map(|name| (format!("{prefix}.{e}.{name}.weight"), h, dim))
                    })
                    .collect(),
                (0..count)
                    .map(|e| (format!("{prefix}.{e}.{down}.weight"), dim, h))
                    .collect(),
            ),
        };
        let stack = |parts: &[(String, usize, usize)], rows: usize, cols: usize, scale: f32| {
            let bf16 = parts.iter().all(|(name, _, _)| self.st.dtype(name) == Some("BF16"));
            let total: usize = parts.iter().map(|(_, rows, _)| rows).sum();
            let data = self.device.buffer(total * cols * if bf16 { 2 } else { 4 })?;
            let mut at = 0;
            for (name, part_rows, part_cols) in parts {
                // A stacked tensor is `[experts, rows, cols]`.
                let shape = match stacked.is_some() {
                    true => vec![count, part_rows / count, *part_cols],
                    false => vec![*part_rows, *part_cols],
                };
                let range = at * cols..(at + part_rows) * cols;
                match bf16 {
                    true => {
                        // SAFETY: the GPU hasn't seen this buffer yet.
                        let memory = unsafe { data.slice_mut::<u16>() };
                        memory[range].copy_from_slice(&self.st.load_bf16(name, &shape)?);
                    }
                    false => {
                        // SAFETY: as above.
                        let memory = unsafe { data.slice_mut::<f32>() };
                        memory[range].copy_from_slice(&self.st.load(name, &shape)?);
                    }
                }
                at += part_rows;
            }
            self.params.set(self.params.get() + total * cols);
            Ok::<_, anyhow::Error>(Matrix {
                data,
                bf16,
                rows,
                cols,
                bias: None,
                scale,
            })
        };
        let correction_name = format!("{router_name}.e_score_correction_bias");
        let correction = match self.st.contains(&correction_name) {
            true => Some(self.upload(&self.st.load(&correction_name, &[count])?)?),
            false => None,
        };
        let shared = match spec.shared_hidden {
            None => None,
            Some(hidden) => {
                let prefix = match spec.shared_gated {
                    true => format!("{p}.mlp.shared_expert"),
                    false => format!("{p}.mlp.shared_experts"),
                };
                Some((
                    self.matrix(
                        &[
                            (format!("{prefix}.gate_proj"), hidden),
                            (format!("{prefix}.up_proj"), hidden),
                        ],
                        dim,
                        1.0,
                    )?,
                    self.matrix(&[(format!("{prefix}.down_proj"), dim)], hidden, scale)?,
                    match spec.shared_gated {
                        true => Some(self.matrix(&[(format!("{p}.mlp.shared_expert_gate"), 1)], dim, 1.0)?),
                        false => None,
                    },
                ))
            }
        };
        Ok(Experts {
            router,
            gate_up: stack(&gate_up_parts, 2 * h, dim, 1.0)?,
            down: stack(&down_parts, dim, h, scale)?,
            correction,
            shared,
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
    eagle: Option<EagleParts>,
    /// The layers whose output (the residual stream after them) each forward keeps for every
    /// row, and where: `[rows, capture.len() * dim]`.
    capture: Vec<usize>,
    features: Buffer,
    /// How many rows the last forward had.
    last_rows: usize,
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
        if let Some(spec) = &architecture.experts {
            ensure!(
                spec.glu.is_none() && spec.naming != ExpertNaming::GptOss,
                "GPT-OSS's experts don't run on Metal yet"
            );
            ensure!(
                spec.top_k <= 16 && spec.count <= 512,
                "the Metal kernels handle at most 512 experts and 16 per token"
            );
            if let Routing::DeepSeek { groups, .. } = spec.routing {
                ensure!(
                    groups <= 32 && spec.count.is_multiple_of(groups),
                    "the Metal kernels handle at most 32 groups of experts"
                );
            }
            ensure!(
                [spec.hidden, spec.shared_hidden.unwrap_or(0)]
                    .iter()
                    .all(|width| width.is_multiple_of(8)),
                "the Metal kernels need experts' widths that are multiples of 8"
            );
        }
        ensure!(
            !architecture.attention_sinks,
            "attention sinks (GPT-OSS) don't run on Metal yet"
        );
        let latent = architecture.latent;
        let eagle3 = architecture.eagle3;
        // An EAGLE head's attention reads each token's embedding next to its hidden state.
        let attn_in = match eagle3 {
            Some(_) => 2 * config.hidden_size,
            None => config.hidden_size,
        };
        if let Some(l) = latent {
            ensure!(
                l.kv_lora_rank <= 512
                    && l.kv_lora_rank + l.rope <= 640
                    && [l.kv_lora_rank, l.rope, l.nope, l.v, l.q_lora_rank.unwrap_or(0)]
                        .iter()
                        .all(|size| size.is_multiple_of(8)),
                "the Metal kernels need a latent of at most 512 and sizes in multiples of 8"
            );
        }
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
        let st = SafeTensors::open_with(dir, options.weights_from.as_deref())?;
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
            let (mut qkv_names, gate_up_names) = match architecture.combined_projections {
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
            // Latent attention starts with the projections of the layer's input.
            let latent_layer = match latent {
                None => None,
                Some(l) => {
                    qkv_names = vec![
                        match l.q_lora_rank {
                            Some(rank) => (format!("{attn}.q_a_proj"), rank),
                            None => (format!("{attn}.q_proj"), n_heads * (l.nope + l.rope)),
                        },
                        (format!("{attn}.kv_a_proj_with_mqa"), l.kv_lora_rank + l.rope),
                    ];
                    let (absorb_k, absorb_v) = loader.absorbed(&attn, l, n_heads)?;
                    Some(LatentLayer {
                        q_low: match l.q_lora_rank {
                            Some(rank) => Some((
                                loader.norm(&format!("{attn}.q_a_layernorm.weight"), rank, false)?,
                                loader.matrix(
                                    &[(format!("{attn}.q_b_proj"), n_heads * (l.nope + l.rope))],
                                    rank,
                                    1.0,
                                )?,
                            )),
                            None => None,
                        },
                        kv_a_norm: loader.norm(
                            &format!("{attn}.kv_a_layernorm.weight"),
                            l.kv_lora_rank,
                            false,
                        )?,
                        absorb_k,
                        absorb_v,
                    })
                }
            };
            let attn_width = latent.map_or(q_dim, |l| n_heads * l.v);
            let feed_forward = match (shape.sparse, &architecture.experts) {
                (true, Some(spec)) => {
                    FeedForward::Sparse(Box::new(loader.experts(&p, spec, dim, residual)?))
                }
                _ => FeedForward::Dense {
                    gate_up: loader.matrix(&gate_up_names, dim, 1.0)?,
                    down: loader.matrix(
                        &[(format!("{p}.mlp.down_proj"), dim)],
                        hidden_dim,
                        residual,
                    )?,
                },
            };
            layers.push(Layer {
                attn_norm: norm("input_layernorm")?,
                qkv: loader.matrix(&qkv_names, attn_in, 1.0)?,
                head_norms,
                o_proj: loader.matrix(&[(format!("{attn}.o_proj"), dim)], attn_width, residual)?,
                norm_after_attn,
                mlp_norm,
                feed_forward,
                norm_after_mlp,
                // Allocated below, once the weights' size is known.
                k_pool: device.buffer(0)?,
                v_pool: device.buffer(0)?,
                rope_table: shape.rope_table,
                window: shape.window.unwrap_or(u32::MAX as usize),
                latent: latent_layer,
            });
        }
        let embed = loader.matrix(&[("model.embed_tokens".into(), vocab_size)], dim, 1.0)?;
        // An EAGLE head scores its own, smaller, vocabulary.
        let out_rows = eagle3.map_or(vocab_size, |e| e.draft_vocab);
        let lm_head = match st.contains("lm_head.weight") {
            true => Some(loader.matrix(&[("lm_head".into(), out_rows)], dim, 1.0)?),
            false => None,
        };
        let eagle = match eagle3 {
            None => None,
            Some(e) => {
                ensure!(
                    config.num_hidden_layers == 1 && lm_head.is_some(),
                    "an EAGLE-3 head has one layer and an output projection of its own"
                );
                let targets = crate::model::eagle_targets(&st, e.draft_vocab, vocab_size)?;
                Some(EagleParts {
                    fc: loader.matrix(&[("fc".into(), dim)], 3 * dim, 1.0)?,
                    hidden_norm: loader.norm("model.layers.0.hidden_norm.weight", dim, false)?,
                    targets: loader.upload(&targets)?,
                    draft_vocab: e.draft_vocab,
                    hidden_in: device.buffer(16)?,
                    width: 0,
                    wide: device.buffer(16)?,
                    draft_logits: device.buffer(16)?,
                })
            }
        };
        let norm = loader.norm("model.norm.weight", dim, false)?;
        let num_params = loader.params.get();

        // The KV cache gets what was asked for, as long as Metal thinks it'll fit.
        let weight_bytes = |m: &Matrix| m.data.bytes;
        let weights = weight_bytes(&embed)
            + lm_head.as_ref().map_or(0, weight_bytes)
            + (layers.iter())
                .map(|l| {
                    let mlp = match &l.feed_forward {
                        FeedForward::Dense { gate_up, down } => [gate_up, down].map(weight_bytes),
                        FeedForward::Sparse(experts) => {
                            [&experts.gate_up, &experts.down].map(weight_bytes)
                        }
                    };
                    weight_bytes(&l.qkv) + weight_bytes(&l.o_proj) + mlp.iter().sum::<usize>()
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
        // Latent attention keeps each token's latent and rotated key part instead.
        let kv_bytes_per_token = match latent {
            Some(l) => layers.len() * (l.kv_lora_rank + l.rope) * 4,
            None => 2 * layers.len() * kv_dim * 4,
        };
        let capacity = (cache_bytes / kv_bytes_per_token).clamp(16, u32::MAX as usize);
        for layer in &mut layers {
            match latent {
                Some(l) => layer.k_pool = device.buffer(capacity * (l.kv_lora_rank + l.rope) * 4)?,
                None => {
                    layer.k_pool = device.buffer(capacity * kv_dim * 4)?;
                    layer.v_pool = device.buffer(capacity * kv_dim * 4)?;
                }
            }
        }

        let experts = architecture.experts.as_ref();
        let shape = Shape {
            dim,
            q_dim,
            kv_dim,
            hidden_dim,
            mlp_width: hidden_dim.max(experts.and_then(|spec| spec.shared_hidden).unwrap_or(0)),
            half: config.rope_dim() / 2,
            rope_tables: architecture.rope_tables.len(),
            experts: experts.map(|spec| (spec.count, spec.top_k, spec.hidden)),
            heads: n_heads,
            qkv_width: match latent {
                Some(l) => {
                    l.q_lora_rank.unwrap_or(n_heads * (l.nope + l.rope)) + l.kv_lora_rank + l.rope
                }
                None => q_dim + 2 * kv_dim,
            },
            attn_width: latent.map_or(q_dim, |l| n_heads * l.v),
            latent,
        };
        let ws = Workspace::new(&device, shape, 16)?;
        let features = device.buffer(16)?;
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
                eagle,
                capture: Vec::new(),
                features,
                last_rows: 0,
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
        let dim = self.shape.dim;
        if let Some(eagle) = &mut self.eagle {
            ensure!(
                eagle.hidden_in.bytes >= rows.len() * eagle.width * 4 && eagle.width > 0,
                "an EAGLE head needs its hidden states first"
            );
            reserve(&self.device, &mut eagle.wide, rows.len() * 2 * dim * 4)?;
            reserve(&self.device, &mut eagle.draft_logits, step.n_logits.max(1) * eagle.draft_vocab * 4)?;
        }
        if !self.capture.is_empty() {
            let bytes = rows.len() * self.capture.len() * dim * 4;
            reserve(&self.device, &mut self.features, bytes)?;
        }
        self.last_rows = rows.len();
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
        if let Some(eagle) = &self.eagle {
            self.eagle_input(e, eagle, batch);
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block: mix in information from earlier tokens.
            let attn_input = match &self.eagle {
                Some(eagle) => &eagle.wide,
                None => {
                    if on(Part::Norms) {
                        self.rmsnorm(e, &ws.x, &ws.xb, &layer.attn_norm, batch, false);
                    }
                    &ws.xb
                }
            };
            if on(Part::Matmuls) {
                self.matmul(e, &layer.qkv, attn_input, &ws.qkv, batch, false);
                self.lora(e, step, l, 0, attn_input, &ws.qkv);
            }
            if let Some(latent) = &layer.latent {
                self.latent_attention(e, layer, latent, batch, &on);
            } else {
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
            let (gate_up, down) = match &layer.feed_forward {
                FeedForward::Dense { gate_up, down } => (gate_up, down),
                FeedForward::Sparse(experts) => {
                    self.experts(e, experts, batch, &on);
                    self.keep_features(e, l, batch);
                    continue;
                }
            };
            if on(Part::Matmuls) {
                self.matmul(e, gate_up, &ws.xb, &ws.gate_up, batch, false);
                self.lora(e, step, l, 2, &ws.xb, &ws.gate_up);
            }
            if on(Part::Other) {
                self.gated(e, &ws.gate_up, &ws.gated, down.cols, batch);
            }
            if on(Part::Matmuls) {
                self.add_block_output(
                    e,
                    down,
                    &ws.gated,
                    batch,
                    layer.norm_after_mlp.as_ref(),
                    (step, l, 3),
                );
            }
            self.keep_features(e, l, batch);
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
            match &self.eagle {
                None => self.matmul(e, lm_head, &ws.xb, &self.logits, n_logits, false),
                // An EAGLE head's scores go to their tokens' places in the model's vocabulary.
                Some(eagle) => {
                    self.matmul(e, lm_head, &ws.xb, &eagle.draft_logits, n_logits, false);
                    e.set_pipeline(&k.scatter_logits);
                    e.set_buffer(0, &eagle.draft_logits, 0);
                    e.set_buffer(1, &eagle.targets, 0);
                    e.set_buffer(2, &self.logits, 0);
                    e.set_bytes(
                        3,
                        &ScatterArgs {
                            draft_vocab: eagle.draft_vocab as u32,
                            vocab: self.vocab_size as u32,
                        },
                    );
                    e.dispatch([n_logits, 1], threads(&k.scatter_logits, 1024));
                }
            }
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

    /// Queue DeepSeek's latent attention (see `kernels.metal`), from the stacked projections of
    /// the layer's input in `ws.qkv` to its result in `ws.attn_out`, storing each row's latent in
    /// the cache.
    fn latent_attention(
        &self,
        e: &Encoder,
        layer: &Layer,
        latent: &LatentLayer,
        batch: usize,
        on: &dyn Fn(Part) -> bool,
    ) {
        let l = self.architecture.latent.expect("a latent layer has latent attention");
        let (ws, k) = (&self.ws, &self.kernels);
        let b = ws.latent.as_ref().expect("allocated for latent attention");
        let (n, rank, rope, nope, v) = (self.n_heads, l.kv_lora_rank, l.rope, l.nope, l.v);
        let q_width = l.q_lora_rank.unwrap_or(n * (nope + rope));
        let stride = q_width + rank + rope;
        let args = |stride: usize| LatentArgs {
            n_heads: n as u32,
            nope: nope as u32,
            rope: rope as u32,
            rank: rank as u32,
            stride: stride as u32,
            rows: ws.rows as u32,
            eps: self.eps,
        };
        // Queries straight from the layer's input, or through a latent of their own.
        let (q, q_stride) = match &latent.q_low {
            Some((norm, q_b)) => {
                if on(Part::Norms) {
                    e.set_pipeline(&k.rmsnorm);
                    e.set_buffer(0, &ws.qkv, 0);
                    e.set_buffer(1, &b.q_low, 0);
                    e.set_buffer(2, norm, 0);
                    e.set_buffer(3, &ws.logit_rows, 0);
                    e.set_bytes(
                        4,
                        &NormArgs {
                            dim: q_width as u32,
                            eps: self.eps,
                            select: 0,
                            stride: stride as u32,
                        },
                    );
                    e.dispatch([batch, 1], NORM_THREADS);
                }
                if on(Part::Matmuls) {
                    self.matmul(e, q_b, &b.q_low, &b.q, batch, false);
                }
                (&b.q, n * (nope + rope))
            }
            None => (&ws.qkv, stride),
        };
        if on(Part::Rope) {
            let table = layer.rope_table * batch * self.shape.half * 4;
            e.set_pipeline(&k.latent_queries);
            e.set_buffer(0, q, 0);
            e.set_buffer(1, &ws.cos, table);
            e.set_buffer(2, &ws.sin, table);
            e.set_buffer(3, &b.q_nope, 0);
            e.set_buffer(4, &b.query, 0);
            e.set_bytes(5, &args(q_stride));
            e.dispatch([batch, 1], threads(&k.latent_queries, ROW_THREADS));
            e.set_pipeline(&k.latent_store);
            e.set_buffer(0, &ws.qkv, q_width * 4);
            e.set_buffer(1, &latent.kv_a_norm, 0);
            e.set_buffer(2, &ws.cos, table);
            e.set_buffer(3, &ws.sin, table);
            e.set_buffer(4, &ws.write_slots, 0);
            e.set_buffer(5, &layer.k_pool, 0);
            e.set_bytes(6, &args(stride));
            e.dispatch([batch, 1], NORM_THREADS);
        }
        // Each head's query through its key matrix, to score against latents directly.
        let key = rank + rope;
        if on(Part::Matmuls) {
            self.matmul_heads(e, &latent.absorb_k, &b.q_nope, &b.query, batch, (n * key, key));
        }
        if on(Part::Attention) {
            e.set_pipeline(&k.attention_latent);
            e.set_buffer(0, &b.query, 0);
            e.set_buffer(1, &layer.k_pool, 0);
            e.set_buffer(2, &self.slot_lists, 0);
            e.set_buffer(3, &ws.row_offset, 0);
            e.set_buffer(4, &ws.row_len, 0);
            e.set_buffer(5, &b.blended, 0);
            e.set_bytes(
                6,
                &LatentAttnArgs {
                    n_heads: n as u32,
                    rank: rank as u32,
                    rope: rope as u32,
                    rows: ws.rows as u32,
                    scale: self.architecture.attention_scale,
                },
            );
            e.set_threadgroup_memory(0, ATT_SIMDS * (rank + 2) * 4);
            e.dispatch([n, batch], ATT_SIMDS * SIMD);
        }
        // Each head's blend of latents through its value matrix, into its place in attn_out.
        if on(Part::Matmuls) {
            self.matmul_heads(e, &latent.absorb_v, &b.blended, &ws.attn_out, batch, (n * v, v));
        }
    }

    /// Queue `out_h = W_h x_h` for each head `h`, with `W_h` one head's `w` (the heads' matrices
    /// one after another), `x_h` its `[rows, cols]` part of `x` (the heads' parts `ws.rows` rows
    /// apart), and `out_h` its columns of `out`, whose rows are `out_stride` apart and whose heads
    /// are `head_offset` columns apart.
    fn matmul_heads(
        &self,
        e: &Encoder,
        w: &Matrix,
        x: &Buffer,
        out: &Buffer,
        batch: usize,
        (out_stride, head_offset): (usize, usize),
    ) {
        let k = &self.kernels;
        let tiles = w.rows.div_ceil(MM_BM) * batch.div_ceil(MM_BN) * self.n_heads;
        let mma = batch >= MMA_MIN_BATCH && tiles >= MMA_MIN_TILES;
        let (pipeline, groups, threads) = match (mma, w.bf16) {
            (false, bf16) => {
                let pipeline = match bf16 {
                    true => &k.matmul_heads_vec_bf16,
                    false => &k.matmul_heads_vec_f32,
                };
                let threads = threads(pipeline, VEC_THREADS);
                let rows_per_group = threads / SIMD * VEC_ROWS;
                let groups = [w.rows.div_ceil(rows_per_group), batch.div_ceil(VEC_BATCH)];
                (pipeline, groups, threads)
            }
            (true, bf16) => {
                let pipeline = match bf16 {
                    true => &k.matmul_heads_mma_bf16,
                    false => &k.matmul_heads_mma_f32,
                };
                (pipeline, [w.rows.div_ceil(MM_BM), batch.div_ceil(MM_BN)], MM_THREADS)
            }
        };
        e.set_pipeline(pipeline);
        e.set_buffer(0, &w.data, 0);
        e.set_buffer(1, x, 0);
        e.set_buffer(2, out, 0);
        e.set_buffer(3, &w.data, 0);
        e.set_bytes(
            4,
            &MatmulArgs {
                m: w.rows as u32,
                n: w.cols as u32,
                batch: batch as u32,
                scale: w.scale,
                accumulate: 0,
                has_bias: 0,
                out_stride: out_stride as u32,
            },
        );
        e.set_bytes(
            5,
            &HeadArgs {
                w_stride: (w.rows * w.cols) as u32,
                x_stride: (self.ws.rows * w.cols) as u32,
                out_offset: head_offset as u32,
            },
        );
        e.dispatch_3d([groups[0], groups[1], self.n_heads], threads);
    }

    /// Queue keeping the residual stream after layer `l`, if it's one of those asked for.
    fn keep_features(&self, e: &Encoder, l: usize, batch: usize) {
        let Some(k) = self.capture.iter().position(|&c| c == l) else {
            return;
        };
        let dim = self.shape.dim;
        e.set_pipeline(&self.kernels.copy_rows);
        e.set_buffer(0, &self.ws.x, 0);
        e.set_buffer(1, &self.features, 0);
        e.set_bytes(
            2,
            &CopyArgs {
                dim: dim as u32,
                stride: (self.capture.len() * dim) as u32,
                offset: (k * dim) as u32,
            },
        );
        e.dispatch([batch, 1], threads(&self.kernels.copy_rows, ROW_THREADS));
    }

    /// Queue an EAGLE head's start (see [`EagleParts`]): with each row's embedding in `ws.x`,
    /// make its hidden state (through `fc`, if it's the model's features) the residual stream,
    /// and put the two, normalized, side by side in `wide`.
    fn eagle_input(&self, e: &Encoder, eagle: &EagleParts, batch: usize) {
        let (ws, k, dim) = (&self.ws, &self.kernels, self.shape.dim);
        let hidden = match eagle.width == 3 * dim {
            true => {
                self.matmul(e, &eagle.fc, &eagle.hidden_in, &ws.added, batch, false);
                &ws.added
            }
            false => &eagle.hidden_in,
        };
        e.set_pipeline(&k.eagle_input);
        e.set_buffer(0, &ws.x, 0);
        e.set_buffer(1, hidden, 0);
        e.set_buffer(2, &self.layers[0].attn_norm, 0);
        e.set_buffer(3, &eagle.hidden_norm, 0);
        e.set_buffer(4, &eagle.wide, 0);
        e.set_bytes(
            5,
            &NormArgs {
                dim: dim as u32,
                eps: self.eps,
                select: 0,
                stride: dim as u32,
            },
        );
        e.dispatch([batch, 1], NORM_THREADS);
        e.set_pipeline(&k.copy_rows);
        e.set_buffer(0, hidden, 0);
        e.set_buffer(1, &ws.x, 0);
        e.set_bytes(
            2,
            &CopyArgs {
                dim: dim as u32,
                stride: dim as u32,
                offset: 0,
            },
        );
        e.dispatch([batch, 1], threads(&k.copy_rows, ROW_THREADS));
    }

    /// Queue `out = act(gate) * up` for `rows` rows of `gate_up`, `hidden` wide.
    fn gated(&self, e: &Encoder, gate_up: &Buffer, out: &Buffer, hidden: usize, rows: usize) {
        let k = &self.kernels;
        e.set_pipeline(&k.gated);
        e.set_buffer(0, gate_up, 0);
        e.set_buffer(1, out, 0);
        e.set_bytes(
            2,
            &GatedArgs {
                hidden: hidden as u32,
                gelu: (self.architecture.activation == Activation::Gelu) as u32,
            },
        );
        let threads = threads(&k.gated, ROW_THREADS);
        e.dispatch([hidden.div_ceil(threads), rows], threads);
    }

    /// Queue a mixture of experts for the normalized inputs in `ws.xb`, adding what each row's
    /// experts make of it, weighted (and what any shared expert does), to `ws.x`.
    fn experts(&self, e: &Encoder, experts: &Experts, batch: usize, on: &dyn Fn(Part) -> bool) {
        let (ws, k) = (&self.ws, &self.kernels);
        let moe = ws.moe.as_ref().expect("allocated for a mixture of experts");
        let spec = (self.architecture.experts.as_ref()).expect("a sparse layer has experts");
        let (count, top_k, hidden, dim) = (spec.count, spec.top_k, spec.hidden, self.shape.dim);
        // Experts with many rows each go through the tiled kernel, in groups of its size.
        let mma = batch * top_k >= MMA_MIN_BATCH * count;
        let group_rows = if mma { MM_BN } else { VEC_BATCH };
        let slots = (batch * top_k + count * (group_rows - 1)).next_multiple_of(group_rows);
        if on(Part::Matmuls) {
            self.matmul(e, &experts.router, &ws.xb, &moe.scores, batch, false);
        }
        if on(Part::Other) {
            let (deepseek, sigmoid, groups, top_groups, renormalize, scale) = match spec.routing {
                Routing::Softmax { .. } => (false, false, 1, 1, spec.renormalize, 1.0),
                Routing::DeepSeek {
                    sigmoid,
                    groups,
                    top_groups,
                    renormalize,
                    scale,
                } => (true, sigmoid, groups, top_groups, renormalize, scale),
            };
            e.set_pipeline(&k.moe_route);
            e.set_buffer(0, &moe.scores, 0);
            e.set_buffer(1, experts.correction.as_ref().unwrap_or(&moe.scores), 0);
            e.set_buffer(2, &moe.topk_ids, 0);
            e.set_buffer(3, &moe.topk_weights, 0);
            e.set_bytes(
                4,
                &RouteArgs {
                    experts: count as u32,
                    top_k: top_k as u32,
                    renormalize: renormalize as u32,
                    deepseek: deepseek as u32,
                    sigmoid: sigmoid as u32,
                    groups: groups as u32,
                    top_groups: top_groups as u32,
                    scale,
                    has_correction: experts.correction.is_some() as u32,
                },
            );
            e.dispatch([batch, 1], SIMD);

            e.set_pipeline(&k.moe_sort);
            e.set_buffer(0, &moe.topk_ids, 0);
            e.set_buffer(1, &moe.sorted_rows, 0);
            e.set_buffer(2, &moe.slot_of, 0);
            e.set_buffer(3, &moe.group_expert, 0);
            e.set_buffer(4, &moe.num_groups, 0);
            e.set_bytes(
                5,
                &SortArgs {
                    rows: batch as u32,
                    top_k: top_k as u32,
                    experts: count as u32,
                    group_rows: group_rows as u32,
                },
            );
            e.dispatch([1, 1], threads(&k.moe_sort, ROW_THREADS));

            e.set_pipeline(&k.moe_gather);
            e.set_buffer(0, &ws.xb, 0);
            e.set_buffer(1, &moe.sorted_rows, 0);
            e.set_buffer(2, &moe.num_groups, 0);
            e.set_buffer(3, &moe.x, 0);
            e.set_bytes(
                4,
                &GatherArgs {
                    dim: dim as u32,
                    group_rows: group_rows as u32,
                },
            );
            e.dispatch([slots, 1], threads(&k.moe_gather, 2 * SIMD));
        }
        if on(Part::Matmuls) {
            self.moe_matmul(e, &experts.gate_up, &moe.x, &moe.gate_up, slots, mma);
        }
        if on(Part::Other) {
            self.gated(e, &moe.gate_up, &moe.gated, hidden, slots);
        }
        if on(Part::Matmuls) {
            self.moe_matmul(e, &experts.down, &moe.gated, &moe.down, slots, mma);
        }
        if on(Part::Other) {
            e.set_pipeline(&k.moe_combine);
            e.set_buffer(0, &moe.down, 0);
            e.set_buffer(1, &moe.slot_of, 0);
            e.set_buffer(2, &moe.topk_weights, 0);
            e.set_buffer(3, &ws.x, 0);
            e.set_bytes(
                4,
                &CombineArgs {
                    dim: dim as u32,
                    top_k: top_k as u32,
                },
            );
            let threads = threads(&k.moe_combine, ROW_THREADS);
            e.dispatch([dim.div_ceil(threads), batch], threads);
        }
        // Every row goes through a shared expert as well, as through a dense MLP.
        let Some((gate_up, down, gate)) = &experts.shared else {
            return;
        };
        if on(Part::Matmuls) {
            self.matmul(e, gate_up, &ws.xb, &ws.gate_up, batch, false);
        }
        if on(Part::Other) {
            self.gated(e, &ws.gate_up, &ws.gated, down.cols, batch);
        }
        let Some(gate) = gate else {
            if on(Part::Matmuls) {
                self.matmul(e, down, &ws.gated, &ws.x, batch, true);
            }
            return;
        };
        // Scaled by its own gate's sigmoid (Qwen2 MoE).
        if on(Part::Matmuls) {
            self.matmul(e, gate, &ws.xb, &moe.shared_gate, batch, false);
            self.matmul(e, down, &ws.gated, &ws.added, batch, false);
        }
        if on(Part::Other) {
            e.set_pipeline(&k.add_gated);
            e.set_buffer(0, &ws.added, 0);
            e.set_buffer(1, &moe.shared_gate, 0);
            e.set_buffer(2, &ws.x, 0);
            e.set_bytes(3, &(dim as u32));
            let threads = threads(&k.add_gated, ROW_THREADS);
            e.dispatch([dim.div_ceil(threads), batch], threads);
        }
    }

    /// Queue `out = W_e x` for each group of rows of `x` (whose rows are in expert order, `slots`
    /// of them at most) and its expert `e`, with the experts' weights one after another in `w`.
    fn moe_matmul(
        &self,
        e: &Encoder,
        w: &Matrix,
        x: &Buffer,
        out: &Buffer,
        slots: usize,
        mma: bool,
    ) {
        let k = &self.kernels;
        let moe = (self.ws.moe.as_ref()).expect("allocated for a mixture of experts");
        let (pipeline, groups, threads) = match (mma, w.bf16) {
            (false, bf16) => {
                let pipeline = match bf16 {
                    true => &k.moe_matmul_vec_bf16,
                    false => &k.moe_matmul_vec_f32,
                };
                let threads = threads(pipeline, VEC_THREADS);
                let rows_per_group = threads / SIMD * VEC_ROWS;
                let groups = [w.rows.div_ceil(rows_per_group), slots / VEC_BATCH];
                (pipeline, groups, threads)
            }
            (true, bf16) => {
                let pipeline = match bf16 {
                    true => &k.moe_matmul_mma_bf16,
                    false => &k.moe_matmul_mma_f32,
                };
                (pipeline, [w.rows.div_ceil(MM_BM), slots / MM_BN], MM_THREADS)
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
                batch: slots as u32,
                scale: w.scale,
                accumulate: 0,
                has_bias: w.bias.is_some() as u32,
                out_stride: w.rows as u32,
            },
        );
        e.set_buffer(5, &moe.group_expert, 0);
        e.set_buffer(6, &moe.num_groups, 0);
        e.dispatch(groups, threads);
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
                stride: self.shape.dim as u32,
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
                stride: self.shape.dim as u32,
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
            out_stride: w.rows as u32,
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

    fn capture_layers(&mut self, layers: &[usize]) -> Result<()> {
        ensure!(
            layers.iter().all(|&l| l < self.layers.len()),
            "there are only {} layers",
            self.layers.len()
        );
        self.capture = layers.to_vec();
        Ok(())
    }

    fn features(&self, row: usize) -> &[f32] {
        let width = self.capture.len() * self.shape.dim;
        if width == 0 || row >= self.last_rows {
            return &[];
        }
        // SAFETY: the GPU finished writing them before `forward` returned.
        let all = unsafe { self.features.slice::<f32>() };
        &all[row * width..(row + 1) * width]
    }

    fn set_hidden_inputs(&mut self, width: usize, values: &[f32]) -> Result<()> {
        let dim = self.shape.dim;
        let Some(eagle) = &mut self.eagle else {
            anyhow::bail!("{} isn't an EAGLE head", self.gpu_name);
        };
        ensure!(
            width == dim || width == 3 * dim,
            "an EAGLE head takes hidden states {dim} or {} wide",
            3 * dim
        );
        reserve(&self.device, &mut eagle.hidden_in, values.len().max(1) * 4)?;
        // SAFETY: the GPU isn't using it between forwards.
        let memory = unsafe { eagle.hidden_in.slice_mut::<f32>() };
        memory[..values.len()].copy_from_slice(values);
        eagle.width = width;
        Ok(())
    }

    fn add_adapter(&mut self, adapter: &LoraAdapter) -> Result<()> {
        ensure!(
            self.architecture.latent.is_none(),
            "LoRA adapters for DeepSeek's latent attention aren't supported"
        );
        ensure!(
            adapter.layers.len() == self.layers.len(),
            "{}: made for a model with {} layers, not {}",
            adapter.name,
            adapter.layers.len(),
            self.layers.len()
        );
        for (l, layer) in self.layers.iter().enumerate() {
            let changes_mlp = [Module::Gate, Module::Up, Module::Down]
                .iter()
                .any(|&m| adapter.layers[l][m as usize].is_some());
            ensure!(
                !(changes_mlp && matches!(layer.feed_forward, FeedForward::Sparse(_))),
                "{}: adapters for mixtures of experts' MLPs aren't supported",
                adapter.name
            );
        }
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
