//! Running the model on an NVIDIA GPU with CUDA.
//!
//! This does the same work as the CPU forward pass in [`crate::model`], one kernel per step. The
//! kernels are in `kernels.cu`. Weights and the KV cache live in GPU memory; each step uploads
//! what's needed to describe the batch, queues the kernels, and reads back the logits.
//!
//! Weights can be kept as 16-bit bfloat16 (the default) or as 32-bit floats. 16-bit weights go
//! through the GPU's tensor cores, which are several times faster, and take half the memory.
//! Most checkpoints are stored in bfloat16 anyway, so nothing is lost by that; the difference is
//! that the input to each matrix multiplication gets rounded to 16 bits as well.

mod api;

pub use api::{Arg, Gpu, Graph, Kernel, Module, Pointer};

use crate::backend::{Backend, LoadedModel, Logits};
use crate::config::{Activation, Architecture, Config, ExpertNaming, Experts as ConfigExperts};
use crate::lora::{LoraAdapter, LoraPair, Module as LoraModule};
use crate::model::{LogitUse, Row};
use crate::pool::Pool;
use crate::safetensors::{FP8_BLOCK, SafeTensors, bf16_to_f32, f32_to_bf16};
use anyhow::{Context, Result, bail, ensure};
use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;

pub const KERNELS_PTX: &str = include_str!("kernels.ptx");

/// The most tokens read back per row when the rest can't matter. It's plenty for most text, and
/// if a row has more in contention, all its logits are read after all.
const MAX_CANDIDATES: usize = 4096;

/// Input and output buffers have room for this many more rows than the batch, for the padding
/// rows that the matrix kernels can touch when the last group of rows isn't full.
const PADDING_ROWS: usize = 64;

/// The 32-bit kernel multiplies this many weight rows at a time, so matrices are padded to
/// multiples of it.
const BLOCK_ROWS: usize = 8;

/// The 16-bit kernels work on tiles this wide and tall.
const TILE: usize = 16;

/// Weight tiles per block in the 16-bit kernels (`WARPS_ACROSS` in `kernels.cu`).
const WARPS_ACROSS: usize = 4;
/// Columns per block in the 16-bit SwiGLU kernel (`SWIGLU_COLUMNS` in `kernels.cu`).
const SWIGLU_COLUMNS: usize = 256;
/// Threads per block in the kernels that go through all of a row's logits (`SCAN_THREADS`).
const SCAN_THREADS: usize = 1024;
/// The largest batch to keep a recorded graph for. Each takes about 2 MB of GPU memory.
const MAX_RECORDED_BATCH: usize = 64;

/// How many 16-element chunks a head can have. `kernels.cu` has an attention kernel for each.
const TILE_CHUNK_COUNTS: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 16];

/// `kernels.cu` holds attention accumulators for at most this many warps of a head.
const MAX_HEAD_DIM: usize = 16 * 32;

/// GPU memory to leave free when sizing the KV cache by default.
const RESERVED_BYTES: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 16-bit weights on tensor cores: much faster, and accurate enough for serving.
    Bf16,
    /// 32-bit floats throughout, matching a 32-bit reference run as closely as the CPU does.
    F32,
}

/// A way of storing weights in fewer bits than they come in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    /// One signed byte per weight, with a scale for each group of 128 along a row, for the
    /// matrices inside each layer. The embeddings and output projection stay 16-bit.
    Int8,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub precision: Precision,
    /// Store the layers' weights in fewer bits (needs 16-bit precision).
    pub quantization: Option<Quantization>,
    /// The most tokens that will ever be passed to `forward` at once.
    pub max_batch: usize,
    /// The KV cache's size, or `None` to use the memory that's left over.
    pub cache_bytes: Option<usize>,
    /// CPU threads for work between GPU steps (choosing the next tokens).
    pub host_threads: usize,
}

/// A weight matrix on the GPU.
#[derive(Clone, Copy)]
struct Matrix {
    data: Pointer,
    rows: usize,
    cols: usize,
    bias: Pointer,
    /// Working space for 16-bit multiplication, when `rows` isn't a multiple of 16.
    scratch: Pointer,
    /// For 8-bit weights, the scale of each group of 128 in each row: weight = byte * scale.
    /// Null otherwise.
    scales: Pointer,
    /// 8-bit weights are floats (e4m3) rather than integers.
    fp8: bool,
    /// For 4-bit weights, the zero point of each group of 128 in each row, as bytes: weight =
    /// (value - zero) * scale. Null otherwise.
    zeros: Pointer,
}

struct Layer {
    attn_norm: Pointer,
    /// q_proj, k_proj and v_proj as one matrix, since they all take the same input.
    qkv: Matrix,
    /// For normalizing each head of the queries and of the keys, if the model does that.
    head_norms: Option<(Pointer, Pointer)>,
    o_proj: Matrix,
    /// Normalizes what attention adds to the residual stream, if the model does that.
    norm_after_attn: Option<Pointer>,
    mlp_norm: Pointer,
    feed_forward: FeedForward,
    /// The same for the MLP.
    norm_after_mlp: Option<Pointer>,
    k_pool: Pointer,
    v_pool: Pointer,
    rope_table: usize,
    /// How many tokens it can see.
    window: usize,
}

enum FeedForward {
    /// gate_proj and up_proj as one matrix, for the same reason as `qkv`, then down_proj.
    Dense {
        gate_up: Matrix,
        down: Matrix,
    },
    Sparse(Experts),
}

/// A layer's mixture of experts, all its experts' weights in the 16-bit tile layout, side by side.
struct Experts {
    router: Matrix,
    /// Each expert's gate_proj and up_proj stacked, `[2 * hidden, dim]`.
    gate_up: Pointer,
    /// Each expert's down_proj, `[dim, hidden]`.
    down: Pointer,
}

/// Working memory for mixtures of experts.
struct MoeBuffers {
    scores: Pointer,
    topk_ids: Pointer,
    topk_weights: Pointer,
    slot_of: Pointer,
    sorted_rows: Pointer,
    group_expert: Pointer,
    num_groups: Pointer,
    /// The rows in expert order, 16-bit tiles `[slots, dim]`.
    x: Pointer,
    gate_up: Pointer,
    gated: Pointer,
    down: Pointer,
}

struct Kernels {
    embed: Kernel,
    rmsnorm: Kernel,
    matmul_1: Kernel,
    matmul_2: Kernel,
    matmul_4: Kernel,
    matmul_8: Kernel,
    rope_store: Kernel,
    attention: Kernel,
    swiglu: Kernel,
    geglu: Kernel,
    head_norm: Kernel,
    finish_logits: Kernel,
    rmsnorm_add: Kernel,
    embed_bf16: Kernel,
    rmsnorm_bf16: Kernel,
    matmul_bf16_1: Kernel,
    matmul_bf16_2: Kernel,
    matmul_bf16_4: Kernel,
    matmul_i8_1: Kernel,
    matmul_i8_2: Kernel,
    matmul_i8_4: Kernel,
    matmul_f8_1: Kernel,
    matmul_f8_2: Kernel,
    matmul_f8_4: Kernel,
    matmul_q4_1: Kernel,
    matmul_q4_2: Kernel,
    matmul_q4_4: Kernel,
    lora_shrink: Kernel,
    lora_expand: Kernel,
    swiglu_bf16: Kernel,
    geglu_bf16: Kernel,
    rope_store_bf16: Kernel,
    /// One for each entry of `TILE_CHUNK_COUNTS`.
    attention_bf16: Vec<Kernel>,
    top_candidates: Kernel,
    draw_tokens: Kernel,
    moe_route: Kernel,
    moe_sort: Kernel,
    moe_gather: Kernel,
    moe_matmul_1: Kernel,
    moe_matmul_2: Kernel,
    moe_matmul_4: Kernel,
    moe_swiglu: Kernel,
    moe_geglu: Kernel,
    moe_combine: Kernel,
}

impl Kernels {
    fn find(gpu: &Gpu, module: &Module) -> Result<Self> {
        Ok(Self {
            embed: gpu.kernel(module, "embed")?,
            rmsnorm: gpu.kernel(module, "rmsnorm")?,
            matmul_1: gpu.kernel(module, "matmul_1")?,
            matmul_2: gpu.kernel(module, "matmul_2")?,
            matmul_4: gpu.kernel(module, "matmul_4")?,
            matmul_8: gpu.kernel(module, "matmul_8")?,
            rope_store: gpu.kernel(module, "rope_store")?,
            attention: gpu.kernel(module, "attention")?,
            swiglu: gpu.kernel(module, "swiglu")?,
            geglu: gpu.kernel(module, "geglu")?,
            head_norm: gpu.kernel(module, "head_norm")?,
            finish_logits: gpu.kernel(module, "finish_logits")?,
            rmsnorm_add: gpu.kernel(module, "rmsnorm_add")?,
            embed_bf16: gpu.kernel(module, "embed_bf16")?,
            rmsnorm_bf16: gpu.kernel(module, "rmsnorm_bf16")?,
            matmul_bf16_1: gpu.kernel(module, "matmul_bf16_1")?,
            matmul_bf16_2: gpu.kernel(module, "matmul_bf16_2")?,
            matmul_bf16_4: gpu.kernel(module, "matmul_bf16_4")?,
            matmul_i8_1: gpu.kernel(module, "matmul_i8_1")?,
            matmul_i8_2: gpu.kernel(module, "matmul_i8_2")?,
            matmul_i8_4: gpu.kernel(module, "matmul_i8_4")?,
            matmul_f8_1: gpu.kernel(module, "matmul_f8_1")?,
            matmul_f8_2: gpu.kernel(module, "matmul_f8_2")?,
            matmul_f8_4: gpu.kernel(module, "matmul_f8_4")?,
            matmul_q4_1: gpu.kernel(module, "matmul_q4_1")?,
            matmul_q4_2: gpu.kernel(module, "matmul_q4_2")?,
            matmul_q4_4: gpu.kernel(module, "matmul_q4_4")?,
            lora_shrink: gpu.kernel(module, "lora_shrink")?,
            lora_expand: gpu.kernel(module, "lora_expand")?,
            moe_route: gpu.kernel(module, "moe_route")?,
            moe_sort: gpu.kernel(module, "moe_sort")?,
            moe_gather: gpu.kernel(module, "moe_gather")?,
            moe_matmul_1: gpu.kernel(module, "moe_matmul_1")?,
            moe_matmul_2: gpu.kernel(module, "moe_matmul_2")?,
            moe_matmul_4: gpu.kernel(module, "moe_matmul_4")?,
            moe_swiglu: gpu.kernel(module, "moe_swiglu")?,
            moe_geglu: gpu.kernel(module, "moe_geglu")?,
            moe_combine: gpu.kernel(module, "moe_combine")?,
            swiglu_bf16: gpu.kernel(module, "swiglu_bf16")?,
            geglu_bf16: gpu.kernel(module, "geglu_bf16")?,
            rope_store_bf16: gpu.kernel(module, "rope_store_bf16")?,
            attention_bf16: TILE_CHUNK_COUNTS
                .iter()
                .map(|chunks| gpu.kernel(module, &format!("attention_bf16_{chunks}")))
                .collect::<Result<_>>()?,
            top_candidates: gpu.kernel(module, "top_candidates")?,
            draw_tokens: gpu.kernel(module, "draw_tokens")?,
        })
    }
}

/// A GPU buffer that grows when it needs to.
#[derive(Default)]
struct Growing {
    data: Pointer,
    capacity: usize,
}

impl Growing {
    /// Returns whether the buffer moved. The caller has to have waited for the GPU to finish
    /// with the old contents.
    fn reserve(&mut self, gpu: &Gpu, elements: usize) -> Result<bool> {
        if elements <= self.capacity {
            return Ok(false);
        }
        if self.capacity > 0 {
            gpu.free(self.data)?;
        }
        self.capacity = elements.next_power_of_two().max(1 << 16);
        self.data = gpu.alloc(self.capacity)?;
        Ok(true)
    }
}

/// Everything uploaded for one step, as `[u32]`, side by side in one buffer.
#[derive(Default)]
struct StepData {
    /// How far behind the best a token can be, per row that wants logits (negative = not this way).
    gaps: Vec<f32>,
    /// How many of the top tokens are enough, per row that wants logits (0 = no limit).
    limits: Vec<u32>,
    /// The temperature to draw a token at, per row that wants logits (negative = not this way).
    temperatures: Vec<f32>,
    /// The two halves of the key for that draw.
    keys: Vec<u32>,
    /// Whether each row that wants logits wants its final hidden state instead.
    hidden: Vec<bool>,
    tokens: Vec<u32>,
    write_slots: Vec<u32>,
    row_offset: Vec<u32>,
    row_len: Vec<u32>,
    /// Each row's LoRA adapter, 0 for none.
    adapters: Vec<u32>,
    team_size: Vec<u32>,
    scratch_offset: Vec<u32>,
    logit_rows: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    combined: Vec<u32>,
    /// The KV slots each sequence in the batch can see, concatenated.
    slot_lists: Vec<u32>,
}

pub struct CudaBackend {
    gpu: Gpu,
    kernels: Kernels,
    precision: Precision,
    quantization: Option<Quantization>,
    embeddings: Pointer,
    lm_head: Matrix,
    norm: Pointer,
    layers: Vec<Layer>,
    architecture: Architecture,
    /// What embeddings get multiplied by, at this precision.
    embedding_scale: f32,

    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    eps: f32,
    max_batch: usize,
    capacity: usize,

    x: Pointer,
    xb: Pointer,
    /// What a block adds to the residual stream, for models that normalize it first.
    added: Pointer,
    qkv: Pointer,
    attn_out: Pointer,
    gate_up: Pointer,
    gated: Pointer,
    logits_device: Pointer,
    step_buffer: Pointer,
    moe: Option<MoeBuffers>,

    slot_list_buffer: Growing,
    scratch: Growing,
    step: StepData,
    /// The kernels of a forward pass, recorded once per batch size, then replayed. That saves
    /// a little GPU time per kernel, which only matters when the batch is small.
    graphs: HashMap<(usize, bool), Graph>,

    /// Two entries per row that wants logits: how many candidates it has, and the token drawn.
    results_device: Pointer,
    candidate_values_device: Pointer,
    candidate_ids_device: Pointer,
    results: Vec<u32>,
    candidate_values: Vec<f32>,
    candidate_ids: Vec<u32>,
    candidates: Vec<(f32, u32)>,
    /// Where each row's candidates start in `candidates`; `u32::MAX` if its logits are in `logits`.
    candidate_start: Vec<u32>,
    candidate_end: Vec<u32>,
    /// Which logit rows were read in full, and where in `logits`.
    full_rows: Vec<usize>,
    logits: Vec<f32>,
    /// Final hidden states of the rows that want logits, when any want those.
    hidden: Vec<f32>,
    pool: Pool,
    gpu_name: String,
    /// LoRA adapters' corrections as given, for each layer and module, the first being adapter 1.
    adapters: Vec<Vec<[Option<LoraPair>; 7]>>,
    /// The same on the GPU, for each layer and each of `LORA_TARGETS`.
    lora: Vec<[LoraTarget; 4]>,
    /// Working space for `A x`, `[rows, largest rank]`.
    lora_hidden: Pointer,
}

/// The matrices LoRA corrections go to, as stacked here, and the modules each stacks.
const LORA_TARGETS: [&[LoraModule]; 4] = [
    &[LoraModule::Q, LoraModule::K, LoraModule::V],
    &[LoraModule::O],
    &[LoraModule::Gate, LoraModule::Up],
    &[LoraModule::Down],
];

/// Every adapter's correction to one stacked matrix: `A` as `[adapters, rank, in]` and `B`
/// transposed, `[adapters, rank, out]`, padded with zeros to the largest rank. A rank of 0 means
/// no adapter changes this matrix.
#[derive(Clone, Copy, Default)]
struct LoraTarget {
    a: Pointer,
    b: Pointer,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
}

impl CudaBackend {
    /// Whether a usable GPU is present.
    pub fn available() -> bool {
        Gpu::open().is_ok()
    }

    /// Load `config.json` and the weights onto the GPU.
    pub fn load(dir: &Path, options: &Options) -> Result<LoadedModel<Self>> {
        let config = Config::load(dir)?;
        let st = SafeTensors::open(dir)?;
        let gpu = Gpu::open()?;
        gpu.attach()?;
        let gpu_name = gpu.name();

        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.n_kv_heads();
        let head_dim = config.head_dim();
        let hidden_dim = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        let architecture = config.architecture()?;
        if head_dim > MAX_HEAD_DIM {
            bail!("head size {head_dim} is too large for the GPU kernels (limit {MAX_HEAD_DIM})");
        }

        let module = gpu
            .load_kernels(KERNELS_PTX)
            .context("the GPU backend needs an Ampere (2020) or newer GPU")?;
        let kernels = Kernels::find(&gpu, &module)?;
        // 16-bit attention works in tiles: all the query heads sharing a key/value head have to
        // fit in one tile's 16 rows, and heads have to be a whole number of 16-element chunks.
        let fits_tiles = n_heads / n_kv_heads <= TILE
            && head_dim.is_multiple_of(TILE)
            && TILE_CHUNK_COUNTS.contains(&(head_dim / TILE));
        if let Some(experts) = &architecture.experts {
            ensure!(
                experts.shared_hidden.is_none(),
                "mixtures of experts with a shared expert only run on the CPU so far"
            );
            ensure!(
                options.precision == Precision::Bf16,
                "mixtures of experts need 16-bit precision on the GPU"
            );
            ensure!(
                experts.top_k <= 16 && experts.count <= 512,
                "the GPU handles at most 512 experts and 16 per token"
            );
        }
        let precision = if options.precision == Precision::Bf16 && !fits_tiles {
            eprintln!("This model's attention heads don't fit 16-bit tiles, so using 32-bit.");
            Precision::F32
        } else {
            options.precision
        };
        // 8-bit weights feed the tensor cores, which take 16-bit inputs.
        let quantization = match (options.quantization, precision) {
            (Some(_), Precision::F32) => {
                eprintln!("8-bit weights need 16-bit precision, so they're staying 32-bit.");
                None
            }
            (quantization, _) => quantization,
        };
        let options = &Options {
            quantization,
            ..options.clone()
        };

        let batch_rows = options.max_batch + PADDING_ROWS;
        let mut num_params = 0;
        let vector_params = Cell::new(0);
        let upload_vector = |name: &str, len: usize| -> Result<Pointer> {
            // A head's norms can be missing, which leaves them doing nothing but normalize.
            let values: Vec<f32> = match st.contains(name) || len == dim {
                // Gemma's norm weights are stored as an offset from 1.
                true => (st.load(name, &[len])?.into_iter())
                    .map(|v| v + architecture.norm_offset)
                    .collect(),
                false => vec![1.0; len],
            };
            vector_params.set(vector_params.get() + len);
            let device = gpu.alloc(len)?;
            gpu.write_now(device, &values)?;
            Ok(device)
        };
        let norm = upload_vector("model.norm.weight", dim)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (l, shape) in architecture.layers.iter().enumerate() {
            let p = format!("model.layers.{l}");
            let attn = format!("{p}.self_attn");
            let attn_norm = upload_vector(&format!("{p}.input_layernorm.weight"), dim)?;
            let head_norms = match (architecture.head_norms, architecture.full_qk_norms) {
                (true, _) => Some((
                    upload_vector(&format!("{attn}.q_norm.weight"), head_dim)?,
                    upload_vector(&format!("{attn}.k_norm.weight"), head_dim)?,
                )),
                // A whole token's queries, and keys, normalized together (OLMoE).
                (_, true) => Some((
                    upload_vector(&format!("{attn}.q_norm.weight"), q_dim)?,
                    upload_vector(&format!("{attn}.k_norm.weight"), kv_dim)?,
                )),
                _ => None,
            };
            let (norm_after_attn, mlp_norm, norm_after_mlp) = if architecture.norms_after_blocks {
                (
                    Some(upload_vector(
                        &format!("{p}.post_attention_layernorm.weight"),
                        dim,
                    )?),
                    upload_vector(&format!("{p}.pre_feedforward_layernorm.weight"), dim)?,
                    Some(upload_vector(
                        &format!("{p}.post_feedforward_layernorm.weight"),
                        dim,
                    )?),
                )
            } else {
                // Elsewhere this name means the norm in front of the MLP.
                (
                    None,
                    upload_vector(&format!("{p}.post_attention_layernorm.weight"), dim)?,
                    None,
                )
            };
            let qkv_names = match architecture.combined_projections {
                true => vec![(format!("{attn}.qkv_proj"), q_dim + 2 * kv_dim)],
                false => vec![
                    (format!("{attn}.q_proj"), q_dim),
                    (format!("{attn}.k_proj"), kv_dim),
                    (format!("{attn}.v_proj"), kv_dim),
                ],
            };
            let qkv = upload_stacked(
                &gpu,
                &st,
                precision,
                &mut num_params,
                &qkv_names,
                dim,
                batch_rows,
                1.0,
                options.quantization,
            )?;
            let o_proj = upload_stacked(
                &gpu,
                &st,
                precision,
                &mut num_params,
                &[(format!("{attn}.o_proj"), dim)],
                q_dim,
                batch_rows,
                architecture.residual_scale,
                options.quantization,
            )?;
            let feed_forward = match (shape.sparse, &architecture.experts) {
                (true, Some(spec)) => FeedForward::Sparse(upload_experts(
                    &gpu,
                    &st,
                    &mut num_params,
                    &p,
                    spec,
                    dim,
                    batch_rows,
                    architecture.residual_scale,
                )?),
                _ => {
                    let gate_up_names = match architecture.combined_projections {
                        true => vec![(format!("{p}.mlp.gate_up_proj"), 2 * hidden_dim)],
                        false => vec![
                            (format!("{p}.mlp.gate_proj"), hidden_dim),
                            (format!("{p}.mlp.up_proj"), hidden_dim),
                        ],
                    };
                    let gate_up = upload_stacked(
                        &gpu,
                        &st,
                        precision,
                        &mut num_params,
                        &gate_up_names,
                        dim,
                        batch_rows,
                        1.0,
                        options.quantization,
                    )?;
                    let down = upload_stacked(
                        &gpu,
                        &st,
                        precision,
                        &mut num_params,
                        &[(format!("{p}.mlp.down_proj"), dim)],
                        hidden_dim,
                        batch_rows,
                        architecture.residual_scale,
                        options.quantization,
                    )?;
                    FeedForward::Dense { gate_up, down }
                }
            };
            layers.push(Layer {
                attn_norm,
                qkv,
                head_norms,
                o_proj,
                norm_after_attn,
                mlp_norm,
                feed_forward,
                norm_after_mlp,
                k_pool: Pointer::NULL,
                v_pool: Pointer::NULL,
                rope_table: shape.rope_table,
                window: shape.window.unwrap_or(i32::MAX as usize),
            });
        }

        let embed = upload_stacked(
            &gpu,
            &st,
            precision,
            &mut num_params,
            &[("model.embed_tokens".to_string(), vocab_size)],
            dim,
            batch_rows,
            1.0,
            None,
        )?;
        let lm_head = if st.contains("lm_head.weight") {
            upload_stacked(
                &gpu,
                &st,
                precision,
                &mut num_params,
                &[("lm_head".to_string(), vocab_size)],
                dim,
                batch_rows,
                1.0,
                None,
            )?
        } else {
            // Tied embeddings: the same matrix does both jobs.
            embed
        };

        // Working memory for the batch.
        let rows = options.max_batch + PADDING_ROWS;
        let logits_rows = options.max_batch;
        let x = gpu.alloc(rows * dim)?;
        let xb = input_buffer(&gpu, precision, rows, dim)?;
        // Also where final hidden states go, for embeddings.
        let added = gpu.alloc(rows * dim)?;
        let qkv = gpu.alloc(rows * (q_dim + 2 * kv_dim))?;
        let attn_out = input_buffer(&gpu, precision, rows, q_dim)?;
        let gate_up = gpu.alloc(rows * 2 * hidden_dim)?;
        let gated = input_buffer(&gpu, precision, rows, hidden_dim)?;
        let moe = match &architecture.experts {
            Some(spec) => {
                let pairs = rows * spec.top_k;
                let slots = moe_slots(rows, spec);
                Some(MoeBuffers {
                    scores: gpu.alloc(rows * spec.count)?,
                    topk_ids: gpu.alloc(pairs)?,
                    topk_weights: gpu.alloc(pairs)?,
                    slot_of: gpu.alloc(pairs)?,
                    sorted_rows: gpu.alloc(slots)?,
                    group_expert: gpu.alloc(slots / TILE)?,
                    num_groups: gpu.alloc(1)?,
                    x: input_buffer(&gpu, Precision::Bf16, slots, dim)?,
                    gate_up: gpu.alloc(slots * 2 * spec.hidden)?,
                    gated: input_buffer(&gpu, Precision::Bf16, slots, spec.hidden)?,
                    down: gpu.alloc(slots * dim)?,
                })
            }
            None => None,
        };
        let logits_device = gpu.alloc((logits_rows + PADDING_ROWS) * vocab_size)?;
        let results_device = gpu.alloc(options.max_batch * 2)?;
        let candidate_values_device = gpu.alloc(options.max_batch * MAX_CANDIDATES)?;
        let candidate_ids_device = gpu.alloc(options.max_batch * MAX_CANDIDATES)?;
        let rope_slots = 2 * options.max_batch * (head_dim / 2) * architecture.rope_tables.len();
        let step_slots = 14 * options.max_batch + rope_slots + 16;
        let step_buffer = gpu.alloc(step_slots)?;

        // The KV cache gets what's left, unless told otherwise.
        let element_bytes = match precision {
            Precision::Bf16 => 2,
            Precision::F32 => 4,
        };
        let kv_bytes_per_token = 2 * config.num_hidden_layers * kv_dim * element_bytes;
        let cache_bytes = match options.cache_bytes {
            Some(bytes) => bytes,
            None => gpu.free_memory()?.saturating_sub(RESERVED_BYTES),
        };
        let capacity = (cache_bytes / kv_bytes_per_token).min(1 << 24);
        // 16-bit pools are organized in whole pages of 16 slots.
        let capacity = match precision {
            Precision::Bf16 => capacity - capacity % TILE,
            Precision::F32 => capacity,
        };
        if capacity < 16 {
            bail!("not enough GPU memory left for the KV cache");
        }
        for layer in &mut layers {
            let elements = (capacity * kv_dim * element_bytes).div_ceil(4);
            layer.k_pool = gpu
                .alloc(elements)
                .context("couldn't allocate the KV cache; try a smaller --cache-mb")?;
            layer.v_pool = gpu
                .alloc(elements)
                .context("couldn't allocate the KV cache; try a smaller --cache-mb")?;
        }

        let eps = config.rms_norm_eps;
        let embedding_scale = match precision {
            Precision::Bf16 => architecture.embedding_scale_bf16,
            Precision::F32 => architecture.embedding_scale,
        };
        Ok(LoadedModel {
            num_params: num_params + vector_params.get(),
            config,
            backend: Self {
                pool: Pool::new(options.host_threads),
                gpu,
                kernels,
                precision,
                quantization,
                embeddings: embed.data,
                lm_head,
                norm,
                layers,
                architecture,
                embedding_scale,
                dim,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden_dim,
                vocab_size,
                eps,
                max_batch: options.max_batch,
                capacity,
                x,
                xb,
                added,
                qkv,
                attn_out,
                gate_up,
                gated,
                moe,
                logits_device,
                step_buffer,
                slot_list_buffer: Growing::default(),
                scratch: Growing::default(),
                step: StepData::default(),
                graphs: HashMap::new(),
                results_device,
                candidate_values_device,
                candidate_ids_device,
                results: Vec::new(),
                candidate_values: Vec::new(),
                candidate_ids: Vec::new(),
                candidates: Vec::new(),
                candidate_start: Vec::new(),
                candidate_end: Vec::new(),
                full_rows: Vec::new(),
                logits: Vec::new(),
                hidden: Vec::new(),
                gpu_name,
                adapters: Vec::new(),
                lora: Vec::new(),
                lora_hidden: Pointer::NULL,
            },
        })
    }

    /// Queue `out = W x`, or `out += W x` if `accumulate`.
    fn matmul(
        &self,
        w: &Matrix,
        x: Pointer,
        batch: usize,
        out: Pointer,
        accumulate: bool,
    ) -> Result<()> {
        launch_matmul(
            &self.gpu,
            &self.kernels,
            self.precision,
            w,
            x,
            batch,
            out,
            accumulate,
        )
    }

    /// Queue the rotation of queries and keys, and storing keys and values in their KV slots.
    fn rope_and_store(&self, layer: &Layer, batch: usize, step: &StepPointers) -> Result<()> {
        let head_dim = self.head_dim;
        let kv_dim = self.n_kv_heads * head_dim;
        let args = [
            Arg::Ptr(self.qkv),
            Arg::Int(self.n_heads * head_dim + 2 * kv_dim),
            Arg::Ptr(step.cos.offset(layer.rope_table * batch * (head_dim / 2))),
            Arg::Ptr(step.sin.offset(layer.rope_table * batch * (head_dim / 2))),
            Arg::Int(self.n_heads),
            Arg::Int(self.n_kv_heads),
            Arg::Int(head_dim),
            Arg::Ptr(step.write_slots),
            Arg::Ptr(layer.k_pool),
            Arg::Ptr(layer.v_pool),
            Arg::Int(kv_dim / TILE),
        ];
        match self.precision {
            Precision::Bf16 => self
                .gpu
                .launch(self.kernels.rope_store_bf16, [batch, 1], &args),
            Precision::F32 => self
                .gpu
                .launch(self.kernels.rope_store, [batch, 1], &args[..10]),
        }
    }

    /// Queue the attention step, which leaves its result in `attn_out`.
    fn attention(&self, layer: &Layer, batch: usize, step: &StepPointers) -> Result<()> {
        let head_dim = self.head_dim;
        let q_dim = self.n_heads * head_dim;
        let kv_dim = self.n_kv_heads * head_dim;
        let stride = q_dim + 2 * kv_dim;
        let scale = self.architecture.attention_scale;
        let softcap = self.architecture.attention_softcap;
        let lists = self.slot_list_buffer.data;
        if self.precision == Precision::Bf16 {
            let tile_bytes = TILE * TILE * 2;
            let shared = (head_dim / TILE) * tile_bytes + 3 * tile_bytes;
            let chunks = head_dim / TILE;
            let kernel = TILE_CHUNK_COUNTS
                .iter()
                .position(|&count| count == chunks)
                .expect("checked when the model was loaded");
            return self.gpu.launch_shared(
                self.kernels.attention_bf16[kernel],
                [batch * self.n_kv_heads, 1],
                shared,
                &[
                    Arg::Ptr(self.qkv),
                    Arg::Int(stride),
                    Arg::Ptr(layer.k_pool),
                    Arg::Ptr(layer.v_pool),
                    Arg::Int(kv_dim / TILE),
                    Arg::Ptr(lists),
                    Arg::Ptr(step.row_offset),
                    Arg::Ptr(step.row_len),
                    Arg::Ptr(step.team_size),
                    Arg::Ptr(self.attn_out),
                    Arg::Int(self.n_heads),
                    Arg::Int(self.n_kv_heads),
                    Arg::Int(head_dim),
                    Arg::Float(scale),
                    Arg::Int(q_dim.div_ceil(TILE)),
                    Arg::Int(layer.window),
                    Arg::Float(softcap),
                ],
            );
        }
        self.gpu.launch(
            self.kernels.attention,
            [batch * self.n_heads, 1],
            &[
                Arg::Ptr(self.qkv),
                Arg::Int(stride),
                Arg::Ptr(layer.k_pool),
                Arg::Ptr(layer.v_pool),
                Arg::Ptr(lists),
                Arg::Ptr(step.row_offset),
                Arg::Ptr(step.row_len),
                Arg::Ptr(self.scratch.data),
                Arg::Ptr(step.scratch_offset),
                Arg::Ptr(self.attn_out),
                Arg::Int(self.n_heads),
                Arg::Int(self.n_kv_heads),
                Arg::Int(head_dim),
                Arg::Float(scale),
                Arg::Int(layer.window),
                Arg::Float(softcap),
            ],
        )
    }

    /// Queue `x += W input`, normalizing the addition first if the model does that.
    #[allow(clippy::too_many_arguments)]
    fn add_block_output(
        &self,
        w: &Matrix,
        input: Pointer,
        batch: usize,
        norm: Option<Pointer>,
        lora: Option<(usize, usize)>,
        step: &StepPointers,
    ) -> Result<()> {
        let out = match norm {
            Some(_) => self.added,
            None => self.x,
        };
        self.matmul(w, input, batch, out, norm.is_none())?;
        if let Some((layer, target)) = lora {
            self.lora(layer, target, input, batch, out, step)?;
        }
        let Some(norm) = norm else {
            return Ok(());
        };
        self.gpu.launch(
            self.kernels.rmsnorm_add,
            [batch, 1],
            &[
                Arg::Ptr(self.added),
                Arg::Ptr(self.x),
                Arg::Ptr(norm),
                Arg::Int(self.dim),
                Arg::Float(self.eps),
            ],
        )
    }

    /// Queue the LoRA corrections to stacked matrix `target` (one of `LORA_TARGETS`) of `layer`:
    /// `out += B (A input)` for each row using an adapter.
    fn lora(
        &self,
        layer: usize,
        target: usize,
        input: Pointer,
        batch: usize,
        out: Pointer,
        step: &StepPointers,
    ) -> Result<()> {
        let Some(t) = self.lora.get(layer).map(|targets| targets[target]) else {
            return Ok(());
        };
        if t.rank == 0 || !step.any_adapter {
            return Ok(());
        }
        // Rows per block in the LoRA kernels (`LORA_ROWS` in `kernels.cu`).
        const ROWS: usize = 4;
        self.gpu.launch_blocks(
            self.kernels.lora_shrink,
            [batch.div_ceil(ROWS), t.rank.div_ceil(8)],
            256,
            &[
                Arg::Ptr(input),
                Arg::Int((self.precision == Precision::Bf16) as usize),
                Arg::Int(t.in_dim),
                Arg::Ptr(t.a),
                Arg::Int(t.rank),
                Arg::Ptr(step.adapters),
                Arg::Int(batch),
                Arg::Ptr(self.lora_hidden),
            ],
        )?;
        self.gpu.launch_blocks(
            self.kernels.lora_expand,
            [batch.div_ceil(ROWS), t.out_dim.div_ceil(256)],
            256,
            &[
                Arg::Ptr(self.lora_hidden),
                Arg::Int(t.rank),
                Arg::Ptr(t.b),
                Arg::Int(t.out_dim),
                Arg::Ptr(step.adapters),
                Arg::Int(batch),
                Arg::Ptr(out),
            ],
        )
    }

    /// `(in, out)` for a module.
    fn module_shape(&self, module: LoraModule) -> (usize, usize) {
        let (q_dim, kv_dim) = (
            self.n_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        );
        match module {
            LoraModule::Q => (self.dim, q_dim),
            LoraModule::K | LoraModule::V => (self.dim, kv_dim),
            LoraModule::O => (q_dim, self.dim),
            LoraModule::Gate | LoraModule::Up => (self.dim, self.hidden_dim),
            LoraModule::Down => (self.hidden_dim, self.dim),
        }
    }

    /// Put every adapter's corrections on the GPU, replacing what was there.
    fn upload_adapters(&mut self) -> Result<()> {
        for targets in self.lora.drain(..) {
            for target in targets.iter().filter(|t| t.rank > 0) {
                self.gpu.free(target.a)?;
                self.gpu.free(target.b)?;
            }
        }
        if self.lora_hidden != Pointer::NULL {
            self.gpu.free(self.lora_hidden)?;
        }
        let residual = self.architecture.residual_scale;
        let mut largest = 1;
        for l in 0..self.layers.len() {
            let mut targets = [LoraTarget::default(); 4];
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
                                LoraModule::O | LoraModule::Down => residual,
                                _ => 1.0,
                            };
                            let at = (i * rank + first_rank) * in_dim;
                            a[at..at + pair.rank * in_dim].copy_from_slice(&pair.a);
                            // B is stored transposed: `[adapters, rank, out]`.
                            for o in 0..out {
                                for j in 0..pair.rank {
                                    b[(i * rank + first_rank + j) * out_dim + first_out + o] =
                                        pair.b[o * pair.rank + j] * scale;
                                }
                            }
                            first_rank += pair.rank;
                        }
                        first_out += out;
                    }
                }
                let a_device = self.gpu.alloc(a.len())?;
                self.gpu.write_now(a_device, &a)?;
                let b_device = self.gpu.alloc(b.len())?;
                self.gpu.write_now(b_device, &b)?;
                targets[t] = LoraTarget {
                    a: a_device,
                    b: b_device,
                    rank,
                    in_dim,
                    out_dim,
                };
                largest = largest.max(rank);
            }
            self.lora.push(targets);
        }
        self.lora_hidden = self.gpu.alloc((self.max_batch + PADDING_ROWS) * largest)?;
        // Recorded steps don't know about the new adapter.
        for (_, graph) in self.graphs.drain() {
            self.gpu.destroy_graph(graph)?;
        }
        Ok(())
    }

    /// Queue a mixture of experts on the normalized input in `xb`, adding its result to `x`.
    fn experts(&self, experts: &Experts, batch: usize) -> Result<()> {
        let moe = self
            .moe
            .as_ref()
            .expect("allocated for a mixture of experts");
        let spec = self
            .architecture
            .experts
            .expect("a sparse layer has experts");
        let (count, top_k, hidden, dim) = (spec.count, spec.top_k, spec.hidden, self.dim);
        self.matmul(&experts.router, self.xb, batch, moe.scores, false)?;
        self.gpu.launch(
            self.kernels.moe_route,
            [batch, 1],
            &[
                Arg::Ptr(moe.scores),
                Arg::Int(count),
                Arg::Int(top_k),
                Arg::Int(spec.renormalize as usize),
                Arg::Ptr(moe.topk_ids),
                Arg::Ptr(moe.topk_weights),
            ],
        )?;
        self.gpu.launch_blocks(
            self.kernels.moe_sort,
            [1, 1],
            256,
            &[
                Arg::Ptr(moe.topk_ids),
                Arg::Int(batch),
                Arg::Int(top_k),
                Arg::Int(count),
                Arg::Ptr(moe.sorted_rows),
                Arg::Ptr(moe.slot_of),
                Arg::Ptr(moe.group_expert),
                Arg::Ptr(moe.num_groups),
            ],
        )?;
        // Launched for as many slots as there could be; the kernels stop at the real number.
        let slots = moe_slots(batch, &spec);
        self.gpu.launch(
            self.kernels.moe_gather,
            [slots, 1],
            &[
                Arg::Ptr(self.xb),
                Arg::Int(dim / TILE),
                Arg::Ptr(moe.sorted_rows),
                Arg::Ptr(moe.num_groups),
                Arg::Ptr(moe.x),
            ],
        )?;
        self.moe_matmul(experts.gate_up, 2 * hidden, dim, slots, moe.x, moe.gate_up)?;
        let gated = match self.architecture.activation {
            Activation::Silu => self.kernels.moe_swiglu,
            Activation::Gelu => self.kernels.moe_geglu,
        };
        self.gpu.launch(
            gated,
            [slots, hidden.div_ceil(SWIGLU_COLUMNS)],
            &[
                Arg::Ptr(moe.gate_up),
                Arg::Ptr(moe.gated),
                Arg::Int(hidden),
                Arg::Int(hidden / TILE),
                Arg::Ptr(moe.num_groups),
            ],
        )?;
        self.moe_matmul(experts.down, dim, hidden, slots, moe.gated, moe.down)?;
        self.gpu.launch(
            self.kernels.moe_combine,
            [batch, 1],
            &[
                Arg::Ptr(moe.down),
                Arg::Int(dim),
                Arg::Ptr(moe.slot_of),
                Arg::Ptr(moe.topk_weights),
                Arg::Int(top_k),
                Arg::Ptr(self.x),
            ],
        )
    }

    /// Queue `out = W_e x` for each 16-row group of `x` and its expert `e`, with each expert's
    /// `[m, cols]` weights one after another from `w`.
    fn moe_matmul(
        &self,
        w: Pointer,
        m: usize,
        cols: usize,
        slots: usize,
        x: Pointer,
        out: Pointer,
    ) -> Result<()> {
        let moe = self
            .moe
            .as_ref()
            .expect("allocated for a mixture of experts");
        let groups = slots / TILE;
        let (kernel, tx) = match groups {
            1 => (self.kernels.moe_matmul_1, 1),
            2 => (self.kernels.moe_matmul_2, 2),
            _ => (self.kernels.moe_matmul_4, 4),
        };
        self.gpu.launch_blocks(
            kernel,
            [groups.div_ceil(tx), m.div_ceil(TILE * WARPS_ACROSS)],
            32 * WARPS_ACROSS * tx,
            &[
                Arg::Ptr(w),
                Arg::Int(m * cols),
                Arg::Ptr(moe.group_expert),
                Arg::Ptr(moe.num_groups),
                Arg::Int(m),
                Arg::Int(cols / TILE),
                Arg::Ptr(x),
                Arg::Ptr(out),
            ],
        )
    }

    /// Queue the normalization of a whole token's queries, and keys, as OLMoE does: the same
    /// kernel as for heads, treating each as one big head.
    fn full_qk_norms(&self, q_weight: Pointer, k_weight: Pointer, batch: usize) -> Result<()> {
        let (q_dim, kv_dim) = (
            self.n_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        );
        let stride = q_dim + 2 * kv_dim;
        for (offset, n_q, n_k, len) in [(0, 1, 0, q_dim), (q_dim, 0, 1, kv_dim)] {
            self.gpu.launch(
                self.kernels.head_norm,
                [batch, 1],
                &[
                    Arg::Ptr(self.qkv.offset(offset)),
                    Arg::Int(stride),
                    Arg::Int(n_q),
                    Arg::Int(n_k),
                    Arg::Int(len),
                    Arg::Ptr(q_weight),
                    Arg::Ptr(k_weight),
                    Arg::Float(self.eps),
                ],
            )?;
        }
        Ok(())
    }

    /// Queue the normalization of each head of the queries and keys.
    fn head_norms(&self, q_weight: Pointer, k_weight: Pointer, batch: usize) -> Result<()> {
        let head_dim = self.head_dim;
        self.gpu.launch(
            self.kernels.head_norm,
            [batch * (self.n_heads + self.n_kv_heads), 1],
            &[
                Arg::Ptr(self.qkv),
                Arg::Int(self.n_heads * head_dim + 2 * self.n_kv_heads * head_dim),
                Arg::Int(self.n_heads),
                Arg::Int(self.n_kv_heads),
                Arg::Int(head_dim),
                Arg::Ptr(q_weight),
                Arg::Ptr(k_weight),
                Arg::Float(self.eps),
            ],
        )
    }

    fn rmsnorm(
        &self,
        x: Pointer,
        out: Pointer,
        weight: Pointer,
        rows: usize,
        select: Pointer,
    ) -> Result<()> {
        let args = [
            Arg::Ptr(x),
            Arg::Ptr(out),
            Arg::Ptr(weight),
            Arg::Int(self.dim),
            Arg::Float(self.eps),
            Arg::Ptr(select),
            Arg::Int(self.dim.div_ceil(TILE)),
        ];
        match self.precision {
            Precision::Bf16 => self.gpu.launch(self.kernels.rmsnorm_bf16, [rows, 1], &args),
            Precision::F32 => self.gpu.launch(self.kernels.rmsnorm, [rows, 1], &args[..6]),
        }
    }

    /// Work out and queue the uploads that describe this batch.
    /// Returns pointers to the arrays inside the step buffer.
    fn upload_step(&mut self, rows: &[Row]) -> Result<StepPointers> {
        let batch = rows.len();
        let half = self.head_dim / 2;
        let step = &mut self.step;
        step.tokens.clear();
        step.write_slots.clear();
        step.row_offset.clear();
        step.row_len.clear();
        step.adapters.clear();
        step.team_size.clear();
        step.scratch_offset.clear();
        step.logit_rows.clear();
        step.slot_lists.clear();
        step.gaps.clear();
        step.limits.clear();
        step.temperatures.clear();
        step.keys.clear();
        step.hidden.clear();
        let tables = self.architecture.rope_tables.len();
        step.cos.resize(tables * batch * half, 0.0);
        step.sin.resize(tables * batch * half, 0.0);

        let mut scratch_total = 0usize;
        let mut r = 0;
        while r < batch {
            // Rows from one sequence are next to each other and share a list of slots: the
            // longest one, belonging to its last row in this batch.
            let start = rows[r].slots.as_ptr();
            let group = rows[r..]
                .iter()
                .take_while(|row| row.slots.as_ptr() == start)
                .count();
            let offset = step.slot_lists.len() as u32;
            let visible = rows[r + group - 1].slots;
            match self.precision {
                Precision::F32 => step.slot_lists.extend_from_slice(visible),
                // In pages, slot `p % 16` of a page holds token `p`, so a list of pages will do.
                Precision::Bf16 => step
                    .slot_lists
                    .extend(visible.iter().step_by(TILE).map(|&slot| slot / TILE as u32)),
            }
            let team = TILE / (self.n_heads / self.n_kv_heads);
            for (i, row) in rows[r..r + group].iter().enumerate() {
                let index = r + i;
                step.team_size.push(match i % team {
                    0 => team.min(group - i) as u32,
                    _ => 0,
                });
                let token = row.token as usize;
                assert!(
                    token < self.vocab_size,
                    "token id {token} is outside the vocabulary"
                );
                step.tokens.push(row.token);
                step.write_slots
                    .push(*row.slots.last().expect("a row needs a KV slot"));
                step.row_offset.push(offset);
                step.row_len.push(row.slots.len() as u32);
                assert!(
                    (row.adapter as usize) <= self.adapters.len(),
                    "there's no adapter {}",
                    row.adapter
                );
                step.adapters.push(row.adapter as u32);
                step.scratch_offset.push(scratch_total as u32);
                scratch_total += row.slots.len() * self.n_heads;
                if row.want_logits {
                    step.logit_rows.push(index as u32);
                    let (gap, limit, temperature, key) = match row.sampling {
                        LogitUse::Candidates { gap, limit } => (gap, limit as u32, -1.0, 0),
                        LogitUse::Draw { temperature, key } => (-1.0, 0, temperature, key),
                        LogitUse::All | LogitUse::Hidden => (-1.0, 0, -1.0, 0),
                    };
                    step.hidden.push(row.sampling == LogitUse::Hidden);
                    step.gaps.push(gap);
                    step.limits.push(limit);
                    step.temperatures.push(temperature);
                    step.keys.push(key as u32);
                    step.keys.push((key >> 32) as u32);
                }
                let position = (row.slots.len() - 1) as f32;
                for (table, frequencies) in self.architecture.rope_tables.iter().enumerate() {
                    let scale = self.architecture.rope_scales[table];
                    let start = (table * batch + index) * half;
                    for (j, &freq) in frequencies.iter().enumerate() {
                        let angle = position * freq;
                        step.cos[start + j] = angle.cos() * scale;
                        step.sin[start + j] = angle.sin() * scale;
                    }
                }
            }
            r += group;
        }

        // Everything's waited for at the end of each step, so buffers can be replaced here.
        let mut moved = self
            .slot_list_buffer
            .reserve(&self.gpu, step.slot_lists.len())?;
        if self.precision == Precision::F32 {
            moved |= self.scratch.reserve(&self.gpu, scratch_total)?;
        }
        if moved {
            // Recorded kernels hold on to the old addresses.
            for (_, graph) in self.graphs.drain() {
                self.gpu.destroy_graph(graph)?;
            }
        }

        // Put the arrays side by side and send them in one go.
        let bits =
            |values: &[f32], into: &mut Vec<u32>| into.extend(values.iter().map(|v| v.to_bits()));
        let combined = &mut step.combined;
        combined.clear();
        let mark = |combined: &Vec<u32>| self.step_buffer.offset(combined.len());
        let tokens = mark(combined);
        combined.extend_from_slice(&step.tokens);
        let write_slots = mark(combined);
        combined.extend_from_slice(&step.write_slots);
        let row_offset = mark(combined);
        combined.extend_from_slice(&step.row_offset);
        let row_len = mark(combined);
        combined.extend_from_slice(&step.row_len);
        let adapters = mark(combined);
        combined.extend_from_slice(&step.adapters);
        let team_size = mark(combined);
        combined.extend_from_slice(&step.team_size);
        let scratch_offset = mark(combined);
        combined.extend_from_slice(&step.scratch_offset);
        let logit_rows = mark(combined);
        combined.extend_from_slice(&step.logit_rows);
        // Always the same room for this list, so that for a given batch size every array is at
        // the same address from one step to the next and recorded kernels stay valid.
        combined.resize(combined.len() + batch - step.logit_rows.len(), 0);
        let cos = mark(combined);
        bits(&step.cos, combined);
        let sin = mark(combined);
        bits(&step.sin, combined);
        // These only depend on which rows want logits, and aren't used by recorded kernels.
        let gaps = mark(combined);
        bits(&step.gaps, combined);
        let limits = mark(combined);
        combined.extend_from_slice(&step.limits);
        let temperatures = mark(combined);
        bits(&step.temperatures, combined);
        let keys = mark(combined);
        combined.extend_from_slice(&step.keys);

        self.gpu.write(self.step_buffer, combined)?;
        self.gpu
            .write(self.slot_list_buffer.data, &step.slot_lists)?;
        Ok(StepPointers {
            tokens,
            write_slots,
            row_offset,
            row_len,
            adapters,
            any_adapter: step.adapters.iter().any(|&a| a != 0),
            team_size,
            scratch_offset,
            logit_rows,
            cos,
            sin,
            gaps,
            limits,
            temperatures,
            keys,
            n_logits: step.logit_rows.len(),
        })
    }

    /// Queue everything up to the last layer's output.
    fn launch_layers(&self, batch: usize, step: &StepPointers) -> Result<()> {
        let dim = self.dim;
        let embed_args = [
            Arg::Ptr(self.embeddings),
            Arg::Ptr(step.tokens),
            Arg::Ptr(self.x),
            Arg::Int(dim),
            Arg::Int(dim.div_ceil(TILE)),
            Arg::Float(self.embedding_scale),
        ];
        match self.precision {
            Precision::Bf16 => self
                .gpu
                .launch(self.kernels.embed_bf16, [batch, 1], &embed_args)?,
            Precision::F32 => self.gpu.launch(
                self.kernels.embed,
                [batch, 1],
                &[
                    Arg::Ptr(self.embeddings),
                    Arg::Ptr(step.tokens),
                    Arg::Ptr(self.x),
                    Arg::Int(dim),
                    Arg::Float(self.embedding_scale),
                ],
            )?,
        }
        let (gated_bf16, gated) = match self.architecture.activation {
            Activation::Silu => (self.kernels.swiglu_bf16, self.kernels.swiglu),
            Activation::Gelu => (self.kernels.geglu_bf16, self.kernels.geglu),
        };

        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block: mix in information from earlier tokens.
            self.rmsnorm(self.x, self.xb, layer.attn_norm, batch, Pointer::NULL)?;
            self.matmul(&layer.qkv, self.xb, batch, self.qkv, false)?;
            self.lora(l, 0, self.xb, batch, self.qkv, step)?;
            if let Some((q_weight, k_weight)) = layer.head_norms {
                match self.architecture.full_qk_norms {
                    true => self.full_qk_norms(q_weight, k_weight, batch)?,
                    false => self.head_norms(q_weight, k_weight, batch)?,
                }
            }
            self.rope_and_store(layer, batch, step)?;
            self.attention(layer, batch, step)?;
            self.add_block_output(
                &layer.o_proj,
                self.attn_out,
                batch,
                layer.norm_after_attn,
                Some((l, 1)),
                step,
            )?;

            // MLP block: process each token on its own.
            self.rmsnorm(self.x, self.xb, layer.mlp_norm, batch, Pointer::NULL)?;
            let (gate_up, down) = match &layer.feed_forward {
                FeedForward::Dense { gate_up, down } => (gate_up, down),
                FeedForward::Sparse(experts) => {
                    self.experts(experts, batch)?;
                    continue;
                }
            };
            self.matmul(gate_up, self.xb, batch, self.gate_up, false)?;
            self.lora(l, 2, self.xb, batch, self.gate_up, step)?;
            let swiglu_args = [
                Arg::Ptr(self.gate_up),
                Arg::Ptr(self.gated),
                Arg::Int(self.hidden_dim),
                Arg::Int(self.hidden_dim.div_ceil(TILE)),
            ];
            match self.precision {
                Precision::Bf16 => self.gpu.launch(
                    gated_bf16,
                    [batch, self.hidden_dim.div_ceil(SWIGLU_COLUMNS)],
                    &swiglu_args,
                )?,
                Precision::F32 => self.gpu.launch(gated, [batch, 1], &swiglu_args[..3])?,
            }
            self.add_block_output(
                down,
                self.gated,
                batch,
                layer.norm_after_mlp,
                Some((l, 3)),
                step,
            )?;
        }
        Ok(())
    }

    /// Record the kernels of one forward pass for this batch size.
    fn record_graph(&self, batch: usize, step: &StepPointers) -> Result<Graph> {
        self.gpu.begin_recording()?;
        let launched = self.launch_layers(batch, step);
        let graph = self.gpu.end_recording();
        launched?;
        graph
    }

    fn try_forward(&mut self, rows: &[Row]) -> Result<()> {
        let batch = rows.len();
        assert!(batch <= self.max_batch, "batch larger than the buffers");
        let step = self.upload_step(rows)?;
        if batch <= MAX_RECORDED_BATCH {
            let key = (batch, step.any_adapter);
            if !self.graphs.contains_key(&key) {
                let graph = self.record_graph(batch, &step)?;
                self.graphs.insert(key, graph);
            }
            self.gpu.launch_graph(&self.graphs[&key])?;
        } else {
            self.launch_layers(batch, &step)?;
        }

        let n_logits = step.n_logits;
        let any_candidates = self.step.gaps.iter().any(|&gap| gap >= 0.0);
        let any_draws = self.step.temperatures.iter().any(|&t| t >= 0.0);
        let any_hidden = self.step.hidden.iter().any(|&hidden| hidden);
        if any_hidden {
            // In 32 bits whatever the precision, for embeddings.
            self.gpu.launch(
                self.kernels.rmsnorm,
                [n_logits, 1],
                &[
                    Arg::Ptr(self.x),
                    Arg::Ptr(self.added),
                    Arg::Ptr(self.norm),
                    Arg::Int(self.dim),
                    Arg::Float(self.eps),
                    Arg::Ptr(step.logit_rows),
                ],
            )?;
            self.hidden.resize(n_logits * self.dim, 0.0);
            self.gpu.read(self.added, &mut self.hidden)?;
        }
        if n_logits > 0 {
            self.rmsnorm(self.x, self.xb, self.norm, n_logits, step.logit_rows)?;
            self.matmul(&self.lm_head, self.xb, n_logits, self.logits_device, false)?;
            let (divisor, softcap) = (
                self.architecture.logit_divisor,
                self.architecture.logit_softcap,
            );
            if divisor != 1.0 || softcap != 0.0 {
                self.gpu.launch_blocks(
                    self.kernels.finish_logits,
                    [n_logits, 1],
                    SCAN_THREADS,
                    &[
                        Arg::Ptr(self.logits_device),
                        Arg::Int(self.vocab_size),
                        Arg::Float(divisor),
                        Arg::Float(softcap),
                    ],
                )?;
            }
            if any_candidates {
                self.gpu.launch_blocks(
                    self.kernels.top_candidates,
                    [n_logits, 1],
                    SCAN_THREADS,
                    &[
                        Arg::Ptr(self.logits_device),
                        Arg::Int(self.vocab_size),
                        Arg::Ptr(step.gaps),
                        Arg::Ptr(step.limits),
                        Arg::Ptr(self.results_device),
                        Arg::Ptr(self.candidate_values_device),
                        Arg::Ptr(self.candidate_ids_device),
                        Arg::Int(MAX_CANDIDATES),
                    ],
                )?;
            }
            if any_draws {
                self.gpu.launch_blocks(
                    self.kernels.draw_tokens,
                    [n_logits, 1],
                    SCAN_THREADS,
                    &[
                        Arg::Ptr(self.logits_device),
                        Arg::Int(self.vocab_size),
                        Arg::Ptr(step.temperatures),
                        Arg::Ptr(step.keys),
                        Arg::Ptr(self.results_device),
                    ],
                )?;
            }
            self.results.resize(n_logits * 2, 0);
            self.gpu.read(self.results_device, &mut self.results)?;
            if any_candidates {
                self.candidate_values.resize(n_logits * MAX_CANDIDATES, 0.0);
                self.candidate_ids.resize(n_logits * MAX_CANDIDATES, 0);
                self.gpu
                    .read(self.candidate_values_device, &mut self.candidate_values)?;
                self.gpu
                    .read(self.candidate_ids_device, &mut self.candidate_ids)?;
            }
        }
        self.gpu.wait()?;
        self.collect_candidates(n_logits)
    }

    /// Turn what came back into one list of `(score, token)`, and fetch all the logits for any
    /// row that wants them or that had too many candidates.
    fn collect_candidates(&mut self, n_logits: usize) -> Result<()> {
        self.candidates.clear();
        self.candidate_start.clear();
        self.candidate_end.clear();
        self.full_rows.clear();
        for row in 0..n_logits {
            if self.step.hidden[row] {
                // No scores wanted.
                self.candidate_start.push(0);
                self.candidate_end.push(0);
                continue;
            }
            if self.step.temperatures[row] >= 0.0 {
                // Already drawn, which leaves just the one candidate.
                self.candidate_start.push(self.candidates.len() as u32);
                self.candidates.push((0.0, self.results[row * 2 + 1]));
                self.candidate_end.push(self.candidates.len() as u32);
                continue;
            }
            let count = self.results[row * 2] as usize;
            if self.step.gaps[row] < 0.0 || count > MAX_CANDIDATES {
                self.candidate_start.push(u32::MAX);
                self.candidate_end.push(self.full_rows.len() as u32);
                self.full_rows.push(row);
                continue;
            }
            self.candidate_start.push(self.candidates.len() as u32);
            let values = &self.candidate_values[row * MAX_CANDIDATES..][..count];
            let ids = &self.candidate_ids[row * MAX_CANDIDATES..][..count];
            self.candidates
                .extend(values.iter().copied().zip(ids.iter().copied()));
            self.candidate_end.push(self.candidates.len() as u32);
        }
        if !self.full_rows.is_empty() {
            let vocab = self.vocab_size;
            self.logits.resize(self.full_rows.len() * vocab, 0.0);
            for (index, &row) in self.full_rows.iter().enumerate() {
                self.gpu.read(
                    self.logits_device.offset(row * vocab),
                    &mut self.logits[index * vocab..(index + 1) * vocab],
                )?;
            }
            self.gpu.wait()?;
        }
        Ok(())
    }
}

struct StepPointers {
    tokens: Pointer,
    write_slots: Pointer,
    row_offset: Pointer,
    row_len: Pointer,
    adapters: Pointer,
    /// Some row uses a LoRA adapter.
    any_adapter: bool,
    team_size: Pointer,
    scratch_offset: Pointer,
    logit_rows: Pointer,
    cos: Pointer,
    sin: Pointer,
    gaps: Pointer,
    limits: Pointer,
    temperatures: Pointer,
    keys: Pointer,
    n_logits: usize,
}

/// Queue `out = W x`, or `out += W x` if `accumulate`, choosing the kernel for this batch size.
#[allow(clippy::too_many_arguments)]
fn launch_matmul(
    gpu: &Gpu,
    kernels: &Kernels,
    precision: Precision,
    w: &Matrix,
    x: Pointer,
    batch: usize,
    out: Pointer,
    accumulate: bool,
) -> Result<()> {
    if w.zeros != Pointer::NULL {
        let (kernel, groups) = match batch {
            1..=16 => (kernels.matmul_q4_1, 1),
            17..=32 => (kernels.matmul_q4_2, 2),
            _ => (kernels.matmul_q4_4, 4),
        };
        return gpu.launch_blocks(
            kernel,
            [
                batch.div_ceil(TILE * groups),
                w.rows.div_ceil(TILE * WARPS_ACROSS),
            ],
            32 * WARPS_ACROSS * groups,
            &[
                Arg::Ptr(w.data),
                Arg::Ptr(w.scales),
                Arg::Ptr(w.zeros),
                Arg::Int(w.rows),
                Arg::Int(w.cols.div_ceil(TILE)),
                Arg::Ptr(x),
                Arg::Int(batch),
                Arg::Ptr(out),
                Arg::Ptr(w.bias),
                Arg::Int(accumulate as usize),
            ],
        );
    }
    if w.scales != Pointer::NULL {
        let (kernel, groups) = match (batch, w.fp8) {
            (1..=16, false) => (kernels.matmul_i8_1, 1),
            (17..=32, false) => (kernels.matmul_i8_2, 2),
            (_, false) => (kernels.matmul_i8_4, 4),
            (1..=16, true) => (kernels.matmul_f8_1, 1),
            (17..=32, true) => (kernels.matmul_f8_2, 2),
            (_, true) => (kernels.matmul_f8_4, 4),
        };
        return gpu.launch_blocks(
            kernel,
            [
                batch.div_ceil(TILE * groups),
                w.rows.div_ceil(TILE * WARPS_ACROSS),
            ],
            32 * WARPS_ACROSS * groups,
            &[
                Arg::Ptr(w.data),
                Arg::Ptr(w.scales),
                Arg::Int(w.rows),
                Arg::Int(w.cols.div_ceil(TILE)),
                Arg::Ptr(x),
                Arg::Int(batch),
                Arg::Ptr(out),
                Arg::Ptr(w.bias),
                Arg::Int(accumulate as usize),
            ],
        );
    }
    if precision == Precision::Bf16 {
        let (kernel, groups) = match batch {
            1..=16 => (kernels.matmul_bf16_1, 1),
            17..=32 => (kernels.matmul_bf16_2, 2),
            _ => (kernels.matmul_bf16_4, 4),
        };
        return gpu.launch_blocks(
            kernel,
            [
                batch.div_ceil(TILE * groups),
                w.rows.div_ceil(TILE * WARPS_ACROSS),
            ],
            32 * WARPS_ACROSS * groups,
            &[
                Arg::Ptr(w.data),
                Arg::Int(w.rows),
                Arg::Int(w.cols.div_ceil(TILE)),
                Arg::Ptr(x),
                Arg::Int(batch),
                Arg::Ptr(out),
                Arg::Ptr(w.bias),
                Arg::Int(accumulate as usize),
                Arg::Ptr(w.scratch),
            ],
        );
    }
    let (kernel, group) = match batch {
        1 => (kernels.matmul_1, 1),
        2 | 3 => (kernels.matmul_2, 2),
        4..=7 => (kernels.matmul_4, 4),
        _ => (kernels.matmul_8, 8),
    };
    gpu.launch(
        kernel,
        [batch.div_ceil(group), w.rows.div_ceil(BLOCK_ROWS)],
        &[
            Arg::Ptr(w.data),
            Arg::Int(w.rows),
            Arg::Int(w.cols),
            Arg::Ptr(x),
            Arg::Int(batch),
            Arg::Ptr(out),
            Arg::Ptr(w.bias),
            Arg::Int(accumulate as usize),
        ],
    )
}

/// A zeroed buffer for matrix inputs: `rows` by `cols`, in whatever layout the precision uses.
fn input_buffer(gpu: &Gpu, precision: Precision, rows: usize, cols: usize) -> Result<Pointer> {
    let elements = match precision {
        Precision::Bf16 => (rows.div_ceil(TILE) * cols.div_ceil(TILE) * TILE * TILE).div_ceil(2),
        Precision::F32 => rows * cols,
    };
    let buffer = gpu.alloc(elements)?;
    gpu.zero(buffer, elements)?;
    Ok(buffer)
}

/// Rearrange row-major bfloat16 values into the tile layout (see `kernels.cu`), padded with zeros
/// to a whole number of 16-row groups and 16-column chunks.
fn to_tiles<T: Copy + Default>(values: &[T], rows: usize, cols: usize) -> Vec<T> {
    let chunks = cols.div_ceil(TILE);
    let groups = rows.div_ceil(TILE);
    let mut tiles = vec![T::default(); groups * chunks * TILE * TILE];
    for (row, row_values) in values.chunks_exact(cols).enumerate() {
        for (column, &value) in row_values.iter().enumerate() {
            let tile = (row / TILE) * chunks + column / TILE;
            tiles[tile * TILE * TILE + (row % TILE) * TILE + column % TILE] = value;
        }
    }
    tiles
}

/// Upload some matrices with the same number of columns as one, one on top of the other. Padded
/// with zero rows to a whole number of blocks. `layers` lists each one's name and row count.
#[allow(clippy::too_many_arguments)]
fn upload_stacked(
    gpu: &Gpu,
    st: &SafeTensors,
    precision: Precision,
    num_params: &mut usize,
    layers: &[(String, usize)],
    cols: usize,
    batch_rows: usize,
    scale: f32,
    quantization: Option<Quantization>,
) -> Result<Matrix> {
    let rows: usize = layers.iter().map(|(_, rows)| rows).sum();
    let padded = rows.div_ceil(BLOCK_ROWS) * BLOCK_ROWS;
    // 4-bit checkpoints keep their 4-bit values, when the kernel can take their groups.
    let all_int4 = layers
        .iter()
        .all(|(name, _)| st.dtype(&format!("{name}.weight")) == Some("INT4"));
    if all_int4 && precision == Precision::Bf16 {
        match upload_int4(gpu, st, num_params, layers, cols, scale)? {
            Some(matrix) => return Ok(matrix),
            None => eprintln!(
                "{}: its 4-bit groups don't fit the GPU kernel, so using 16-bit weights",
                layers[0].0
            ),
        }
    }
    // FP8 checkpoints keep their 8-bit floats, with each block scale repeated down its rows.
    let all_fp8 = layers
        .iter()
        .all(|(name, _)| st.dtype(&format!("{name}.weight")) == Some("F8_E4M3"));
    if all_fp8 && precision == Precision::Bf16 {
        let groups = cols.div_ceil(QUANT_GROUP);
        let (mut bytes, mut scales) = (Vec::with_capacity(rows * cols), Vec::new());
        let mut bias: Option<Vec<f32>> = None;
        let mut row = 0;
        for (name, count) in layers {
            let (values, blocks) = st
                .load_fp8(&format!("{name}.weight"), &[*count, cols])?
                .expect("checked above");
            bytes.extend(values);
            for r in 0..*count {
                let blocks = &blocks[(r / FP8_BLOCK) * groups..][..groups];
                scales.extend(blocks.iter().map(|s| s * scale));
            }
            *num_params += count * cols;
            let bias_name = format!("{name}.bias");
            if st.contains(&bias_name) {
                let values = st.load(&bias_name, &[*count])?;
                bias.get_or_insert_with(|| vec![0.0; rows])[row..row + count]
                    .copy_from_slice(&values);
            }
            row += count;
        }
        bias.iter_mut().flatten().for_each(|v| *v *= scale);
        let tiles = to_lane_tiles(&bytes, rows, cols);
        let data = gpu.alloc(tiles.len().div_ceil(4))?;
        gpu.write_now(data, &tiles)?;
        let scales_device = gpu.alloc(scales.len())?;
        gpu.write_now(scales_device, &scales)?;
        let bias = match bias {
            Some(values) => {
                let device = gpu.alloc(rows)?;
                gpu.write_now(device, &values)?;
                device
            }
            None => Pointer::NULL,
        };
        let _ = batch_rows;
        return Ok(Matrix {
            data,
            rows,
            cols,
            bias,
            scratch: Pointer::NULL,
            scales: scales_device,
            fp8: true,
            zeros: Pointer::NULL,
        });
    }
    let mut bias: Option<Vec<f32>> = None;
    let mut f32_values: Vec<f32> = Vec::new();
    let mut bf16_values: Vec<u16> = Vec::new();
    let mut row = 0;
    for (name, count) in layers {
        let weight = format!("{name}.weight");
        match precision {
            Precision::F32 => f32_values.extend(st.load(&weight, &[*count, cols])?),
            Precision::Bf16 => bf16_values.extend(st.load_bf16(&weight, &[*count, cols])?),
        }
        *num_params += count * cols;
        let bias_name = format!("{name}.bias");
        if st.contains(&bias_name) {
            let values = st.load(&bias_name, &[*count])?;
            *num_params += values.len();
            bias.get_or_insert_with(|| vec![0.0; rows])[row..row + count].copy_from_slice(&values);
        }
        row += count;
    }
    // A layer whose output is always scaled has its weights scaled instead.
    if scale != 1.0 {
        f32_values.iter_mut().for_each(|v| *v *= scale);
        for v in &mut bf16_values {
            *v = f32_to_bf16(bf16_to_f32(*v) * scale);
        }
        bias.iter_mut().flatten().for_each(|v| *v *= scale);
    }

    if quantization == Some(Quantization::Int8) {
        let values: Vec<f32> = match precision {
            Precision::F32 => f32_values,
            Precision::Bf16 => bf16_values.iter().map(|&v| bf16_to_f32(v)).collect(),
        };
        let (bytes, scales) = quantize_groups(&values, rows, cols);
        let tiles = to_lane_tiles(&bytes, rows, cols);
        let data = gpu.alloc(tiles.len().div_ceil(4))?;
        gpu.write_now(data, &tiles)?;
        let scales_device = gpu.alloc(scales.len())?;
        gpu.write_now(scales_device, &scales)?;
        let bias = match bias {
            Some(values) => {
                let device = gpu.alloc(rows)?;
                gpu.write_now(device, &values)?;
                device
            }
            None => Pointer::NULL,
        };
        let _ = batch_rows;
        return Ok(Matrix {
            data,
            rows,
            cols,
            bias,
            scratch: Pointer::NULL,
            scales: scales_device,
            fp8: false,
            zeros: Pointer::NULL,
        });
    }

    let data = match precision {
        Precision::F32 => {
            let data = gpu.alloc(padded * cols)?;
            gpu.write_now(data, &f32_values)?;
            if padded > rows {
                gpu.zero(data.offset(rows * cols), (padded - rows) * cols)?;
            }
            data
        }
        Precision::Bf16 => {
            let tiles = to_tiles(&bf16_values, rows, cols);
            let data = gpu.alloc(tiles.len().div_ceil(2))?;
            gpu.write_now(data, &tiles)?;
            data
        }
    };
    let bias = match bias {
        Some(values) => {
            let device = gpu.alloc(rows)?;
            gpu.write_now(device, &values)?;
            device
        }
        None => Pointer::NULL,
    };
    let scratch = match precision == Precision::Bf16 && !rows.is_multiple_of(TILE) {
        true => gpu.alloc(tile_scratch_elements(batch_rows, rows))?,
        false => Pointer::NULL,
    };
    Ok(Matrix {
        data,
        rows,
        cols,
        bias,
        scratch,
        scales: Pointer::NULL,
        fp8: false,
        zeros: Pointer::NULL,
    })
}

/// Stack 4-bit weights for the GPU kernel: their values two to a byte in the lane order of the
/// 8-bit kernel, and a scale (times `scale`) and zero point for each row and group of 128
/// columns. `None` if the kernel can't take them, because their groups aren't a whole number of
/// 128-column groups or GPTQ reordered their columns.
fn upload_int4(
    gpu: &Gpu,
    st: &SafeTensors,
    num_params: &mut usize,
    layers: &[(String, usize)],
    cols: usize,
    scale: f32,
) -> Result<Option<Matrix>> {
    let rows: usize = layers.iter().map(|(_, rows)| rows).sum();
    let groups = cols.div_ceil(QUANT_GROUP);
    let (mut values, mut scales, mut zeros) = (
        Vec::with_capacity(rows * cols),
        Vec::with_capacity(rows * groups),
        Vec::with_capacity(rows * groups),
    );
    let mut bias: Option<Vec<f32>> = None;
    let mut row = 0;
    for (name, count) in layers {
        let Ok(Some(matrix)) = st.load_int4(&format!("{name}.weight"), &[*count, cols]) else {
            return Ok(None);
        };
        if !matrix.group.is_multiple_of(QUANT_GROUP) || !cols.is_multiple_of(QUANT_GROUP) {
            return Ok(None);
        }
        // Each of the checkpoint's groups covers this many of the kernel's.
        let (own_groups, repeat) = (cols.div_ceil(matrix.group), matrix.group / QUANT_GROUP);
        values.extend_from_slice(&matrix.values);
        for r in 0..*count {
            for g in 0..groups {
                let at = r * own_groups + g / repeat;
                scales.push(matrix.scales[at] * scale);
                zeros.push(matrix.zeros[at]);
            }
        }
        *num_params += count * cols;
        let bias_name = format!("{name}.bias");
        if st.contains(&bias_name) {
            let values = st.load(&bias_name, &[*count])?;
            *num_params += values.len();
            bias.get_or_insert_with(|| vec![0.0; rows])[row..row + count].copy_from_slice(&values);
        }
        row += count;
    }
    bias.iter_mut().flatten().for_each(|v| *v *= scale);
    let packed = pack_int4(&values, rows, cols);
    let data = gpu.alloc(packed.len().div_ceil(4))?;
    gpu.write_now(data, &packed)?;
    let scales_device = gpu.alloc(scales.len())?;
    gpu.write_now(scales_device, &scales)?;
    let zeros_device = gpu.alloc(zeros.len().div_ceil(4))?;
    gpu.write_now(zeros_device, &zeros)?;
    let bias = match bias {
        Some(values) => {
            let device = gpu.alloc(rows)?;
            gpu.write_now(device, &values)?;
            device
        }
        None => Pointer::NULL,
    };
    Ok(Some(Matrix {
        data,
        rows,
        cols,
        bias,
        scratch: Pointer::NULL,
        scales: scales_device,
        fp8: false,
        zeros: zeros_device,
    }))
}

/// 4-bit values `[rows, cols]` in the 8-bit kernel's lane order (see [`to_lane_tiles`]), two to a
/// byte: each lane's 8 values of a tile become one 32-bit word, the first in the lowest nibble.
/// Row groups are packed side by side on all the CPU's threads.
fn pack_int4(values: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let chunks = cols.div_ceil(TILE);
    let groups = rows.div_ceil(TILE);
    let group_bytes = chunks * TILE * TILE / 2;
    let mut packed = vec![0u8; groups * group_bytes];
    let at = |row: usize, column: usize| -> u32 {
        match row < rows && column < cols {
            true => values[row * cols + column] as u32,
            false => 0,
        }
    };
    let pack_group = |group: usize, out: &mut [u8]| {
        for chunk in 0..chunks {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let mut word = 0u32;
                for half in 0..2 {
                    let row = group * TILE + half * 8 + g;
                    let column = chunk * TILE + 2 * t;
                    for (j, k) in [0, 1, 8, 9].into_iter().enumerate() {
                        word |= at(row, column + k) << (4 * (half * 4 + j));
                    }
                }
                let index = (chunk * 32 + lane) * 4;
                out[index..index + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per_thread = groups.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (part, out) in packed.chunks_mut(per_thread * group_bytes).enumerate() {
            scope.spawn(move || {
                for (i, group_out) in out.chunks_mut(group_bytes).enumerate() {
                    pack_group(part * per_thread + i, group_out);
                }
            });
        }
    });
    packed
}

/// The most slots a mixture of experts can need for `rows` rows: every (row, expert) pair, and
/// each expert's run padded to a whole 16-row group.
fn moe_slots(rows: usize, spec: &ConfigExperts) -> usize {
    (rows * spec.top_k + spec.count * (TILE - 1)).next_multiple_of(TILE)
}

/// Upload a layer's mixture of experts: the router, and every expert's weights in the 16-bit tile
/// layout, one expert after another.
#[allow(clippy::too_many_arguments)]
fn upload_experts(
    gpu: &Gpu,
    st: &SafeTensors,
    num_params: &mut usize,
    p: &str,
    spec: &ConfigExperts,
    dim: usize,
    batch_rows: usize,
    residual: f32,
) -> Result<Experts> {
    let h = spec.hidden;
    ensure!(
        dim.is_multiple_of(TILE) && h.is_multiple_of(TILE),
        "experts' sizes have to be multiples of 16 on the GPU"
    );
    let router_name = match spec.naming {
        ExpertNaming::Qwen => format!("{p}.mlp.gate"),
        ExpertNaming::Mixtral => format!("{p}.block_sparse_moe.gate"),
        ExpertNaming::Granite => format!("{p}.block_sparse_moe.router.layer"),
    };
    let router = upload_stacked(
        gpu,
        st,
        Precision::Bf16,
        num_params,
        &[(router_name, spec.count)],
        dim,
        batch_rows,
        1.0,
        None,
    )?;
    let scaled = |values: Vec<u16>| -> Vec<u16> {
        match residual == 1.0 {
            true => values,
            false => values
                .into_iter()
                .map(|v| f32_to_bf16(bf16_to_f32(v) * residual))
                .collect(),
        }
    };
    let mut gate_up: Vec<u16> = Vec::with_capacity(spec.count * 2 * h * dim);
    let mut down: Vec<u16> = Vec::with_capacity(spec.count * dim * h);
    match spec.naming {
        ExpertNaming::Granite => {
            let input_name = format!("{p}.block_sparse_moe.input_linear.weight");
            let output_name = format!("{p}.block_sparse_moe.output_linear.weight");
            let input = st.load_bf16(&input_name, &[spec.count, 2 * h, dim])?;
            let output = st.load_bf16(&output_name, &[spec.count, dim, h])?;
            for e in 0..spec.count {
                gate_up.extend(to_tiles(
                    &input[e * 2 * h * dim..][..2 * h * dim],
                    2 * h,
                    dim,
                ));
                let expert_down = scaled(output[e * dim * h..][..dim * h].to_vec());
                down.extend(to_tiles(&expert_down, dim, h));
            }
        }
        ExpertNaming::Qwen | ExpertNaming::Mixtral => {
            let (prefix, [gate, up, down_name]) = match spec.naming {
                ExpertNaming::Qwen => (
                    format!("{p}.mlp.experts"),
                    ["gate_proj", "up_proj", "down_proj"],
                ),
                _ => (format!("{p}.block_sparse_moe.experts"), ["w1", "w3", "w2"]),
            };
            for e in 0..spec.count {
                let mut stacked =
                    st.load_bf16(&format!("{prefix}.{e}.{gate}.weight"), &[h, dim])?;
                stacked.extend(st.load_bf16(&format!("{prefix}.{e}.{up}.weight"), &[h, dim])?);
                gate_up.extend(to_tiles(&stacked, 2 * h, dim));
                let expert_down =
                    st.load_bf16(&format!("{prefix}.{e}.{down_name}.weight"), &[dim, h])?;
                down.extend(to_tiles(&scaled(expert_down), dim, h));
            }
        }
    }
    *num_params += spec.count * 3 * h * dim;
    let upload = |values: &[u16]| -> Result<Pointer> {
        let device = gpu.alloc(values.len().div_ceil(2))?;
        gpu.write_now(device, values)?;
        Ok(device)
    };
    Ok(Experts {
        router,
        gate_up: upload(&gate_up)?,
        down: upload(&down)?,
    })
}

/// Rearrange row-major bytes into 16x16 tiles in the order the 8-bit kernel's lanes read them (see
/// "8-bit weights" in `kernels.cu`), padded with zeros to whole tiles.
fn to_lane_tiles<T: Copy + Default>(values: &[T], rows: usize, cols: usize) -> Vec<T> {
    let chunks = cols.div_ceil(TILE);
    let groups = rows.div_ceil(TILE);
    let mut tiles = vec![T::default(); groups * chunks * TILE * TILE];
    let at = |row: usize, column: usize| -> T {
        match row < rows && column < cols {
            true => values[row * cols + column],
            false => T::default(),
        }
    };
    for group in 0..groups {
        for chunk in 0..chunks {
            let tile = &mut tiles[(group * chunks + chunk) * TILE * TILE..][..TILE * TILE];
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                for half in 0..2 {
                    let row = group * TILE + half * 8 + g;
                    for (j, k) in [2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9]
                        .into_iter()
                        .enumerate()
                    {
                        tile[lane * 8 + half * 4 + j] = at(row, chunk * TILE + k);
                    }
                }
            }
        }
    }
    tiles
}

/// Weights in a group sharing one scale, for 8-bit weights.
const QUANT_GROUP: usize = 128;

/// Each row as signed bytes, with a scale for each group of 128 (its largest magnitude over 127)
/// so that `value ≈ byte * scale`. Scales are `[rows, groups]`.
fn quantize_groups(values: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let groups = cols.div_ceil(QUANT_GROUP);
    let mut bytes = vec![0i8; rows * cols];
    let mut scales = vec![0.0f32; rows * groups];
    for (r, row) in values.chunks_exact(cols).enumerate() {
        for (g, group) in row.chunks(QUANT_GROUP).enumerate() {
            let largest = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = largest / 127.0;
            scales[r * groups + g] = scale;
            if scale > 0.0 {
                let out = &mut bytes[r * cols + g * QUANT_GROUP..][..group.len()];
                for (byte, &v) in out.iter_mut().zip(group) {
                    *byte = (v / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (bytes, scales)
}

/// Floats needed for one 16x16 tile for each tile of a `[batch_rows, rows]` output.
fn tile_scratch_elements(batch_rows: usize, rows: usize) -> usize {
    batch_rows.div_ceil(TILE) * rows.div_ceil(TILE) * TILE * TILE
}

impl Backend for CudaBackend {
    fn attach(&mut self) {
        self.gpu
            .attach()
            .expect("couldn't use the GPU from this thread");
    }

    fn describe(&self) -> String {
        let weights = match (self.precision, self.quantization) {
            _ if self.layers.first().is_some_and(|l| l.qkv.fp8) => "8-bit float",
            _ if self
                .layers
                .first()
                .is_some_and(|l| l.qkv.zeros != Pointer::NULL) =>
            {
                "4-bit"
            }
            (_, Some(Quantization::Int8)) => "8-bit",
            (Precision::Bf16, None) => "16-bit",
            (Precision::F32, None) => "32-bit",
        };
        format!("{} (CUDA, {weights} weights)", self.gpu_name)
    }

    fn forward(&mut self, rows: &[Row]) {
        self.try_forward(rows).expect("the GPU failed");
    }

    fn hidden(&self, row: usize) -> &[f32] {
        &self.hidden[row * self.dim..(row + 1) * self.dim]
    }

    fn logits(&self, row: usize) -> Logits<'_> {
        let (start, end) = (self.candidate_start[row], self.candidate_end[row]);
        if start == u32::MAX {
            let index = end as usize * self.vocab_size;
            Logits::All(&self.logits[index..index + self.vocab_size])
        } else {
            Logits::Candidates(&self.candidates[start as usize..end as usize])
        }
    }

    fn pool(&self) -> &Pool {
        &self.pool
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn page_size(&self) -> usize {
        match self.precision {
            Precision::Bf16 => TILE,
            Precision::F32 => 1,
        }
    }

    fn add_adapter(&mut self, adapter: &LoraAdapter) -> Result<()> {
        ensure!(
            adapter.layers.len() == self.layers.len(),
            "{}: made for a model with {} layers, not {}",
            adapter.name,
            adapter.layers.len(),
            self.layers.len()
        );
        for (l, layer) in self.layers.iter().enumerate() {
            let changes_mlp = [LoraModule::Gate, LoraModule::Up, LoraModule::Down]
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
    use crate::safetensors::{bf16_to_f32, f32_to_bf16};

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// Multiply on the GPU and compare with doing the sum by hand, with shapes that land on and
    /// off tile boundaries in each direction. Skips itself on machines without a suitable GPU.
    #[test]
    fn top_candidates_finds_exactly_the_tokens_it_should() {
        let Ok(gpu) = Gpu::open() else {
            return;
        };
        gpu.attach().unwrap();
        let module = gpu.load_kernels(KERNELS_PTX).unwrap();
        let kernels = Kernels::find(&gpu, &module).unwrap();
        let vocab = 151_936;
        let row: Vec<f32> = pseudo_random(vocab, 21).iter().map(|v| v * 40.0).collect();
        let best = row.iter().copied().fold(f32::MIN, f32::max);
        // (gap, limit): no limit; a limit below the count in the smallest band; a large limit.
        let settings = [(0.25, 0u32), (2.0, 1u32), (0.5, 1000u32), (-1.0, 0u32)];
        let rows = settings.len();
        let logits = gpu.alloc(rows * vocab).unwrap();
        let all: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
        gpu.write_now(logits, &all).unwrap();
        let gaps = gpu.alloc(rows).unwrap();
        let limits = gpu.alloc(rows).unwrap();
        gpu.write_now(gaps, &settings.map(|s| s.0)).unwrap();
        gpu.write_now(limits, &settings.map(|s| s.1)).unwrap();
        let results_device = gpu.alloc(rows * 2).unwrap();
        gpu.zero(results_device, rows * 2).unwrap();
        let capacity = 8192;
        let values_device = gpu.alloc(rows * capacity).unwrap();
        let ids_device = gpu.alloc(rows * capacity).unwrap();
        gpu.launch_blocks(
            kernels.top_candidates,
            [rows, 1],
            SCAN_THREADS,
            &[
                Arg::Ptr(logits),
                Arg::Int(vocab),
                Arg::Ptr(gaps),
                Arg::Ptr(limits),
                Arg::Ptr(results_device),
                Arg::Ptr(values_device),
                Arg::Ptr(ids_device),
                Arg::Int(capacity),
            ],
        )
        .unwrap();
        let mut results = vec![0u32; rows * 2];
        let mut values = vec![0f32; rows * capacity];
        let mut ids = vec![0u32; rows * capacity];
        gpu.read(results_device, &mut results).unwrap();
        gpu.read(values_device, &mut values).unwrap();
        gpu.read(ids_device, &mut ids).unwrap();
        gpu.wait().unwrap();

        for (index, &(gap, limit)) in settings.iter().enumerate() {
            let count = results[index * 2] as usize;
            if gap < 0.0 {
                assert_eq!(count, 0, "a negative gap skips the row");
                continue;
            }
            // Without a limit, it's exactly the tokens within the gap. With one, the band the
            // kernel settled on: it must contain the limit's worth of the best tokens.
            let expected: Vec<u32> = (0..vocab as u32)
                .filter(|&i| row[i as usize] >= best - gap)
                .collect();
            let mut got: Vec<u32> = ids[index * capacity..][..count].to_vec();
            got.sort_unstable();
            if limit == 0 {
                assert_eq!(got, expected);
            } else {
                assert!(count >= (limit as usize).min(expected.len()));
                assert!(got.iter().all(|i| expected.binary_search(i).is_ok()));
                // Nothing that made the cut can be worse than something left out of it.
                let cut = got
                    .iter()
                    .map(|&i| row[i as usize])
                    .fold(f32::MAX, f32::min);
                assert!(
                    (0..vocab as u32)
                        .all(|i| got.binary_search(&i).is_ok() || row[i as usize] <= cut)
                );
            }
            for (&value, &id) in values[index * capacity..][..count]
                .iter()
                .zip(&ids[index * capacity..][..count])
            {
                assert_eq!(value, row[id as usize]);
            }
        }
    }

    #[test]
    fn drawn_tokens_follow_the_distribution() {
        let Ok(gpu) = Gpu::open() else {
            return;
        };
        gpu.attach().unwrap();
        let module = gpu.load_kernels(KERNELS_PTX).unwrap();
        let kernels = Kernels::find(&gpu, &module).unwrap();
        let (vocab, rows, rounds) = (5000, 512, 200);
        let temperature = 0.7f32;
        // A few likely tokens, among many unlikely ones.
        let row: Vec<f32> = pseudo_random(vocab, 33)
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 197 == 3 { 2.0 + v } else { v - 14.0 })
            .collect();
        let logits = gpu.alloc(rows * vocab).unwrap();
        let all: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
        gpu.write_now(logits, &all).unwrap();
        let temperatures = gpu.alloc(rows).unwrap();
        gpu.write_now(temperatures, &vec![temperature; rows])
            .unwrap();
        let keys = gpu.alloc(rows * 2).unwrap();
        let results_device = gpu.alloc(rows * 2).unwrap();
        let mut counts = vec![0u64; vocab];
        let mut draw = 0usize;
        for _ in 0..rounds {
            let round: Vec<u32> = (0..rows)
                .flat_map(|_| {
                    draw += 1;
                    let key = crate::sampler::draw_key(1234, draw);
                    [key as u32, (key >> 32) as u32]
                })
                .collect();
            gpu.write_now(keys, &round).unwrap();
            gpu.launch_blocks(
                kernels.draw_tokens,
                [rows, 1],
                SCAN_THREADS,
                &[
                    Arg::Ptr(logits),
                    Arg::Int(vocab),
                    Arg::Ptr(temperatures),
                    Arg::Ptr(keys),
                    Arg::Ptr(results_device),
                ],
            )
            .unwrap();
            let mut results = vec![0u32; rows * 2];
            gpu.read(results_device, &mut results).unwrap();
            gpu.wait().unwrap();
            for result in results.chunks(2) {
                counts[result[1] as usize] += 1;
            }
        }

        let max = row.iter().copied().fold(f32::MIN, f32::max);
        let weights: Vec<f64> = row
            .iter()
            .map(|&l| (((l - max) / temperature) as f64).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let draws = (rows * rounds) as f64;
        // Pearson's chi-squared over the 40 likeliest tokens, plus everything else as one bin.
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
        let (top, rest) = order.split_at(40);
        let bins: Vec<(f64, f64)> = top
            .iter()
            .map(|&i| (counts[i] as f64, draws * weights[i] / total))
            .chain(std::iter::once((
                rest.iter().map(|&i| counts[i] as f64).sum(),
                rest.iter().map(|&i| draws * weights[i] / total).sum(),
            )))
            .collect();
        let chi2: f64 = bins.iter().map(|(o, e)| (o - e) * (o - e) / e).sum();
        // With 40 degrees of freedom, 100 is out past the 1 in 10 million mark.
        assert!(chi2 < 100.0, "chi-squared {chi2}: {bins:?}");
    }

    #[test]
    fn matmul_with_4_bit_weights_matches_arithmetic() {
        let Ok(gpu) = Gpu::open() else {
            return;
        };
        gpu.attach().unwrap();
        let module = gpu.load_kernels(KERNELS_PTX).unwrap();
        let kernels = Kernels::find(&gpu, &module).unwrap();
        // Whole groups of 128 columns, rows past one tile and not a whole number of them.
        for (m, n, batch) in [
            (16, 128, 1),
            (40, 256, 17),
            (24, 384, 3),
            (64, 256, 64),
            (8, 128, 2),
        ] {
            for (use_bias, accumulate) in [(false, false), (true, true)] {
                let groups = n / QUANT_GROUP;
                let values: Vec<u8> = (0..m * n).map(|i| ((i * 7 + i / 5) % 16) as u8).collect();
                let zeros: Vec<u8> = (0..m * groups).map(|i| ((i * 5) % 17) as u8).collect();
                let scales: Vec<f32> = pseudo_random(m * groups, 13)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect();
                let dequantized: Vec<f32> = (0..m * n)
                    .map(|i| {
                        let g = (i / n) * groups + (i % n) / QUANT_GROUP;
                        (values[i] as f32 - zeros[g] as f32) * scales[g]
                    })
                    .collect();
                let x_values = pseudo_random(batch * n, 7);
                let bias_values = pseudo_random(m, 5);
                let start_values = pseudo_random(batch * m, 3);
                let packed = pack_int4(&values, m, n);
                let data = gpu.alloc(packed.len().div_ceil(4)).unwrap();
                gpu.write_now(data, &packed).unwrap();
                let scales_device = gpu.alloc(scales.len()).unwrap();
                gpu.write_now(scales_device, &scales).unwrap();
                let zeros_device = gpu.alloc(zeros.len().div_ceil(4)).unwrap();
                gpu.write_now(zeros_device, &zeros).unwrap();
                let x16: Vec<u16> = x_values.iter().map(|&v| f32_to_bf16(v)).collect();
                let rounded_x: Vec<f32> = x16.iter().map(|&h| bf16_to_f32(h)).collect();
                let x = input_buffer(&gpu, Precision::Bf16, batch + PADDING_ROWS, n).unwrap();
                gpu.write_now(x, &to_tiles(&x16, batch, n)).unwrap();
                let bias = match use_bias {
                    true => {
                        let device = gpu.alloc(m).unwrap();
                        gpu.write_now(device, &bias_values).unwrap();
                        device
                    }
                    false => Pointer::NULL,
                };
                let out = gpu.alloc((batch + PADDING_ROWS) * m).unwrap();
                gpu.write_now(out, &start_values).unwrap();
                let matrix = Matrix {
                    data,
                    rows: m,
                    cols: n,
                    bias,
                    scratch: Pointer::NULL,
                    scales: scales_device,
                    fp8: false,
                    zeros: zeros_device,
                };
                launch_matmul(
                    &gpu,
                    &kernels,
                    Precision::Bf16,
                    &matrix,
                    x,
                    batch,
                    out,
                    accumulate,
                )
                .unwrap();
                let mut result = vec![0.0f32; batch * m];
                gpu.read(out, &mut result).unwrap();
                gpu.wait().unwrap();
                for b in 0..batch {
                    for r in 0..m {
                        let mut expected: f64 = (0..n)
                            .map(|i| dequantized[r * n + i] as f64 * rounded_x[b * n + i] as f64)
                            .sum();
                        if use_bias {
                            expected += bias_values[r] as f64;
                        }
                        if accumulate {
                            expected += start_values[b * m + r] as f64;
                        }
                        let got = result[b * m + r] as f64;
                        assert!(
                            (got - expected).abs() < 1e-4 * (1.0 + expected.abs()),
                            "{m}x{n} batch {batch}: [{b}][{r}] = {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matmul_with_8_bit_weights_matches_arithmetic() {
        let Ok(gpu) = Gpu::open() else {
            return;
        };
        gpu.attach().unwrap();
        let module = gpu.load_kernels(KERNELS_PTX).unwrap();
        let kernels = Kernels::find(&gpu, &module).unwrap();
        // Columns past one group of 128 (and not a whole number of them), rows past one tile.
        for (m, n, batch) in [
            (16, 16, 1),
            (40, 144, 17),
            (33, 300, 3),
            (64, 256, 64),
            (7, 129, 2),
        ] {
            for (use_bias, accumulate) in [(false, false), (true, true)] {
                let w_values = pseudo_random(m * n, 11);
                let x_values = pseudo_random(batch * n, 7);
                let bias_values = pseudo_random(m, 5);
                let start_values = pseudo_random(batch * m, 3);
                let (bytes, scales) = quantize_groups(&w_values, m, n);
                let groups = n.div_ceil(QUANT_GROUP);
                let dequantized: Vec<f32> = (0..m * n)
                    .map(|i| bytes[i] as f32 * scales[(i / n) * groups + (i % n) / QUANT_GROUP])
                    .collect();
                let tiles = to_lane_tiles(&bytes, m, n);
                let data = gpu.alloc(tiles.len().div_ceil(4)).unwrap();
                gpu.write_now(data, &tiles).unwrap();
                let scales_device = gpu.alloc(scales.len()).unwrap();
                gpu.write_now(scales_device, &scales).unwrap();
                let x16: Vec<u16> = x_values.iter().map(|&v| f32_to_bf16(v)).collect();
                let rounded_x: Vec<f32> = x16.iter().map(|&h| bf16_to_f32(h)).collect();
                let x = input_buffer(&gpu, Precision::Bf16, batch + PADDING_ROWS, n).unwrap();
                gpu.write_now(x, &to_tiles(&x16, batch, n)).unwrap();
                let bias = match use_bias {
                    true => {
                        let device = gpu.alloc(m).unwrap();
                        gpu.write_now(device, &bias_values).unwrap();
                        device
                    }
                    false => Pointer::NULL,
                };
                let out = gpu.alloc((batch + PADDING_ROWS) * m).unwrap();
                gpu.write_now(out, &start_values).unwrap();
                let matrix = Matrix {
                    data,
                    rows: m,
                    cols: n,
                    bias,
                    scratch: Pointer::NULL,
                    scales: scales_device,
                    fp8: false,
                    zeros: Pointer::NULL,
                };
                launch_matmul(
                    &gpu,
                    &kernels,
                    Precision::Bf16,
                    &matrix,
                    x,
                    batch,
                    out,
                    accumulate,
                )
                .unwrap();
                let mut result = vec![0.0f32; batch * m];
                gpu.read(out, &mut result).unwrap();
                gpu.wait().unwrap();
                for b in 0..batch {
                    for r in 0..m {
                        let mut expected: f64 = (0..n)
                            .map(|i| dequantized[r * n + i] as f64 * rounded_x[b * n + i] as f64)
                            .sum();
                        if use_bias {
                            expected += bias_values[r] as f64;
                        }
                        if accumulate {
                            expected += start_values[b * m + r] as f64;
                        }
                        let got = result[b * m + r] as f64;
                        assert!(
                            (got - expected).abs() < 5e-4 * (1.0 + expected.abs()),
                            "{m}x{n} batch {batch}: [{b}][{r}] = {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matmul_kernels_match_arithmetic() {
        let Ok(gpu) = Gpu::open() else {
            return;
        };
        gpu.attach().unwrap();
        let module = gpu.load_kernels(KERNELS_PTX).unwrap();
        let kernels = Kernels::find(&gpu, &module).unwrap();
        let shapes = [
            (16, 16, 1),
            (32, 16, 16),
            (20, 16, 3),
            (16, 40, 17),
            (40, 24, 33),
            (33, 17, 65),
            (7, 5, 2),
            (64, 64, 64),
            (65, 33, 100),
        ];
        for precision in [Precision::Bf16, Precision::F32] {
            for (m, n, batch) in shapes {
                for (use_bias, accumulate) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let w_values = pseudo_random(m * n, 11);
                    let x_values = pseudo_random(batch * n, 7);
                    let bias_values = pseudo_random(m, 5);
                    let start_values = pseudo_random(batch * m, 3);

                    let (data, x, rounded_w, rounded_x) = match precision {
                        Precision::Bf16 => {
                            let w16: Vec<u16> = w_values.iter().map(|&v| f32_to_bf16(v)).collect();
                            let x16: Vec<u16> = x_values.iter().map(|&v| f32_to_bf16(v)).collect();
                            let tiles = to_tiles(&w16, m, n);
                            let data = gpu.alloc(tiles.len().div_ceil(2)).unwrap();
                            gpu.write_now(data, &tiles).unwrap();
                            let x_tiles = to_tiles(&x16, batch, n);
                            let x = input_buffer(&gpu, precision, batch + PADDING_ROWS, n).unwrap();
                            gpu.write_now(x, &x_tiles).unwrap();
                            let widen =
                                |v: &[u16]| v.iter().map(|&h| bf16_to_f32(h)).collect::<Vec<f32>>();
                            (data, x, widen(&w16), widen(&x16))
                        }
                        Precision::F32 => {
                            let padded = m.div_ceil(BLOCK_ROWS) * BLOCK_ROWS;
                            let data = gpu.alloc(padded * n).unwrap();
                            gpu.zero(data, padded * n).unwrap();
                            gpu.write_now(data, &w_values).unwrap();
                            let x = input_buffer(&gpu, precision, batch + PADDING_ROWS, n).unwrap();
                            gpu.write_now(x, &x_values).unwrap();
                            (data, x, w_values.clone(), x_values.clone())
                        }
                    };
                    let bias = if use_bias {
                        let device = gpu.alloc(m).unwrap();
                        gpu.write_now(device, &bias_values).unwrap();
                        device
                    } else {
                        Pointer::NULL
                    };
                    let out = gpu.alloc((batch + PADDING_ROWS) * m).unwrap();
                    gpu.write_now(out, &start_values).unwrap();
                    let scratch = gpu
                        .alloc(tile_scratch_elements(batch + PADDING_ROWS, m))
                        .unwrap();
                    let matrix = Matrix {
                        data,
                        rows: m,
                        cols: n,
                        bias,
                        scratch,
                        scales: Pointer::NULL,
                        fp8: false,
                        zeros: Pointer::NULL,
                    };
                    launch_matmul(
                        &gpu, &kernels, precision, &matrix, x, batch, out, accumulate,
                    )
                    .unwrap();
                    let mut result = vec![0.0f32; batch * m];
                    gpu.read(out, &mut result).unwrap();
                    gpu.wait().unwrap();

                    for b in 0..batch {
                        for r in 0..m {
                            let mut expected: f64 = (0..n)
                                .map(|i| rounded_w[r * n + i] as f64 * rounded_x[b * n + i] as f64)
                                .sum();
                            if use_bias {
                                expected += bias_values[r] as f64;
                            }
                            if accumulate {
                                expected += start_values[b * m + r] as f64;
                            }
                            let got = result[b * m + r] as f64;
                            assert!(
                                (got - expected).abs() < 5e-4 * (1.0 + expected.abs()),
                                "{precision:?} {m}x{n} batch {batch} bias {use_bias} add {accumulate}: [{b}][{r}] = {got}, expected {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}
