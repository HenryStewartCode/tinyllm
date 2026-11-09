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
use crate::config::{Activation, Architecture, Config};
use crate::model::{LogitUse, Row};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
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

#[derive(Clone, Debug)]
pub struct Options {
    pub precision: Precision,
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
    /// gate_proj and up_proj as one matrix, for the same reason.
    gate_up: Matrix,
    down: Matrix,
    /// The same for the MLP.
    norm_after_mlp: Option<Pointer>,
    k_pool: Pointer,
    v_pool: Pointer,
    rope_table: usize,
    /// How many tokens it can see.
    window: usize,
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
    swiglu_bf16: Kernel,
    geglu_bf16: Kernel,
    rope_store_bf16: Kernel,
    /// One for each entry of `TILE_CHUNK_COUNTS`.
    attention_bf16: Vec<Kernel>,
    top_candidates: Kernel,
    draw_tokens: Kernel,
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
    tokens: Vec<u32>,
    write_slots: Vec<u32>,
    row_offset: Vec<u32>,
    row_len: Vec<u32>,
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

    slot_list_buffer: Growing,
    scratch: Growing,
    step: StepData,
    /// The kernels of a forward pass, recorded once per batch size, then replayed. That saves
    /// a little GPU time per kernel, which only matters when the batch is small.
    graphs: HashMap<usize, Graph>,

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
    pool: Pool,
    gpu_name: String,
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
        let precision = if options.precision == Precision::Bf16 && !fits_tiles {
            eprintln!("This model's attention heads don't fit 16-bit tiles, so using 32-bit.");
            Precision::F32
        } else {
            options.precision
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
            let head_norms = match architecture.head_norms {
                true => Some((
                    upload_vector(&format!("{attn}.q_norm.weight"), head_dim)?,
                    upload_vector(&format!("{attn}.k_norm.weight"), head_dim)?,
                )),
                false => None,
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
            )?;
            let o_proj = upload_stacked(
                &gpu,
                &st,
                precision,
                &mut num_params,
                &[(format!("{attn}.o_proj"), dim)],
                q_dim,
                batch_rows,
            )?;
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
            )?;
            let down = upload_stacked(
                &gpu,
                &st,
                precision,
                &mut num_params,
                &[(format!("{p}.mlp.down_proj"), dim)],
                hidden_dim,
                batch_rows,
            )?;
            layers.push(Layer {
                attn_norm,
                qkv,
                head_norms,
                o_proj,
                norm_after_attn,
                mlp_norm,
                gate_up,
                down,
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
        let added = match architecture.norms_after_blocks {
            true => gpu.alloc(rows * dim)?,
            false => Pointer::NULL,
        };
        let qkv = gpu.alloc(rows * (q_dim + 2 * kv_dim))?;
        let attn_out = input_buffer(&gpu, precision, rows, q_dim)?;
        let gate_up = gpu.alloc(rows * 2 * hidden_dim)?;
        let gated = input_buffer(&gpu, precision, rows, hidden_dim)?;
        let logits_device = gpu.alloc((logits_rows + PADDING_ROWS) * vocab_size)?;
        let results_device = gpu.alloc(options.max_batch * 2)?;
        let candidate_values_device = gpu.alloc(options.max_batch * MAX_CANDIDATES)?;
        let candidate_ids_device = gpu.alloc(options.max_batch * MAX_CANDIDATES)?;
        let rope_slots = 2 * options.max_batch * (head_dim / 2) * architecture.rope_tables.len();
        let step_slots = 13 * options.max_batch + rope_slots + 16;
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
                gpu_name,
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
    fn add_block_output(
        &self,
        w: &Matrix,
        input: Pointer,
        batch: usize,
        norm: Option<Pointer>,
    ) -> Result<()> {
        let Some(norm) = norm else {
            return self.matmul(w, input, batch, self.x, true);
        };
        self.matmul(w, input, batch, self.added, false)?;
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
        step.team_size.clear();
        step.scratch_offset.clear();
        step.logit_rows.clear();
        step.slot_lists.clear();
        step.gaps.clear();
        step.limits.clear();
        step.temperatures.clear();
        step.keys.clear();
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
                step.scratch_offset.push(scratch_total as u32);
                scratch_total += row.slots.len() * self.n_heads;
                if row.want_logits {
                    step.logit_rows.push(index as u32);
                    let (gap, limit, temperature, key) = match row.sampling {
                        LogitUse::Candidates { gap, limit } => (gap, limit as u32, -1.0, 0),
                        LogitUse::Draw { temperature, key } => (-1.0, 0, temperature, key),
                        LogitUse::All => (-1.0, 0, -1.0, 0),
                    };
                    step.gaps.push(gap);
                    step.limits.push(limit);
                    step.temperatures.push(temperature);
                    step.keys.push(key as u32);
                    step.keys.push((key >> 32) as u32);
                }
                let position = (row.slots.len() - 1) as f32;
                for (table, frequencies) in self.architecture.rope_tables.iter().enumerate() {
                    let start = (table * batch + index) * half;
                    for (j, &freq) in frequencies.iter().enumerate() {
                        let angle = position * freq;
                        step.cos[start + j] = angle.cos();
                        step.sin[start + j] = angle.sin();
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

        for layer in &self.layers {
            // Attention block: mix in information from earlier tokens.
            self.rmsnorm(self.x, self.xb, layer.attn_norm, batch, Pointer::NULL)?;
            self.matmul(&layer.qkv, self.xb, batch, self.qkv, false)?;
            if let Some((q_weight, k_weight)) = layer.head_norms {
                self.head_norms(q_weight, k_weight, batch)?;
            }
            self.rope_and_store(layer, batch, step)?;
            self.attention(layer, batch, step)?;
            self.add_block_output(&layer.o_proj, self.attn_out, batch, layer.norm_after_attn)?;

            // MLP block: process each token on its own.
            self.rmsnorm(self.x, self.xb, layer.mlp_norm, batch, Pointer::NULL)?;
            self.matmul(&layer.gate_up, self.xb, batch, self.gate_up, false)?;
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
            self.add_block_output(&layer.down, self.gated, batch, layer.norm_after_mlp)?;
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
            if !self.graphs.contains_key(&batch) {
                let graph = self.record_graph(batch, &step)?;
                self.graphs.insert(batch, graph);
            }
            self.gpu.launch_graph(&self.graphs[&batch])?;
        } else {
            self.launch_layers(batch, &step)?;
        }

        let n_logits = step.n_logits;
        let any_candidates = self.step.gaps.iter().any(|&gap| gap >= 0.0);
        let any_draws = self.step.temperatures.iter().any(|&t| t >= 0.0);
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
fn to_tiles(values: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    let chunks = cols.div_ceil(TILE);
    let groups = rows.div_ceil(TILE);
    let mut tiles = vec![0u16; groups * chunks * TILE * TILE];
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
fn upload_stacked(
    gpu: &Gpu,
    st: &SafeTensors,
    precision: Precision,
    num_params: &mut usize,
    layers: &[(String, usize)],
    cols: usize,
    batch_rows: usize,
) -> Result<Matrix> {
    let rows: usize = layers.iter().map(|(_, rows)| rows).sum();
    let padded = rows.div_ceil(BLOCK_ROWS) * BLOCK_ROWS;
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
    })
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
        let weights = match self.precision {
            Precision::Bf16 => "16-bit",
            Precision::F32 => "32-bit",
        };
        format!("{} (CUDA, {weights} weights)", self.gpu_name)
    }

    fn forward(&mut self, rows: &[Row]) {
        self.try_forward(rows).expect("the GPU failed");
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
