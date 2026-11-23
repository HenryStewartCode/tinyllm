//! Qwen2.5-VL's vision encoder on NVIDIA GPUs: the same steps as on the CPU
//! ([`crate::vision::VisionEncoder`]), with the language model's matrix, norm and RoPE kernels.
//! Attention has its own kernel (`vision_attention`), since the language model's keep a score per
//! key (too much for a whole image) or read keys in pages of 16 (which windows don't fill).

use super::api::{Arg, Gpu, Pointer};
use super::{
    CudaBackend, Matrix, PADDING_ROWS, Precision, SWIGLU_COLUMNS, TILE, input_buffer,
    launch_matmul, to_tiles,
};
use crate::config::VisionConfig;
use crate::safetensors::SafeTensors;
use crate::vision::{EPS, Layout, Patches, layout, unorder};
use anyhow::{Context, Result, ensure};

/// The largest head `vision_attention` takes.
const VISION_HEAD: usize = 128;

struct Block {
    norm1: Pointer,
    qkv: Matrix,
    proj: Matrix,
    norm2: Pointer,
    /// The MLP's gate and up projections, stacked, each padded with zero rows to a multiple of
    /// 16, as `down` is with zero columns.
    gate_up: Matrix,
    down: Matrix,
}

/// The vision encoder's weights on the GPU.
pub(super) struct VisionTower {
    config: VisionConfig,
    /// The MLP's width, padded to a multiple of 16.
    inner: usize,
    patch_embed: Matrix,
    blocks: Vec<Block>,
    merger_norm: Pointer,
    fc1: Matrix,
    fc2: Matrix,
}

/// What loading needs.
struct Loader<'a> {
    gpu: &'a Gpu,
    st: &'a SafeTensors,
    precision: Precision,
    num_params: &'a mut usize,
}

impl VisionTower {
    pub(super) fn load(
        gpu: &Gpu,
        st: &SafeTensors,
        precision: Precision,
        num_params: &mut usize,
        c: &VisionConfig,
    ) -> Result<Self> {
        let d = c.hidden_size;
        let head_dim = d / c.num_heads;
        let (patch_len, merged) = (c.patch_len(), d * c.spatial_merge_size.pow(2));
        ensure!(
            head_dim <= VISION_HEAD
                && head_dim.is_multiple_of(2)
                && [d, merged, c.out_hidden_size]
                    .iter()
                    .all(|size| size.is_multiple_of(TILE)),
            "the CUDA kernels need the vision encoder's sizes in multiples of 16"
        );
        let inner = c.intermediate_size.next_multiple_of(TILE);
        let mut l = Loader {
            gpu,
            st,
            precision,
            num_params,
        };
        let mut blocks = Vec::with_capacity(c.depth);
        for i in 0..c.depth {
            let p = format!("visual.blocks.{i}");
            let mlp = |name: &str| format!("{p}.mlp.{name}");
            blocks.push(Block {
                norm1: l.vector(&format!("{p}.norm1.weight"), d)?,
                qkv: l.padded(&[format!("{p}.attn.qkv")], [3 * d, d], [3 * d, d])?,
                proj: l.padded(&[format!("{p}.attn.proj")], [d, d], [d, d])?,
                norm2: l.vector(&format!("{p}.norm2.weight"), d)?,
                gate_up: l.padded(
                    &[mlp("gate_proj"), mlp("up_proj")],
                    [c.intermediate_size, d],
                    [inner, d],
                )?,
                down: l.padded(&[mlp("down_proj")], [d, c.intermediate_size], [d, inner])?,
            });
        }
        Ok(Self {
            config: c.clone(),
            inner,
            // A convolution with as many weights as a patch has values.
            patch_embed: l.padded(
                &["visual.patch_embed.proj".to_string()],
                [d, patch_len],
                [d, patch_len],
            )?,
            blocks,
            merger_norm: l.vector("visual.merger.ln_q.weight", d)?,
            fc1: l.padded(
                &["visual.merger.mlp.0".to_string()],
                [merged, merged],
                [merged, merged],
            )?,
            fc2: l.padded(
                &["visual.merger.mlp.2".to_string()],
                [c.out_hidden_size, merged],
                [c.out_hidden_size, merged],
            )?,
        })
    }
}

impl Loader<'_> {
    fn vector(&mut self, name: &str, len: usize) -> Result<Pointer> {
        let values = self.st.load(name, &[len])?;
        *self.num_params += len;
        let device = self.gpu.alloc(len)?;
        self.gpu.write_now(device, &values)?;
        Ok(device)
    }

    /// Matrices of `[rows, cols]` (whatever shape they're stored in), each laid out in
    /// `[padded_rows, padded_cols]` with zeros around it, stacked, and their biases if any.
    fn padded(
        &mut self,
        parts: &[String],
        [rows, cols]: [usize; 2],
        [padded_rows, padded_cols]: [usize; 2],
    ) -> Result<Matrix> {
        let total = parts.len() * padded_rows;
        let mut f32_values = Vec::new();
        let mut bf16_values = Vec::new();
        match self.precision {
            Precision::F32 => f32_values.resize(total * padded_cols, 0.0f32),
            Precision::Bf16 => bf16_values.resize(total * padded_cols, 0u16),
        }
        let mut bias: Option<Vec<f32>> = None;
        for (i, part) in parts.iter().enumerate() {
            let name = format!("{part}.weight");
            let shape = self
                .st
                .shape(&name)
                .context("a vision weight is missing")?
                .to_vec();
            ensure!(
                shape.iter().product::<usize>() == rows * cols,
                "{name} has {shape:?} weights, not {rows} x {cols}"
            );
            let first = i * padded_rows * padded_cols;
            match self.precision {
                Precision::F32 => place(
                    &mut f32_values,
                    &self.st.load(&name, &shape)?,
                    first,
                    cols,
                    padded_cols,
                ),
                Precision::Bf16 => place(
                    &mut bf16_values,
                    &self.st.load_bf16(&name, &shape)?,
                    first,
                    cols,
                    padded_cols,
                ),
            }
            *self.num_params += rows * cols;
            let bias_name = format!("{part}.bias");
            if self.st.contains(&bias_name) {
                let values = self.st.load(&bias_name, &[rows])?;
                *self.num_params += rows;
                bias.get_or_insert_with(|| vec![0.0; total])[i * padded_rows..][..rows]
                    .copy_from_slice(&values);
            }
        }
        let gpu = self.gpu;
        let data = match self.precision {
            Precision::F32 => {
                let data = gpu.alloc(total * padded_cols)?;
                gpu.write_now(data, &f32_values)?;
                data
            }
            Precision::Bf16 => {
                let tiles = to_tiles(&bf16_values, total, padded_cols);
                let data = gpu.alloc(tiles.len().div_ceil(2))?;
                gpu.write_now(data, &tiles)?;
                data
            }
        };
        let bias = match bias {
            Some(values) => {
                let device = gpu.alloc(total)?;
                gpu.write_now(device, &values)?;
                device
            }
            None => Pointer::NULL,
        };
        Ok(Matrix {
            data,
            rows: total,
            cols: padded_cols,
            bias,
            scratch: Pointer::NULL,
            scales: Pointer::NULL,
            fp8: false,
            zeros: Pointer::NULL,
        })
    }
}

/// Copy `values`' rows (`cols` long) into `to` from `first` on, `padded_cols` apart.
fn place<T: Copy>(to: &mut [T], values: &[T], first: usize, cols: usize, padded_cols: usize) {
    for (r, row) in values.chunks(cols).enumerate() {
        to[first + r * padded_cols..][..cols].copy_from_slice(row);
    }
}

/// GPU memory for one image, freed when dropped (after waiting for the GPU to be done with it).
struct Buffers<'a> {
    gpu: &'a Gpu,
    pointers: Vec<Pointer>,
}

impl Buffers<'_> {
    fn keep(&mut self, pointer: Pointer) -> Pointer {
        self.pointers.push(pointer);
        pointer
    }

    fn floats(&mut self, count: usize) -> Result<Pointer> {
        let pointer = self.gpu.alloc(count)?;
        Ok(self.keep(pointer))
    }

    fn upload<T: Copy>(&mut self, values: &[T]) -> Result<Pointer> {
        let pointer = self.floats(values.len())?;
        self.gpu.write_now(pointer, values)?;
        Ok(pointer)
    }

    /// A matrix input in the precision's layout.
    fn input(&mut self, precision: Precision, rows: usize, cols: usize) -> Result<Pointer> {
        let pointer = input_buffer(self.gpu, precision, rows, cols)?;
        Ok(self.keep(pointer))
    }
}

impl Drop for Buffers<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.wait();
        for &pointer in &self.pointers {
            let _ = self.gpu.free(pointer);
        }
    }
}

impl CudaBackend {
    /// An image's tokens for the language model, as the CPU's encoder gives them.
    pub(super) fn encode_image_on_gpu(&mut self, patches: &Patches) -> Result<Vec<f32>> {
        // Another backend (a draft model's) may have made its own context current.
        self.gpu.attach()?;
        let tower = self
            .vision
            .as_ref()
            .context("this model has no vision encoder")?;
        let (gpu, k, precision) = (&self.gpu, &self.kernels, self.precision);
        let c = &tower.config;
        let (d, heads, inner) = (c.hidden_size, c.num_heads, tower.inner);
        let head_dim = d / heads;
        let unit = c.spatial_merge_size.pow(2);
        let (patch_len, merged, out_dim) = (c.patch_len(), d * unit, c.out_hidden_size);
        let Layout {
            order,
            windows,
            input,
            cos,
            sin,
        } = layout(c, patches)?;
        let n = input.len() / patch_len;
        let tokens = n / unit;
        // The matrix kernels can touch rows past the last.
        let rows = n + PADDING_ROWS;

        let mut b = Buffers {
            gpu,
            pointers: Vec::new(),
        };
        // Room for the padding rows too.
        let x_in = b.floats(rows * patch_len)?;
        gpu.zero(x_in, rows * patch_len)?;
        gpu.write_now(x_in, &input)?;
        let in_tiles = match precision {
            Precision::Bf16 => {
                let tiles = b.input(precision, rows, patch_len)?;
                gpu.launch(
                    k.tiles_bf16,
                    [n, 1],
                    &[
                        Arg::Ptr(x_in),
                        Arg::Int(patch_len),
                        Arg::Ptr(tiles),
                        Arg::Int(patch_len.div_ceil(TILE)),
                    ],
                )?;
                tiles
            }
            Precision::F32 => x_in,
        };
        let x = b.floats(rows * d)?;
        let xb = b.input(precision, rows, d)?;
        let qkv = b.floats(rows * 3 * d)?;
        let (keys, values, attended) = (b.floats(n * d)?, b.floats(n * d)?, b.floats(n * d)?);
        let gate_up = b.floats(rows * 2 * inner)?;
        let gated = b.input(precision, rows, inner)?;
        let (cos, sin) = (b.upload(&cos)?, b.upload(&sin)?);
        // The patches are their own KV slots. Each sees its window's, or every one.
        let slots = b.upload(&(0..n as i32).collect::<Vec<_>>())?;
        let (mut window_start, mut window_len) = (vec![0i32; n], vec![0i32; n]);
        for w in windows.windows(2) {
            window_start[w[0]..w[1]].fill(w[0] as i32);
            window_len[w[0]..w[1]].fill((w[1] - w[0]) as i32);
        }
        let windowed = (b.upload(&window_start)?, b.upload(&window_len)?);
        let whole = (b.upload(&vec![0i32; n])?, b.upload(&vec![n as i32; n])?);
        let token_rows = tokens + PADDING_ROWS;
        // `[n, d]` row-major is `[tokens, merged]`.
        let normed = b.floats(token_rows * merged)?;
        let m_input = match precision {
            Precision::Bf16 => b.input(precision, token_rows, merged)?,
            Precision::F32 => normed,
        };
        let hidden = b.floats(token_rows * merged)?;
        let out = b.floats(token_rows * out_dim)?;

        let matmul = |w: &Matrix, input: Pointer, batch: usize, output: Pointer, add: bool| {
            launch_matmul(gpu, k, precision, w, input, batch, output, add)
        };
        let norm = |weight: Pointer| match precision {
            Precision::Bf16 => gpu.launch(
                k.rmsnorm_bf16,
                [n, 1],
                &[
                    Arg::Ptr(x),
                    Arg::Ptr(xb),
                    Arg::Ptr(weight),
                    Arg::Int(d),
                    Arg::Float(EPS),
                    Arg::Ptr(Pointer::NULL),
                    Arg::Int(d / TILE),
                    Arg::Int(d),
                ],
            ),
            Precision::F32 => gpu.launch(
                k.rmsnorm,
                [n, 1],
                &[
                    Arg::Ptr(x),
                    Arg::Ptr(xb),
                    Arg::Ptr(weight),
                    Arg::Int(d),
                    Arg::Float(EPS),
                    Arg::Ptr(Pointer::NULL),
                ],
            ),
        };
        // 32-bit values into the tile layout, for the next 16-bit matrix.
        let tiles = |from: Pointer, count: usize, cols: usize, to: Pointer| {
            gpu.launch(
                k.tiles_bf16,
                [count, 1],
                &[
                    Arg::Ptr(from),
                    Arg::Int(cols),
                    Arg::Ptr(to),
                    Arg::Int(cols / TILE),
                ],
            )
        };

        matmul(&tower.patch_embed, in_tiles, n, x, false)?;
        for (l, block) in tower.blocks.iter().enumerate() {
            let (starts, lens) = match c.fullatt_block_indexes.contains(&l) {
                true => whole,
                false => windowed,
            };
            norm(block.norm1)?;
            matmul(&block.qkv, xb, n, qkv, false)?;
            gpu.launch(
                k.rope_store,
                [n, 1],
                &[
                    Arg::Ptr(qkv),
                    Arg::Int(3 * d),
                    Arg::Ptr(cos),
                    Arg::Ptr(sin),
                    Arg::Int(heads),
                    Arg::Int(heads),
                    Arg::Int(head_dim),
                    Arg::Ptr(slots),
                    Arg::Ptr(keys),
                    Arg::Ptr(values),
                ],
            )?;
            gpu.launch(
                k.vision_attention,
                [n, heads],
                &[
                    Arg::Ptr(qkv),
                    Arg::Int(3 * d),
                    Arg::Ptr(keys),
                    Arg::Ptr(values),
                    Arg::Int(d),
                    Arg::Int(head_dim),
                    Arg::Ptr(starts),
                    Arg::Ptr(lens),
                    Arg::Float(1.0 / (head_dim as f32).sqrt()),
                    Arg::Ptr(attended),
                ],
            )?;
            match precision {
                Precision::Bf16 => {
                    tiles(attended, n, d, xb)?;
                    matmul(&block.proj, xb, n, x, true)?;
                }
                Precision::F32 => matmul(&block.proj, attended, n, x, true)?,
            }
            norm(block.norm2)?;
            matmul(&block.gate_up, xb, n, gate_up, false)?;
            match precision {
                Precision::Bf16 => gpu.launch(
                    k.swiglu_bf16,
                    [n, inner.div_ceil(SWIGLU_COLUMNS)],
                    &[
                        Arg::Ptr(gate_up),
                        Arg::Ptr(gated),
                        Arg::Int(inner),
                        Arg::Int(inner / TILE),
                    ],
                )?,
                Precision::F32 => gpu.launch(
                    k.swiglu,
                    [n, 1],
                    &[Arg::Ptr(gate_up), Arg::Ptr(gated), Arg::Int(inner)],
                )?,
            }
            matmul(&block.down, gated, n, x, true)?;
        }

        // Each block of patches becomes one token: normalized row-major in both precisions,
        // since `[n, d]` only reads as `[tokens, merged]` that way.
        gpu.launch(
            k.rmsnorm,
            [n, 1],
            &[
                Arg::Ptr(x),
                Arg::Ptr(normed),
                Arg::Ptr(tower.merger_norm),
                Arg::Int(d),
                Arg::Float(EPS),
                Arg::Ptr(Pointer::NULL),
            ],
        )?;
        if precision == Precision::Bf16 {
            tiles(normed, tokens, merged, m_input)?;
        }
        matmul(&tower.fc1, m_input, tokens, hidden, false)?;
        let count = tokens * merged;
        gpu.launch_blocks(
            k.gelu_erf,
            [count.div_ceil(256), 1],
            256,
            &[Arg::Ptr(hidden), Arg::Int(count)],
        )?;
        let fc2_input = match precision {
            Precision::Bf16 => {
                tiles(hidden, tokens, merged, m_input)?;
                m_input
            }
            Precision::F32 => hidden,
        };
        matmul(&tower.fc2, fc2_input, tokens, out, false)?;
        let mut host = vec![0.0f32; tokens * out_dim];
        gpu.read(out, &mut host)?;
        gpu.wait()?;
        drop(b);
        Ok(unorder(&order, &host))
    }
}
