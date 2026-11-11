//! Qwen2.5-VL's vision encoder on the Apple GPU: the same steps as on the CPU
//! ([`crate::vision::VisionEncoder`]), with the language model's kernels. Attention is the
//! language model's too: the image's keys and values are its KV slots, and each patch sees the
//! slots of its own window (or of the whole image, in the layers that look at all of it).

use super::api::{Buffer, Device, Encoder};
use super::{
    ATT_SIMDS, AttnArgs, GatedArgs, Loader, MAX_HEAD_DIM, Matrix, MetalBackend, NORM_THREADS,
    NormArgs, ROW_THREADS, RopeArgs, SIMD, threads,
};
use crate::config::VisionConfig;
use crate::vision::{EPS, Layout, Patches, layout, unorder};
use anyhow::{Context, Result, ensure};

struct Block {
    norm1: Buffer,
    qkv: Matrix,
    proj: Matrix,
    norm2: Buffer,
    /// The MLP's gate and up projections, stacked, each padded with zero rows to a multiple of
    /// 8 (the matrix kernels' step), as `down` is with zero columns.
    gate_up: Matrix,
    down: Matrix,
}

/// The vision encoder's weights on the GPU.
pub(super) struct VisionTower {
    config: VisionConfig,
    /// The MLP's width, padded to a multiple of 8.
    inner: usize,
    patch_embed: Matrix,
    blocks: Vec<Block>,
    merger_norm: Buffer,
    fc1: Matrix,
    fc2: Matrix,
}

impl VisionTower {
    pub(super) fn load(loader: &Loader, c: &VisionConfig) -> Result<Self> {
        let d = c.hidden_size;
        let head_dim = d / c.num_heads;
        let (patch_len, merged) = (c.patch_len(), d * c.spatial_merge_size.pow(2));
        ensure!(
            head_dim <= MAX_HEAD_DIM
                && head_dim.is_multiple_of(4)
                && [d, patch_len, c.out_hidden_size]
                    .iter()
                    .all(|size| size.is_multiple_of(8)),
            "the Metal kernels need the vision encoder's sizes in multiples of 8"
        );
        let inner = c.intermediate_size.next_multiple_of(8);
        let padded = |parts: &[String], [rows, cols]: [usize; 2], shape: [usize; 2]| {
            loader.padded(parts, [rows, cols], shape)
        };
        let blocks = (0..c.depth)
            .map(|i| {
                let p = format!("visual.blocks.{i}");
                let mlp = |name: &str| format!("{p}.mlp.{name}");
                Ok(Block {
                    norm1: loader.norm(&format!("{p}.norm1.weight"), d, false)?,
                    qkv: loader.matrix(&[(format!("{p}.attn.qkv"), 3 * d)], d, 1.0)?,
                    proj: loader.matrix(&[(format!("{p}.attn.proj"), d)], d, 1.0)?,
                    norm2: loader.norm(&format!("{p}.norm2.weight"), d, false)?,
                    gate_up: padded(
                        &[mlp("gate_proj"), mlp("up_proj")],
                        [c.intermediate_size, d],
                        [inner, d],
                    )?,
                    down: padded(&[mlp("down_proj")], [d, c.intermediate_size], [d, inner])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: c.clone(),
            inner,
            // A convolution with as many weights as a patch has values.
            patch_embed: padded(
                &["visual.patch_embed.proj".to_string()],
                [d, patch_len],
                [d, patch_len],
            )?,
            blocks,
            merger_norm: loader.norm("visual.merger.ln_q.weight", d, false)?,
            fc1: loader.matrix(&[("visual.merger.mlp.0".into(), merged)], merged, 1.0)?,
            fc2: loader.matrix(
                &[("visual.merger.mlp.2".into(), c.out_hidden_size)],
                merged,
                1.0,
            )?,
        })
    }

    pub(super) fn weight_bytes(&self) -> usize {
        let blocks: usize = (self.blocks.iter())
            .map(|b| {
                [&b.qkv, &b.proj, &b.gate_up, &b.down]
                    .map(|m| m.data.bytes)
                    .iter()
                    .sum::<usize>()
            })
            .sum();
        blocks
            + [&self.patch_embed, &self.fc1, &self.fc2]
                .map(|m| m.data.bytes)
                .iter()
                .sum::<usize>()
    }
}

impl Loader<'_> {
    /// Matrices of `[rows, cols]` (whatever shape they're stored in), each laid out in
    /// `[padded_rows, padded_cols]` with zeros around it, stacked, and their biases if any.
    fn padded(
        &self,
        parts: &[String],
        [rows, cols]: [usize; 2],
        [padded_rows, padded_cols]: [usize; 2],
    ) -> Result<Matrix> {
        let names: Vec<String> = parts.iter().map(|p| format!("{p}.weight")).collect();
        let bf16 = names.iter().all(|name| self.st.dtype(name) == Some("BF16"));
        let total = parts.len() * padded_rows;
        let data = self
            .device
            .buffer(total * padded_cols * if bf16 { 2 } else { 4 })?;
        // SAFETY: the GPU hasn't seen this buffer yet.
        unsafe { data.slice_mut::<u8>().fill(0) };
        for (i, name) in names.iter().enumerate() {
            let shape = self
                .st
                .shape(name)
                .context("a vision weight is missing")?
                .to_vec();
            ensure!(
                shape.iter().product::<usize>() == rows * cols,
                "{name} has {shape:?} weights, not {rows} x {cols}"
            );
            let first = i * padded_rows * padded_cols;
            // SAFETY: the GPU hasn't seen this buffer yet.
            match bf16 {
                true => place(
                    unsafe { data.slice_mut::<u16>() },
                    &self.st.load_bf16(name, &shape)?,
                    first,
                    cols,
                    padded_cols,
                ),
                false => place(
                    unsafe { data.slice_mut::<f32>() },
                    &self.st.load(name, &shape)?,
                    first,
                    cols,
                    padded_cols,
                ),
            }
        }
        let biases: Vec<String> = parts.iter().map(|p| format!("{p}.bias")).collect();
        let bias = match biases.iter().any(|name| self.st.contains(name)) {
            false => None,
            true => {
                let mut values = vec![0.0; total];
                for (i, name) in biases.iter().enumerate() {
                    if self.st.contains(name) {
                        let bias = self.st.load(name, &[rows])?;
                        values[i * padded_rows..][..rows].copy_from_slice(&bias);
                        self.params.set(self.params.get() + rows);
                    }
                }
                Some(self.upload(&values)?)
            }
        };
        self.params
            .set(self.params.get() + parts.len() * rows * cols);
        Ok(Matrix {
            data,
            bf16,
            rows: total,
            cols: padded_cols,
            bias,
            scale: 1.0,
        })
    }
}

/// Copy `values`' rows (`cols` long) into `to` from `first` on, `padded_cols` apart.
fn place<T: Copy>(to: &mut [T], values: &[T], first: usize, cols: usize, padded_cols: usize) {
    for (r, row) in values.chunks(cols).enumerate() {
        to[first + r * padded_cols..][..cols].copy_from_slice(row);
    }
}

fn upload<T: Copy>(device: &Device, values: &[T]) -> Result<Buffer> {
    let buffer = device.buffer(size_of_val(values).max(4))?;
    // SAFETY: the GPU hasn't seen this buffer yet, and T is plain data.
    unsafe { buffer.slice_mut::<T>()[..values.len()].copy_from_slice(values) };
    Ok(buffer)
}

impl MetalBackend {
    /// An image's tokens for the language model, as the CPU's encoder gives them.
    pub(super) fn encode_image_on_gpu(&self, patches: &Patches) -> Result<Vec<f32>> {
        let tower = self
            .vision
            .as_ref()
            .context("this model has no vision encoder")?;
        let c = &tower.config;
        let (d, heads, inner) = (c.hidden_size, c.num_heads, tower.inner);
        let head_dim = d / heads;
        let unit = c.spatial_merge_size.pow(2);
        let Layout {
            order,
            windows,
            input,
            cos,
            sin,
        } = layout(c, patches)?;
        let n = input.len() / c.patch_len();
        let tokens = n / unit;
        let device = &self.device;
        let floats = |count: usize| device.buffer(count.max(1) * 4);
        let (input, cos, sin) = (
            upload(device, &input)?,
            upload(device, &cos)?,
            upload(device, &sin)?,
        );
        let (x, normed, attended) = (floats(n * d)?, floats(n * d)?, floats(n * d)?);
        let (qkv, keys, values) = (floats(n * 3 * d)?, floats(n * d)?, floats(n * d)?);
        let (gate_up, gated) = (floats(n * 2 * inner)?, floats(n * inner)?);
        let (hidden, out) = (
            floats(tokens * unit * d)?,
            floats(tokens * c.out_hidden_size)?,
        );
        // The patches are their own KV slots. Each sees its window's, or every one.
        let slots: Vec<u32> = (0..n as u32).collect();
        let (mut window_start, mut window_len) = (vec![0u32; n], vec![0u32; n]);
        for w in windows.windows(2) {
            window_start[w[0]..w[1]].fill(w[0] as u32);
            window_len[w[0]..w[1]].fill((w[1] - w[0]) as u32);
        }
        let slots = upload(device, &slots)?;
        let windowed = (upload(device, &window_start)?, upload(device, &window_len)?);
        let whole = (
            upload(device, &vec![0u32; n])?,
            upload(device, &vec![n as u32; n])?,
        );

        let k = &self.kernels;
        let norm = |e: &Encoder, from: &Buffer, to: &Buffer, weight: &Buffer| {
            e.set_pipeline(&k.rmsnorm);
            e.set_buffer(0, from, 0);
            e.set_buffer(1, to, 0);
            e.set_buffer(2, weight, 0);
            e.set_buffer(3, &slots, 0);
            let (dim, stride) = (d as u32, d as u32);
            e.set_bytes(
                4,
                &NormArgs {
                    dim,
                    eps: EPS,
                    select: 0,
                    stride,
                },
            );
            e.dispatch([n, 1], NORM_THREADS);
        };
        let shape = |stride: usize| (stride as u32, heads as u32, head_dim as u32, d as u32);
        device.run(|e| {
            self.matmul(e, &tower.patch_embed, &input, &x, n, false);
            for (l, block) in tower.blocks.iter().enumerate() {
                let (starts, lens) = match c.fullatt_block_indexes.contains(&l) {
                    true => &whole,
                    false => &windowed,
                };
                norm(e, &x, &normed, &block.norm1);
                self.matmul(e, &block.qkv, &normed, &qkv, n, false);
                let (stride, heads, head_dim, d) = shape(3 * d);
                e.set_pipeline(&k.rope_store);
                for (i, buffer) in [&qkv, &cos, &sin, &slots, &keys, &values]
                    .iter()
                    .enumerate()
                {
                    e.set_buffer(i, buffer, 0);
                }
                let (n_heads, n_kv_heads, q_dim, kv_dim) = (heads, heads, d, d);
                let rope = RopeArgs {
                    stride,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    q_dim,
                    kv_dim,
                };
                e.set_bytes(6, &rope);
                e.dispatch([n, 1], threads(&k.rope_store, ROW_THREADS));
                e.set_pipeline(&k.attention);
                for (i, buffer) in [&qkv, &keys, &values, &slots, starts, lens, &attended]
                    .iter()
                    .enumerate()
                {
                    e.set_buffer(i, buffer, 0);
                }
                e.set_bytes(
                    7,
                    &AttnArgs {
                        stride,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        q_dim,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        softcap: 0.0,
                        window: u32::MAX,
                    },
                );
                e.set_threadgroup_memory(0, ATT_SIMDS * (head_dim as usize + 2) * 4);
                e.dispatch([heads as usize, n], ATT_SIMDS * SIMD);
                self.matmul(e, &block.proj, &attended, &x, n, true);
                norm(e, &x, &normed, &block.norm2);
                self.matmul(e, &block.gate_up, &normed, &gate_up, n, false);
                e.set_pipeline(&k.gated);
                e.set_buffer(0, &gate_up, 0);
                e.set_buffer(1, &gated, 0);
                e.set_bytes(
                    2,
                    &GatedArgs {
                        hidden: inner as u32,
                        gelu: 0,
                    },
                );
                let threads = threads(&k.gated, ROW_THREADS);
                e.dispatch([inner.div_ceil(threads), n], threads);
                self.matmul(e, &block.down, &gated, &x, n, true);
            }
            // Each block of patches becomes one token.
            norm(e, &x, &normed, &tower.merger_norm);
            self.matmul(e, &tower.fc1, &normed, &hidden, tokens, false);
            let count = tokens * unit * d;
            e.set_pipeline(&k.gelu_erf);
            e.set_buffer(0, &hidden, 0);
            e.set_bytes(1, &(count as u32));
            e.dispatch([count.div_ceil(ROW_THREADS), 1], ROW_THREADS);
            self.matmul(e, &tower.fc2, &hidden, &out, tokens, false);
            Ok(())
        })?;
        // SAFETY: the GPU has finished with it.
        let out = unsafe { out.slice::<f32>() };
        Ok(unorder(&order, &out[..tokens * c.out_hidden_size]))
    }
}
