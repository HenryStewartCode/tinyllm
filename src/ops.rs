//! Numeric kernels. Everything is f32 on plain slices (weight matrices may be stored as bfloat16);
//! batches are matrices stored row-major as one flat slice, so row `b` of a `[batch, dim]` matrix
//! is `m[b * dim..(b + 1) * dim]`.
//!
//! The kernels add numbers up in the same order whatever the shape of the batch, so a sequence
//! gets bit-identical results no matter which other sequences it shares a batch with.

use crate::pool::Pool;
use std::cell::Cell;

/// Accumulators in `dot`. Several are needed for the compiler to use SIMD, because float
/// addition isn't associative and a single running sum would have to stay in order.
const DOT_LANES: usize = 16;

/// Accumulators per (weight row, input row) pair in the `matmul` kernel: one SIMD register.
const BLOCK_LANES: usize = if cfg!(target_feature = "avx512f") {
    16
} else if cfg!(target_feature = "avx2") {
    8
} else {
    4
};

/// Rows of a `matmul` handled together, in each direction. Sharing loads between 4 weight rows
/// and 4 input rows is what lets the CPU keep all its floating-point units busy.
const BLOCK_ROWS: usize = 4;

/// `a * b + c`, as a single fused instruction where the CPU has one: half the work of a
/// separate multiply and add. Where it doesn't, `mul_add` would be a slow library call.
#[inline(always)]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "neon", target_feature = "fma")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Add up an array of accumulators pairwise, which also vectorizes.
#[inline(always)]
fn reduce<const N: usize>(mut acc: [f32; N]) -> f32 {
    let mut width = N / 2;
    while width > 0 {
        for i in 0..width {
            acc[i] += acc[i + width];
        }
        width /= 2;
    }
    acc[0]
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_tail) = a.as_chunks::<DOT_LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<DOT_LANES>();
    let mut acc = [0.0f32; DOT_LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for i in 0..DOT_LANES {
            acc[i] = fmadd(x[i], y[i], acc[i]);
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum = fmadd(*x, *y, sum);
    }
    sum
}

/// What a weight matrix's elements can be stored as.
pub trait Weight: Copy + Default + Send + Sync + 'static {
    fn to_f32(self) -> f32;

    /// Four consecutive elements as one NEON register.
    ///
    /// # Safety
    /// `ptr` must point to four readable elements.
    #[cfg(target_arch = "aarch64")]
    unsafe fn load4(ptr: *const Self) -> std::arch::aarch64::float32x4_t;
}

impl Weight for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(ptr: *const Self) -> std::arch::aarch64::float32x4_t {
        // SAFETY: the caller promises four readable elements.
        unsafe { std::arch::aarch64::vld1q_f32(ptr) }
    }
}

/// A bfloat16 number: the top half of an f32's bits. Most checkpoints are stored this way, and
/// widening one to an f32 is exact, so keeping weights like this halves the memory they take
/// and the time spent reading them, and changes no result.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Bf16(pub u16);

impl Weight for Bf16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(ptr: *const Self) -> std::arch::aarch64::float32x4_t {
        use std::arch::aarch64::*;
        // SAFETY: the caller promises four readable elements; `Bf16` is a `u16`.
        unsafe { vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(ptr.cast::<u16>()))) }
    }
}

/// A weight matrix laid out for fast multiplication.
///
/// Rows are grouped into blocks of `BLOCK_ROWS`, and the rows of a block are interleaved every
/// `BLOCK_LANES` elements. Reading a block is then one continuous sweep through memory, which
/// the CPU's prefetcher follows well, and one read brings in the elements of 4 rows at once.
/// Columns that don't fill a whole SIMD register are kept to one side in `tail`.
pub struct Matrix {
    data: Data,
    rows: usize,
    cols: usize,
}

enum Data {
    F32(Layout<f32>),
    Bf16(Layout<Bf16>),
}

struct Layout<T> {
    /// `[blocks, BLOCK_ROWS * interleaved columns]`. A last, partial block is padded with zeros.
    blocks: Vec<T>,
    /// `[rows, tail columns]`
    tail: Vec<T>,
}

impl<T: Weight> Layout<T> {
    fn new(values: &[T], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols, "matrix shape");
        let tail_cols = cols % BLOCK_LANES;
        let chunks = cols / BLOCK_LANES;
        let n_blocks = rows.div_ceil(BLOCK_ROWS);
        let mut blocks = vec![T::default(); n_blocks * BLOCK_ROWS * chunks * BLOCK_LANES];
        let mut tail = Vec::with_capacity(rows * tail_cols);
        for (r, row) in values.chunks_exact(cols).enumerate() {
            let (block, in_block) = (r / BLOCK_ROWS, r % BLOCK_ROWS);
            let (row_chunks, _) = row.as_chunks::<BLOCK_LANES>();
            for (c, values) in row_chunks.iter().enumerate() {
                let at = (block * chunks + c) * BLOCK_ROWS * BLOCK_LANES + in_block * BLOCK_LANES;
                blocks[at..at + BLOCK_LANES].copy_from_slice(values);
            }
            tail.extend_from_slice(&row[chunks * BLOCK_LANES..]);
        }
        Self { blocks, tail }
    }

    fn copy_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        let tail_cols = cols % BLOCK_LANES;
        let chunks = cols / BLOCK_LANES;
        let (block, in_block) = (r / BLOCK_ROWS, r % BLOCK_ROWS);
        let (out_chunks, _) = out.as_chunks_mut::<BLOCK_LANES>();
        for (c, out) in out_chunks.iter_mut().enumerate() {
            let at = (block * chunks + c) * BLOCK_ROWS * BLOCK_LANES + in_block * BLOCK_LANES;
            for (o, w) in out.iter_mut().zip(&self.blocks[at..at + BLOCK_LANES]) {
                *o = w.to_f32();
            }
        }
        let tail = &self.tail[r * tail_cols..(r + 1) * tail_cols];
        for (o, w) in out[chunks * BLOCK_LANES..].iter_mut().zip(tail) {
            *o = w.to_f32();
        }
    }
}

impl Matrix {
    /// Build from row-major `[rows, cols]` values.
    pub fn from_rows(values: &[f32], rows: usize, cols: usize) -> Self {
        Self {
            data: Data::F32(Layout::new(values, rows, cols)),
            rows,
            cols,
        }
    }

    /// Build from row-major `[rows, cols]` bfloat16 values, keeping them that way.
    pub fn from_rows_bf16(values: &[u16], rows: usize, cols: usize) -> Self {
        // SAFETY: `Bf16` is a `repr(Rust)` newtype of `u16` with the same size and alignment,
        // and every bit pattern is valid for both.
        let values: &[Bf16] =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) };
        Self {
            data: Data::Bf16(Layout::new(values, rows, cols)),
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes taken by the elements.
    pub fn bytes(&self) -> usize {
        match &self.data {
            Data::F32(layout) => (layout.blocks.len() + layout.tail.len()) * 4,
            Data::Bf16(layout) => (layout.blocks.len() + layout.tail.len()) * 2,
        }
    }

    /// Copy row `r` into `out`, which is how an embedding is looked up.
    pub fn copy_row(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows && out.len() == self.cols);
        match &self.data {
            Data::F32(layout) => layout.copy_row(r, self.cols, out),
            Data::Bf16(layout) => layout.copy_row(r, self.cols, out),
        }
    }
}

/// Dot products of one block of weight rows with `X` input rows.
#[inline(always)]
fn dot_block<T: Weight, const X: usize>(
    block: &[T],
    tail: &[T],
    x: [&[f32]; X],
) -> [[f32; X]; BLOCK_ROWS] {
    let (chunks, _) = block.as_chunks::<{ BLOCK_ROWS * BLOCK_LANES }>();
    let inputs = x.map(|x| x.as_chunks::<BLOCK_LANES>().0);
    assert!(inputs.iter().all(|i| i.len() >= chunks.len()));
    let mut out = block_sums(chunks, inputs);
    let tail_cols = tail.len() / BLOCK_ROWS;
    for a in 0..BLOCK_ROWS {
        for b in 0..X {
            for i in 0..tail_cols {
                let xi = x[b][chunks.len() * BLOCK_LANES + i];
                out[a][b] = fmadd(tail[a * tail_cols + i].to_f32(), xi, out[a][b]);
            }
        }
    }
    out
}

/// The heart of the engine. Written by hand for ARM, because the compiler's own vectorization
/// of this loop was around half as fast: each weight vector is loaded once and multiplied by
/// `X` input vectors, with one accumulator register per pair.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn block_sums<T: Weight, const X: usize>(
    chunks: &[[T; BLOCK_ROWS * BLOCK_LANES]],
    inputs: [&[[f32; BLOCK_LANES]]; X],
) -> [[f32; X]; BLOCK_ROWS] {
    use std::arch::aarch64::*;
    // SAFETY: NEON is always available on aarch64, and `dot_block` checked that each input has
    // at least `chunks.len()` chunks. Pointers come from those slices and stay in bounds.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); X]; BLOCK_ROWS];
        let ptrs = inputs.map(|i| i.as_ptr().cast::<f32>());
        for (c, w) in chunks.iter().enumerate() {
            let w = w.as_ptr();
            let wv = [
                T::load4(w),
                T::load4(w.add(BLOCK_LANES)),
                T::load4(w.add(2 * BLOCK_LANES)),
                T::load4(w.add(3 * BLOCK_LANES)),
            ];
            for b in 0..X {
                let xv = vld1q_f32(ptrs[b].add(c * BLOCK_LANES));
                acc[0][b] = vfmaq_f32(acc[0][b], wv[0], xv);
                acc[1][b] = vfmaq_f32(acc[1][b], wv[1], xv);
                acc[2][b] = vfmaq_f32(acc[2][b], wv[2], xv);
                acc[3][b] = vfmaq_f32(acc[3][b], wv[3], xv);
            }
        }
        // Same order as `reduce` on the 4 lanes: (0 + 2) + (1 + 3).
        acc.map(|row| {
            row.map(|a| {
                let [l0, l1, l2, l3]: [f32; 4] = std::mem::transmute(a);
                (l0 + l2) + (l1 + l3)
            })
        })
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn block_sums<T: Weight, const X: usize>(
    chunks: &[[T; BLOCK_ROWS * BLOCK_LANES]],
    inputs: [&[[f32; BLOCK_LANES]]; X],
) -> [[f32; X]; BLOCK_ROWS] {
    let mut acc = [[[0.0f32; BLOCK_LANES]; X]; BLOCK_ROWS];
    for (c, w) in chunks.iter().enumerate() {
        for a in 0..BLOCK_ROWS {
            for b in 0..X {
                let x = &inputs[b][c];
                for l in 0..BLOCK_LANES {
                    acc[a][b][l] = fmadd(w[a * BLOCK_LANES + l].to_f32(), x[l], acc[a][b][l]);
                }
            }
        }
    }
    acc.map(|row| row.map(reduce))
}

/// One block of weight rows times every row of `x`, into `out: [BLOCK_ROWS, batch]`.
#[inline(always)]
fn dot_rows<T: Weight>(out: &mut [f32], block: &[T], tail: &[T], x: &[f32], cols: usize) {
    let batch = x.len() / cols;
    assert!(block.len() <= BLOCK_ROWS * cols && out.len() == BLOCK_ROWS * batch);
    let row = |b: usize| &x[b * cols..(b + 1) * cols];
    fn store<const X: usize>(out: &mut [f32], b: usize, r: [[f32; X]; BLOCK_ROWS]) {
        let batch = out.len() / BLOCK_ROWS;
        for (a, values) in r.iter().enumerate() {
            out[a * batch + b..][..X].copy_from_slice(values);
        }
    }
    let mut b = 0;
    while batch - b >= BLOCK_ROWS {
        let rows = [row(b), row(b + 1), row(b + 2), row(b + 3)];
        store(out, b, dot_block(block, tail, rows));
        b += BLOCK_ROWS;
    }
    match batch - b {
        3 => store(
            out,
            b,
            dot_block(block, tail, [row(b), row(b + 1), row(b + 2)]),
        ),
        2 => store(out, b, dot_block(block, tail, [row(b), row(b + 1)])),
        1 => store(out, b, dot_block(block, tail, [row(b)])),
        _ => {}
    }
}

/// Matrix product: `out[b][m] = dot(w[m], x[b])`, with `x: [batch, w.cols]` and
/// `out: [batch, w.rows]`. `scratch` is working space, reused between calls.
///
/// Nearly all of the time per step goes here. Each block of weight rows is multiplied with
/// every input row while it's in the CPU cache, so a batch costs one trip through memory for the
/// weights however large it is, which is why batching pays off so much.
pub fn matmul(pool: &Pool, out: &mut [f32], scratch: &mut Vec<f32>, w: &Matrix, x: &[f32]) {
    match &w.data {
        Data::F32(layout) => matmul_layout(pool, out, scratch, layout, w.rows, w.cols, x),
        Data::Bf16(layout) => matmul_layout(pool, out, scratch, layout, w.rows, w.cols, x),
    }
}

fn matmul_layout<T: Weight>(
    pool: &Pool,
    out: &mut [f32],
    scratch: &mut Vec<f32>,
    w: &Layout<T>,
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    let batch = x.len() / cols;
    assert_eq!(x.len(), batch * cols, "matmul: input shape");
    assert_eq!(out.len(), batch * rows, "matmul: output shape");
    let n_blocks = rows.div_ceil(BLOCK_ROWS);
    let tail_cols = cols % BLOCK_LANES;
    let block_len = BLOCK_ROWS * (cols - tail_cols);

    // The threads divide up the weight rows, so results come out as [m, batch] and get flipped
    // afterwards. For a single input and whole blocks, that's already the final layout.
    let in_place = batch == 1 && rows.is_multiple_of(BLOCK_ROWS);
    let target: &mut [f32] = if in_place {
        out
    } else {
        scratch.resize(n_blocks * BLOCK_ROWS * batch, 0.0);
        scratch
    };
    let grain = (GRAIN_FLOPS / (BLOCK_ROWS * cols * batch)).max(1);
    pool.for_chunks(target, BLOCK_ROWS * batch, grain, |i, out| {
        let block = &w.blocks[i * block_len..(i + 1) * block_len];
        // The tail of the last block is padded with zeros if that block isn't full.
        if tail_cols == 0 {
            dot_rows(out, block, &[], x, cols);
        } else {
            let mut tail = [T::default(); BLOCK_ROWS * BLOCK_LANES];
            let block_rows = BLOCK_ROWS.min(rows - i * BLOCK_ROWS);
            tail[..block_rows * tail_cols]
                .copy_from_slice(&w.tail[i * BLOCK_ROWS * tail_cols..][..block_rows * tail_cols]);
            dot_rows(out, block, &tail[..BLOCK_ROWS * tail_cols], x, cols);
        }
    });

    if !in_place {
        let scratch = &*scratch;
        pool.for_chunks(out, rows, row_grain(rows), |b, row| {
            for (i, o) in row.iter_mut().enumerate() {
                *o = scratch[i * batch + b];
            }
        });
    }
}

/// Don't split work across threads into pieces smaller than this many multiply-adds.
const GRAIN_FLOPS: usize = 16 * 1024;

/// How many rows of `dim` elements make up one piece of cheap per-element work.
#[inline]
fn row_grain(dim: usize) -> usize {
    (GRAIN_FLOPS / dim).max(1)
}

/// RMSNorm of each stretch of `weight.len()` elements of `x`, in place. For normalizing the
/// heads of the queries and keys.
pub fn rmsnorm_heads(pool: &Pool, x: &mut [f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    pool.for_chunks(x, dim, row_grain(dim), |_, x| {
        let mean_square = dot(x, x) / dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for (v, &w) in x.iter_mut().zip(weight) {
            *v *= scale * w;
        }
    });
}

/// RMSNorm of each row of `x`: `out = x / sqrt(mean(x²) + eps) * weight`.
pub fn rmsnorm(pool: &Pool, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    pool.for_chunks(out, dim, row_grain(dim), |r, out| {
        let x = &x[r * dim..(r + 1) * dim];
        let mean_square = dot(x, x) / dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
            *o = xi * scale * wi;
        }
    });
}

/// `e^x`, to about 1 part in 10^7. Unlike `f32::exp` it's plain arithmetic, so the compiler can
/// run many at once with SIMD. (Polynomial from the Cephes library's `expf`.)
#[inline(always)]
#[allow(clippy::manual_clamp)] // min and max are single SIMD instructions; clamp isn't
fn fast_exp(x: f32) -> f32 {
    const LN2_INV: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.min(88.0).max(-87.0);
    let n = (x * LN2_INV).round();
    let r = x - n * LN2_HI - n * LN2_LO;
    let mut p = 1.987_569_1e-4;
    p = p * r + 1.398_2e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 0.166_666_65;
    p = p * r + 0.5;
    let value = p * (r * r) + r + 1.0;
    value * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// In-place softmax, shifted by the max for numerical stability.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = fast_exp(*v - max);
    }
    let inv_sum = 1.0 / sum(x);
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

fn sum(x: &[f32]) -> f32 {
    let (chunks, tail) = x.as_chunks::<DOT_LANES>();
    let mut acc = [0.0f32; DOT_LANES];
    for chunk in chunks {
        for i in 0..DOT_LANES {
            acc[i] += chunk[i];
        }
    }
    reduce(acc) + tail.iter().sum::<f32>()
}

/// The SwiGLU activation over a batch: `gate = silu(gate) * up`, where `silu(x)` is
/// `x * sigmoid(x)`.
pub fn swiglu(pool: &Pool, gate: &mut [f32], up: &[f32], dim: usize) {
    pool.for_chunks(gate, dim, row_grain(4 * dim), |r, gate| {
        let up = &up[r * dim..(r + 1) * dim];
        for (g, &u) in gate.iter_mut().zip(up) {
            *g = *g / (1.0 + fast_exp(-*g)) * u;
        }
    });
}

/// The gated GELU activation over a batch: `gate = gelu(gate) * up`, where
/// `gelu(x) = x * (1 + tanh(c * (x + 0.044715 * x^3))) / 2`.
///
/// That's the same as `x / (1 + e^(-2y))` with `y` the inside of the `tanh`, which only needs
/// the exponential.
pub fn gated_gelu(pool: &Pool, gate: &mut [f32], up: &[f32], dim: usize) {
    pool.for_chunks(gate, dim, row_grain(4 * dim), |r, gate| {
        let up = &up[r * dim..(r + 1) * dim];
        for (g, &u) in gate.iter_mut().zip(up) {
            let x = *g;
            let y = GELU_SCALE * (x + 0.044715 * x * x * x);
            *g = x / (1.0 + fast_exp(-2.0 * y)) * u;
        }
    });
}

/// `sqrt(2 / pi)`
const GELU_SCALE: f32 = 0.797_884_6;

/// `cap * tanh(x / cap)`, which leaves small numbers about the same and keeps large ones from
/// going past `cap`. As in [`gated_gelu`] this is written so that it only needs the exponential.
#[inline(always)]
pub fn soft_cap(x: f32, cap: f32) -> f32 {
    cap * (1.0 - 2.0 / (1.0 + fast_exp(2.0 * x / cap)))
}

/// `out[row] += x[row]`
pub fn add_rows(pool: &Pool, out: &mut [f32], x: &[f32], dim: usize) {
    pool.for_chunks(out, dim, row_grain(dim), |r, out| {
        let x = &x[r * dim..(r + 1) * dim];
        for (o, v) in out.iter_mut().zip(x) {
            *o += v;
        }
    });
}

/// `out[row] += bias` for every row.
pub fn add_bias(out: &mut [f32], bias: &[f32]) {
    for row in out.chunks_exact_mut(bias.len()) {
        for (o, v) in row.iter_mut().zip(bias) {
            *o += v;
        }
    }
}

/// `out += weight * x`
#[inline]
pub fn add_scaled(out: &mut [f32], weight: f32, x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = fmadd(weight, v, *o);
    }
}

/// Rotary position embedding over a batch of `x: [batch, n_heads * head_dim]`, where `cos` and
/// `sin` are `[batch, head_dim / 2]`, one row of rotations for each row's position.
///
/// Pairs of elements are rotated by an angle proportional to the token's position, so the dot
/// product of a query and a key depends on how far apart they are. Hugging Face checkpoints use
/// the "rotate half" layout: element `i` is paired with element `i + head_dim / 2`.
pub fn rope(pool: &Pool, x: &mut [f32], width: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    pool.for_chunks(x, width, row_grain(width), |r, row| {
        let cos = &cos[r * half..(r + 1) * half];
        let sin = &sin[r * half..(r + 1) * half];
        for head in row.chunks_exact_mut(head_dim) {
            let (lo, hi) = head.split_at_mut(half);
            for i in 0..half {
                let (a, b) = (lo[i], hi[i]);
                lo[i] = a * cos[i] - b * sin[i];
                hi[i] = b * cos[i] + a * sin[i];
            }
        }
    });
}

thread_local! {
    static SCRATCH: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// Run `f` with a per-thread buffer of `len` floats, for working space inside parallel loops.
#[inline]
pub fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    let mut buffer = SCRATCH.take();
    buffer.resize(len, 0.0);
    let result = f(&mut buffer);
    SCRATCH.set(buffer);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, cols: usize, seed: f32) -> Vec<f32> {
        (0..n * cols)
            .map(|i| ((i as f32 + seed) * 0.37).sin())
            .collect()
    }

    #[test]
    fn dot_matches_naive_sum() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.25).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-3);
    }

    #[test]
    fn matmul_multiplies_rows() {
        let w = Matrix::from_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]; // 2 inputs
        let mut out = [0.0; 4];
        matmul(&Pool::new(2), &mut out, &mut Vec::new(), &w, &x);
        assert_eq!(out, [-2.0, -2.0, 2.0, 5.0]);
    }

    #[test]
    fn matrix_rows_round_trip() {
        for (m, cols) in [(1, 1), (3, 5), (4, 8), (9, 16), (10, 37)] {
            let values = rows(m, cols, 2.0);
            let w = Matrix::from_rows(&values, m, cols);
            let mut row = vec![0.0; cols];
            for r in 0..m {
                w.copy_row(r, &mut row);
                assert_eq!(row, values[r * cols..(r + 1) * cols]);
            }
        }
    }

    #[test]
    fn bf16_storage_gives_the_same_results() {
        for (m, cols) in [(5, 3), (8, 16), (13, 37), (64, 128)] {
            let bits: Vec<u16> = (0..m * cols)
                .map(|i| crate::safetensors::f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 7.0))
                .collect();
            let values: Vec<f32> = bits.iter().map(|&b| Bf16(b).to_f32()).collect();
            let (full, half) = (
                Matrix::from_rows(&values, m, cols),
                Matrix::from_rows_bf16(&bits, m, cols),
            );
            let pool = Pool::new(2);
            for batch in [1, 3, 6] {
                let x: Vec<f32> = (0..batch * cols).map(|i| (i % 11) as f32 * 0.25).collect();
                let (mut a, mut b) = (vec![0.0; batch * m], vec![0.0; batch * m]);
                matmul(&pool, &mut a, &mut Vec::new(), &full, &x);
                matmul(&pool, &mut b, &mut Vec::new(), &half, &x);
                assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
            }
            let (mut a, mut b) = (vec![0.0; cols], vec![0.0; cols]);
            full.copy_row(m - 1, &mut a);
            half.copy_row(m - 1, &mut b);
            assert_eq!(a, b);
            assert_eq!(half.bytes() * 2, full.bytes());
        }
    }

    #[test]
    fn matmul_is_independent_of_batch_shape() {
        // Every combination of block shapes, with lengths that leave a tail after the SIMD chunks.
        let pool = Pool::new(3);
        for cols in [1, 7, 16, 37] {
            for m in 1..10 {
                let values = rows(m, cols, 1.0);
                let w = Matrix::from_rows(&values, m, cols);
                let x = rows(11, cols, 5.0);
                let mut together = vec![0.0; 11 * m];
                matmul(&pool, &mut together, &mut Vec::new(), &w, &x);
                for b in 0..11 {
                    let mut alone = vec![0.0; m];
                    let input = &x[b * cols..(b + 1) * cols];
                    matmul(&pool, &mut alone, &mut Vec::new(), &w, input);
                    assert_eq!(
                        alone,
                        together[b * m..(b + 1) * m],
                        "cols={cols} m={m} b={b}"
                    );
                    for (i, value) in alone.iter().enumerate() {
                        let row = &values[i * cols..(i + 1) * cols];
                        assert!((value - dot(row, input)).abs() < 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn fast_exp_is_accurate() {
        let mut x = -80.0f32;
        while x < 80.0 {
            let (ours, exact) = (fast_exp(x), (x as f64).exp());
            assert!(
                ((ours as f64 - exact) / exact).abs() < 4e-7,
                "exp({x}) = {ours}, not {exact}"
            );
            x += 0.0173;
        }
        assert!(fast_exp(1000.0) > 1e37);
        assert!(fast_exp(-1000.0) < 1e-37);
    }

    #[test]
    fn softmax_normalizes() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn rope_rotates_pairs() {
        // One head of size 4: pairs are (0, 2) and (1, 3).
        let mut x = [1.0, 2.0, 0.0, 0.0];
        let angle = std::f32::consts::FRAC_PI_2;
        rope(
            &Pool::new(1),
            &mut x,
            4,
            4,
            &[angle.cos(), 1.0],
            &[angle.sin(), 0.0],
        );
        let expected = [0.0, 2.0, 1.0, 0.0]; // first pair rotated 90°, second unchanged
        for (a, b) in x.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{x:?}");
        }
    }
}
