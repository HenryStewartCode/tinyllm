//! A Llama-style decoder-only transformer. This one layout covers Llama 2/3, Mistral, Qwen2/3,
//! SmolLM, Phi-3, Granite, TinyLlama and, with a few additions, Gemma.
//!
//! The forward pass works on a batch of rows, one row per token. Rows can come from different
//! sequences, or be several consecutive tokens of one sequence. Each row's keys and values are
//! written to its slot in a [`KvPool`], so earlier tokens are never recomputed:
//!
//! ```text
//! x = embed[token]
//! for each layer:
//!     x += attention(rmsnorm(x))   // grouped-query attention, RoPE on queries and keys
//!     x += mlp(rmsnorm(x))         // SwiGLU: down(silu(gate(x)) * up(x))
//! logits = lm_head(rmsnorm(x))
//! ```
//!
//! Where families of models differ from this is described by [`Architecture`]: other
//! activations, an extra normalization of what each block adds, and so on.
//!
//! Weights stored as bfloat16 stay that way and are widened to f32 inside the matrix kernel,
//! which is exact; others become f32 when they're loaded. Every kernel does plain f32 math.

use crate::cache::KvPool;
use crate::config::{Activation, Architecture, ClampedGlu, Config, ExpertNaming, Experts, Routing};
use crate::lora::{LoraAdapter, Module};
use crate::ops::{
    Matrix, add_bias, add_rows, add_scaled, dot, gated_gelu, matmul, rmsnorm, rmsnorm_heads, rope,
    soft_cap, softmax, swiglu, with_scratch,
};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;

/// A dense layer `y = W x + b`, with `W` stored row-major as `[out_dim, in_dim]`.
pub struct Linear {
    weight: Matrix,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn load(st: &SafeTensors, name: &str, out_dim: usize, in_dim: usize) -> Result<Self> {
        Self::load_scaled(st, name, out_dim, in_dim, 1.0)
    }

    /// Load a layer whose output is always multiplied by `scale`, by multiplying its weights.
    fn load_scaled(
        st: &SafeTensors,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        Ok(Self::load_split(st, name, &[out_dim], in_dim, scale)?.remove(0))
    }

    /// Load one matrix as several layers, taking `out_dims[i]` rows for the `i`-th.
    fn load_split(
        st: &SafeTensors,
        name: &str,
        out_dims: &[usize],
        in_dim: usize,
        scale: f32,
    ) -> Result<Vec<Self>> {
        let total = out_dims.iter().sum();
        let weight_name = format!("{name}.weight");
        let bias_name = format!("{name}.bias");
        let mut bias = match st.contains(&bias_name) {
            true => Some(st.load(&bias_name, &[total])?),
            false => None,
        };
        bias.iter_mut().flatten().for_each(|v| *v *= scale);
        // Weights stored as bfloat16 stay that way, which changes no result (see `ops::Bf16`).
        // Scaling them would round them, so scaled ones become 32-bit.
        let keep_bf16 = st.dtype(&weight_name) == Some("BF16") && scale == 1.0;
        let (mut full, mut half) = (Vec::new(), Vec::new());
        match keep_bf16 {
            true => half = st.load_bf16(&weight_name, &[total, in_dim])?,
            false => {
                full = st.load(&weight_name, &[total, in_dim])?;
                full.iter_mut().for_each(|v| *v *= scale);
            }
        }
        let mut start = 0;
        let mut layers = Vec::with_capacity(out_dims.len());
        for &out_dim in out_dims {
            let rows = start * in_dim..(start + out_dim) * in_dim;
            layers.push(Self {
                weight: match keep_bf16 {
                    true => Matrix::from_rows_bf16(&half[rows], out_dim, in_dim),
                    false => Matrix::from_rows(&full[rows], out_dim, in_dim),
                },
                bias: bias.as_ref().map(|b| b[start..start + out_dim].to_vec()),
            });
            start += out_dim;
        }
        Ok(layers)
    }

    /// Apply the layer to every row of `x`.
    fn forward(&self, pool: &Pool, out: &mut [f32], scratch: &mut Vec<f32>, x: &[f32]) {
        matmul(pool, out, scratch, &self.weight, x);
        if let Some(bias) = &self.bias {
            add_bias(out, bias);
        }
    }

    fn num_params(&self) -> usize {
        self.weight.len() + self.bias.as_ref().map_or(0, Vec::len)
    }

    /// One of several matrices stored together in one tensor, `[count, rows, cols]` (or one
    /// piece of such a matrix), taken from `values`, which are bfloat16 if `half` is.
    fn from_part(full: &[f32], half: &[u16], rows: usize, cols: usize) -> Self {
        Self {
            weight: match half.is_empty() {
                true => Matrix::from_rows(full, rows, cols),
                false => Matrix::from_rows_bf16(half, rows, cols),
            },
            bias: None,
        }
    }
}

/// GPT-OSS's activation over a batch: `gate = (u + 1) * g * sigmoid(alpha * g)`, with
/// `g = min(gate, limit)` and `u = clamp(up, -limit, limit)`.
fn clamped_glu(pool: &Pool, gate: &mut [f32], up: &[f32], dim: usize, glu: ClampedGlu) {
    pool.for_chunks(gate, dim, 256, |r, gate| {
        let up = &up[r * dim..(r + 1) * dim];
        for (g, &u) in gate.iter_mut().zip(up) {
            let x = g.min(glu.limit);
            let u = u.clamp(-glu.limit, glu.limit);
            *g = (u + 1.0) * (x / (1.0 + (-glu.alpha * x).exp()));
        }
    });
}

/// A gated MLP: `down(act(gate(x)) * up(x))`.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn num_params(&self) -> usize {
        [&self.gate_proj, &self.up_proj, &self.down_proj]
            .iter()
            .map(|lin| lin.num_params())
            .sum()
    }
}

/// What a layer's feed-forward block is.
enum FeedForward {
    Dense(Mlp),
    /// A mixture of experts (see [`crate::config::Experts`]).
    Sparse {
        router: Linear,
        /// What's added to the router's scores to choose experts (not to weigh them), if
        /// anything (DeepSeek V3's `e_score_correction_bias`).
        correction: Option<Vec<f32>>,
        experts: Vec<Mlp>,
        /// An expert every token goes through, scaled by `sigmoid(gate(x))` if it has a gate.
        shared: Option<Box<(Mlp, Option<Linear>)>>,
    },
}

impl FeedForward {
    fn num_params(&self) -> usize {
        match self {
            FeedForward::Dense(mlp) => mlp.num_params(),
            FeedForward::Sparse {
                router,
                experts,
                shared,
                ..
            } => {
                router.num_params()
                    + experts.iter().map(Mlp::num_params).sum::<usize>()
                    + shared.as_ref().map_or(0, |shared| {
                        shared.0.num_params() + shared.1.as_ref().map_or(0, Linear::num_params)
                    })
            }
        }
    }
}

/// Load a layer's mixture of experts.
fn load_experts(
    st: &SafeTensors,
    p: &str,
    spec: &Experts,
    dim: usize,
    residual: f32,
) -> Result<FeedForward> {
    let h = spec.hidden;
    let mlp = |prefix: &str, [gate, up, down]: [&str; 3], hidden: usize| -> Result<Mlp> {
        Ok(Mlp {
            gate_proj: Linear::load(st, &format!("{prefix}.{gate}"), hidden, dim)?,
            up_proj: Linear::load(st, &format!("{prefix}.{up}"), hidden, dim)?,
            down_proj: Linear::load_scaled(st, &format!("{prefix}.{down}"), dim, hidden, residual)?,
        })
    };
    let (router, experts) = match spec.naming {
        ExpertNaming::Qwen => (
            format!("{p}.mlp.gate"),
            (0..spec.count)
                .map(|e| {
                    mlp(
                        &format!("{p}.mlp.experts.{e}"),
                        ["gate_proj", "up_proj", "down_proj"],
                        h,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        ExpertNaming::Mixtral => (
            format!("{p}.block_sparse_moe.gate"),
            (0..spec.count)
                .map(|e| {
                    mlp(
                        &format!("{p}.block_sparse_moe.experts.{e}"),
                        ["w1", "w3", "w2"],
                        h,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        ExpertNaming::Granite => {
            // Every expert's gate and up projections stacked in one tensor, and every down.
            let input_name = format!("{p}.block_sparse_moe.input_linear.weight");
            let output_name = format!("{p}.block_sparse_moe.output_linear.weight");
            let (input_shape, output_shape) = ([spec.count, 2 * h, dim], [spec.count, dim, h]);
            let keep_bf16 = st.dtype(&input_name) == Some("BF16")
                && st.dtype(&output_name) == Some("BF16")
                && residual == 1.0;
            let (mut input, mut input16, mut output, mut output16) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            match keep_bf16 {
                true => {
                    input16 = st.load_bf16(&input_name, &input_shape)?;
                    output16 = st.load_bf16(&output_name, &output_shape)?;
                }
                false => {
                    input = st.load(&input_name, &input_shape)?;
                    output = st.load(&output_name, &output_shape)?;
                    output.iter_mut().for_each(|v| *v *= residual);
                }
            }
            let part = |full: &[f32], half: &[u16], at: usize, len: usize, rows, cols| {
                let full = full.get(at..at + len).unwrap_or(&[]);
                let half = half.get(at..at + len).unwrap_or(&[]);
                Linear::from_part(full, half, rows, cols)
            };
            let experts = (0..spec.count)
                .map(|e| Mlp {
                    gate_proj: part(&input, &input16, e * 2 * h * dim, h * dim, h, dim),
                    up_proj: part(&input, &input16, (e * 2 + 1) * h * dim, h * dim, h, dim),
                    down_proj: part(&output, &output16, e * dim * h, dim * h, dim, h),
                })
                .collect();
            (format!("{p}.block_sparse_moe.router.layer"), experts)
        }
        ExpertNaming::GptOss => {
            anyhow::ensure!(residual == 1.0, "GPT-OSS experts don't scale their outputs");
            (
                format!("{p}.mlp.router"),
                load_gpt_oss_experts(st, p, spec, dim)?,
            )
        }
        ExpertNaming::DeepSeek => {
            // Each expert by name, or (as transformers 5 saves them) all stacked together.
            let stacked = format!("{p}.mlp.experts.gate_up_proj");
            let experts = match st.contains(&stacked) {
                false => (0..spec.count)
                    .map(|e| {
                        mlp(
                            &format!("{p}.mlp.experts.{e}"),
                            ["gate_proj", "up_proj", "down_proj"],
                            h,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                true => {
                    let gate_up = st.load(&stacked, &[spec.count, 2 * h, dim])?;
                    let mut down =
                        st.load(&format!("{p}.mlp.experts.down_proj"), &[spec.count, dim, h])?;
                    down.iter_mut().for_each(|v| *v *= residual);
                    let part = |values: &[f32], at: usize, rows: usize, cols: usize| {
                        Linear::from_part(&values[at..at + rows * cols], &[], rows, cols)
                    };
                    (0..spec.count)
                        .map(|e| Mlp {
                            gate_proj: part(&gate_up, e * 2 * h * dim, h, dim),
                            up_proj: part(&gate_up, (e * 2 + 1) * h * dim, h, dim),
                            down_proj: part(&down, e * dim * h, dim, h),
                        })
                        .collect()
                }
            };
            (format!("{p}.mlp.gate"), experts)
        }
    };
    let shared = match (spec.shared_hidden, spec.shared_gated) {
        (Some(hidden), true) => Some(Box::new((
            mlp(
                &format!("{p}.mlp.shared_expert"),
                ["gate_proj", "up_proj", "down_proj"],
                hidden,
            )?,
            Some(Linear::load(
                st,
                &format!("{p}.mlp.shared_expert_gate"),
                1,
                dim,
            )?),
        ))),
        (Some(hidden), false) => Some(Box::new((
            mlp(
                &format!("{p}.mlp.shared_experts"),
                ["gate_proj", "up_proj", "down_proj"],
                hidden,
            )?,
            None,
        ))),
        (None, _) => None,
    };
    let correction_name = format!("{router}.e_score_correction_bias");
    let correction = match st.contains(&correction_name) {
        true => Some(st.load(&correction_name, &[spec.count])?),
        false => None,
    };
    Ok(FeedForward::Sparse {
        router: Linear::load(st, &router, spec.count, dim)?,
        correction,
        experts,
        shared,
    })
}

/// GPT-OSS's experts: `gate_up_proj` holds each expert's gate and up projections with their rows
/// alternating, and `down_proj` its down projection, all with biases. They're either plain
/// matrices, `[experts, in, out]`, or packed in MXFP4 as `[experts, out, in]`, which become
/// bfloat16 (exactly).
fn load_gpt_oss_experts(st: &SafeTensors, p: &str, spec: &Experts, dim: usize) -> Result<Vec<Mlp>> {
    let (count, h) = (spec.count, spec.hidden);
    let gate_up = format!("{p}.mlp.experts.gate_up_proj");
    let down = format!("{p}.mlp.experts.down_proj");
    let gate_up_bias = st.load(&format!("{gate_up}_bias"), &[count, 2 * h])?;
    let down_bias = st.load(&format!("{down}_bias"), &[count, dim])?;
    // Each expert's matrices as [out, in].
    let (gate_up_half, down_half, gate_up_full, down_full) =
        match st.load_mxfp4_bf16(&gate_up, count, 2 * h, dim)? {
            Some(packed) => {
                let down = st
                    .load_mxfp4_bf16(&down, count, dim, h)?
                    .context("down_proj needs to be MXFP4 as well")?;
                (packed, down, Vec::new(), Vec::new())
            }
            None => {
                let transpose = |values: Vec<f32>, rows: usize, cols: usize| -> Vec<f32> {
                    let mut out = vec![0.0; values.len()];
                    for e in 0..count {
                        let (from, to) = (&values[e * rows * cols..], &mut out[e * rows * cols..]);
                        for r in 0..rows {
                            for c in 0..cols {
                                to[c * rows + r] = from[r * cols + c];
                            }
                        }
                    }
                    out
                };
                let full_gate_up = st.load(&gate_up, &[count, dim, 2 * h])?;
                let full_down = st.load(&down, &[count, h, dim])?;
                (
                    Vec::new(),
                    Vec::new(),
                    transpose(full_gate_up, dim, 2 * h),
                    transpose(full_down, h, dim),
                )
            }
        };
    let half = !gate_up_half.is_empty();
    Ok((0..count)
        .map(|e| {
            // Gate rows are the even ones, up rows the odd ones.
            let alternate = |parity: usize| -> Linear {
                let rows = (0..h).map(|i| e * 2 * h + 2 * i + parity);
                let bias = rows.clone().map(|r| gate_up_bias[r]).collect();
                let weight = match half {
                    true => {
                        let values: Vec<u16> = rows
                            .flat_map(|r| gate_up_half[r * dim..(r + 1) * dim].iter().copied())
                            .collect();
                        Matrix::from_rows_bf16(&values, h, dim)
                    }
                    false => {
                        let values: Vec<f32> = rows
                            .flat_map(|r| gate_up_full[r * dim..(r + 1) * dim].iter().copied())
                            .collect();
                        Matrix::from_rows(&values, h, dim)
                    }
                };
                Linear {
                    weight,
                    bias: Some(bias),
                }
            };
            let block = e * dim * h..(e + 1) * dim * h;
            Mlp {
                gate_proj: alternate(0),
                up_proj: alternate(1),
                down_proj: Linear {
                    weight: match half {
                        true => Matrix::from_rows_bf16(&down_half[block], dim, h),
                        false => Matrix::from_rows(&down_full[block], dim, h),
                    },
                    bias: Some(down_bias[e * dim..(e + 1) * dim].to_vec()),
                },
            }
        })
        .collect())
}

/// The experts a token goes through, in order of expert, with their weights: the softmax of the
/// router's scores, the `top_k` best (the earlier expert of any tie), scaled to add up to 1 if
/// `renormalize`.
fn route(
    scores: &[f32],
    top_k: usize,
    routing: Routing,
    correction: Option<&[f32]>,
) -> Vec<(usize, f32)> {
    let (sigmoid, groups, top_groups, renormalize, scale) = match routing {
        Routing::Softmax { renormalize } => return route_softmax(scores, top_k, renormalize),
        Routing::DeepSeek {
            sigmoid,
            groups,
            top_groups,
            renormalize,
            scale,
        } => (sigmoid, groups, top_groups, renormalize, scale),
    };
    // DeepSeek: weights from the softmax or the sigmoids; choices by those plus the correction.
    let weights: Vec<f32> = match sigmoid {
        true => scores.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect(),
        false => {
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        }
    };
    let choice: Vec<f32> = match correction {
        Some(bias) => weights.iter().zip(bias).map(|(w, b)| w + b).collect(),
        None => weights.clone(),
    };
    // Only the best groups are chosen from: scored by their best two (V3) or best one (V2).
    let per_group = scores.len() / groups;
    let group_score = |g: usize| -> f32 {
        let mut members: Vec<f32> = choice[g * per_group..(g + 1) * per_group].to_vec();
        members.sort_by(|a, b| b.total_cmp(a));
        match sigmoid {
            true => members.iter().take(2).sum(),
            false => members[0],
        }
    };
    let mut group_order: Vec<usize> = (0..groups).collect();
    group_order.sort_by(|&a, &b| group_score(b).total_cmp(&group_score(a)).then(a.cmp(&b)));
    let allowed = &group_order[..top_groups.min(groups)];
    let mut order: Vec<usize> = (0..scores.len())
        .filter(|&e| allowed.contains(&(e / per_group)))
        .collect();
    order.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]).then(a.cmp(&b)));
    let mut chosen: Vec<(usize, f32)> = order[..top_k].iter().map(|&e| (e, weights[e])).collect();
    if renormalize {
        let total: f32 = chosen.iter().map(|&(_, w)| w).sum::<f32>() + 1e-20;
        chosen.iter_mut().for_each(|(_, w)| *w /= total);
    }
    chosen.iter_mut().for_each(|(_, w)| *w *= scale);
    chosen.sort_by_key(|&(e, _)| e);
    chosen
}

/// The usual routing (see [`route`]).
fn route_softmax(scores: &[f32], top_k: usize, renormalize: bool) -> Vec<(usize, f32)> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| exps[b].total_cmp(&exps[a]).then(a.cmp(&b)));
    let mut chosen: Vec<(usize, f32)> =
        order[..top_k].iter().map(|&e| (e, exps[e] / sum)).collect();
    if renormalize {
        let total: f32 = chosen.iter().map(|&(_, w)| w).sum();
        chosen.iter_mut().for_each(|(_, w)| *w /= total);
    }
    chosen.sort_by_key(|&(e, _)| e);
    chosen
}

struct Layer {
    attn_norm: Vec<f32>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    /// Normalize each head of the queries and of the keys, if the model does that.
    head_norms: Option<(Vec<f32>, Vec<f32>)>,
    /// Each head's attention sink, if the model has them.
    sinks: Option<Vec<f32>>,
    /// DeepSeek's latent attention, which replaces `q_proj`, `k_proj` and `v_proj` (left empty).
    latent: Option<Box<Latent>>,
    o_proj: Linear,
    /// Normalizes what attention adds to the residual stream, if the model does that.
    norm_after_attn: Option<Vec<f32>>,
    mlp_norm: Vec<f32>,
    feed_forward: FeedForward,
    /// The same for the MLP.
    norm_after_mlp: Option<Vec<f32>>,
    rope_table: usize,
    window: Option<usize>,
}

pub struct Model {
    pub config: Config,
    /// `[vocab, dim]`. Row `t` is token `t`'s embedding. Many small models reuse this matrix as
    /// the output projection ("tied embeddings") instead of storing a separate `lm_head`.
    embed_tokens: Linear,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    lm_head: Option<Linear>,
    architecture: Architecture,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Each head's key width (for queries too) and value width, and how much of it RoPE
    /// rotates: `head_dim` for all three, except with DeepSeek's latent attention.
    head_k: usize,
    head_v: usize,
    rope_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    /// LoRA adapters, which rows ask for by number: the first is 1.
    adapters: Vec<Adapter>,
}

/// A LoRA adapter's corrections, as matrices: for each layer and module, `A` and `B` (with the
/// adapter's scale, and for the modules whose output is scaled, that too).
struct Adapter {
    layers: Vec<[Option<(Matrix, Matrix)>; 7]>,
}

/// DeepSeek's latent attention weights for a layer (see [`LatentAttention`]).
struct Latent {
    /// `q_a_proj` and `q_a_layernorm`, when queries go through a latent of their own.
    q_low: Option<(Linear, Vec<f32>)>,
    /// `q_proj`, or `q_b_proj` after the latent: `[heads * (nope + rope), ...]`.
    q: Linear,
    /// `kv_a_proj_with_mqa`: the latent, then the keys' rotated part, `[kv_lora_rank + rope, dim]`.
    kv_a: Linear,
    kv_a_norm: Vec<f32>,
    /// `kv_b_proj`: each head's unrotated key and its value, `[heads * (nope + v), kv_lora_rank]`.
    kv_b: Linear,
}

/// Rotate adjacent pairs `(x[2i], x[2i + 1])` of `x` by RoPE's angle `i`, as DeepSeek lays them out.
fn rotate_pairs(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    for (i, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * cos[i] - b * sin[i];
        pair[1] = b * cos[i] + a * sin[i];
    }
}

/// Working memory for latent attention.
#[derive(Default)]
struct LatentBuffers {
    q_low: Vec<f32>,   // [batch, q_lora_rank]
    q_norm: Vec<f32>,  // the same, normalized
    kv_low: Vec<f32>,  // [batch, kv_lora_rank + rope]
    latent: Vec<f32>,  // [batch, kv_lora_rank]
    normed: Vec<f32>,  // the same, normalized
    up: Vec<f32>,      // [batch, heads * (nope + v)]
    rotated: Vec<f32>, // one token's rotated key part
}

/// One token to run through the model.
pub struct Row<'a> {
    pub token: u32,
    /// The KV slots of this token's sequence, up to and including this token's own, which is
    /// where its keys and values get written. Its position is `slots.len() - 1`.
    pub slots: &'a [u32],
    /// Compute the scores for the token that follows this one.
    pub want_logits: bool,
    /// How the scores will be used, which can save moving them all to the CPU.
    pub sampling: LogitUse,
    /// The LoRA adapter to apply (see [`crate::lora`]): 0 for none, or the number the backend
    /// loaded it as.
    pub adapter: u16,
}

/// What a row's logits are needed for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LogitUse {
    /// Choosing a token from those within `gap` of the best score. Anything further behind can't
    /// affect the result, and with a `limit` over 0, nor can anything but the top `limit` tokens.
    Candidates { gap: f32, limit: usize },
    /// Drawing a token with probability proportional to `exp(score / temperature)`, using `key`
    /// for the randomness, and nothing else. A backend that can do this itself returns the
    /// chosen token as the only candidate; otherwise this is the same as `All`.
    Draw { temperature: f32, key: u64 },
    /// Everything: all the logits have to be available.
    All,
    /// No scores at all, just the final hidden state (after the last norm), as embedding
    /// models use it.
    Hidden,
}

/// The worker threads and working memory for the forward pass.
pub struct Workspace {
    pool: Pool,
    x: Vec<f32>,        // residual stream [batch, dim]
    xb: Vec<f32>,       // input or output of a sub-block [batch, dim]
    xc: Vec<f32>,       // what a sub-block adds to the stream, normalized [batch, dim]
    q: Vec<f32>,        // [batch, n_heads * head_dim]
    k: Vec<f32>,        // [batch, kv_dim]
    v: Vec<f32>,        // [batch, kv_dim]
    attn_out: Vec<f32>, // [batch, n_heads * head_dim]
    gate: Vec<f32>,     // [batch, hidden_dim]
    up: Vec<f32>,       // [batch, hidden_dim]
    cos: Vec<f32>,      // RoPE rotations for each row's position [tables, batch, head_dim / 2]
    sin: Vec<f32>,
    final_x: Vec<f32>, // the rows that want logits [n_logits, dim]
    scratch: Vec<f32>,
    logits: Vec<f32>, // [n_logits, vocab]
    vocab: usize,
    hidden: Vec<f32>, // final hidden states of the rows that want logits, if any want those [n_logits, dim]
    dim: usize,
    moe: MoeBuffers,
    lora: LoraBuffers,
    latent: LatentBuffers,
}

/// Working memory for LoRA corrections.
#[derive(Default)]
struct LoraBuffers {
    rows: Vec<usize>,  // the rows using one adapter
    input: Vec<f32>,   // their inputs [rows, in]
    reduced: Vec<f32>, // A times those [rows, rank]
    delta: Vec<f32>,   // B times that [rows, out]
    scratch: Vec<f32>,
}

/// Working memory for a mixture of experts.
#[derive(Default)]
struct MoeBuffers {
    scores: Vec<f32>, // the router's scores for each expert [batch, experts]
    input: Vec<f32>,  // the rows going to one expert [rows, dim]
    output: Vec<f32>, // what it makes of them [rows, dim]
    sum: Vec<f32>,    // the weighted sum of the experts' outputs [batch, dim]
    gate: Vec<f32>,   // [rows, expert hidden]
    up: Vec<f32>,
    scratch: Vec<f32>,
}

impl Workspace {
    pub fn new(threads: usize) -> Self {
        Self {
            pool: Pool::new(threads),
            x: Vec::new(),
            xb: Vec::new(),
            xc: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            cos: Vec::new(),
            sin: Vec::new(),
            final_x: Vec::new(),
            scratch: Vec::new(),
            logits: Vec::new(),
            vocab: 0,
            hidden: Vec::new(),
            dim: 0,
            moe: MoeBuffers::default(),
            lora: LoraBuffers::default(),
            latent: LatentBuffers::default(),
        }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Scores for the next token after the `i`-th row that asked for them.
    pub fn logits(&self, i: usize) -> &[f32] {
        &self.logits[i * self.vocab..(i + 1) * self.vocab]
    }

    /// The final hidden state of the `i`-th row that asked for logits, if it asked for that.
    pub fn hidden(&self, i: usize) -> &[f32] {
        &self.hidden[i * self.dim..(i + 1) * self.dim]
    }
}

impl Model {
    /// Load `config.json` and the safetensors weights from a Hugging Face model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = Config::load(dir)?;
        let st = SafeTensors::open(dir)?;

        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let architecture = config.architecture()?;
        let latent = architecture.latent;
        // Latent attention has a key and value for every head.
        let n_kv_heads = match latent {
            Some(_) => n_heads,
            None => config.n_kv_heads(),
        };
        let head_dim = config.head_dim();
        let (head_k, head_v, rope_dim) = match latent {
            Some(l) => (l.nope + l.rope, l.v, l.rope),
            None => (head_dim, head_dim, head_dim),
        };
        let hidden_dim = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);

        // Gemma's norm weights are stored as an offset from 1.
        let load_norm = |name: &str, len: usize| -> Result<Vec<f32>> {
            let mut weight = st.load(name, &[len])?;
            weight
                .iter_mut()
                .for_each(|w| *w += architecture.norm_offset);
            Ok(weight)
        };
        // Load the layers side by side: most of the time is spent reading and rearranging.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let results = Mutex::new(
            (0..config.num_hidden_layers)
                .map(|_| None)
                .collect::<Vec<_>>(),
        );
        let residual = architecture.residual_scale;
        let load_layer = |i: usize| -> Result<Layer> {
            let p = format!("model.layers.{i}");
            let attn = format!("{p}.self_attn");
            let shape = architecture.layers[i];
            let dense = |name: &str| match shape.sparse {
                true => Ok(None),
                false => Linear::load(&st, &format!("{p}.mlp.{name}"), hidden_dim, dim).map(Some),
            };
            let empty = || Linear {
                weight: Matrix::from_rows(&[], 0, dim),
                bias: None,
            };
            let (q_proj, k_proj, v_proj, gate_proj, up_proj) = if latent.is_some() {
                (
                    empty(),
                    empty(),
                    empty(),
                    dense("gate_proj")?,
                    dense("up_proj")?,
                )
            } else if architecture.combined_projections {
                let mut qkv = Linear::load_split(
                    &st,
                    &format!("{attn}.qkv_proj"),
                    &[q_dim, kv_dim, kv_dim],
                    dim,
                    1.0,
                )?;
                let mut gate_up = Linear::load_split(
                    &st,
                    &format!("{p}.mlp.gate_up_proj"),
                    &[hidden_dim, hidden_dim],
                    dim,
                    1.0,
                )?;
                let (v, k, q) = (qkv.remove(2), qkv.remove(1), qkv.remove(0));
                let (up, gate) = (gate_up.remove(1), gate_up.remove(0));
                (q, k, v, Some(gate), Some(up))
            } else {
                // Layers with a mixture of experts have no MLP of their own.
                (
                    Linear::load(&st, &format!("{attn}.q_proj"), q_dim, dim)?,
                    Linear::load(&st, &format!("{attn}.k_proj"), kv_dim, dim)?,
                    Linear::load(&st, &format!("{attn}.v_proj"), kv_dim, dim)?,
                    dense("gate_proj")?,
                    dense("up_proj")?,
                )
            };
            // Each head's worth, or (OLMoE) a whole token's queries or keys.
            let head_norm = |name: &str, len: usize| -> Result<Vec<f32>> {
                let name = format!("{attn}.{name}.weight");
                match st.contains(&name) {
                    true => load_norm(&name, len),
                    // A checkpoint without them leaves them doing nothing but normalize.
                    false => Ok(vec![1.0; len]),
                }
            };
            let head_norms = match (architecture.head_norms, architecture.full_qk_norms) {
                (true, _) => Some((
                    head_norm("q_norm", head_dim)?,
                    head_norm("k_norm", head_dim)?,
                )),
                (_, true) => Some((head_norm("q_norm", q_dim)?, head_norm("k_norm", kv_dim)?)),
                _ => None,
            };
            let (norm_after_attn, mlp_norm, norm_after_mlp) = if architecture.norms_after_blocks {
                (
                    Some(load_norm(
                        &format!("{p}.post_attention_layernorm.weight"),
                        dim,
                    )?),
                    load_norm(&format!("{p}.pre_feedforward_layernorm.weight"), dim)?,
                    Some(load_norm(
                        &format!("{p}.post_feedforward_layernorm.weight"),
                        dim,
                    )?),
                )
            } else {
                // Elsewhere this name means the norm in front of the MLP.
                (
                    None,
                    load_norm(&format!("{p}.post_attention_layernorm.weight"), dim)?,
                    None,
                )
            };
            let sinks = match architecture.attention_sinks {
                true => Some(st.load(&format!("{attn}.sinks"), &[n_heads])?),
                false => None,
            };
            let latent = match latent {
                None => None,
                Some(l) => {
                    let q_width = n_heads * (l.nope + l.rope);
                    let q_low = match l.q_lora_rank {
                        Some(rank) => Some((
                            Linear::load(&st, &format!("{attn}.q_a_proj"), rank, dim)?,
                            load_norm(&format!("{attn}.q_a_layernorm.weight"), rank)?,
                        )),
                        None => None,
                    };
                    let q = match l.q_lora_rank {
                        Some(rank) => {
                            Linear::load(&st, &format!("{attn}.q_b_proj"), q_width, rank)?
                        }
                        None => Linear::load(&st, &format!("{attn}.q_proj"), q_width, dim)?,
                    };
                    let rank = l.kv_lora_rank;
                    Some(Box::new(Latent {
                        q_low,
                        q,
                        kv_a: Linear::load(
                            &st,
                            &format!("{attn}.kv_a_proj_with_mqa"),
                            rank + l.rope,
                            dim,
                        )?,
                        kv_a_norm: load_norm(&format!("{attn}.kv_a_layernorm.weight"), rank)?,
                        kv_b: Linear::load(
                            &st,
                            &format!("{attn}.kv_b_proj"),
                            n_heads * (l.nope + l.v),
                            rank,
                        )?,
                    }))
                }
            };
            Ok(Layer {
                sinks,
                latent,
                attn_norm: load_norm(&format!("{p}.input_layernorm.weight"), dim)?,
                q_proj,
                k_proj,
                v_proj,
                head_norms,
                o_proj: Linear::load_scaled(
                    &st,
                    &format!("{attn}.o_proj"),
                    dim,
                    n_heads * head_v,
                    residual,
                )?,
                norm_after_attn,
                mlp_norm,
                feed_forward: match (shape.sparse, &architecture.experts) {
                    (true, Some(spec)) => load_experts(&st, &p, spec, dim, residual)?,
                    _ => FeedForward::Dense(Mlp {
                        gate_proj: gate_proj.context("a dense layer needs gate_proj")?,
                        up_proj: up_proj.context("a dense layer needs up_proj")?,
                        down_proj: Linear::load_scaled(
                            &st,
                            &format!("{p}.mlp.down_proj"),
                            dim,
                            hidden_dim,
                            residual,
                        )?,
                    }),
                },
                norm_after_mlp,
                rope_table: shape.rope_table,
                window: shape.window,
            })
        };
        Pool::new(threads).par_for(config.num_hidden_layers, 1, |i| {
            let layer = load_layer(i);
            results.lock().unwrap()[i] = Some(layer);
        });
        let layers = results
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|layer| layer.expect("every layer was visited"))
            .collect::<Result<Vec<_>>>()?;

        let embed_tokens = Linear::load(&st, "model.embed_tokens", vocab_size, dim)?;
        let norm = load_norm("model.norm.weight", dim)?;
        let lm_head = match st.contains("lm_head.weight") {
            true => Some(Linear::load(&st, "lm_head", vocab_size, dim)?),
            false => None, // tied embeddings
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            architecture,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            head_k,
            head_v,
            rope_dim,
            hidden_dim,
            vocab_size,
            adapters: Vec::new(),
        })
    }

    /// Add a LoRA adapter, which rows then ask for by its number (the first is 1).
    pub fn add_adapter(&mut self, adapter: &LoraAdapter) -> Result<()> {
        anyhow::ensure!(
            self.architecture.latent.is_none(),
            "LoRA adapters for DeepSeek's latent attention aren't supported"
        );
        anyhow::ensure!(
            adapter.layers.len() == self.layers.len(),
            "{}: made for a model with {} layers, not {}",
            adapter.name,
            adapter.layers.len(),
            self.layers.len()
        );
        let residual = self.architecture.residual_scale;
        let layers = adapter
            .layers
            .iter()
            .map(|modules| {
                std::array::from_fn(|m| {
                    let pair = modules[m].as_ref()?;
                    let (in_dim, out_dim) = Module::ALL[m].shape(&self.config);
                    // The output projections' outputs are scaled, so their corrections are too.
                    let scale = match Module::ALL[m] {
                        Module::O | Module::Down => residual,
                        _ => 1.0,
                    };
                    let b: Vec<f32> = pair.b.iter().map(|v| v * scale).collect();
                    Some((
                        Matrix::from_rows(&pair.a, pair.rank, in_dim),
                        Matrix::from_rows(&b, out_dim, pair.rank),
                    ))
                })
            })
            .collect();
        self.adapters.push(Adapter { layers });
        Ok(())
    }

    /// Add each adapter's correction for `module` of `layer` to the rows of `out` that use it:
    /// `out[r] += B (A x[r])`, where `x` holds the module's inputs. Each row's correction is
    /// worked out the same way whatever else is in the batch.
    #[allow(clippy::too_many_arguments)]
    fn add_lora(
        &self,
        pool: &Pool,
        layer: usize,
        module: Module,
        rows: &[Row],
        x: &[f32],
        out: &mut [f32],
        lora: &mut LoraBuffers,
    ) {
        let (in_dim, out_dim) = module.shape(&self.config);
        for (index, adapter) in self.adapters.iter().enumerate() {
            let Some((a, b)) = &adapter.layers[layer][module as usize] else {
                continue;
            };
            let number = index as u16 + 1;
            lora.rows.clear();
            lora.rows.extend(
                (rows.iter().enumerate())
                    .filter(|(_, row)| row.adapter == number)
                    .map(|(r, _)| r),
            );
            if lora.rows.is_empty() {
                continue;
            }
            let n = lora.rows.len();
            lora.input.clear();
            for &r in &lora.rows {
                lora.input
                    .extend_from_slice(&x[r * in_dim..(r + 1) * in_dim]);
            }
            lora.reduced.resize(n * a.rows(), 0.0);
            matmul(pool, &mut lora.reduced, &mut lora.scratch, a, &lora.input);
            lora.delta.resize(n * out_dim, 0.0);
            matmul(pool, &mut lora.delta, &mut lora.scratch, b, &lora.reduced);
            for (j, &r) in lora.rows.iter().enumerate() {
                let delta = &lora.delta[j * out_dim..(j + 1) * out_dim];
                for (o, d) in out[r * out_dim..(r + 1) * out_dim].iter_mut().zip(delta) {
                    *o += d;
                }
            }
        }
    }

    pub fn num_params(&self) -> usize {
        let per_layer = |l: &Layer| {
            [&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj]
                .iter()
                .map(|lin| lin.num_params())
                .sum::<usize>()
                + l.latent.as_ref().map_or(0, |latent| {
                    latent.q.num_params()
                        + latent.kv_a.num_params()
                        + latent.kv_b.num_params()
                        + latent.kv_a_norm.len()
                        + latent
                            .q_low
                            .as_ref()
                            .map_or(0, |(a, norm)| a.num_params() + norm.len())
                })
                + l.feed_forward.num_params()
                + l.attn_norm.len()
                + l.mlp_norm.len()
                + l.head_norms.as_ref().map_or(0, |(q, k)| q.len() + k.len())
                + l.norm_after_attn.as_ref().map_or(0, Vec::len)
                + l.norm_after_mlp.as_ref().map_or(0, Vec::len)
        };
        self.embed_tokens.num_params()
            + self.layers.iter().map(per_layer).sum::<usize>()
            + self.norm.len()
            + self.lm_head.as_ref().map_or(0, Linear::num_params)
    }

    /// A mixture of experts: route each row of `ws.xb` to its experts, and leave the weighted
    /// sum of their outputs in `ws.xb`. Rows are grouped by expert, so each expert's weights are
    /// read once for all the rows it has. Each row's outputs are added up in order of expert,
    /// whatever else is in the batch.
    #[allow(clippy::too_many_arguments)]
    fn experts(
        &self,
        pool: &Pool,
        router: &Linear,
        correction: Option<&[f32]>,
        experts: &[Mlp],
        shared: Option<&(Mlp, Option<Linear>)>,
        xb: &mut [f32],
        moe: &mut MoeBuffers,
        batch: usize,
    ) {
        let dim = self.dim;
        let spec = self
            .architecture
            .experts
            .expect("sparse layers have experts");
        let count = spec.count;
        moe.scores.resize(batch * count, 0.0);
        router.forward(pool, &mut moe.scores, &mut moe.scratch, xb);
        let mut assigned: Vec<Vec<(usize, f32)>> = vec![Vec::new(); count];
        for r in 0..batch {
            let scores = &moe.scores[r * count..(r + 1) * count];
            for (e, weight) in route(scores, spec.top_k, spec.routing, correction) {
                assigned[e].push((r, weight));
            }
        }
        moe.sum.clear();
        moe.sum.resize(batch * dim, 0.0);
        for (e, rows) in assigned.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            moe.input.clear();
            for &(r, _) in rows {
                moe.input.extend_from_slice(&xb[r * dim..(r + 1) * dim]);
            }
            self.run_mlp(pool, &experts[e], spec.hidden, moe);
            for (i, &(r, weight)) in rows.iter().enumerate() {
                let out = &moe.output[i * dim..(i + 1) * dim];
                for (sum, &y) in moe.sum[r * dim..(r + 1) * dim].iter_mut().zip(out) {
                    *sum += weight * y;
                }
            }
        }
        if let Some((mlp, gate)) = shared {
            // Every row goes through this one, scaled by its own gate.
            moe.input.clear();
            moe.input.extend_from_slice(&xb[..batch * dim]);
            let hidden = spec.shared_hidden.expect("a shared expert has a width");
            self.run_mlp(pool, mlp, hidden, moe);
            moe.scores.resize(batch, 0.0);
            if let Some(gate) = gate {
                gate.forward(pool, &mut moe.scores[..batch], &mut moe.scratch, xb);
            }
            for r in 0..batch {
                let scale = match gate {
                    Some(_) => 1.0 / (1.0 + (-moe.scores[r]).exp()),
                    None => 1.0,
                };
                let out = &moe.output[r * dim..(r + 1) * dim];
                for (sum, &y) in moe.sum[r * dim..(r + 1) * dim].iter_mut().zip(out) {
                    *sum += scale * y;
                }
            }
        }
        xb[..batch * dim].copy_from_slice(&moe.sum);
    }

    /// Put `moe.input` through one expert's MLP, into `moe.output`.
    fn run_mlp(&self, pool: &Pool, mlp: &Mlp, hidden: usize, moe: &mut MoeBuffers) {
        let rows = moe.input.len() / self.dim;
        moe.gate.resize(rows * hidden, 0.0);
        moe.up.resize(rows * hidden, 0.0);
        moe.output.resize(rows * self.dim, 0.0);
        mlp.gate_proj
            .forward(pool, &mut moe.gate, &mut moe.scratch, &moe.input);
        mlp.up_proj
            .forward(pool, &mut moe.up, &mut moe.scratch, &moe.input);
        let glu = self.architecture.experts.and_then(|spec| spec.glu);
        match (glu, self.architecture.activation) {
            (Some(glu), _) => clamped_glu(pool, &mut moe.gate, &moe.up, hidden, glu),
            (None, Activation::Silu) => swiglu(pool, &mut moe.gate, &moe.up, hidden),
            (None, Activation::Gelu) => gated_gelu(pool, &mut moe.gate, &moe.up, hidden),
        }
        mlp.down_proj
            .forward(pool, &mut moe.output, &mut moe.scratch, &moe.gate);
    }

    /// DeepSeek's latent attention, for a batch whose normalized inputs are in `ws.xb`: queries
    /// into `ws.q`, keys into `ws.k` and values into `ws.v`, with their rotated parts rotated by
    /// the angles in `table` of the RoPE tables.
    #[allow(clippy::too_many_arguments)]
    fn latent_qkv(
        &self,
        pool: &Pool,
        latent: &Latent,
        xb: &[f32],
        (cos, sin): (&[f32], &[f32]),
        (q, k, v): (&mut [f32], &mut [f32], &mut [f32]),
        scratch: &mut Vec<f32>,
        b: &mut LatentBuffers,
        batch: usize,
    ) {
        let l = self
            .architecture
            .latent
            .expect("the model has latent attention");
        let (n, eps) = (self.n_heads, self.config.rms_norm_eps);
        let (rank, head_k) = (l.kv_lora_rank, l.nope + l.rope);
        let half = l.rope / 2;
        match &latent.q_low {
            Some((a, norm)) => {
                b.q_low.resize(batch * a.weight.rows(), 0.0);
                a.forward(pool, &mut b.q_low, scratch, xb);
                b.q_norm.resize(b.q_low.len(), 0.0);
                rmsnorm(pool, &mut b.q_norm, &b.q_low, norm, eps);
                latent.q.forward(pool, q, scratch, &b.q_norm);
            }
            None => latent.q.forward(pool, q, scratch, xb),
        }
        let width = rank + l.rope;
        b.kv_low.resize(batch * width, 0.0);
        latent.kv_a.forward(pool, &mut b.kv_low, scratch, xb);
        b.latent.clear();
        for r in 0..batch {
            b.latent
                .extend_from_slice(&b.kv_low[r * width..r * width + rank]);
        }
        b.normed.resize(batch * rank, 0.0);
        rmsnorm(pool, &mut b.normed, &b.latent, &latent.kv_a_norm, eps);
        b.up.resize(batch * n * (l.nope + l.v), 0.0);
        latent.kv_b.forward(pool, &mut b.up, scratch, &b.normed);
        for r in 0..batch {
            let (c, s) = (
                &cos[r * half..(r + 1) * half],
                &sin[r * half..(r + 1) * half],
            );
            // The keys' rotated part is the same for every head.
            b.rotated.clear();
            b.rotated
                .extend_from_slice(&b.kv_low[r * width + rank..(r + 1) * width]);
            rotate_pairs(&mut b.rotated, c, s);
            for h in 0..n {
                let up = &b.up[(r * n + h) * (l.nope + l.v)..][..l.nope + l.v];
                let key = &mut k[(r * n + h) * head_k..][..head_k];
                key[..l.nope].copy_from_slice(&up[..l.nope]);
                key[l.nope..].copy_from_slice(&b.rotated);
                v[(r * n + h) * l.v..][..l.v].copy_from_slice(&up[l.nope..]);
                rotate_pairs(&mut q[(r * n + h) * head_k + l.nope..][..l.rope], c, s);
            }
        }
    }

    /// Bytes of KV cache each token takes up.
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers.len() * self.n_kv_heads * (self.head_k + self.head_v) * size_of::<f32>()
    }

    /// A pool with room for the keys and values of `slots` tokens.
    pub fn new_kv_pool(&self, slots: usize) -> KvPool {
        KvPool::new(
            self.layers.len(),
            self.n_kv_heads * self.head_k,
            self.n_kv_heads * self.head_v,
            slots,
        )
    }

    /// Run a batch of tokens through the model, writing their keys and values into `kv`.
    ///
    /// Rows from one sequence must be in order and consecutive positions; a row can see every
    /// earlier token of its sequence but nothing after it. Logits land in `ws`.
    pub fn forward(&self, rows: &[Row], kv: &mut KvPool, ws: &mut Workspace) {
        let batch = rows.len();
        let (dim, head_dim, eps) = (self.dim, self.head_dim, self.config.rms_norm_eps);
        let (q_dim, kv_dim) = (self.n_heads * self.head_k, self.n_kv_heads * self.head_k);
        let v_dim = self.n_kv_heads * self.head_v;
        let out_dim = self.n_heads * self.head_v;
        let half = self.rope_dim / 2;
        let pool = &ws.pool;

        let tables = self.architecture.rope_tables.len();
        ws.x.resize(batch * dim, 0.0);
        ws.xb.resize(batch * dim, 0.0);
        ws.xc.resize(batch * dim, 0.0);
        ws.q.resize(batch * q_dim, 0.0);
        ws.k.resize(batch * kv_dim, 0.0);
        ws.v.resize(batch * v_dim, 0.0);
        ws.attn_out.resize(batch * out_dim, 0.0);
        ws.gate.resize(batch * self.hidden_dim, 0.0);
        ws.up.resize(batch * self.hidden_dim, 0.0);
        ws.cos.resize(tables * batch * half, 0.0);
        ws.sin.resize(tables * batch * half, 0.0);

        for (r, row) in rows.iter().enumerate() {
            let t = row.token as usize;
            assert!(
                t < self.vocab_size,
                "token id {} is outside the vocabulary",
                row.token
            );
            let embedding = &mut ws.x[r * dim..(r + 1) * dim];
            self.embed_tokens.weight.copy_row(t, embedding);
            if self.architecture.embedding_scale != 1.0 {
                let scale = self.architecture.embedding_scale;
                embedding.iter_mut().for_each(|v| *v *= scale);
            }
            let pos = (row.slots.len() - 1) as f32;
            for (table, frequencies) in self.architecture.rope_tables.iter().enumerate() {
                let scale = self.architecture.rope_scales[table];
                let start = (table * batch + r) * half;
                for (i, &freq) in frequencies.iter().enumerate() {
                    let angle = pos * freq;
                    ws.cos[start + i] = angle.cos() * scale;
                    ws.sin[start + i] = angle.sin() * scale;
                }
            }
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block: mix in information from earlier tokens.
            rmsnorm(pool, &mut ws.xb, &ws.x, &layer.attn_norm, eps);
            let table = layer.rope_table * batch * half..(layer.rope_table + 1) * batch * half;
            if let Some(latent) = &layer.latent {
                self.latent_qkv(
                    pool,
                    latent,
                    &ws.xb,
                    (&ws.cos[table.clone()], &ws.sin[table.clone()]),
                    (&mut ws.q, &mut ws.k, &mut ws.v),
                    &mut ws.scratch,
                    &mut ws.latent,
                    batch,
                );
            } else {
                layer
                    .q_proj
                    .forward(pool, &mut ws.q, &mut ws.scratch, &ws.xb);
                layer
                    .k_proj
                    .forward(pool, &mut ws.k, &mut ws.scratch, &ws.xb);
                layer
                    .v_proj
                    .forward(pool, &mut ws.v, &mut ws.scratch, &ws.xb);
                if !self.adapters.is_empty() {
                    for (module, out) in [
                        (Module::Q, &mut ws.q),
                        (Module::K, &mut ws.k),
                        (Module::V, &mut ws.v),
                    ] {
                        self.add_lora(pool, l, module, rows, &ws.xb, out, &mut ws.lora);
                    }
                }
                if let Some((q_norm, k_norm)) = &layer.head_norms {
                    rmsnorm_heads(pool, &mut ws.q, q_norm, eps);
                    rmsnorm_heads(pool, &mut ws.k, k_norm, eps);
                }
                let (cos, sin) = (&ws.cos[table.clone()], &ws.sin[table]);
                rope(pool, &mut ws.q, q_dim, head_dim, cos, sin);
                rope(pool, &mut ws.k, kv_dim, head_dim, cos, sin);
            }
            // Keys and values have to be in place before attention, because later rows of a
            // sequence attend to earlier rows in the same batch.
            for (r, row) in rows.iter().enumerate() {
                let slot = *row.slots.last().expect("a row needs a KV slot") as usize;
                kv.k[l][slot * kv_dim..(slot + 1) * kv_dim]
                    .copy_from_slice(&ws.k[r * kv_dim..(r + 1) * kv_dim]);
                kv.v[l][slot * v_dim..(slot + 1) * v_dim]
                    .copy_from_slice(&ws.v[r * v_dim..(r + 1) * v_dim]);
            }
            self.attention(
                pool,
                &mut ws.attn_out,
                &ws.q,
                rows,
                (&kv.k[l], &kv.v[l]),
                layer.window,
                layer.sinks.as_deref(),
            );
            layer
                .o_proj
                .forward(pool, &mut ws.xb, &mut ws.scratch, &ws.attn_out);
            if !self.adapters.is_empty() {
                let (attn_out, out) = (&ws.attn_out, &mut ws.xb);
                self.add_lora(pool, l, Module::O, rows, attn_out, out, &mut ws.lora);
            }
            let added = match &layer.norm_after_attn {
                Some(norm) => {
                    rmsnorm(pool, &mut ws.xc, &ws.xb, norm, eps);
                    &ws.xc
                }
                None => &ws.xb,
            };
            add_rows(pool, &mut ws.x, added, dim);

            // MLP block: process each token on its own.
            rmsnorm(pool, &mut ws.xb, &ws.x, &layer.mlp_norm, eps);
            match &layer.feed_forward {
                FeedForward::Dense(mlp) => {
                    mlp.gate_proj
                        .forward(pool, &mut ws.gate, &mut ws.scratch, &ws.xb);
                    mlp.up_proj
                        .forward(pool, &mut ws.up, &mut ws.scratch, &ws.xb);
                    if !self.adapters.is_empty() {
                        for (module, out) in
                            [(Module::Gate, &mut ws.gate), (Module::Up, &mut ws.up)]
                        {
                            self.add_lora(pool, l, module, rows, &ws.xb, out, &mut ws.lora);
                        }
                    }
                    match self.architecture.activation {
                        Activation::Silu => swiglu(pool, &mut ws.gate, &ws.up, self.hidden_dim),
                        Activation::Gelu => gated_gelu(pool, &mut ws.gate, &ws.up, self.hidden_dim),
                    }
                    mlp.down_proj
                        .forward(pool, &mut ws.xb, &mut ws.scratch, &ws.gate);
                    if !self.adapters.is_empty() {
                        let (gate, out) = (&ws.gate, &mut ws.xb);
                        self.add_lora(pool, l, Module::Down, rows, gate, out, &mut ws.lora);
                    }
                }
                FeedForward::Sparse {
                    router,
                    correction,
                    experts,
                    shared,
                } => {
                    let shared = shared.as_deref();
                    self.experts(
                        pool,
                        router,
                        correction.as_deref(),
                        experts,
                        shared,
                        &mut ws.xb,
                        &mut ws.moe,
                        batch,
                    )
                }
            }
            let added = match &layer.norm_after_mlp {
                Some(norm) => {
                    rmsnorm(pool, &mut ws.xc, &ws.xb, norm, eps);
                    &ws.xc
                }
                None => &ws.xb,
            };
            add_rows(pool, &mut ws.x, added, dim);
        }

        // Only some rows need the output projection, which is by far the largest matrix.
        let n_logits = rows.iter().filter(|r| r.want_logits).count();
        ws.vocab = self.vocab_size;
        ws.logits.resize(n_logits * self.vocab_size, 0.0);
        if n_logits > 0 {
            ws.final_x.clear();
            for (r, row) in rows.iter().enumerate() {
                if row.want_logits {
                    ws.final_x.extend_from_slice(&ws.x[r * dim..(r + 1) * dim]);
                }
            }
            ws.xb.resize(n_logits * dim, 0.0);
            rmsnorm(
                pool,
                &mut ws.xb[..n_logits * dim],
                &ws.final_x,
                &self.norm,
                eps,
            );
            let wanted = || rows.iter().filter(|r| r.want_logits);
            ws.dim = dim;
            if wanted().any(|r| r.sampling == LogitUse::Hidden) {
                ws.hidden.clear();
                ws.hidden.extend_from_slice(&ws.xb[..n_logits * dim]);
            }
            // A batch of nothing but embeddings needs no scores.
            if wanted().any(|r| r.sampling != LogitUse::Hidden) {
                let lm_head = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
                lm_head.forward(
                    pool,
                    &mut ws.logits,
                    &mut ws.scratch,
                    &ws.xb[..n_logits * dim],
                );
                self.finish_logits(pool, &mut ws.logits);
            }
        }
    }

    /// The last adjustments some models make to their scores.
    fn finish_logits(&self, pool: &Pool, logits: &mut [f32]) {
        let (divisor, cap) = (
            self.architecture.logit_divisor,
            self.architecture.logit_softcap,
        );
        if divisor == 1.0 && cap == 0.0 {
            return;
        }
        pool.for_chunks(logits, self.vocab_size, 1, |_, row| {
            for value in row {
                *value /= divisor;
                if cap > 0.0 {
                    *value = soft_cap(*value, cap);
                }
            }
        });
    }

    /// Causal self-attention for each row, with (row, head) pairs spread across threads.
    ///
    /// With grouped-query attention, `n_heads / n_kv_heads` query heads share each key/value
    /// head, which keeps the cache small.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        pool: &Pool,
        out: &mut [f32],
        q: &[f32],
        rows: &[Row],
        (k_pool, v_pool): (&[f32], &[f32]),
        window: Option<usize>,
        sinks: Option<&[f32]>,
    ) {
        let (n_heads, head_k, head_v) = (self.n_heads, self.head_k, self.head_v);
        let (k_dim, v_dim) = (self.n_kv_heads * head_k, self.n_kv_heads * head_v);
        let group = n_heads / self.n_kv_heads;
        let scale = self.architecture.attention_scale;
        let cap = self.architecture.attention_softcap;

        pool.for_chunks(out, head_v, 2, |i, out| {
            let (row, head) = (i / n_heads, i % n_heads);
            let slots = rows[row].slots;
            let slots = &slots[slots.len().saturating_sub(window.unwrap_or(usize::MAX))..];
            let q = &q[i * head_k..(i + 1) * head_k];
            let kv_head = head / group;
            with_scratch(slots.len(), |scores| {
                // How relevant is each earlier token to this one?
                for (score, &slot) in scores.iter_mut().zip(slots) {
                    let k = &k_pool[slot as usize * k_dim + kv_head * head_k..][..head_k];
                    *score = dot(q, k) * scale;
                    if cap > 0.0 {
                        *score = soft_cap(*score, cap);
                    }
                }
                match sinks {
                    None => softmax(scores),
                    Some(sinks) => {
                        // The sink joins the softmax, and takes its share of the weight.
                        let sink = sinks[head];
                        let max = scores.iter().copied().fold(sink, f32::max);
                        let mut total = (sink - max).exp();
                        for score in scores.iter_mut() {
                            *score = (*score - max).exp();
                            total += *score;
                        }
                        scores.iter_mut().for_each(|score| *score /= total);
                    }
                }
                // Blend their values by that relevance.
                out.fill(0.0);
                for (&weight, &slot) in scores.iter().zip(slots) {
                    let v = &v_pool[slot as usize * v_dim + kv_head * head_v..][..head_v];
                    add_scaled(out, weight, v);
                }
            });
        });
    }
}
