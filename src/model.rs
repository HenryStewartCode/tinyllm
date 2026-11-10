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
use crate::config::{Activation, Architecture, Config};
use crate::ops::{
    Matrix, add_bias, add_rows, add_scaled, dot, gated_gelu, matmul, rmsnorm, rmsnorm_heads, rope,
    soft_cap, softmax, swiglu, with_scratch,
};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::Result;
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
}

struct Layer {
    attn_norm: Vec<f32>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    /// Normalize each head of the queries and of the keys, if the model does that.
    head_norms: Option<(Vec<f32>, Vec<f32>)>,
    o_proj: Linear,
    /// Normalizes what attention adds to the residual stream, if the model does that.
    norm_after_attn: Option<Vec<f32>>,
    mlp_norm: Vec<f32>,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
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
    hidden_dim: usize,
    vocab_size: usize,
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
        let n_kv_heads = config.n_kv_heads();
        let head_dim = config.head_dim();
        let hidden_dim = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        let architecture = config.architecture()?;

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
            let (q_proj, k_proj, v_proj, gate_proj, up_proj) = if architecture.combined_projections
            {
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
                (q, k, v, gate, up)
            } else {
                (
                    Linear::load(&st, &format!("{attn}.q_proj"), q_dim, dim)?,
                    Linear::load(&st, &format!("{attn}.k_proj"), kv_dim, dim)?,
                    Linear::load(&st, &format!("{attn}.v_proj"), kv_dim, dim)?,
                    Linear::load(&st, &format!("{p}.mlp.gate_proj"), hidden_dim, dim)?,
                    Linear::load(&st, &format!("{p}.mlp.up_proj"), hidden_dim, dim)?,
                )
            };
            let head_norm = |name: &str| -> Result<Vec<f32>> {
                let name = format!("{attn}.{name}.weight");
                match st.contains(&name) {
                    true => load_norm(&name, head_dim),
                    // A checkpoint without them leaves them doing nothing but normalize.
                    false => Ok(vec![1.0; head_dim]),
                }
            };
            let head_norms = match architecture.head_norms {
                true => Some((head_norm("q_norm")?, head_norm("k_norm")?)),
                false => None,
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
            Ok(Layer {
                attn_norm: load_norm(&format!("{p}.input_layernorm.weight"), dim)?,
                q_proj,
                k_proj,
                v_proj,
                head_norms,
                o_proj: Linear::load_scaled(&st, &format!("{attn}.o_proj"), dim, q_dim, residual)?,
                norm_after_attn,
                mlp_norm,
                gate_proj,
                up_proj,
                down_proj: Linear::load_scaled(
                    &st,
                    &format!("{p}.mlp.down_proj"),
                    dim,
                    hidden_dim,
                    residual,
                )?,
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
            hidden_dim,
            vocab_size,
        })
    }

    pub fn num_params(&self) -> usize {
        let per_layer = |l: &Layer| {
            [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ]
            .iter()
            .map(|lin| lin.num_params())
            .sum::<usize>()
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

    /// Bytes of KV cache each token takes up.
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.layers.len() * self.n_kv_heads * self.head_dim * size_of::<f32>()
    }

    /// A pool with room for the keys and values of `slots` tokens.
    pub fn new_kv_pool(&self, slots: usize) -> KvPool {
        KvPool::new(self.layers.len(), self.n_kv_heads * self.head_dim, slots)
    }

    /// Run a batch of tokens through the model, writing their keys and values into `kv`.
    ///
    /// Rows from one sequence must be in order and consecutive positions; a row can see every
    /// earlier token of its sequence but nothing after it. Logits land in `ws`.
    pub fn forward(&self, rows: &[Row], kv: &mut KvPool, ws: &mut Workspace) {
        let batch = rows.len();
        let (dim, head_dim, eps) = (self.dim, self.head_dim, self.config.rms_norm_eps);
        let (q_dim, kv_dim) = (self.n_heads * head_dim, self.n_kv_heads * head_dim);
        let half = head_dim / 2;
        let pool = &ws.pool;

        let tables = self.architecture.rope_tables.len();
        ws.x.resize(batch * dim, 0.0);
        ws.xb.resize(batch * dim, 0.0);
        ws.xc.resize(batch * dim, 0.0);
        ws.q.resize(batch * q_dim, 0.0);
        ws.k.resize(batch * kv_dim, 0.0);
        ws.v.resize(batch * kv_dim, 0.0);
        ws.attn_out.resize(batch * q_dim, 0.0);
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
                let start = (table * batch + r) * half;
                for (i, &freq) in frequencies.iter().enumerate() {
                    let angle = pos * freq;
                    ws.cos[start + i] = angle.cos();
                    ws.sin[start + i] = angle.sin();
                }
            }
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block: mix in information from earlier tokens.
            rmsnorm(pool, &mut ws.xb, &ws.x, &layer.attn_norm, eps);
            layer
                .q_proj
                .forward(pool, &mut ws.q, &mut ws.scratch, &ws.xb);
            layer
                .k_proj
                .forward(pool, &mut ws.k, &mut ws.scratch, &ws.xb);
            layer
                .v_proj
                .forward(pool, &mut ws.v, &mut ws.scratch, &ws.xb);
            if let Some((q_norm, k_norm)) = &layer.head_norms {
                rmsnorm_heads(pool, &mut ws.q, q_norm, eps);
                rmsnorm_heads(pool, &mut ws.k, k_norm, eps);
            }
            let table = layer.rope_table * batch * half..(layer.rope_table + 1) * batch * half;
            let (cos, sin) = (&ws.cos[table.clone()], &ws.sin[table]);
            rope(pool, &mut ws.q, q_dim, head_dim, cos, sin);
            rope(pool, &mut ws.k, kv_dim, head_dim, cos, sin);
            // Keys and values have to be in place before attention, because later rows of a
            // sequence attend to earlier rows in the same batch.
            for (r, row) in rows.iter().enumerate() {
                let slot = *row.slots.last().expect("a row needs a KV slot") as usize;
                kv.k[l][slot * kv_dim..(slot + 1) * kv_dim]
                    .copy_from_slice(&ws.k[r * kv_dim..(r + 1) * kv_dim]);
                kv.v[l][slot * kv_dim..(slot + 1) * kv_dim]
                    .copy_from_slice(&ws.v[r * kv_dim..(r + 1) * kv_dim]);
            }
            self.attention(
                pool,
                &mut ws.attn_out,
                &ws.q,
                rows,
                (&kv.k[l], &kv.v[l]),
                layer.window,
            );
            layer
                .o_proj
                .forward(pool, &mut ws.xb, &mut ws.scratch, &ws.attn_out);
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
            layer
                .gate_proj
                .forward(pool, &mut ws.gate, &mut ws.scratch, &ws.xb);
            layer
                .up_proj
                .forward(pool, &mut ws.up, &mut ws.scratch, &ws.xb);
            match self.architecture.activation {
                Activation::Silu => swiglu(pool, &mut ws.gate, &ws.up, self.hidden_dim),
                Activation::Gelu => gated_gelu(pool, &mut ws.gate, &ws.up, self.hidden_dim),
            }
            layer
                .down_proj
                .forward(pool, &mut ws.xb, &mut ws.scratch, &ws.gate);
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
    fn attention(
        &self,
        pool: &Pool,
        out: &mut [f32],
        q: &[f32],
        rows: &[Row],
        (k_pool, v_pool): (&[f32], &[f32]),
        window: Option<usize>,
    ) {
        let (n_heads, head_dim) = (self.n_heads, self.head_dim);
        let kv_dim = self.n_kv_heads * head_dim;
        let group = n_heads / self.n_kv_heads;
        let scale = self.architecture.attention_scale;
        let cap = self.architecture.attention_softcap;

        pool.for_chunks(out, head_dim, 2, |i, out| {
            let (row, head) = (i / n_heads, i % n_heads);
            let slots = rows[row].slots;
            let slots = &slots[slots.len().saturating_sub(window.unwrap_or(usize::MAX))..];
            let q = &q[i * head_dim..(i + 1) * head_dim];
            let kv_head = (head / group) * head_dim;
            with_scratch(slots.len(), |scores| {
                // How relevant is each earlier token to this one?
                for (score, &slot) in scores.iter_mut().zip(slots) {
                    let k = &k_pool[slot as usize * kv_dim + kv_head..][..head_dim];
                    *score = dot(q, k) * scale;
                    if cap > 0.0 {
                        *score = soft_cap(*score, cap);
                    }
                }
                softmax(scores);
                // Blend their values by that relevance.
                out.fill(0.0);
                for (&weight, &slot) in scores.iter().zip(slots) {
                    let v = &v_pool[slot as usize * kv_dim + kv_head..][..head_dim];
                    add_scaled(out, weight, v);
                }
            });
        });
    }
}
