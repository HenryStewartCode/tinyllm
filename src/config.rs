//! Model hyper-parameters, read from a Hugging Face `config.json`, and what they mean for the
//! layout of the model.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::path::Path;

/// `model_type` values whose checkpoints use a layout implemented in [`crate::model`].
const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "smollm3",
    "phi3",
    "granite",
    "gemma",
    "gemma2",
    "gemma3_text",
    "qwen3_moe",
    "qwen2_moe",
    "mixtral",
    "olmoe",
    "granitemoe",
    "gpt_oss",
];

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: String,
    /// Width of the residual stream ("dim").
    pub hidden_size: usize,
    /// Width of the MLP's hidden layer.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Fewer key/value heads than query heads means grouped-query attention.
    pub num_key_value_heads: Option<usize>,
    /// Usually `hidden_size / num_attention_heads`, but some models set it explicitly.
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// `hidden_activation` in Gemma 2 and 3.
    #[serde(alias = "hidden_activation")]
    pub hidden_act: Option<String>,
    pub eos_token_id: Option<TokenIds>,

    // Position embeddings. Newer configs put everything in `rope_parameters`, per kind of layer
    // if the kinds differ; older ones use the separate fields.
    rope_theta: Option<f64>,
    rope_local_base_freq: Option<f64>,
    rope_scaling: Option<RopeScaling>,
    rope_parameters: Option<Value>,
    partial_rotary_factor: Option<f64>,
    /// 0 for each layer without position embeddings, 1 for the others. SmolLM3.
    no_rope_layers: Option<Vec<u8>>,
    no_rope_layer_interval: Option<usize>,

    // Sliding windows: how many tokens back a layer can see, for the layers that have a limit.
    layer_types: Option<Vec<String>>,
    #[serde(default, deserialize_with = "present")]
    sliding_window: Option<Option<usize>>,
    sliding_window_pattern: Option<usize>,
    use_sliding_window: Option<bool>,
    max_window_layers: Option<usize>,
    use_bidirectional_attention: Option<bool>,

    // Gemma.
    query_pre_attn_scalar: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    attn_logit_softcapping: Option<Option<f32>>,
    #[serde(default, deserialize_with = "present")]
    final_logit_softcapping: Option<Option<f32>>,

    /// How the checkpoint's weights are quantized, if they are.
    quantization_config: Option<Value>,

    // Mixture of experts.
    #[serde(alias = "num_local_experts")]
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    /// The width of each expert's hidden layer, where it isn't `intermediate_size`.
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    norm_topk_prob: Option<bool>,
    decoder_sparse_step: Option<usize>,
    mlp_only_layers: Option<Vec<usize>>,
    clip_qkv: Option<f32>,
    // GPT-OSS's activation.
    swiglu_alpha: Option<f32>,
    swiglu_limit: Option<f32>,

    // Granite.
    attention_multiplier: Option<f32>,
    logits_scaling: Option<f32>,
    residual_multiplier: Option<f32>,
    embedding_multiplier: Option<f32>,
}

/// Tells a field that was left out (`None`) from one set to null (`Some(None)`), for settings
/// whose default isn't null.
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Adjustments to the RoPE frequencies that some models use to stretch their context window.
#[derive(Debug, Clone, Deserialize)]
struct RopeScaling {
    rope_type: Option<String>,
    /// Older configs call `rope_type` just `type`.
    #[serde(rename = "type")]
    legacy_type: Option<String>,
    rope_theta: Option<f64>,
    factor: Option<f64>,
    low_freq_factor: Option<f64>,
    high_freq_factor: Option<f64>,
    original_max_position_embeddings: Option<f64>,
    partial_rotary_factor: Option<f64>,
    // YaRN's settings.
    beta_fast: Option<f64>,
    beta_slow: Option<f64>,
    mscale: Option<f64>,
    mscale_all_dim: Option<f64>,
    attention_factor: Option<f64>,
    truncate: Option<bool>,
}

impl RopeScaling {
    fn kind(&self) -> &str {
        self.rope_type
            .as_deref()
            .or(self.legacy_type.as_deref())
            .unwrap_or("default")
    }
}

/// Token id fields may hold a single id or a list of ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            TokenIds::One(id) => vec![*id],
            TokenIds::Many(ids) => ids.clone(),
        }
    }
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_max_position_embeddings() -> usize {
    2048
}

/// The nonlinearity in the MLP. Either way it's gated: `down(act(gate(x)) * up(x))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    /// Gaussian error linear units, in the form that uses `tanh`.
    Gelu,
}

/// How one layer differs from the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerShape {
    /// Which of [`Architecture::rope_tables`] it uses.
    pub rope_table: usize,
    /// How many tokens it can see, counting the current one. `None` for no limit.
    pub window: Option<usize>,
    /// Its MLP is a mixture of experts (see [`Architecture::experts`]).
    pub sparse: bool,
}

/// Where a family keeps its experts' weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertNaming {
    /// `mlp.gate`, `mlp.experts.N.{gate,up,down}_proj` (Qwen MoE, OLMoE).
    Qwen,
    /// `block_sparse_moe.gate`, `block_sparse_moe.experts.N.{w1,w3,w2}` (Mixtral).
    Mixtral,
    /// `block_sparse_moe.router.layer`, and every expert in one `input_linear` and one
    /// `output_linear` (Granite MoE).
    Granite,
    /// `mlp.router`, and every expert's gate and up projections in one `gate_up_proj` (their
    /// rows alternating) and down projections in one `down_proj`, all with biases, and either
    /// as plain matrices `[experts, in, out]` or packed in MXFP4 (GPT-OSS).
    GptOss,
}

/// GPT-OSS's activation: with `g = min(gate, limit)` and `u` the up projection clamped to
/// `[-limit, limit]`, `(u + 1) * g * sigmoid(alpha * g)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClampedGlu {
    pub alpha: f32,
    pub limit: f32,
}

/// A mixture-of-experts MLP: a router scores the experts for each token, and the token goes
/// through its best few, whose outputs are added up weighted by their scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    pub count: usize,
    /// How many each token goes through.
    pub top_k: usize,
    /// The width of each expert's hidden layer.
    pub hidden: usize,
    /// The chosen experts' weights are scaled to add up to 1.
    pub renormalize: bool,
    /// The width of an expert every token goes through, gated by its own score (Qwen2 MoE).
    pub shared_hidden: Option<usize>,
    pub naming: ExpertNaming,
    /// The experts' activation, if it isn't the model's usual one.
    pub glu: Option<ClampedGlu>,
}

/// What makes a family of models different from the plain Llama layout. Both backends build
/// the model from this.
#[derive(Clone, Debug)]
pub struct Architecture {
    pub activation: Activation,
    /// What embeddings get multiplied by, when the weights are used as 32-bit numbers.
    pub embedding_scale: f32,
    /// The same when they're 16-bit, for models whose scale is rounded along with them.
    pub embedding_scale_bf16: f32,
    /// Added to every RMSNorm weight. Gemma's weights are stored minus 1.
    pub norm_offset: f32,
    /// What attention scores get multiplied by.
    pub attention_scale: f32,
    /// If not 0, attention scores `s` become `cap * tanh(s / cap)`.
    pub attention_softcap: f32,
    /// If not 0, logits `l` become `cap * tanh(l / cap)`.
    pub logit_softcap: f32,
    /// What logits get divided by.
    pub logit_divisor: f32,
    /// What each block's output is multiplied by before it joins the residual stream.
    pub residual_scale: f32,
    /// Queries, keys and values come from one `qkv_proj` matrix, and gate and up from one
    /// `gate_up_proj`, instead of from separate ones.
    pub combined_projections: bool,
    /// Gemma 2 and 3 normalize what attention and the MLP each add to the residual stream.
    pub norms_after_blocks: bool,
    /// Qwen3 and Gemma 3 normalize each head of the queries and keys.
    pub head_norms: bool,
    /// OLMoE normalizes all the queries, and all the keys, of a token together.
    pub full_qk_norms: bool,
    /// Each head has a learned score (`self_attn.sinks`) that joins its attention scores in the
    /// softmax without adding any value, so attention can go nowhere (GPT-OSS).
    pub attention_sinks: bool,
    /// The mixture of experts in the layers marked `sparse`, if the model has one.
    pub experts: Option<Experts>,
    /// The RoPE speed of each element pair in a head: one list per kind of layer.
    pub rope_tables: Vec<Vec<f32>>,
    /// What each table's rotations (its cosines and sines) are multiplied by: 1, except for
    /// YaRN, which scales queries and keys up a little along with the context.
    pub rope_scales: Vec<f32>,
    pub layers: Vec<LayerShape>,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let config: Config =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if !SUPPORTED_MODEL_TYPES.contains(&self.model_type.as_str()) {
            bail!(
                "model_type '{}' is not supported (supported: {})",
                self.model_type,
                SUPPORTED_MODEL_TYPES.join(", ")
            );
        }
        ensure!(
            self.num_attention_heads.is_multiple_of(self.n_kv_heads()),
            "num_attention_heads must be a multiple of num_key_value_heads"
        );
        ensure!(
            self.head_dim().is_multiple_of(2),
            "head_dim must be even for RoPE"
        );
        ensure!(
            !self.use_bidirectional_attention.unwrap_or(false),
            "bidirectional attention is not supported"
        );
        ensure!(
            self.partial_rotary_factor.is_none_or(|f| f == 1.0),
            "a partial_rotary_factor other than 1 is not supported"
        );
        ensure!(self.clip_qkv.is_none(), "clip_qkv is not supported");
        if let Some(quantization) = &self.quantization_config {
            match quantization["quant_method"].as_str() {
                Some("fp8") => {
                    let block = &quantization["weight_block_size"];
                    ensure!(
                        block.is_null() || *block == serde_json::json!([128, 128]),
                        "FP8 checkpoints need 128x128 weight blocks, not {block}"
                    );
                }
                Some("awq" | "gptq") => {
                    crate::safetensors::Int4Format::from_config(quantization)?;
                }
                // GPT-OSS's experts, which the loaders recognize by their tensors.
                Some("mxfp4") => {}
                method => bail!(
                    "{} quantization isn't supported yet (FP8 checkpoints with 128x128 blocks, and \
                     4-bit AWQ and GPTQ ones, are)",
                    method.unwrap_or("this kind of")
                ),
            }
        }
        self.architecture()?;
        Ok(())
    }

    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn is_gemma(&self) -> bool {
        matches!(self.model_type.as_str(), "gemma" | "gemma2" | "gemma3_text")
    }

    /// Work out how the model is put together.
    pub fn architecture(&self) -> Result<Architecture> {
        let model_type = self.model_type.as_str();
        let gemma = self.is_gemma();

        let activation = match self.hidden_act.as_deref() {
            None if gemma => Activation::Gelu,
            None | Some("silu") => Activation::Silu,
            Some("gelu_pytorch_tanh") => Activation::Gelu,
            // Gemma's early configs said this, but meant the form above.
            Some("gelu") if gemma => Activation::Gelu,
            Some(other) => bail!("activation '{other}' is not supported"),
        };

        // Gemma keeps this scale as a number of the same type as its weights, so at 16 bits it
        // gets rounded. That's how the models were trained.
        let (embedding_scale, embedding_scale_bf16) = match model_type {
            "gemma" | "gemma2" | "gemma3_text" => {
                let scale = (self.hidden_size as f32).sqrt();
                (scale, round_to_bf16(scale))
            }
            _ => {
                let scale = self.embedding_multiplier.unwrap_or(1.0);
                (scale, scale)
            }
        };

        let attention_scale = match model_type {
            "gemma2" | "gemma3_text" => self.query_pre_attn_scalar.unwrap_or(256.0).powf(-0.5),
            "granite" | "granitemoe" => self.attention_multiplier.unwrap_or(1.0),
            _ => (self.head_dim() as f32).powf(-0.5),
        };
        // Gemma 3 has the setting, but its attention doesn't use it.
        let attention_softcap = match model_type {
            "gemma2" => self.attn_logit_softcapping.unwrap_or(Some(50.0)),
            _ => None,
        };
        let logit_softcap = match model_type {
            "gemma2" => self.final_logit_softcapping.unwrap_or(Some(30.0)),
            "gemma3_text" => self.final_logit_softcapping.flatten(),
            _ => None,
        };

        let mut rope_tables: Vec<Vec<f32>> = Vec::new();
        let mut rope_scales: Vec<f32> = Vec::new();
        let mut layers = Vec::with_capacity(self.num_hidden_layers);
        for l in 0..self.num_hidden_layers {
            let kind = self.layer_type(l);
            let (table, scale) = match self.has_rope(l) {
                true => self.rope_frequencies(&kind)?,
                false => (vec![0.0; self.head_dim() / 2], 1.0),
            };
            let known = (rope_tables.iter().zip(&rope_scales))
                .position(|(known, &known_scale)| *known == table && known_scale == scale);
            let rope_table = match known {
                Some(index) => index,
                None => {
                    rope_tables.push(table);
                    rope_scales.push(scale);
                    rope_tables.len() - 1
                }
            };
            let window = match kind.as_str() {
                "sliding_attention" => self.window_size(),
                _ => None,
            };
            layers.push(LayerShape {
                rope_table,
                window,
                sparse: self.is_sparse(l),
            });
        }

        Ok(Architecture {
            activation,
            embedding_scale,
            embedding_scale_bf16,
            norm_offset: if gemma { 1.0 } else { 0.0 },
            attention_scale,
            attention_softcap: attention_softcap.unwrap_or(0.0),
            logit_softcap: logit_softcap.unwrap_or(0.0),
            logit_divisor: self.logits_scaling.unwrap_or(1.0),
            residual_scale: self.residual_multiplier.unwrap_or(1.0),
            combined_projections: model_type == "phi3",
            norms_after_blocks: matches!(model_type, "gemma2" | "gemma3_text"),
            head_norms: matches!(model_type, "qwen3" | "gemma3_text" | "qwen3_moe"),
            full_qk_norms: model_type == "olmoe",
            attention_sinks: model_type == "gpt_oss",
            experts: self.experts()?,
            rope_tables,
            rope_scales,
            layers,
        })
    }

    /// The model's mixture of experts, if it has one.
    fn experts(&self) -> Result<Option<Experts>> {
        let model_type = self.model_type.as_str();
        let naming = match model_type {
            "qwen3_moe" | "qwen2_moe" | "olmoe" => ExpertNaming::Qwen,
            "mixtral" => ExpertNaming::Mixtral,
            "granitemoe" => ExpertNaming::Granite,
            "gpt_oss" => ExpertNaming::GptOss,
            _ => return Ok(None),
        };
        let count = self
            .num_experts
            .context("a mixture of experts needs num_experts")?;
        let top_k = self.num_experts_per_tok.unwrap_or(2);
        ensure!(
            (1..=count).contains(&top_k),
            "num_experts_per_tok must be between 1 and num_experts"
        );
        Ok(Some(Experts {
            count,
            top_k,
            hidden: match model_type {
                "qwen3_moe" | "qwen2_moe" => self
                    .moe_intermediate_size
                    .context("moe_intermediate_size is missing")?,
                _ => self.intermediate_size,
            },
            // Mixtral, Granite and GPT-OSS take the softmax of just the chosen experts' scores,
            // which comes to the same thing.
            renormalize: match model_type {
                "mixtral" | "granitemoe" | "gpt_oss" => true,
                _ => self.norm_topk_prob.unwrap_or(false),
            },
            shared_hidden: match model_type {
                "qwen2_moe" => self.shared_expert_intermediate_size,
                _ => None,
            },
            naming,
            glu: (model_type == "gpt_oss").then(|| ClampedGlu {
                alpha: self.swiglu_alpha.unwrap_or(1.702),
                limit: self.swiglu_limit.unwrap_or(7.0),
            }),
        }))
    }

    /// Whether layer `l` has a mixture of experts rather than a plain MLP.
    fn is_sparse(&self, l: usize) -> bool {
        match self.model_type.as_str() {
            "qwen3_moe" | "qwen2_moe" => {
                let step = self.decoder_sparse_step.unwrap_or(1).max(1);
                let dense = self
                    .mlp_only_layers
                    .as_ref()
                    .is_some_and(|d| d.contains(&l));
                !dense && (l + 1).is_multiple_of(step)
            }
            "mixtral" | "olmoe" | "granitemoe" | "gpt_oss" => true,
            _ => false,
        }
    }

    /// `full_attention` or `sliding_attention`.
    fn layer_type(&self, l: usize) -> String {
        if let Some(kind) = self.layer_types.as_ref().and_then(|kinds| kinds.get(l)) {
            return kind.clone();
        }
        let sliding = match self.model_type.as_str() {
            // These have no kinds of layer: a window, if there is one, is for all of them.
            "mistral" | "phi3" => self.window_size().is_some(),
            "qwen2" | "qwen3" => {
                self.use_sliding_window.unwrap_or(false)
                    && l >= self.max_window_layers.unwrap_or(28)
            }
            "smollm3" => self.use_sliding_window.unwrap_or(false) && !self.has_rope(l),
            "gemma3_text" => !(l + 1).is_multiple_of(self.sliding_window_pattern.unwrap_or(6)),
            // Alternating, starting with a window.
            "gpt_oss" => l.is_multiple_of(2),
            _ => false,
        };
        match sliding {
            true => "sliding_attention".to_string(),
            false => "full_attention".to_string(),
        }
    }

    fn window_size(&self) -> Option<usize> {
        match (self.model_type.as_str(), self.sliding_window) {
            // Mistral and Qwen and Gemma 3 have a default, where others have none.
            ("mistral" | "qwen2" | "qwen3" | "smollm3" | "gemma3_text", None) => Some(4096),
            ("gpt_oss", None) => Some(128),
            (_, setting) => setting.flatten(),
        }
    }

    fn has_rope(&self, l: usize) -> bool {
        if let Some(flags) = &self.no_rope_layers {
            return flags.get(l).is_none_or(|&flag| flag != 0);
        }
        match (self.model_type.as_str(), self.no_rope_layer_interval) {
            ("smollm3", interval) => !(l + 1).is_multiple_of(interval.unwrap_or(4)),
            _ => true,
        }
    }

    /// RoPE frequencies for a kind of layer: pair `i` of each head rotates by
    /// `pos * theta^(-2i / head_dim)` radians, optionally adjusted by the config's scaling.
    /// The RoPE speed of each element pair in a head, for layers of `kind`, and what their
    /// rotations are multiplied by.
    fn rope_frequencies(&self, kind: &str) -> Result<(Vec<f32>, f32)> {
        let default_theta = match (self.model_type.as_str(), kind) {
            ("gemma3_text", "full_attention") => 1_000_000.0,
            _ => 10_000.0,
        };
        let legacy_theta = match (self.model_type.as_str(), kind) {
            ("gemma3_text", "sliding_attention") => self.rope_local_base_freq,
            _ => self.rope_theta,
        };
        // Newer configs keep it all together, per kind of layer or for every layer alike.
        let parameters = self
            .rope_parameters
            .as_ref()
            .map(|parameters| parameters.get(kind).unwrap_or(parameters))
            .filter(|parameters| parameters.is_object())
            .map(|parameters| serde_json::from_value::<RopeScaling>(parameters.clone()))
            .transpose()
            .context("couldn't read rope_parameters")?;
        let scaling = parameters.as_ref().or(self.rope_scaling.as_ref());
        ensure!(
            scaling
                .and_then(|s| s.partial_rotary_factor)
                .is_none_or(|f| f == 1.0),
            "a partial_rotary_factor other than 1 is not supported"
        );
        let theta = scaling
            .and_then(|s| s.rope_theta)
            .or(legacy_theta)
            .unwrap_or(default_theta);

        let head_dim = self.head_dim();
        let mut inv_freq: Vec<f64> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();
        let mut scale = 1.0;
        if let Some(scaling) = scaling {
            let factor = scaling.factor.unwrap_or(1.0);
            match scaling.kind() {
                "default" => {}
                "linear" => inv_freq.iter_mut().for_each(|f| *f /= factor),
                "llama3" => {
                    // Llama 3.1+: slow down the low-frequency pairs, leave the high-frequency
                    // ones, and blend in between.
                    let low = scaling.low_freq_factor.unwrap_or(1.0);
                    let high = scaling.high_freq_factor.unwrap_or(4.0);
                    let original_ctx = scaling.original_max_position_embeddings.unwrap_or(8192.0);
                    for f in &mut inv_freq {
                        let wavelen = 2.0 * std::f64::consts::PI / *f;
                        if wavelen > original_ctx / low {
                            *f /= factor;
                        } else if wavelen >= original_ctx / high {
                            let smooth = (original_ctx / wavelen - low) / (high - low);
                            *f = (1.0 - smooth) * *f / factor + smooth * *f;
                        }
                    }
                }
                "yarn" => {
                    // YaRN (as transformers computes it): the fastest-turning pairs keep their
                    // speed, the slowest are slowed down by `factor`, with a ramp between, and
                    // queries and keys are scaled up a little to keep attention as sharp.
                    let original = scaling
                        .original_max_position_embeddings
                        .unwrap_or(self.max_position_embeddings as f64);
                    let factor = scaling
                        .factor
                        .unwrap_or(self.max_position_embeddings as f64 / original);
                    let mscale = |scale: f64, m: f64| match scale <= 1.0 {
                        true => 1.0,
                        false => 0.1 * m * scale.ln() + 1.0,
                    };
                    scale = match (
                        scaling.attention_factor,
                        scaling.mscale,
                        scaling.mscale_all_dim,
                    ) {
                        (Some(attention), _, _) => attention,
                        (None, Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                            mscale(factor, m) / mscale(factor, all)
                        }
                        _ => mscale(factor, 1.0),
                    };
                    let beta_fast = scaling.beta_fast.filter(|&b| b != 0.0).unwrap_or(32.0);
                    let beta_slow = scaling.beta_slow.filter(|&b| b != 0.0).unwrap_or(1.0);
                    let dim = head_dim as f64;
                    let correction = |rotations: f64| {
                        dim * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
                            / (2.0 * theta.ln())
                    };
                    let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
                    if scaling.truncate.unwrap_or(true) {
                        (low, high) = (low.floor(), high.ceil());
                    }
                    let (low, mut high) = (low.max(0.0), high.min(dim - 1.0));
                    if low == high {
                        high += 0.001;
                    }
                    for (i, f) in inv_freq.iter_mut().enumerate() {
                        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                        let keep = 1.0 - ramp;
                        *f = (*f / factor) * (1.0 - keep) + *f * keep;
                    }
                }
                other => bail!("rope_scaling type '{other}' is not supported"),
            }
        }
        Ok((
            inv_freq.into_iter().map(|f| f as f32).collect(),
            scale as f32,
        ))
    }
}

/// The nearest bfloat16 number, which has only 8 bits of precision.
fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(extra: Value) -> Config {
        let mut value = json!({
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "vocab_size": 100,
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let config: Config = serde_json::from_value(value).unwrap();
        config.validate().unwrap();
        config
    }

    #[test]
    fn llama_is_the_plain_case() {
        let arch = config(json!({"model_type": "llama"}))
            .architecture()
            .unwrap();
        assert_eq!(arch.activation, Activation::Silu);
        assert_eq!(arch.embedding_scale, 1.0);
        assert_eq!(arch.norm_offset, 0.0);
        assert_eq!(arch.attention_scale, (32f32).powf(-0.5));
        assert_eq!(arch.rope_tables.len(), 1);
        assert!(
            arch.layers
                .iter()
                .all(|l| l.rope_table == 0 && l.window.is_none())
        );
        assert!(!arch.norms_after_blocks && !arch.combined_projections && !arch.head_norms);
    }

    #[test]
    fn rope_frequencies_drop_by_theta() {
        let arch = config(json!({"model_type": "llama", "rope_theta": 100.0}))
            .architecture()
            .unwrap();
        assert_eq!(arch.rope_tables[0][0], 1.0);
        assert!((arch.rope_tables[0][8] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn llama3_scaling_is_applied() {
        let arch = config(json!({"model_type": "llama", "rope_scaling": {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 100.0,
        }}))
        .architecture()
        .unwrap();
        let plain = config(json!({"model_type": "llama"}))
            .architecture()
            .unwrap();
        let (first, last) = (0, arch.rope_tables[0].len() - 1);
        // Wavelengths: 2 pi at the start, which is short; very long at the end.
        assert_eq!(arch.rope_tables[0][first], plain.rope_tables[0][first]);
        assert!((arch.rope_tables[0][last] - plain.rope_tables[0][last] / 8.0).abs() < 1e-7);
    }

    #[test]
    fn rope_parameters_work_like_the_older_fields() {
        let new = config(json!({"model_type": "llama", "rope_parameters": {
            "rope_theta": 500.0, "rope_type": "linear", "factor": 2.0,
        }}))
        .architecture()
        .unwrap();
        let old = config(
            json!({"model_type": "llama", "rope_theta": 500.0, "rope_scaling": {
                "type": "linear", "factor": 2.0,
            }}),
        )
        .architecture()
        .unwrap();
        assert_eq!(new.rope_tables, old.rope_tables);
    }

    #[test]
    fn unknown_scalings_are_refused() {
        let value = json!({
            "model_type": "phi3", "hidden_size": 128, "intermediate_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 100,
            "rope_scaling": {"type": "longrope", "long_factor": [1.0], "short_factor": [1.0]},
        });
        let config: Config = serde_json::from_value(value).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn smollm3_turns_rope_off_in_every_fourth_layer() {
        let arch = config(json!({"model_type": "smollm3"}))
            .architecture()
            .unwrap();
        assert_eq!(arch.rope_tables.len(), 2);
        assert!(arch.rope_tables[1].iter().all(|&f| f == 0.0));
        let tables: Vec<usize> = arch.layers.iter().map(|l| l.rope_table).collect();
        assert_eq!(tables, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn an_explicit_list_of_no_rope_layers_wins() {
        let flags = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
        let arch = config(json!({"model_type": "smollm3", "no_rope_layers": flags}))
            .architecture()
            .unwrap();
        let no_rope = arch.layers[0].rope_table;
        assert!(arch.rope_tables[no_rope].iter().all(|&f| f == 0.0));
        assert_ne!(arch.layers[1].rope_table, no_rope);
        assert_eq!(arch.layers[11].rope_table, no_rope);
    }

    #[test]
    fn gemma_differs_in_several_ways() {
        let arch = config(json!({"model_type": "gemma"}))
            .architecture()
            .unwrap();
        assert_eq!(arch.activation, Activation::Gelu);
        assert_eq!(arch.norm_offset, 1.0);
        assert_eq!(arch.embedding_scale, (128f32).sqrt());
        assert!(!arch.norms_after_blocks);
    }

    #[test]
    fn gemma_rounds_its_scale_at_16_bits() {
        let arch = config(json!({"model_type": "gemma", "hidden_size": 3072}))
            .architecture()
            .unwrap();
        assert_eq!(arch.embedding_scale, 3072f32.sqrt());
        assert_eq!(arch.embedding_scale_bf16, 55.5);
    }

    #[test]
    fn gemma2_caps_scores_by_default() {
        let c = config(json!({"model_type": "gemma2", "query_pre_attn_scalar": 16.0}));
        let arch = c.architecture().unwrap();
        assert_eq!(arch.attention_scale, 0.25);
        assert_eq!(arch.attention_softcap, 50.0);
        assert_eq!(arch.logit_softcap, 30.0);
        assert!(arch.norms_after_blocks && !arch.head_norms);
        let off = json!({"model_type": "gemma2", "attn_logit_softcapping": null,
                         "final_logit_softcapping": null});
        let arch = config(off).architecture().unwrap();
        assert_eq!((arch.attention_softcap, arch.logit_softcap), (0.0, 0.0));
    }

    #[test]
    fn gemma3_alternates_kinds_of_layer() {
        let arch = config(json!({"model_type": "gemma3_text", "sliding_window": 512}))
            .architecture()
            .unwrap();
        assert!(arch.head_norms && arch.norms_after_blocks);
        assert_eq!((arch.attention_softcap, arch.logit_softcap), (0.0, 0.0));
        assert_eq!(arch.rope_tables.len(), 2);
        for (l, layer) in arch.layers.iter().enumerate() {
            let global = (l + 1) % 6 == 0;
            assert_eq!(
                layer.window,
                if global { None } else { Some(512) },
                "layer {l}"
            );
            assert_eq!(layer.rope_table, global as usize);
        }
        // The first kind seen is sliding, at theta = 10,000; the other has 1,000,000.
        assert!(arch.rope_tables[0][8] > arch.rope_tables[1][8]);
    }

    #[test]
    fn gemma3_rope_can_be_given_per_kind_of_layer() {
        let arch = config(json!({"model_type": "gemma3_text", "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 100.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10.0},
        }}))
        .architecture()
        .unwrap();
        assert!((arch.rope_tables[0][8] - 0.316_227_8).abs() < 1e-5);
        assert!((arch.rope_tables[1][8] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn listed_layer_types_win() {
        let types: Vec<&str> = (0..12)
            .map(|l| match l {
                0 => "sliding_attention",
                _ => "full_attention",
            })
            .collect();
        let arch = config(json!({"model_type": "gemma3_text", "layer_types": types,
                                 "sliding_window": 9}))
        .architecture()
        .unwrap();
        assert_eq!(arch.layers[0].window, Some(9));
        assert_eq!(arch.layers[5].window, None);
    }

    #[test]
    fn mistral_windows_default_on_and_can_be_off() {
        let on = config(json!({"model_type": "mistral"}))
            .architecture()
            .unwrap();
        assert!(on.layers.iter().all(|l| l.window == Some(4096)));
        let off = config(json!({"model_type": "mistral", "sliding_window": null}))
            .architecture()
            .unwrap();
        assert!(off.layers.iter().all(|l| l.window.is_none()));
    }

    #[test]
    fn qwen_windows_need_switching_on() {
        let plain = config(json!({"model_type": "qwen3"}))
            .architecture()
            .unwrap();
        assert!(plain.layers.iter().all(|l| l.window.is_none()));
        let windowed = config(json!({"model_type": "qwen3", "use_sliding_window": true,
                                     "max_window_layers": 10, "sliding_window": 32}))
        .architecture()
        .unwrap();
        assert_eq!(windowed.layers[9].window, None);
        assert_eq!(windowed.layers[10].window, Some(32));
    }

    #[test]
    fn phi3_combines_its_projections() {
        let arch = config(json!({"model_type": "phi3"}))
            .architecture()
            .unwrap();
        assert!(arch.combined_projections);
        assert!(arch.layers.iter().all(|l| l.window.is_none()));
    }

    #[test]
    fn granite_scales_what_blocks_add() {
        let arch = config(json!({"model_type": "granite", "residual_multiplier": 0.22}))
            .architecture()
            .unwrap();
        assert_eq!(arch.residual_scale, 0.22);
    }

    #[test]
    fn granite_can_scale_its_embeddings_and_logits() {
        let arch = config(json!({"model_type": "granite", "embedding_multiplier": 2.0,
                                 "logits_scaling": 4.0}))
        .architecture()
        .unwrap();
        assert_eq!(arch.embedding_scale, 2.0);
        assert_eq!(arch.embedding_scale_bf16, 2.0);
        assert_eq!(arch.logit_divisor, 4.0);
        // Its attention isn't scaled by the size of a head, unlike the others.
        assert_eq!(arch.attention_scale, 1.0);
    }

    #[test]
    fn qwen3_normalizes_its_heads_and_qwen2_doesnt() {
        assert!(
            config(json!({"model_type": "qwen3"}))
                .architecture()
                .unwrap()
                .head_norms
        );
        assert!(
            !config(json!({"model_type": "qwen2"}))
                .architecture()
                .unwrap()
                .head_norms
        );
    }

    #[test]
    fn other_activations_and_types_are_refused() {
        for extra in [
            json!({"model_type": "llama", "hidden_act": "relu"}),
            json!({"model_type": "llama", "hidden_act": "gelu"}),
            json!({"model_type": "falcon"}),
            json!({"model_type": "gemma3_text", "use_bidirectional_attention": true}),
        ] {
            let mut value = json!({
                "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                "num_attention_heads": 4, "vocab_size": 100,
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let config: Config = serde_json::from_value(value).unwrap();
            assert!(config.validate().is_err(), "{extra}");
        }
    }

    #[test]
    fn rounding_to_bf16_keeps_8_bits() {
        assert_eq!(round_to_bf16(55.4256), 55.5);
        assert_eq!(round_to_bf16(27.7128), 27.75);
        assert_eq!(round_to_bf16(48.0), 48.0);
        assert_eq!(round_to_bf16(1.0), 1.0);
    }
}
