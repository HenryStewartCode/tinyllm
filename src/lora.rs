//! LoRA adapters: small corrections to a model's weights, trained for one task, which requests
//! can choose between while sharing the one copy of the model.
//!
//! An adapter replaces some of a layer's matrices `W` with `W + scale * B A`, where `A` is
//! `[rank, in]` and `B` is `[out, rank]` for a small rank (8, 16, 64...). Rather than change the
//! weights, which would need a copy of the model for each adapter, the backends add
//! `scale * B (A x)` to the output of `W x` for the rows that use the adapter: two thin matrix
//! products, cheap next to the full one, so a batch can mix requests using different adapters
//! and none.
//!
//! Adapters are read from the PEFT library's format, which is how they're shared on Hugging Face:
//! `adapter_config.json` and `adapter_model.safetensors`.

use crate::config::Config;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::path::Path;

/// The matrices of a layer that an adapter can change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Module {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Module {
    pub const ALL: [Module; 7] = [
        Module::Q,
        Module::K,
        Module::V,
        Module::O,
        Module::Gate,
        Module::Up,
        Module::Down,
    ];

    /// What the module is called in checkpoints.
    pub fn name(self) -> &'static str {
        match self {
            Module::Q => "q_proj",
            Module::K => "k_proj",
            Module::V => "v_proj",
            Module::O => "o_proj",
            Module::Gate => "gate_proj",
            Module::Up => "up_proj",
            Module::Down => "down_proj",
        }
    }

    /// `(in, out)`: the sizes of the module's input and output.
    pub fn shape(self, config: &Config) -> (usize, usize) {
        let (dim, hidden) = (config.hidden_size, config.intermediate_size);
        let q_dim = config.num_attention_heads * config.head_dim();
        let kv_dim = config.n_kv_heads() * config.head_dim();
        match self {
            Module::Q => (dim, q_dim),
            Module::K | Module::V => (dim, kv_dim),
            Module::O => (q_dim, dim),
            Module::Gate | Module::Up => (dim, hidden),
            Module::Down => (hidden, dim),
        }
    }
}

/// One module's correction: `scale * B (A x)` is added to its output.
#[derive(Clone, Debug)]
pub struct LoraPair {
    pub rank: usize,
    /// `[rank, in]`
    pub a: Vec<f32>,
    /// `[out, rank]`, already multiplied by the adapter's scale.
    pub b: Vec<f32>,
}

/// An adapter, read and checked against the model it's for.
pub struct LoraAdapter {
    pub name: String,
    /// For each layer, the correction to each module (in the order of [`Module::ALL`]), if the
    /// adapter changes it.
    pub layers: Vec<[Option<LoraPair>; 7]>,
}

impl LoraAdapter {
    /// Read the adapter in `dir`, for a model with `config`.
    pub fn load(name: &str, dir: &Path, config: &Config) -> Result<Self> {
        let path = dir.join("adapter_config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("no adapter_config.json in {}", dir.display()))?;
        let settings: Value =
            serde_json::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        let peft_type = settings["peft_type"].as_str().unwrap_or("LORA");
        ensure!(
            peft_type.eq_ignore_ascii_case("lora"),
            "{name}: {peft_type} adapters aren't supported, only LoRA"
        );
        ensure!(
            !settings["use_dora"].as_bool().unwrap_or(false),
            "{name}: DoRA adapters aren't supported"
        );
        ensure!(
            !settings["fan_in_fan_out"].as_bool().unwrap_or(false),
            "{name}: adapters with fan_in_fan_out aren't supported"
        );
        let bias = settings["bias"].as_str().unwrap_or("none");
        ensure!(
            bias == "none",
            "{name}: adapters that train biases aren't supported"
        );
        let saved = &settings["modules_to_save"];
        ensure!(
            saved.is_null() || saved.as_array().is_some_and(|a| a.is_empty()),
            "{name}: adapters that replace whole modules ({saved}) aren't supported"
        );
        let alpha = settings["lora_alpha"].as_f64().unwrap_or(8.0);
        let rslora = settings["use_rslora"].as_bool().unwrap_or(false);
        let file = dir.join("adapter_model.safetensors");
        ensure!(
            file.exists(),
            "{name}: needs adapter_model.safetensors in {}",
            dir.display()
        );
        let st = SafeTensors::open_file(&file)?;

        let mut layers: Vec<[Option<LoraPair>; 7]> = (0..config.num_hidden_layers)
            .map(|_| Default::default())
            .collect();
        let mut found = 0;
        for name_a in st.names().filter(|n| n.contains(".lora_A.")) {
            let name_b = name_a.replace(".lora_A.", ".lora_B.");
            let (layer, module) = locate(name_a)
                .with_context(|| format!("{name}: don't know where {name_a} goes"))?;
            ensure!(
                layer < layers.len(),
                "{name}: {name_a} is for layer {layer}, but the model has {}",
                layers.len()
            );
            let pattern_value = |key: &str| {
                settings[key].as_object().and_then(|patterns| {
                    patterns
                        .iter()
                        .find(|(pattern, _)| name_a.contains(pattern.as_str()))
                        .and_then(|(_, v)| v.as_f64())
                })
            };
            let alpha = pattern_value("alpha_pattern").unwrap_or(alpha);
            // The rank is in the tensors' shapes.
            let rank = st
                .shape(name_a)
                .and_then(|shape| shape.first().copied())
                .with_context(|| format!("{name}: {name_a} has no shape"))?;
            ensure!(rank > 0, "{name}: {name_a} has rank 0");
            let scale = match rslora {
                true => alpha / (rank as f64).sqrt(),
                false => alpha / rank as f64,
            } as f32;
            let parts: &[Module] = match module {
                Target::One(module) => &[module][..],
                Target::Qkv => &[Module::Q, Module::K, Module::V],
                Target::GateUp => &[Module::Gate, Module::Up],
            };
            let in_dim = parts[0].shape(config).0;
            let out_dims: Vec<usize> = parts.iter().map(|m| m.shape(config).1).collect();
            let total_out: usize = out_dims.iter().sum();
            let a = st.load(name_a, &[rank, in_dim])?;
            let b = st.load(&name_b, &[total_out, rank])?;
            let mut start = 0;
            for (&part, &out) in parts.iter().zip(&out_dims) {
                let slot = &mut layers[layer][part as usize];
                ensure!(
                    slot.is_none(),
                    "{name}: layer {layer} has two {part:?} corrections"
                );
                *slot = Some(LoraPair {
                    rank,
                    a: a.clone(),
                    b: b[start * rank..(start + out) * rank]
                        .iter()
                        .map(|v| v * scale)
                        .collect(),
                });
                start += out;
            }
            found += 1;
        }
        ensure!(found > 0, "{name}: no LoRA weights in {}", file.display());
        Ok(Self {
            name: name.to_string(),
            layers,
        })
    }

    /// The largest rank of any of the adapter's corrections.
    pub fn max_rank(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .flatten()
            .map(|pair| pair.rank)
            .max()
            .unwrap_or(0)
    }
}

/// Which module (or modules stored together) a LoRA tensor corrects.
enum Target {
    One(Module),
    /// Phi-3's `qkv_proj`.
    Qkv,
    /// Phi-3's `gate_up_proj`.
    GateUp,
}

/// The layer and module of a tensor named like
/// `base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight`.
fn locate(name: &str) -> Option<(usize, Target)> {
    let after = &name[name.find("layers.")? + "layers.".len()..];
    let layer: usize = after[..after.find('.')?].parse().ok()?;
    let before = &name[..name.find(".lora_A.")?];
    let module = &before[before.rfind('.')? + 1..];
    if before.contains("experts") {
        return None;
    }
    let target = match module {
        "qkv_proj" => Target::Qkv,
        "gate_up_proj" => Target::GateUp,
        other => Target::One(*Module::ALL.iter().find(|m| m.name() == other)?),
    };
    Some((layer, target))
}

/// The adapters a server has, by name. Each keeps its number for good: one that's unloaded
/// leaves a gap, so requests still running with it are unaffected and a new one never takes
/// its number (or its cached prompts).
#[derive(Default)]
pub struct Adapters {
    names: Vec<Option<String>>,
}

impl Adapters {
    pub fn new(names: Vec<String>) -> Self {
        Self {
            names: names.into_iter().map(Some).collect(),
        }
    }

    /// The number rows use for adapter `name` (1 for the first; 0 means none), if there is one.
    pub fn find(&self, name: &str) -> Option<u16> {
        self.names
            .iter()
            .position(|n| n.as_deref() == Some(name))
            .map(|i| i as u16 + 1)
    }

    /// The adapters' names, in order.
    pub fn names(&self) -> Vec<String> {
        self.names.iter().flatten().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.names.iter().all(Option::is_none)
    }

    /// Note that the backend has been given another adapter, which becomes the next number.
    pub fn push(&mut self, name: String) -> u16 {
        self.names.push(Some(name));
        self.names.len() as u16
    }

    /// Stop offering adapter `name`. Returns whether there was one.
    pub fn remove(&mut self, name: &str) -> bool {
        match self.find(name) {
            Some(number) => {
                self.names[number as usize - 1] = None;
                true
            }
            None => false,
        }
    }
}

/// Parse a `NAME=PATH` (or just `PATH`, named after its last directory) adapter argument.
pub fn parse_argument(argument: &str) -> Result<(String, std::path::PathBuf)> {
    let (name, path) = match argument.split_once('=') {
        Some((name, path)) => (name.to_string(), std::path::PathBuf::from(path)),
        None => {
            let path = std::path::PathBuf::from(argument);
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .context("an adapter path needs a name: use NAME=PATH")?
                .to_string();
            (name, path)
        }
    };
    if name.is_empty() {
        bail!("an adapter needs a name: use NAME=PATH");
    }
    Ok((name, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_names_are_located() {
        let (layer, target) =
            locate("base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight").unwrap();
        assert_eq!(layer, 12);
        assert!(matches!(target, Target::One(Module::Q)));
        let (layer, target) =
            locate("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight").unwrap();
        assert_eq!(layer, 0);
        assert!(matches!(target, Target::One(Module::Down)));
        let (_, target) =
            locate("base_model.model.model.layers.1.self_attn.qkv_proj.lora_A.weight").unwrap();
        assert!(matches!(target, Target::Qkv));
        assert!(locate("base_model.model.model.layers.1.mlp.experts.3.w1.lora_A.weight").is_none());
        assert!(locate("base_model.model.lm_head.lora_A.weight").is_none());
    }

    #[test]
    fn adapter_arguments_parse() {
        let (name, path) = parse_argument("sql=/adapters/sql-lora").unwrap();
        assert_eq!(
            (name.as_str(), path.to_str().unwrap()),
            ("sql", "/adapters/sql-lora")
        );
        let (name, _) = parse_argument("/adapters/chat-lora/").unwrap();
        assert_eq!(name, "chat-lora");
    }
}
