//! Minimal reader for Hugging Face `.safetensors` checkpoints.
//!
//! A file is an 8-byte little-endian header length `N`, then `N` bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets: [begin, end]}`, then the raw tensor bytes
//! (offsets count from the end of the header). Large models are split across several files
//! that are listed in `model.safetensors.index.json`.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Entry {
    file: usize,
    /// Where the data section of `file` starts.
    data_start: u64,
    info: TensorInfo,
}

pub struct SafeTensors {
    files: Vec<PathBuf>,
    tensors: HashMap<String, Entry>,
}

impl SafeTensors {
    /// Index `model.safetensors`, or every shard named in `model.safetensors.index.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        let files = find_files(dir)?;
        let mut tensors = HashMap::new();
        for (i, path) in files.iter().enumerate() {
            let mut file =
                File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let mut len = [0u8; 8];
            file.read_exact(&mut len)?;
            let header_len = u64::from_le_bytes(len);
            ensure!(
                header_len < 100 << 20,
                "{}: not a safetensors file",
                path.display()
            );
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)?;
            let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .with_context(|| format!("{}: bad header", path.display()))?;
            for (name, value) in header {
                if name == "__metadata__" {
                    continue;
                }
                let info: TensorInfo =
                    serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
                tensors.insert(
                    name,
                    Entry {
                        file: i,
                        data_start: 8 + header_len,
                        info,
                    },
                );
            }
        }
        // Checkpoints of a bare model (embedding models, often) leave off the "model." that the
        // names of the same weights carry inside a model for generating text.
        let bare = tensors.contains_key("embed_tokens.weight")
            && !tensors.keys().any(|name| name.starts_with("model."));
        if bare {
            tensors = tensors
                .into_iter()
                .map(|(name, entry)| match name.starts_with("lm_head.") {
                    true => (name, entry),
                    false => (format!("model.{name}"), entry),
                })
                .collect();
        }
        Ok(Self { files, tensors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// How tensor `name` is stored: `"BF16"`, `"F16"`, `"F32"`...
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.tensors
            .get(name)
            .map(|entry| entry.info.dtype.as_str())
    }

    /// Read tensor `name`, check that it has the expected `shape`, and convert it to f32.
    pub fn load(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (bytes, dtype) = self.read(name, shape)?;
        let halves = || halves(&bytes);
        Ok(match dtype {
            "F32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            "BF16" => halves().map(bf16_to_f32).collect(),
            "F16" => halves().map(f16_to_f32).collect(),
            _ => unreachable!(),
        })
    }

    /// Read tensor `name` as bfloat16. Tensors stored that way come through exactly; others are
    /// rounded.
    pub fn load_bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let (bytes, dtype) = self.read(name, shape)?;
        Ok(match dtype {
            "BF16" => halves(&bytes).collect(),
            "F32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32_to_bf16(f32::from_le_bytes(b)))
                .collect(),
            "F16" => halves(&bytes)
                .map(|half| f32_to_bf16(f16_to_f32(half)))
                .collect(),
            _ => unreachable!(),
        })
    }

    /// The raw bytes of a tensor, after checking its shape.
    fn read(&self, name: &str, shape: &[usize]) -> Result<(Vec<u8>, &str)> {
        let entry = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        let info = &entry.info;
        ensure!(
            info.shape == shape,
            "tensor {name}: expected shape {shape:?}, found {:?}",
            info.shape
        );
        let elem_size = match info.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => bail!("tensor {name}: unsupported dtype {other}"),
        };
        let [begin, end] = info.data_offsets;
        let numel: usize = shape.iter().product();
        ensure!(
            end - begin == (numel * elem_size) as u64,
            "tensor {name}: wrong byte size"
        );

        let mut bytes = vec![0u8; (end - begin) as usize];
        let mut file = File::open(&self.files[entry.file])?;
        file.seek(SeekFrom::Start(entry.data_start + begin))?;
        file.read_exact(&mut bytes)
            .with_context(|| format!("reading tensor {name}"))?;
        Ok((bytes, info.dtype.as_str()))
    }
}

fn find_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path).with_context(|| {
        format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            dir.display()
        )
    })?;
    let index: serde_json::Value = serde_json::from_str(&text)?;
    let weight_map = index["weight_map"]
        .as_object()
        .context("index has no weight_map")?;
    let mut names: Vec<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    Ok(names.into_iter().map(|name| dir.join(name)).collect())
}

fn halves(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
}

/// Round to the nearest bfloat16, ties to even.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return (bits >> 16) as u16 | 0x40;
    }
    // Rounding up from the largest finite value properly gives infinity.
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// bfloat16 is the top half of an f32, so widening is a shift.
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// IEEE half precision: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
pub fn f16_to_f32(x: u16) -> f32 {
    let sign = if x & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((x >> 10) & 0x1f) as i32;
    let mantissa = (x & 0x3ff) as f32;
    match exp {
        0 => sign * mantissa * 2f32.powi(-24), // zero and subnormals
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_to_bfloat16() {
        for x in [0.0f32, 1.0, -2.5, 0.15625, 1e-5, -3.5e7] {
            let back = bf16_to_f32(f32_to_bf16(x));
            assert!(
                ((back - x) / x.abs().max(1e-9)).abs() < 0.005,
                "{x} -> {back}"
            );
        }
        // Halfway between two values: ties go to the one with an even last bit.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
    }

    #[test]
    fn half_precision_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x3555), 0.33325195);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
    }
}
