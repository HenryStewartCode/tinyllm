//! Minimal reader for Hugging Face `.safetensors` checkpoints.
//!
//! A file is an 8-byte little-endian header length `N`, then `N` bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets: [begin, end]}`, then the raw tensor bytes
//! (offsets count from the end of the header). Large models are split across several files
//! that are listed in `model.safetensors.index.json`.
//!
//! Weights quantized to 4 bits (AWQ or GPTQ checkpoints) are stored as several tensors in place
//! of `{layer}.weight`. Asking for that weight unpacks them, so the rest of tinyllm can treat them
//! like any other; [`SafeTensors::load_int4`] gives them still packed, for kernels that keep them
//! that way.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone)]
struct Entry {
    file: usize,
    /// Where the data section of `file` starts.
    data_start: u64,
    info: TensorInfo,
}

pub struct SafeTensors {
    files: Vec<PathBuf>,
    tensors: HashMap<String, Entry>,
    /// How 4-bit weights are packed, if the checkpoint has them.
    int4: Option<Int4Format>,
}

/// How a checkpoint packs its 4-bit weights, from `quantization_config` in `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Int4Format {
    pub packing: Packing,
    /// How many inputs share each scale and zero point, or `None` for a whole row.
    pub group_size: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Packing {
    /// AutoAWQ's "gemm" layout: `qweight` is `[in, out / 8]`, eight outputs to a 32-bit word in
    /// the order 0, 2, 4, 6, 1, 3, 5, 7; zero points are packed the same way.
    Awq,
    /// GPTQ: `qweight` is `[in / 8, out]`, eight inputs to a word in order; zero points are packed
    /// along the outputs, and in the original format stored one less than they are.
    Gptq { zero_offset: u8 },
}

impl Int4Format {
    /// The format described by a `quantization_config`, if it's 4-bit AWQ or GPTQ.
    pub fn from_config(config: &serde_json::Value) -> Result<Option<Self>> {
        let packing = match config["quant_method"].as_str() {
            Some("awq") => {
                let version = config["version"].as_str().unwrap_or("gemm");
                ensure!(
                    version.eq_ignore_ascii_case("gemm"),
                    "AWQ checkpoints in the {version} layout aren't supported (gemm ones are)"
                );
                ensure!(
                    config["zero_point"].as_bool().unwrap_or(true),
                    "AWQ checkpoints without zero points aren't supported"
                );
                Packing::Awq
            }
            Some("gptq") => {
                let format = config["checkpoint_format"].as_str().unwrap_or("gptq");
                let zero_offset = match format {
                    "gptq" => 1,
                    "gptq_v2" => 0,
                    other => bail!("GPTQ checkpoints in the {other} format aren't supported"),
                };
                Packing::Gptq { zero_offset }
            }
            _ => return Ok(None),
        };
        let bits = config["bits"].as_u64().unwrap_or(4);
        ensure!(
            bits == 4,
            "{bits}-bit AWQ and GPTQ checkpoints aren't supported (4-bit ones are)"
        );
        let group_size = match config["group_size"].as_i64().unwrap_or(128) {
            -1 => None,
            size if size > 0 && size % 8 == 0 => Some(size as usize),
            size => bail!("a quantization group size of {size} isn't supported"),
        };
        Ok(Some(Self {
            packing,
            group_size,
        }))
    }
}

/// A 4-bit weight matrix, unpacked: `values` (0 to 15) for each `[out, in]`, and for each output
/// and group of `group` inputs a zero point and scale, `[out, in / group]`:
/// weight = (value - zero) * scale.
pub struct Int4Matrix {
    pub values: Vec<u8>,
    pub zeros: Vec<u8>,
    pub scales: Vec<f32>,
    pub group: usize,
}

impl Int4Matrix {
    /// The weights as 32-bit floats, which hold them exactly.
    pub fn dequantize(&self, rows: usize, cols: usize) -> Vec<f32> {
        let groups = cols.div_ceil(self.group);
        (0..rows * cols)
            .map(|i| {
                let (row, column) = (i / cols, i % cols);
                let g = row * groups + column / self.group;
                (self.values[i] as f32 - self.zeros[g] as f32) * self.scales[g]
            })
            .collect()
    }
}

/// Where AWQ puts output `c` of each group of eight in a word (as the nibble's index).
const AWQ_SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

impl SafeTensors {
    /// Index `model.safetensors`, or every shard named in `model.safetensors.index.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with(dir, None)
    }

    /// `open`, taking the embeddings from the checkpoint in `fallback` if this one has none (an
    /// EAGLE-3 draft head uses its model's). An EAGLE-3 head's tensors are renamed to where a
    /// one-layer model's would be: `midlayer.*` to `model.layers.0.*` and `norm` to
    /// `model.norm`.
    pub fn open_with(dir: &Path, fallback: Option<&Path>) -> Result<Self> {
        let mut st = Self::index(find_files(dir)?)?;
        if st.tensors.contains_key("fc.weight")
            && st.tensors.keys().any(|name| name.starts_with("midlayer."))
        {
            st.tensors = std::mem::take(&mut st.tensors)
                .into_iter()
                .map(|(name, entry)| {
                    let name = match name.strip_prefix("midlayer.") {
                        Some(rest) => format!("model.layers.0.{rest}"),
                        None if name == "norm.weight" || name == "embed_tokens.weight" => {
                            format!("model.{name}")
                        }
                        None => name,
                    };
                    (name, entry)
                })
                .collect();
        }
        if let Some(fallback) = fallback
            && !st.tensors.contains_key("model.embed_tokens.weight")
        {
            let other = Self::open(fallback)?;
            let name = "model.embed_tokens.weight";
            let entry = other
                .tensors
                .get(name)
                .with_context(|| format!("{} has no embeddings to borrow", fallback.display()))?;
            st.files.push(other.files[entry.file].clone());
            let mut entry = entry.clone();
            entry.file = st.files.len() - 1;
            st.tensors.insert(name.to_string(), entry);
        }
        // Checkpoints of a bare model (embedding models, often) leave off the "model." that the
        // names of the same weights carry inside a model for generating text.
        let bare = st.tensors.contains_key("embed_tokens.weight")
            && !st.tensors.keys().any(|name| name.starts_with("model."));
        if bare {
            st.tensors = std::mem::take(&mut st.tensors)
                .into_iter()
                .map(|(name, entry)| match name.starts_with("lm_head.") {
                    true => (name, entry),
                    false => (format!("model.{name}"), entry),
                })
                .collect();
        }
        // How 4-bit weights are packed comes from the model's configuration.
        if let Ok(text) = std::fs::read_to_string(dir.join("config.json")) {
            let config: serde_json::Value = serde_json::from_str(&text)?;
            if let Some(quantization) = config.get("quantization_config") {
                st.int4 = Int4Format::from_config(quantization)?;
            }
        }
        Ok(st)
    }

    /// Index one file, taking tensor names as they are.
    pub fn open_file(path: &Path) -> Result<Self> {
        Self::index(vec![path.to_path_buf()])
    }

    fn index(files: Vec<PathBuf>) -> Result<Self> {
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
        Ok(Self {
            files,
            tensors,
            int4: None,
        })
    }

    /// The names of all the tensors.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The shape tensor `name` is stored with.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .get(name)
            .map(|entry| entry.info.shape.as_slice())
    }

    /// A tensor of 64-bit integers (an EAGLE-3 head's `d2t`, say).
    pub fn load_i64(&self, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
        let (bytes, dtype) = self.read(name, shape)?;
        ensure!(dtype == "I64", "tensor {name}: expected I64, found {dtype}");
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name) || self.int4_prefix(name).is_some()
    }

    /// How tensor `name` is stored: `"BF16"`, `"F16"`, `"F32"`..., or `"INT4"` for a weight
    /// quantized to 4 bits.
    pub fn dtype(&self, name: &str) -> Option<&str> {
        match self.int4_prefix(name) {
            Some(_) => Some("INT4"),
            None => self
                .tensors
                .get(name)
                .map(|entry| entry.info.dtype.as_str()),
        }
    }

    /// For a weight stored as 4-bit values (`{layer}.weight` in a quantized checkpoint), the
    /// `{layer}` its packed tensors are named after.
    fn int4_prefix<'a>(&self, name: &'a str) -> Option<&'a str> {
        self.int4?;
        let prefix = name.strip_suffix(".weight")?;
        let packed = !self.tensors.contains_key(name)
            && self.tensors.contains_key(&format!("{prefix}.qweight"));
        packed.then_some(prefix)
    }

    /// A `[rows, cols]` weight stored as 4-bit values, unpacked but not multiplied out, or `None`
    /// if it's stored some other way. Its zero points and scales come in groups of `group`
    /// columns, which for GPTQ with the columns reordered ("act order") isn't possible, so that's
    /// an error here (though [`SafeTensors::load`] can still read the weight).
    pub fn load_int4(&self, name: &str, shape: &[usize]) -> Result<Option<Int4Matrix>> {
        match self.int4_parts(name, shape)? {
            None => Ok(None),
            Some((matrix, None)) => Ok(Some(matrix)),
            Some((_, Some(_))) => bail!("{name}: GPTQ with reordered columns (act order)"),
        }
    }

    /// The unpacked parts of a 4-bit weight, and for GPTQ with reordered columns, the group of
    /// each column (when `matrix.zeros` and `scales` are indexed by that rather than by
    /// `column / group`).
    #[allow(clippy::type_complexity)]
    fn int4_parts(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<Option<(Int4Matrix, Option<Vec<usize>>)>> {
        let (Some(prefix), Some(format)) = (self.int4_prefix(name), self.int4) else {
            return Ok(None);
        };
        ensure!(shape.len() == 2, "{name}: 4-bit weights only for matrices");
        let (rows, cols) = (shape[0], shape[1]);
        ensure!(
            rows.is_multiple_of(8) && cols.is_multiple_of(8),
            "{name}: 4-bit weights need sizes that are multiples of 8"
        );
        let group = format.group_size.unwrap_or(cols);
        let groups = cols.div_ceil(group);
        let words = |suffix: &str, shape: &[usize]| -> Result<Vec<u32>> {
            let name = format!("{prefix}.{suffix}");
            let (bytes, dtype) = self.read(&name, shape)?;
            ensure!(
                dtype == "I32",
                "{name} should be 32-bit integers, not {dtype}"
            );
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| u32::from_le_bytes(b))
                .collect())
        };
        let nibble = |word: u32, index: usize| ((word >> (4 * index)) & 15) as u8;
        // Scales are `[groups, out]` in both formats.
        let stored_scales = self.load(&format!("{prefix}.scales"), &[groups, rows])?;
        let mut scales = vec![0.0f32; rows * groups];
        let mut zeros = vec![0u8; rows * groups];
        let mut values = vec![0u8; rows * cols];
        let packed_zeros = words("qzeros", &[groups, rows / 8])?;
        let mut column_groups = None;
        match format.packing {
            Packing::Awq => {
                // Eight rows at a time, so the values are written in order along each row.
                let packed = words("qweight", &[cols, rows / 8])?;
                in_parallel(&mut values, 8 * cols, |w, eight| {
                    for column in 0..cols {
                        let word = packed[column * (rows / 8) + w];
                        for (k, &slot) in AWQ_SLOT.iter().enumerate() {
                            eight[k * cols + column] = nibble(word, slot);
                        }
                    }
                });
                for g in 0..groups {
                    for row in 0..rows {
                        let word = packed_zeros[g * (rows / 8) + row / 8];
                        zeros[row * groups + g] = nibble(word, AWQ_SLOT[row % 8]);
                        scales[row * groups + g] = stored_scales[g * rows + row];
                    }
                }
            }
            Packing::Gptq { zero_offset } => {
                let packed = words("qweight", &[cols / 8, rows])?;
                in_parallel(&mut values, cols, |row, out| {
                    for (w, eight) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let word = packed[w * rows + row];
                        for (k, value) in eight.iter_mut().enumerate() {
                            *value = nibble(word, k);
                        }
                    }
                });
                for g in 0..groups {
                    for row in 0..rows {
                        let word = packed_zeros[g * (rows / 8) + row / 8];
                        zeros[row * groups + g] = nibble(word, row % 8) + zero_offset;
                        scales[row * groups + g] = stored_scales[g * rows + row];
                    }
                }
                let g_idx = format!("{prefix}.g_idx");
                if self.tensors.contains_key(&g_idx) {
                    let order: Vec<usize> = words("g_idx", &[cols])?
                        .into_iter()
                        .map(|g| g as usize)
                        .collect();
                    ensure!(
                        order.iter().all(|&g| g < groups),
                        "{g_idx} names a group that doesn't exist"
                    );
                    if order.iter().enumerate().any(|(c, &g)| g != c / group) {
                        column_groups = Some(order);
                    }
                }
            }
        }
        let matrix = Int4Matrix {
            values,
            zeros,
            scales,
            group,
        };
        Ok(Some((matrix, column_groups)))
    }

    /// Read tensor `name`, check that it has the expected `shape`, and convert it to f32. Weights
    /// stored as 8-bit floats come out multiplied by their block scales.
    pub fn load(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        if let Some((matrix, column_groups)) = self.int4_parts(name, shape)? {
            let Some(column_groups) = column_groups else {
                return Ok(matrix.dequantize(shape[0], shape[1]));
            };
            // GPTQ with reordered columns: each column's group comes from the list.
            let (rows, cols) = (shape[0], shape[1]);
            let groups = cols.div_ceil(matrix.group);
            return Ok((0..rows * cols)
                .map(|i| {
                    let (row, column) = (i / cols, i % cols);
                    let g = row * groups + column_groups[column];
                    (matrix.values[i] as f32 - matrix.zeros[g] as f32) * matrix.scales[g]
                })
                .collect());
        }
        if let Some((bytes, scales)) = self.load_fp8(name, shape)? {
            let (rows, cols) = (shape[0], shape[1]);
            let column_blocks = cols.div_ceil(FP8_BLOCK);
            return Ok((0..rows * cols)
                .map(|i| {
                    let (row, column) = (i / cols, i % cols);
                    let scale = scales[(row / FP8_BLOCK) * column_blocks + column / FP8_BLOCK];
                    e4m3_to_f32(bytes[i]) * scale
                })
                .collect());
        }
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

    /// For a `[rows, cols]` weight stored as 8-bit floats (e4m3, as in Qwen's and DeepSeek's FP8
    /// checkpoints), its bytes and the scale of each 128x128 block, `[row blocks, column blocks]`:
    /// weight = value * scale. `None` for weights stored any other way.
    pub fn load_fp8(&self, name: &str, shape: &[usize]) -> Result<Option<(Vec<u8>, Vec<f32>)>> {
        if self.dtype(name) != Some("F8_E4M3") {
            return Ok(None);
        }
        ensure!(
            shape.len() == 2,
            "tensor {name}: 8-bit floats only for matrices"
        );
        let (bytes, _) = self.read(name, shape)?;
        let scale_name = format!("{name}_scale_inv");
        let blocks = [shape[0].div_ceil(FP8_BLOCK), shape[1].div_ceil(FP8_BLOCK)];
        let scales = self
            .load(&scale_name, &blocks)
            .with_context(|| format!("{name} needs 128x128 block scales in {scale_name}"))?;
        Ok(Some((bytes, scales)))
    }

    /// Read tensor `name` as bfloat16. Tensors stored that way come through exactly; others are
    /// rounded.
    pub fn load_bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        if matches!(self.dtype(name), Some("F8_E4M3" | "INT4")) {
            return Ok(self
                .load(name, shape)?
                .into_iter()
                .map(f32_to_bf16)
                .collect());
        }
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

    /// Experts' weights packed in MXFP4, as GPT-OSS stores them: `{name}_blocks`, `[experts,
    /// rows, cols / 32, 16]` bytes of 4-bit floats (two to a byte, the first in the low nibble),
    /// and `{name}_scales`, `[experts, rows, cols / 32]`, a power of two for each 32 as an
    /// exponent biased by 127. The two as stored, or `None` if the weights aren't stored this way.
    pub fn load_mxfp4(
        &self,
        name: &str,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let (blocks, scales) = (format!("{name}_blocks"), format!("{name}_scales"));
        if !self.tensors.contains_key(&blocks) {
            return Ok(None);
        }
        ensure!(cols.is_multiple_of(32), "{name}: MXFP4 rows come in 32s");
        let (packed, _) = self.read(&blocks, &[experts, rows, cols / 32, 16])?;
        let (exponents, _) = self.read(&scales, &[experts, rows, cols / 32])?;
        Ok(Some((packed, exponents)))
    }

    /// MXFP4 experts' weights (see [`SafeTensors::load_mxfp4`]) as bfloat16, `[experts, rows,
    /// cols]`, which holds them exactly.
    pub fn load_mxfp4_bf16(
        &self,
        name: &str,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Vec<u16>>> {
        let Some((packed, exponents)) = self.load_mxfp4(name, experts, rows, cols)? else {
            return Ok(None);
        };
        let mut values = Vec::with_capacity(experts * rows * cols);
        for (block, &exponent) in packed.as_chunks::<16>().0.iter().zip(&exponents) {
            let scale = power_of_two(exponent);
            for &byte in block {
                for code in [byte & 15, byte >> 4] {
                    values.push(f32_to_bf16(FP4_VALUES[code as usize] * scale));
                }
            }
        }
        Ok(Some(values))
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
            "U8" | "I8" | "BOOL" => 1,
            "F32" | "I32" => 4,
            "I64" => 8,
            "F16" | "BF16" => 2,
            "F8_E4M3" => 1,
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

/// Run `f(index, piece)` on each `piece_len` piece of `values`, on all the CPU's threads.
fn in_parallel(values: &mut [u8], piece_len: usize, f: impl Fn(usize, &mut [u8]) + Sync) {
    let pieces = values.len() / piece_len;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per_thread = pieces.div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        for (part, chunk) in values.chunks_mut(per_thread * piece_len).enumerate() {
            scope.spawn(move || {
                for (i, piece) in chunk.chunks_exact_mut(piece_len).enumerate() {
                    f(part * per_thread + i, piece);
                }
            });
        }
    });
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

/// The values of 4-bit floats (2 exponent bits, 1 mantissa bit, as in MXFP4), by their bits.
pub const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2^(exponent - 127), as MXFP4 stores its scales.
pub fn power_of_two(exponent: u8) -> f32 {
    match exponent {
        0 => f32::from_bits(1 << 22), // 2^-127, below normal
        255 => f32::NAN,
        e => f32::from_bits((e as u32) << 23),
    }
}

/// The block size of 8-bit float weights' scales.
pub const FP8_BLOCK: usize = 128;

/// An 8-bit float with 4 exponent bits and 3 mantissa bits, the "fn" kind with no infinities.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let (exponent, mantissa) = ((byte >> 3) & 15, byte & 7);
    let magnitude = match exponent {
        0 => mantissa as f32 / 512.0,
        15 if mantissa == 7 => f32::NAN,
        _ => (8 + mantissa) as f32 * 2f32.powi(exponent as i32 - 10),
    };
    if byte & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
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

/// IEEE half precision: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits. Every value is
/// exact in f32, so this just moves the bits (only numbers below normal need arithmetic).
pub fn f16_to_f32(x: u16) -> f32 {
    let sign = ((x & 0x8000) as u32) << 16;
    let exp = ((x >> 10) & 0x1f) as u32;
    let mantissa = (x & 0x3ff) as u32;
    match exp {
        // Zero and numbers below normal: mantissa * 2^-24, exact.
        0 => f32::from_bits(sign | (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits()),
        // Infinities and NaNs keep their mantissa bits, moved up.
        31 => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decodes() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exponent 7, the bias
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // the largest
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0); // the smallest, below normal
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

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
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x83ff), -1023.0 * 2f32.powi(-24));
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
    }
}
