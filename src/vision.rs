//! Images, for vision-language models (Qwen2.5-VL): decoding them, cutting them into the patches
//! the model's vision encoder takes (as transformers' image processor does), and the encoder
//! itself on the CPU.
//!
//! An image is resized so that both sides are a whole number of `patch_size * merge_size` pixels
//! (28), cut into `patch_size` (14) squares, and each square is repeated `temporal_patch_size` (2)
//! times, as a still frame of video would be. The encoder is a transformer over those patches,
//! with 2D RoPE (half of each head turns with the patch's row, half with its column); most of its
//! layers only attend within windows of `window_size` pixels, a few across the whole image. At
//! the end each `merge_size` × `merge_size` block of patches becomes one token for the language
//! model, which reads it in place of a placeholder token, at a 3D position (see
//! [`crate::config::Architecture::mrope`]).

use crate::config::VisionConfig;
use crate::model::Linear;
use crate::ops::{add_rows, rmsnorm, rope, swiglu, with_scratch};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::path::Path;

/// How images are prepared for the model, from `preprocessor_config.json`.
#[derive(Clone, Debug)]
pub struct Preprocessor {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Preprocessor {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("preprocessor_config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let json: Value = serde_json::from_str(&text)?;
        let number = |key: &str, default: usize| json[key].as_u64().map_or(default, |v| v as usize);
        let triple = |key: &str, default: [f32; 3]| -> [f32; 3] {
            match json[key].as_array() {
                Some(values) if values.len() == 3 => {
                    let get = |i: usize| values[i].as_f64().unwrap_or(default[i] as f64) as f32;
                    [get(0), get(1), get(2)]
                }
                _ => default,
            }
        };
        // Newer files say `size: {shortest_edge, longest_edge}` for the pixel limits.
        let size = &json["size"];
        Ok(Self {
            min_pixels: (json["min_pixels"].as_u64())
                .or(size["shortest_edge"].as_u64())
                .map_or(56 * 56, |v| v as usize),
            max_pixels: (json["max_pixels"].as_u64())
                .or(size["longest_edge"].as_u64())
                .map_or(28 * 28 * 1280, |v| v as usize),
            patch_size: number("patch_size", 14),
            temporal_patch_size: number("temporal_patch_size", 2),
            merge_size: number("merge_size", 2),
            mean: triple("image_mean", [0.481_454_66, 0.457_827_5, 0.408_210_73]),
            std: triple("image_std", [0.268_629_54, 0.261_302_6, 0.275_777_1]),
        })
    }

    /// Turn an image into patches for the vision encoder.
    pub fn prepare(&self, picture: &Picture) -> Result<Patches> {
        let factor = self.patch_size * self.merge_size;
        let (height, width) = smart_resize(
            picture.height,
            picture.width,
            factor,
            self.min_pixels,
            self.max_pixels,
        )?;
        let resized;
        let picture = match (height, width) == (picture.height, picture.width) {
            true => picture,
            false => {
                resized = resize(picture, width, height);
                &resized
            }
        };
        Ok(self.patchify(picture))
    }

    /// Normalize an image whose sides are whole numbers of blocks, and lay out its patches:
    /// blocks of `merge_size` × `merge_size` patches in rows, each block's patches in rows, each
    /// patch's values by channel, then frame (repeated), then pixel row and column.
    fn patchify(&self, picture: &Picture) -> Patches {
        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        let (grid_h, grid_w) = (picture.height / p, picture.width / p);
        let width = picture.width;
        let value = |row: usize, col: usize, channel: usize| -> f32 {
            let byte = picture.rgb[(row * width + col) * 3 + channel];
            (byte as f32 / 255.0 - self.mean[channel]) / self.std[channel]
        };
        let patch_len = 3 * t * p * p;
        let mut values = Vec::with_capacity(grid_h * grid_w * patch_len);
        for block_row in 0..grid_h / m {
            for block_col in 0..grid_w / m {
                for inner_row in 0..m {
                    for inner_col in 0..m {
                        let (row, col) = (block_row * m + inner_row, block_col * m + inner_col);
                        for channel in 0..3 {
                            for _ in 0..t {
                                for y in 0..p {
                                    for x in 0..p {
                                        values.push(value(row * p + y, col * p + x, channel));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let grid = [1, grid_h, grid_w];
        let hash = fnv1a(
            (grid.iter().map(|&g| g as u32))
                .chain(values.iter().map(|v| v.to_bits()))
                .flat_map(u32::to_le_bytes),
        );
        Patches { values, grid, hash }
    }
}

/// What a server needs to take images with a prompt: how to prepare them, and the token a chat
/// template puts where each one goes.
pub struct ImageInput {
    pub preprocessor: Preprocessor,
    pub image_token: u32,
    /// How many patches a side of each of the language model's image tokens covers.
    pub merge: usize,
}

impl ImageInput {
    /// For a vision-language model; `None` for any other.
    pub fn load(dir: &Path, config: &crate::config::Config) -> Result<Option<Self>> {
        let Some(vision) = &config.vision_config else {
            return Ok(None);
        };
        Ok(Some(Self {
            preprocessor: Preprocessor::load(dir)?,
            image_token: config
                .image_token_id
                .context("config.json has no image_token_id")?,
            merge: vision.spatial_merge_size,
        }))
    }

    /// Read an image a request sent, as a data URL or bare base64, and cut it into patches.
    pub fn prepare(&self, source: &str) -> Result<Patches> {
        self.preprocessor.prepare(&decode(&image_bytes(source)?)?)
    }
}

/// The bytes of an image a request sent: a data URL (`data:image/png;base64,...`) or bare base64.
pub fn image_bytes(source: &str) -> Result<Vec<u8>> {
    let source = source.trim();
    let data = match source.strip_prefix("data:") {
        Some(url) => {
            let (kind, data) = url.split_once(',').context("a data URL needs a comma")?;
            ensure!(
                kind.ends_with(";base64"),
                "an image's data URL must be base64"
            );
            data
        }
        None if source.starts_with("http://") || source.starts_with("https://") => bail!(
            "images can't be fetched from URLs yet: send the image itself, as a data URL \
             (data:image/png;base64,...)"
        ),
        None => source,
    };
    base64_decode(data).context("the image isn't valid base64")
}

/// Decode base64 in either alphabet (standard or URL-safe), padded or not, ignoring whitespace.
pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut buffer, mut bits) = (0u32, 0);
    for (i, byte) in text.bytes().enumerate() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            // Padding ends the data.
            b'=' => {
                return text.as_bytes()[i..]
                    .iter()
                    .all(|b| b"= \t\r\n".contains(b))
                    .then_some(out);
            }
            _ => return None,
        };
        buffer = (buffer << 6 | value as u32) & 0xffff;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    Some(out)
}

/// An image, as 8-bit RGB, row by row.
#[derive(Clone)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Read a PNG, JPEG or WebP image.
pub fn decode(bytes: &[u8]) -> Result<Picture> {
    let image = image::load_from_memory(bytes).context("couldn't read the image")?;
    let rgb = image.to_rgb8();
    Ok(Picture {
        width: rgb.width() as usize,
        height: rgb.height() as usize,
        rgb: rgb.into_raw(),
    })
}

/// An image cut into patches for the vision encoder.
pub struct Patches {
    /// `[patches, 3 * temporal_patch_size * patch_size^2]`, in the encoder's order (see
    /// [`Preprocessor::patchify`]).
    pub values: Vec<f32>,
    /// How many patches the image is in time, height and width.
    pub grid: [usize; 3],
    /// Tells images apart, for the prefix cache.
    pub hash: u64,
}

impl Patches {
    /// How many tokens the language model reads for the image: one per block of patches.
    pub fn tokens(&self, merge: usize) -> usize {
        self.grid.iter().product::<usize>() / (merge * merge)
    }
}

/// The size to resize an image to, as transformers' Qwen2-VL processor works it out: both sides
/// a multiple of `factor`, the area between the limits, the shape as close as that allows.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(height > 0 && width > 0, "the image is empty");
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    ensure!(
        h.max(w) / h.min(w) <= 200.0,
        "the image is too long and thin (more than 200 to 1)"
    );
    // Python rounds halves to even.
    let round = |x: f64| x.round_ties_even();
    let mut h_bar = (round(h / f) * f).max(f);
    let mut w_bar = (round(w / f) * f).max(f);
    if h_bar * w_bar > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Resize with antialiased bicubic interpolation, as PIL (and torchvision, which copies it) does
/// for 8-bit images: across the rows first, then down the columns, rounding to bytes after each.
fn resize(picture: &Picture, width: usize, height: usize) -> Picture {
    let across = resample(
        &picture.rgb,
        picture.width,
        picture.height,
        width,
        |x, y| (y * picture.width + x) * 3,
        |x, y| (y * width + x) * 3,
    );
    let down = resample(
        &across,
        picture.height,
        width,
        height,
        |y, x| (y * width + x) * 3,
        |y, x| (y * width + x) * 3,
    );
    Picture {
        width,
        height,
        rgb: down,
    }
}

/// One pass of [`resize`] along one axis of `from`, which has `lines` lines of `size` pixels,
/// into `new_size` pixels a line. `at(i, line)` and `to(i, line)` give where pixel `i` of a line is
/// in the input and output.
fn resample(
    from: &[u8],
    size: usize,
    lines: usize,
    new_size: usize,
    at: impl Fn(usize, usize) -> usize,
    to: impl Fn(usize, usize) -> usize,
) -> Vec<u8> {
    // PIL's bicubic filter (a = -0.5), widened by the scale when shrinking.
    let cubic = |x: f64| {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = size as f64 / new_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let taps: Vec<(usize, Vec<f64>)> = (0..new_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let last = ((center + support + 0.5) as isize).min(size as isize) as usize;
            let mut weights: Vec<f64> = (first..last)
                .map(|j| cubic((j as f64 - center + 0.5) / filter_scale))
                .collect();
            let total: f64 = weights.iter().sum();
            if total != 0.0 {
                weights.iter_mut().for_each(|w| *w /= total);
            }
            (first, weights)
        })
        .collect();
    let mut out = vec![0u8; new_size * lines * 3];
    for line in 0..lines {
        for (i, (first, weights)) in taps.iter().enumerate() {
            for channel in 0..3 {
                let sum: f64 = (weights.iter().enumerate())
                    .map(|(k, w)| w * from[at(first + k, line) + channel] as f64)
                    .sum();
                out[to(i, line) + channel] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Where each window's blocks are once they're put window by window (each window's blocks in
/// rows, windows in rows), and where each window starts in patches, as transformers' Qwen2.5-VL
/// works them out: `order[i]` is the block that goes `i`-th.
pub fn window_order(
    grid: [usize; 3],
    merge: usize,
    window_blocks: usize,
) -> (Vec<usize>, Vec<usize>) {
    let [t, h, w] = grid;
    let (blocks_h, blocks_w) = (h / merge, w / merge);
    let windows_h = blocks_h.div_ceil(window_blocks);
    let windows_w = blocks_w.div_ceil(window_blocks);
    let unit = merge * merge;
    let (mut order, mut starts) = (Vec::new(), vec![0]);
    for frame in 0..t {
        for wy in 0..windows_h {
            for wx in 0..windows_w {
                let before = order.len();
                for by in wy * window_blocks..((wy + 1) * window_blocks).min(blocks_h) {
                    for bx in wx * window_blocks..((wx + 1) * window_blocks).min(blocks_w) {
                        order.push((frame * blocks_h + by) * blocks_w + bx);
                    }
                }
                if order.len() > before {
                    starts.push(order.len() * unit);
                }
            }
        }
    }
    (order, starts)
}

/// `0.5 x (1 + erf(x / sqrt 2))`: GELU without the tanh approximation.
fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

/// The error function, to about 1e-7 (Numerical Recipes' Chebyshev fit to erfc).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    match x >= 0.0 {
        true => 1.0 - erfc,
        false => erfc - 1.0,
    }
}

/// One layer of the vision encoder.
struct Block {
    norm1: Vec<f32>,
    qkv: Linear,
    proj: Linear,
    norm2: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// Qwen2.5-VL's vision encoder, on the CPU (see the module's documentation).
pub struct VisionEncoder {
    config: VisionConfig,
    patch_embed: Linear,
    blocks: Vec<Block>,
    merger_norm: Vec<f32>,
    merger_fc1: Linear,
    merger_fc2: Linear,
}

/// The vision encoder's RMS norms' epsilon.
pub const EPS: f32 = 1e-6;

impl VisionEncoder {
    pub fn load(st: &SafeTensors, config: &VisionConfig) -> Result<Self> {
        let c = config;
        let (d, inner) = (c.hidden_size, c.intermediate_size);
        let patch_len = c.patch_len();
        // The patch embedding is a convolution with as many weights as a patch has values.
        let name = "visual.patch_embed.proj.weight";
        let shape = [
            d,
            c.in_channels,
            c.temporal_patch_size,
            c.patch_size,
            c.patch_size,
        ];
        let patch_embed = match st.dtype(name) {
            Some("BF16") => Linear::from_part(&[], &st.load_bf16(name, &shape)?, d, patch_len),
            _ => Linear::from_part(&st.load(name, &shape)?, &[], d, patch_len),
        };
        let norm = |name: &str, len: usize| st.load(name, &[len]);
        let blocks = (0..c.depth)
            .map(|i| {
                let p = format!("visual.blocks.{i}");
                Ok(Block {
                    norm1: norm(&format!("{p}.norm1.weight"), d)?,
                    qkv: Linear::load(st, &format!("{p}.attn.qkv"), 3 * d, d)?,
                    proj: Linear::load(st, &format!("{p}.attn.proj"), d, d)?,
                    norm2: norm(&format!("{p}.norm2.weight"), d)?,
                    gate: Linear::load(st, &format!("{p}.mlp.gate_proj"), inner, d)?,
                    up: Linear::load(st, &format!("{p}.mlp.up_proj"), inner, d)?,
                    down: Linear::load(st, &format!("{p}.mlp.down_proj"), d, inner)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let merged = d * c.spatial_merge_size * c.spatial_merge_size;
        Ok(Self {
            config: config.clone(),
            patch_embed,
            blocks,
            merger_norm: norm("visual.merger.ln_q.weight", d)?,
            merger_fc1: Linear::load(st, "visual.merger.mlp.0", merged, merged)?,
            merger_fc2: Linear::load(st, "visual.merger.mlp.2", c.out_hidden_size, merged)?,
        })
    }

    pub fn num_params(&self) -> usize {
        self.merger_norm.len()
            + self.patch_embed.num_params()
            + self.merger_fc1.num_params()
            + self.merger_fc2.num_params()
            + (self.blocks.iter())
                .map(|b| {
                    b.qkv.num_params()
                        + b.proj.num_params()
                        + b.gate.num_params()
                        + b.up.num_params()
                        + b.down.num_params()
                        + 2 * b.norm1.len()
                })
                .sum::<usize>()
    }

    /// The image's tokens for the language model, `[tokens, out_hidden_size]`, in the order of
    /// its blocks of patches (rows of blocks, top to bottom).
    pub fn encode(&self, pool: &Pool, patches: &Patches) -> Result<Vec<f32>> {
        let c = &self.config;
        let (d, heads) = (c.hidden_size, c.num_heads);
        let head_dim = d / heads;
        let unit = c.spatial_merge_size * c.spatial_merge_size;
        let Layout {
            order,
            windows,
            input,
            cos,
            sin,
        } = layout(c, patches)?;
        let n = input.len() / c.patch_len();
        let mut scratch = Vec::new();
        let mut x = vec![0.0; n * d];
        self.patch_embed.forward(pool, &mut x, &mut scratch, &input);

        let whole = [0, n];
        let mut normed = vec![0.0; n * d];
        let mut qkv = vec![0.0; n * 3 * d];
        let (mut q, mut k, mut v) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        let mut attended = vec![0.0; n * d];
        let mut added = vec![0.0; n * d];
        let inner = c.intermediate_size;
        let (mut gate, mut up) = (vec![0.0; n * inner], vec![0.0; n * inner]);
        for (l, block) in self.blocks.iter().enumerate() {
            let segments: &[usize] = match c.fullatt_block_indexes.contains(&l) {
                true => &whole,
                false => &windows,
            };
            rmsnorm(pool, &mut normed, &x, &block.norm1, EPS);
            block.qkv.forward(pool, &mut qkv, &mut scratch, &normed);
            for i in 0..n {
                let row = &qkv[i * 3 * d..(i + 1) * 3 * d];
                q[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
                k[i * d..(i + 1) * d].copy_from_slice(&row[d..2 * d]);
                v[i * d..(i + 1) * d].copy_from_slice(&row[2 * d..]);
            }
            rope(pool, &mut q, d, head_dim, &cos, &sin);
            rope(pool, &mut k, d, head_dim, &cos, &sin);
            segment_attention(pool, &mut attended, (&q, &k, &v), heads, segments);
            block
                .proj
                .forward(pool, &mut added, &mut scratch, &attended);
            add_rows(pool, &mut x, &added, d);
            rmsnorm(pool, &mut normed, &x, &block.norm2, EPS);
            block.gate.forward(pool, &mut gate, &mut scratch, &normed);
            block.up.forward(pool, &mut up, &mut scratch, &normed);
            swiglu(pool, &mut gate, &up, inner);
            block.down.forward(pool, &mut added, &mut scratch, &gate);
            add_rows(pool, &mut x, &added, d);
        }

        // Each block of patches becomes one token.
        rmsnorm(pool, &mut normed, &x, &self.merger_norm, EPS);
        let tokens = n / unit;
        let merged = d * unit;
        let mut hidden = vec![0.0; tokens * merged];
        self.merger_fc1
            .forward(pool, &mut hidden, &mut scratch, &normed);
        hidden.iter_mut().for_each(|v| *v = gelu(*v));
        let out_dim = c.out_hidden_size;
        let mut out = vec![0.0; tokens * out_dim];
        self.merger_fc2
            .forward(pool, &mut out, &mut scratch, &hidden);
        Ok(unorder(&order, &out))
    }
}

/// How the encoder goes through an image: block by block in `order` (window by window), with
/// each patch's 2D RoPE rotation. The same for the GPUs.
pub struct Layout {
    /// The image's blocks of patches, in the order the encoder takes them.
    pub order: Vec<usize>,
    /// Where each window starts, in patches, then where the last ends.
    pub windows: Vec<usize>,
    /// The patches' values in that order, `[patches, patch_len]`.
    pub input: Vec<f32>,
    /// Each patch's rotations, `[patches, head_dim / 2]`: the first quarter of each head's
    /// frequencies turn with the patch's row, the second with its column (then the same again,
    /// for the rotation's other half).
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl VisionConfig {
    /// How many values a patch has.
    pub fn patch_len(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// Work out how the encoder goes through `patches` (see [`Layout`]).
pub fn layout(c: &VisionConfig, patches: &Patches) -> Result<Layout> {
    let head_dim = c.hidden_size / c.num_heads;
    let m = c.spatial_merge_size;
    let unit = m * m;
    let patch_len = c.patch_len();
    let n = patches.grid.iter().product::<usize>();
    ensure!(
        patches.values.len() == n * patch_len && n % unit == 0,
        "the patches don't match the vision encoder"
    );
    ensure!(
        head_dim.is_multiple_of(4),
        "vision heads of {head_dim} don't split in four for 2D RoPE"
    );
    let (order, windows) = window_order(patches.grid, m, c.window_size / m / c.patch_size);
    let reorder = |values: &[f32], width: usize| -> Vec<f32> {
        let block = unit * width;
        (order.iter())
            .flat_map(|&b| &values[b * block..(b + 1) * block])
            .copied()
            .collect()
    };
    let quarter = head_dim / 4;
    let inv_freq: Vec<f64> = (0..quarter)
        .map(|k| 1.0 / 10000f64.powf((2 * k) as f64 / (head_dim / 2) as f64))
        .collect();
    let [frames, grid_h, grid_w] = patches.grid;
    let mut positions = Vec::with_capacity(n * 2);
    for _ in 0..frames {
        for by in 0..grid_h / m {
            for bx in 0..grid_w / m {
                for iy in 0..m {
                    for ix in 0..m {
                        positions.push((by * m + iy) as f32);
                        positions.push((bx * m + ix) as f32);
                    }
                }
            }
        }
    }
    let positions = reorder(&positions, 2);
    let half = head_dim / 2;
    let (mut cos, mut sin) = (vec![0.0; n * half], vec![0.0; n * half]);
    for i in 0..n {
        for j in 0..half {
            let (axis, k) = (j / quarter, j % quarter);
            let angle = positions[i * 2 + axis] as f64 * inv_freq[k];
            cos[i * half + j] = angle.cos() as f32;
            sin[i * half + j] = angle.sin() as f32;
        }
    }
    let input = reorder(&patches.values, patch_len);
    Ok(Layout {
        order,
        windows,
        input,
        cos,
        sin,
    })
}

/// The encoder's output, one row a block in window order, put back in the image's order.
pub fn unorder(order: &[usize], out: &[f32]) -> Vec<f32> {
    let width = out.len() / order.len().max(1);
    let mut result = vec![0.0; out.len()];
    for (i, &b) in order.iter().enumerate() {
        result[b * width..(b + 1) * width].copy_from_slice(&out[i * width..(i + 1) * width]);
    }
    result
}

/// Attention with no causal mask, each row seeing the rows of its own segment: `segments` holds
/// where each starts, then where the last ends.
fn segment_attention(
    pool: &Pool,
    out: &mut [f32],
    (q, k, v): (&[f32], &[f32], &[f32]),
    heads: usize,
    segments: &[usize],
) {
    let d = q.len() / (segments[segments.len() - 1].max(1));
    let head_dim = d / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    pool.for_chunks(out, d, 1, |i, out| {
        let s = segments.partition_point(|&start| start <= i) - 1;
        let (first, last) = (segments[s], segments[s + 1]);
        with_scratch(last - first, |scores| {
            for h in 0..heads {
                let query = &q[i * d + h * head_dim..][..head_dim];
                let mut best = f32::NEG_INFINITY;
                for (j, score) in (first..last).zip(scores.iter_mut()) {
                    let key = &k[j * d + h * head_dim..][..head_dim];
                    *score = crate::ops::dot(query, key) * scale;
                    best = best.max(*score);
                }
                let mut total = 0.0;
                for score in scores.iter_mut() {
                    *score = (*score - best).exp();
                    total += *score;
                }
                let to = &mut out[h * head_dim..(h + 1) * head_dim];
                to.fill(0.0);
                for (j, &weight) in (first..last).zip(scores.iter()) {
                    let value = &v[j * d + h * head_dim..][..head_dim];
                    for (o, &x) in to.iter_mut().zip(value) {
                        *o += weight * x;
                    }
                }
                to.iter_mut().for_each(|o| *o /= total);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_transformers() {
        // A size already in whole blocks stays; others round to the nearest block.
        assert_eq!(
            smart_resize(448, 336, 28, 3136, 12845056).unwrap(),
            (448, 336)
        );
        assert_eq!(
            smart_resize(500, 375, 28, 3136, 12845056).unwrap(),
            (504, 364)
        );
        // Too many pixels shrinks it, keeping its shape as far as blocks allow.
        assert_eq!(
            smart_resize(4000, 3000, 28, 3136, 1003520).unwrap(),
            (1148, 840)
        );
        // Too few grows it.
        assert_eq!(smart_resize(20, 20, 28, 3136, 12845056).unwrap(), (56, 56));
    }

    #[test]
    fn windows_cover_every_block_once() {
        let (order, starts) = window_order([1, 20, 12], 2, 4);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..60).collect::<Vec<_>>());
        // 10 x 6 blocks in windows of 4 x 4: 3 x 2 windows, the last row and column partial.
        assert_eq!(starts.len(), 7);
        assert_eq!(*starts.last().unwrap(), 240);
        assert_eq!(&order[..5], &[0, 1, 2, 3, 6]);
    }

    #[test]
    fn base64_reads_both_alphabets() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8h").unwrap(), b"hello!");
        assert_eq!(base64_decode("-_8=").unwrap(), [0xfb, 0xff]);
        assert_eq!(base64_decode("+/8=").unwrap(), [0xfb, 0xff]);
        assert!(base64_decode("aGV*").is_none());
        assert!(base64_decode("aG=Vs").is_none());
        let url = "data:image/png;base64,aGVsbG8=";
        assert_eq!(image_bytes(url).unwrap(), b"hello");
        assert!(image_bytes("https://example.com/cat.png").is_err());
        assert!(image_bytes("data:image/png,hello").is_err());
    }

    #[test]
    fn erf_is_accurate() {
        for (x, expected) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (-2.0, -0.995_322_265),
        ] {
            assert!((erf(x) - expected).abs() < 2e-7, "erf({x}) = {}", erf(x));
        }
    }
}
