//! Byte-pair-encoding (BPE) tokenizer that reads a Hugging Face `tokenizer.json`.
//!
//! Encoding runs the same stages as the `tokenizers` library, limited to the options that
//! Llama-family models use:
//!
//! 1. split out added tokens such as `<|im_start|>`, which are matched literally;
//! 2. normalize the rest (SentencePiece-style models turn `' '` into `'▁'`);
//! 3. pre-tokenize into words (regex and digit splitting, GPT-2's byte-to-character mapping);
//! 4. BPE each word: start from single characters and keep applying the highest-priority merge
//!    from the `merges` list until none applies;
//! 5. add the special tokens the post-processor asks for (e.g. `<s>` at the start).
//!
//! Decoding turns each token back into raw bytes. Byte-level vocabularies (GPT-2, SmolLM, Qwen,
//! Llama 3) write bytes as printable stand-in characters; SentencePiece-style vocabularies
//! (Llama 2, TinyLlama, Mistral) use `'▁'` for spaces and `<0xNN>` tokens for raw bytes.

use anyhow::{Context, Result, bail};
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

/// GPT-2's word-splitting regex, used by `ByteLevel` pre-tokenizers with `use_regex: true`.
const GPT2_SPLIT: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

enum Normalizer {
    Prepend(String),
    Replace {
        from: String,
        to: String,
    },
    /// Unicode canonical composition, e.g. "e" + combining accent becomes "é".
    Nfc,
}

enum PreTokenizer {
    /// Split around regex matches; matches become their own pieces unless removed.
    Split { regex: Regex, keep_matches: bool },
    /// Split digits from other text, either one digit per piece or in runs.
    Digits { individual: bool },
    /// GPT-2 style: optionally split with [`GPT2_SPLIT`], then map each byte to a character.
    ByteLevel {
        add_prefix_space: bool,
        regex: Option<Regex>,
    },
    /// SentencePiece style: replace spaces with `replacement` and maybe prepend one.
    Metaspace {
        replacement: char,
        prepend: Prepend,
        split: bool,
    },
}

#[derive(PartialEq)]
enum Prepend {
    Always,
    First,
    Never,
}

enum Segment<'a> {
    Text(&'a str),
    Added(u32),
}

pub struct Tokenizer {
    /// BPE vocabulary: token string to id.
    vocab: HashMap<String, u32>,
    /// Id to token string, covering the vocabulary and the added tokens.
    tokens: Vec<String>,
    /// Which ids are special tokens (never shown to the user).
    special: Vec<bool>,
    /// `(left, right)` to `(rank, merged)`. Lower ranks were learned earlier and win.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Tokens matched verbatim in the input, longest first.
    added: Vec<(String, u32)>,
    normalizers: Vec<Normalizer>,
    pre_tokenizers: Vec<PreTokenizer>,
    /// Byte-level vocabulary (true) or SentencePiece-style (false), for decoding.
    byte_level: bool,
    /// Leading spaces [`Self::decode`] removes: SentencePiece-style tokenizers put a `'▁'` in
    /// front of the text when encoding, and take it off again when decoding.
    strip_leading_spaces: usize,
    /// `<0xNN>` ids used when a character isn't in a SentencePiece-style vocabulary.
    byte_fallback: [Option<u32>; 256],
    unk: Option<u32>,
    fuse_unk: bool,
    /// Llama 3: if a whole word is already a token, use it without running BPE.
    ignore_merges: bool,
    /// Special tokens to put around an encoded text, e.g. `<s>` (beginning of sequence).
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let json: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &Value) -> Result<Self> {
        let model = &json["model"];
        if model["type"].as_str() != Some("BPE") {
            bail!(
                "tokenizer type {} is not supported (only BPE)",
                model["type"]
            );
        }

        let vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .context("tokenizer.json has no model.vocab")?
            .iter()
            .map(|(token, id)| Ok((token.clone(), id.as_u64().context("bad token id")? as u32)))
            .collect::<Result<_>>()?;

        let mut added = Vec::new();
        let mut special_ids = Vec::new();
        for token in json["added_tokens"].as_array().into_iter().flatten() {
            let id = token["id"].as_u64().context("added token without id")? as u32;
            let content = token["content"]
                .as_str()
                .context("added token without content")?;
            added.push((content.to_string(), id));
            if token["special"].as_bool().unwrap_or(false) {
                special_ids.push(id);
            }
        }
        added.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));

        let n_tokens = vocab
            .values()
            .chain(added.iter().map(|(_, id)| id))
            .max()
            .map_or(0, |&id| id as usize + 1);
        let mut tokens = vec![String::new(); n_tokens];
        for (token, &id) in &vocab {
            tokens[id as usize] = token.clone();
        }
        for (token, id) in &added {
            tokens[*id as usize] = token.clone();
        }
        let mut special = vec![false; n_tokens];
        for id in special_ids {
            special[id as usize] = true;
        }

        // Merges are listed in priority order, as "left right" strings or ["left", "right"] pairs.
        let mut merges = HashMap::new();
        for (rank, merge) in model["merges"].as_array().into_iter().flatten().enumerate() {
            let (left, right) = match merge {
                Value::String(s) => s.split_once(' ').context("bad merge")?,
                Value::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().context("bad merge")?,
                    pair[1].as_str().context("bad merge")?,
                ),
                _ => bail!("bad merge {merge}"),
            };
            let ids = (
                vocab.get(left),
                vocab.get(right),
                vocab.get(&format!("{left}{right}")),
            );
            if let (Some(&l), Some(&r), Some(&merged)) = ids {
                merges.entry((l, r)).or_insert((rank as u32, merged));
            }
        }

        let mut byte_fallback = [None; 256];
        if model["byte_fallback"].as_bool().unwrap_or(false) {
            for (byte, slot) in byte_fallback.iter_mut().enumerate() {
                *slot = vocab.get(&format!("<0x{byte:02X}>")).copied();
            }
        }

        let mut normalizers = Vec::new();
        parse_normalizer(&json["normalizer"], &mut normalizers)?;
        let mut pre_tokenizers = Vec::new();
        parse_pre_tokenizer(&json["pre_tokenizer"], &mut pre_tokenizers)?;
        let (mut prefix, mut suffix) = (Vec::new(), Vec::new());
        parse_post_processor(&json["post_processor"], &mut prefix, &mut suffix);

        let byte_to_char = byte_to_char_table();
        let char_to_byte = (0..=255u8).map(|b| (byte_to_char[b as usize], b)).collect();

        Ok(Self {
            unk: model["unk_token"]
                .as_str()
                .and_then(|unk| vocab.get(unk).copied()),
            fuse_unk: model["fuse_unk"].as_bool().unwrap_or(false),
            ignore_merges: model["ignore_merges"].as_bool().unwrap_or(false),
            byte_level: has_type(&json["decoder"], "ByteLevel"),
            strip_leading_spaces: leading_spaces_to_strip(&json["decoder"]),
            vocab,
            tokens,
            special,
            merges,
            added,
            normalizers,
            pre_tokenizers,
            byte_fallback,
            prefix,
            suffix,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Text to token ids. `add_special_tokens` wraps the result in e.g. a `<s>` token if the
    /// tokenizer's post-processor asks for it.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special_tokens {
            ids.extend_from_slice(&self.prefix);
        }
        for (i, segment) in self.split_added_tokens(text).into_iter().enumerate() {
            match segment {
                Segment::Added(id) => ids.push(id),
                Segment::Text(text) => {
                    for word in self.pre_tokenize(self.normalize(text), i == 0) {
                        self.bpe(&word, &mut ids);
                    }
                }
            }
        }
        if add_special_tokens {
            ids.extend_from_slice(&self.suffix);
        }
        ids
    }

    /// Token ids to text. Invalid UTF-8 (e.g. half of a multi-byte character) becomes U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id)).collect();
        let text = String::from_utf8_lossy(&bytes);
        let mut text: &str = &text;
        for _ in 0..self.strip_leading_spaces {
            text = text.strip_prefix(' ').unwrap_or(text);
        }
        text.to_string()
    }

    /// The raw bytes one token stands for. A multi-byte character can be split across several
    /// tokens, so when streaming, buffer bytes until they form valid UTF-8.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(token) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.special[id as usize] {
            token.as_bytes().to_vec()
        } else if self.byte_level {
            let mut bytes = Vec::with_capacity(token.len());
            for c in token.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                }
            }
            bytes
        } else if let Some(byte) = parse_byte_token(token) {
            vec![byte]
        } else {
            token.replace('▁', " ").into_bytes()
        }
    }

    /// How many leading spaces `decode` removes (for tokenizers that mark word boundaries).
    pub fn leading_spaces_to_strip(&self) -> usize {
        self.strip_leading_spaces
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// Look up a token by its exact string, e.g. `"<|im_end|>"`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        let added = self.added.iter().find(|(content, _)| content == token);
        added
            .map(|&(_, id)| id)
            .or_else(|| self.vocab.get(token).copied())
    }

    /// Special tokens [`Self::encode`] puts in front of a text (usually just a BOS token, if any).
    pub fn prefix_tokens(&self) -> &[u32] {
        &self.prefix
    }

    fn split_added_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let (mut start, mut i) = (0, 0);
        while i < text.len() {
            let rest = &text[i..];
            let found = self
                .added
                .iter()
                .find(|(content, _)| !content.is_empty() && rest.starts_with(content.as_str()));
            if let Some((content, id)) = found {
                if start < i {
                    segments.push(Segment::Text(&text[start..i]));
                }
                segments.push(Segment::Added(*id));
                i += content.len();
                start = i;
            } else {
                i += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        if start < text.len() {
            segments.push(Segment::Text(&text[start..]));
        }
        segments
    }

    fn normalize(&self, text: &str) -> String {
        let mut text = text.to_string();
        for normalizer in &self.normalizers {
            match normalizer {
                Normalizer::Prepend(prefix) if !text.is_empty() => text.insert_str(0, prefix),
                Normalizer::Prepend(_) => {}
                Normalizer::Replace { from, to } => text = text.replace(from.as_str(), to),
                Normalizer::Nfc => text = text.nfc().collect(),
            }
        }
        text
    }

    /// Split normalized text into the words BPE runs on. `first_segment` is true when the text
    /// starts at the beginning of the input (not after an added token).
    fn pre_tokenize(&self, text: String, first_segment: bool) -> Vec<String> {
        let mut pieces = vec![text];
        for pre_tokenizer in &self.pre_tokenizers {
            let mut next = Vec::with_capacity(pieces.len());
            for (j, piece) in pieces.into_iter().enumerate() {
                match pre_tokenizer {
                    PreTokenizer::Split {
                        regex,
                        keep_matches,
                    } => split_regex(regex, &piece, *keep_matches, &mut next),
                    PreTokenizer::Digits { individual } => {
                        split_digits(&piece, *individual, &mut next)
                    }
                    PreTokenizer::ByteLevel {
                        add_prefix_space,
                        regex,
                    } => {
                        let piece = match *add_prefix_space && !piece.starts_with(' ') {
                            true => format!(" {piece}"),
                            false => piece,
                        };
                        let mut words = Vec::new();
                        match regex {
                            Some(regex) => split_regex(regex, &piece, true, &mut words),
                            None => words.push(piece),
                        }
                        let to_chars = |w: &String| -> String {
                            w.bytes().map(|b| self.byte_to_char[b as usize]).collect()
                        };
                        next.extend(words.iter().map(to_chars));
                    }
                    PreTokenizer::Metaspace {
                        replacement,
                        prepend,
                        split,
                    } => {
                        let mut s = piece.replace(' ', &replacement.to_string());
                        let at_start = first_segment && j == 0;
                        let wants_prefix =
                            *prepend == Prepend::Always || (*prepend == Prepend::First && at_start);
                        if wants_prefix && !s.starts_with(*replacement) {
                            s.insert(0, *replacement);
                        }
                        match split {
                            true => split_before(&s, *replacement, &mut next),
                            false => next.push(s),
                        }
                    }
                }
            }
            pieces = next;
        }
        pieces
    }

    /// Byte-pair encoding of one word, appending its token ids to `out`.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(word)
        {
            out.push(id);
            return;
        }

        // Start with one symbol per character.
        let mut symbols: Vec<u32> = Vec::with_capacity(word.len());
        let mut buf = [0u8; 4];
        for c in word.chars() {
            let c = c.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(&*c) {
                symbols.push(id);
            } else if c.bytes().all(|b| self.byte_fallback[b as usize].is_some()) {
                symbols.extend(c.bytes().map(|b| self.byte_fallback[b as usize].unwrap()));
            } else if let Some(unk) = self.unk
                && !(self.fuse_unk && symbols.last() == Some(&unk))
            {
                symbols.push(unk);
            }
        }

        // Repeatedly merge the adjacent pair with the best (lowest) rank.
        loop {
            let mut best: Option<(u32, usize, u32)> = None; // (rank, position, merged id)
            for (i, pair) in symbols.windows(2).enumerate() {
                if let Some(&(rank, merged)) = self.merges.get(&(pair[0], pair[1]))
                    && best.is_none_or(|(best_rank, _, _)| rank < best_rank)
                {
                    best = Some((rank, i, merged));
                }
            }
            let Some((_, i, merged)) = best else { break };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
        out.extend_from_slice(&symbols);
    }
}

fn parse_normalizer(config: &Value, out: &mut Vec<Normalizer>) -> Result<()> {
    match config["type"].as_str() {
        None => {}
        Some("Sequence") => {
            for normalizer in config["normalizers"].as_array().into_iter().flatten() {
                parse_normalizer(normalizer, out)?;
            }
        }
        Some("Prepend") => out.push(Normalizer::Prepend(
            config["prepend"].as_str().context("Prepend")?.into(),
        )),
        Some("Replace") => out.push(Normalizer::Replace {
            from: config["pattern"]["String"]
                .as_str()
                .context("only string Replace patterns are supported")?
                .into(),
            to: config["content"]
                .as_str()
                .context("Replace without content")?
                .into(),
        }),
        Some("NFC") => out.push(Normalizer::Nfc),
        Some(other) => bail!("normalizer {other} is not supported"),
    }
    Ok(())
}

fn parse_pre_tokenizer(config: &Value, out: &mut Vec<PreTokenizer>) -> Result<()> {
    match config["type"].as_str() {
        None => {}
        Some("Sequence") => {
            for pre_tokenizer in config["pretokenizers"].as_array().into_iter().flatten() {
                parse_pre_tokenizer(pre_tokenizer, out)?;
            }
        }
        Some("Split") => {
            let pattern = match (
                config["pattern"]["Regex"].as_str(),
                config["pattern"]["String"].as_str(),
            ) {
                (Some(regex), _) => regex.to_string(),
                (None, Some(literal)) => regex_escape(literal),
                _ => bail!("Split pre-tokenizer without a pattern"),
            };
            let keep_matches = match config["behavior"].as_str() {
                Some("Isolated") => true,
                Some("Removed") => false,
                other => bail!("Split behavior {other:?} is not supported"),
            };
            if config["invert"].as_bool() == Some(true) {
                bail!("inverted Split pre-tokenizers are not supported");
            }
            out.push(PreTokenizer::Split {
                regex: Regex::new(&pattern)?,
                keep_matches,
            });
        }
        Some("Digits") => out.push(PreTokenizer::Digits {
            individual: config["individual_digits"].as_bool().unwrap_or(false),
        }),
        Some("ByteLevel") => out.push(PreTokenizer::ByteLevel {
            add_prefix_space: config["add_prefix_space"].as_bool().unwrap_or(false),
            regex: match config["use_regex"].as_bool().unwrap_or(true) {
                true => Some(Regex::new(GPT2_SPLIT)?),
                false => None,
            },
        }),
        Some("Metaspace") => out.push(PreTokenizer::Metaspace {
            replacement: config["replacement"]
                .as_str()
                .and_then(|s| s.chars().next())
                .unwrap_or('▁'),
            prepend: match config["prepend_scheme"].as_str() {
                Some("always") => Prepend::Always,
                Some("first") => Prepend::First,
                Some("never") => Prepend::Never,
                _ => match config["add_prefix_space"].as_bool().unwrap_or(true) {
                    true => Prepend::Always,
                    false => Prepend::Never,
                },
            },
            split: config["split"].as_bool().unwrap_or(true),
        }),
        Some(other) => bail!("pre-tokenizer {other} is not supported"),
    }
    Ok(())
}

/// Collect the special tokens a `TemplateProcessing` post-processor puts before and after a
/// single sequence.
fn parse_post_processor(config: &Value, prefix: &mut Vec<u32>, suffix: &mut Vec<u32>) {
    match config["type"].as_str() {
        Some("Sequence") => {
            for processor in config["processors"].as_array().into_iter().flatten() {
                parse_post_processor(processor, prefix, suffix);
            }
        }
        Some("TemplateProcessing") => {
            let mut after_text = false;
            for piece in config["single"].as_array().into_iter().flatten() {
                if piece.get("Sequence").is_some() {
                    after_text = true;
                } else if let Some(name) = piece["SpecialToken"]["id"].as_str() {
                    let ids = config["special_tokens"][name]["ids"]
                        .as_array()
                        .into_iter()
                        .flatten();
                    let ids = ids.filter_map(|id| id.as_u64()).map(|id| id as u32);
                    match after_text {
                        true => suffix.extend(ids),
                        false => prefix.extend(ids),
                    }
                }
            }
        }
        _ => {}
    }
}

/// How many leading spaces the decoder removes from decoded text.
fn leading_spaces_to_strip(decoder: &Value) -> usize {
    match decoder["type"].as_str() {
        Some("Strip") if decoder["content"] == " " => {
            decoder["start"].as_u64().unwrap_or(0) as usize
        }
        Some("Metaspace") => {
            let prepends =
                decoder["prepend_scheme"] != "never" && decoder["add_prefix_space"] != false;
            prepends as usize
        }
        Some("Sequence") => decoder["decoders"]
            .as_array()
            .into_iter()
            .flatten()
            .map(leading_spaces_to_strip)
            .sum(),
        _ => 0,
    }
}

/// Does this component, or any component nested in a `Sequence`, have the given type?
fn has_type(config: &Value, kind: &str) -> bool {
    config["type"].as_str() == Some(kind)
        || ["decoders", "pretokenizers", "processors"]
            .iter()
            .any(|key| {
                config[key]
                    .as_array()
                    .is_some_and(|items| items.iter().any(|c| has_type(c, kind)))
            })
}

/// GPT-2's reversible mapping from bytes to printable characters: printable Latin-1 bytes map
/// to themselves and the rest (control characters, space, ...) to U+0100 onwards, which is why
/// a leading space shows up as `Ġ` in byte-level vocabularies.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256;
    for b in 0..=255u8 {
        table[b as usize] = if matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF) {
            b as char
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    table
}

/// `"<0x0A>"` -> `Some(0x0A)`
fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Split around regex matches, keeping the text between them (and the matches if asked).
fn split_regex(regex: &Regex, text: &str, keep_matches: bool, out: &mut Vec<String>) {
    let mut last = 0;
    for m in regex.find_iter(text).filter_map(Result::ok) {
        if m.start() > last {
            out.push(text[last..m.start()].to_string());
        }
        if keep_matches && !m.as_str().is_empty() {
            out.push(m.as_str().to_string());
        }
        last = m.end();
    }
    if last < text.len() {
        out.push(text[last..].to_string());
    }
}

/// Separate numeric characters from the rest: `"abc123"` becomes `["abc", "1", "2", "3"]`
/// (or `["abc", "123"]` when not `individual`).
fn split_digits(text: &str, individual: bool, out: &mut Vec<String>) {
    let mut current = String::new();
    let mut current_is_digit = false;
    for c in text.chars() {
        let is_digit = c.is_numeric();
        if !current.is_empty() && (is_digit != current_is_digit || (is_digit && individual)) {
            out.push(std::mem::take(&mut current));
        }
        current.push(c);
        current_is_digit = is_digit;
    }
    if !current.is_empty() {
        out.push(current);
    }
}

/// Start a new piece at every `delimiter`: `"▁a▁b"` becomes `["▁a", "▁b"]`.
fn split_before(text: &str, delimiter: char, out: &mut Vec<String>) {
    let mut current = String::new();
    for c in text.chars() {
        if c == delimiter && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push(c);
    }
    if !current.is_empty() {
        out.push(current);
    }
}

fn regex_escape(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len());
    for c in literal.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A toy byte-level tokenizer: single characters plus a few merges.
    fn toy() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, token) in [
            "h", "e", "l", "o", "Ġ", "w", "r", "d", "he", "ll", "hell", "hello", "Ġw",
        ]
        .iter()
        .enumerate()
        {
            vocab.insert(token.to_string(), json!(i));
        }
        vocab.insert("<|end|>".into(), json!(13));
        Tokenizer::from_json(&json!({
            "added_tokens": [{"id": 13, "content": "<|end|>", "special": true}],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": ["h e", "l l", "he ll", ["hell", "o"], "Ġ w"],
            },
        }))
        .unwrap()
    }

    #[test]
    fn byte_to_char_matches_gpt2() {
        let table = byte_to_char_table();
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        assert_eq!(table[b'a' as usize], 'a');
    }

    #[test]
    fn encodes_with_merges_and_added_tokens() {
        let tok = toy();
        // "hello" merges all the way; " world" splits into "Ġw" + leftover characters.
        let ids = tok.encode("hello world<|end|>", false);
        assert_eq!(ids, vec![11, 12, 3, 6, 2, 7, 13]);
        assert_eq!(tok.decode(&ids), "hello world<|end|>");
        assert!(tok.is_special(13));
    }

    #[test]
    fn splits_digits() {
        let mut out = Vec::new();
        split_digits("ab12c", true, &mut out);
        assert_eq!(out, ["ab", "1", "2", "c"]);
        out.clear();
        split_digits("ab12c", false, &mut out);
        assert_eq!(out, ["ab", "12", "c"]);
    }

    #[test]
    fn parses_byte_tokens() {
        assert_eq!(parse_byte_token("<0x0A>"), Some(10));
        assert_eq!(parse_byte_token("<0xFF>"), Some(255));
        assert_eq!(parse_byte_token("<s>"), None);
    }
}
