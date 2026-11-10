//! Reasoning models think out loud before they answer, between markers such as `<think>` and
//! `</think>`. This separates the thinking from the answer, as SGLang's reasoning parsers do, so
//! the API can return it as `reasoning_content` and look for tool calls only in the answer.
//!
//! Whether a reply starts out thinking depends on the prompt. Some templates open the block
//! themselves (DeepSeek-R1, Qwen3's "thinking" models), some leave it to the model (Qwen3), and
//! some close an empty one to turn thinking off (Qwen3 with `enable_thinking: false`). Rather
//! than know each model, [`ReasoningFormat::opened_by`] looks at how the prompt ends.
//!
//! GPT-OSS writes its replies in OpenAI's "harmony" format instead: a series of messages, each
//! `<|channel|>NAME<|message|>TEXT<|end|>`, where the `analysis` channel is its thinking and the
//! `final` channel its answer.

use crate::tokenizer::Tokenizer;

/// The markers around a model's thinking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningFormat {
    pub start: String,
    pub end: String,
    /// Replies are harmony messages (GPT-OSS); `start` and `end` mark where thinking begins
    /// and where the answer does.
    pub harmony: bool,
}

/// The special tokens harmony messages are made of, which have to stay in the text to be read.
const HARMONY_MARKERS: [&str; 5] = [
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|constrain|>",
];

impl ReasoningFormat {
    /// `<think>` ... `</think>`, which nearly every open reasoning model uses.
    pub fn think() -> Self {
        Self {
            start: "<think>".to_string(),
            end: "</think>".to_string(),
            harmony: false,
        }
    }

    /// OpenAI's harmony format (GPT-OSS): the thinking is the `analysis` channel's messages,
    /// and the answer starts with the `final` channel's.
    pub fn harmony() -> Self {
        Self {
            start: "<|channel|>analysis<|message|>".to_string(),
            end: "<|channel|>final<|message|>".to_string(),
            harmony: true,
        }
    }

    /// The format a model thinks in, judging by its chat template and vocabulary: both need to
    /// know about the end marker.
    pub fn detect(template: &str, tokenizer: &Tokenizer) -> Option<Self> {
        if tokenizer.token_id("<|channel|>").is_some() && template.contains("<|channel|>") {
            return Some(Self::harmony());
        }
        let format = Self::think();
        let known = tokenizer.token_id(&format.end).is_some() && template.contains(&format.end);
        known.then_some(format)
    }

    /// Whether a rendered prompt leaves a block of thinking open, so that the reply starts in it.
    /// A harmony reply always starts by thinking (it can skip straight to its answer, which
    /// counts as the thinking ending at once).
    pub fn opened_by(&self, prompt: &str) -> bool {
        self.harmony
            || prompt
                .rfind(&self.start)
                .is_some_and(|at| !prompt[at..].contains(&self.end))
    }

    /// The token that opens a block, if the marker is a single token.
    pub fn start_token(&self, tokenizer: &Tokenizer) -> Option<u32> {
        tokenizer.token_id(&self.start)
    }

    /// The token that closes a block, if the marker is a single token.
    pub fn end_token(&self, tokenizer: &Tokenizer) -> Option<u32> {
        tokenizer.token_id(&self.end)
    }

    /// Special tokens that have to be kept in the text of a reply for it to be read.
    pub fn marker_tokens(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        match self.harmony {
            true => HARMONY_MARKERS
                .iter()
                .filter_map(|marker| tokenizer.token_id(marker))
                .collect(),
            false => Vec::new(),
        }
    }
}

/// A piece of a reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Nothing but whitespace yet, so the reply could still open a block.
    Start,
    /// Inside the block. `started` once it has written something other than whitespace.
    Reasoning { started: bool },
    /// The answer. `trim` while whitespace left over from the end of a block should go.
    Content { trim: bool },
}

/// Splits a reply into thinking and answer as it streams in. Whitespace next to the markers is
/// dropped, so the pieces read the same whichever way the text was cut up.
pub struct ReasoningParser {
    format: ReasoningFormat,
    state: State,
    /// Text that can't be placed yet: it might be part of a marker, or whitespace that may or
    /// may not end up mattering.
    held: String,
    /// For harmony replies, the kind of message being read, or `None` between messages.
    message: Option<Channel>,
}

/// What a harmony message is, by its channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    /// `analysis`: thinking.
    Analysis,
    /// `final`, or `commentary` with no recipient: the answer.
    Answer,
    /// Addressed to a tool, which isn't read as a call yet, so it's left out.
    ToTool,
}

impl ReasoningParser {
    /// `open` if the prompt left a block open, so the reply starts inside it.
    pub fn new(format: ReasoningFormat, open: bool) -> Self {
        Self {
            format,
            state: match open {
                true => State::Reasoning { started: false },
                false => State::Start,
            },
            held: String::new(),
            message: None,
        }
    }

    /// Whether the reply is in its block of thinking right now.
    pub fn is_reasoning(&self) -> bool {
        match self.format.harmony {
            true => self.message == Some(Channel::Analysis),
            false => matches!(self.state, State::Reasoning { .. }),
        }
    }

    /// Add more of a harmony reply: each message's text goes where its channel says.
    fn push_harmony(&mut self, text: &str) -> Vec<Piece> {
        const MESSAGE: &str = "<|message|>";
        const END: &str = "<|end|>";
        self.held.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.message {
                // A header: `<|start|>assistant<|channel|>NAME to=RECIPIENT ...<|message|>`.
                None => {
                    let Some(at) = self.held.find(MESSAGE) else {
                        return out;
                    };
                    let header: String = self.held.drain(..at + MESSAGE.len()).collect();
                    let channel = header
                        .split("<|channel|>")
                        .nth(1)
                        .and_then(|rest| rest.split(|c: char| c.is_whitespace() || c == '<').next())
                        .unwrap_or("final");
                    self.message = Some(match channel {
                        "analysis" => Channel::Analysis,
                        _ if header.contains(" to=") => Channel::ToTool,
                        _ => Channel::Answer,
                    });
                }
                Some(channel) => {
                    let (text, ended) = match self.held.find(END) {
                        Some(at) => {
                            let text: String = self.held.drain(..at).collect();
                            self.held.drain(..END.len());
                            (text, true)
                        }
                        None => {
                            let keep = crate::engine::partial_match_len(&self.held, END);
                            let ready = self.held.len() - keep;
                            (self.held.drain(..ready).collect(), false)
                        }
                    };
                    emit_message(&mut out, channel, text);
                    if !ended {
                        return out;
                    }
                    self.message = None;
                }
            }
        }
    }

    /// Add more of the reply, getting back whatever can be placed.
    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        if self.format.harmony {
            return self.push_harmony(text);
        }
        self.held.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Start => {
                    let rest = self.held.trim_start();
                    if rest.starts_with(&self.format.start) {
                        let after = rest[self.format.start.len()..].to_string();
                        self.held = after;
                        self.state = State::Reasoning { started: false };
                        continue;
                    }
                    if rest.is_empty() || self.format.start.starts_with(rest) {
                        return out; // still might be the opening marker
                    }
                    self.state = State::Content { trim: false };
                }
                State::Reasoning { started } => {
                    if let Some(at) = self.held.find(&self.format.end) {
                        let text = self.held[..at].to_string();
                        self.held.drain(..at + self.format.end.len());
                        emit_reasoning(&mut out, &text, started, true);
                        self.state = State::Content { trim: true };
                        continue;
                    }
                    // Hold back what might be the start of the end marker, and trailing
                    // whitespace, which only counts if more thinking follows it.
                    let keep = crate::engine::partial_match_len(&self.held, &self.format.end);
                    let ready = self.held.len() - keep;
                    let ready = self.held[..ready].trim_end().len();
                    let text: String = self.held.drain(..ready).collect();
                    let wrote = emit_reasoning(&mut out, &text, started, false);
                    self.state = State::Reasoning {
                        started: started || wrote,
                    };
                    return out;
                }
                State::Content { trim } => {
                    let text = std::mem::take(&mut self.held);
                    let text = match trim {
                        true => text.trim_start().to_string(),
                        false => text,
                    };
                    if !text.is_empty() {
                        self.state = State::Content { trim: false };
                        out.push(Piece::Content(text));
                    }
                    return out;
                }
            }
        }
    }

    /// The reply is over: place whatever was held back.
    pub fn finish(&mut self) -> Vec<Piece> {
        let text = std::mem::take(&mut self.held);
        if self.format.harmony {
            let mut out = Vec::new();
            if let Some(channel) = self.message {
                emit_message(&mut out, channel, text);
            }
            return out;
        }
        match self.state {
            // Never got past the marker, or past text that looked like it might be one.
            State::Start if text.trim().is_empty() => Vec::new(),
            State::Start => vec![Piece::Content(text)],
            // Cut off while thinking.
            State::Reasoning { started } => {
                let mut out = Vec::new();
                emit_reasoning(&mut out, &text, started, true);
                out
            }
            State::Content { trim } => {
                let text = if trim { text.trim_start() } else { &text };
                match text.is_empty() {
                    true => Vec::new(),
                    false => vec![Piece::Content(text.to_string())],
                }
            }
        }
    }
}

/// Send some of a harmony message on, where its channel says.
fn emit_message(out: &mut Vec<Piece>, channel: Channel, text: String) {
    if text.is_empty() {
        return;
    }
    match (channel, out.last_mut()) {
        (Channel::Analysis, Some(Piece::Reasoning(so_far))) => so_far.push_str(&text),
        (Channel::Analysis, _) => out.push(Piece::Reasoning(text)),
        (Channel::Answer, Some(Piece::Content(so_far))) => so_far.push_str(&text),
        (Channel::Answer, _) => out.push(Piece::Content(text)),
        (Channel::ToTool, _) => {}
    }
}

/// Send some of the thinking on, without the whitespace at its very start or (when `last`) its
/// very end. Returns whether anything was sent.
fn emit_reasoning(out: &mut Vec<Piece>, text: &str, started: bool, last: bool) -> bool {
    let text = if started { text } else { text.trim_start() };
    let text = if last { text.trim_end() } else { text };
    if text.is_empty() {
        return false;
    }
    match out.last_mut() {
        Some(Piece::Reasoning(so_far)) => so_far.push_str(text),
        _ => out.push(Piece::Reasoning(text.to_string())),
    }
    true
}

/// Split a whole reply at once.
pub fn split(format: &ReasoningFormat, open: bool, text: &str) -> (String, String) {
    let mut parser = ReasoningParser::new(format.clone(), open);
    let mut pieces = parser.push(text);
    pieces.extend(parser.finish());
    let (mut reasoning, mut content) = (String::new(), String::new());
    for piece in pieces {
        match piece {
            Piece::Reasoning(text) => reasoning.push_str(&text),
            Piece::Content(text) => content.push_str(&text),
        }
    }
    (reasoning, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(open: bool, chunks: &[&str]) -> (String, String) {
        let mut parser = ReasoningParser::new(ReasoningFormat::think(), open);
        let mut pieces = Vec::new();
        for chunk in chunks {
            pieces.extend(parser.push(chunk));
        }
        pieces.extend(parser.finish());
        let (mut reasoning, mut content) = (String::new(), String::new());
        for piece in pieces {
            match piece {
                Piece::Reasoning(text) => reasoning.push_str(&text),
                Piece::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
    }

    /// Every way of cutting `text` into two chunks gives the same result as all at once.
    fn same_however_cut(open: bool, text: &str) -> (String, String) {
        let whole = parts(open, &[text]);
        for at in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            assert_eq!(
                parts(open, &[&text[..at], &text[at..]]),
                whole,
                "cut at {at}"
            );
        }
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(parts(open, &refs), whole, "one character at a time");
        whole
    }

    #[test]
    fn a_block_the_model_opens_is_split_off() {
        let (reasoning, content) = same_however_cut(
            false,
            "<think>\nLet me add.\n2 + 2 = 4.\n</think>\n\nIt's 4.",
        );
        assert_eq!(reasoning, "Let me add.\n2 + 2 = 4.");
        assert_eq!(content, "It's 4.");
    }

    #[test]
    fn a_block_the_prompt_opened_is_split_off() {
        let (reasoning, content) = same_however_cut(true, "Hmm, 4.\n</think>\n\nFour.");
        assert_eq!(reasoning, "Hmm, 4.");
        assert_eq!(content, "Four.");
    }

    #[test]
    fn replies_without_a_block_are_all_answer() {
        assert_eq!(
            same_however_cut(false, "Just an answer, with <b>tags</b>."),
            (
                String::new(),
                "Just an answer, with <b>tags</b>.".to_string()
            )
        );
        // Text that starts out like the marker but isn't it.
        assert_eq!(
            same_however_cut(false, "<thin ice"),
            (String::new(), "<thin ice".to_string())
        );
    }

    #[test]
    fn a_reply_cut_off_while_thinking_is_all_reasoning() {
        assert_eq!(
            same_however_cut(false, "<think>\nStill going </thi"),
            ("Still going </thi".to_string(), String::new())
        );
    }

    #[test]
    fn an_empty_block_gives_no_reasoning() {
        assert_eq!(
            same_however_cut(false, "<think>\n\n</think>\n\nHi."),
            (String::new(), "Hi.".to_string())
        );
    }

    #[test]
    fn only_the_first_block_counts() {
        let (reasoning, content) = same_however_cut(false, "<think>a</think>b<think>c</think>");
        assert_eq!(reasoning, "a");
        assert_eq!(content, "b<think>c</think>");
    }

    #[test]
    fn whitespace_inside_the_thinking_is_kept() {
        let (reasoning, _) = same_however_cut(true, "  one\n\n  two  \n</think>x");
        assert_eq!(reasoning, "one\n\n  two");
    }

    #[test]
    fn prompts_that_leave_a_block_open_are_recognised() {
        let format = ReasoningFormat::think();
        assert!(format.opened_by("<|Assistant|><think>\n"));
        assert!(!format.opened_by("<|im_start|>assistant\n"));
        assert!(!format.opened_by("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(format.opened_by("<think>old</think> more <think>\n"));
    }

    #[test]
    fn harmony_messages_go_where_their_channels_say() {
        let reply = "<|channel|>analysis<|message|>The user wants a capital.<|end|><|start|>\
                     assistant<|channel|>final<|message|>Paris.";
        let (reasoning, content) = split(&ReasoningFormat::harmony(), true, reply);
        assert_eq!(reasoning, "The user wants a capital.");
        assert_eq!(content, "Paris.");
        // Cut up anywhere, it reads the same.
        for cut in 1..reply.len() {
            if !reply.is_char_boundary(cut) {
                continue;
            }
            let mut parser = ReasoningParser::new(ReasoningFormat::harmony(), true);
            let mut pieces = parser.push(&reply[..cut]);
            pieces.extend(parser.push(&reply[cut..]));
            pieces.extend(parser.finish());
            let (mut r, mut c) = (String::new(), String::new());
            for piece in pieces {
                match piece {
                    Piece::Reasoning(text) => r.push_str(&text),
                    Piece::Content(text) => c.push_str(&text),
                }
            }
            assert_eq!(
                (r.as_str(), c.as_str()),
                (reasoning.as_str(), content.as_str()),
                "cut at {cut}"
            );
        }
    }
}
