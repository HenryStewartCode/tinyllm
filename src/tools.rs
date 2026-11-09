//! Tool calling: letting a model ask the client to run a function.
//!
//! Models are trained to write calls in a notation of their own, as part of the text of their
//! reply, so there are three things to do:
//!
//! - [`plan`] works out what a request needs: which tools the chat template shows the model, and
//!   whether a grammar has to hold it to writing a valid call.
//! - [`ToolFormat`] knows each notation, both for writing that grammar and for reading calls.
//! - [`ToolParser`] reads calls out of a reply as it streams in, turning them into OpenAI's
//!   format. Anything that isn't a call for one of the request's tools stays ordinary text.

use crate::engine::{GrammarTrigger, partial_match_len};
use crate::grammar::{CallFrame, CallsGrammar};
use crate::tokenizer::Tokenizer;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The notations models write calls in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`: Qwen, Hermes.
    Hermes,
    /// `[TOOL_CALLS] [{"name": ..., "arguments": ...}]`: Mistral.
    Mistral,
    /// `[TOOL_CALLS]name[ARGS]{...}`: newer Mistral models.
    MistralCompact,
    /// `{"name": ..., "parameters": ...}`: Llama 3.
    Llama3,
    /// `[{"name": ..., "parameters": ...}]`, for models with no notation of their own.
    JsonArray,
}

/// How calls are arranged in a reply.
struct Layout {
    /// Text that opens a group of calls.
    opening: Option<&'static str>,
    /// Text that closes each group.
    closing: Option<&'static str>,
    /// Characters that open a group of calls when they're the first thing in a reply.
    leading: &'static str,
    /// What goes between calls that aren't in a list.
    separator: Option<u8>,
    /// Whether a group can be `name[ARGS]{...}`.
    compact: bool,
}

const HERMES: Layout = Layout {
    opening: Some("<tool_call>"),
    closing: Some("</tool_call>"),
    leading: "",
    separator: None,
    compact: false,
};
const MISTRAL: Layout = Layout {
    opening: Some("[TOOL_CALLS]"),
    closing: None,
    leading: "",
    separator: None,
    compact: true,
};
const LLAMA3: Layout = Layout {
    opening: Some("<|python_tag|>"),
    closing: None,
    leading: "{",
    separator: Some(b';'),
    compact: false,
};
const JSON_ARRAY: Layout = Layout {
    opening: None,
    closing: None,
    leading: "[",
    separator: None,
    compact: false,
};

impl ToolFormat {
    /// Work out a model's notation from its chat template, which has to spell it out to show
    /// the model its earlier calls. `None` if the template doesn't handle tools.
    pub fn detect(template: &str) -> Option<Self> {
        if template.contains("<tool_call>") {
            Some(Self::Hermes)
        } else if template.contains("[TOOL_CALLS]") {
            Some(match template.contains("[ARGS]") {
                true => Self::MistralCompact,
                false => Self::Mistral,
            })
        } else if template.contains("<|python_tag|>")
            || (template.contains("<|eot_id|>") && template.contains("\"parameters\""))
        {
            Some(Self::Llama3)
        } else {
            None
        }
    }

    fn layout(self) -> &'static Layout {
        match self {
            Self::Hermes => &HERMES,
            Self::Mistral | Self::MistralCompact => &MISTRAL,
            Self::Llama3 => &LLAMA3,
            Self::JsonArray => &JSON_ARRAY,
        }
    }

    /// Tokens that have to stay in the text for calls to be readable, even though they're
    /// special tokens, which are normally left out.
    pub fn marker_tokens(self, tokenizer: &Tokenizer) -> Vec<u32> {
        let markers: &[&str] = match self {
            Self::Hermes => &["<tool_call>", "</tool_call>"],
            Self::Mistral | Self::MistralCompact => &["[TOOL_CALLS]", "[ARGS]", "[CALL_ID]"],
            Self::Llama3 => &["<|python_tag|>"],
            Self::JsonArray => &[],
        };
        markers
            .iter()
            .filter_map(|marker| tokenizer.token_id(marker))
            .filter(|&id| tokenizer.is_special(id))
            .collect()
    }

    /// The grammar for replies made only of calls to these tools.
    fn grammar(self, tools: &[&Tool], max_calls: Option<usize>) -> CallsGrammar {
        // Llama 3's chat templates refuse to show more than one call in a message, so if there
        // were more the conversation couldn't go on.
        let max_calls = if self == Self::Llama3 {
            Some(1)
        } else {
            max_calls
        };
        let calls = tools
            .iter()
            .map(|tool| {
                let name = Value::from(tool.name.as_str()).to_string();
                let json = |key: &str| format!("{{\"name\": {name}, \"{key}\": ");
                let (before, after) = match self {
                    Self::Hermes => (
                        format!("<tool_call>\n{}", json("arguments")),
                        "}\n</tool_call>",
                    ),
                    Self::Mistral => (json("arguments"), "}"),
                    Self::MistralCompact => (format!("[TOOL_CALLS]{}[ARGS]", tool.name), ""),
                    Self::Llama3 | Self::JsonArray => (json("parameters"), "}"),
                };
                let no_parameters =
                    || json!({ "type": "object", "properties": {}, "additionalProperties": false });
                CallFrame {
                    before,
                    schema: tool.parameters.clone().unwrap_or_else(no_parameters),
                    strict: tool.strict,
                    after: after.to_string(),
                }
            })
            .collect();
        let (opening, separator, closing): (&[&str], &str, &str) = match self {
            Self::Hermes => (&[""], "\n", ""),
            Self::Mistral => (&["[TOOL_CALLS] ["], ", ", "]"),
            Self::MistralCompact => (&[""], "", ""),
            Self::Llama3 => (&["", "<|python_tag|>"], ";", ""),
            Self::JsonArray => (&["["], ", ", "]"),
        };
        CallsGrammar {
            opening: opening.iter().map(|s| s.to_string()).collect(),
            calls,
            separator: separator.to_string(),
            closing: closing.to_string(),
            max_calls,
        }
    }

    /// What in a free-form reply means a call has begun, and the grammar applies from there.
    fn triggers(self) -> Vec<GrammarTrigger> {
        let layout = self.layout();
        let marker = layout.opening.map(|text| GrammarTrigger {
            text: text.to_string(),
            at_start: false,
        });
        let leading = layout.leading.chars().map(|c| GrammarTrigger {
            text: c.to_string(),
            at_start: true,
        });
        marker.into_iter().chain(leading).collect()
    }

    /// A new id for a call. Mistral's chat templates only accept nine alphanumeric characters.
    pub fn call_id(self, random: u64) -> String {
        match self {
            Self::Mistral | Self::MistralCompact => {
                const DIGITS: &[u8] =
                    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                let mut rest = random;
                (0..9)
                    .map(|_| {
                        let digit = DIGITS[(rest % DIGITS.len() as u64) as usize];
                        rest /= DIGITS.len() as u64;
                        digit as char
                    })
                    .collect()
            }
            _ => format!("call_{random:016x}"),
        }
    }
}

struct Tool {
    name: String,
    parameters: Option<Value>,
    strict: bool,
    /// As the client wrote it, for the chat template.
    definition: Value,
}

fn read_tools(tools: Option<&Value>) -> Result<Vec<Tool>, String> {
    let list = match tools {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(list)) => list,
        Some(_) => return Err("tools must be a list".to_string()),
    };
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(list.len());
    for (i, tool) in list.iter().enumerate() {
        if tool.get("type").is_some_and(|kind| kind != "function") {
            return Err(format!("tool {i}: only tools of type \"function\" exist"));
        }
        let function = tool
            .get("function")
            .filter(|f| f.is_object())
            .ok_or_else(|| format!("tool {i} has no function"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| format!("tool {i} has no name"))?;
        if !seen.insert(name) {
            return Err(format!("there are two tools called '{name}'"));
        }
        let parameters = function.get("parameters").filter(|p| !p.is_null());
        if parameters.is_some_and(|p| !p.is_object()) {
            return Err(format!("the parameters of '{name}' must be a JSON schema"));
        }
        result.push(Tool {
            name: name.to_string(),
            parameters: parameters.cloned(),
            strict: function.get("strict") == Some(&Value::Bool(true)),
            definition: tool.clone(),
        });
    }
    Ok(result)
}

#[derive(Debug, PartialEq, Eq)]
enum Choice {
    None,
    Auto,
    Required,
    Named(String),
}

fn read_choice(choice: Option<&Value>) -> Result<Choice, String> {
    let invalid = || "tool_choice must be \"none\", \"auto\", \"required\" or name a function";
    match choice {
        None | Some(Value::Null) => Ok(Choice::Auto),
        Some(Value::String(choice)) => match choice.as_str() {
            "none" => Ok(Choice::None),
            "auto" => Ok(Choice::Auto),
            "required" => Ok(Choice::Required),
            _ => Err(invalid().to_string()),
        },
        Some(Value::Object(choice)) => {
            if choice.get("type").is_some_and(|kind| kind != "function") {
                return Err(invalid().to_string());
            }
            // `{"function": {"name": ...}}` in chat completions, `{"name": ...}` in the newer API.
            choice
                .get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| choice.get("name"))
                .and_then(Value::as_str)
                .map(|name| Choice::Named(name.to_string()))
                .ok_or_else(|| invalid().to_string())
        }
        Some(_) => Err(invalid().to_string()),
    }
}

/// What a request's tools mean for how it's run.
#[derive(Default)]
pub struct ToolPlan {
    /// What the chat template gets as `tools`.
    pub shown: Option<Value>,
    /// How to read calls out of the reply, if they should be.
    pub parser: Option<ToolReading>,
    /// What the reply is held to, if anything.
    pub grammar: Option<CallsGrammar>,
    /// If not empty, the reply is free until one of these turns up.
    pub triggers: Vec<GrammarTrigger>,
}

/// What it takes to start reading a reply for calls.
#[derive(Clone)]
pub struct ToolReading {
    pub format: ToolFormat,
    names: Arc<HashSet<String>>,
}

impl ToolReading {
    pub fn parser(&self) -> ToolParser {
        ToolParser::new(self.format, self.names.clone())
    }
}

pub struct ToolOptions<'a> {
    pub tools: Option<&'a Value>,
    pub tool_choice: Option<&'a Value>,
    pub parallel_tool_calls: bool,
    /// The model's notation, if it has one.
    pub format: Option<ToolFormat>,
    /// Whether the request sets a constraint of its own, such as a response format.
    pub output_constrained: bool,
}

/// Validate a request's tools and decide how to serve them.
pub fn plan(options: ToolOptions) -> Result<ToolPlan, String> {
    let tools = read_tools(options.tools)?;
    let choice = read_choice(options.tool_choice)?;
    if tools.is_empty() {
        return match choice {
            Choice::Required => Err("tool_choice \"required\" needs at least one tool".to_string()),
            Choice::Named(name) => Err(format!("tool '{name}' not found in tools")),
            _ => Ok(ToolPlan::default()),
        };
    }
    if choice == Choice::None {
        return Ok(ToolPlan::default());
    }
    let forced = matches!(choice, Choice::Required | Choice::Named(_));
    if forced && options.output_constrained {
        return Err(
            "tool_choice \"required\" or a named tool can't be combined with response_format, \
             regex, ebnf or json_schema"
                .to_string(),
        );
    }

    // Naming a tool puts only that tool in front of the model.
    let offered: Vec<&Tool> = match &choice {
        Choice::Named(name) => {
            let tool = tools
                .iter()
                .find(|tool| tool.name == *name)
                .ok_or_else(|| format!("tool '{name}' not found in tools"))?;
            vec![tool]
        }
        _ => tools.iter().collect(),
    };
    let shown = offered
        .iter()
        .map(|t| t.definition.clone())
        .collect::<Vec<_>>();
    let mut plan = ToolPlan {
        shown: Some(shown.into()),
        ..Default::default()
    };
    // Its own constraint decides what the whole reply looks like, which leaves no room for calls.
    if options.output_constrained {
        return Ok(plan);
    }

    // A call can be forced on a model with no notation for one, using a plain JSON list.
    let format = match (options.format, forced) {
        (Some(format), _) => format,
        (None, true) => ToolFormat::JsonArray,
        (None, false) => return Ok(plan),
    };
    let names = offered.iter().map(|tool| tool.name.clone()).collect();
    plan.parser = Some(ToolReading {
        format,
        names: Arc::new(names),
    });

    let parallel = options.parallel_tool_calls;
    let max_calls = (!parallel).then_some(1);
    if forced {
        plan.grammar = Some(format.grammar(&offered, max_calls));
    } else if !parallel || offered.iter().any(|tool| tool.strict) {
        // Only the calls have to follow rules, and those may never happen, so the grammar waits
        // for one to begin. The text around them then costs nothing extra.
        plan.grammar = Some(format.grammar(&offered, max_calls));
        plan.triggers = format.triggers();
    }
    Ok(plan)
}

/// Get a conversation ready for a chat template. In the API an assistant's earlier calls hold
/// their arguments as JSON text, whereas templates work with them as objects.
pub fn prepare_messages(messages: &mut [Value]) -> Result<(), String> {
    check_tool_replies(messages)?;
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        // Clients that send back a reply as they received it can leave these empty, and some
        // templates take the mere presence of the key to mean there are calls.
        if message
            .get("tool_calls")
            .is_some_and(|calls| calls.is_null() || calls.as_array().is_some_and(Vec::is_empty))
        {
            message.remove("tool_calls");
        }
        let Some(Value::Array(calls)) = message.get_mut("tool_calls") else {
            continue;
        };
        for call in calls {
            let arguments = call
                .get_mut("function")
                .and_then(|function| function.get_mut("arguments"));
            let Some(arguments) = arguments else {
                continue;
            };
            if let Value::String(text) = arguments {
                *arguments = parse_arguments(text)?;
            }
        }
    }
    Ok(())
}

fn parse_arguments(text: &str) -> Result<Value, String> {
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    let value: Value = serde_json::from_str(text)
        .map_err(|_| "the arguments of a tool call must be valid JSON".to_string())?;
    if !value.is_object() {
        return Err("the arguments of a tool call must be a JSON object".to_string());
    }
    Ok(value)
}

/// Each call must be answered by one tool message right after it, except at the end of the
/// conversation, where it may be waiting for one.
fn check_tool_replies(messages: &[Value]) -> Result<(), String> {
    let unanswered_error = |pending: &HashMap<&str, ()>| {
        let mut ids: Vec<&str> = pending.keys().copied().collect();
        ids.sort_unstable();
        format!("no tool message answers the tool calls {ids:?}")
    };
    // `None` when the last message wasn't an assistant's with calls (or a reply to those).
    let mut pending: Option<HashMap<&str, ()>> = None;
    // Calls without ids can't be matched with their results, so those aren't checked.
    let mut unchecked = false;
    for message in messages {
        let role = message.get("role").and_then(Value::as_str);
        if role == Some("tool") {
            if unchecked {
                continue;
            }
            let Some(calls) = &mut pending else {
                return Err(
                    "a tool message must follow an assistant message with tool_calls".to_string(),
                );
            };
            let id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            if calls.remove(id).is_none() {
                return Err(format!(
                    "tool_call_id '{id}' doesn't match an unanswered call"
                ));
            }
            continue;
        }
        if let Some(calls) = &pending
            && !calls.is_empty()
        {
            return Err(unanswered_error(calls));
        }
        pending = None;
        unchecked = false;
        let calls = message.get("tool_calls").and_then(Value::as_array);
        if role == Some("assistant")
            && let Some(calls) = calls.filter(|calls| !calls.is_empty())
        {
            let ids: Option<HashMap<&str, ()>> = calls
                .iter()
                .map(|call| call.get("id").and_then(Value::as_str).map(|id| (id, ())))
                .collect();
            unchecked = ids.is_none();
            pending = ids;
        }
    }
    let last_is_tool = messages
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("tool"));
    match &pending {
        Some(calls) if last_is_tool && !calls.is_empty() => Err(unanswered_error(calls)),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------------
// Reading calls
// ---------------------------------------------------------------------------------------------

/// A piece of a reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolOutput {
    /// Ordinary text.
    Content(String),
    /// The start of a call, with its arguments to follow.
    Call { index: usize, name: String },
    /// More of a call's arguments, which are JSON text.
    Arguments { index: usize, text: String },
}

/// Names and keys that run longer than this aren't what they seemed.
const MAX_SHORT_STRING: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Name,
    /// `function`, which small models write in place of `name`, copying how tools are shown to
    /// them: either the name itself, or an object holding the name and arguments.
    Function,
    Arguments,
    Other,
}

#[derive(Clone, Copy)]
enum State {
    /// Ordinary text, watching for the start of a group of calls.
    Text,
    /// Just inside a group, which says how the calls are arranged.
    Opening,
    /// Expecting the `{` of a call.
    BeforeObject,
    /// Between the members of a call's object.
    BeforeKey,
    Key,
    Colon(Field),
    Value(Field),
    Name,
    /// Inside `{...}` arguments, sending them on as they arrive.
    Arguments,
    /// Inside arguments written as a string of JSON text, which must arrive whole.
    ArgumentsString,
    /// Inside a value that isn't wanted.
    Skipping,
    /// After a call in a list.
    NextInList,
    /// After the last call of a group.
    AfterGroup,
    /// After a broken call, waiting for the marker at its end.
    SkipToClosing,
    CompactName,
}

/// Follows a JSON value a byte at a time, without keeping it, to find where it ends.
#[derive(Clone, Copy, Default)]
struct Skim {
    depth: u32,
    in_string: bool,
    escaped: bool,
    started: bool,
}

enum Skimmed {
    More,
    /// The value ends just before this position.
    Ended(usize),
    /// This can't be part of JSON.
    Invalid(usize),
}

impl Skim {
    fn run(&mut self, bytes: &[u8], from: usize) -> Skimmed {
        for (i, &byte) in bytes.iter().enumerate().skip(from) {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                    if self.depth == 0 {
                        return Skimmed::Ended(i + 1);
                    }
                }
                continue;
            }
            match byte {
                b'"' => {
                    self.in_string = true;
                    self.started = true;
                }
                b'{' | b'[' => {
                    self.depth += 1;
                    self.started = true;
                }
                b'}' | b']' => {
                    // Closing something this value isn't part of: it was a number, say.
                    if self.depth == 0 {
                        return Skimmed::Ended(i);
                    }
                    self.depth -= 1;
                    if self.depth == 0 {
                        return Skimmed::Ended(i + 1);
                    }
                }
                b',' if self.depth == 0 => return Skimmed::Ended(i),
                b',' | b':' => {}
                byte if byte.is_ascii_whitespace() => {
                    if self.depth == 0 && self.started {
                        return Skimmed::Ended(i);
                    }
                }
                b'-' | b'+' | b'.' | b'0'..=b'9' | b'a'..=b'z' | b'E' => self.started = true,
                _ => return Skimmed::Invalid(i),
            }
        }
        Skimmed::More
    }
}

/// Reads a reply a piece at a time, separating calls from ordinary text. Text that might turn
/// out to be part of a call is held until that's settled, and a call is only announced once
/// it's known to be for one of the request's tools. After that its arguments go out as they
/// arrive, and if they turn out to be broken the client gets what there was, as with OpenAI.
pub struct ToolParser {
    layout: &'static Layout,
    names: Arc<HashSet<String>>,
    /// Text not yet dealt with, from `pos` on, and text that might still be sent as ordinary
    /// text, from `attempt` on.
    buffer: String,
    pos: usize,
    state: State,
    out: Vec<ToolOutput>,
    calls: usize,
    /// There's been nothing since the last call, so any whitespace is only there for layout.
    call_was_last: bool,
    /// Nothing but whitespace has been written so far.
    at_start: bool,
    /// Where a possible call began, until it's certain.
    attempt: Option<usize>,
    /// The possible call came after a marker, or after another call, rather than the reply
    /// merely starting in a way that calls can.
    marked: bool,
    /// The current call's name is known, and has been sent.
    announced: bool,
    has_arguments: bool,
    /// Arguments that came before the name, so couldn't be sent yet.
    early_arguments: Option<String>,
    /// Where the arguments not yet sent begin.
    arguments_from: usize,
    in_list: bool,
    /// How many objects deep inside the call, for `function: {...}`.
    nesting: u32,
    /// The call is `name[ARGS]{...}`, which ends with its arguments.
    compact_call: bool,
    skim: Skim,
}

impl ToolParser {
    fn new(format: ToolFormat, names: Arc<HashSet<String>>) -> Self {
        Self {
            layout: format.layout(),
            names,
            buffer: String::new(),
            pos: 0,
            state: State::Text,
            out: Vec::new(),
            calls: 0,
            call_was_last: false,
            at_start: true,
            attempt: None,
            marked: false,
            announced: false,
            has_arguments: false,
            early_arguments: None,
            arguments_from: 0,
            in_list: false,
            nesting: 0,
            compact_call: false,
            skim: Skim::default(),
        }
    }

    /// How many calls have been found so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Add the next piece of the reply, and get whatever can be said for certain now.
    pub fn push(&mut self, text: &str) -> Vec<ToolOutput> {
        self.buffer.push_str(text);
        while self.step() {}
        // Forget what's been dealt with.
        let needed_from = match self.state {
            State::ArgumentsString => self.arguments_from,
            _ => self.pos,
        };
        let done = self.attempt.unwrap_or(needed_from).min(needed_from);
        self.buffer.drain(..done);
        self.pos -= done;
        self.arguments_from = self.arguments_from.saturating_sub(done);
        if let Some(attempt) = &mut self.attempt {
            *attempt -= done;
        }
        std::mem::take(&mut self.out)
    }

    /// The reply is complete: settle anything still being held.
    pub fn finish(&mut self) -> Vec<ToolOutput> {
        while self.step() {}
        if let State::Text = self.state {
            let rest = self.buffer[self.pos..].to_string();
            if !self.call_was_last || !rest.trim().is_empty() {
                self.send_text(&rest);
            }
        } else if self.attempt.is_some() {
            // Cut off before it was clear what this was. After a marker it's a call that never
            // got as far as saying what it was calling, which is of no use to anyone as text.
            if !self.marked {
                self.pos = self.buffer.len();
                self.not_a_call();
            }
        } else if self.announced && !self.has_arguments {
            self.send_no_arguments();
        }
        self.buffer.clear();
        self.pos = 0;
        std::mem::take(&mut self.out)
    }

    fn step(&mut self) -> bool {
        match self.state {
            State::Text => self.in_text(),
            State::Opening => self.opening(),
            State::BeforeObject => self.before_object(),
            State::BeforeKey => self.before_key(),
            State::Key => self.key(),
            State::Colon(field) => self.colon(field),
            State::Value(field) => self.value(field),
            State::Name => self.name(),
            State::Arguments => self.arguments(),
            State::ArgumentsString => self.arguments_string(),
            State::Skipping => self.skipping(),
            State::NextInList => self.next_in_list(),
            State::AfterGroup => self.after_group(),
            State::SkipToClosing => self.skip_to_closing(),
            State::CompactName => self.compact_name(),
        }
    }

    fn send_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.call_was_last = false;
        match self.out.last_mut() {
            Some(ToolOutput::Content(last)) => last.push_str(text),
            _ => self.out.push(ToolOutput::Content(text.to_string())),
        }
    }

    fn send_arguments(&mut self, text: String) {
        self.out.push(ToolOutput::Arguments {
            index: self.calls - 1,
            text,
        });
    }

    fn send_no_arguments(&mut self) {
        self.send_arguments("{}".to_string());
    }

    /// Skip whitespace, and look at what's after it.
    fn next_significant(&mut self) -> Option<u8> {
        let bytes = self.buffer.as_bytes();
        while let Some(&byte) = bytes.get(self.pos) {
            if !byte.is_ascii_whitespace() {
                return Some(byte);
            }
            self.pos += 1;
        }
        None
    }

    fn in_text(&mut self) -> bool {
        let rest = &self.buffer[self.pos..];
        if self.at_start {
            let trimmed = rest.trim_start();
            let Some(first) = trimmed.chars().next() else {
                return false;
            };
            self.at_start = false;
            if self.layout.leading.contains(first) {
                self.pos += rest.len() - trimmed.len();
                self.attempt = Some(self.pos);
                self.marked = false;
                self.state = State::Opening;
                return true;
            }
        }
        // Whitespace is held back as well: if a call comes next it's only there for layout.
        let Some(marker) = self.layout.opening else {
            let text = rest.trim_end().to_string();
            self.pos += text.len();
            self.send_text(&text);
            return false;
        };
        if let Some(at) = rest.find(marker) {
            let text = rest[..at].trim_end().to_string();
            self.send_text(&text);
            self.attempt = Some(self.pos + at);
            self.marked = true;
            self.pos += at + marker.len();
            self.state = State::Opening;
            return true;
        }
        let safe = rest.len() - partial_match_len(rest, marker);
        let text = rest[..safe].trim_end().to_string();
        self.pos += text.len();
        self.send_text(&text);
        false
    }

    fn opening(&mut self) -> bool {
        let Some(first) = self.next_significant() else {
            return false;
        };
        match first {
            b'[' => {
                self.pos += 1;
                self.in_list = true;
                self.state = State::BeforeObject;
                true
            }
            b'{' => {
                self.state = State::BeforeObject;
                true
            }
            _ if self.layout.compact => {
                self.state = State::CompactName;
                true
            }
            _ => self.not_a_call(),
        }
    }

    fn before_object(&mut self) -> bool {
        let Some(first) = self.next_significant() else {
            return false;
        };
        if self.attempt.is_none() {
            self.attempt = Some(self.pos);
            self.marked = true;
        }
        if first != b'{' {
            return self.not_a_call();
        }
        self.pos += 1;
        self.state = State::BeforeKey;
        true
    }

    fn before_key(&mut self) -> bool {
        let Some(first) = self.next_significant() else {
            return false;
        };
        match first {
            b',' => {
                self.pos += 1;
                true
            }
            b'"' => {
                self.state = State::Key;
                true
            }
            b'}' => {
                self.pos += 1;
                if self.nesting > 0 {
                    self.nesting -= 1;
                    return true;
                }
                self.end_of_call()
            }
            _ => self.broken(),
        }
    }

    /// The JSON string at the current position, if all of it has arrived. `Err` if it can't be
    /// what it's supposed to be.
    fn short_string(&mut self) -> Result<Option<String>, ()> {
        let bytes = &self.buffer.as_bytes()[self.pos..];
        let mut escaped = false;
        for (i, &byte) in bytes.iter().enumerate().skip(1).take(MAX_SHORT_STRING) {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                let end = self.pos + i + 1;
                let text = serde_json::from_str(&self.buffer[self.pos..end]).map_err(|_| ())?;
                self.pos = end;
                return Ok(Some(text));
            }
        }
        if bytes.len() > MAX_SHORT_STRING {
            Err(())
        } else {
            Ok(None)
        }
    }

    fn key(&mut self) -> bool {
        match self.short_string() {
            Ok(None) => false,
            Err(()) => self.broken(),
            Ok(Some(key)) => {
                let field = match key.as_str() {
                    "name" if !self.announced => Field::Name,
                    "function" if !self.announced => Field::Function,
                    "arguments" | "parameters"
                        if !self.has_arguments && self.early_arguments.is_none() =>
                    {
                        Field::Arguments
                    }
                    _ => Field::Other,
                };
                self.state = State::Colon(field);
                true
            }
        }
    }

    fn colon(&mut self, field: Field) -> bool {
        match self.next_significant() {
            None => false,
            Some(b':') => {
                self.pos += 1;
                self.state = State::Value(field);
                true
            }
            Some(_) => self.broken(),
        }
    }

    fn value(&mut self, field: Field) -> bool {
        let Some(first) = self.next_significant() else {
            return false;
        };
        self.skim = Skim::default();
        self.arguments_from = self.pos;
        self.state = match (field, first) {
            (Field::Name | Field::Function, b'"') => State::Name,
            (Field::Name, _) => return self.broken(),
            (Field::Function, b'{') => {
                self.pos += 1;
                self.nesting += 1;
                State::BeforeKey
            }
            (Field::Arguments, b'{' | b'[') => {
                self.has_arguments = true;
                State::Arguments
            }
            (Field::Arguments, b'"') => {
                self.has_arguments = true;
                State::ArgumentsString
            }
            _ => State::Skipping,
        };
        true
    }

    fn name(&mut self) -> bool {
        match self.short_string() {
            Ok(None) => false,
            Err(()) => self.broken(),
            Ok(Some(name)) if self.names.contains(&name) => {
                self.announce(name);
                self.state = State::BeforeKey;
                true
            }
            Ok(Some(_)) => self.not_a_call(),
        }
    }

    fn announce(&mut self, name: String) {
        self.out.push(ToolOutput::Call {
            index: self.calls,
            name,
        });
        self.calls += 1;
        self.call_was_last = true;
        self.announced = true;
        self.attempt = None;
        if let Some(arguments) = self.early_arguments.take() {
            self.has_arguments = true;
            self.send_arguments(arguments);
        }
    }

    /// Send the arguments that have arrived.
    fn flush_arguments(&mut self) {
        if self.announced && self.pos > self.arguments_from {
            let text = self.buffer[self.arguments_from..self.pos].to_string();
            self.arguments_from = self.pos;
            self.send_arguments(text);
        }
    }

    fn arguments(&mut self) -> bool {
        match self.skim.run(self.buffer.as_bytes(), self.pos) {
            Skimmed::More => {
                self.pos = self.buffer.len();
                self.flush_arguments();
                false
            }
            Skimmed::Ended(end) => {
                self.pos = end;
                if !self.announced {
                    let text = self.buffer[self.arguments_from..end].to_string();
                    self.early_arguments = Some(text);
                    self.has_arguments = false;
                }
                self.flush_arguments();
                self.after_value()
            }
            Skimmed::Invalid(at) => {
                self.pos = at;
                self.flush_arguments();
                self.broken()
            }
        }
    }

    fn arguments_string(&mut self) -> bool {
        match self.skim.run(self.buffer.as_bytes(), self.pos) {
            Skimmed::More => {
                self.pos = self.buffer.len();
                false
            }
            Skimmed::Ended(end) => {
                self.pos = end;
                let raw = &self.buffer[self.arguments_from..end];
                let text = serde_json::from_str(raw).unwrap_or_else(|_| raw.to_string());
                if self.announced {
                    self.send_arguments(text);
                } else {
                    self.early_arguments = Some(text);
                    self.has_arguments = false;
                }
                self.after_value()
            }
            Skimmed::Invalid(at) => {
                self.pos = at;
                self.broken()
            }
        }
    }

    fn skipping(&mut self) -> bool {
        match self.skim.run(self.buffer.as_bytes(), self.pos) {
            Skimmed::More => {
                self.pos = self.buffer.len();
                false
            }
            Skimmed::Ended(end) => {
                self.pos = end;
                self.after_value()
            }
            Skimmed::Invalid(at) => {
                self.pos = at;
                self.broken()
            }
        }
    }

    fn after_value(&mut self) -> bool {
        if self.compact_call {
            return self.end_of_call();
        }
        self.state = State::BeforeKey;
        true
    }

    fn end_of_call(&mut self) -> bool {
        if !self.announced {
            return self.not_a_call();
        }
        if !self.has_arguments {
            self.send_no_arguments();
        }
        self.forget_call();
        self.state = if self.in_list {
            State::NextInList
        } else {
            State::AfterGroup
        };
        true
    }

    fn forget_call(&mut self) {
        self.attempt = None;
        self.nesting = 0;
        self.announced = false;
        self.has_arguments = false;
        self.early_arguments = None;
        self.compact_call = false;
    }

    /// What looked like a call isn't one: everything since it began is ordinary text.
    fn not_a_call(&mut self) -> bool {
        let start = self.attempt.take().unwrap_or(self.pos);
        let text = self.buffer[start..self.pos].to_string();
        self.send_text(&text);
        self.forget_call();
        self.in_list = false;
        self.state = State::Text;
        true
    }

    /// The text can't be read any further.
    fn broken(&mut self) -> bool {
        if !self.announced {
            return self.not_a_call();
        }
        if !self.has_arguments {
            self.send_no_arguments();
        }
        self.forget_call();
        self.in_list = false;
        self.state = match self.layout.closing {
            Some(_) => State::SkipToClosing,
            None => State::Text,
        };
        true
    }

    fn next_in_list(&mut self) -> bool {
        let Some(next) = self.next_significant() else {
            return false;
        };
        match next {
            b',' => {
                self.pos += 1;
                self.state = State::BeforeObject;
            }
            b']' => {
                self.pos += 1;
                self.in_list = false;
                self.state = State::AfterGroup;
            }
            _ => {
                self.in_list = false;
                self.state = State::AfterGroup;
            }
        }
        true
    }

    fn after_group(&mut self) -> bool {
        let Some(next) = self.next_significant() else {
            return false;
        };
        if let Some(closing) = self.layout.closing {
            let rest = &self.buffer[self.pos..];
            if rest.starts_with(closing) {
                self.pos += closing.len();
            } else if closing.starts_with(rest) {
                return false;
            }
        } else if self.layout.separator == Some(next) {
            self.pos += 1;
            self.state = State::BeforeObject;
            return true;
        }
        self.state = State::Text;
        true
    }

    fn skip_to_closing(&mut self) -> bool {
        let closing = self.layout.closing.expect("only entered if there is one");
        let rest = &self.buffer[self.pos..];
        if let Some(at) = rest.find(closing) {
            self.pos += at + closing.len();
            self.state = State::Text;
            return true;
        }
        self.pos += rest.len() - partial_match_len(rest, closing);
        false
    }

    fn compact_name(&mut self) -> bool {
        const ARGS: &str = "[ARGS]";
        let rest = &self.buffer[self.pos..];
        let Some(at) = rest.find(ARGS) else {
            if rest.len() > MAX_SHORT_STRING || rest.contains('\n') {
                return self.not_a_call();
            }
            return false;
        };
        let name = rest[..at].split("[CALL_ID]").next().unwrap_or("").trim();
        if !self.names.contains(name) {
            return self.not_a_call();
        }
        let name = name.to_string();
        self.pos += at + ARGS.len();
        self.announce(name);
        self.compact_call = true;
        self.state = State::Value(Field::Arguments);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ToolOutput::{Arguments, Call, Content};

    fn reading(format: ToolFormat) -> ToolReading {
        ToolReading {
            format,
            names: Arc::new(["get_weather", "get_time"].map(String::from).into()),
        }
    }

    /// Combine runs of the same kind of output, so that different chunkings can be compared.
    fn merged(outputs: Vec<ToolOutput>) -> Vec<ToolOutput> {
        let mut result: Vec<ToolOutput> = Vec::new();
        for output in outputs {
            match (result.last_mut(), &output) {
                (Some(Content(last)), Content(text)) => last.push_str(text),
                (
                    Some(Arguments {
                        index: i,
                        text: last,
                    }),
                    Arguments { index: j, text },
                ) if i == j => last.push_str(text),
                _ => result.push(output),
            }
        }
        result
    }

    /// Parse the text whole, then one character at a time and split in every place, checking
    /// that it makes no difference.
    fn parse(format: ToolFormat, text: &str) -> Vec<ToolOutput> {
        let whole = {
            let mut parser = reading(format).parser();
            let mut out = parser.push(text);
            out.extend(parser.finish());
            merged(out)
        };
        let mut parser = reading(format).parser();
        let mut chars = Vec::new();
        let mut buffer = [0; 4];
        for c in text.chars() {
            chars.extend(parser.push(c.encode_utf8(&mut buffer)));
        }
        chars.extend(parser.finish());
        assert_eq!(merged(chars), whole, "one character at a time: {text}");
        for (at, _) in text.char_indices() {
            let mut parser = reading(format).parser();
            let mut out = parser.push(&text[..at]);
            out.extend(parser.push(&text[at..]));
            out.extend(parser.finish());
            assert_eq!(merged(out), whole, "split at {at}: {text}");
        }
        whole
    }

    fn call(index: usize, name: &str) -> ToolOutput {
        Call {
            index,
            name: name.to_string(),
        }
    }

    fn args(index: usize, text: &str) -> ToolOutput {
        Arguments {
            index,
            text: text.to_string(),
        }
    }

    fn text(text: &str) -> ToolOutput {
        Content(text.to_string())
    }

    #[test]
    fn hermes_calls() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
            ),
            [call(0, "get_weather"), args(0, "{\"city\": \"Paris\"}")]
        );
    }

    #[test]
    fn text_around_several_calls() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "Let me look.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>\n\
                 <tool_call>{\"name\":\"get_time\",\"arguments\":{\"tz\":\"é}\\\"\"}}</tool_call>\n\nDone.\n"
            ),
            [
                text("Let me look."),
                call(0, "get_weather"),
                args(0, "{\"city\": \"Rome\"}"),
                call(1, "get_time"),
                args(1, "{\"tz\":\"é}\\\"\"}"),
                text("\n\nDone.\n"),
            ]
        );
    }

    #[test]
    fn plain_replies_are_untouched() {
        for reply in [
            "",
            "Hello",
            "  a < b\n\n",
            "<tool",
            "2 < 3 and <tool_ca",
            "{\"a\": 1}",
        ] {
            let expected = if reply.is_empty() {
                vec![]
            } else {
                vec![text(reply)]
            };
            assert_eq!(parse(ToolFormat::Hermes, reply), expected);
        }
    }

    #[test]
    fn whitespace_after_the_last_call_is_dropped() {
        let reply = "<tool_call>{\"name\": \"get_time\"}</tool_call>\n ";
        assert_eq!(
            parse(ToolFormat::Hermes, reply),
            [call(0, "get_time"), args(0, "{}")]
        );
    }

    #[test]
    fn what_isnt_a_call_stays_text() {
        for reply in [
            "<tool_call>not json</tool_call>",
            "<tool_call>\n{\"name\": \"launch_rockets\", \"arguments\": {}}\n</tool_call>",
            "<tool_call>{\"arguments\": {}}</tool_call>",
            "<tool_call>{\"name\": 3}</tool_call>",
        ] {
            assert_eq!(parse(ToolFormat::Hermes, reply), [text(reply)], "{reply}");
        }
    }

    #[test]
    fn a_call_cut_off_before_its_name_is_dropped() {
        for reply in [
            "<tool_call>",
            "<tool_call>{\"na",
            "<tool_call>\n{\"name\": \"get_wea",
        ] {
            assert_eq!(parse(ToolFormat::Hermes, reply), [], "{reply}");
        }
        assert_eq!(
            parse(ToolFormat::Hermes, "Checking.<tool_call>{\"na"),
            [text("Checking.")]
        );
        assert_eq!(
            parse(ToolFormat::JsonArray, "[{\"name\": \"get_time\"}, {\"na"),
            [call(0, "get_time"), args(0, "{}")]
        );
        // With no marker, it could still have been an answer that happened to be JSON.
        assert_eq!(parse(ToolFormat::Llama3, "{\"na"), [text("{\"na")]);
    }

    #[test]
    fn function_can_stand_in_for_name() {
        assert_eq!(
            parse(
                ToolFormat::Llama3,
                "<|python_tag|>{\"type\": \"function\", \"function\": \"get_time\", \"parameters\": {\"tz\": 1}}"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1}")]
        );
        assert_eq!(
            parse(
                ToolFormat::Llama3,
                "{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"parameters\": {\"tz\": 1}}}"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1}")]
        );
        let not_ours = "{\"function\": \"nothing\"}";
        assert_eq!(parse(ToolFormat::Llama3, not_ours), [text(not_ours)]);
    }

    #[test]
    fn arguments_may_come_first() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>{\"arguments\": {\"city\": \"Oslo\"}, \"name\": \"get_weather\"}</tool_call>"
            ),
            [call(0, "get_weather"), args(0, "{\"city\": \"Oslo\"}")]
        );
    }

    #[test]
    fn other_members_are_ignored() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>{\"id\": 12, \"name\": \"get_time\", \"x\": [1, {\"a\": \"}\"}], \"parameters\": {\"tz\": 1}, \"y\": null}</tool_call>"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1}")]
        );
    }

    #[test]
    fn arguments_as_a_string_are_unwrapped() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>{\"name\": \"get_time\", \"arguments\": \"{\\\"tz\\\": 1}\"}</tool_call>"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1}")]
        );
    }

    #[test]
    fn cut_off_calls_keep_what_arrived() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>{\"name\": \"get_time\", \"arguments\": {\"tz\": \"Eu"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": \"Eu")]
        );
        assert_eq!(
            parse(ToolFormat::Hermes, "<tool_call>{\"name\": \"get_time\""),
            [call(0, "get_time"), args(0, "{}")]
        );
    }

    #[test]
    fn broken_arguments_end_at_the_closing_marker() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>{\"name\": \"get_time\", \"arguments\": {\"tz\": 1\n</tool_call>after"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1\n"), text("after")]
        );
    }

    #[test]
    fn a_list_inside_one_marker() {
        assert_eq!(
            parse(
                ToolFormat::Hermes,
                "<tool_call>[{\"name\": \"get_time\"}, {\"name\": \"get_weather\", \"arguments\": {}}]</tool_call>"
            ),
            [
                call(0, "get_time"),
                args(0, "{}"),
                call(1, "get_weather"),
                args(1, "{}")
            ]
        );
    }

    #[test]
    fn mistral_lists() {
        assert_eq!(
            parse(
                ToolFormat::Mistral,
                "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}, \"id\": \"abc\"}, {\"name\": \"get_time\", \"arguments\": {}}]"
            ),
            [
                call(0, "get_weather"),
                args(0, "{\"city\": \"Rome\"}"),
                call(1, "get_time"),
                args(1, "{}"),
            ]
        );
    }

    #[test]
    fn mistral_compact_calls() {
        assert_eq!(
            parse(
                ToolFormat::MistralCompact,
                "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Rome\"}[TOOL_CALLS]get_time[CALL_ID]abc123def[ARGS]{}"
            ),
            [
                call(0, "get_weather"),
                args(0, "{\"city\": \"Rome\"}"),
                call(1, "get_time"),
                args(1, "{}"),
            ]
        );
        let unknown = "[TOOL_CALLS]nothing[ARGS]{}";
        assert_eq!(parse(ToolFormat::MistralCompact, unknown), [text(unknown)]);
    }

    #[test]
    fn llama_calls() {
        assert_eq!(
            parse(
                ToolFormat::Llama3,
                "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}; {\"name\": \"get_time\", \"parameters\": {}}"
            ),
            [
                call(0, "get_weather"),
                args(0, "{\"city\": \"Rome\"}"),
                call(1, "get_time"),
                args(1, "{}"),
            ]
        );
        assert_eq!(
            parse(
                ToolFormat::Llama3,
                "Sure.<|python_tag|>{\"name\": \"get_time\", \"parameters\": {}}"
            ),
            [text("Sure."), call(0, "get_time"), args(0, "{}")]
        );
        // JSON that isn't a call is only an answer that happens to be JSON.
        let answer = "{\"name\": \"Ada\", \"age\": 36}";
        assert_eq!(parse(ToolFormat::Llama3, answer), [text(answer)]);
        let code = "<|python_tag|>print(1)";
        assert_eq!(parse(ToolFormat::Llama3, code), [text(code)]);
    }

    #[test]
    fn json_arrays() {
        assert_eq!(
            parse(
                ToolFormat::JsonArray,
                "[{\"name\": \"get_time\", \"parameters\": {\"tz\": 1}}]"
            ),
            [call(0, "get_time"), args(0, "{\"tz\": 1}")]
        );
    }

    #[test]
    fn formats_are_recognized_from_templates() {
        assert_eq!(
            ToolFormat::detect("x<tool_call>y"),
            Some(ToolFormat::Hermes)
        );
        assert_eq!(
            ToolFormat::detect("[TOOL_CALLS] ["),
            Some(ToolFormat::Mistral)
        );
        assert_eq!(
            ToolFormat::detect("[TOOL_CALLS]x[ARGS]"),
            Some(ToolFormat::MistralCompact)
        );
        assert_eq!(
            ToolFormat::detect("<|eot_id|>'\"parameters\": '"),
            Some(ToolFormat::Llama3)
        );
        assert_eq!(ToolFormat::detect("<|im_start|>"), None);
    }

    fn options<'a>(tools: &'a Value, choice: Option<&'a Value>) -> ToolOptions<'a> {
        ToolOptions {
            tools: Some(tools),
            tool_choice: choice,
            parallel_tool_calls: true,
            format: Some(ToolFormat::Hermes),
            output_constrained: false,
        }
    }

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "b"}},
        ])
    }

    #[test]
    fn auto_leaves_the_model_free() {
        let tools = tools();
        let plan = plan(options(&tools, None)).unwrap();
        assert_eq!(plan.shown.as_ref(), Some(&tools));
        assert!(plan.parser.is_some() && plan.grammar.is_none());
    }

    #[test]
    fn strict_tools_are_enforced_only_once_a_call_starts() {
        let tools = json!([{"function": {"name": "a", "strict": true}}]);
        let plan = plan(options(&tools, None)).unwrap();
        assert!(plan.grammar.is_some());
        assert_eq!(plan.triggers.len(), 1);
        assert_eq!(plan.triggers[0].text, "<tool_call>");
    }

    #[test]
    fn one_call_at_most_needs_a_grammar() {
        let tools = tools();
        let mut options = options(&tools, None);
        options.parallel_tool_calls = false;
        let plan = plan(options).unwrap();
        assert_eq!(plan.grammar.unwrap().max_calls, Some(1));
        assert!(!plan.triggers.is_empty());
    }

    #[test]
    fn none_hides_the_tools() {
        let tools = tools();
        let choice = json!("none");
        let plan = plan(options(&tools, Some(&choice))).unwrap();
        assert!(plan.shown.is_none() && plan.parser.is_none());
    }

    #[test]
    fn required_is_enforced_from_the_start() {
        let tools = tools();
        let choice = json!("required");
        let plan = plan(options(&tools, Some(&choice))).unwrap();
        assert_eq!(plan.grammar.unwrap().calls.len(), 2);
        assert!(plan.triggers.is_empty());
    }

    #[test]
    fn naming_a_tool_shows_only_that_one() {
        let tools = tools();
        for choice in [
            json!({"type": "function", "function": {"name": "b"}}),
            json!({"type": "function", "name": "b"}),
        ] {
            let plan = plan(options(&tools, Some(&choice))).unwrap();
            assert_eq!(plan.shown.as_ref().unwrap().as_array().unwrap().len(), 1);
            assert_eq!(plan.grammar.unwrap().calls.len(), 1);
        }
    }

    #[test]
    fn a_model_with_no_notation_can_still_be_forced() {
        let tools = tools();
        let choice = json!("required");
        let mut opts = options(&tools, Some(&choice));
        opts.format = None;
        let forced = plan(opts).unwrap();
        assert_eq!(forced.parser.unwrap().format, ToolFormat::JsonArray);
        let mut opts = options(&tools, None);
        opts.format = None;
        assert!(plan(opts).unwrap().parser.is_none());
    }

    #[test]
    fn another_constraint_takes_over() {
        let tools = tools();
        let mut opts = options(&tools, None);
        opts.output_constrained = true;
        let auto = plan(opts).unwrap();
        assert!(auto.parser.is_none() && auto.grammar.is_none() && auto.shown.is_some());
        let choice = json!("required");
        let mut opts = options(&tools, Some(&choice));
        opts.output_constrained = true;
        assert!(plan(opts).is_err());
    }

    #[test]
    fn mistakes_are_reported() {
        let empty = json!([]);
        let good = tools();
        let required = json!("required");
        let unknown = json!({"function": {"name": "nope"}});
        let wrong = json!("sometimes");
        for (tools, choice) in [
            (&empty, Some(&required)),
            (&empty, Some(&unknown)),
            (&good, Some(&unknown)),
            (&good, Some(&wrong)),
        ] {
            assert!(plan(options(tools, choice)).is_err());
        }
        for tools in [
            json!({"function": {"name": "a"}}),
            json!([{"type": "web_search", "function": {"name": "a"}}]),
            json!([{"type": "function"}]),
            json!([{"function": {"description": "no name"}}]),
            json!([{"function": {"name": "a"}}, {"function": {"name": "a"}}]),
            json!([{"function": {"name": "a", "parameters": "x"}}]),
        ] {
            assert!(plan(options(&tools, None)).is_err(), "{tools}");
        }
    }

    #[test]
    fn no_tools_is_the_same_as_none_given() {
        let empty = json!([]);
        let auto = json!("auto");
        let plan = plan(options(&empty, Some(&auto))).unwrap();
        assert!(plan.shown.is_none() && plan.parser.is_none());
    }

    #[test]
    fn arguments_become_objects_for_templates() {
        let mut messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "1", "function": {"name": "a", "arguments": "{\"b\": 1, \"a\": 2}"}},
                {"id": "2", "function": {"name": "b", "arguments": ""}},
            ]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "x"}),
            json!({"role": "tool", "tool_call_id": "2", "content": "y"}),
        ];
        prepare_messages(&mut messages).unwrap();
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"].to_string(),
            "{\"b\":1,\"a\":2}",
            "the order is kept"
        );
        assert_eq!(
            messages[1]["tool_calls"][1]["function"]["arguments"],
            json!({})
        );
    }

    #[test]
    fn empty_tool_calls_are_removed() {
        let mut messages = vec![
            json!({"role": "assistant", "content": "a", "tool_calls": null}),
            json!({"role": "assistant", "content": "b", "tool_calls": []}),
        ];
        prepare_messages(&mut messages).unwrap();
        assert!(messages.iter().all(|m| m.get("tool_calls").is_none()));
    }

    #[test]
    fn calls_and_results_have_to_match_up() {
        let assistant = |ids: &[&str]| {
            let calls: Vec<Value> = ids
                .iter()
                .map(|id| json!({"id": id, "function": {"name": "a", "arguments": "{}"}}))
                .collect();
            json!({"role": "assistant", "tool_calls": calls})
        };
        let result = |id: &str| json!({"role": "tool", "tool_call_id": id, "content": "x"});
        let user = || json!({"role": "user", "content": "x"});
        let check = |mut messages: Vec<Value>| prepare_messages(&mut messages).is_ok();
        assert!(check(vec![user(), assistant(&["1"]), result("1")]));
        assert!(check(vec![
            assistant(&["1", "2"]),
            result("2"),
            result("1"),
            user()
        ]));
        // The last message can be one still waiting for its results.
        assert!(check(vec![user(), assistant(&["1"])]));
        // A result with nothing before it.
        assert!(!check(vec![user(), result("1")]));
        // The wrong call.
        assert!(!check(vec![assistant(&["1"]), result("2")]));
        // A result given twice.
        assert!(!check(vec![assistant(&["1"]), result("1"), result("1")]));
        // Moving on with a call unanswered.
        assert!(!check(vec![assistant(&["1"]), user()]));
        assert!(!check(vec![assistant(&["1", "2"]), result("1")]));
        // Calls without ids can't be matched up.
        let mut no_id = assistant(&["1"]);
        no_id["tool_calls"][0].as_object_mut().unwrap().remove("id");
        assert!(check(vec![no_id, result("anything")]));
    }

    #[test]
    fn invalid_arguments_in_history_are_refused() {
        for arguments in ["{", "[1]", "3"] {
            let mut messages = vec![json!({"role": "assistant", "tool_calls": [
                {"function": {"name": "a", "arguments": arguments}},
            ]})];
            assert!(prepare_messages(&mut messages).is_err(), "{arguments}");
        }
    }

    #[test]
    fn ids_suit_the_format() {
        let id = ToolFormat::Hermes.call_id(0x1234);
        assert_eq!(id, "call_0000000000001234");
        let id = ToolFormat::Mistral.call_id(u64::MAX);
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
