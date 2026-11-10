//! Anthropic's Messages API (`/v1/messages`), as SGLang offers it, so that clients written for
//! Claude can use a model served here. Requests are translated into the OpenAI form the rest of
//! the server works with, and replies are translated back: thinking becomes `thinking` blocks and
//! tool calls `tool_use` blocks, streamed with Anthropic's events.

use super::{
    ApiError, App, ChatRequest, OneOrMany, Reply, ReplyPart, ReplyReading, SamplingFields, collect,
    json_response, parse_body, prepare, prepare_chat, start_chat,
};
use crate::engine::{Event, FinishReason, Usage};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::Response;
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::convert::Infallible;
use std::sync::Arc;

#[derive(Deserialize, Default)]
#[serde(default)]
struct MessagesRequest {
    model: Option<String>,
    max_tokens: Option<usize>,
    messages: Vec<Value>,
    system: Option<Value>,
    stop_sequences: Option<Vec<String>>,
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
}

/// An error in Anthropic's format.
fn error(err: ApiError) -> Response {
    let kind = match err.status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        _ => "api_error",
    };
    let body = json!({ "type": "error", "error": { "type": kind, "message": err.message } });
    json_response(err.status, &body)
}

/// The text of a `system` field or a block's `content`: a string, or a list of text blocks.
fn text_of(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
                    other => return Err(unsupported_block(other)),
                }
            }
            Ok(text)
        }
        _ => Err(ApiError::invalid(
            "content must be a string or a list of blocks",
        )),
    }
}

fn unsupported_block(kind: Option<&str>) -> ApiError {
    match kind {
        Some(kind @ ("image" | "document")) => {
            ApiError::invalid(format!("{kind} blocks aren't supported by this model"))
        }
        Some(kind) => ApiError::invalid(format!("unknown content block type '{kind}'")),
        None => ApiError::invalid("content blocks need a type"),
    }
}

/// Turn Anthropic's messages into OpenAI's: a system message first, tool results as `tool`
/// messages, and an assistant's thinking and tool use as `reasoning_content` and `tool_calls`.
fn openai_messages(request: &MessagesRequest) -> Result<Vec<Value>, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "system", "content": text_of(system)? }));
    }
    for message in &request.messages {
        let role = message["role"].as_str().unwrap_or_default();
        let blocks = match &message["content"] {
            Value::String(text) => {
                messages.push(json!({ "role": role, "content": text }));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => {
                return Err(ApiError::invalid(
                    "content must be a string or a list of blocks",
                ));
            }
        };
        match role {
            "user" => {
                let mut text = String::new();
                for block in blocks {
                    match block["type"].as_str() {
                        Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
                        Some("tool_result") => {
                            let mut content = text_of(&block["content"])?;
                            if block["is_error"].as_bool() == Some(true) {
                                content = format!("Error: {content}");
                            }
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": content,
                            }));
                        }
                        other => return Err(unsupported_block(other)),
                    }
                }
                if !text.is_empty() {
                    messages.push(json!({ "role": "user", "content": text }));
                }
            }
            "assistant" => {
                let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), vec![]);
                for block in blocks {
                    match block["type"].as_str() {
                        Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
                        Some("thinking") => {
                            reasoning.push_str(block["thinking"].as_str().unwrap_or_default());
                        }
                        Some("redacted_thinking") => {}
                        Some("tool_use") => calls.push(json!({
                            "id": block["id"],
                            "type": "function",
                            "function": { "name": block["name"], "arguments": block["input"] },
                        })),
                        other => return Err(unsupported_block(other)),
                    }
                }
                let mut message = json!({ "role": "assistant", "content": text });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = reasoning.into();
                }
                if !calls.is_empty() {
                    message["tool_calls"] = calls.into();
                }
                messages.push(message);
            }
            other => {
                return Err(ApiError::invalid(format!(
                    "a message's role must be user or assistant, not '{other}'"
                )));
            }
        }
    }
    Ok(messages)
}

/// The same request in OpenAI's form.
fn chat_request(request: &MessagesRequest) -> Result<ChatRequest, ApiError> {
    let tools = match &request.tools {
        None => None,
        Some(tools) => {
            let mut converted = Vec::new();
            for tool in tools {
                let Some(schema) = tool.get("input_schema") else {
                    let kind = tool["type"].as_str().unwrap_or("unknown");
                    return Err(ApiError::invalid(format!(
                        "only client tools with an input_schema are supported, not '{kind}'"
                    )));
                };
                converted.push(json!({ "type": "function", "function": {
                    "name": tool["name"],
                    "description": tool["description"].as_str().unwrap_or_default(),
                    "parameters": schema,
                }}));
            }
            Some(Value::Array(converted))
        }
    };
    let mut parallel_tool_calls = None;
    let tool_choice = match &request.tool_choice {
        None => None,
        Some(choice) => {
            if choice["disable_parallel_tool_use"].as_bool() == Some(true) {
                parallel_tool_calls = Some(false);
            }
            Some(match choice["type"].as_str() {
                Some("auto") => json!("auto"),
                Some("any") => json!("required"),
                Some("none") => json!("none"),
                Some("tool") => {
                    json!({ "type": "function", "function": { "name": choice["name"] } })
                }
                _ => return Err(ApiError::invalid("unknown tool_choice type")),
            })
        }
    };
    // Extended thinking: on with a budget, off, or left to the model.
    let mut kwargs = Map::new();
    let mut thinking_budget = None;
    match request.thinking.as_ref().and_then(|t| t["type"].as_str()) {
        Some("enabled") => {
            kwargs.insert("enable_thinking".into(), true.into());
            let thinking = request.thinking.as_ref().expect("matched above");
            thinking_budget = thinking["budget_tokens"].as_u64().map(|b| b as usize);
        }
        Some("disabled") => {
            kwargs.insert("enable_thinking".into(), false.into());
        }
        _ => {}
    }
    Ok(ChatRequest {
        messages: openai_messages(request)?,
        tools,
        tool_choice,
        parallel_tool_calls,
        chat_template_kwargs: (!kwargs.is_empty()).then_some(kwargs),
        thinking_budget,
        separate_reasoning: Some(true),
        sampling: SamplingFields {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            stop: request.stop_sequences.clone().map(OneOrMany::Many),
            model: request.model.clone(),
            ..Default::default()
        },
        ..Default::default()
    })
}

fn usage_json(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens - usage.cached_tokens,
        "output_tokens": usage.completion_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
        "cache_creation_input_tokens": 0,
    })
}

/// Why a reply stopped, and the stop sequence that did it, if one did.
fn stop_reason(reason: FinishReason, calls: usize, stop: Option<&str>) -> (&'static str, Value) {
    match (reason, calls, stop) {
        (FinishReason::Length, ..) => ("max_tokens", Value::Null),
        (FinishReason::Stop, 1.., _) => ("tool_use", Value::Null),
        (FinishReason::Stop, _, Some(stop)) => ("stop_sequence", stop.into()),
        (FinishReason::Stop, _, None) => ("end_turn", Value::Null),
    }
}

fn tool_use_id(app: &App) -> String {
    let count = app
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let seed = super::new_seed();
    let (high, low) = (
        crate::sampler::draw_key(seed, count as usize),
        crate::sampler::draw_key(!seed, count as usize),
    );
    format!("toolu_{high:016x}{:08x}", low as u32)
}

/// A finished reply as Anthropic content blocks.
fn content_blocks(app: &App, reply: &Reply) -> Vec<Value> {
    let mut blocks = Vec::new();
    if !reply.reasoning.is_empty() {
        blocks.push(json!({ "type": "thinking", "thinking": reply.reasoning, "signature": "" }));
    }
    if !reply.content.is_empty() {
        blocks.push(json!({ "type": "text", "text": reply.content }));
    }
    for (name, arguments) in &reply.calls {
        let input: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        blocks.push(
            json!({ "type": "tool_use", "id": tool_use_id(app), "name": name, "input": input }),
        );
    }
    blocks
}

pub(super) async fn messages(State(app): State<Arc<App>>, body: Bytes) -> Response {
    match respond(app, body).await {
        Ok(response) => response,
        Err(err) => error(err),
    }
}

async fn respond(app: Arc<App>, body: Bytes) -> Result<Response, ApiError> {
    let request: MessagesRequest = parse_body(&body)?;
    let chat = chat_request(&request)?;
    let job = start_chat(&app, &chat)?;
    let id = app.next_id("msg");
    let model = app.server.model_name.clone();
    if request.stream {
        let (_, receiver) = job.receivers.into_iter().next().expect("one choice");
        let events = MessageStream::new(app, id, model, job.reading, job.prompt_tokens);
        return Ok(stream_response(events.run(receiver)));
    }
    let (results, usage) = collect(job.receivers, 1).await?;
    let result = &results[0];
    let reply = job.reading.read_all(&result.text);
    let (reason, stop_sequence) = stop_reason(
        result.finish.unwrap_or(FinishReason::Stop),
        reply.calls.len(),
        result.stop_string.as_deref(),
    );
    Ok(json_response(
        StatusCode::OK,
        &json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks(&app, &reply),
            "stop_reason": reason,
            "stop_sequence": stop_sequence,
            "usage": usage_json(&usage),
        }),
    ))
}

/// `/v1/messages/count_tokens`: how long the prompt would be.
pub(super) async fn count_tokens(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let counted = (|| {
        let request: MessagesRequest = parse_body(&body)?;
        let chat = chat_request(&request)?;
        prepare(|| prepare_chat(&app, &chat)).map(|prompt| prompt.tokens.len())
    })();
    match counted {
        Ok(tokens) => json_response(StatusCode::OK, &json!({ "input_tokens": tokens })),
        Err(err) => error(err),
    }
}

fn event(kind: &str, data: &Value) -> Result<Bytes, Infallible> {
    Ok(Bytes::from(format!("event: {kind}\ndata: {data}\n\n")))
}

fn stream_response(
    events: impl futures_util::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(events))
        .expect("the headers are valid")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
    ToolUse,
}

/// Turns a streamed reply into Anthropic's events: each kind of content gets a block of its own,
/// opened and closed around its deltas.
struct MessageStream {
    app: Arc<App>,
    id: String,
    model: String,
    reading: ReplyReading,
    prompt_tokens: usize,
    open: Option<Block>,
    blocks: usize,
    calls: usize,
}

impl MessageStream {
    fn new(
        app: Arc<App>,
        id: String,
        model: String,
        reading: ReplyReading,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            app,
            id,
            model,
            reading,
            prompt_tokens,
            open: None,
            blocks: 0,
            calls: 0,
        }
    }

    fn close(&mut self, out: &mut Vec<Result<Bytes, Infallible>>) {
        let Some(block) = self.open.take() else {
            return;
        };
        let index = self.blocks - 1;
        if block == Block::Thinking {
            let delta = json!({ "type": "signature_delta", "signature": "" });
            out.push(event(
                "content_block_delta",
                &json!({ "type": "content_block_delta", "index": index, "delta": delta }),
            ));
        }
        out.push(event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": index }),
        ));
    }

    /// Make sure a block of this kind is open, starting one with `start` if not.
    fn open(&mut self, block: Block, start: Value, out: &mut Vec<Result<Bytes, Infallible>>) {
        if self.open == Some(block) && block != Block::ToolUse {
            return;
        }
        self.close(out);
        out.push(event(
            "content_block_start",
            &json!({ "type": "content_block_start", "index": self.blocks, "content_block": start }),
        ));
        self.blocks += 1;
        self.open = Some(block);
    }

    fn delta(&self, delta: Value, out: &mut Vec<Result<Bytes, Infallible>>) {
        out.push(event(
            "content_block_delta",
            &json!({ "type": "content_block_delta", "index": self.blocks - 1, "delta": delta }),
        ));
    }

    fn part(&mut self, part: ReplyPart, out: &mut Vec<Result<Bytes, Infallible>>) {
        match part {
            ReplyPart::Reasoning(text) => {
                let start = json!({ "type": "thinking", "thinking": "", "signature": "" });
                self.open(Block::Thinking, start, out);
                self.delta(json!({ "type": "thinking_delta", "thinking": text }), out);
            }
            ReplyPart::Content(text) => {
                self.open(Block::Text, json!({ "type": "text", "text": "" }), out);
                self.delta(json!({ "type": "text_delta", "text": text }), out);
            }
            ReplyPart::Call { name, .. } => {
                self.calls += 1;
                let start = json!({
                    "type": "tool_use",
                    "id": tool_use_id(&self.app),
                    "name": name,
                    "input": {},
                });
                self.open(Block::ToolUse, start, out);
            }
            ReplyPart::Arguments { text, .. } => {
                self.delta(
                    json!({ "type": "input_json_delta", "partial_json": text }),
                    out,
                );
            }
        }
    }

    fn run(
        mut self,
        receiver: tokio::sync::mpsc::UnboundedReceiver<Event>,
    ) -> impl futures_util::Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
        let start = json!({ "type": "message_start", "message": {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": { "input_tokens": self.prompt_tokens, "output_tokens": 0 },
        }});
        let opening = stream::iter([event("message_start", &start)]);
        let mut reader = self.reading.reader();
        let events = stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|event| (event, receiver))
        })
        .flat_map(move |item| {
            let mut out = Vec::new();
            match item {
                Event::Text { text, .. } => {
                    for part in reader.read(Some(&text)) {
                        self.part(part, &mut out);
                    }
                }
                Event::Done {
                    reason,
                    usage,
                    stop_string,
                } => {
                    for part in reader.read(None) {
                        self.part(part, &mut out);
                    }
                    self.close(&mut out);
                    let (reason, stop_sequence) =
                        stop_reason(reason, self.calls, stop_string.as_deref());
                    let delta = json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": reason, "stop_sequence": stop_sequence },
                        "usage": usage_json(&usage),
                    });
                    out.push(event("message_delta", &delta));
                    out.push(event("message_stop", &json!({ "type": "message_stop" })));
                }
                Event::Embedding(_) => {}
                Event::Error(message) => {
                    let body = json!({ "type": "error", "error": {
                        "type": "api_error",
                        "message": message,
                    }});
                    out.push(event("error", &body));
                }
            }
            stream::iter(out)
        });
        opening.chain(events)
    }
}
