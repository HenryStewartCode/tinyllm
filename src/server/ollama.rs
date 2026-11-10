//! Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, ...), as SGLang offers it, so that
//! tools built for Ollama can use a model served here. Replies stream as one JSON object per
//! line, and generation settings come in `options`.

use super::embed as embed_prompts;
use super::{
    ApiError, App, ChatRequest, OneOrMany, ReplyPart, SamplingFields, json_response, parse_body,
    prepare, start_chat,
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
use std::time::Instant;

/// The version of Ollama's API this follows.
const VERSION: &str = "0.12.0";

#[derive(Deserialize, Default)]
#[serde(default)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    min_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
    /// Tokens to generate; -1 for no limit, -2 to fill the context.
    num_predict: Option<i64>,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
}

#[derive(Deserialize)]
struct ChatBody {
    messages: Vec<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    options: Options,
    #[serde(default = "yes")]
    stream: bool,
    #[serde(default)]
    think: Option<Value>,
}

#[derive(Deserialize)]
struct GenerateBody {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    images: Option<Vec<Value>>,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    options: Options,
    #[serde(default = "yes")]
    stream: bool,
    #[serde(default)]
    raw: bool,
    #[serde(default)]
    think: Option<Value>,
}

fn yes() -> bool {
    true
}

/// An error in Ollama's format.
fn error(err: ApiError) -> Response {
    json_response(err.status, &json!({ "error": err.message }))
}

fn sampling(options: &Options, format: Option<&Value>) -> Result<SamplingFields, ApiError> {
    let response_format = match format {
        None | Some(Value::Null) => None,
        Some(Value::String(kind)) if kind == "json" => Some(json!({ "type": "json_object" })),
        Some(Value::String(kind)) if kind.is_empty() => None,
        Some(schema @ Value::Object(_)) => Some(json!({
            "type": "json_schema",
            "json_schema": { "name": "format", "schema": schema },
        })),
        Some(_) => {
            return Err(ApiError::invalid(
                "format must be \"json\" or a JSON schema",
            ));
        }
    };
    Ok(SamplingFields {
        max_tokens: options
            .num_predict
            .and_then(|n| usize::try_from(n).ok().filter(|&n| n > 0)),
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        min_p: options.min_p,
        seed: options.seed,
        stop: options.stop.clone().map(OneOrMany::Many),
        repetition_penalty: options.repeat_penalty,
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        response_format,
        ..Default::default()
    })
}

/// `think` as template settings: `true` (or a level such as "high") turns thinking on.
fn thinking_kwargs(think: Option<&Value>) -> Option<Map<String, Value>> {
    let on = match think? {
        Value::Bool(on) => *on,
        Value::String(_) => true,
        _ => return None,
    };
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), on.into());
    Some(kwargs)
}

/// Ollama's messages are OpenAI's, except that tool calls' arguments are objects and images
/// come separately.
fn openai_messages(messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::with_capacity(messages.len());
    for (i, message) in messages.iter().enumerate() {
        if message["images"]
            .as_array()
            .is_some_and(|images| !images.is_empty())
        {
            return Err(ApiError::invalid("images aren't supported by this model"));
        }
        let mut message = message.clone();
        if let Some(map) = message.as_object_mut() {
            map.remove("images");
            if let Some(thinking) = map.remove("thinking") {
                map.insert("reasoning_content".into(), thinking);
            }
            if let Some(Value::Array(calls)) = map.get_mut("tool_calls") {
                for (j, call) in calls.iter_mut().enumerate() {
                    call["type"] = "function".into();
                    if call.get("id").is_none() {
                        call["id"] = format!("call_{i}_{j}").into();
                    }
                }
            }
        }
        out.push(message);
    }
    // Ollama's tool messages don't say which call they answer: take them in order.
    let mut pending: Vec<String> = Vec::new();
    for message in &mut out {
        if let Some(calls) = message["tool_calls"].as_array() {
            pending = calls
                .iter()
                .filter_map(|c| c["id"].as_str().map(String::from))
                .rev()
                .collect();
        } else if message["role"] == "tool"
            && message.get("tool_call_id").is_none()
            && let Some(id) = pending.pop()
        {
            message["tool_call_id"] = id.into();
        }
    }
    Ok(out)
}

fn created_at() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

fn done_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

/// The timing and counting fields of a final reply. Durations are in nanoseconds.
fn stats(usage: &Usage, start: Instant, first: Option<Instant>) -> Map<String, Value> {
    let nanos = |d: std::time::Duration| d.as_nanos() as u64;
    let first = first.unwrap_or_else(Instant::now);
    let mut out = Map::new();
    out.insert("total_duration".into(), nanos(start.elapsed()).into());
    out.insert("load_duration".into(), 0.into());
    out.insert("prompt_eval_count".into(), usage.prompt_tokens.into());
    out.insert(
        "prompt_eval_duration".into(),
        nanos(first.duration_since(start)).into(),
    );
    out.insert("eval_count".into(), usage.completion_tokens.into());
    out.insert("eval_duration".into(), nanos(first.elapsed()).into());
    out
}

fn ndjson_response(
    lines: impl futures_util::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(lines))
        .expect("the headers are valid")
}

fn line(value: &Value) -> Result<Bytes, Infallible> {
    Ok(Bytes::from(format!("{value}\n")))
}

pub(super) async fn chat(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let result = (|| {
        let body: ChatBody = parse_body(&body)?;
        let request = ChatRequest {
            messages: openai_messages(&body.messages)?,
            tools: body.tools.clone(),
            chat_template_kwargs: thinking_kwargs(body.think.as_ref()),
            separate_reasoning: Some(true),
            sampling: sampling(&body.options, body.format.as_ref())?,
            ..Default::default()
        };
        Ok((body.stream, request))
    })();
    let (streaming, request) = match result {
        Ok(ok) => ok,
        Err(err) => return error(err),
    };
    match start_chat(&app, &request) {
        Ok(job) => reply(app, job, streaming, true).await,
        Err(err) => error(err),
    }
}

pub(super) async fn generate(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let body: GenerateBody = match parse_body(&body) {
        Ok(body) => body,
        Err(err) => return error(err),
    };
    if body
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty())
    {
        return error(ApiError::invalid("images aren't supported by this model"));
    }
    if body
        .suffix
        .as_ref()
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return error(ApiError::invalid("suffix isn't supported"));
    }
    // An empty prompt just loads the model, which is already loaded.
    if body.prompt.is_empty() {
        let done = json!({
            "model": app.server.model_name,
            "created_at": created_at(),
            "response": "",
            "done": true,
            "done_reason": "load",
        });
        return json_response(StatusCode::OK, &done);
    }
    let sampling = match sampling(&body.options, body.format.as_ref()) {
        Ok(sampling) => sampling,
        Err(err) => return error(err),
    };
    if body.raw || app.server.template.is_none() {
        return raw_generate(app, body, sampling).await;
    }
    let mut messages = Vec::new();
    if let Some(system) = &body.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.push(json!({ "role": "user", "content": body.prompt }));
    let request = ChatRequest {
        messages,
        chat_template_kwargs: thinking_kwargs(body.think.as_ref()),
        separate_reasoning: Some(true),
        sampling,
        ..Default::default()
    };
    match start_chat(&app, &request) {
        Ok(job) => reply(app, job, body.stream, false).await,
        Err(err) => error(err),
    }
}

/// `/api/generate` with `raw`: the prompt goes to the model as it is.
async fn raw_generate(app: Arc<App>, body: GenerateBody, fields: SamplingFields) -> Response {
    let prepared = prepare(|| {
        let tokens = app.server.tokenizer.encode(&body.prompt, true);
        let params = app.params(&fields, tokens.len())?;
        Ok((tokens, params))
    });
    let (tokens, params) = match prepared {
        Ok(ok) => ok,
        Err(err) => return error(err),
    };
    let receivers = app
        .start(&tokens, &params, 1)
        .into_iter()
        .enumerate()
        .collect();
    let job = super::ChatJob {
        receivers,
        reading: Default::default(),
        n: 1,
        prompt_tokens: tokens.len(),
    };
    reply(app, job, body.stream, false).await
}

/// Send a reply, as `/api/chat` (a `message`) or `/api/generate` (a `response`) does.
async fn reply(app: Arc<App>, job: super::ChatJob, streaming: bool, is_chat: bool) -> Response {
    let start = Instant::now();
    let model = app.server.model_name.clone();
    let (_, receiver) = job.receivers.into_iter().next().expect("one choice");
    let mut reader = job.reading.reader();
    // One line of the reply: text and thinking so far (since the last line, when streaming).
    let piece = move |content: &str, thinking: &str, calls: &[(String, String)]| -> Value {
        if is_chat {
            let mut message = json!({ "role": "assistant", "content": content });
            if !thinking.is_empty() {
                message["thinking"] = thinking.into();
            }
            if !calls.is_empty() {
                let calls: Vec<Value> = calls
                    .iter()
                    .map(|(name, arguments)| {
                        let arguments: Value =
                            serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                        json!({ "function": { "name": name, "arguments": arguments } })
                    })
                    .collect();
                message["tool_calls"] = calls.into();
            }
            json!({ "message": message })
        } else {
            let mut out = json!({ "response": content });
            if !thinking.is_empty() {
                out["thinking"] = thinking.into();
            }
            out
        }
    };
    let finish = move |mut value: Value, model: &str| -> Value {
        value["model"] = model.into();
        value["created_at"] = created_at().into();
        value
    };

    if !streaming {
        let mut text = String::new();
        let mut receiver = receiver;
        let mut first = None;
        while let Some(event) = receiver.recv().await {
            match event {
                Event::Text { text: more, .. } => {
                    first.get_or_insert_with(Instant::now);
                    text.push_str(&more);
                }
                Event::Done { reason, usage, .. } => {
                    let reply = job.reading.read_all(&text);
                    let mut value = piece(&reply.content, &reply.reasoning, &reply.calls);
                    value["done"] = true.into();
                    value["done_reason"] = done_reason(reason).into();
                    if let Some(map) = value.as_object_mut() {
                        map.extend(stats(&usage, start, first));
                    }
                    return json_response(StatusCode::OK, &finish(value, &model));
                }
                Event::Embedding(_) => {}
                Event::Error(message) => return error(ApiError::invalid(message)),
            }
        }
        return error(ApiError::internal("the engine stopped unexpectedly"));
    }

    // Tool calls go out whole, once their arguments are complete: at the end.
    let mut calls: Vec<(String, String)> = Vec::new();
    let mut first = None;
    let lines = stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|event| (event, receiver))
    })
    .flat_map(move |event| {
        let mut out = Vec::new();
        let mut send = |parts: Vec<ReplyPart>, out: &mut Vec<Result<Bytes, Infallible>>| {
            let (mut content, mut thinking) = (String::new(), String::new());
            for part in parts {
                match part {
                    ReplyPart::Reasoning(text) => thinking.push_str(&text),
                    ReplyPart::Content(text) => content.push_str(&text),
                    ReplyPart::Call { name, .. } => calls.push((name, String::new())),
                    ReplyPart::Arguments { index, text } => calls[index].1.push_str(&text),
                }
            }
            if !content.is_empty() || !thinking.is_empty() {
                let mut value = piece(&content, &thinking, &[]);
                value["done"] = false.into();
                out.push(line(&finish(value, &model)));
            }
        };
        match event {
            Event::Text { text, .. } => {
                first.get_or_insert_with(Instant::now);
                send(reader.read(Some(&text)), &mut out);
            }
            Event::Done { reason, usage, .. } => {
                send(reader.read(None), &mut out);
                if !calls.is_empty() {
                    let mut value = piece("", "", &calls);
                    value["done"] = false.into();
                    out.push(line(&finish(value, &model)));
                }
                let mut value = piece("", "", &[]);
                value["done"] = true.into();
                value["done_reason"] = done_reason(reason).into();
                if let Some(map) = value.as_object_mut() {
                    map.extend(stats(&usage, start, first));
                }
                out.push(line(&finish(value, &model)));
            }
            Event::Embedding(_) => {}
            Event::Error(message) => out.push(line(&json!({ "error": message }))),
        }
        stream::iter(out)
    });
    ndjson_response(lines)
}

/// What `/api/tags` and `/api/show` say about the model.
fn details(app: &App) -> Value {
    let params = app.server.num_params as f64;
    let size = match params {
        p if p >= 1e9 => format!("{:.1}B", p / 1e9),
        p => format!("{:.0}M", p / 1e6),
    };
    json!({
        "parent_model": "",
        "format": "safetensors",
        "family": app.server.model_type,
        "families": [app.server.model_type],
        "parameter_size": size,
        "quantization_level": "BF16",
    })
}

pub(super) async fn tags(State(app): State<Arc<App>>) -> Response {
    let name = &app.server.model_name;
    json_response(
        StatusCode::OK,
        &json!({ "models": [{
            "name": name,
            "model": name,
            "modified_at": created_at(),
            "size": app.server.num_params * 2,
            "digest": "",
            "details": details(&app),
        }]}),
    )
}

pub(super) async fn show(State(app): State<Arc<App>>) -> Response {
    let mut capabilities = vec!["completion"];
    if app.server.tool_format.is_some() {
        capabilities.push("tools");
    }
    if app.server.reasoning.is_some() {
        capabilities.push("thinking");
    }
    let template = app
        .server
        .template
        .as_ref()
        .map_or("", |template| template.source());
    json_response(
        StatusCode::OK,
        &json!({
            "modelfile": "",
            "parameters": "",
            "template": template,
            "details": details(&app),
            "model_info": {
                "general.architecture": app.server.model_type,
                "general.parameter_count": app.server.num_params,
            },
            "capabilities": capabilities,
        }),
    )
}

pub(super) async fn ps(State(app): State<Arc<App>>) -> Response {
    let name = &app.server.model_name;
    json_response(
        StatusCode::OK,
        &json!({ "models": [{
            "name": name,
            "model": name,
            "size": app.server.num_params * 2,
            "digest": "",
            "details": details(&app),
            "expires_at": "2999-01-01T00:00:00Z",
            "size_vram": 0,
        }]}),
    )
}

#[derive(Deserialize)]
struct EmbedBody {
    input: OneOrMany<String>,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    dimensions: Option<usize>,
}

/// Token ids for each text, cut to fit the context unless `truncate` is false.
fn embedding_prompts(app: &App, texts: &[String], truncate: bool) -> Vec<Vec<u32>> {
    let limit = app.context_limit().saturating_sub(1);
    texts
        .iter()
        .map(|text| {
            let mut tokens = app.server.tokenizer.encode(text, true);
            if truncate {
                tokens.truncate(limit);
            }
            tokens
        })
        .collect()
}

/// `/api/embed`: vectors for one or several texts.
pub(super) async fn embed(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let start = Instant::now();
    let body: EmbedBody = match parse_body(&body) {
        Ok(body) => body,
        Err(err) => return error(err),
    };
    let texts = match body.input {
        OneOrMany::One(text) => vec![text],
        OneOrMany::Many(texts) => texts,
    };
    let prompts = embedding_prompts(&app, &texts, body.truncate.unwrap_or(true));
    match embed_prompts(&app, &prompts, body.dimensions).await {
        Ok((vectors, tokens)) => json_response(
            StatusCode::OK,
            &json!({
                "model": app.server.model_name,
                "embeddings": vectors,
                "total_duration": start.elapsed().as_nanos() as u64,
                "load_duration": 0,
                "prompt_eval_count": tokens,
            }),
        ),
        Err(err) => error(err),
    }
}

#[derive(Deserialize)]
struct LegacyEmbedBody {
    prompt: String,
}

/// `/api/embeddings`, the older form: one text, one vector.
pub(super) async fn embeddings(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let body: LegacyEmbedBody = match parse_body(&body) {
        Ok(body) => body,
        Err(err) => return error(err),
    };
    let prompts = embedding_prompts(&app, &[body.prompt], true);
    match embed_prompts(&app, &prompts, None).await {
        Ok((vectors, _)) => json_response(StatusCode::OK, &json!({ "embedding": vectors[0] })),
        Err(err) => error(err),
    }
}

pub(super) async fn version() -> Response {
    json_response(StatusCode::OK, &json!({ "version": VERSION }))
}
