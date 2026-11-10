//! An HTTP server with the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`),
//! so that existing clients and SDKs work unchanged, plus SGLang's native `/generate`.

use crate::chat::{self, ChatTemplate};
use crate::engine::{
    EngineHandle, Event, FinishReason, GenerateParams, Request, STACK_BYTES, Thinking,
    TokenLogprob, Usage,
};
use crate::grammar::{Grammar, Vocabulary};
use crate::lora::Adapters;
use crate::reasoning::{Piece, ReasoningFormat, ReasoningParser};
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use crate::tools::{self, ToolFormat, ToolOptions, ToolOutput, ToolParser, ToolReading};
use anyhow::{Context, Result};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use futures_util::stream::{self, BoxStream};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime};
use tokio::sync::mpsc::UnboundedReceiver;
use tower_http::cors::CorsLayer;

mod anthropic;
mod ollama;

/// Largest request body accepted, which is what limits prompt size in practice.
const MAX_BODY_BYTES: usize = 64 << 20;
const MAX_CHOICES: usize = 128;

/// Sampling settings the model's authors chose, from `generation_config.json`.
#[derive(Clone, Copy)]
pub struct SamplingDefaults {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
}

impl SamplingDefaults {
    pub fn load(dir: &Path) -> Self {
        let config: Value = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(Value::Null);
        let float = |name: &str, default: f32| config[name].as_f64().map_or(default, |v| v as f32);
        Self {
            temperature: float("temperature", 1.0),
            top_p: float("top_p", 1.0),
            top_k: config["top_k"].as_u64().unwrap_or(0) as usize,
            repeat_penalty: float("repetition_penalty", 1.0),
        }
    }
}

pub struct Server {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Option<ChatTemplate>,
    pub model_name: String,
    pub stop_tokens: HashSet<u32>,
    pub defaults: SamplingDefaults,
    pub vocab_size: usize,
    /// The notation the model writes tool calls in, if it has one.
    pub tool_format: Option<ToolFormat>,
    /// How the model marks its thinking, if it's a reasoning model.
    pub reasoning: Option<ReasoningFormat>,
    /// The model's architecture, as `config.json` names it.
    pub model_type: String,
    pub num_params: usize,
    /// The LoRA adapters requests can choose, by name.
    pub adapters: Adapters,
}

struct App {
    server: Server,
    created: u64,
    next_id: AtomicU64,
    vocabulary: Arc<Vocabulary>,
    /// For tool calls in a notation that needs special tokens, made if it's ever needed.
    marker_vocabulary: OnceLock<Arc<Vocabulary>>,
    /// Compiled grammars, so that repeating a schema doesn't compile it again.
    grammars: Mutex<HashMap<String, Arc<Grammar>>>,
}

/// Run the server until Ctrl-C.
pub fn serve(server: Server, address: &str) -> Result<()> {
    let vocabulary = Arc::new(Vocabulary::new(&server.tokenizer, server.vocab_size));
    let app = Arc::new(App {
        server,
        created: unix_time(),
        next_id: AtomicU64::new(1),
        vocabulary,
        marker_vocabulary: OnceLock::new(),
        grammars: Mutex::new(HashMap::new()),
    });
    let router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/generate", post(generate))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        .route("/api/ps", get(ollama::ps))
        .route("/api/version", get(ollama::version))
        .route("/api/embed", post(ollama::embed))
        .route("/api/embeddings", post(ollama::embeddings))
        .route("/", get(|| async { "tinyllm is running" }))
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/metrics", get(metrics))
        .route("/get_model_info", get(model_info))
        .route("/get_server_info", get(server_info))
        .route("/flush_cache", post(flush_cache))
        .route("/health_generate", get(health_generate))
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CorsLayer::permissive())
        .with_state(app);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(STACK_BYTES)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .with_context(|| format!("couldn't listen on {address}"))?;
        eprintln!("Serving an OpenAI-compatible API on http://{address}/v1");
        // Small writes are the whole point of streaming, and without this the kernel holds each
        // one back until the last has been acknowledged, which can take 40 ms.
        let listener = listener.tap_io(|connection| {
            let _ = connection.set_nodelay(true);
        });
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .context("the server failed")
    })
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// An error in OpenAI's format.
struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
    code: Option<&'static str>,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            code: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            code: None,
        }
    }

    fn body(&self) -> Value {
        json!({ "error": {
            "message": self.message,
            "type": self.kind,
            "param": null,
            "code": self.code,
        }})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        json_response(self.status, &self.body())
    }
}

fn json_response(status: StatusCode, value: &Value) -> Response {
    let mut response = (status, value.to_string()).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

fn parse_body<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|err| ApiError::invalid(format!("invalid request: {err}")))
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

/// A value that may be given as one item or a list of them.
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SamplingFields {
    #[serde(alias = "max_new_tokens", alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    min_p: Option<f32>,
    n: Option<usize>,
    stop: Option<OneOrMany<String>>,
    stop_token_ids: Option<Vec<u32>>,
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    logit_bias: Option<HashMap<String, f32>>,
    ignore_eos: bool,
    skip_special_tokens: Option<bool>,
    // Structured output.
    response_format: Option<Value>,
    regex: Option<String>,
    ebnf: Option<String>,
    json_schema: Option<Value>,
    // Which weights: the model, or one of its LoRA adapters, chosen by naming it as the model
    // or (as SGLang does it) in `lora_path`.
    model: Option<String>,
    lora_path: Option<String>,
}

impl SamplingFields {
    /// Whether the request sets its own constraint on the reply.
    fn constrains_output(&self) -> bool {
        self.regex.is_some()
            || self.ebnf.is_some()
            || self.json_schema.is_some()
            || self
                .response_format
                .as_ref()
                .is_some_and(|format| !matches!(format["type"].as_str(), None | Some("text")))
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StreamOptions {
    include_usage: bool,
}

/// A chat request in OpenAI's format, which the other chat APIs are translated into.
#[derive(Deserialize, Default)]
#[serde(default)]
struct ChatRequest {
    messages: Vec<Value>,
    stream: bool,
    stream_options: StreamOptions,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    chat_template_kwargs: Option<Map<String, Value>>,
    logprobs: bool,
    top_logprobs: Option<usize>,
    /// Return a reasoning model's thinking as `reasoning_content` (the default), rather than as
    /// part of `content`.
    separate_reasoning: Option<bool>,
    /// The most tokens a reasoning model may think for before it has to answer.
    thinking_budget: Option<usize>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// The ways `/v1/completions` allows a prompt to be written.
#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Texts(Vec<String>),
    TokenLists(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: Prompt,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    logprobs: Option<usize>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct NativeRequest {
    text: Option<String>,
    input_ids: Option<Vec<u32>>,
    stream: bool,
    sampling_params: SamplingFields,
    /// The LoRA adapter to use, by name.
    lora_path: Option<String>,
}

fn new_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

impl App {
    fn next_id(&self, prefix: &str) -> String {
        format!("{prefix}-{:016x}", self.next_id.fetch_add(1, Relaxed))
    }

    fn context_limit(&self) -> usize {
        let config = self.server.engine.config();
        config.ctx_size.min(config.cache_tokens)
    }

    /// Turn the API's sampling fields into engine parameters.
    fn params(
        &self,
        fields: &SamplingFields,
        prompt_len: usize,
    ) -> Result<GenerateParams, ApiError> {
        let defaults = self.server.defaults;
        let temperature = fields.temperature.unwrap_or(defaults.temperature);
        let top_p = fields.top_p.unwrap_or(defaults.top_p);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::invalid("temperature must be between 0 and 2"));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ApiError::invalid(
                "top_p must be greater than 0 and at most 1",
            ));
        }
        let limit = self.context_limit();
        if prompt_len >= limit {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!(
                    "the prompt is {prompt_len} tokens, but this model's context limit is {limit}"
                ),
                kind: "invalid_request_error",
                code: Some("context_length_exceeded"),
            });
        }
        let room = limit - prompt_len;
        let max_tokens = match fields.max_tokens {
            Some(0) => return Err(ApiError::invalid("max_tokens must be at least 1")),
            Some(n) => n.min(room),
            None => room,
        };

        let mut logit_bias = Vec::new();
        for (token, bias) in fields.logit_bias.iter().flatten() {
            let id = token.parse().map_err(|_| {
                ApiError::invalid(format!("logit_bias key '{token}' isn't a token id"))
            })?;
            logit_bias.push((id, *bias));
        }
        if let Some(n) = fields.n
            && !(1..=MAX_CHOICES).contains(&n)
        {
            return Err(ApiError::invalid(format!(
                "n must be between 1 and {MAX_CHOICES}"
            )));
        }
        let mut stop_tokens = self.server.stop_tokens.clone();
        for &id in fields.stop_token_ids.iter().flatten() {
            if id as usize >= self.server.vocab_size {
                return Err(ApiError::invalid(format!(
                    "stop token {id} is outside the vocabulary"
                )));
            }
            stop_tokens.insert(id);
        }
        let stop_strings: Vec<String> = match &fields.stop {
            None => Vec::new(),
            Some(OneOrMany::One(stop)) => vec![stop.clone()],
            Some(OneOrMany::Many(stops)) => stops.clone(),
        };
        let stop_strings = stop_strings.into_iter().filter(|s| !s.is_empty()).collect();

        let adapters = &self.server.adapters;
        let adapter = match (&fields.lora_path, &fields.model) {
            (Some(name), _) => adapters.find(name).ok_or_else(|| {
                ApiError::invalid(match adapters.is_empty() {
                    true => format!("there's no LoRA adapter named {name}: this server has none"),
                    false => format!(
                        "there's no LoRA adapter named {name}; this server has {}",
                        adapters.names().join(", ")
                    ),
                })
            })?,
            (None, Some(model)) => adapters.find(model).unwrap_or(0),
            (None, None) => 0,
        };

        Ok(GenerateParams {
            adapter,
            max_tokens,
            sampling: SamplingParams {
                temperature,
                top_k: match fields.top_k {
                    Some(k) => k.max(0) as usize,
                    None => defaults.top_k,
                },
                top_p,
                min_p: fields.min_p.unwrap_or(0.0),
                repeat_penalty: fields.repetition_penalty.unwrap_or(defaults.repeat_penalty),
                frequency_penalty: fields.frequency_penalty.unwrap_or(0.0),
                presence_penalty: fields.presence_penalty.unwrap_or(0.0),
                logit_bias,
                seed: fields.seed.unwrap_or_else(new_seed),
            },
            stop_tokens,
            stop_strings,
            ignore_eos: fields.ignore_eos,
            skip_special_tokens: fields.skip_special_tokens.unwrap_or(true),
            strip_leading_space: false,
            logprobs: None,
            grammar: self.grammar(fields)?,
            grammar_triggers: Vec::new(),
            keep_tokens: Vec::new(),
            thinking: None,
            embed: false,
        })
    }

    /// How a reply to `prompt` may think, for a reasoning model.
    fn thinking(&self, prompt: &str, budget: Option<usize>) -> Option<Thinking> {
        let format = self.server.reasoning.as_ref()?;
        let tokenizer = &self.server.tokenizer;
        let newline = tokenizer.encode("\n", false);
        Some(Thinking {
            open: format.opened_by(prompt),
            start_token: format.start_token(tokenizer),
            end: format.end.clone(),
            end_token: format.end_token(tokenizer),
            newline_token: (newline.len() == 1).then(|| newline[0]),
            budget,
        })
    }

    /// Compile the structured-output constraint a request asks for, if any.
    fn grammar(&self, fields: &SamplingFields) -> Result<Option<Arc<Grammar>>, ApiError> {
        let mut specs = Vec::new();
        if let Some(pattern) = &fields.regex {
            specs.push(format!("regex:{pattern}"));
        }
        if let Some(ebnf) = &fields.ebnf {
            specs.push(format!("ebnf:{ebnf}"));
        }
        if let Some(schema) = &fields.json_schema {
            let schema = match schema {
                // SGLang's native API passes the schema as a string.
                Value::String(text) => serde_json::from_str(text)
                    .map_err(|err| ApiError::invalid(format!("invalid json_schema: {err}")))?,
                other => other.clone(),
            };
            specs.push(format!("schema:{schema}"));
        }
        if let Some(format) = &fields.response_format {
            match format["type"].as_str() {
                None | Some("text") => {}
                Some("json_object") => specs.push("json".to_string()),
                Some("json_schema") => {
                    let schema = &format["json_schema"]["schema"];
                    if schema.is_null() {
                        return Err(ApiError::invalid(
                            "response_format needs json_schema.schema",
                        ));
                    }
                    specs.push(format!("schema:{schema}"));
                }
                Some(other) => {
                    return Err(ApiError::invalid(format!(
                        "unknown response_format '{other}'"
                    )));
                }
            }
        }
        if specs.len() > 1 {
            return Err(ApiError::invalid(
                "regex, ebnf, json_schema and response_format can't be combined",
            ));
        }
        let Some(spec) = specs.pop() else {
            return Ok(None);
        };
        self.compiled(spec, self.vocabulary.clone()).map(Some)
    }

    /// Compile a grammar, unless it's been done already.
    fn compiled(
        &self,
        spec: String,
        vocabulary: Arc<Vocabulary>,
    ) -> Result<Arc<Grammar>, ApiError> {
        if let Some(grammar) = self.grammars.lock().unwrap().get(&spec) {
            return Ok(grammar.clone());
        }
        let grammar = Grammar::compile(&spec, vocabulary)
            .map_err(|err| ApiError::invalid(format!("invalid grammar: {err:#}")))?;
        let grammar = Arc::new(grammar);
        let mut grammars = self.grammars.lock().unwrap();
        if grammars.len() >= 256 {
            grammars.clear();
        }
        grammars.insert(spec, grammar.clone());
        Ok(grammar)
    }

    /// The vocabulary for grammars of tool calls written in `format`.
    fn tool_vocabulary(&self, format: ToolFormat) -> Arc<Vocabulary> {
        let tokenizer = &self.server.tokenizer;
        let markers = format.marker_tokens(tokenizer);
        if markers.is_empty() {
            return self.vocabulary.clone();
        }
        self.marker_vocabulary
            .get_or_init(|| {
                let size = self.server.vocab_size;
                Arc::new(Vocabulary::with_markers(tokenizer, size, &markers))
            })
            .clone()
    }

    /// A new id for a tool call. These need to be different from those in a client's earlier
    /// conversations, including ones from before the server last started.
    fn call_id(&self, format: ToolFormat) -> String {
        let count = self.next_id.fetch_add(1, Relaxed);
        format.call_id(crate::sampler::draw_key(new_seed(), count as usize))
    }

    /// Start `n` generations of the same prompt, with different random seeds.
    fn start(
        &self,
        tokens: &[u32],
        params: &GenerateParams,
        n: usize,
    ) -> Vec<UnboundedReceiver<Event>> {
        (0..n)
            .map(|i| {
                let mut params = params.clone();
                params.sampling.seed = params.sampling.seed.wrapping_add(i as u64);
                self.server.engine.submit(Request {
                    tokens: tokens.to_vec(),
                    params,
                })
            })
            .collect()
    }

    fn token_text(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.server.tokenizer.token_bytes(id)).into_owned()
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.server.tokenizer.token_bytes(id)
    }

    /// Logprobs in the format of the chat API.
    fn chat_logprobs(&self, logprobs: &[TokenLogprob]) -> Value {
        let entry = |id: u32, logprob: f32| json!({ "token": self.token_text(id), "logprob": logprob, "bytes": self.token_bytes(id) });
        let content: Vec<Value> = logprobs
            .iter()
            .map(|l| {
                let mut value = entry(l.token, l.logprob);
                let top: Vec<Value> = l.top.iter().map(|&(id, p)| entry(id, p)).collect();
                value["top_logprobs"] = top.into();
                value
            })
            .collect();
        json!({ "content": content })
    }

    /// Logprobs in the format of the older completions API.
    fn completion_logprobs(&self, logprobs: &[TokenLogprob]) -> Value {
        let tokens: Vec<String> = logprobs.iter().map(|l| self.token_text(l.token)).collect();
        let token_logprobs: Vec<f32> = logprobs.iter().map(|l| l.logprob).collect();
        let top: Vec<Value> = logprobs
            .iter()
            .map(|l| {
                let map: Map<String, Value> = l
                    .top
                    .iter()
                    .map(|&(id, p)| (self.token_text(id), p.into()))
                    .collect();
                Value::Object(map)
            })
            .collect();
        json!({ "tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top })
    }
}

/// Run a piece of CPU-heavy request preparation (rendering, tokenizing, compiling a grammar),
/// letting the async runtime move other connections off this thread in the meantime.
fn prepare<T>(work: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
    tokio::task::block_in_place(work)
}

fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

/// The reason a chat reply ended. Ending normally after calling tools has its own name.
fn chat_finish_reason(reason: FinishReason, tool_calls: usize) -> &'static str {
    match (reason, tool_calls) {
        (FinishReason::Stop, 1..) => "tool_calls",
        _ => finish_reason(reason),
    }
}

fn usage_json(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": usage.cached_tokens },
    })
}

/// Combine the events of several generations, each tagged with its choice index.
fn merge(receivers: Vec<(usize, UnboundedReceiver<Event>)>) -> BoxStream<'static, (usize, Event)> {
    let streams = receivers.into_iter().map(|(index, receiver)| {
        stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|event| (event, receiver))
        })
        .map(move |event| (index, event))
        .boxed()
    });
    stream::select_all(streams).boxed()
}

/// One choice's result, for non-streaming responses.
#[derive(Default)]
struct Collected {
    text: String,
    logprobs: Vec<TokenLogprob>,
    finish: Option<FinishReason>,
    stop_string: Option<String>,
}

/// Wait for every generation to finish.
async fn collect(
    receivers: Vec<(usize, UnboundedReceiver<Event>)>,
    prompts: usize,
) -> Result<(Vec<Collected>, Usage), ApiError> {
    let count = receivers
        .iter()
        .map(|(index, _)| index + 1)
        .max()
        .unwrap_or(0);
    let mut results: Vec<Collected> = (0..count).map(|_| Collected::default()).collect();
    let mut usage = UsageTotal::new(prompts);
    let mut events = merge(receivers);
    while let Some((index, event)) = events.next().await {
        match event {
            Event::Text { text, logprobs } => {
                results[index].text.push_str(&text);
                results[index].logprobs.extend(logprobs);
            }
            Event::Done {
                reason,
                usage: u,
                stop_string,
            } => {
                results[index].finish = Some(reason);
                results[index].stop_string = stop_string;
                usage.add(index, &u);
            }
            Event::Embedding(_) => {}
            Event::Error(message) => return Err(ApiError::invalid(message)),
        }
    }
    Ok((results, usage.total))
}

/// Adds up usage across choices. Each prompt's tokens count once, however many choices it has.
struct UsageTotal {
    total: Usage,
    choices_per_prompt: usize,
}

impl UsageTotal {
    fn new(choices_per_prompt: usize) -> Self {
        Self {
            total: Usage::default(),
            choices_per_prompt: choices_per_prompt.max(1),
        }
    }

    fn add(&mut self, index: usize, usage: &Usage) {
        self.total.completion_tokens += usage.completion_tokens;
        if index.is_multiple_of(self.choices_per_prompt) {
            self.total.prompt_tokens += usage.prompt_tokens;
            self.total.cached_tokens += usage.cached_tokens;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------------------------

fn sse(value: &Value) -> Result<Bytes, Infallible> {
    Ok(Bytes::from(format!("data: {value}\n\n")))
}

fn sse_response(
    stream: impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .expect("the headers are valid")
}

/// Which of the OpenAI APIs a stream is for.
#[derive(Clone)]
struct ChunkFormat {
    is_chat: bool,
    id: String,
    created: u64,
    model: String,
}

impl ChunkFormat {
    fn chunk(&self, choices: Value) -> Value {
        json!({
            "id": self.id,
            "object": if self.is_chat { "chat.completion.chunk" } else { "text_completion" },
            "created": self.created,
            "model": self.model,
            "choices": choices,
        })
    }

    fn delta(&self, index: usize, text: &str, logprobs: Value) -> Value {
        if self.is_chat {
            self.chunk(json!([{
                "index": index,
                "delta": { "content": text },
                "logprobs": logprobs,
                "finish_reason": null,
            }]))
        } else {
            self.chunk(json!([{
                "index": index,
                "text": text,
                "logprobs": logprobs,
                "finish_reason": null,
            }]))
        }
    }

    /// More of a reasoning model's thinking.
    fn reasoning_delta(&self, index: usize, text: &str, logprobs: Value) -> Value {
        self.chunk(json!([{
            "index": index,
            "delta": { "reasoning_content": text },
            "logprobs": logprobs,
            "finish_reason": null,
        }]))
    }

    /// Part of a tool call, as an entry for the delta's `tool_calls`.
    fn call_delta(&self, index: usize, call: Value, logprobs: Value) -> Value {
        self.chunk(json!([{
            "index": index,
            "delta": { "tool_calls": [call] },
            "logprobs": logprobs,
            "finish_reason": null,
        }]))
    }

    fn end(&self, index: usize, reason: &str) -> Value {
        if self.is_chat {
            self.chunk(json!([{
                "index": index,
                "delta": {},
                "logprobs": null,
                "finish_reason": reason,
            }]))
        } else {
            self.chunk(json!([{
                "index": index,
                "text": "",
                "logprobs": null,
                "finish_reason": reason,
            }]))
        }
    }
}

/// What a chat reply is made of, as it streams.
#[derive(Debug)]
enum ReplyPart {
    /// More of a reasoning model's thinking.
    Reasoning(String),
    /// More of the answer.
    Content(String),
    /// A tool call begins.
    Call { index: usize, name: String },
    /// More of a tool call's arguments, as JSON text.
    Arguments { index: usize, text: String },
}

impl From<ToolOutput> for ReplyPart {
    fn from(output: ToolOutput) -> Self {
        match output {
            ToolOutput::Content(text) => ReplyPart::Content(text),
            ToolOutput::Call { index, name } => ReplyPart::Call { index, name },
            ToolOutput::Arguments { index, text } => ReplyPart::Arguments { index, text },
        }
    }
}

/// How to read a chat reply: where its thinking is, and whether it may call tools.
#[derive(Clone, Default)]
struct ReplyReading {
    tools: Option<ToolReading>,
    /// The model's reasoning format, and whether the prompt opened a block of thinking.
    thinking: Option<(ReasoningFormat, bool)>,
}

impl ReplyReading {
    /// Whether replies are plain text, with nothing to pick out of them.
    fn is_plain(&self) -> bool {
        self.tools.is_none() && self.thinking.is_none()
    }

    fn tool_format(&self) -> Option<ToolFormat> {
        self.tools.as_ref().map(|tools| tools.format)
    }

    fn reader(&self) -> ChoiceReader {
        ChoiceReader {
            thinking: self
                .thinking
                .as_ref()
                .map(|(format, open)| ReasoningParser::new(format.clone(), *open)),
            calls: self.tools.as_ref().map(ToolReading::parser),
        }
    }

    /// Read a whole reply at once.
    fn read_all(&self, text: &str) -> Reply {
        let mut reader = self.reader();
        let mut parts = reader.read(Some(text));
        parts.extend(reader.read(None));
        let mut reply = Reply::default();
        for part in parts {
            match part {
                ReplyPart::Reasoning(text) => reply.reasoning.push_str(&text),
                ReplyPart::Content(text) => reply.content.push_str(&text),
                ReplyPart::Call { name, .. } => reply.calls.push((name, String::new())),
                ReplyPart::Arguments { index, text } => reply.calls[index].1.push_str(&text),
            }
        }
        reply
    }
}

/// A whole chat reply, taken apart.
#[derive(Default)]
struct Reply {
    reasoning: String,
    content: String,
    /// Each tool call's name and arguments (as JSON text).
    calls: Vec<(String, String)>,
}

/// Reads one chat choice as it streams: its thinking, if it's a reasoning model, and its tool
/// calls, if it might make any.
struct ChoiceReader {
    thinking: Option<ReasoningParser>,
    calls: Option<ToolParser>,
}

impl ChoiceReader {
    /// Read more of the reply. `None` means it's finished.
    fn read(&mut self, text: Option<&str>) -> Vec<ReplyPart> {
        let pieces = match (&mut self.thinking, text) {
            (Some(thinking), Some(text)) => thinking.push(text),
            (Some(thinking), None) => thinking.finish(),
            (None, Some(text)) => vec![Piece::Content(text.to_string())],
            (None, None) => Vec::new(),
        };
        let mut parts = Vec::new();
        for piece in pieces {
            match (piece, &mut self.calls) {
                (Piece::Reasoning(text), _) => parts.push(ReplyPart::Reasoning(text)),
                (Piece::Content(text), Some(parser)) => {
                    parts.extend(parser.push(&text).into_iter().map(ReplyPart::from));
                }
                (Piece::Content(text), None) => parts.push(ReplyPart::Content(text)),
            }
        }
        if text.is_none()
            && let Some(parser) = &mut self.calls
        {
            parts.extend(parser.finish().into_iter().map(ReplyPart::from));
        }
        parts
    }

    fn calls_made(&self) -> usize {
        self.calls.as_ref().map_or(0, ToolParser::calls)
    }
}

struct StreamJob {
    app: Arc<App>,
    format: ChunkFormat,
    receivers: Vec<(usize, UnboundedReceiver<Event>)>,
    include_usage: bool,
    want_logprobs: bool,
    choices_per_prompt: usize,
    /// Text to send first on each choice (`echo`).
    prefixes: Vec<String>,
    /// What to pick out of each chat choice.
    reading: ReplyReading,
}

/// A piece of a chat reply as an OpenAI stream chunk.
fn part_chunk(
    app: &App,
    format: &ChunkFormat,
    index: usize,
    part: ReplyPart,
    logprobs: Value,
    tool_format: Option<ToolFormat>,
) -> Value {
    match part {
        ReplyPart::Reasoning(text) => format.reasoning_delta(index, &text, logprobs),
        ReplyPart::Content(text) => format.delta(index, &text, logprobs),
        ReplyPart::Call { index: call, name } => {
            let id = app.call_id(tool_format.expect("calls are only read with a format"));
            let call = json!({
                "index": call,
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": "" },
            });
            format.call_delta(index, call, logprobs)
        }
        ReplyPart::Arguments { index: call, text } => {
            let call = json!({ "index": call, "function": { "arguments": text } });
            format.call_delta(index, call, logprobs)
        }
    }
}

fn stream_events(job: StreamJob) -> impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
    let StreamJob {
        app,
        format,
        receivers,
        include_usage,
        want_logprobs,
        choices_per_prompt,
        prefixes,
        reading,
    } = job;
    let mut remaining = receivers.len();
    let mut usage = UsageTotal::new(choices_per_prompt);
    let tool_format = reading.tool_format();
    // Each chat choice's reader, and logprobs for text it hasn't let go of yet: tokens don't line
    // up with what gets sent, so they go with the next chunk out.
    let mut readers: Vec<(ChoiceReader, Vec<TokenLogprob>)> = match reading.is_plain() {
        true => Vec::new(),
        false => receivers
            .iter()
            .map(|_| (reading.reader(), Vec::new()))
            .collect(),
    };

    // Chat streams open by announcing the role.
    let opening: Vec<Result<Bytes, Infallible>> = if format.is_chat {
        receivers
            .iter()
            .map(|(index, _)| {
                sse(&format.chunk(json!([{
                    "index": index,
                    "delta": { "role": "assistant", "content": "" },
                    "logprobs": null,
                    "finish_reason": null,
                }])))
            })
            .collect()
    } else {
        prefixes
            .iter()
            .enumerate()
            .filter(|(_, text)| !text.is_empty())
            .map(|(index, text)| sse(&format.delta(index, text, Value::Null)))
            .collect()
    };

    let events = merge(receivers).flat_map(move |(index, event)| {
        let mut out = Vec::new();
        let send_parts = |reader: &mut (ChoiceReader, Vec<TokenLogprob>),
                          text: Option<&str>,
                          out: &mut Vec<Result<Bytes, Infallible>>| {
            for part in reader.0.read(text) {
                let logprobs = match want_logprobs {
                    true => app.chat_logprobs(&std::mem::take(&mut reader.1)),
                    false => Value::Null,
                };
                out.push(sse(&part_chunk(
                    &app,
                    &format,
                    index,
                    part,
                    logprobs,
                    tool_format,
                )));
            }
        };
        match event {
            Event::Text { text, logprobs } => {
                if let Some(reader) = readers.get_mut(index) {
                    reader.1.extend(logprobs);
                    send_parts(reader, Some(&text), &mut out);
                } else {
                    let logprobs = match (want_logprobs, format.is_chat) {
                        (false, _) => Value::Null,
                        (true, true) => app.chat_logprobs(&logprobs),
                        (true, false) => app.completion_logprobs(&logprobs),
                    };
                    out.push(sse(&format.delta(index, &text, logprobs)));
                }
            }
            Event::Done {
                reason, usage: u, ..
            } => {
                usage.add(index, &u);
                let reason = match readers.get_mut(index) {
                    Some(reader) => {
                        send_parts(reader, None, &mut out);
                        chat_finish_reason(reason, reader.0.calls_made())
                    }
                    None => finish_reason(reason),
                };
                out.push(sse(&format.end(index, reason)));
                remaining -= 1;
                if remaining == 0 && include_usage {
                    let mut chunk = format.chunk(json!([]));
                    chunk["usage"] = usage_json(&usage.total);
                    out.push(sse(&chunk));
                }
            }
            Event::Embedding(_) => {}
            Event::Error(message) => {
                out.push(sse(&ApiError::invalid(message).body()));
            }
        }
        stream::iter(out)
    });

    stream::iter(opening)
        .chain(events)
        .chain(stream::once(async {
            Ok(Bytes::from_static(b"data: [DONE]\n\n"))
        }))
}

// ---------------------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------------------

/// A chat request that's been started.
struct ChatJob {
    receivers: Vec<(usize, UnboundedReceiver<Event>)>,
    /// What to pick out of the replies.
    reading: ReplyReading,
    n: usize,
    prompt_tokens: usize,
}

/// A chat request, laid out and ready to start.
struct ChatPrompt {
    tokens: Vec<u32>,
    params: GenerateParams,
    reading: ReplyReading,
}

/// Lay out a conversation with the model's template and start generating replies. Every API
/// that chats comes through here.
fn start_chat(app: &App, request: &ChatRequest) -> Result<ChatJob, ApiError> {
    let ChatPrompt {
        tokens,
        params,
        reading,
    } = prepare_chat(app, request)?;
    let n = request.sampling.n.unwrap_or(1);
    let receivers = app
        .start(&tokens, &params, n)
        .into_iter()
        .enumerate()
        .collect();
    Ok(ChatJob {
        receivers,
        reading,
        n,
        prompt_tokens: tokens.len(),
    })
}

/// Lay out a conversation and work out how to generate and read the reply.
fn prepare_chat(app: &App, request: &ChatRequest) -> Result<ChatPrompt, ApiError> {
    let template = app.server.template.as_ref().ok_or_else(|| {
        ApiError::invalid("this model has no chat template; use /v1/completions instead")
    })?;
    if request.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    // Some templates only understand plain strings, so a text-only list of parts is joined.
    let mut messages: Vec<Value> = request.messages.iter().map(flatten_content).collect();
    tools::prepare_messages(&mut messages).map_err(ApiError::invalid)?;
    let plan = tools::plan(ToolOptions {
        tools: request.tools.as_ref(),
        tool_choice: request.tool_choice.as_ref(),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        format: app.server.tool_format,
        output_constrained: request.sampling.constrains_output(),
    })
    .map_err(ApiError::invalid)?;
    let tool_reading = plan.parser.clone();
    let (tokens, mut params) = prepare(|| {
        let text = template
            .render(
                &Value::Array(messages),
                plan.shown.as_ref(),
                request.chat_template_kwargs.as_ref(),
                true,
            )
            .map_err(|err| ApiError::invalid(format!("{err:#}")))?;
        let tokens = chat::encode_rendered(&app.server.tokenizer, &text);
        let mut params = app.params(&request.sampling, tokens.len())?;
        if let (Some(grammar), Some(reading)) = (&plan.grammar, &tool_reading) {
            let spec = format!(
                "calls:{}",
                serde_json::to_string(grammar).expect("it's plain data")
            );
            let vocabulary = app.tool_vocabulary(reading.format);
            params.grammar = Some(app.compiled(spec, vocabulary)?);
            params.grammar_triggers = plan.triggers.clone();
        }
        if let Some(reading) = &tool_reading {
            params.keep_tokens = reading.format.marker_tokens(&app.server.tokenizer);
        }
        params.thinking = app.thinking(&text, request.thinking_budget);
        Ok((tokens, params))
    })?;
    params.strip_leading_space = true;
    if request.logprobs {
        params.logprobs = Some(request.top_logprobs.unwrap_or(0));
    }
    // Where to put a reasoning model's thinking, and whether the prompt has already started it.
    let thinking = match (&app.server.reasoning, &params.thinking) {
        (Some(format), Some(thinking)) if request.separate_reasoning.unwrap_or(true) => {
            Some((format.clone(), thinking.open))
        }
        _ => None,
    };
    Ok(ChatPrompt {
        tokens,
        params,
        reading: ReplyReading {
            tools: tool_reading,
            thinking,
        },
    })
}

async fn chat_completions(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: ChatRequest = parse_body(&body)?;
    let job = start_chat(&app, &request)?;
    let format = ChunkFormat {
        is_chat: true,
        id: app.next_id("chatcmpl"),
        created: unix_time(),
        model: app.server.model_name.clone(),
    };
    if request.stream {
        return Ok(sse_response(stream_events(StreamJob {
            app: app.clone(),
            format,
            receivers: job.receivers,
            include_usage: request.stream_options.include_usage,
            want_logprobs: request.logprobs,
            choices_per_prompt: job.n,
            prefixes: Vec::new(),
            reading: job.reading,
        })));
    }

    let (results, usage) = collect(job.receivers, job.n).await?;
    let choices: Vec<Value> = results
        .iter()
        .enumerate()
        .map(|(index, result)| {
            let reply = job.reading.read_all(&result.text);
            json!({
                "index": index,
                "message": openai_message(&app, &reply, job.reading.tool_format()),
                "logprobs": if request.logprobs { app.chat_logprobs(&result.logprobs) } else { Value::Null },
                "finish_reason": result.finish.map(|reason| chat_finish_reason(reason, reply.calls.len())),
                "matched_stop": result.stop_string,
            })
        })
        .collect();
    Ok(json_response(
        StatusCode::OK,
        &json!({
            "id": format.id,
            "object": "chat.completion",
            "created": format.created,
            "model": format.model,
            "choices": choices,
            "usage": usage_json(&usage),
        }),
    ))
}

/// A finished reply as an OpenAI message.
fn openai_message(app: &App, reply: &Reply, tool_format: Option<ToolFormat>) -> Value {
    let mut message = json!({ "role": "assistant", "content": reply.content });
    if let Some(format) = tool_format.filter(|_| !reply.calls.is_empty()) {
        let calls: Vec<Value> = reply
            .calls
            .iter()
            .map(|(name, arguments)| {
                json!({
                    "id": app.call_id(format),
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                })
            })
            .collect();
        message["tool_calls"] = calls.into();
        if reply.content.is_empty() {
            message["content"] = Value::Null;
        }
    }
    if !reply.reasoning.is_empty() {
        message["reasoning_content"] = reply.reasoning.clone().into();
    }
    message
}

/// Replace a `content` list of `{"type": "text", "text": ...}` parts with one string.
fn flatten_content(message: &Value) -> Value {
    let mut message = message.clone();
    if let Some(parts) = message["content"].as_array()
        && parts.iter().all(|p| p["type"] == "text")
    {
        let text: String = parts.iter().filter_map(|p| p["text"].as_str()).collect();
        message["content"] = text.into();
    }
    message
}

async fn completions(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: CompletionRequest = parse_body(&body)?;
    let tokenizer = &app.server.tokenizer;
    let n = request.sampling.n.unwrap_or(1);
    let encode = |text: &str| tokenizer.encode(text, true);
    let (prompts, texts): (Vec<Vec<u32>>, Vec<String>) = prepare(|| {
        Ok(match &request.prompt {
            Prompt::Text(text) => (vec![encode(text)], vec![text.clone()]),
            Prompt::Texts(texts) => (texts.iter().map(|t| encode(t)).collect(), texts.clone()),
            Prompt::Tokens(tokens) => (vec![tokens.clone()], vec![tokenizer.decode(tokens)]),
            Prompt::TokenLists(lists) => (
                lists.clone(),
                lists.iter().map(|t| tokenizer.decode(t)).collect(),
            ),
        })
    })?;
    if prompts.iter().any(Vec::is_empty) {
        return Err(ApiError::invalid("the prompt is empty"));
    }

    let mut receivers = Vec::new();
    let mut prefixes = Vec::new();
    for (p, tokens) in prompts.iter().enumerate() {
        let mut params = prepare(|| app.params(&request.sampling, tokens.len()))?;
        params.logprobs = request.logprobs;
        for (i, receiver) in app.start(tokens, &params, n).into_iter().enumerate() {
            receivers.push((p * n + i, receiver));
            prefixes.push(if request.echo {
                texts[p].clone()
            } else {
                String::new()
            });
        }
    }

    let format = ChunkFormat {
        is_chat: false,
        id: app.next_id("cmpl"),
        created: unix_time(),
        model: app.server.model_name.clone(),
    };
    let want_logprobs = request.logprobs.is_some();
    if request.stream {
        return Ok(sse_response(stream_events(StreamJob {
            app: app.clone(),
            format,
            receivers,
            include_usage: request.stream_options.include_usage,
            want_logprobs,
            choices_per_prompt: n,
            prefixes,
            reading: ReplyReading::default(),
        })));
    }

    let (results, usage) = collect(receivers, n).await?;
    let choices: Vec<Value> = results
        .iter()
        .enumerate()
        .map(|(index, result)| {
            json!({
                "index": index,
                "text": format!("{}{}", prefixes[index], result.text),
                "logprobs": if want_logprobs { app.completion_logprobs(&result.logprobs) } else { Value::Null },
                "finish_reason": result.finish.map(finish_reason),
            })
        })
        .collect();
    Ok(json_response(
        StatusCode::OK,
        &json!({
            "id": format.id,
            "object": "text_completion",
            "created": format.created,
            "model": format.model,
            "choices": choices,
            "usage": usage_json(&usage),
        }),
    ))
}

/// SGLang's native API.
async fn generate(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: NativeRequest = parse_body(&body)?;
    let tokens = match (&request.input_ids, &request.text) {
        (Some(ids), _) => ids.clone(),
        (None, Some(text)) => prepare(|| Ok(app.server.tokenizer.encode(text, true)))?,
        (None, None) => return Err(ApiError::invalid("either text or input_ids is required")),
    };
    if tokens.is_empty() {
        return Err(ApiError::invalid("the prompt is empty"));
    }
    if request.sampling_params.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("/generate only supports n = 1 so far"));
    }
    let mut fields = request.sampling_params;
    fields.lora_path = request.lora_path.or(fields.lora_path);
    // SGLang's defaults for this endpoint, rather than the OpenAI ones.
    fields.max_tokens = fields.max_tokens.or(Some(128));
    fields.temperature = fields.temperature.or(Some(1.0));
    let params = prepare(|| app.params(&fields, tokens.len()))?;
    let id = app.next_id("gen");
    let start = Instant::now();
    let mut receiver = app
        .start(&tokens, &params, 1)
        .pop()
        .expect("one was started");

    let meta = move |usage: &Usage, reason: Option<FinishReason>, id: &str| {
        json!({
            "id": id,
            "finish_reason": reason.map(|r| json!({ "type": finish_reason(r) })),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": usage.cached_tokens,
            "e2e_latency": start.elapsed().as_secs_f64(),
        })
    };

    if request.stream {
        let stream = stream::unfold((receiver, String::new()), move |(mut receiver, mut text)| {
            let (id, meta) = (id.clone(), meta);
            async move {
                let event = receiver.recv().await?;
                let chunk = match event {
                    Event::Text { text: piece, .. } => {
                        text.push_str(&piece);
                        sse(&json!({ "text": text, "meta_info": meta(&Usage::default(), None, &id) }))
                    }
                    Event::Done { reason, usage, .. } => {
                        sse(&json!({ "text": text, "meta_info": meta(&usage, Some(reason), &id) }))
                    }
                    Event::Embedding(_) => unreachable!("only embedding requests get these"),
                    Event::Error(message) => sse(&ApiError::invalid(message).body()),
                };
                Some((chunk, (receiver, text)))
            }
        })
        .chain(stream::once(async {
            Ok(Bytes::from_static(b"data: [DONE]\n\n"))
        }));
        return Ok(sse_response(stream));
    }

    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        match event {
            Event::Text { text: piece, .. } => text.push_str(&piece),
            Event::Done { reason, usage, .. } => {
                return Ok(json_response(
                    StatusCode::OK,
                    &json!({ "text": text, "meta_info": meta(&usage, Some(reason), &id) }),
                ));
            }
            Event::Embedding(_) => {}
            Event::Error(message) => return Err(ApiError::invalid(message)),
        }
    }
    Err(ApiError::internal("the engine stopped unexpectedly"))
}

async fn models(State(app): State<Arc<App>>) -> Response {
    let base = &app.server.model_name;
    let mut data = vec![json!({
        "id": base,
        "object": "model",
        "created": app.created,
        "owned_by": "tinyllm",
        "max_model_len": app.context_limit(),
    })];
    // Each adapter is a model of its own, as vLLM lists them.
    for name in app.server.adapters.names() {
        data.push(json!({
            "id": name,
            "object": "model",
            "created": app.created,
            "owned_by": "tinyllm",
            "parent": base,
            "max_model_len": app.context_limit(),
        }));
    }
    json_response(StatusCode::OK, &json!({ "object": "list", "data": data }))
}

async fn model_info(State(app): State<Arc<App>>) -> Response {
    json_response(
        StatusCode::OK,
        &json!({ "model_path": app.server.model_name, "is_generation": true }),
    )
}

async fn server_info(State(app): State<Arc<App>>) -> Response {
    let config = app.server.engine.config();
    json_response(
        StatusCode::OK,
        &json!({
            "threads": config.threads,
            "max_running_requests": config.max_parallel,
            "max_batch_tokens": config.max_batch,
            "cache_tokens": config.cache_tokens,
            "context_length": app.context_limit(),
        }),
    )
}

async fn flush_cache(State(app): State<Arc<App>>) -> Response {
    app.server.engine.flush_cache();
    (StatusCode::OK, "Cache flushed.\n").into_response()
}

/// The ways `/v1/embeddings` allows input to be written.
#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenLists(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
}

/// Embed each prompt: the final hidden state of its last token, cut to `dimensions` if given
/// (for models trained to allow that), and scaled to length 1. They run as one batch. Returns
/// the vectors and how many tokens they took.
async fn embed(
    app: &App,
    prompts: &[Vec<u32>],
    dimensions: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let limit = app.context_limit();
    let mut receivers = Vec::with_capacity(prompts.len());
    for tokens in prompts {
        if tokens.is_empty() {
            return Err(ApiError::invalid("an input is empty"));
        }
        if tokens.len() >= limit {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!(
                    "an input is {} tokens, but this model's context limit is {limit}",
                    tokens.len()
                ),
                kind: "invalid_request_error",
                code: Some("context_length_exceeded"),
            });
        }
        let params = GenerateParams {
            embed: true,
            max_tokens: 1,
            ..Default::default()
        };
        receivers.push(
            app.start(tokens, &params, 1)
                .pop()
                .expect("one was started"),
        );
    }
    let mut vectors = Vec::with_capacity(prompts.len());
    let mut tokens = 0;
    for mut receiver in receivers {
        let mut vector = None;
        while let Some(event) = receiver.recv().await {
            match event {
                Event::Embedding(hidden) => vector = Some(hidden),
                Event::Done { usage, .. } => {
                    tokens += usage.prompt_tokens;
                    break;
                }
                Event::Error(message) => return Err(ApiError::invalid(message)),
                Event::Text { .. } => {}
            }
        }
        let mut vector = vector.ok_or_else(|| ApiError::internal("no embedding came back"))?;
        if let Some(dimensions) = dimensions {
            if dimensions == 0 || dimensions > vector.len() {
                return Err(ApiError::invalid(format!(
                    "dimensions must be between 1 and {}",
                    vector.len()
                )));
            }
            vector.truncate(dimensions);
        }
        let length = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if length > 0.0 {
            vector.iter_mut().for_each(|v| *v /= length);
        }
        vectors.push(vector);
    }
    Ok((vectors, tokens))
}

/// Standard base64, as OpenAI's `encoding_format: "base64"` sends vectors.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            match i <= chunk.len() {
                true => out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char),
                false => out.push('='),
            }
        }
    }
    out
}

/// OpenAI's `/v1/embeddings`.
async fn embeddings(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: EmbeddingRequest = parse_body(&body)?;
    let tokenizer = &app.server.tokenizer;
    let prompts: Vec<Vec<u32>> = prepare(|| {
        Ok(match &request.input {
            EmbeddingInput::Text(text) => vec![tokenizer.encode(text, true)],
            EmbeddingInput::Texts(texts) => {
                texts.iter().map(|t| tokenizer.encode(t, true)).collect()
            }
            EmbeddingInput::Tokens(tokens) => vec![tokens.clone()],
            EmbeddingInput::TokenLists(lists) => lists.clone(),
        })
    })?;
    if prompts.is_empty() {
        return Err(ApiError::invalid("input must not be empty"));
    }
    let base64_output = match request.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unknown encoding_format '{other}'"
            )));
        }
    };
    let (vectors, tokens) = embed(&app, &prompts, request.dimensions).await?;
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let embedding = match base64_output {
                true => {
                    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Value::String(base64(&bytes))
                }
                false => json!(vector),
            };
            json!({ "object": "embedding", "index": index, "embedding": embedding })
        })
        .collect();
    Ok(json_response(
        StatusCode::OK,
        &json!({
            "object": "list",
            "data": data,
            "model": app.server.model_name,
            "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
        }),
    ))
}

/// Healthy only if the model can actually generate a token.
async fn health_generate(State(app): State<Arc<App>>) -> Response {
    let tokens = app.server.tokenizer.encode("Hi", true);
    let params = GenerateParams {
        max_tokens: 1,
        ..Default::default()
    };
    let mut receiver = app
        .start(&tokens, &params, 1)
        .pop()
        .expect("one was started");
    while let Some(event) = receiver.recv().await {
        match event {
            Event::Done { .. } => return StatusCode::OK.into_response(),
            Event::Embedding(_) => {}
            Event::Error(message) => {
                return (StatusCode::SERVICE_UNAVAILABLE, message).into_response();
            }
            Event::Text { .. } => {}
        }
    }
    StatusCode::SERVICE_UNAVAILABLE.into_response()
}

#[derive(Deserialize)]
struct TokenizeRequest {
    prompt: OneOrMany<String>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
}

fn default_true() -> bool {
    true
}

/// SGLang's `/tokenize`: token ids for a text, or for each of a list of texts.
async fn tokenize(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: TokenizeRequest = parse_body(&body)?;
    let tokenizer = &app.server.tokenizer;
    let encode = |text: &str| tokenizer.encode(text, request.add_special_tokens);
    let (tokens, count) = match &request.prompt {
        OneOrMany::One(text) => {
            let tokens = encode(text);
            (json!(tokens), json!(tokens.len()))
        }
        OneOrMany::Many(texts) => {
            let lists: Vec<Vec<u32>> = texts.iter().map(|t| encode(t)).collect();
            let counts: Vec<usize> = lists.iter().map(Vec::len).collect();
            (json!(lists), json!(counts))
        }
    };
    Ok(json_response(
        StatusCode::OK,
        &json!({ "tokens": tokens, "count": count, "max_model_len": app.context_limit() }),
    ))
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    tokens: OneOrMany<Vec<u32>>,
}

/// SGLang's `/detokenize`: the text for a list of token ids, or for each of several lists.
async fn detokenize(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, ApiError> {
    let request: DetokenizeRequest = parse_body(&body)?;
    let tokenizer = &app.server.tokenizer;
    let vocab = app.server.vocab_size;
    let check = |ids: &[u32]| match ids.iter().find(|&&id| id as usize >= vocab) {
        Some(id) => Err(ApiError::invalid(format!(
            "token {id} is outside the vocabulary"
        ))),
        None => Ok(()),
    };
    let text = match &request.tokens {
        OneOrMany::One(ids) => {
            check(ids)?;
            json!(tokenizer.decode(ids))
        }
        OneOrMany::Many(lists) => {
            let mut texts = Vec::new();
            for ids in lists {
                check(ids)?;
                texts.push(tokenizer.decode(ids));
            }
            json!(texts)
        }
    };
    Ok(json_response(StatusCode::OK, &json!({ "text": text })))
}

/// Counters in Prometheus's text format.
async fn metrics(State(app): State<Arc<App>>) -> Response {
    let stats = app.server.engine.stats();
    let lines = [
        (
            "counter",
            "requests_total",
            "Requests received",
            stats.requests.load(Relaxed),
        ),
        (
            "counter",
            "prompt_tokens_total",
            "Prompt tokens received",
            stats.prompt_tokens.load(Relaxed),
        ),
        (
            "counter",
            "cached_prompt_tokens_total",
            "Prompt tokens served from the prefix cache",
            stats.cached_prompt_tokens.load(Relaxed),
        ),
        (
            "counter",
            "generated_tokens_total",
            "Tokens generated",
            stats.generated_tokens.load(Relaxed),
        ),
        (
            "counter",
            "steps_total",
            "Batches run through the model",
            stats.steps.load(Relaxed),
        ),
        (
            "counter",
            "batch_tokens_total",
            "Tokens run through the model",
            stats.rows.load(Relaxed),
        ),
        (
            "counter",
            "retractions_total",
            "Requests put back in the queue to free KV cache",
            stats.retractions.load(Relaxed),
        ),
        (
            "counter",
            "spec_drafted_tokens_total",
            "Tokens guessed ahead by speculative decoding",
            stats.drafted_tokens.load(Relaxed),
        ),
        (
            "counter",
            "spec_accepted_tokens_total",
            "Guessed tokens that turned out right",
            stats.accepted_tokens.load(Relaxed),
        ),
        (
            "gauge",
            "running_requests",
            "Requests in progress",
            stats.running.load(Relaxed),
        ),
        (
            "gauge",
            "queued_requests",
            "Requests waiting to start",
            stats.waiting.load(Relaxed),
        ),
        (
            "gauge",
            "cache_used_tokens",
            "Tokens held by the prefix cache",
            stats.cache_used.load(Relaxed),
        ),
        (
            "gauge",
            "cache_capacity_tokens",
            "Size of the KV cache in tokens",
            stats.cache_capacity.load(Relaxed),
        ),
    ];
    let mut out = format!(
        "# HELP tinyllm_step_seconds_total Time spent running batches\n# TYPE tinyllm_step_seconds_total counter\ntinyllm_step_seconds_total {}\n\
         # HELP tinyllm_forward_seconds_total The part of that spent in the model's forward pass\n# TYPE tinyllm_forward_seconds_total counter\ntinyllm_forward_seconds_total {}\n",
        stats.step_micros.load(Relaxed) as f64 / 1e6,
        stats.forward_micros.load(Relaxed) as f64 / 1e6
    );
    for (kind, name, help, value) in lines {
        out.push_str(&format!(
            "# HELP tinyllm_{name} {help}\n# TYPE tinyllm_{name} {kind}\ntinyllm_{name} {value}\n"
        ));
    }
    (StatusCode::OK, out).into_response()
}
