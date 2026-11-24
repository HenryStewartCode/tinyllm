//! A router in front of several tinyllm servers (SGLang's router does the same job): each request
//! goes to the server most likely to have its prompt cached already, unless that would load the
//! servers unevenly, in which case it goes to the least busy one.
//!
//! The router can't see the servers' caches, so it keeps its own guess: the recent prompts it sent
//! each server, as text. A request's match with a server is the longest prefix it shares with one
//! of those. Replies stream through unchanged.

use anyhow::{Context, Result, ensure};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

const MAX_BODY_BYTES: usize = 64 << 20;
/// How much prompt text to remember per server.
const REMEMBERED_BYTES: usize = 32 << 20;
/// A request goes to the server sharing the longest prefix with it if that's at least this
/// fraction of its prompt (SGLang's `--cache-threshold`).
const CACHE_THRESHOLD: f64 = 0.3;
/// The servers count as unevenly loaded when the busiest has this many more requests than the
/// least busy, and this many times as many (SGLang's `--balance-abs-threshold` and
/// `--balance-rel-threshold`).
const BALANCE_ABS: usize = 64;
const BALANCE_REL: f64 = 1.5;
const HEALTH_INTERVAL: Duration = Duration::from_secs(5);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);

type HttpClient = Client<HttpConnector, Full<Bytes>>;

struct Worker {
    url: String,
    /// Requests sent to it that haven't finished (their replies are still streaming).
    load: AtomicUsize,
    healthy: AtomicBool,
    /// Recent prompts sent to it, newest first, and their total length.
    prompts: Mutex<(VecDeque<Arc<str>>, usize)>,
}

impl Worker {
    fn new(url: &str) -> Arc<Self> {
        Arc::new(Self {
            url: url.trim_end_matches('/').to_string(),
            load: AtomicUsize::new(0),
            healthy: AtomicBool::new(true),
            prompts: Mutex::new((VecDeque::new(), 0)),
        })
    }

    /// The longest prefix `text` shares with a prompt sent here, in bytes.
    fn matched(&self, text: &str) -> usize {
        let prompts = self.prompts.lock().unwrap();
        (prompts.0.iter())
            .map(|p| common_prefix(p.as_bytes(), text.as_bytes()))
            .max()
            .unwrap_or(0)
    }

    fn remember(&self, text: &str) {
        let mut guard = self.prompts.lock().unwrap();
        let (prompts, bytes) = &mut *guard;
        // A prompt that extends one already here replaces it.
        if let Some(i) = prompts.iter().position(|p| text.starts_with(&**p)) {
            *bytes -= prompts.remove(i).expect("found").len();
        }
        prompts.push_front(text.into());
        *bytes += text.len();
        while *bytes > REMEMBERED_BYTES {
            let Some(oldest) = prompts.pop_back() else {
                break;
            };
            *bytes -= oldest.len();
        }
    }

    fn forget(&self) {
        *self.prompts.lock().unwrap() = (VecDeque::new(), 0);
    }
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Decrements a worker's load when dropped: when its reply has been passed on, or abandoned.
struct InFlight(Arc<Worker>);

impl InFlight {
    fn new(worker: &Arc<Worker>) -> Self {
        worker.load.fetch_add(1, Ordering::Relaxed);
        Self(worker.clone())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.load.fetch_sub(1, Ordering::Relaxed);
    }
}

struct App {
    workers: RwLock<Vec<Arc<Worker>>>,
    client: HttpClient,
}

impl App {
    fn healthy(&self) -> Vec<Arc<Worker>> {
        (self.workers.read().unwrap().iter())
            .filter(|w| w.healthy.load(Ordering::Relaxed))
            .cloned()
            .collect()
    }

    /// Where a request with this prompt text should go.
    fn choose(&self, text: &str, skip: &[String]) -> Option<Arc<Worker>> {
        let workers: Vec<_> = (self.healthy().into_iter())
            .filter(|w| !skip.contains(&w.url))
            .collect();
        let load = |w: &Arc<Worker>| w.load.load(Ordering::Relaxed);
        // Ties go to the server holding the least, as new prompts have the most room there.
        let least = (workers.iter())
            .min_by_key(|w| (load(w), w.prompts.lock().unwrap().1))?
            .clone();
        let most = workers.iter().map(load).max().unwrap_or(0);
        let uneven =
            most - load(&least) > BALANCE_ABS && most as f64 > BALANCE_REL * load(&least) as f64;
        if text.is_empty() || uneven {
            return Some(least);
        }
        let (best, matched) = (workers.iter())
            .map(|w| (w, w.matched(text)))
            .max_by_key(|&(w, matched)| (matched, std::cmp::Reverse(load(w))))?;
        match matched as f64 >= CACHE_THRESHOLD * text.len() as f64 {
            true => Some(best.clone()),
            false => Some(least),
        }
    }
}

/// The text of a request's prompt, as far as the router needs it: every string in its messages,
/// prompt, system prompt or input, in order. Pictures count by their data, so the same picture
/// matches itself.
fn prompt_text(body: &[u8]) -> String {
    let Ok(Value::Object(fields)) = serde_json::from_slice::<Value>(body) else {
        return String::new();
    };
    let mut text = String::new();
    // The model first, so requests for different LoRA adapters don't count as sharing a prefix.
    for key in [
        "model",
        "lora_path",
        "system",
        "messages",
        "prompt",
        "text",
        "input",
    ] {
        if let Some(value) = fields.get(key) {
            collect_strings(value, &mut text);
        }
    }
    text
}

fn collect_strings(value: &Value, out: &mut String) {
    match value {
        Value::String(s) => {
            out.push_str(s);
            out.push('\u{1}');
        }
        Value::Array(items) => items.iter().for_each(|v| collect_strings(v, out)),
        Value::Object(fields) => fields.values().for_each(|v| collect_strings(v, out)),
        _ => {}
    }
}

/// Serve a router on `address` in front of the servers at `urls`.
pub fn serve(urls: Vec<String>, address: &str) -> Result<()> {
    ensure!(!urls.is_empty(), "a router needs at least one --worker");
    let client = Client::builder(TokioExecutor::new()).build_http();
    let app = Arc::new(App {
        workers: RwLock::new(urls.iter().map(|u| Worker::new(u)).collect()),
        client,
    });
    let router = Router::new()
        .route("/health", get(health))
        .route(
            "/workers",
            get(list_workers).post(add_worker).delete(remove_worker),
        )
        .route("/flush_cache", axum::routing::post(flush_cache))
        .fallback(forward)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(app.clone());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        tokio::spawn(check_health(app.clone()));
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .with_context(|| format!("couldn't listen on {address}"))?;
        eprintln!(
            "Routing http://{address} to {} server(s): {}",
            urls.len(),
            urls.join(", ")
        );
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .context("the router failed")
    })
}

/// Send `body` to `worker` as `method` `path`.
async fn send(
    app: &App,
    worker: &Worker,
    method: &Method,
    uri: &Uri,
    headers: &axum::http::HeaderMap,
    body: Bytes,
) -> Result<hyper::Response<hyper::body::Incoming>, hyper_util::client::legacy::Error> {
    let path = uri.path_and_query().map_or("/", |p| p.as_str());
    let mut request = hyper::Request::builder()
        .method(method.clone())
        .uri(format!("{}{path}", worker.url));
    for (name, value) in headers {
        if name != "host" && name != "content-length" {
            request = request.header(name, value);
        }
    }
    let request = request
        .body(Full::new(body))
        .expect("a valid request was received");
    app.client.request(request).await
}

/// Every other request: to the server chosen for it, with its reply streamed back.
async fn forward(State(app): State<Arc<App>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let Ok(body) = axum::body::to_bytes(body, MAX_BODY_BYTES).await else {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "the request is too large");
    };
    let text = match parts.method == Method::POST {
        true => prompt_text(&body),
        false => String::new(),
    };
    let mut tried = Vec::new();
    loop {
        let Some(worker) = app.choose(&text, &tried) else {
            return error(StatusCode::SERVICE_UNAVAILABLE, "no server is available");
        };
        let in_flight = InFlight::new(&worker);
        // Remembered now, so requests sharing the prefix that arrive meanwhile follow it.
        if !text.is_empty() {
            worker.remember(&text);
        }
        match send(
            &app,
            &worker,
            &parts.method,
            &parts.uri,
            &parts.headers,
            body.clone(),
        )
        .await
        {
            Ok(reply) => {
                let (mut parts, incoming) = reply.into_parts();
                if let Ok(url) = worker.url.parse() {
                    parts.headers.insert("x-tinyllm-worker", url);
                }
                // The load drops when the whole reply has gone through.
                let stream = incoming.into_data_stream().map(move |chunk| {
                    let _ = &in_flight;
                    chunk
                });
                return Response::from_parts(parts, Body::from_stream(stream));
            }
            // It couldn't be reached: try another, and leave it to the health check.
            Err(e) => {
                eprintln!("router: {} failed: {e}", worker.url);
                worker.healthy.store(false, Ordering::Relaxed);
                worker.forget();
                tried.push(worker.url.clone());
            }
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

async fn check_health(app: Arc<App>) {
    loop {
        let workers = app.workers.read().unwrap().clone();
        for worker in workers {
            let ok = tokio::time::timeout(HEALTH_TIMEOUT, async {
                let uri = format!("{}/health", worker.url).parse().ok()?;
                let reply = app.client.get(uri).await.ok()?;
                reply.status().is_success().then_some(())
            })
            .await
            .ok()
            .flatten()
            .is_some();
            if ok != worker.healthy.swap(ok, Ordering::Relaxed) {
                let state = if ok { "is back" } else { "is down" };
                eprintln!("router: {} {state}", worker.url);
                if !ok {
                    worker.forget();
                }
            }
        }
        tokio::time::sleep(HEALTH_INTERVAL).await;
    }
}

async fn health(State(app): State<Arc<App>>) -> StatusCode {
    match app.healthy().is_empty() {
        true => StatusCode::SERVICE_UNAVAILABLE,
        false => StatusCode::OK,
    }
}

async fn list_workers(State(app): State<Arc<App>>) -> Json<Value> {
    let workers = app.workers.read().unwrap();
    Json(json!({"workers": workers.iter().map(|w| json!({
        "url": w.url,
        "healthy": w.healthy.load(Ordering::Relaxed),
        "load": w.load.load(Ordering::Relaxed),
        "remembered_bytes": w.prompts.lock().unwrap().1,
    })).collect::<Vec<_>>()}))
}

fn url_of(body: &Value) -> Option<String> {
    let url = body.get("url")?.as_str()?;
    Some(url.trim_end_matches('/').to_string())
}

/// `POST /workers {"url": ...}`: add a server.
async fn add_worker(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let Some(url) = url_of(&body) else {
        return error(StatusCode::BAD_REQUEST, "give the server's \"url\"");
    };
    let mut workers = app.workers.write().unwrap();
    if !workers.iter().any(|w| w.url == url) {
        workers.push(Worker::new(&url));
    }
    StatusCode::OK.into_response()
}

/// `DELETE /workers {"url": ...}`: stop sending to a server. Requests it has already finish.
async fn remove_worker(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let Some(url) = url_of(&body) else {
        return error(StatusCode::BAD_REQUEST, "give the server's \"url\"");
    };
    let mut workers = app.workers.write().unwrap();
    let before = workers.len();
    workers.retain(|w| w.url != url);
    match workers.len() < before {
        true => StatusCode::OK.into_response(),
        false => error(StatusCode::NOT_FOUND, "no such server"),
    }
}

/// Empty every server's cache, and the router's guesses about them.
async fn flush_cache(State(app): State<Arc<App>>) -> StatusCode {
    let workers = app.workers.read().unwrap().clone();
    let mut ok = true;
    for worker in workers {
        worker.forget();
        let request = hyper::Request::post(format!("{}/flush_cache", worker.url))
            .body(Full::new(Bytes::new()))
            .expect("a valid request");
        ok &= app
            .client
            .request(request)
            .await
            .is_ok_and(|r| r.status().is_success());
    }
    match ok {
        true => StatusCode::OK,
        false => StatusCode::BAD_GATEWAY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(urls: &[&str]) -> App {
        App {
            workers: RwLock::new(urls.iter().map(|u| Worker::new(u)).collect()),
            client: Client::builder(TokioExecutor::new()).build_http(),
        }
    }

    #[test]
    fn prompts_follow_their_cache() {
        let app = app(&["http://a", "http://b"]);
        let long = "x".repeat(1000);
        let first = format!("{long} question one");
        let w = app.choose(&first, &[]).unwrap();
        w.remember(&first);
        // Sharing most of its prompt, a second request goes to the same server.
        let second = format!("{long} question two");
        assert_eq!(app.choose(&second, &[]).unwrap().url, w.url);
        // A different prompt goes to the least busy, which the first server isn't while busy.
        let _busy = InFlight::new(&w);
        assert_ne!(app.choose("something else", &[]).unwrap().url, w.url);
    }

    #[test]
    fn uneven_load_wins_over_the_cache() {
        let app = app(&["http://a", "http://b"]);
        let text = "y".repeat(500);
        let a = app.workers.read().unwrap()[0].clone();
        a.remember(&text);
        let busy: Vec<_> = (0..BALANCE_ABS + 1).map(|_| InFlight::new(&a)).collect();
        assert_eq!(app.choose(&text, &[]).unwrap().url, "http://b");
        drop(busy);
        assert_eq!(app.choose(&text, &[]).unwrap().url, "http://a");
    }

    #[test]
    fn prompt_text_reads_each_api() {
        let chat = br#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#;
        assert_eq!(prompt_text(chat), "m\u{1}user\u{1}text\u{1}hi\u{1}");
        assert_eq!(prompt_text(br#"{"prompt":"abc"}"#), "abc\u{1}");
        assert_eq!(prompt_text(b"not json"), "");
    }

    #[test]
    fn longer_prompts_replace_their_prefixes() {
        let w = Worker::new("http://a");
        w.remember("abc");
        w.remember("abcdef");
        assert_eq!(w.prompts.lock().unwrap().0.len(), 1);
        assert_eq!(w.matched("abcdxx"), 4);
    }
}
