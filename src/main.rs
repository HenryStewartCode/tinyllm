//! Command-line front end: load a model directory, then complete a prompt, chat, or serve.

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde_json::Value;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tinyllm::backend::{Backend, CpuBackend, LoadedModel};
use tinyllm::chat::{self, ChatTemplate};
use tinyllm::config::Config;
use tinyllm::cuda::{self, CudaBackend};
use tinyllm::engine::{
    self, EngineConfig, EngineHandle, Event, GenerateParams, Request, Speculative,
};
use tinyllm::reasoning::ReasoningFormat;
use tinyllm::sampler::SamplingParams;
use tinyllm::server::{self, SamplingDefaults, Server};
use tinyllm::tokenizer::Tokenizer;
use tinyllm::tools::ToolFormat;

/// What to run the model on.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Device {
    /// An NVIDIA GPU if there is one, otherwise the CPU
    Auto,
    Cpu,
    Cuda,
}

/// How precisely the GPU does its matrix multiplications.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PrecisionArg {
    /// 16-bit weights on tensor cores: much faster, and it's how most checkpoints are stored
    Bf16,
    /// 32-bit floats throughout
    F32,
}

/// How to read tool calls out of a model's replies.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ToolParserArg {
    /// Go by the model's chat template
    Auto,
    /// Leave tool calls as text
    None,
    /// <tool_call>{...}</tool_call>
    Qwen25,
    /// The same as qwen25
    Qwen,
    /// The same as qwen25
    Hermes,
    /// [TOOL_CALLS] [{...}]
    Mistral,
    /// {"name": ..., "parameters": ...}
    Llama3,
}

/// How to store weights in fewer bits.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum QuantizationArg {
    /// One byte per weight, with a scale per row, for the layers (GPU only for now)
    Int8,
}

/// How to guess tokens ahead (speculative decoding).
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SpeculativeArg {
    /// Don't guess
    None,
    /// Copy what followed the last few tokens earlier in the text (SGLang's NGRAM)
    Ngram,
}

/// How to separate a reasoning model's thinking from its answer.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReasoningParserArg {
    /// Go by the model's chat template and vocabulary
    Auto,
    /// Leave the thinking in the answer
    None,
    /// <think> ... </think>
    Qwen3,
    /// The same as qwen3; the template opening the block is noticed either way
    #[value(name = "deepseek-r1")]
    DeepseekR1,
}

/// Run a Llama-family language model (Llama, Mistral, Qwen, SmolLM, Phi-3, Granite, Gemma).
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Model directory holding config.json, tokenizer.json and *.safetensors (see download.sh)
    #[arg(short, long)]
    model: PathBuf,

    /// Text to continue. With --chat it is sent as one user message instead
    #[arg(short, long)]
    prompt: Option<String>,

    /// Use the model's chat template; without --prompt this starts an interactive chat
    #[arg(short, long)]
    chat: bool,

    /// System prompt for --chat (default: the model template's own, if it has one)
    #[arg(long)]
    system: Option<String>,

    /// Maximum number of tokens to generate per reply
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature; 0 always picks the most likely token (greedy)
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Sample only from the k most likely tokens (0 = no limit)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Sample only from the most likely tokens whose probabilities add up to p
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Drop tokens less than this fraction as likely as the most likely one
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,

    /// Penalty for repeating any of the last 64 tokens (1.0 = off)
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Random seed, for reproducible sampling (default: taken from the clock)
    #[arg(long)]
    seed: Option<u64>,

    /// Maximum context length of one conversation, in tokens (default: the model's maximum)
    #[arg(long)]
    ctx_size: Option<usize>,

    /// What to run the model on
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,

    /// Number precision on the GPU
    #[arg(long, value_enum, default_value_t = PrecisionArg::Bf16)]
    precision: PrecisionArg,

    /// Store the layers' weights in fewer bits: less memory, faster steps, a little rounding
    #[arg(long, value_enum)]
    quantization: Option<QuantizationArg>,

    /// Number of CPU worker threads (default: performance cores minus two)
    #[arg(long)]
    threads: Option<usize>,

    /// Print the prompt's token ids before generating
    #[arg(long)]
    show_tokens: bool,

    /// How many requests can run at the same time (default: 32 on CPU, 128 on GPU)
    #[arg(long)]
    parallel: Option<usize>,

    /// How many tokens go through the model in each step (default: 64 on CPU, 512 on GPU)
    #[arg(long)]
    batch_size: Option<usize>,

    /// Memory for the KV cache, in MiB (default on CPU: a quarter of RAM, at most 8 GiB, of which
    /// only the part in use is committed; on GPU: all that's left after loading the weights)
    #[arg(long)]
    cache_mb: Option<usize>,

    /// Run an OpenAI-compatible HTTP server
    #[arg(long)]
    serve: bool,

    /// Address for --serve to listen on
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port for --serve to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// How --serve reads tool calls out of the model's replies
    #[arg(long, value_enum, default_value_t = ToolParserArg::Auto)]
    tool_call_parser: ToolParserArg,

    /// How --serve separates a reasoning model's thinking (`reasoning_content`) from its answer
    #[arg(long, value_enum, default_value_t = ReasoningParserArg::Auto)]
    reasoning_parser: ReasoningParserArg,

    /// Guess tokens ahead and check them in one pass (speculative decoding). Replies come out
    /// exactly the same, sooner when the guesses are right
    #[arg(long, value_enum, default_value_t = SpeculativeArg::None)]
    speculative_algorithm: SpeculativeArg,

    /// The most tokens to guess ahead for one request
    #[arg(long, default_value_t = 6)]
    speculative_num_draft_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.serve && !args.chat && args.prompt.is_none() {
        bail!("nothing to do: pass --prompt \"some text\", --chat, or --serve");
    }
    let start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::load(&args.model)?);
    let device = match args.device {
        Device::Auto if CudaBackend::available() => Device::Cuda,
        Device::Auto => Device::Cpu,
        chosen => chosen,
    };
    let on_gpu = device == Device::Cuda;
    // On a GPU the threads only choose tokens between steps, so use what's there.
    let threads = args
        .threads
        .unwrap_or_else(|| match on_gpu {
            true => std::thread::available_parallelism().map_or(4, |n| n.get().min(16)),
            false => default_threads(),
        })
        .max(1);
    let parallel = args
        .parallel
        .unwrap_or(if on_gpu { 128 } else { 32 })
        .max(1);
    let batch_size = args
        .batch_size
        .unwrap_or(if on_gpu { 512 } else { 64 })
        .max(1);
    let cache_bytes = args.cache_mb.map(|mb| mb << 20);

    let cpu = |cache_bytes: Option<usize>| -> Result<(Config, usize, Box<dyn Backend>)> {
        let bytes = cache_bytes.unwrap_or_else(|| default_cache_mb() << 20);
        let LoadedModel {
            config,
            num_params,
            backend,
        } = CpuBackend::load(&args.model, threads, bytes)?;
        Ok((config, num_params, Box::new(backend)))
    };
    let (config, num_params, backend) = if on_gpu {
        let options = cuda::Options {
            quantization: args.quantization.map(|q| match q {
                QuantizationArg::Int8 => cuda::Quantization::Int8,
            }),
            precision: match args.precision {
                PrecisionArg::Bf16 => cuda::Precision::Bf16,
                PrecisionArg::F32 => cuda::Precision::F32,
            },
            max_batch: batch_size.max(parallel + 8),
            cache_bytes,
            host_threads: threads,
        };
        match CudaBackend::load(&args.model, &options) {
            Ok(LoadedModel {
                config,
                num_params,
                backend,
            }) => (config, num_params, Box::new(backend) as Box<dyn Backend>),
            Err(err) if args.device == Device::Auto => {
                eprintln!("Can't use the GPU, so falling back to the CPU: {err:#}");
                cpu(cache_bytes)?
            }
            Err(err) => return Err(err),
        }
    } else {
        cpu(cache_bytes)?
    };
    eprintln!(
        "Loaded {} model with {:.0}M parameters ({} layers, {} query / {} key-value heads) on {} in {:.1}s",
        config.model_type,
        num_params as f64 / 1e6,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.n_kv_heads(),
        backend.describe(),
        start.elapsed().as_secs_f64()
    );

    let engine_config = EngineConfig {
        threads,
        max_parallel: parallel,
        max_batch: batch_size,
        cache_tokens: backend.capacity(),
        ctx_size: args.ctx_size.unwrap_or(config.max_position_embeddings),
        speculative: match args.speculative_algorithm {
            SpeculativeArg::None => None,
            SpeculativeArg::Ngram => Some(Speculative {
                draft_tokens: args.speculative_num_draft_tokens,
                // Past this many rows a batch stops being nearly free.
                max_draft_rows: if on_gpu { 64 } else { 16 },
            }),
        },
    };
    let params = GenerateParams {
        max_tokens: args.max_tokens,
        sampling: SamplingParams {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            repeat_penalty: args.repeat_penalty,
            seed: args.seed.unwrap_or_else(clock_seed),
            ..Default::default()
        },
        stop_tokens: chat::stop_tokens(
            &args.model,
            config
                .eos_token_id
                .iter()
                .flat_map(|ids| ids.to_vec())
                .collect(),
            &tokenizer,
        ),
        ..Default::default()
    };
    let engine = engine::start(backend, tokenizer.clone(), engine_config, config.vocab_size);

    if args.serve {
        let model_name = args.model.file_name().map_or_else(
            || args.model.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        let template = ChatTemplate::load(&args.model)?;
        let detected = template.as_ref().and_then(ChatTemplate::tool_format);
        let tool_format = match args.tool_call_parser {
            ToolParserArg::Auto => detected,
            ToolParserArg::None => None,
            ToolParserArg::Qwen25 | ToolParserArg::Qwen | ToolParserArg::Hermes => {
                Some(ToolFormat::Hermes)
            }
            ToolParserArg::Mistral => match detected {
                Some(ToolFormat::MistralCompact) => detected,
                _ => Some(ToolFormat::Mistral),
            },
            ToolParserArg::Llama3 => Some(ToolFormat::Llama3),
        };
        match tool_format {
            Some(format) => eprintln!("Reading tool calls in the {format:?} format"),
            None => eprintln!("This model has no format for tool calls"),
        }
        let reasoning = match args.reasoning_parser {
            ReasoningParserArg::Auto => template
                .as_ref()
                .and_then(|t| ReasoningFormat::detect(t.source(), &tokenizer)),
            ReasoningParserArg::None => None,
            ReasoningParserArg::Qwen3 | ReasoningParserArg::DeepseekR1 => {
                Some(ReasoningFormat::think())
            }
        };
        if let Some(format) = &reasoning {
            eprintln!(
                "Separating reasoning between {} and {}",
                format.start, format.end
            );
        }
        return server::serve(
            Server {
                engine,
                tokenizer: tokenizer.clone(),
                template,
                model_name,
                stop_tokens: params.stop_tokens,
                defaults: SamplingDefaults::load(&args.model),
                vocab_size: config.vocab_size,
                tool_format,
                reasoning,
                model_type: config.model_type.clone(),
                num_params,
            },
            &format!("{}:{}", args.host, args.port),
        );
    }

    let cli = Cli {
        engine,
        tokenizer: tokenizer.clone(),
        params,
        show_tokens: args.show_tokens,
    };

    if args.chat {
        chat(&cli, &args)
    } else {
        let prompt = args.prompt.as_deref().unwrap_or_default();
        print!("{prompt}");
        cli.generate(tokenizer.encode(prompt, true), false)?;
        Ok(())
    }
}

/// How many threads to use by default. Using every core is counterproductive: a section of work
/// is only finished when its slowest thread is, so a thread that has to share its core with
/// other programs, or lands on an efficiency core, holds all the others up.
fn default_threads() -> usize {
    match performance_cores() {
        Some(cores) => cores.saturating_sub(2).max(1),
        None => std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .max(1),
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        old: *mut std::ffi::c_void,
        old_len: *mut usize,
        new: *mut std::ffi::c_void,
        new_len: usize,
    ) -> std::ffi::c_int;
}

#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    let (mut value, mut len) = (0i32, size_of::<i32>());
    // SAFETY: the name is a C string, and `value` and `len` describe a valid buffer.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

/// A quarter of RAM, at most 8 GiB.
fn default_cache_mb() -> usize {
    match total_memory_mb() {
        Some(total) => (total / 4).clamp(256, 8192),
        None => 4096,
    }
}

#[cfg(target_os = "macos")]
fn total_memory_mb() -> Option<usize> {
    let (mut value, mut len) = (0u64, size_of::<u64>());
    // SAFETY: the name is a C string, and `value` and `len` describe a valid buffer.
    let status = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0).then_some((value >> 20) as usize)
}

#[cfg(target_os = "linux")]
fn total_memory_mb() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn total_memory_mb() -> Option<usize> {
    None
}

fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

struct Cli {
    engine: EngineHandle,
    tokenizer: Arc<Tokenizer>,
    params: GenerateParams,
    show_tokens: bool,
}

impl Cli {
    /// Generate a reply to `tokens`, printing it as it arrives. Returns the reply's text.
    fn generate(&self, tokens: Vec<u32>, is_chat: bool) -> Result<String> {
        if self.show_tokens {
            let piece =
                |t: u32| String::from_utf8_lossy(&self.tokenizer.token_bytes(t)).into_owned();
            let pieces: Vec<String> = tokens
                .iter()
                .map(|&t| format!("{t}:{:?}", piece(t)))
                .collect();
            eprintln!("[{} prompt tokens] {}", tokens.len(), pieces.join(" "));
        }

        let params = GenerateParams {
            strip_leading_space: is_chat,
            ..self.params.clone()
        };
        let start = Instant::now();
        let mut events = self.engine.submit(Request { tokens, params });
        let mut stdout = std::io::stdout().lock();
        let mut reply = String::new();
        let mut first_token = None;
        while let Some(event) = events.blocking_recv() {
            match event {
                Event::Text { text, .. } => {
                    first_token.get_or_insert_with(Instant::now);
                    stdout.write_all(text.as_bytes())?;
                    stdout.flush()?;
                    reply.push_str(&text);
                }
                Event::Done { usage, .. } => {
                    writeln!(stdout)?;
                    let prompt_secs = first_token.unwrap_or(start).duration_since(start);
                    let reply_secs = first_token.map_or(0.0, |t| t.elapsed().as_secs_f64());
                    let processed = usage.prompt_tokens - usage.cached_tokens;
                    eprintln!(
                        "[prompt: {processed} tokens at {:.1} tok/s, {} cached | reply: {} tokens at {:.1} tok/s]",
                        processed as f64 / prompt_secs.as_secs_f64(),
                        usage.cached_tokens,
                        usage.completion_tokens,
                        usage.completion_tokens.saturating_sub(1) as f64 / reply_secs
                    );
                    return Ok(reply);
                }
                Event::Embedding(_) => {}
                Event::Error(message) => bail!(message),
            }
        }
        bail!("the engine stopped unexpectedly")
    }
}

fn chat(cli: &Cli, args: &Args) -> Result<()> {
    let template = ChatTemplate::load(&args.model)?
        .context("this model has no chat template; run without --chat to complete text instead")?;
    let system = args.system.as_ref().map(|s| chat::message("system", s));
    let mut history: Vec<Value> = system.into_iter().collect();

    if let Some(prompt) = &args.prompt {
        history.push(chat::message("user", prompt));
        return respond(cli, &template, &mut history);
    }

    eprintln!(
        "Chat started. Type a message and press Enter; /reset clears the conversation, /exit quits."
    );
    let initial_len = history.len();
    let mut stdin = std::io::stdin().lock();
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // end of input (Ctrl-D)
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.truncate(initial_len);
                eprintln!("[conversation cleared]");
            }
            message => {
                history.push(chat::message("user", message));
                if let Err(err) = respond(cli, &template, &mut history) {
                    history.pop();
                    eprintln!("error: {err:#} (try /reset)");
                }
            }
        }
    }
    Ok(())
}

/// Generate the assistant's next message and add it to `history`.
fn respond(cli: &Cli, template: &ChatTemplate, history: &mut Vec<Value>) -> Result<()> {
    let tokens = template.encode(&cli.tokenizer, &Value::Array(history.clone()))?;
    let reply = cli.generate(tokens, true)?;
    history.push(chat::message("assistant", &reply));
    Ok(())
}
