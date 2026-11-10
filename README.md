# tinyllm

A small LLM serving engine in Rust, built on the same ideas as
[SGLang](https://github.com/sgl-project/sglang): continuous batching, a radix-tree prefix cache
and fast structured output, behind OpenAI-, Anthropic- and Ollama-compatible APIs. It runs
Llama-style models (Llama 2/3, Qwen2.5, Qwen3, Mistral, SmolLM2/3, Phi-3, Granite, Gemma 1–3,
TinyLlama), reasoning models, and embedding models straight from Hugging Face checkpoints, on
NVIDIA GPUs or on the CPU, with no ML framework underneath. Everything from weight loading and
tokenization to the GPU and SIMD kernels and the HTTP server is here, in about 18,000 lines
including tests. See [How it compares with SGLang](#how-it-compares-with-sglang) for what's still
missing.

```text
$ cargo run --release -- -m models/SmolLM2-135M-Instruct --chat
Loaded llama model with 135M parameters (30 layers, 9 query / 3 key-value heads) in 0.1s
Chat started. Type a message and press Enter; /reset clears the conversation, /exit quits.

> What is the capital of France?
The capital of France is Paris.
[prompt: 37 tokens at 894.6 tok/s, 0 cached | reply: 8 tokens at 123.8 tok/s]

> And of Japan?
The capital of Japan is Tokyo.
[prompt: 15 tokens at 1115.5 tok/s, 44 cached | reply: 8 tokens at 135.5 tok/s]
```

## Quick start

```bash
./download.sh HuggingFaceTB/SmolLM2-135M-Instruct     # 270 MB, into models/SmolLM2-135M-Instruct
cargo run --release -- -m models/SmolLM2-135M-Instruct --chat
```

`download.sh` takes any Hugging Face repo name. Some good small models to try:

| Model | Download | Speed, one stream (M1 Max) | Notes |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 270 MB | ~140 tok/s | tiny and fast, not very smart. Verified. |
| `Qwen/Qwen2.5-0.5B-Instruct` | 990 MB | ~50 tok/s | noticeably better answers. Verified. |
| `meta-llama/Llama-3.2-1B-Instruct` | 2.5 GB | ~18 tok/s | gated (see below). Verified. |
| `Qwen/Qwen3-0.6B` | 1.5 GB | ~37 tok/s | a reasoning model: thinks before it answers. Verified. |
| `ibm-granite/granite-3.3-2b-instruct` | 5 GB | | Verified. |
| `HuggingFaceTB/SmolLM3-3B` | 6 GB | | reasoning model. Verified. |
| `microsoft/Phi-3-mini-4k-instruct` | 7.6 GB | | Verified. |
| `Qwen/Qwen3-Embedding-0.6B` | 1.2 GB | | an embedding model, for `/v1/embeddings`. Verified. |
| `HuggingFaceTB/SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct` | 720 MB, 3.4 GB | | same architecture as the 135M |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2.2 GB | | Llama 2 style tokenizer |
| `google/gemma-3-1b-it`, `gemma-2-2b-it` | 2 GB, 5 GB | | gated. Checked with random weights only. |

"Verified" means checked against Hugging Face's reference implementation (see
[Correctness](#correctness)); the others use the same architecture and tokenizer code but
haven't been run. The speeds are for the CPU; see [Performance](#performance) for a GPU.

Gated repos such as `meta-llama/Llama-3.2-1B-Instruct` need an access token:
`HF_TOKEN=hf_... ./download.sh meta-llama/Llama-3.2-1B-Instruct`.

## GPUs

With an NVIDIA GPU from 2020 or later (Ampere or newer) on Linux, tinyllm uses it automatically.
It needs only the NVIDIA driver: not the CUDA toolkit, to build or to run. It loads the driver when
it starts, and falls back to the CPU if there's no suitable GPU. `--device cpu` or `--device cuda`
makes the choice explicit, and `CUDA_VISIBLE_DEVICES` chooses the GPU (one for now).

Weights are 16-bit `bfloat16` on the GPU by default. That's how most checkpoints are stored, so
they're used exactly, and they run on the GPU's tensor cores, which multiply several times faster
and take half the memory. The one approximation is that the input to each matrix multiplication
gets rounded to 16 bits. `--precision f32` uses 32-bit numbers throughout instead, which is slower
and agrees with a 32-bit reference run as closely as the CPU does.

On the GPU, the KV cache is organized in pages of 16 tokens, and only whole pages are shared
between requests, so a cached prompt can have up to 15 more tokens to recompute than on the CPU.

The compiled kernels are in [`src/cuda/kernels.ptx`](src/cuda/kernels.ptx). To change them, edit
[`kernels.cu`](src/cuda/kernels.cu) and run [`build_kernels.sh`](src/cuda/build_kernels.sh), which
does need the CUDA toolkit.

## Serving

```bash
cargo run --release -- -m models/Qwen2.5-0.5B-Instruct --serve        # http://127.0.0.1:8000/v1
```

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
for chunk in client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Structured output guarantees a reply's shape. Use OpenAI's `response_format`, or SGLang's `regex`
and `ebnf` fields:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Invent a person."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"]
  }}}
}'
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | Runs the model's own chat template. Streaming, `n`, logprobs, tool calls, `reasoning_content`. |
| `POST /v1/completions` | Text or token prompts, one or many. Streaming, `n`, logprobs, `echo`. |
| `POST /v1/embeddings` | Text or token inputs, `dimensions`, `encoding_format: "base64"`. |
| `POST /v1/messages` | Anthropic's Messages API: content blocks, `tool_use`, extended thinking, streaming events. Also `/v1/messages/count_tokens`. |
| `POST /api/chat`, `/api/generate`, `/api/embed` | Ollama's API, streaming one JSON object per line. Also `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/embeddings`. |
| `GET /v1/models` | |
| `POST /generate` | SGLang's native API: `text` or `input_ids`, `sampling_params`, `stream`. |
| `POST /tokenize`, `/detokenize` | Token ids for text and back (also under `/v1`). |
| `GET /metrics` | Throughput, queue length and cache use, in Prometheus format. |
| `GET /health`, `/health_generate`, `/get_model_info`, `/get_server_info` | `/health_generate` checks that a token can actually be generated. |
| `POST /flush_cache` | Drop everything in the prefix cache that isn't in use. |

Requests can set `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k`,
`min_p`, `stop`, `stop_token_ids`, `seed`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `ignore_eos`,
`skip_special_tokens`, `stream_options.include_usage`, `response_format`, `regex`, `ebnf`,
`json_schema`, `chat_template_kwargs`, `tools`, `tool_choice`, `parallel_tool_calls`,
`separate_reasoning` and `thinking_budget`. Sampling settings left out come from the model's
`generation_config.json`. Replies say which stop string ended them, if one did, in
`matched_stop`.
`usage.prompt_tokens_details.cached_tokens` reports how much of the prompt came from the prefix
cache.

### Reasoning models

Models such as Qwen3, SmolLM3 and DeepSeek-R1 think between `<think>` and `</think>` before they
answer. As with SGLang's reasoning parsers, the thinking comes back as `reasoning_content`, in
streamed deltas too, and tool calls are only looked for in the answer. Structured output and
forced tool calls apply to the answer, after the thinking.

| | |
|---|---|
| `chat_template_kwargs: {"enable_thinking": false}` | Skip the thinking, for templates that offer that (Qwen3). |
| `thinking_budget: N` | Think for at most N tokens, after which the end of the thinking is forced. |
| `separate_reasoning: false` | Leave the thinking in `content`. |

Whether a model thinks is worked out from its chat template and vocabulary; `--reasoning-parser`
(`qwen3`, `deepseek-r1`, `none`) overrides that. A template that opens the block itself, or closes
an empty one to turn thinking off, is noticed from how the prompt ends. The Anthropic API's
`thinking` blocks and Ollama's `thinking` field come from the same place.

### Embeddings

Embedding models built like decoders, such as Qwen3-Embedding, gte-Qwen2 and e5-mistral, use the
final hidden state of the last token, scaled to length 1. `/v1/embeddings` runs the inputs as one
batch; `dimensions` cuts the vectors short first, for models trained to allow that.

```bash
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["The capital of China is Beijing.", "Gravity attracts bodies."]}'
```

### Speculative decoding

`--speculative-algorithm ngram` guesses up to `--speculative-num-draft-tokens` (6) tokens ahead by
copying what followed the last few tokens the previous time they appeared, as SGLang's NGRAM
algorithm does, and checks every guess in the same pass through the model. Each position is
sampled as ordinary decoding would sample it (every random draw depends only on the seed and the
position), so replies come out exactly the same, only sooner. On an RTX 2000 Ada, with prompts
that ask for code edits and copied text as well as poems and lists:

| | 1 request | 8 at once | 32 at once |
|---|---|---|---|
| Qwen2.5-0.5B, tok/s without / with | 162 / 259 (1.6×) | 1,000 / 1,389 (1.4×) | 1,517 / 2,202 (1.5×) |
| Qwen3-0.6B (mostly free-form thinking) | 133 / 156 (1.2×) | 843 / 915 (1.1×) | 2,150 / 2,153 (1.0×) |

All 20 replies were identical with and without guessing, in each case. Requests with grammars,
logprobs or thinking budgets don't guess, and the number of guesses per step is capped so that
large batches don't spend their time checking guesses.

### Tool calling

Pass `tools` as you would to OpenAI. Calls come back in `tool_calls`, a piece at a time if you're
streaming, and the model's answer to the results is one more request:

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}}]
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

reply = client.chat.completions.create(model="any", messages=messages, tools=tools).choices[0].message
messages.append(reply)
for call in reply.tool_calls or []:
    messages.append({"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 21}'})

print(client.chat.completions.create(model="any", messages=messages, tools=tools).choices[0].message.content)
```

| `tool_choice` | |
|---|---|
| `"auto"` (the default) | The model decides whether to call anything. |
| `"none"` | The model isn't shown the tools. |
| `"required"` | A grammar makes the model call at least one, with arguments that follow its schema. |
| `{"type": "function", "function": {"name": ...}}` | The same, with only that tool. |

Under `auto`, a tool with `"strict": true` has its arguments held to its schema, and
`parallel_tool_calls: false` allows at most one call. Both are done with a grammar that only
starts once a call does, so ordinary replies run at full speed. A schema that can't be followed is
an error for a strict tool; for any other tool, the arguments can be any JSON object instead.

The way the model writes calls comes from its chat template: `<tool_call>` tags (Qwen, Hermes),
`[TOOL_CALLS]` (Mistral, in both its layouts) or plain JSON (Llama 3). `--tool-call-parser` sets it
by hand, using SGLang's names (`qwen25`, `qwen`, `hermes`, `mistral`, `llama3`), and `none` leaves
calls as text. A template with no place for tools (SmolLM2, TinyLlama) means the model can't see
them, so `auto` just gets an answer, but `required` still works, using a plain JSON list.

Text that turns out not to be a call for one of the request's tools stays in `content`. Arguments
aren't guaranteed to be valid under `auto` without `strict`, as with OpenAI: check them before
acting on them. A forced reply from a Llama 3 model has only one call, since its templates can't
show more than that in a conversation.

## Command line

```bash
# Continue a piece of text (works with base and instruct models)
cargo run --release -- -m models/SmolLM2-135M-Instruct -p "Once upon a time"

# Ask one question using the model's chat format, then exit
cargo run --release -- -m models/Qwen2.5-0.5B-Instruct --chat -p "Explain recursion in one sentence."

# Interactive chat with a custom system prompt
cargo run --release -- -m models/Qwen2.5-0.5B-Instruct --chat --system "You are a pirate."
```

| Flag | Default | |
|---|---|---|
| `-m, --model DIR` | | directory with `config.json`, `tokenizer.json`, `*.safetensors` |
| `-p, --prompt TEXT` | | text to continue; with `--chat`, a single user message |
| `-c, --chat` | off | use the model's chat template; interactive without `--prompt` |
| `--serve` | off | run the HTTP server |
| `--host`, `--port` | 127.0.0.1, 8000 | where `--serve` listens |
| `--reasoning-parser P` | auto | `auto`, `none`, `qwen3` or `deepseek-r1`: how `--serve` finds a reasoning model's thinking |
| `--tool-call-parser P` | auto | how `--serve` reads tool calls (see [Tool calling](#tool-calling)) |
| `--device D` | auto | `auto`, `cuda` or `cpu` |
| `--precision P` | bf16 | on a GPU, `bf16` (16-bit weights) or `f32` |
| `--system TEXT` | template's own | system prompt for `--chat` |
| `-n, --max-tokens N` | 256 | tokens to generate per reply (command line only) |
| `-t, --temperature T` | 0.7 | 0 = greedy (always the most likely token) |
| `--top-k K` | 40 | sample from the K most likely tokens (0 = off) |
| `--top-p P` | 0.9 | ...and from the smallest set covering probability P |
| `--min-p P` | 0 | ...and drop tokens less than P times as likely as the best one |
| `--repeat-penalty R` | 1.0 | > 1 discourages repeating the last 64 tokens |
| `--seed N` | clock | for reproducible sampling |
| `--ctx-size N` | model max | longest allowed conversation, in tokens |
| `--threads N` | performance cores − 2 | CPU worker threads (on a GPU, all available, for choosing tokens) |
| `--parallel N` | 32 CPU, 128 GPU | requests in progress at once |
| `--speculative-algorithm A` | none | `ngram` to guess tokens ahead (see [Speculative decoding](#speculative-decoding)) |
| `--speculative-num-draft-tokens N` | 6 | the most tokens to guess ahead per request |
| `--batch-size N` | 64 CPU, 512 GPU | tokens run through the model per step |
| `--cache-mb N` | see right | KV cache size. CPU: ¼ of RAM up to 8 GiB, committed as used. GPU: what's left after loading |
| `--show-tokens` | off | print the prompt's token ids |

Sampling flags apply to the command line; the server takes those settings per request. Generated
text goes to stdout; timing and status go to stderr.

## How it works

Generating text is a loop: the model turns the tokens so far into a score (logit) for every
possible next token, the sampler picks one, and the new token is fed back in. A KV cache keeps
each earlier token's attention keys and values, so each step only runs new tokens through the
network. The forward pass, from [`model.rs`](src/model.rs):

```text
x = embed[token]
for each layer:
    x += attention(rmsnorm(x))   // grouped-query attention, RoPE on queries and keys
    x += mlp(rmsnorm(x))         // SwiGLU: down(silu(gate(x)) * up(x))
logits = lm_head(rmsnorm(x))
```

What makes it a server rather than a loop:

- **Continuous batching** ([`engine.rs`](src/engine.rs)). Every step is one batch: the next token
  of each generating sequence, plus as many prompt tokens as fit. Requests join and leave between
  steps. Speed on a CPU is mostly limited by how fast the weights can be read from memory, and a
  batch reads them once however many tokens it holds, so 16 streams together get several times
  the total throughput of one.
- **Prefix caching** ([`cache.rs`](src/cache.rs)), which SGLang calls RadixAttention. Every token's
  keys and values live in one pool of slots, and a radix tree maps token sequences to their slots.
  A new request starts from its longest cached prefix, and shares those slots rather than copying
  them. Old branches are evicted when the pool fills up, and if running requests need more room
  than there is, the newest one goes back in the queue and resumes later. Concurrent requests
  sharing a long prefix wait for the first one so the work isn't repeated.
- **Structured output** ([`grammar/`](src/grammar/mod.rs)). JSON schemas, regexes and EBNF all
  compile to a context-free grammar over bytes, run by a pushdown automaton whose states are
  discovered lazily and remembered, like a DFA. Walking a trie of the vocabulary alongside the
  automaton finds every allowed token in one sweep, and that's remembered per state too.
- **Determinism.** Kernels add numbers up in the same order whatever the batch looks like, so a
  seeded request gets exactly the same reply whether it runs alone or with 31 others.
- **GPU execution** ([`cuda/`](src/cuda/mod.rs)). The same forward pass, one kernel per step, with
  the driver loaded at run time. Weights are stored as contiguous 16×16 tiles for the tensor
  cores, and the KV cache lives in GPU memory. A forward pass is recorded as a CUDA graph and
  replayed for batches of up to 64. To avoid moving a whole vocabulary of scores across the bus
  for every token, a request sampled by temperature alone gets its token drawn on the GPU (adding
  Gumbel noise to the scores and taking the largest is the same as sampling from them); with
  top-k or top-p, the GPU first narrows each row to the tokens still in contention.

| File | What it does |
|---|---|
| [`src/server.rs`](src/server.rs) | The HTTP API. |
| [`src/engine.rs`](src/engine.rs) | The scheduler: admission, batching, sampling, streaming, stop conditions. |
| [`src/cache.rs`](src/cache.rs) | The KV slot pool and radix tree. |
| [`src/grammar/`](src/grammar/mod.rs) | Structured output: automaton, regex, EBNF, JSON schema. |
| [`src/backend.rs`](src/backend.rs) | What a device has to provide, and the CPU implementation. |
| [`src/cuda/`](src/cuda/mod.rs) | The NVIDIA backend: [loading the driver](src/cuda/api.rs), [the kernels](src/cuda/kernels.cu), and running a batch. |
| [`src/model.rs`](src/model.rs) | The transformer's batched forward pass on the CPU. |
| [`src/ops.rs`](src/ops.rs) | Numeric kernels: interleaved weight storage and a NEON matrix kernel, RMSNorm, RoPE, softmax, SwiGLU, vectorized `exp`. |
| [`src/pool.rs`](src/pool.rs) | A spinning fork-join thread pool: a forward pass is hundreds of parallel sections lasting microseconds each. |
| [`src/sampler.rs`](src/sampler.rs) | Temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, logit bias. |
| [`src/chat.rs`](src/chat.rs) | Runs each model's Jinja chat template the way `transformers` does. |
| [`src/tokenizer.rs`](src/tokenizer.rs) | A BPE tokenizer that reads Hugging Face `tokenizer.json`. |
| [`src/safetensors.rs`](src/safetensors.rs), [`src/config.rs`](src/config.rs) | Checkpoint loading. |
| [`src/main.rs`](src/main.rs) | The command line. |

## Performance

On an RTX 2000 Ada, a 70 W, 16 GB workstation GPU, compared with SGLang 0.5.23 serving the same
checkpoint on the same GPU through the same client. Requests ran concurrently and generated 128
tokens each, at temperature 0.8. SGLang was warmed up first, because it compiles a graph for each
batch size on first use. Total tokens per second:

| Concurrent requests | Qwen2.5-0.5B, tinyllm | Qwen2.5-0.5B, SGLang | SmolLM2-135M, tinyllm | SmolLM2-135M, SGLang |
|---|---|---|---|---|
| 1 | 162 | 179 | 402 | 478 |
| 4 | 640 | 676 | 1,575 | 1,686 |
| 16 | 2,449 | 1,039 | 6,026 | 852 |
| 64 | 7,644 | 3,382 | 16,736 | 3,334 |
| 128 | 10,035 | 5,454 | 23,345 | 6,273 |
| 256 | 10,840 | 7,941 | 26,062 | 11,032 |

| Time to first token, ~1,400-token prompt | tinyllm | SGLang |
|---|---|---|
| Qwen2.5-0.5B, first time | 86 ms | 87 ms |
| Qwen2.5-0.5B, prompt already cached | 13–14 ms | 18–21 ms |
| SmolLM2-135M, first time | 42 ms | 35 ms |
| SmolLM2-135M, prompt already cached | 10–12 ms | 17–22 ms |

So from 16 concurrent requests up, tinyllm is ahead by 1.4× to 7× on these two models, and for a
single stream SGLang is still about 10–20% faster. Long prompts are close to even. SGLang's own
numbers dip at 16 requests in every run, for reasons not looked into, so the largest ratios there
may say more about its default settings than about the GPU work.

These requests set only a temperature. SmolLM2 has no sampling defaults of its own, so its tokens
are drawn on the GPU. Qwen2.5's generation config adds top-k, top-p and a repetition penalty, so
there the GPU narrows each row to its candidates instead.

Most of a single stream's time is now reading the weights from memory, which every step has to do
once: about 1 GB for Qwen2.5 and 270 MB for SmolLM2. At the 200 GB/s this GPU sustains, that's
about 5 of Qwen2.5's 6 ms per token, and 1.3 of SmolLM2's 2.2 ms.

On the CPU, an M1 Max with 6 threads (other applications running), through the HTTP server:

| Concurrent requests | SmolLM2-135M, total tok/s | Qwen2.5-0.5B, total tok/s | Qwen2.5 per stream |
|---|---|---|---|
| 1 | 125 | 46 | 49 |
| 4 | 377 | 140 | 39 |
| 8 | 591 | 213 | 32 |
| 16 | 693 | 293 | 21 |
| 32 | 853 | 300 | 11 |

On the CPU too, the prefix cache matters more than anything else for long shared prompts. For a
request with a system prompt of about 1,400 tokens:

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| First request, time to first token | 1,775 ms | 3,718 ms |
| Later requests with that system prompt | 26 ms | 41 ms |

## Correctness

Checked against Hugging Face `transformers` (float32) and `tokenizers`:

- **Model:** for SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct, the
  logits at every position of three prompts (plain text, a chat prompt, code) match to within
  5e-4, with each prompt run as a single batch using deliberately scattered KV slots. The most
  likely next token agrees at every position, and 48-token greedy generations are identical token
  for token. The same holds for Qwen3-0.6B, Granite-3.3-2B, SmolLM3-3B and Phi-3-mini-4k (largest
  difference 4e-4, 32-token greedy generations identical on the CPU and GPU).
- **Every architecture:** random-weight models built by `transformers` for Llama, Qwen3 (with and
  without sliding windows), Mistral with a window, Phi-3, Granite, SmolLM3, Gemma, Gemma 2 (with
  soft-capping, and with 256-wide heads) and Gemma 3 (alternating sliding and global layers) match
  to within 4e-4 on the CPU and the GPU in 32 bits, and batching never changes a result.
- **Embeddings:** Qwen3-Embedding-0.6B gives vectors with cosine similarity 1.000000 to
  `transformers`' in 32 bits (largest difference 3e-7), on the CPU and GPU, and at least 0.99996 in
  16 bits.
- **Reasoning and the other APIs:** `scripts/reasoning_test.py`, `anthropic_test.py` and
  `ollama_test.py` check thinking, budgets, structured output and tool calls after thinking, and
  the Anthropic and Ollama APIs end to end, on Qwen3-0.6B.
- **Batching:** logits are bit-identical (millions compared, none different) whether a sequence
  runs alone or batched with others. Through the server, seeded samples are identical too,
  including when an undersized KV cache forces requests to be retracted and resumed.
- **GPU, 32-bit (`--precision f32`):** for SmolLM2 and Qwen2.5, logits match to within 2e-4 and
  48-token greedy generations are identical.
- **GPU, 16-bit (the default):** compared against 32-bit `transformers` on four prompts, the
  largest logit difference is 0.2 to 1.8, where `transformers`' own `bfloat16` mode differs by
  0.7 to 3.6. The most likely next token agrees at every position but in one prompt, at 97.3%,
  where `transformers` in `bfloat16` agrees just as often, and it manages 88–100% across the
  others. Greedy generations stay the same for all 48 tokens in 6 of the 8 cases, but they can
  diverge where two tokens are nearly tied.
- **GPU batching:** logits are bit-identical whether a sequence runs alone or batched, in both
  precisions, tensor cores included (24 million compared in each, none different).
- **Tokenizer:** identical token ids for 25 tricky strings (unicode, emoji, CJK, code, numbers,
  whitespace runs, special tokens), with and without special tokens, for five tokenizer styles:
  SmolLM2, Qwen2.5, Llama 3.2, TinyLlama (SentencePiece-style with byte fallback) and Mistral
  v0.3 (Metaspace). Decoding matches too. Qwen3, Granite, SmolLM3 and Phi-3 match on 26 more,
  checked against the `tokenizers` library reading `tokenizer.json` (`transformers` 5 rebuilds
  SmolLM2's tokenizer without its digit splitting, and differs from the file on one of them).
- **Chat templates:** token-identical to `apply_chat_template` for SmolLM2, Qwen2.5, Qwen3,
  Llama 3.2, Granite, SmolLM3, Phi-3 and Mistral v0.3, across single- and multi-turn conversations
  with and without tools, including
  earlier calls and their results as API clients send them. TinyLlama's
  rendered text is identical, but we add the `<s>` its training format expects and which its
  template leaves out, and `transformers` tokenizes the text after `</s>` slightly differently.
- **Structured output:** random walks through the allowed tokens, for JSON schemas, free-form
  JSON, regexes and EBNF, on all five vocabularies: every finished walk was valid (validated with
  Python's `jsonschema` and `re`).
- **Tool calling:** run through the official `openai` Python client on Qwen2.5 0.5B and Llama 3.2
  1B: each `tool_choice`, strict tools, parallel and limited calls, `n`, logprobs, cut-off replies,
  and whole conversations. Streamed replies came out the same as non-streamed ones, and of 256
  forced calls sent at once on each model all were valid. Random walks through the call grammars,
  for every notation on all five vocabularies, gave about 11,000 calls that the parser read back
  correctly and that followed their schemas. The Mistral notation has only been run on a model
  with random weights and the real tokenizer.
- **API:** exercised with the official `openai` Python client.

`cargo test` runs the unit tests. The scripts behind these checks are in
[`scripts/`](scripts/README.md), driving the library through the `harness` example.

## How it compares with SGLang

SGLang is a large project aimed at GPU clusters. The serving design here is the same, and on one
GPU, with the two models tested, throughput is ahead from 16 concurrent requests up. What's
missing is breadth, and the last 10–20% of single-stream speed.

| | SGLang | tinyllm |
|---|---|---|
| OpenAI-compatible API, native `/generate`, streaming | yes | yes |
| Anthropic and Ollama APIs | yes | yes |
| `/tokenize`, `/detokenize`, `/health_generate` | yes | yes |
| Continuous batching | yes | yes |
| Prefix caching with a radix tree, eviction, retraction | yes | yes |
| Longest-prefix-aware scheduling | yes | waits for in-flight prompts; otherwise first come, first served |
| JSON schema / regex / EBNF constraints | yes | yes |
| Model's own Jinja chat templates | yes | yes |
| Deterministic results across batch shapes | optional | always |
| Prometheus metrics | yes | yes |
| GPU execution | yes | **one NVIDIA GPU**; no AMD or Apple. Slightly slower for one stream, faster under load: see [Performance](#performance) |
| Reusable compiled batches (CUDA graphs) | yes | yes, for batches of up to 64 |
| Quantized weights (bf16, int8, GGUF) | yes | **bf16 on GPU only**, f32 on the CPU |
| Tool calling, with `tool_choice` and strict schemas | yes, with parsers for many model families | yes, for the families that run here: Qwen and Hermes, Mistral, Llama 3 |
| Reasoning parsers, thinking budgets | yes, for many families | yes, for `<think>` models (Qwen3, SmolLM3, DeepSeek-R1 and others) |
| Speculative decoding | yes: EAGLE, MTP, draft models, n-grams | n-grams only, with replies exactly as without it |
| Architectures | many, including mixture-of-experts | dense Llama-style (Llama, Mistral, Qwen2/3, SmolLM2/3, Phi-3, Granite, Gemma 1–3); **no mixture-of-experts** |
| Sliding-window attention, logit soft-capping | yes | yes |
| Embedding models | yes | yes, with last-token pooling |
| Vision, rerank, reward models | yes | **no** |
| Multi-GPU, multi-node, router | yes | **no** |
| The SGL frontend language (`gen`, `select`, `fork`) | yes | **no** |

Biggest gaps first, the plan is: quantized weights, mixture-of-experts models, overlapping CPU
scheduling with GPU work, a Metal backend for Apple GPUs, LoRA adapters, then draft-model
speculative decoding.

Other limitations: only BPE tokenizers are supported (not SentencePiece Unigram or WordPiece,
though that covers the Llama family), embeddings only use last-token pooling, and sliding windows
limit what attention sees but don't yet save KV cache memory.
