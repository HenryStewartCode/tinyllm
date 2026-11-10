# Verification scripts

How the claims in the main README were checked. Most compare tinyllm with Hugging Face
`transformers`, so they need Python with `torch`, `transformers` and `numpy` (and `aiohttp` or
`openai` for the server tests). The model comparisons drive the library directly through the
`harness` example:

```bash
cargo build --release --example harness      # target/release/examples/harness
```

`DEVICE=cpu|cuda|metal` and `PRECISION=bf16|f32` choose where the harness runs the model, and
`harness profile MODEL CONTEXT BATCH` shows where an Apple GPU's step time goes.

| Script | What it checks |
|---|---|
| `compare_random.py HARNESS WORKDIR [case...]` | Every supported architecture, with small random-weight models built by `transformers`: logits on the CPU and GPU against Hugging Face, and bit-identical results whether batched or alone. Needs the SmolLM2 tokenizer at the path at the top. |
| `compare_checkpoint.py HARNESS MODEL_DIR` | A real checkpoint: logits at every position and greedy generations, CPU and GPU, against Hugging Face in 32 and 16 bits. |
| `compare_int4.py HARNESS INT4_DIR WORKDIR` | A 4-bit AWQ or GPTQ checkpoint: unpacks its weights independently of tinyllm, runs `transformers` on them in 32 bits, and compares logits and greedy generations on the CPU and GPU. |
| `make_mxfp4.py GPT_OSS_DIR MXFP4_DIR UNPACKED_DIR` | A GPT-OSS checkpoint's experts packed in MXFP4, and the same values unpacked, which must give the same results. Needs only `numpy`. |
| `make_lora.py MODEL_DIR ADAPTER_DIR [--merged DIR]` | A random LoRA adapter for a model in PEFT's format, and optionally the model with it merged into the weights. Needs only `numpy`. |
| `compare_lora.py HARNESS MODEL_DIR ADAPTER_DIR MERGED_DIR [DEVICE]` | A model running an adapter against the merged model (they must agree) and against the model alone (they mustn't). `LORA=DIR` also makes the harness apply an adapter to every row. |
| `lora_test.py BASE_URL ADAPTER_1 ADAPTER_2` | Adapters through the server: mixed batches, the prefix cache, `model` and `lora_path`. |
| `compare_devices.py HARNESS MODEL_DIR [DEVICE [REFERENCE]]` | One backend against another (by default the Apple GPU against the CPU) on a real checkpoint: logits at every position and greedy generations. Needs only `numpy`. |
| `compare_model.py`, `compare_bf16.py` | Older versions of the same, holding both references on the GPU. |
| `compare_text.py HARNESS MODEL_DIR` | Tokenizer (ids and decoding of tricky strings) and chat template (text and ids) against `transformers`. |
| `reasoning_test.py BASE_URL` | A reasoning model through the server: `reasoning_content`, streaming, `enable_thinking`, thinking budgets, structured output and tool calls after the thinking. |
| `serve_test.py BASE_URL CHECK` | Server behaviour: determinism, structured output, the prefix cache, cancellation. |
| `tools_test.py`, `tools_forced.py`, `tools_schemas.py`, `tools_diff.py` | Tool calling through the official `openai` client, compared with SGLang. |
| `client.py BASE_URL/v1` | The `openai` client end to end. |
| `load.py BASE_URL MAX_TOKENS CONCURRENCY...` | Throughput at several levels of concurrency. |
| `ttft.py BASE_URL` | Time to first token for a long shared prompt, cold and cached. |
| `pressure.py` | Retraction and resumption when the KV cache is too small. |
