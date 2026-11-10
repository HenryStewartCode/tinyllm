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
