"""Check a 4-bit AWQ or GPTQ checkpoint: unpack its weights here, independently of tinyllm (following
AutoAWQ's and GPTQ's own layouts), run `transformers` in 32 bits on the result, and compare
tinyllm's logits and greedy generations with that, on the CPU and the GPU.

usage: compare_int4.py HARNESS INT4_DIR WORKDIR"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, torch
from safetensors.numpy import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tool, source, work = sys.argv[1:4]
config = json.load(open(os.path.join(source, "config.json")))
quant = config.pop("quantization_config")
method, group = quant["quant_method"], quant.get("group_size", 128)
assert quant.get("bits", 4) == 4

def nibbles(words):
    """[..., n] int32 -> [..., n * 8] values, nibble k of each word at position 8 * i + k."""
    words = words.astype(np.uint32)
    return np.stack([(words >> (4 * k)) & 15 for k in range(8)], axis=-1).reshape(*words.shape[:-1], -1)

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def awq_unpack(words):
    """AutoAWQ packs columns in the order 0, 2, 4, 6, 1, 3, 5, 7; undo that."""
    values = nibbles(words)
    order = np.arange(values.shape[-1]).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    return values[..., order]

tensors = {}
for name in sorted(os.listdir(source)):
    if name.endswith(".safetensors"):
        tensors.update(load_file(os.path.join(source, name)))
state = {}
for name, value in tensors.items():
    if name.endswith((".qzeros", ".scales", ".g_idx")):
        continue
    if not name.endswith(".qweight"):
        state[name] = value.astype(np.float32) if value.dtype != np.int32 else value
        continue
    prefix = name[: -len(".qweight")]
    scales = tensors[prefix + ".scales"].astype(np.float32)          # [groups, out]
    if method == "awq":
        q = awq_unpack(value)                                         # [in, out]
        zeros = awq_unpack(tensors[prefix + ".qzeros"])               # [groups, out]
        g_idx = np.arange(q.shape[0]) // (group if group > 0 else q.shape[0])
    else:
        q = nibbles(value.T).T                                        # [in / 8, out] -> [in, out]
        offset = 0 if quant.get("checkpoint_format") == "gptq_v2" else 1
        zeros = nibbles(tensors[prefix + ".qzeros"]) + offset         # [groups, out]
        g_idx = tensors.get(prefix + ".g_idx", np.arange(q.shape[0]) // group).astype(np.int64)
    weight = (q.astype(np.float32) - zeros[g_idx].astype(np.float32)) * scales[g_idx]
    state[prefix + ".weight"] = np.ascontiguousarray(weight.T)        # [out, in]

full = os.path.join(work, "dequantized")
shutil.rmtree(full, ignore_errors=True)
model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**config), dtype=torch.float32).eval()
missing, unexpected = model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=False)
missing = [m for m in missing if not (m == "lm_head.weight" and config.get("tie_word_embeddings"))]
assert not missing and not unexpected, (missing, unexpected)
if config.get("tie_word_embeddings"):
    model.tie_weights()

tok = AutoTokenizer.from_pretrained(source)
texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3,
         tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}], tokenize=False, add_generation_prompt=True)]

def ours(ids, **env):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logits.bin")
        environment = {**os.environ, **env}
        subprocess.run([tool, "logits", source, path, *map(str, ids)], check=True, env=environment, capture_output=True)
        logits = np.fromfile(path, dtype=np.float32).reshape(len(ids), -1)
        greedy = json.loads(subprocess.run([tool, "greedy", source, "32", *map(str, ids)], check=True, env=environment,
                                           capture_output=True, text=True).stdout)
    return logits, greedy

same = lambda a, b: next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
for i, text in enumerate(texts):
    ids = tok(text, add_special_tokens=(i < 2))["input_ids"]
    with torch.no_grad():
        reference = model(torch.tensor([ids])).logits[0].numpy()
        generated = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                   max_new_tokens=32, do_sample=False)[0, len(ids):].tolist()
    line = f"{len(ids):3d} tokens"
    for label, env in [("CPU", dict(DEVICE="cpu")), ("GPU", dict(DEVICE="cuda", PRECISION="bf16"))]:
        if label == "GPU" and not torch.cuda.is_available():
            continue
        logits, greedy = ours(ids, **env)
        line += (f" | {label}: largest difference {np.abs(logits - reference).max():.2e}, top-1 "
                 f"{(logits.argmax(-1) == reference.argmax(-1)).mean():.0%}, greedy {same(greedy, generated)}/32")
    print(line)
