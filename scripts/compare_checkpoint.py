"""Compare tinyllm with Hugging Face on a real checkpoint: logits at every position of a few
prompts, and greedy generations, on the CPU (32-bit) and the GPU (16-bit, and 32-bit if it fits).

Hugging Face's 32-bit reference runs on the CPU and its 16-bit one on the GPU, one at a time,
so this works for models that would be too big to hold both on the GPU at once.

usage: compare_checkpoint.py HARNESS MODEL_DIR [--no-gpu] [--no-gpu32]"""
import gc, json, os, subprocess, sys, tempfile
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tool, model_dir = sys.argv[1:3]
use_gpu = "--no-gpu" not in sys.argv and torch.cuda.is_available()
use_gpu32 = use_gpu and "--no-gpu32" not in sys.argv
torch.set_num_threads(min(8, os.cpu_count() or 4))
name = os.path.basename(model_dir.rstrip("/"))
tok = AutoTokenizer.from_pretrained(model_dir)
N_GEN = 32

texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3]
if tok.chat_template:
    texts.append(tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}], tokenize=False, add_generation_prompt=True))
prompts = [tok(text, add_special_tokens=(i < 2))["input_ids"] for i, text in enumerate(texts)]


def reference(dtype, device):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    out = []
    for ids in prompts:
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float().cpu().numpy()
            greedy = model.generate(t, max_new_tokens=N_GEN, do_sample=False, temperature=None, top_p=None, top_k=None,
                                    repetition_penalty=1.0, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        out.append((logits, greedy))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def ours(ids, **env):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logits.bin")
        environment = {**os.environ, **env}
        subprocess.run([tool, "logits", model_dir, path, *map(str, ids)], check=True, env=environment, capture_output=True)
        logits = np.fromfile(path, dtype=np.float32).reshape(len(ids), -1)
        greedy = json.loads(subprocess.run([tool, "greedy", model_dir, str(N_GEN), *map(str, ids)], check=True, env=environment,
                                           capture_output=True, text=True).stdout)
    return logits, greedy


def same_for(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


full = reference(torch.float32, "cpu")
half = reference(torch.bfloat16, "cuda") if use_gpu else None
runs = [("CPU", dict(DEVICE="cpu"))]
if use_gpu32:
    runs.append(("GPU 32-bit", dict(DEVICE="cuda", PRECISION="f32")))
if use_gpu:
    runs.append(("GPU 16-bit", dict(DEVICE="cuda", PRECISION="bf16")))
worst32 = 0.0
for p, ids in enumerate(prompts):
    a, greedy = full[p]
    line = f"{name} {len(ids):4d} tokens | range ±{np.abs(a).max():5.1f}"
    for label, env in runs:
        logits, generated = ours(ids, **env)
        diff = float(np.abs(logits - a).max())
        if "16" not in label:
            worst32 = max(worst32, diff)
        agree = (logits.argmax(-1) == a.argmax(-1)).mean()
        line += f" | {label}: {diff:.1e}, top-1 {agree:.0%}, greedy {same_for(generated, greedy)}/{N_GEN}"
    if half is not None:
        b, greedy16 = half[p]
        line += f" | HF 16-bit: {float(np.abs(b - a).max()):.1e}, top-1 {(b.argmax(-1) == a.argmax(-1)).mean():.0%}, greedy {same_for(greedy16, greedy)}/{N_GEN}"
    print(line, flush=True)
print(f"{name}: largest 32-bit difference {worst32:.1e}")
