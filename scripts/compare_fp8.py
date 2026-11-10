"""Check an FP8 checkpoint: tinyllm's GPU (keeping the 8-bit floats) against its CPU (which
dequantizes them to 32 bits), and both against the 16-bit original the FP8 model was made from.

usage: compare_fp8.py HARNESS FP8_DIR ORIGINAL_DIR"""
import json, os, subprocess, sys, tempfile
import numpy as np
from transformers import AutoTokenizer

tool, fp8_dir, original_dir = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(fp8_dir)
texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3,
         tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}], tokenize=False, add_generation_prompt=True)]


def run(model_dir, ids, **env):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logits.bin")
        environment = {**os.environ, **env}
        subprocess.run([tool, "logits", model_dir, path, *map(str, ids)], check=True, env=environment, capture_output=True)
        logits = np.fromfile(path, dtype=np.float32).reshape(len(ids), -1)
        greedy = json.loads(subprocess.run([tool, "greedy", model_dir, "32", *map(str, ids)], check=True, env=environment,
                                           capture_output=True, text=True).stdout)
    return logits, greedy


for i, text in enumerate(texts):
    ids = tok(text, add_special_tokens=(i < 2))["input_ids"]
    cpu, cpu_greedy = run(fp8_dir, ids, DEVICE="cpu")
    gpu, gpu_greedy = run(fp8_dir, ids, DEVICE="cuda", PRECISION="bf16")
    orig, orig_greedy = run(original_dir, ids, DEVICE="cpu")
    same = lambda a, b: next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    print(f"{len(ids):3d} tokens | FP8 GPU vs FP8 CPU: {np.abs(gpu - cpu).max():.2e}, top-1 {(gpu.argmax(-1) == cpu.argmax(-1)).mean():.0%}, "
          f"greedy {same(gpu_greedy, cpu_greedy)}/32 | FP8 CPU vs 16-bit original: {np.abs(cpu - orig).max():.2e}, "
          f"top-1 {(cpu.argmax(-1) == orig.argmax(-1)).mean():.0%}, greedy {same(cpu_greedy, orig_greedy)}/32")
