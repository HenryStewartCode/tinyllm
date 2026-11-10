"""Check LoRA adapters: a model running an adapter against the same model with the adapter merged
into its weights (see make_lora.py), which must agree to within rounding, and against the model
alone, which shouldn't, so that the adapter is seen to matter. Only numpy is needed.

usage: compare_lora.py HARNESS MODEL_DIR ADAPTER_DIR MERGED_DIR [DEVICE]

DEVICE (cpu, cuda or metal; default cpu) runs the adapter; the merged model runs on the CPU."""
import json, os, subprocess, sys, tempfile
import numpy as np

tool, model, adapter, merged = sys.argv[1:5]
device = sys.argv[5] if len(sys.argv) > 5 else "cpu"
texts = ["The capital of France is",
         "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3,
         "Once upon a time, in a land far away, there lived a dragon who loved to read. " * 6]
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "texts.json")
    json.dump(texts, open(path, "w"))
    tokens = json.loads(subprocess.run([tool, "tokenize", model, path], check=True, capture_output=True,
                                       text=True).stdout)["special"]

def run(model_dir, ids, device, lora=None):
    environment = {k: v for k, v in os.environ.items() if k != "LORA"}
    environment["DEVICE"] = device
    if lora:
        environment["LORA"] = lora
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "logits.bin")
        subprocess.run([tool, "logits", model_dir, out, *map(str, ids)], check=True, capture_output=True,
                       env=environment)
        logits = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)
    greedy = json.loads(subprocess.run([tool, "greedy", model_dir, "32", *map(str, ids)], check=True,
                                       capture_output=True, text=True, env=environment).stdout)
    return logits, greedy

same = lambda a, b: next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
for ids in tokens:
    with_adapter, greedy = run(model, ids, device, adapter)
    reference, reference_greedy = run(merged, ids, "cpu")
    alone, _ = run(model, ids, "cpu")
    print(f"{len(ids):4d} tokens | adapter on {device} vs merged: largest difference "
          f"{np.abs(with_adapter - reference).max():.2e}, top-1 {(with_adapter.argmax(-1) == reference.argmax(-1)).mean():.0%}, "
          f"greedy {same(greedy, reference_greedy)}/32 | the adapter changes logits by up to "
          f"{np.abs(alone - reference).max():.1f}")
