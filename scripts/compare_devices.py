"""Check one of tinyllm's backends against another on a real checkpoint: the logits at every
position of a few prompts, and 32 greedily generated tokens. No Python ML libraries needed, so it
runs anywhere the harness does (the CPU is the reference that the other scripts check against
transformers).

usage: compare_devices.py HARNESS MODEL_DIR [DEVICE [REFERENCE]]   (default: metal against cpu)

Extra environment variables (PRECISION, QUANTIZATION, THREADS) pass through to the harness."""
import json, os, subprocess, sys, tempfile
import numpy as np

tool, model = sys.argv[1:3]
device = sys.argv[3] if len(sys.argv) > 3 else "metal"
reference = sys.argv[4] if len(sys.argv) > 4 else "cpu"
texts = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3,
    "Once upon a time, in a land far away, there lived a dragon who loved to read. " * 6,
]

with tempfile.TemporaryDirectory() as tmp:
    strings = os.path.join(tmp, "strings.json")
    with open(strings, "w") as f:
        json.dump(texts, f)
    tokens = json.loads(subprocess.run([tool, "tokenize", model, strings], check=True,
                                       capture_output=True, text=True).stdout)["special"]


def run(ids, device):
    environment = {**os.environ, "DEVICE": device}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logits.bin")
        subprocess.run([tool, "logits", model, path, *map(str, ids)], check=True, env=environment,
                       capture_output=True)
        logits = np.fromfile(path, dtype=np.float32).reshape(len(ids), -1)
    greedy = json.loads(subprocess.run([tool, "greedy", model, "32", *map(str, ids)], check=True,
                                       env=environment, capture_output=True, text=True).stdout)
    return logits, greedy


worst = 0.0
for ids in tokens:
    got, got_greedy = run(ids, device)
    want, want_greedy = run(ids, reference)
    same = next((j for j, (x, y) in enumerate(zip(got_greedy, want_greedy)) if x != y), len(want_greedy))
    diff = np.abs(got - want).max()
    worst = max(worst, diff)
    print(f"{len(ids):4d} tokens | largest logit difference {diff:.2e} (scale {np.abs(want).max():.1f}) | "
          f"top-1 agrees {(got.argmax(-1) == want.argmax(-1)).mean():.0%} | greedy {same}/32 identical")
print(f"{device} against {reference}: largest difference {worst:.2e}")
