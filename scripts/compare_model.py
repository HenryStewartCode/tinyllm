import json, sys, subprocess, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tool, model_dir, tmp = sys.argv[1:4]
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
name = os.path.basename(model_dir)
chat = tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}], tokenize=False, add_generation_prompt=True)
prompts = ["The capital of France is", chat, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3]
for text in prompts:
    ids = tok(text, add_special_tokens=(text != chat))["input_ids"]
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].numpy()
    out = os.path.join(tmp, "logits.bin")
    subprocess.run([tool, "logits", model_dir, out, *map(str, ids)], check=True)
    ours = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)
    diff = np.abs(ours - ref)
    agree = (ours.argmax(-1) == ref.argmax(-1)).mean()
    print(f"{name}: {len(ids):3d} tokens | max |logit diff| {diff.max():.2e} (logit range ±{np.abs(ref).max():.1f}) | argmax agrees at {agree:.0%} of positions")
    # greedy continuation
    n = 48
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]), attention_mask=torch.ones_like(torch.tensor([ids])), max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None,
                           repetition_penalty=1.0, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
    o = json.loads(subprocess.check_output([tool, "greedy", model_dir, str(n), *map(str, ids)]))
    same = next((i for i, (a, b) in enumerate(zip(g, o)) if a != b), n)
    print(f"    greedy: first {same}/{n} generated tokens identical  -> {tok.decode(o[:24])!r}")
