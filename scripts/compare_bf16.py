# How close is the 16-bit GPU path to the 32-bit reference, compared to Hugging Face's own 16-bit mode?
import json, sys, subprocess, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tool, model_dir, tmp = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_dir)
name = os.path.basename(model_dir)
ref32 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
ref16 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
chat = tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}], tokenize=False, add_generation_prompt=True)
prompts = ["The capital of France is", chat, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 3, "Once upon a time, in a land far away, there lived a young dragon who"]
n_gen = 48
for text in prompts:
    ids = tok(text, add_special_tokens=(text != chat))["input_ids"]
    t = torch.tensor([ids]).cuda()
    with torch.no_grad():
        a = ref32(t).logits[0].float().cpu().numpy()
        b = ref16(t).logits[0].float().cpu().numpy()
    out = os.path.join(tmp, "logits.bin")
    subprocess.run([tool, "logits", model_dir, out, *map(str, ids)], check=True)
    ours = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)
    with torch.no_grad():
        g32 = ref32.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=n_gen, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
    o = json.loads(subprocess.check_output([tool, "greedy", model_dir, str(n_gen), *map(str, ids)]))
    same = next((i for i, (x, y) in enumerate(zip(g32, o)) if x != y), n_gen)
    p = lambda x: torch.tensor(x).log_softmax(-1)
    kl = lambda x, y: float((p(x).exp() * (p(x) - p(y))).sum(-1).mean())
    print(f"{name} {len(ids):3d} tokens | logits vs 32-bit: ours {np.abs(ours-a).max():.3f} (HF 16-bit {np.abs(b-a).max():.3f}) | top-1 agrees: ours {(ours.argmax(-1)==a.argmax(-1)).mean():.1%} (HF 16-bit {(b.argmax(-1)==a.argmax(-1)).mean():.1%}) | KL: ours {kl(a, ours):.5f} (HF 16-bit {kl(a, b):.5f}) | greedy same for {same}/{n_gen}")
