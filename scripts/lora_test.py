"""Check LoRA adapters through the server: requests choosing different adapters (and none) in the
same batches get exactly the replies they get alone, a prompt cached for one adapter isn't reused
for another, and each way of choosing an adapter works.

usage: lora_test.py BASE_URL ADAPTER_1 ADAPTER_2   (the names the server was given with --lora)"""
import json, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

base, first, second = sys.argv[1:4]
failures = 0

def post(path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    parse = lambda text: json.loads(text) if text.strip().startswith((b"{", b"[")) else {}
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, parse(response.read())
    except urllib.error.HTTPError as error:
        return error.code, parse(error.read())

def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")

def chat(model, text, **extra):
    status, reply = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}],
                                                    "max_tokens": 24, "temperature": 0, **extra})
    assert status == 200, reply
    return reply["choices"][0]["message"]["content"], reply["usage"]

prompts = ["Name a colour.", "Write a word that rhymes with cat.", "What is 3 + 4?", "Say hello in French."]
models = ["base-model", first, second]
alone = {(m, p): chat(m, p)[0] for m in models for p in prompts}
check("adapters change the replies", len({alone[(m, prompts[0])] for m in models}) == 3,
      " / ".join(repr(alone[(m, prompts[0])][:30]) for m in models))

post("/flush_cache", {})
jobs = [(m, p) for p in prompts for m in models] * 2
with ThreadPoolExecutor(len(jobs)) as pool:
    together = list(pool.map(lambda job: chat(*job)[0], jobs))
same = sum(reply == alone[job] for job, reply in zip(jobs, together))
check("mixed batches give the same replies as alone", same == len(jobs), f"{same}/{len(jobs)}")

post("/flush_cache", {})
long_prompt = "Here is a long system message that a server would cache. " * 12 + "Now name an animal."
_, usage = chat("base-model", long_prompt)
_, usage_adapter = chat(first, long_prompt)
_, usage_again = chat(first, long_prompt)
cached = lambda u: u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
check("an adapter doesn't reuse the model's cached prompt", cached(usage_adapter) == 0,
      f"cached {cached(usage_adapter)} of {usage_adapter['prompt_tokens']}")
check("an adapter reuses its own cached prompt", cached(usage_again) > 0,
      f"cached {cached(usage_again)} of {usage_again['prompt_tokens']}")

status, models_list = 200, json.loads(urllib.request.urlopen(base + "/v1/models").read())
ids = [m["id"] for m in models_list["data"]]
check("/v1/models lists the adapters", first in ids and second in ids, ", ".join(ids))

by_path, _ = chat("base-model", prompts[0], lora_path=first)
check("lora_path chooses an adapter", by_path == alone[(first, prompts[0])])
status, native = post("/generate", {"text": "The capital of France is", "lora_path": first,
                                     "sampling_params": {"max_new_tokens": 8, "temperature": 0}})
status_base, native_base = post("/generate", {"text": "The capital of France is",
                                              "sampling_params": {"max_new_tokens": 8, "temperature": 0}})
check("/generate takes lora_path", status == 200 and native["text"] != native_base["text"],
      f"{native.get('text')!r} vs {native_base.get('text')!r}")
status, error = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "lora_path": "nope"})
check("an unknown adapter is an error", status == 400, error.get("error", {}).get("message", ""))
sys.exit(1 if failures else 0)
