"""Time long prompts against a server: for each length, a fresh prompt (nothing cached) of about
that many tokens, streamed, reporting the time to the first token (prefill speed) and the decoding
speed after it. Works with tinyllm or SGLang.

    python3 scripts/long_prompt.py URL [LENGTH...]

Standard library only. Lengths are approximate: words are about 1.3 tokens each."""
import json, random, sys, time, urllib.request

url = sys.argv[1].rstrip("/")
lengths = [int(n) for n in sys.argv[2:]] or [1000, 4000, 16000, 30000]
words = "river stone cloud lamp garden window paper music engine forest silver orange bridge".split()

def run(n, seed):
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(int(n / 1.3)))
    body = {"messages": [{"role": "user", "content": f"{seed} {text}\nSummarize this in one line."}],
            "max_tokens": 64, "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.time(); first = None; chunks = 0; prompt = 0
    with urllib.request.urlopen(req) as reply:
        for line in reply:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                prompt = event["usage"]["prompt_tokens"]
            delta = event["choices"][0]["delta"] if event.get("choices") else {}
            if delta.get("content") or delta.get("reasoning_content"):
                chunks += 1
                first = first or time.time()
    end = time.time()
    return prompt, first - start, (chunks - 1) / (end - first)

run(500, 0)  # warm up
for i, n in enumerate(lengths):
    prompt, ttft, decode = run(n, i + 1)
    print(f"{prompt:6d} prompt tokens: first token after {ttft * 1000:7.0f} ms "
          f"({prompt / ttft:6.0f} tok/s prefill), then {decode:5.1f} tok/s")
