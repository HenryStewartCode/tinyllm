"""Load a tinyllm router (or one server) with requests that share long prefixes, as chat
conversations with long system prompts do, and report throughput, how much of each prompt was
cached, and how requests were spread over the servers.

    python3 scripts/router_test.py URL [GROUPS] [REQUESTS_PER_GROUP] [CONCURRENCY]

URL can also be several servers' URLs joined by commas, to send requests to them in turn (round
robin) for comparison.

Standard library only. Each group's system prompt is about 1,500 tokens of words."""
import json, random, sys, threading, time, urllib.request
from collections import Counter

urls = [u.rstrip("/") for u in sys.argv[1].split(",")]
turn = 0
lock = threading.Lock()
groups = int(sys.argv[2]) if len(sys.argv) > 2 else 8
per_group = int(sys.argv[3]) if len(sys.argv) > 3 else 16
concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 32
rng = random.Random(0)
words = "river stone cloud lamp garden window paper music engine forest silver orange bridge".split()
systems = [f"Context {g}: " + " ".join(rng.choice(words) for _ in range(1500)) for g in range(groups)]
jobs = [(g, i) for i in range(per_group) for g in range(groups)]
rng.shuffle(jobs)

def call(body, stream=False):
    global turn
    with lock:
        url = urls[turn % len(urls)]
        turn += 1
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as reply:
        worker = reply.headers.get("x-tinyllm-worker", url)
        return (reply.read() if stream else json.load(reply)), worker

prompt = cached = done = 0
workers, placement = Counter(), {}
def run():
    global prompt, cached, done
    while True:
        with lock:
            if not jobs:
                return
            g, i = jobs.pop()
        body = {"messages": [{"role": "system", "content": systems[g]},
                             {"role": "user", "content": f"Question {i}: which word comes first?"}],
                "max_tokens": 16, "temperature": 0}
        out, worker = call(body)
        usage = out["usage"]
        with lock:
            prompt += usage["prompt_tokens"]
            cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            done += 1
            workers[worker] += 1
            placement.setdefault(g, Counter())[worker] += 1

start = time.time()
threads = [threading.Thread(target=run) for _ in range(concurrency)]
for t in threads: t.start()
for t in threads: t.join()
elapsed = time.time() - start
print(f"{done} requests in {elapsed:.1f}s ({done / elapsed:.1f}/s); {cached / prompt:.0%} of prompt tokens cached")
print("per server:", dict(workers))
print("servers per group:", [len(c) for _, c in sorted(placement.items())])
raw, worker = call({"messages": [{"role": "user", "content": "Count to five."}], "max_tokens": 20,
                    "temperature": 0, "stream": True}, stream=True)
chunks = raw.decode().count("data: ")
print(f"streaming through {worker}: {chunks} events, ends with [DONE]: {raw.decode().rstrip().endswith('[DONE]')}")
