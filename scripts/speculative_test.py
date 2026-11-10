"""Check that speculative decoding changes nothing but speed: the same requests to a server without
it (PLAIN) and one with it (SPEC) must give identical replies.

usage: speculative_test.py PLAIN_URL SPEC_URL [concurrency]"""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

plain, spec = sys.argv[1:3]
concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 1

CODE = '''def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
'''
TEXT = ("The lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a can of oil "
        "and a cloth. He polished the great lens until it shone, trimmed the wick, and lit the lamp. "
        "Ships far out at sea saw the beam sweep across the water and knew the rocks were near.")
PROMPTS = [
    f"Rewrite this Python function, adding type hints and a docstring. Keep the body the same.\n\n{CODE}",
    f"Repeat the following paragraph exactly, word for word:\n\n{TEXT}",
    f"Here is a paragraph:\n\n{TEXT}\n\nCopy it, but replace 'lighthouse keeper' with 'old sailor'.",
    "Write a short poem about autumn leaves.",
    "List the numbers from 1 to 30, one per line.",
]
SETTINGS = [
    {"temperature": 0},
    {"temperature": 0.8, "seed": 11},
    {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 5},
    {"temperature": 1.0, "repetition_penalty": 1.1, "seed": 3},
]


def ask(base, prompt, settings):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 200, **settings}
    request = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    return reply["choices"][0]["message"]["content"], reply["usage"]["completion_tokens"]


def metric(base, name):
    text = urllib.request.urlopen(f"{base}/metrics").read().decode()
    found = re.search(rf"^tinyllm_{name} (\S+)$", text, re.M)
    return float(found.group(1)) if found else 0.0


def run(base):
    jobs = [(p, s) for p in PROMPTS for s in SETTINGS]
    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda job: ask(base, *job), jobs))
    elapsed = time.perf_counter() - start
    return results, sum(tokens for _, tokens in results) / elapsed


drafted0, accepted0 = metric(spec, "spec_drafted_tokens_total"), metric(spec, "spec_accepted_tokens_total")
plain_results, plain_speed = run(plain)
spec_results, spec_speed = run(spec)
drafted = metric(spec, "spec_drafted_tokens_total") - drafted0
accepted = metric(spec, "spec_accepted_tokens_total") - accepted0
same = sum(a[0] == b[0] for a, b in zip(plain_results, spec_results))
print(f"identical replies: {same}/{len(plain_results)}")
for (prompt, settings), a, b in zip([(p, s) for p in PROMPTS for s in SETTINGS], plain_results, spec_results):
    if a[0] != b[0]:
        at = next((i for i, (x, y) in enumerate(zip(a[0], b[0])) if x != y), min(len(a[0]), len(b[0])))
        print(f"  differs at {at}: {settings} {prompt[:40]!r}: {a[0][at:at+30]!r} vs {b[0][at:at+30]!r}")
print(f"throughput at concurrency {concurrency}: {plain_speed:.1f} tok/s plain, {spec_speed:.1f} tok/s speculative ({spec_speed / plain_speed:.2f}x)")
print(f"guesses: {int(accepted)}/{int(drafted)} right ({accepted / max(drafted, 1):.0%})")
sys.exit(0 if same == len(plain_results) else 1)
