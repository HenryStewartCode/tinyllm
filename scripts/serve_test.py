import asyncio, json, sys, time, statistics
import aiohttp, jsonschema
BASE = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "all"

async def stream_chat(session, messages, max_tokens=64, **extra):
    """Returns (text, ttft, total, usage)."""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}, **extra}
    t0 = time.perf_counter(); ttft = None; text = ""; usage = None; finish = None
    async with session.post(f"{BASE}/v1/chat/completions", json=body) as r:
        assert r.status == 200, await r.text()
        async for line in r.content:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            data = line[6:]
            if data == "[DONE]": break
            obj = json.loads(data)
            if "error" in obj: raise Exception(obj["error"])
            if obj.get("usage"): usage = obj["usage"]
            for c in obj["choices"]:
                piece = c.get("delta", {}).get("content")
                if piece:
                    if ttft is None: ttft = time.perf_counter() - t0
                    text += piece
                if c.get("finish_reason"): finish = c["finish_reason"]
    return text, ttft, time.perf_counter() - t0, usage, finish

async def complete(session, messages, max_tokens=64, **extra):
    body = {"messages": messages, "max_tokens": max_tokens, **extra}
    async with session.post(f"{BASE}/v1/chat/completions", json=body) as r:
        out = await r.json()
        assert r.status == 200, out
        return out

async def metric(session, name):
    async with session.get(f"{BASE}/metrics") as r:
        for line in (await r.text()).splitlines():
            if line.startswith(f"tinyllm_{name} "): return float(line.split()[1])

TOPICS = ["the sea","mountains","cats","volcanoes","rivers","bicycles","chess","honey","trains","clouds","owls","maps","glass","tea","bridges","comets","rain","wolves","maple trees","lighthouses","kites","coral","deserts","fog","violins","ferns","anchors","salt","moss","ink","tides","foxes"]
def msg(i): return [{"role":"user","content":f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}, with details and history. ({i})"}]

async def throughput(session):
    print("concurrency | total tok/s | tok/s per stream | median first token | max first token")
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        await session.post(f"{BASE}/flush_cache")
        t0 = time.perf_counter()
        results = await asyncio.gather(*[stream_chat(session, msg(i), max_tokens=64, ignore_eos=True, temperature=0.8, seed=i) for i in range(n)])
        dt = time.perf_counter() - t0
        tokens = sum(r[3]["completion_tokens"] for r in results)
        rates = [(r[3]["completion_tokens"]-1)/(r[2]-r[1]) for r in results]
        ttfts = [r[1] for r in results]
        print(f"{n:11d} | {tokens/dt:11.1f} | {statistics.mean(rates):16.1f} | {statistics.median(ttfts)*1e3:15.0f}ms | {max(ttfts)*1e3:12.0f}ms")

async def prefix_cache(session):
    await session.post(f"{BASE}/flush_cache")
    system = "You are a meticulous assistant for a shipping company. " + " ".join(f"Rule {i}: always check the manifest for crate number {i*7} before departure and record the time." for i in range(60))
    questions = ["What is rule 3?", "What is rule 12?", "Who are you?", "What is rule 40?"]
    print("prefix cache, one long shared system prompt:")
    for i, q in enumerate(questions):
        t, ttft, total, usage, _ = await stream_chat(session, [{"role":"system","content":system},{"role":"user","content":q}], max_tokens=4, temperature=0)
        print(f"  request {i+1}: {usage['prompt_tokens']} prompt tokens, {usage['prompt_tokens_details']['cached_tokens']} cached, first token after {ttft*1e3:.0f} ms")
    # Multi-turn: each turn reuses everything so far
    history = [{"role":"user","content":"Tell me a fact about Jupiter."}]
    print("multi-turn chat:")
    for turn in range(3):
        t, ttft, total, usage, _ = await stream_chat(session, history, max_tokens=24, temperature=0)
        print(f"  turn {turn+1}: {usage['prompt_tokens']} prompt tokens, {usage['prompt_tokens_details']['cached_tokens']} cached, first token after {ttft*1e3:.0f} ms")
        history += [{"role":"assistant","content":t},{"role":"user","content":"And another?"}]
    # Burst of identical long prompts: only one should compute it.
    await session.post(f"{BASE}/flush_cache")
    p0 = await metric(session, "batch_tokens_total")
    results = await asyncio.gather(*[stream_chat(session, [{"role":"system","content":system},{"role":"user","content":f"Say the number {i}."}], max_tokens=2, temperature=0) for i in range(8)])
    p1 = await metric(session, "batch_tokens_total")
    prompt = sum(r[3]["prompt_tokens"] for r in results); cached = sum(r[3]["prompt_tokens_details"]["cached_tokens"] for r in results)
    print(f"burst of 8 sharing a long prefix at the same moment: {prompt} prompt tokens, {cached} cached, {int(p1-p0)} tokens actually computed")

async def determinism(session):
    prompts = [msg(i) for i in range(12)]
    alone = []
    for p in prompts:
        await session.post(f"{BASE}/flush_cache")
        alone.append((await complete(session, p, max_tokens=48, temperature=0))["choices"][0]["message"]["content"])
    await session.post(f"{BASE}/flush_cache")
    together = await asyncio.gather(*[complete(session, p, max_tokens=48, temperature=0) for p in prompts])
    same = sum(a == t["choices"][0]["message"]["content"] for a, t in zip(alone, together))
    print(f"determinism: {same}/{len(prompts)} greedy replies identical alone vs. in a batch of {len(prompts)}")
    # the same seed gives the same sample whether alone or in a batch
    await session.post(f"{BASE}/flush_cache")
    alone = []
    for i, p in enumerate(prompts):
        alone.append((await complete(session, p, max_tokens=32, temperature=0.9, seed=100+i))["choices"][0]["message"]["content"])
    together = await asyncio.gather(*[complete(session, p, max_tokens=32, temperature=0.9, seed=100+i) for i, p in enumerate(prompts)])
    same = sum(a == t["choices"][0]["message"]["content"] for a, t in zip(alone, together))
    print(f"determinism: {same}/{len(prompts)} seeded samples identical alone vs. in a batch")

async def cancellation(session):
    before = await metric(session, "generated_tokens_total")
    async def drop():
        body = {"messages": msg(1), "max_tokens": 400, "stream": True, "ignore_eos": True}
        async with session.post(f"{BASE}/v1/chat/completions", json=body) as r:
            n = 0
            async for line in r.content:
                n += 1
                if n > 10: return  # hang up early
    await asyncio.gather(*[drop() for _ in range(4)])
    await asyncio.sleep(0.5)
    running = await metric(session, "running_requests")
    after = await metric(session, "generated_tokens_total")
    await asyncio.sleep(0.3)
    final = await metric(session, "generated_tokens_total")
    print(f"cancellation: running after hang-up = {int(running)}; generated {int(after-before)} tokens of 1600 asked for; still generating: {final != after}")

async def structured(session):
    person = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":120},"city":{"type":"string"},"hobbies":{"type":"array","items":{"type":"string"},"maxItems":3}},"required":["name","age","city","hobbies"],"additionalProperties":False}
    ask = lambda i: [{"role":"user","content":f"Invent a fictional person number {i} and describe them as JSON with name, age, city, hobbies."}]
    fmt = {"type":"json_schema","json_schema":{"name":"person","schema":person,"strict":True}}
    t0 = time.perf_counter()
    res = await asyncio.gather(*[complete(session, ask(i), max_tokens=160, temperature=0.7, seed=i, response_format=fmt) for i in range(24)])
    dt = time.perf_counter() - t0
    ok = 0
    for r in res:
        c = r["choices"][0]
        try:
            jsonschema.validate(json.loads(c["message"]["content"]), person); ok += 1
        except Exception as e:
            print("  INVALID", c["finish_reason"], repr(c["message"]["content"])[:200], repr(e)[:100])
    print(f"json_schema: {ok}/24 valid in {dt:.1f}s; e.g. {res[0]['choices'][0]['message']['content']!r} ({res[0]['choices'][0]['finish_reason']})")
    r = await complete(session, [{"role":"user","content":"List three fruits as a JSON object."}], max_tokens=300, temperature=0, response_format={"type":"json_object"})
    c = r["choices"][0]; print("json_object:", repr(c["message"]["content"]), c["finish_reason"]); json.loads(c["message"]["content"])
    r = await complete(session, [{"role":"user","content":"Give me a US phone number."}], max_tokens=30, temperature=0.5, regex=r"\(\d{3}\) \d{3}-\d{4}")
    c = r["choices"][0]; print("regex:", repr(c["message"]["content"]), c["finish_reason"])
    r = await complete(session, [{"role":"user","content":"Is the sky blue? Answer."}], max_tokens=10, temperature=0, ebnf='root ::= "Yes" | "No"')
    c = r["choices"][0]; print("ebnf:", repr(c["message"]["content"]), c["finish_reason"])
    r = await session.post(f"{BASE}/v1/chat/completions", json={"messages":msg(0),"max_tokens":4,"regex":"(unclosed"})
    print("bad regex:", r.status, (await r.json())["error"]["message"][:80])

async def main():
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        for name, f in [("determinism", determinism), ("structured", structured), ("cache", prefix_cache), ("cancel", cancellation), ("throughput", throughput)]:
            if mode in ("all", name):
                await f(session)
asyncio.run(main())
