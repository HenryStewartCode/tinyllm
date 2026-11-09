# Time to first token for a long shared prompt, and throughput-style use, against any OpenAI server.
import asyncio, json, sys, time, aiohttp
BASE = sys.argv[1]
SYSTEM = "You are a meticulous assistant for a shipping company. " + " ".join(f"Rule {i}: always check the manifest for crate number {i*7} before departure and record the time." for i in range(60))
async def ask(s, q):
    body = {"messages": [{"role":"system","content":SYSTEM},{"role":"user","content":q}], "max_tokens": 4, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; usage = None
    async with s.post(f"{BASE}/v1/chat/completions", json=body) as r:
        assert r.status == 200, await r.text()
        async for line in r.content:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            o = json.loads(line[6:])
            if o.get("usage"): usage = o["usage"]
            if o["choices"] and o["choices"][0].get("delta", {}).get("content") and first is None: first = time.perf_counter() - t0
    return first, usage
async def main():
    async with aiohttp.ClientSession() as s:
        await s.post(f"{BASE}/flush_cache")
        for i, q in enumerate(["What is rule 3?", "What is rule 12?", "Who are you?"]):
            t, u = await ask(s, q)
            d = (u or {}).get("prompt_tokens_details") or {}
            print(f"  request {i+1}: {u['prompt_tokens']} prompt tokens, {d.get('cached_tokens', '?')} cached, first token after {t*1e3:.0f} ms")
asyncio.run(main())
