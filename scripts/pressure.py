import asyncio, json, sys, time
import aiohttp
A, B = sys.argv[1:3]
TOPICS = ["the sea","mountains","cats","volcanoes","rivers","bicycles","chess","honey","trains","clouds","owls","maps","glass","tea","bridges","comets"]
msgs = [[{"role":"user","content":f"Write a short paragraph about {t}."}] for t in TOPICS]
async def ask(session, base, m, seed):
    body = {"messages": m, "max_tokens": 64, "temperature": 0.8, "seed": seed, "ignore_eos": True}
    async with session.post(f"{base}/v1/chat/completions", json=body) as r:
        out = await r.json()
        if r.status != 200: return {"error": out}
        return {"text": out["choices"][0]["message"]["content"], "n": out["usage"]["completion_tokens"]}
async def metric(session, base, name):
    async with session.get(f"{base}/metrics") as r:
        for line in (await r.text()).splitlines():
            if line.startswith(f"tinyllm_{name} "): return float(line.split()[1])
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=aiohttp.TCPConnector(limit=0)) as s:
        ref = await asyncio.gather(*[ask(s, A, m, i) for i, m in enumerate(msgs)])
        print("capacity:", await metric(s, B, "cache_capacity_tokens"), "tokens")
        t0 = time.perf_counter()
        got = await asyncio.gather(*[ask(s, B, m, i) for i, m in enumerate(msgs)])
        dt = time.perf_counter() - t0
        errs = [g for g in got if "error" in g]
        same = sum(1 for a, b in zip(ref, got) if a.get("text") == b.get("text"))
        print(f"{len(got)} requests needing {16*(64+36)} tokens of KV cache in total: {len(errs)} errors, {same}/16 replies identical to the unconstrained server, {dt:.1f}s; retractions = {await metric(s, B, 'retractions_total')}")
        if errs: print(errs[0])
        # A request that alone can't fit
        body = {"messages": [{"role":"user","content":"word " * 800}], "max_tokens": 4}
        async with s.post(f"{B}/v1/chat/completions", json=body) as r:
            print("too long for the cache:", r.status, (await r.text())[:160])
        # and it still works afterwards
        print("afterwards:", (await ask(s, B, msgs[0], 1)).get("n"), "tokens")
asyncio.run(main())
