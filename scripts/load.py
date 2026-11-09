# Server throughput, measured without per-token streaming so the client isn't the bottleneck.
# usage: load.py <base-url> <max-tokens> <concurrency>...
import asyncio, json, sys, time, multiprocessing as mp
import aiohttp
BASE, MAX_TOK = sys.argv[1], int(sys.argv[2])
LEVELS = list(map(int, sys.argv[3:]))
TOPICS = ["the sea","mountains","cats","volcanoes","rivers","bicycles","chess","honey","trains","clouds","owls","maps","glass","tea","bridges","comets","rain","wolves","maple trees","lighthouses","kites","coral","deserts","fog","violins","ferns","anchors","salt","moss","ink","tides","foxes"]
def msg(i): return [{"role":"user","content":f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}, with details and history. ({i})"}]
async def one(s, i):
    body = {"messages": msg(i), "max_tokens": MAX_TOK, "temperature": 0.8, "seed": i, "ignore_eos": True, "stream": False}
    async with s.post(f"{BASE}/v1/chat/completions", json=body) as r:
        out = await r.json()
        assert r.status == 200, out
        return out["usage"]["completion_tokens"]
async def clients(n, start):
    t = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=t, connector=aiohttp.TCPConnector(limit=0)) as s:
        return sum(await asyncio.gather(*[one(s, start + i) for i in range(n)]))
def proc(n, start, q): q.put(asyncio.run(clients(n, start)))
async def flush():
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE}/flush_cache") as r: await r.read()
print(f"{'clients':>8} {'total tok/s':>12} {'tok/s per stream':>17}")
for n in LEVELS:
    asyncio.run(flush())
    procs = 8 if n >= 8 else 1
    q = mp.Queue(); ps = []
    each = n // procs
    t0 = time.perf_counter()
    for p in range(procs):
        count = each + (1 if p < n % procs else 0)
        pr = mp.Process(target=proc, args=(count, p * 1000, q)); pr.start(); ps.append(pr)
    total = sum(q.get() for _ in ps)
    for pr in ps: pr.join()
    dt = time.perf_counter() - t0
    print(f"{n:8d} {total/dt:12.1f} {total/dt/n:17.1f}")
