"""For models that can't be expected to choose tools well: forced calls must still be valid."""
import json, sys, asyncio
import jsonschema
from openai import AsyncOpenAI
base = sys.argv[1]
c = AsyncOpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=300)
exec(open("/root/tools_schemas.py").read())
async def one(i, **kw):
    r = await c.chat.completions.create(model="m", messages=[{"role": "user", "content": ASKS[i % len(ASKS)]}], tools=TOOLS, temperature=0.8, seed=i, max_tokens=400, **kw)
    return r.choices[0]
async def stream_one(i, **kw):
    content, calls, finish = "", {}, None
    async for chunk in await c.chat.completions.create(model="m", messages=[{"role": "user", "content": ASKS[i % len(ASKS)]}], tools=TOOLS, temperature=0.8, seed=i, max_tokens=400, stream=True, **kw):
        if not chunk.choices: continue
        ch = chunk.choices[0]
        content += ch.delta.content or ""
        for t in ch.delta.tool_calls or []:
            call = calls.setdefault(t.index, ["", ""])
            call[0] += (t.function.name or "") if t.function else ""
            call[1] += (t.function.arguments or "") if t.function else ""
        finish = ch.finish_reason or finish
    return content, [calls[i] for i in sorted(calls)], finish
def valid(calls):
    for name, args in calls:
        jsonschema.validate(json.loads(args), SCHEMAS[name])
async def main():
    named = {"type": "function", "function": {"name": "get_weather"}}
    for label, kw in [("required", {"tool_choice": "required"}), ("named", {"tool_choice": named}), ("one", {"tool_choice": "required", "parallel_tool_calls": False})]:
        rs = await asyncio.gather(*[one(i, **kw) for i in range(64)])
        streams = await asyncio.gather(*[stream_one(i, **kw) for i in range(64)])
        good = same = finished = 0; bad = None
        for r, (content, calls, finish) in zip(rs, streams):
            ours = [[t.function.name, t.function.arguments] for t in r.message.tool_calls or []]
            same += ours == calls and (r.message.content or "") == content
            finished += r.finish_reason == "tool_calls"
            try:
                assert ours and not r.message.content
                if r.finish_reason == "tool_calls":
                    valid(ours)
                    if label == "named": assert all(n == "get_weather" for n, _ in ours)
                    if label == "one": assert len(ours) == 1
                good += 1
            except Exception as e:
                bad = (ours, r.message.content, r.finish_reason, repr(e)[:200])
        print("BAD", bad[-2:] if bad else None); print(f"  {label}: {good}/64 valid, {finished} finished normally, stream matched in {same}; e.g. {[[t.function.name, t.function.arguments] for t in rs[0].message.tool_calls][:2]}"[:300], bad or "")
asyncio.run(main())
