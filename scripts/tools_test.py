"""Exercise tool calling through the official OpenAI client. usage: tools_test.py URL [--json FILE]"""
import asyncio, json, sys, time
import jsonschema
from openai import OpenAI, AsyncOpenAI, BadRequestError

base = sys.argv[1]
dump = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
c = OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=120)
model = c.models.list().data[0].id

WEATHER = {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"], "additionalProperties": False}
TIME = {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Europe/Paris"}}, "required": ["timezone"], "additionalProperties": False}
FLIGHT = {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "passengers": {"type": "integer", "minimum": 1, "maximum": 9}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}, "required": ["origin", "destination", "passengers"], "additionalProperties": False}
def tool(name, description, parameters, **extra):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters, **extra}}
TOOLS = [
    tool("get_weather", "Get the current weather in a city", WEATHER),
    tool("get_time", "Get the current time in a timezone", TIME),
    tool("search_flights", "Search for flights between two cities", FLIGHT),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
ASKS_FOR_TOOLS = [
    "What's the weather like in Paris right now, in celsius?",
    "What time is it in Tokyo?",
    "Find me flights from London to New York for 2 passengers.",
    "How hot is it in Cairo today? Use fahrenheit.",
    "I need to know the current time in Europe/Berlin.",
    "Please check the weather in Sydney in celsius.",
    "Book search: flights from Rome to Madrid, 1 passenger.",
    "What is the temperature in Toronto in fahrenheit?",
]
PLAIN = ["What is the capital of France?", "Write a haiku about the sea.", "What is 7 times 8?", "Tell me a one-sentence joke."]

results = {}
failures = []
def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)

def user(text): return [{"role": "user", "content": text}]

def valid(call):
    """Is this call for a known tool, with arguments that follow its schema?"""
    try:
        args = json.loads(call.function.arguments)
        jsonschema.validate(args, SCHEMAS[call.function.name])
        return True
    except Exception:
        return False

def summarize(message, finish):
    return {"content": message.content, "calls": [(t.function.name, t.function.arguments) for t in (message.tool_calls or [])], "finish": finish}

def chat(messages, **kw):
    kw.setdefault("temperature", 0)
    kw.setdefault("max_tokens", 150)
    return c.chat.completions.create(model=model, messages=messages, **kw)

def stream(messages, **kw):
    """Collect a stream the way a client does, checking its shape as it goes."""
    kw.setdefault("temperature", 0)
    kw.setdefault("max_tokens", 150)
    content, calls, finish, ids = "", {}, None, set()
    for chunk in c.chat.completions.create(model=model, messages=messages, stream=True, **kw):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.content:
            content += choice.delta.content
        for t in choice.delta.tool_calls or []:
            call = calls.setdefault(t.index, {"name": None, "arguments": "", "id": None})
            if t.id:
                assert call["id"] is None, "a call was given two ids"
                call["id"] = t.id
                ids.add(t.id)
            if t.function and t.function.name:
                assert call["name"] is None, "a call was given two names"
                call["name"] = t.function.name
            if t.function and t.function.arguments:
                call["arguments"] += t.function.arguments
        finish = choice.finish_reason or finish
    assert len(ids) == len(calls), "every call needs one id"
    return {"content": content or None, "calls": [(calls[i]["name"], calls[i]["arguments"]) for i in sorted(calls)], "finish": finish}

def section(title): print(f"--- {title}")

section("auto: questions a tool can answer")
auto = []
for ask in ASKS_FOR_TOOLS:
    r = chat(user(ask), tools=TOOLS)
    s = summarize(r.choices[0].message, r.choices[0].finish_reason)
    auto.append({"ask": ask, **s})
    print("   ", ask[:44].ljust(44), "->", s["calls"] or repr((s["content"] or "")[:70]), s["finish"])
results["auto"] = auto
calls = [x for x in auto if x["calls"]]
check("at least half the questions led to a call", len(calls) * 2 >= len(auto), f"{len(calls)}/{len(auto)}")
check("calls end with finish_reason tool_calls", all(x["finish"] == "tool_calls" for x in calls))
check("no call is left in the text", not any("<tool_call>" in (x["content"] or "") or "tool_call" in (x["content"] or "") for x in calls))

section("auto: questions that don't need a tool")
plain = []
for ask in PLAIN:
    r = chat(user(ask), tools=TOOLS)
    s = summarize(r.choices[0].message, r.choices[0].finish_reason)
    plain.append({"ask": ask, **s})
    print("   ", ask[:44].ljust(44), "->", s["calls"] or repr((s["content"] or "")[:70]), s["finish"])
results["plain"] = plain

section("none")
r = chat(user(ASKS_FOR_TOOLS[0]), tools=TOOLS, tool_choice="none", max_tokens=40)
check("no calls and a normal finish", not r.choices[0].message.tool_calls and r.choices[0].message.content and r.choices[0].finish_reason in ("stop", "length"))
results["none"] = summarize(r.choices[0].message, r.choices[0].finish_reason)

section("required")
req = []
for ask in ASKS_FOR_TOOLS + PLAIN:
    r = chat(user(ask), tools=TOOLS, tool_choice="required")
    s = summarize(r.choices[0].message, r.choices[0].finish_reason)
    req.append({"ask": ask, **s})
    ok = bool(r.choices[0].message.tool_calls) and all(valid(t) for t in r.choices[0].message.tool_calls)
    print("   ", ask[:44].ljust(44), "->", s["calls"][:2], s["finish"], "" if ok else "INVALID")
    check("  a valid call", ok, json.dumps(s)[:200])
results["required"] = req

section("named tool")
named = []
for ask, name in [(ASKS_FOR_TOOLS[0], "get_weather"), (ASKS_FOR_TOOLS[1], "get_time"), (PLAIN[0], "search_flights")]:
    r = chat(user(ask), tools=TOOLS, tool_choice={"type": "function", "function": {"name": name}})
    s = summarize(r.choices[0].message, r.choices[0].finish_reason)
    named.append({"ask": ask, **s})
    m = r.choices[0].message
    check(f"calls {name}", bool(m.tool_calls) and all(t.function.name == name and valid(t) for t in m.tool_calls), json.dumps(s)[:200])
results["named"] = named

section("parallel calls")
ask = "Get the weather in Paris in celsius, and also the weather in London in celsius. Call the tool once per city."
r = chat(user(ask), tools=TOOLS)
results["parallel"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
print("    auto:    ", results["parallel"]["calls"])
r = chat(user(ask), tools=TOOLS, tool_choice="required")
results["parallel_required"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
print("    required:", results["parallel_required"]["calls"])
r = chat(user(ask), tools=TOOLS, parallel_tool_calls=False)
n = len(r.choices[0].message.tool_calls or [])
results["one_auto"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
check("parallel_tool_calls=false allows at most one (auto)", n <= 1, str(n))
r = chat(user(ask), tools=TOOLS, tool_choice="required", parallel_tool_calls=False)
n = len(r.choices[0].message.tool_calls or [])
results["one_required"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
check("parallel_tool_calls=false allows exactly one (required)", n == 1, str(n))

section("strict tools in auto mode")
STRICT = [tool(t["function"]["name"], t["function"]["description"], t["function"]["parameters"], strict=True) for t in TOOLS]
strict_ok = strict_calls = 0
for ask in ASKS_FOR_TOOLS:
    r = chat(user(ask), tools=STRICT)
    for t in r.choices[0].message.tool_calls or []:
        strict_calls += 1
        strict_ok += valid(t)
r = chat(user(PLAIN[0]), tools=STRICT)
results["strict_plain"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
check("every strict call follows its schema", strict_ok == strict_calls, f"{strict_ok}/{strict_calls}")
print(f"    {strict_calls} calls, {strict_ok} valid. Plain question: {results['strict_plain']['content'] and results['strict_plain']['content'][:60]!r}")

section("streaming matches non-streaming")
same = 0
cases = [(a, {}) for a in ASKS_FOR_TOOLS[:4] + PLAIN[:2]] + [(ASKS_FOR_TOOLS[0], {"tool_choice": "required"}), (ASKS_FOR_TOOLS[2], {"tool_choice": {"type": "function", "function": {"name": "search_flights"}}})]
for ask, kw in cases:
    r = chat(user(ask), tools=TOOLS, seed=7, **kw)
    a = summarize(r.choices[0].message, r.choices[0].finish_reason)
    b = stream(user(ask), tools=TOOLS, seed=7, **kw)
    if a == b:
        same += 1
    else:
        print("    differs:", json.dumps(a)[:200], "\n         vs", json.dumps(b)[:200])
check(f"{len(cases)} requests come out the same both ways", same == len(cases), f"{same}/{len(cases)}")

section("cut off by max_tokens")
r = chat(user(ASKS_FOR_TOOLS[0]), tools=TOOLS, max_tokens=12)
results["short"] = summarize(r.choices[0].message, r.choices[0].finish_reason)
check("finish_reason is length", r.choices[0].finish_reason == "length", str(r.choices[0].finish_reason))
print("    ", results["short"])

section("several choices")
r = chat(user(ASKS_FOR_TOOLS[0]), tools=TOOLS, n=3, temperature=0.7, seed=1, tool_choice="required")
check("three choices, each with a valid call", len(r.choices) == 3 and all(ch.message.tool_calls and all(valid(t) for t in ch.message.tool_calls) for ch in r.choices))
ids = [t.id for ch in r.choices for t in ch.message.tool_calls]
check("call ids are all different", len(set(ids)) == len(ids))

section("logprobs")
r = chat(user(ASKS_FOR_TOOLS[0]), tools=TOOLS, logprobs=True, top_logprobs=2, tool_choice="required", max_tokens=40)
lp = r.choices[0].logprobs
check("logprobs cover the generated tokens", lp is not None and len(lp.content) == r.usage.completion_tokens, f"{lp and len(lp.content)} vs {r.usage.completion_tokens}")

section("a whole conversation")
messages = user("What's the weather like in Paris right now, in celsius?")
r = chat(messages, tools=TOOLS, tool_choice="required")
m = r.choices[0].message
messages.append(m.model_dump(exclude_none=True))
for t in m.tool_calls:
    messages.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps({"temperature_c": 21, "sky": "sunny"})})
r2 = chat(messages, tools=TOOLS, max_tokens=60)
results["round_trip"] = summarize(r2.choices[0].message, r2.choices[0].finish_reason)
print("    answer:", results["round_trip"])
check("the model answers after the result", bool(r2.choices[0].message.content or r2.choices[0].message.tool_calls))
# The same again, with the assistant message exactly as the client got it (empty fields included).
messages[1] = m.model_dump()
r3 = chat(messages, tools=TOOLS, max_tokens=60)
check("a reply sent back as received works too", summarize(r3.choices[0].message, r3.choices[0].finish_reason) == results["round_trip"])

section("mistakes")
def refused(**kw):
    try:
        c.chat.completions.create(model=model, messages=kw.pop("messages", user("hi")), max_tokens=5, **kw)
        return False
    except BadRequestError:
        return True
check("required without tools", refused(tool_choice="required"))
check("naming a tool that isn't there", refused(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}}))
check("a tool result out of nowhere", refused(messages=user("hi") + [{"role": "tool", "tool_call_id": "x", "content": "y"}]))

print(f"=== {len(failures)} failures" + (": " + ", ".join(failures) if failures else ""))
if dump:
    json.dump(results, open(dump, "w"), indent=1)


async def load(concurrency=64):
    """Forced calls at once. Every one must be valid."""
    ac = AsyncOpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=300)
    start = time.time()
    async def one(i):
        ask = ASKS_FOR_TOOLS[i % len(ASKS_FOR_TOOLS)] + f" (request {i})"
        r = await ac.chat.completions.create(model=model, messages=user(ask), tools=TOOLS, tool_choice="required", temperature=0.5, seed=i, max_tokens=120, parallel_tool_calls=False)
        return r
    rs = await asyncio.gather(*[one(i) for i in range(concurrency * 4)])
    elapsed = time.time() - start
    good = sum(1 for r in rs if r.choices[0].message.tool_calls and all(valid(t) for t in r.choices[0].message.tool_calls))
    toks = sum(r.usage.completion_tokens for r in rs)
    print(f"load: {len(rs)} forced calls, {concurrency} at a time (sent all at once): {good} valid; {toks/elapsed:.0f} tok/s, {elapsed:.2f}s")

asyncio.run(load())
