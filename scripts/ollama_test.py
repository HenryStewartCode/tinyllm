"""Check Ollama's API (/api/chat, /api/generate, ...) against a running server.

usage: ollama_test.py BASE_URL [--thinking]     (--thinking for a reasoning model such as Qwen3)"""
import json, sys, urllib.request

BASE = sys.argv[1]
THINKING = "--thinking" in sys.argv


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"{BASE}{path}", data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            if body is not None and body.get("stream", True) and path in ("/api/chat", "/api/generate") and body.get("prompt", "x"):
                return response.status, [json.loads(line) for line in response if line.strip()]
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


ok = True
def check(name, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"{'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")


opts = {"temperature": 0, "num_predict": 300}
no_think = {"think": False} if THINKING else {}
question = [{"role": "user", "content": "What is the capital of France? One word."}]

status, r = call("/api/chat", {"messages": question, "stream": False, "options": opts, **no_think})
check("chat", status == 200 and r["done"] and "Paris" in r["message"]["content"] and r["done_reason"] == "stop",
      repr(r["message"]["content"]))
check("chat reports counts and timings", r["prompt_eval_count"] > 0 and r["eval_count"] > 0 and r["total_duration"] > 0)

status, lines = call("/api/chat", {"messages": question, "options": opts, **no_think})
text = "".join(l["message"]["content"] for l in lines)
check("chat streams by default", len(lines) > 1 and lines[-1]["done"] and not any(l["done"] for l in lines[:-1])
      and text == r["message"]["content"], f"{len(lines)} lines")

status, r = call("/api/generate", {"prompt": "What is the capital of Japan? One word.", "stream": False, "options": opts, **no_think})
check("generate", status == 200 and "Tokyo" in r["response"], repr(r["response"]))

status, lines = call("/api/generate", {"prompt": "The capital of Italy is", "raw": True, "options": {"temperature": 0, "num_predict": 4}})
check("raw generate streams", "Rome" in "".join(l["response"] for l in lines) and lines[-1]["done_reason"] == "length")

schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
status, r = call("/api/chat", {"messages": question, "stream": False, "format": schema, "options": opts, **no_think})
check("format takes a schema", json.loads(r["message"]["content"]).get("city") == "Paris", r["message"]["content"])

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
weather = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
status, r = call("/api/chat", {"messages": weather, "tools": tools, "stream": False, "options": opts, **no_think})
calls = r["message"].get("tool_calls") or []
check("tool calls come with object arguments", calls and calls[0]["function"]["name"] == "get_weather"
      and calls[0]["function"]["arguments"].get("city") == "Paris", json.dumps(calls))
status, lines = call("/api/chat", {"messages": weather, "tools": tools, "options": opts, **no_think})
streamed = [c for l in lines for c in l["message"].get("tool_calls") or []]
check("tool calls stream too", streamed == calls)

answer = weather + [{"role": "assistant", "content": "", "tool_calls": calls}, {"role": "tool", "content": "Sunny, 21 C"}]
status, r = call("/api/chat", {"messages": answer, "tools": tools, "stream": False, "options": opts, **no_think})
check("a tool result is answered", "21" in r["message"]["content"] or "unny" in r["message"]["content"], repr(r["message"]["content"][:80]))

status, tags = call("/api/tags")
check("tags", status == 200 and len(tags["models"]) == 1, tags["models"][0]["details"]["family"])
status, show = call("/api/show", {"model": tags["models"][0]["name"]})
check("show", status == 200 and "completion" in show["capabilities"], json.dumps(show["capabilities"]))
status, version = call("/api/version")
check("version", status == 200 and version["version"])

if THINKING:
    status, r = call("/api/chat", {"messages": [{"role": "user", "content": "What is 12 + 30?"}], "think": True, "stream": False,
                                   "options": {"temperature": 0, "num_predict": 1000}})
    check("think: true separates the thinking", r["message"].get("thinking") and "42" in r["message"]["content"]
          and "<think>" not in r["message"]["content"], repr(r["message"]["content"][:60]))
    status, lines = call("/api/chat", {"messages": [{"role": "user", "content": "What is 12 + 30?"}], "think": True,
                                       "options": {"temperature": 0, "num_predict": 1000}})
    check("thinking streams", "".join(l["message"].get("thinking", "") for l in lines) == r["message"]["thinking"])
sys.exit(0 if ok else 1)
