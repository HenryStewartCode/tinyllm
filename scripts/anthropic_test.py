"""Check the Anthropic Messages API (/v1/messages) against a running server.

usage: anthropic_test.py BASE_URL [--thinking]     (--thinking for a reasoning model such as Qwen3)"""
import json, sys, urllib.request

BASE = sys.argv[1]
THINKING = "--thinking" in sys.argv


def post(path, body):
    request = urllib.request.Request(f"{BASE}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            if not body.get("stream"):
                return response.status, json.load(response)
            events, kind = [], None
            for line in response:
                line = line.decode().rstrip("\n")
                if line.startswith("event: "):
                    kind = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    assert data["type"] == kind, (kind, data)
                    events.append(data)
            return response.status, events
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def assemble(events):
    """Rebuild a message from stream events, the way the SDKs do."""
    message = None
    for e in events:
        if e["type"] == "message_start":
            message = e["message"]
        elif e["type"] == "content_block_start":
            assert e["index"] == len(message["content"])
            message["content"].append(dict(e["content_block"]))
        elif e["type"] == "content_block_delta":
            block, delta = message["content"][e["index"]], e["delta"]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] += delta["signature"]
            elif delta["type"] == "input_json_delta":
                block["partial"] = block.get("partial", "") + delta["partial_json"]
        elif e["type"] == "content_block_stop":
            block = message["content"][e["index"]]
            if block["type"] == "tool_use":
                block["input"] = json.loads(block.pop("partial", "{}") or "{}")
        elif e["type"] == "message_delta":
            message.update(e["delta"])
            message["usage"].update(e["usage"])
    assert events[-1]["type"] == "message_stop"
    return message


ok = True
def check(name, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"{'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")


def blocks(message, kind):
    return [b for b in message["content"] if b["type"] == kind]


no_thinking = {"thinking": {"type": "disabled"}} if THINKING else {}
base = {"max_tokens": 300, "temperature": 0, **no_thinking}

status, m = post("/v1/messages", {**base, "system": "Answer in one word.", "messages": [{"role": "user", "content": "What is the capital of France?"}]})
check("a plain message", status == 200 and m["type"] == "message" and m["role"] == "assistant"
      and "Paris" in blocks(m, "text")[0]["text"] and m["stop_reason"] == "end_turn", json.dumps(m["content"])[:100])
check("usage is reported", m["usage"]["input_tokens"] > 0 and m["usage"]["output_tokens"] > 0, json.dumps(m["usage"]))

status, events = post("/v1/messages", {**base, "stream": True, "system": [{"type": "text", "text": "Answer in one word."}],
                                        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]})
streamed = assemble(events)
check("streaming gives the same message", [b["text"] for b in blocks(streamed, "text")] == [b["text"] for b in blocks(m, "text")]
      and streamed["stop_reason"] == "end_turn", f"{len(events)} events")

status, m = post("/v1/messages", {**base, "max_tokens": 3, "messages": [{"role": "user", "content": "Count from 1 to 100."}]})
check("max_tokens stops it", m["stop_reason"] == "max_tokens" and m["usage"]["output_tokens"] == 3)

status, m = post("/v1/messages", {**base, "stop_sequences": ["5"], "messages": [{"role": "user", "content": "Count from 1 to 10, separated by commas."}]})
check("stop sequences", m["stop_reason"] == "stop_sequence" and m["stop_sequence"] == "5", json.dumps(blocks(m, "text"))[:80])

tools = [{"name": "get_weather", "description": "Get the weather in a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
question = {"role": "user", "content": "What's the weather in Paris?"}
for stream in [False, True]:
    status, result = post("/v1/messages", {**base, "stream": stream, "tools": tools, "tool_choice": {"type": "any"}, "messages": [question]})
    m = assemble(result) if stream else result
    uses = blocks(m, "tool_use")
    check(f"tool use{' (streaming)' if stream else ''}", m["stop_reason"] == "tool_use" and uses and uses[0]["name"] == "get_weather"
          and uses[0]["input"].get("city") == "Paris" and uses[0]["id"].startswith("toolu_"), json.dumps(uses)[:120])

status, m = post("/v1/messages", {**base, "tools": tools, "messages": [
    question,
    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 21 C"}]},
]})
text = " ".join(b["text"] for b in blocks(m, "text"))
check("a tool result is answered", status == 200 and ("21" in text or "unny" in text), text[:100])

status, counted = post("/v1/messages/count_tokens", {"messages": [question]})
check("count_tokens", status == 200 and counted["input_tokens"] > 5, json.dumps(counted))

status, err = post("/v1/messages", {**base, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]})
check("errors come in Anthropic's format", status == 400 and err["type"] == "error" and err["error"]["type"] == "invalid_request_error",
      err.get("error", {}).get("message", ""))

if THINKING:
    body = {"max_tokens": 1000, "temperature": 0, "thinking": {"type": "enabled", "budget_tokens": 400},
            "messages": [{"role": "user", "content": "What is 12 + 30?"}]}
    status, m = post("/v1/messages", body)
    check("thinking comes first, in its own block", m["content"][0]["type"] == "thinking" and blocks(m, "text")
          and "42" in blocks(m, "text")[0]["text"], json.dumps([b["type"] for b in m["content"]]))
    status, events = post("/v1/messages", {**body, "stream": True})
    streamed = assemble(events)
    check("thinking, streaming", [b["type"] for b in streamed["content"]] == [b["type"] for b in m["content"]]
          and streamed["content"][0]["thinking"] == m["content"][0]["thinking"])
sys.exit(0 if ok else 1)
