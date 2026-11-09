"""Check how a reasoning model's thinking is handled by the server.

usage: reasoning_test.py BASE_URL      (e.g. http://127.0.0.1:8000, serving Qwen3)"""
import json, sys, urllib.request

BASE = sys.argv[1]


def post(body, stream=False):
    request = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps({**body, "stream": stream}).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        if not stream:
            return json.load(response)
        reasoning, content, calls, finish = "", "", [], None
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for choice in chunk["choices"]:
                delta = choice["delta"]
                reasoning += delta.get("reasoning_content") or ""
                content += delta.get("content") or ""
                calls += delta.get("tool_calls") or []
                finish = choice.get("finish_reason") or finish
        return reasoning, content, calls, finish


ok = True
def check(name, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"{'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")


question = [{"role": "user", "content": "What is 12 + 30? Reply with just the number."}]
base = {"messages": question, "temperature": 0, "max_tokens": 600}

reply = post(base)["choices"][0]["message"]
check("thinking is separated", reply.get("reasoning_content") and "<think>" not in reply["content"],
      f"reasoning {len(reply.get('reasoning_content') or '')} chars, content {reply['content']!r}")
check("the answer is in content", "42" in reply["content"])

reasoning, content, _, finish = post(base, stream=True)
check("streaming gives the same pieces", (reasoning, content) == (reply.get("reasoning_content"), reply["content"]),
      f"finish {finish}")

reply = post({**base, "separate_reasoning": False})["choices"][0]["message"]
check("separate_reasoning: false keeps it in content", "</think>" in reply["content"] and not reply.get("reasoning_content"))

reply = post({**base, "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]
check("enable_thinking: false skips thinking", not reply.get("reasoning_content") and "42" in reply["content"],
      repr(reply["content"]))

response = post({**base, "thinking_budget": 20})
reply = response["choices"][0]["message"]
check("a thinking budget cuts the thinking short", reply.get("reasoning_content") and reply["content"],
      f"{response['usage']['completion_tokens']} tokens, content {reply['content'][:60]!r}")

schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
response_format = {"type": "json_schema", "json_schema": {"name": "a", "schema": schema}}
reply = post({**base, "response_format": response_format})["choices"][0]["message"]
try:
    parsed = json.loads(reply["content"])
except ValueError:
    parsed = None
check("structured output applies after the thinking", reply.get("reasoning_content") and parsed == {"answer": 42},
      repr(reply["content"]))

tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers",
          "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
for choice in ["auto", "required"]:
    body = {**base, "messages": [{"role": "user", "content": "Use the add tool to add 12 and 30."}], "tools": tools, "tool_choice": choice}
    reply = post(body)["choices"][0]["message"]
    calls = reply.get("tool_calls") or []
    check(f"tool_choice {choice}: thinks, then calls", reply.get("reasoning_content") and calls
          and json.loads(calls[0]["function"]["arguments"]) == {"a": 12, "b": 30}, json.dumps(calls)[:120])
    reasoning, content, streamed, finish = post(body, stream=True)
    check(f"tool_choice {choice}, streaming", reasoning and streamed and finish == "tool_calls", f"finish {finish}")
sys.exit(0 if ok else 1)
