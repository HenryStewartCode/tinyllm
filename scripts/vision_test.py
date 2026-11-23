"""Send a picture to a tinyllm server in each API's format, and check that a picture and a text
request give the same replies together as alone.

    python3 scripts/vision_test.py PICTURE.png PORT

Needs only the standard library."""
import base64, json, sys, threading, time, urllib.request
picture, port = sys.argv[1], sys.argv[2]
url = "data:image/png;base64," + base64.b64encode(open(picture, "rb").read()).decode()
def call(body, path="/v1/chat/completions"):
    req = urllib.request.Request(f"http://localhost:{port}" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        out = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        out = json.load(e)
    return out, time.time() - t
def content(out):
    return out["choices"][0]["message"]["content"] if "choices" in out else out
body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe this picture in one sentence."}]}], "temperature": 0, "max_tokens": 40}
out, t = call(body)
print(f"{t:.1f}s", repr(content(out)), out.get("usage"))
out, t = call(body)
print(f"{t:.1f}s again (cached)", repr(content(out)), out.get("usage"))
body["messages"] += [{"role": "assistant", "content": content(out)}, {"role": "user", "content": "What colour is the circle?"}]
out, t = call(body)
print(f"{t:.1f}s follow-up", repr(content(out)))
b64 = url.split(",", 1)[1]
out, t = call({"model": "x", "max_tokens": 30, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}, {"type": "text", "text": "How many shapes are there?"}]}]}, "/v1/messages")
print(f"{t:.1f}s anthropic", out.get("content", out))
out, t = call({"model": "x", "stream": False, "options": {"temperature": 0}, "messages": [{"role": "user", "content": "What shape is red?", "images": [b64]}]}, "/api/chat")
print(f"{t:.1f}s ollama chat", out.get("message", out))
out, t = call({"model": "x", "stream": False, "options": {"temperature": 0}, "prompt": "What shape is green?", "images": [b64]}, "/api/generate")
print(f"{t:.1f}s ollama generate", out.get("response", out))
out, _ = call({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}]})
print("a URL:", out)
out, _ = call({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
print("not an image:", out)

# Batching changes nothing: each reply is the same whether the other request shares its batch.
text = {"messages": [{"role": "user", "content": "Write a short poem about the sea."}], "temperature": 0, "max_tokens": 60}
picture_body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "List every shape you can see, with its colour."}]}], "temperature": 0, "max_tokens": 60}
alone = [content(call(b)[0]) for b in (text, picture_body)]
together = [None, None]
def run(i, b):
    together[i] = content(call(b)[0])
threads = [threading.Thread(target=run, args=(i, b)) for i, b in enumerate((text, picture_body))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("together same as alone:", together == alone)
if together != alone:
    print(" alone:", alone, "\n together:", together)
