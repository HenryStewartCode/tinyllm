"""Compare tinyllm's tokenizer and chat template with Hugging Face's, for one model directory.

usage: compare_text.py HARNESS MODEL_DIR

HARNESS is the `harness` example (target/release/examples/harness). Prints a line per mismatch and
a summary for each part; exits non-zero if anything differs."""
import json, os, subprocess, sys, tempfile
from transformers import AutoTokenizer

tool, model_dir = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(model_dir)

STRINGS = [
    "Hello, world!",
    "  two leading spaces",
    "trailing spaces   ",
    "multiple   spaces\tand\ttabs\n\nand newlines\n",
    "I'm sure they'll say we've done what you'd've done",
    "1234567890 3.14159 1,000,000 -42 1e-9",
    "émigré café naïve façade Ünïcödé ÀÉÎÕÜ",
    "日本語のテキストです。中文文本。한국어 텍스트.",
    "emoji 😀🎉👍🏽 and a family 👨‍👩‍👧",
    "def square(x):\n    return x ** 2  # four spaces\n\n\tclass A: pass\n",
    "<|im_start|>user\nhi<|im_end|>",
    "<think>\nreasoning goes here\n</think>\n\nanswer",
    "<s>[INST] hello [/INST]</s>",
    "URL: https://example.com/path?q=1&r=2#frag",
    " non-breaking em​zero-width",
    "mixed123numbers456and789letters",
    "   ",
    "\n\n\n\n",
    "Ελληνικά Русский العربية עברית हिन्दी ไทย",
    '{"json": [1, 2, {"a": null, "b": true}], "s": "x\\ny"}',
    "a" * 50,
    "The quick brown fox jumps over the lazy dog.",
    "tab\tseparated\tvalues\t\t",
    "math: ∑∫√∞≈≠≤≥ αβγ ½ ⅓",
    "Mr. O'Neil's 3rd-place finish—an upset!—stunned fans…",
    "    indented\n        more indented\n",
]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]},
}}]
CONVERSATIONS = [
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"messages": [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "Hello there!"}]},
    {"messages": [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "And another?"},
    ]},
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}], "tools": TOOLS},
    {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21}'},
    ], "tools": TOOLS},
]


def harness(command, data):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
    try:
        out = subprocess.run([tool, command, model_dir, f.name], check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(f.name)
    return json.loads(out)


def ids_of(result):
    return list(result["input_ids"] if hasattr(result, "keys") else result)


failures = 0
ours = harness("tokenize", STRINGS)
bad = 0
for i, text in enumerate(STRINGS):
    special = tok(text, add_special_tokens=True)["input_ids"]
    plain = tok(text, add_special_tokens=False)["input_ids"]
    decoded = tok.decode(plain)
    problems = []
    if ours["special"][i] != special:
        problems.append(f"with special tokens: ours {ours['special'][i][:12]}... theirs {special[:12]}...")
    if ours["plain"][i] != plain:
        problems.append(f"without: ours {ours['plain'][i][:12]}... theirs {plain[:12]}...")
    if ours["decoded"][i] != decoded:
        problems.append(f"decoded: ours {ours['decoded'][i]!r} theirs {decoded!r}")
    if problems:
        bad += 1
        print(f"  string {i} {text[:30]!r}: " + "; ".join(problems))
print(f"tokenizer: {len(STRINGS) - bad}/{len(STRINGS)} strings identical (ids with and without special tokens, and decoding)")
failures += bad

if tok.chat_template:
    ours = harness("chat", CONVERSATIONS)
    bad = 0
    for i, (conversation, got) in enumerate(zip(CONVERSATIONS, ours)):
        kwargs = dict(tools=conversation.get("tools"), add_generation_prompt=True)
        try:
            text = tok.apply_chat_template(conversation["messages"], tokenize=False, **kwargs)
            ids = ids_of(tok.apply_chat_template(conversation["messages"], tokenize=True, **kwargs))
        except Exception as e:  # templates may refuse some conversations; ours should too
            if "error" not in got:
                bad += 1
                print(f"  conversation {i}: theirs refused ({e}), ours rendered it")
            continue
        if "error" in got:
            bad += 1
            print(f"  conversation {i}: ours refused ({got['error']}), theirs rendered it")
        elif got["text"] != text:
            bad += 1
            at = next((j for j, (a, b) in enumerate(zip(got["text"], text)) if a != b), min(len(got["text"]), len(text)))
            print(f"  conversation {i}: text differs at {at}: ours {got['text'][at-20:at+40]!r} theirs {text[at-20:at+40]!r}")
        elif got["ids"] != ids:
            bad += 1
            print(f"  conversation {i}: same text, different ids")
    print(f"chat template: {len(CONVERSATIONS) - bad}/{len(CONVERSATIONS)} conversations identical (text and token ids)")
    failures += bad
else:
    print("chat template: none")
sys.exit(1 if failures else 0)
