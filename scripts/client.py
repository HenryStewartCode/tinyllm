import asyncio, sys
from openai import OpenAI, AsyncOpenAI
base = sys.argv[1]
c = OpenAI(base_url=base, api_key="none")
print([m.id for m in c.models.list().data])
r = c.chat.completions.create(model="m", messages=[{"role":"user","content":"Name a colour."}], max_tokens=12, temperature=0, logprobs=True, top_logprobs=2, n=2)
print(repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage, r.choices[0].logprobs.content[0].token, r.choices[0].logprobs.content[0].top_logprobs[0].token)
print(len(r.choices))
s = c.chat.completions.create(model="m", messages=[{"role":"user","content":"Count to five."}], max_tokens=30, temperature=0, stream=True, stream_options={"include_usage": True})
text = ""
for chunk in s:
    if chunk.choices and chunk.choices[0].delta.content: text += chunk.choices[0].delta.content
    if chunk.usage: print("usage", chunk.usage)
print(repr(text))
r = c.completions.create(model="m", prompt=["Hello my name is", "2+2="], max_tokens=5, temperature=0, n=1)
print([x.text for x in r.choices], r.usage)
r = c.chat.completions.create(model="m", messages=[{"role":"user","content":"Count to five."}], max_tokens=40, temperature=0, stop=["3"])
print(repr(r.choices[0].message.content), r.choices[0].finish_reason)
r = c.chat.completions.create(model="m", messages=[{"role":"user","content":"Hi"}], max_tokens=5, seed=3, temperature=1.0, extra_body={"top_k": 5, "min_p": 0.05, "repetition_penalty": 1.1}, frequency_penalty=0.5, presence_penalty=0.5, logit_bias={"33": -100})
print(repr(r.choices[0].message.content))
