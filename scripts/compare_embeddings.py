"""Compare tinyllm's /v1/embeddings with Hugging Face for a last-token-pooling embedding model
(such as Qwen3-Embedding): the final hidden state of the last token, normalized.

usage: compare_embeddings.py BASE_URL MODEL_DIR"""
import json, sys, urllib.request
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

base, model_dir = sys.argv[1:3]
texts = [
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?",
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other.",
    "def add(a, b):\n    return a + b",
    "短い日本語の文です。",
]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
with torch.no_grad():
    reference = []
    for text in texts:
        ids = tok(text, return_tensors="pt")
        hidden = model(**ids).last_hidden_state[0, -1]
        reference.append(torch.nn.functional.normalize(hidden, dim=-1).numpy())
reference = np.stack(reference)

request = urllib.request.Request(f"{base}/v1/embeddings", json.dumps({"input": texts}).encode(), {"Content-Type": "application/json"})
response = json.load(urllib.request.urlopen(request))
ours = np.array([d["embedding"] for d in response["data"]])
cosine = (ours * reference).sum(-1)
print(f"{len(texts)} texts, {ours.shape[1]} dimensions, {response['usage']['prompt_tokens']} tokens")
print(f"cosine similarity with Hugging Face: min {cosine.min():.6f}  largest difference {np.abs(ours - reference).max():.2e}")
print("similarities between texts, ours:  ", np.round(ours[0] @ ours[1:].T, 4))
print("similarities between texts, theirs:", np.round(reference[0] @ reference[1:].T, 4))

request = urllib.request.Request(f"{base}/v1/embeddings", json.dumps({"input": texts[1], "dimensions": 256}).encode(), {"Content-Type": "application/json"})
short = np.array(json.load(urllib.request.urlopen(request))["data"][0]["embedding"])
cut = reference[1][:256] / np.linalg.norm(reference[1][:256])
print(f"dimensions=256: {short.shape[0]} values, cosine with the cut reference {float(short @ cut):.6f}")
