"""Make a LoRA adapter with random weights for a model, in PEFT's format, and optionally the model
with the adapter merged into its weights (W + scale * B A), which gives the same results as the
model running the adapter. Comparing the two checks the adapter code; only numpy is needed.

usage: make_lora.py MODEL_DIR ADAPTER_DIR [--rank R] [--alpha A] [--modules q_proj,v_proj,...]
                    [--merged MERGED_DIR] [--seed S]"""
import argparse, json, os, shutil, struct
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("adapter")
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--alpha", type=float, default=16.0)
parser.add_argument("--modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--merged")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I32": np.int32}

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        array = np.frombuffer(data[begin:end], dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
        if info["dtype"] == "BF16":
            array = (array.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = array
    return tensors

def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array, dtype=np.float32)
        blob = array.tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)

config = json.load(open(os.path.join(args.model, "config.json")))
dim, hidden = config["hidden_size"], config["intermediate_size"]
heads, kv_heads = config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"])
head_dim = config.get("head_dim") or dim // heads
shapes = {  # (in, out)
    "q_proj": (dim, heads * head_dim), "k_proj": (dim, kv_heads * head_dim), "v_proj": (dim, kv_heads * head_dim),
    "o_proj": (heads * head_dim, dim), "gate_proj": (dim, hidden), "up_proj": (dim, hidden), "down_proj": (hidden, dim),
}
place = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
         "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
modules = args.modules.split(",")
rng = np.random.default_rng(args.seed)
scale = args.alpha / args.rank
adapter = {}
for layer in range(config["num_hidden_layers"]):
    for module in modules:
        n_in, n_out = shapes[module]
        prefix = f"base_model.model.model.layers.{layer}.{place[module]}.{module}"
        # Big enough to change the results clearly, small enough to keep the model sensible.
        a = rng.standard_normal((args.rank, n_in)) / np.sqrt(n_in)
        b = rng.standard_normal((n_out, args.rank)) * 0.02
        adapter[f"{prefix}.lora_A.weight"] = a.astype(np.float32)
        adapter[f"{prefix}.lora_B.weight"] = b.astype(np.float32)
os.makedirs(args.adapter, exist_ok=True)
write_safetensors(os.path.join(args.adapter, "adapter_model.safetensors"), adapter)
json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": args.rank, "lora_alpha": args.alpha,
           "target_modules": modules, "bias": "none", "lora_dropout": 0.0, "use_rslora": False,
           "base_model_name_or_path": os.path.abspath(args.model)},
          open(os.path.join(args.adapter, "adapter_config.json"), "w"), indent=2)
print(f"wrote a rank-{args.rank} adapter for {len(modules)} modules to {args.adapter}")

if args.merged:
    os.makedirs(args.merged, exist_ok=True)
    weights = {}
    files = sorted(f for f in os.listdir(args.model) if f.endswith(".safetensors"))
    for name in files:
        weights.update(read_safetensors(os.path.join(args.model, name)))
    for layer in range(config["num_hidden_layers"]):
        for module in modules:
            prefix = f"model.layers.{layer}.{place[module]}.{module}"
            a = adapter[f"base_model.{prefix.replace('model.', 'model.model.', 1)}.lora_A.weight"]
            b = adapter[f"base_model.{prefix.replace('model.', 'model.model.', 1)}.lora_B.weight"]
            merged = weights[f"{prefix}.weight"].astype(np.float64) + scale * (b.astype(np.float64) @ a.astype(np.float64))
            weights[f"{prefix}.weight"] = merged
    write_safetensors(os.path.join(args.merged, "model.safetensors"), weights)
    for name in os.listdir(args.model):
        if name.endswith(".json") and "safetensors" not in name:
            shutil.copy(os.path.join(args.model, name), args.merged)
    print(f"wrote the merged model to {args.merged}")
