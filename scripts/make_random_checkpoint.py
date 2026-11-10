"""Write a checkpoint with random weights for a model's config, for measuring speed at a real
model's sizes without downloading it: the tensors transformers would make for the config, as
bfloat16, optionally with fewer layers. Only the names and shapes come from transformers, so
nothing big is ever held in memory as floats.

usage: make_random_checkpoint.py CONFIG_JSON TOKENIZER_DIR OUT_DIR [LAYERS]

The tokenizer files are copied from TOKENIZER_DIR."""
import json, os, shutil, struct, sys
import numpy as np, torch
from transformers import AutoConfig, AutoModelForCausalLM

config_path, tokenizer_dir, out = sys.argv[1:4]
settings = json.load(open(config_path))
if len(sys.argv) > 4:
    settings["num_hidden_layers"] = int(sys.argv[4])
settings.pop("auto_map", None)
model_type = settings.pop("model_type")
config = AutoConfig.for_model(model_type, **settings)
# Imported first: importing under the meta device breaks torch.
model_class = AutoModelForCausalLM._model_mapping[type(config)]
with torch.device("meta"):
    model = model_class(config)
shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}

rng = np.random.default_rng(0)
SHARD_BYTES = 4 << 30

def values(name, shape):
    """Norms start at 1 and biases at 0, as in training; everything else is small and random."""
    n = int(np.prod(shape))
    if name.endswith(("norm.weight", "layernorm.weight")):
        return np.full(n, 0x3F80, dtype=np.uint16)
    if name.endswith(("bias", "e_score_correction_bias")):
        return np.zeros(n, dtype=np.uint16)
    floats = rng.standard_normal(n, dtype=np.float32) * 0.02
    return (floats.view(np.uint32) >> 16).astype(np.uint16)

os.makedirs(out, exist_ok=True)
shards, current, size = [], [], 0
for name, shape in shapes.items():
    nbytes = 2 * int(np.prod(shape))
    if current and size + nbytes > SHARD_BYTES:
        shards.append(current)
        current, size = [], 0
    current.append((name, shape))
    size += nbytes
shards.append(current)

weight_map = {}
for i, shard in enumerate(shards):
    file = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    header, offset = {}, 0
    for name, shape in shard:
        nbytes = 2 * int(np.prod(shape))
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        weight_map[name] = file
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(os.path.join(out, file), "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for name, shape in shard:
            f.write(values(name, shape).tobytes())
    print(f"{file}: {len(shard)} tensors, {offset / 1e9:.2f} GB")
json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
json.dump({"model_type": model_type, **settings}, open(os.path.join(out, "config.json"), "w"), indent=2)
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    if os.path.exists(os.path.join(tokenizer_dir, name)):
        shutil.copy(os.path.join(tokenizer_dir, name), out)
total = sum(2 * int(np.prod(s)) for s in shapes.values())
print(f"{out}: {total / 2 / 1e9:.2f}B parameters")
