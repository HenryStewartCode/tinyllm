"""Quantize a GPT-OSS checkpoint's experts (plain matrices, as transformers saves a model it built)
into MXFP4, as OpenAI's checkpoints store them, and also save the same model with the quantized
values unpacked into plain matrices. The two must give the same results, which checks reading
MXFP4. Only numpy is needed.

usage: make_mxfp4.py GPT_OSS_DIR MXFP4_DIR UNPACKED_DIR"""
import json, os, shutil, struct, sys
import numpy as np

source, packed_dir, plain_dir = sys.argv[1:4]
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
DTYPES = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16, "U8": np.uint8, "I32": np.int32}

def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        array = np.frombuffer(data[begin:end], dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
        if info["dtype"] == "BF16":
            array = (array.astype(np.uint32) << 16).view(np.float32)
        out[name] = array
    return out

def write(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        dtype = "U8" if array.dtype == np.uint8 else "F32"
        array = np.ascontiguousarray(array, dtype=np.uint8 if dtype == "U8" else np.float32)
        blob = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)

def quantize(w):
    """[..., cols] -> (blocks [..., cols / 32, 16], scales [..., cols / 32], the values they hold)."""
    groups = w.reshape(*w.shape[:-1], -1, 32).astype(np.float64)
    largest = np.abs(groups).max(-1)
    exponent = np.clip(np.ceil(np.log2(np.maximum(largest, 1e-30) / 6.0)), -126, 127).astype(np.int64)
    scaled = groups / np.exp2(exponent)[..., None]
    magnitude = np.abs(np.abs(scaled)[..., None] - FP4[:8]).argmin(-1)
    codes = (magnitude | np.where(scaled < 0, 8, 0)).astype(np.uint8)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    values = FP4[codes] * np.exp2(exponent)[..., None]
    return blocks, (exponent + 127).astype(np.uint8), values.reshape(w.shape)

tensors = {}
for name in sorted(os.listdir(source)):
    if name.endswith(".safetensors"):
        tensors.update(read(os.path.join(source, name)))
packed, plain = {}, {}
for name, value in tensors.items():
    if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
        # Plain matrices are [experts, in, out]; MXFP4 is packed along `in`, as [experts, out, in].
        blocks, scales, values = quantize(np.swapaxes(value, 1, 2))
        packed[name + "_blocks"] = blocks
        packed[name + "_scales"] = scales
        plain[name] = np.swapaxes(values, 1, 2)
    else:
        packed[name] = plain[name] = value
for directory, weights in [(packed_dir, packed), (plain_dir, plain)]:
    os.makedirs(directory, exist_ok=True)
    write(os.path.join(directory, "model.safetensors"), weights)
    for name in os.listdir(source):
        if name.endswith(".json") and "safetensors" not in name:
            shutil.copy(os.path.join(source, name), directory)
print(f"wrote {packed_dir} (MXFP4 experts) and {plain_dir} (the same values unpacked)")
