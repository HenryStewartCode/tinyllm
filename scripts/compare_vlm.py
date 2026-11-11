"""Check a vision-language model (Qwen2.5-VL) on a prompt with an image: the prompt's tokens, the
logits at every position and a greedy reply, against transformers in 32 bits.

usage: compare_vlm.py HARNESS MODEL_DIR [IMAGE] [DEVICE]

Without IMAGE, a test picture is drawn (448 x 336, already a whole number of patches, so nothing
is resampled). DEVICE (cpu, cuda or metal; default cpu) is where tinyllm runs."""
import json, os, subprocess, sys, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

tool, model_dir = sys.argv[1:3]
image_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
text = "Describe this picture in one sentence."
steps = 24

work = tempfile.mkdtemp()
if image_path is None:
    image = Image.new("RGB", (448, 336), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 40, 170, 180], fill=(220, 30, 30))
    draw.ellipse([230, 60, 400, 230], fill=(30, 60, 220))
    draw.polygon([(120, 300), (220, 200), (320, 300)], fill=(30, 170, 60))
    image_path = os.path.join(work, "picture.png")
    image.save(image_path)
image = Image.open(image_path).convert("RGB")

# tinyllm first, so that the two models aren't in memory at once.
out = os.path.join(work, "logits.bin")
run = subprocess.run([tool, "vlm", model_dir, image_path, out, str(steps), text], check=True, capture_output=True, text=True,
                     env={**os.environ, "DEVICE": device})
ours = json.loads(run.stdout)
logits = np.fromfile(out, dtype=np.float32).reshape(len(ours["tokens"]), -1)

processor = AutoProcessor.from_pretrained(model_dir)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
with torch.no_grad():
    reference = model(**inputs).logits[0].numpy()
    # Plain greedy: generate() would otherwise apply the repetition penalty in the model's
    # generation_config.json, which the harness's greedy reply doesn't.
    generated = model.generate(**inputs, max_new_tokens=steps, do_sample=False,
                               repetition_penalty=1.0)[0, inputs["input_ids"].shape[1]:].tolist()
    # Its reply appended to the prompt, for comparing logits along the same reply below.
    full = torch.cat([inputs["input_ids"], torch.tensor([generated])], dim=1)
    types = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, len(generated), dtype=inputs["mm_token_type_ids"].dtype)], dim=1)
    along = model(input_ids=full, attention_mask=torch.ones_like(full), mm_token_type_ids=types,
                  pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]).logits[0].numpy()
image_token_id = model.config.image_token_id
del model
import gc; gc.collect()

hf_tokens = inputs["input_ids"][0].tolist()
print(f"tokens: {len(hf_tokens)} from transformers, {len(ours['tokens'])} from tinyllm, "
      f"{'the same' if hf_tokens == ours['tokens'] else 'DIFFERENT'} ({run.stderr.strip().splitlines()[-1]})")
image_rows = [i for i, t in enumerate(hf_tokens) if t == image_token_id]
differences = np.abs(logits - reference).max(axis=1)
print(f"largest logit difference: {differences.max():.2e} (image tokens {differences[image_rows].max():.2e}, "
      f"last token {differences[-1]:.2e}); logits up to {np.abs(reference).max():.1f}")
print(f"top-1 agrees at {(logits.argmax(1) == reference.argmax(1)).mean():.0%} of positions")
# The harness carries on past the end of the reply, where transformers stops.
eos = json.load(open(os.path.join(model_dir, "generation_config.json"))).get("eos_token_id", [])
stops = set(eos if isinstance(eos, list) else [eos])
ended = next((i + 1 for i, t in enumerate(ours["greedy"]) if t in stops), len(ours["greedy"]))
ours["greedy"] = ours["greedy"][:ended]
ours["text"] = processor.decode(ours["greedy"])
same = next((i for i, (a, b) in enumerate(zip(ours["greedy"], generated)) if a != b), min(len(generated), len(ours["greedy"])))
print(f"greedy: {same} of {len(generated)} tokens the same")
print("transformers:", repr(processor.decode(generated)))
print("tinyllm:     ", repr(ours["text"]))

# The same reply through both, to see whether they disagree about it or only part ways on a near
# tie: transformers' greedy reply, appended to the prompt.
forced = generated
run = subprocess.run([tool, "vlm", model_dir, image_path, out, "1", text], check=True, capture_output=True, text=True,
                     env={**os.environ, "DEVICE": device, "VLM_FORCE": ",".join(map(str, forced))})
logits = np.fromfile(out, dtype=np.float32).reshape(-1, along.shape[1])
replied = slice(len(hf_tokens) - 1, len(hf_tokens) - 1 + len(forced))
gaps = np.sort(along[replied], axis=1)
print(f"along transformers' reply: largest logit difference {np.abs(logits[replied] - along[replied]).max():.2e}, "
      f"top-1 agrees {(logits[replied].argmax(1) == along[replied].argmax(1)).mean():.0%}, "
      f"smallest gap between transformers' top two {(gaps[:, -1] - gaps[:, -2]).min():.3f}")
