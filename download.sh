#!/usr/bin/env bash
# Download the files tinyllm needs from a Hugging Face model repository.
#
#   ./download.sh HuggingFaceTB/SmolLM2-135M-Instruct     # into models/SmolLM2-135M-Instruct
#   ./download.sh Qwen/Qwen2.5-0.5B-Instruct some/dir     # into some/dir
#
# Gated repositories (e.g. meta-llama/*) need an access token:
#   HF_TOKEN=hf_xxx ./download.sh meta-llama/Llama-3.2-1B-Instruct
set -euo pipefail

repo="${1:?usage: $0 <huggingface-repo> [destination-dir]}"
dest="${2:-models/${repo##*/}}"
base="https://huggingface.co/$repo/resolve/main"
auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi

# fetch FILE MODE: download one file. MODE is "required", "optional" (missing is fine) or
# "large" (show a progress bar).
fetch() {
  local flags=(-fL -sS)
  [[ $2 == optional ]] && flags=(-fL -s)
  [[ $2 == large ]] && flags=(-fL --progress-bar)
  if curl "${flags[@]}" ${auth[@]+"${auth[@]}"} -o "$dest/$1.part" "$base/$1"; then
    mv "$dest/$1.part" "$dest/$1"
  else
    rm -f "$dest/$1.part"
    [[ $2 == optional ]] || { echo "error: couldn't download $1 from $repo" >&2; exit 1; }
    return 1
  fi
}

mkdir -p "$dest"
echo "Downloading $repo into $dest"
fetch config.json required
fetch tokenizer.json required
for file in tokenizer_config.json generation_config.json chat_template.jinja; do
  fetch "$file" optional || true
done
if fetch model.safetensors.index.json optional; then
  # Large models are split into shards listed in the index.
  for shard in $(grep -o '"[^"]*\.safetensors"' "$dest/model.safetensors.index.json" | tr -d '"' | sort -u); do
    echo "$shard"
    fetch "$shard" large
  done
else
  echo "model.safetensors"
  fetch model.safetensors large
fi
echo "Done. Try: cargo run --release -- -m $dest --chat"
