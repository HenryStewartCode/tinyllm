"""Build small models with random weights for each architecture and compare our logits with Hugging Face's.

The maths doesn't need real weights to be checked: any difference in the forward pass shows up as a
difference in logits. usage: compare_random.py BENCH WORKDIR [case...]"""
import json, os, shutil, subprocess, sys
import numpy as np, torch
from transformers import AutoConfig, AutoModelForCausalLM

tool, work = sys.argv[1:3]
only = sys.argv[3:]
# Any Llama-style tokenizer will do; by default the one in the repository's models/.
tok_source = os.environ.get("TOKENIZER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "SmolLM2-135M-Instruct"))
vocab = 49152
common = dict(vocab_size=vocab, hidden_size=256, intermediate_size=512, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
              max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
gemma = dict(head_dim=32)
cases = {
    "llama": ("llama", {}),
    "qwen3": ("qwen3", {}),
    "qwen3-window": ("qwen3", dict(use_sliding_window=True, sliding_window=9, max_window_layers=3)),
    "mistral-window": ("mistral", dict(sliding_window=9)),
    # YaRN, with a short original context so that the prompts go well past it.
    "qwen2-yarn": ("qwen2", dict(rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16})),
    "llama-yarn-mscale": ("llama", dict(rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 12,
                                                      "beta_fast": 16, "beta_slow": 2, "mscale": 0.707, "mscale_all_dim": 1.0})),
    "phi3": ("phi3", dict(pad_token_id=0)),
    "phi3-window": ("phi3", dict(sliding_window=7)),
    "granite": ("granite", dict(embedding_multiplier=3.0, logits_scaling=2.0, residual_multiplier=0.4, attention_multiplier=0.1)),
    "smollm3": ("smollm3", dict(no_rope_layer_interval=2)),
    "gemma": ("gemma", dict(gemma)),
    "gemma2": ("gemma2", dict(gemma, query_pre_attn_scalar=16, attn_logit_softcapping=2.0, final_logit_softcapping=3.0)),
    "gemma2-head256": ("gemma2", dict(hidden_size=512, head_dim=256, num_attention_heads=2, num_key_value_heads=1, query_pre_attn_scalar=64, attn_logit_softcapping=None, final_logit_softcapping=None)),
    "gemma3": ("gemma3_text", dict(gemma, sliding_window=9, sliding_window_pattern=2, query_pre_attn_scalar=16)),
    "gemma3-long": ("gemma3_text", dict(gemma, sliding_window=33, sliding_window_pattern=3, query_pre_attn_scalar=64, final_logit_softcapping=3.0)),
    # Mixtures of experts.
    "qwen3_moe": ("qwen3_moe", dict(moe_intermediate_size=96, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[1])),
    "mixtral": ("mixtral", dict(num_local_experts=6, num_experts_per_tok=2)),
    "olmoe": ("olmoe", dict(num_experts=8, num_experts_per_tok=3)),
    "qwen2_moe": ("qwen2_moe", dict(moe_intermediate_size=96, shared_expert_intermediate_size=192, num_experts=8, num_experts_per_tok=2)),
    "granitemoe": ("granitemoe", dict(num_local_experts=8, num_experts_per_tok=2, embedding_multiplier=2.0, residual_multiplier=0.5, attention_multiplier=0.1, logits_scaling=2.0)),
    # Attention sinks, alternating windows, YaRN, biases everywhere and a clamped activation.
    "gpt_oss": ("gpt_oss", dict(head_dim=32, num_local_experts=8, num_experts_per_tok=2, sliding_window=9)),
    # DeepSeek's latent attention, with its experts chosen by group (V2: softmax, best one per
    # group; V3: sigmoids plus a correction, best two per group), shared experts, dense first layers.
    "deepseek_v2": ("deepseek_v2", dict(q_lora_rank=None, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
                                        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=2, moe_intermediate_size=64,
                                        first_k_dense_replace=1, topk_method="group_limited_greedy", n_group=2, topk_group=1,
                                        routed_scaling_factor=1.5, num_key_value_heads=8)),
    "deepseek_v3": ("deepseek_v3", dict(q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
                                        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, moe_intermediate_size=64,
                                        first_k_dense_replace=1, n_group=4, topk_group=2, routed_scaling_factor=2.5, num_key_value_heads=8,
                                        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                                                      "mscale": 1.0, "mscale_all_dim": 1.0, "beta_fast": 32, "beta_slow": 1})),
    # More heads than one 16-row tile holds (24: a full tile and a part), a wider latent, and V2's
    # plain greedy choice of experts.
    "deepseek_v2-heads": ("deepseek_v2", dict(hidden_size=384, num_attention_heads=24, num_key_value_heads=24, q_lora_rank=None, kv_lora_rank=64,
                                              qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, n_routed_experts=8,
                                              num_experts_per_tok=3, n_shared_experts=1, moe_intermediate_size=64, first_k_dense_replace=2,
                                              topk_method="greedy", routed_scaling_factor=1.0,
                                              rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 16,
                                                            "mscale": 0.707, "mscale_all_dim": 0.707, "beta_fast": 32, "beta_slow": 1})),
    # Latent attention alone: dense layers only, so that no choice of experts can tip one way on
    # one device and the other way on another.
    "deepseek_v3-dense": ("deepseek_v3", dict(q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
                                              n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, moe_intermediate_size=64,
                                              first_k_dense_replace=6, num_key_value_heads=8)),
}
# The GPU runs these only in 16 bits.
CPU_ONLY = set()
BF16_ONLY = {"qwen3_moe", "mixtral", "olmoe", "granitemoe", "gpt_oss", "qwen2_moe", "deepseek_v2", "deepseek_v3",
             "deepseek_v2-heads", "deepseek_v3-dense"}
torch.manual_seed(5)
rng = np.random.default_rng(5)
lengths = [3, 17, 40, 70]
prompts = [[1] + rng.integers(3, vocab, n - 1).tolist() for n in lengths]

def ours(model_dir, ids, **env):
    out = os.path.join(work, "logits.bin")
    subprocess.run([tool, "logits", model_dir, out, *map(str, ids)], check=True, env={**os.environ, **env}, capture_output=True)
    return np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)

print(f"{'case':16} {'tokens':>6} | {'CPU':>9} {'GPU 32':>9} {'GPU 16':>9} {'HF 16':>9} | {'range':>6} | top-1 agrees (GPU 16)")
worst = 0
for name, (kind, extra) in cases.items():
    if only and name not in only:
        continue
    config = AutoConfig.for_model(kind, **{**common, **extra})
    if kind == "gemma3_text":
        config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    with torch.no_grad():
        for p in model.parameters():
            # Bigger weights than the usual initialization, so that everything actually matters.
            if p.dim() >= 2 and p.shape[0] != vocab:
                p.mul_(5.0)
            elif p.dim() == 1:
                p.copy_(torch.randn_like(p) * 0.2)
        for buffer_name, buffer in model.named_buffers():
            # DeepSeek V3 chooses experts by their scores plus this, which starts at zero.
            if buffer_name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn_like(buffer) * 0.3)
    path = os.path.join(work, name)
    shutil.rmtree(path, ignore_errors=True)
    model.save_pretrained(path, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(os.path.join(tok_source, f), path)
    half = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    full = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    for ids in prompts:
        with torch.no_grad():
            a = full(torch.tensor([ids])).logits[0].float().numpy()
            b = half(torch.tensor([ids])).logits[0].float().numpy()
        cpu = ours(path, ids, DEVICE="cpu")
        if name in CPU_ONLY:
            g32 = g16 = cpu
        else:
            g32 = cpu if name in BF16_ONLY else ours(path, ids, DEVICE="cuda", PRECISION="f32")
            g16 = ours(path, ids, DEVICE="cuda", PRECISION="bf16")
        d = lambda x: float(np.abs(x - a).max())
        worst = max(worst, d(cpu), d(g32))
        print(f"{name:16} {len(ids):6d} | {d(cpu):9.2e} {d(g32):9.2e} {d(g16):9.2e} {d(b):9.2e} | {np.abs(a).max():6.1f} | {(g16.argmax(-1) == a.argmax(-1)).mean():.0%}")
    devices = [("cpu", "f32")] if name in CPU_ONLY else [("cuda", "bf16")] if name in BF16_ONLY else [("cuda", "f32"), ("cuda", "bf16")]
    for device, precision in devices:
        r = subprocess.run([tool, "invariance", path], capture_output=True, text=True, env={**os.environ, "DEVICE": device, "PRECISION": precision})
        print(f"    batching, {device} {precision}: {r.stdout.strip() or r.stderr[-300:]}")
print("largest difference from 32-bit Hugging Face in 32-bit runs:", worst)
