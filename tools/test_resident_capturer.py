"""Equivalence test of capture.capture_resident.ResidentCapturer against the streaming capture.capture.Capturer on a TINY random Qwen3 (a few MB, light on the GPU).
Same packed record, same weights: captured vectors must be bit-identical (same device, same kernels; only weight placement differs). No real model, no data.
    PYTHONPATH=. .venv/Scripts/python.exe tools/test_resident_capturer.py"""
import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from capture.capture import Capturer
from capture.capture_resident import ResidentCapturer

torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
D = [2, 4, 6]
for dtype in (torch.float32, torch.float16):
    base = Qwen3ForCausalLM(cfg).to(dtype).eval()
    R = dict(prefix_ids=list(range(5, 45)), query_pos=39, suffix_ids=[[7, 8, 9], [11, 12], [13, 14, 15, 16], [21]])
    stream = Capturer(copy.deepcopy(base), D)
    res_model = copy.deepcopy(base).to("cuda")
    res = ResidentCapturer(res_model, D)
    a, _ = stream.packed(R)
    b, _ = res.packed(R)
    worst = max(max((a[d][0] - b[d][0]).abs().max().item(), (a[d][1] - b[d][1]).abs().max().item()) for d in D)
    pa, pb = stream.plain(R, 1), res.plain(R, 1)
    worst_plain = max(max((pa[d][0] - pb[d][0]).abs().max().item(), (pa[d][1] - pb[d][1]).abs().max().item()) for d in D)
    print(dtype, "packed max abs diff", worst, "| plain max abs diff", worst_plain, "| lm_head calls", stream.lm_calls, res.lm_calls)
    assert worst == 0.0 and worst_plain == 0.0 and stream.lm_calls == 0 and res.lm_calls == 0
print("PASS: resident capture is bit-identical to streaming capture on a tiny random Qwen3 (fp32 and fp16)")

# ---- no_gqa switch is harmless: same vectors with and without it. The math-kernel blow-up itself is torch<=2.10-specific (transformers PR 45776 reports it on a T4 with torch 2.10.0) and does NOT
# ---- reproduce on torch 2.14 even with flash off; it is measured directly on the Kaggle T4 by the notebook pre-flight cell.
import transformers.integrations.sdpa_attention as S
torch.backends.cuda.enable_flash_sdp(False)
big = Qwen3Config(vocab_size=500, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=8, head_dim=64, tie_word_embeddings=True)
mdl = Qwen3ForCausalLM(big).half().eval().to("cuda")
Rb = dict(prefix_ids=[i % 400 + 5 for i in range(3000)], query_pos=2999, suffix_ids=[[7, 8, 9, 10]])
orig = S.use_gqa_in_sdpa
res = {}
for flag in (False, True):
    S.use_gqa_in_sdpa = orig
    cap = ResidentCapturer(mdl, [1, 2], no_gqa=flag)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    out = cap.plain(Rb, 0)
    res[flag] = ((torch.cuda.max_memory_allocated() - base_mem) / 2**20, out)
    for h in list(mdl.model.layers):
        h._forward_hooks.clear()
    mdl.lm_head._forward_hooks.clear()
S.use_gqa_in_sdpa = orig
cos = min(torch.nn.functional.cosine_similarity(res[False][1][d][0].double(), res[True][1][d][0].double(), dim=-1).item() for d in (1, 2))
print(f"flash off, plain forward at 3,004 tokens, 32 heads: extra peak memory with enable_gqa {res[False][0]:.0f} MiB, with the fix {res[True][0]:.0f} MiB; cosine between the two results {cos:.7f}")
assert cos >= 0.9999
print("PASS: switching GQA-in-SDPA off keeps the vectors (memory effect is not reproducible on this torch version)")
