"""Pre-flight on the Kaggle GPU: which SDPA variants fit and run at a real sequence length. Run from the notebook after the bundle is copied: exec(open(W + "/tools/sdpa_preflight.py").read())."""
import time
import torch
# pre-flight: which SDPA variants fit and run on THIS GPU at a real sequence length (8,650 tokens, 32 query heads, 8 kv heads, head_dim 128, fp16)
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
L, dev = 8650, "cuda:0"
q = torch.randn(1, 32, L, 128, dtype=torch.float16, device=dev); k = torch.randn(1, 8, L, 128, dtype=torch.float16, device=dev); v = torch.randn_like(k)
kk, vv = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))[None, None]
def probe(name, fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); b = torch.cuda.memory_allocated()
    try:
        t0 = time.time(); fn(); torch.cuda.synchronize()
        print(f"{name}: OK, extra peak {(torch.cuda.max_memory_allocated() - b) / 2**20:.0f} MiB, {time.time() - t0:.2f} s")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:140]}")
    torch.cuda.empty_cache()
def only_eff(f):
    def g():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            f()
    return g
probe("A  enable_gqa=True, causal, no mask (what transformers does on the plain path), default dispatch", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
probe("A2 enable_gqa=True with the memory-efficient kernel only", only_eff(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)))
probe("B  kv heads repeated, causal, no mask (the fix)", lambda: F.scaled_dot_product_attention(q, kk, vv, is_causal=True))
probe("C  kv heads repeated, bool block mask (the packed path)", lambda: F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask))
del q, k, v, kk, vv, mask
torch.cuda.empty_cache()
