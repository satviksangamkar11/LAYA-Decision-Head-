"""Resident-weights variant of capture.capture.Capturer for GPUs that hold the truncated model (30 layers, about 7 GB fp16/bf16) in VRAM.
Same API (run, packed, plain, lm_calls) and same representation (depth d = residual after layer d, query = hidden at query_pos, option = fp32 mean over its span;
lm_head never run). Only the weight placement differs: no per-layer host-to-device copies. The model must already be on the GPU (model.to("cuda")).
The streaming Capturer stays the reference path; numerics on a new device or dtype are gated by tools/capture_pool_v3_t4.py (hard gates A, B, plain, repeat).

GPUs without the flash kernel (compute capability < 8, e.g. the Kaggle T4): transformers' sdpa_attention_forward asks SDPA for grouped-query attention (enable_gqa=True) whenever no mask
is passed (the plain-forward path). Only the flash and math backends accept enable_gqa, so on sm_75 SDPA silently uses the math kernel and materialises a full L x L x heads fp32
matrix (8.9 GiB at 8.6k tokens: the CUDA OOM seen on Kaggle). Same bug and same intent as huggingface/transformers PR 45776 (not merged), so here use_gqa_in_sdpa is switched off on such
GPUs: transformers then repeats the key/value heads and SDPA keeps the memory-efficient causal kernel. The packed path always passes a mask and is unaffected."""
import torch

from capture.capture import pack


def _no_gqa_in_sdpa():
    import transformers.integrations.sdpa_attention as S
    S.use_gqa_in_sdpa = lambda *a, **k: False


class ResidentCapturer:
    def __init__(self, model, depths, no_gqa=None):
        self.base, self.depths, self.cap, self.plan = model.model, set(depths), {}, None
        self.lm_calls = 0
        self.dev = next(self.base.parameters()).device
        assert self.dev.type == "cuda", "ResidentCapturer needs the model on the GPU"
        self.no_gqa = torch.cuda.get_device_capability(self.dev)[0] < 8 if no_gqa is None else no_gqa
        if self.no_gqa:
            _no_gqa_in_sdpa()
        model.lm_head.register_forward_hook(lambda *a: setattr(self, "lm_calls", self.lm_calls + 1))
        for i, l in enumerate(self.base.layers, 1):
            l.register_forward_hook(self._post(i), with_kwargs=True)

    def _post(self, i):
        def post(mod, args, kwargs, o):
            if i in self.depths:
                x = (o[0] if isinstance(o, tuple) else o)[0]
                q, spans = self.plan
                self.cap[i] = (x[q].float().cpu(), torch.stack([x[a:b].float().mean(0) for a, b in spans]).cpu())
            return o
        return post

    @torch.inference_mode()
    def run(self, ids, query, spans, position_ids=None, mask=None):
        self.cap, self.plan = {}, (query, spans)
        kw = {} if mask is None else dict(position_ids=position_ids.to(self.dev), attention_mask=mask)
        self.base(input_ids=ids.to(self.dev), use_cache=False, **kw)
        torch.cuda.synchronize()
        return self.cap

    def packed(self, r):
        ids, pos, mask, spans = pack(r)
        return self.run(ids, r["query_pos"], spans, pos, mask), (ids, spans)

    def plain(self, r, i):
        P, s = len(r["prefix_ids"]), r["suffix_ids"][i]
        ids = torch.tensor([list(r["prefix_ids"]) + list(s)])
        return self.run(ids, r["query_pos"], [(P, P + len(s))])
