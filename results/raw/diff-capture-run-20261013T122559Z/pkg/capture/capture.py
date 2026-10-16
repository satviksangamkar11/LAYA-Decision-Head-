"""Packed shared-prefix capture (design B). One forward over [prefix, suffix_1..suffix_N] with a block mask: every suffix attends
to the whole prefix and to its own earlier tokens only, with position ids restarting at the prefix length, so each option's
hidden states equal those of a separate [prefix + suffix_i] sequence. Layer weights stream through the GPU via hooks.
Representation: depth d = residual stream after decoder layer d (before the final norm); query = hidden at query_pos;
option = fp32 mean over the option's suffix span. lm_head is never run. No model name or size is hard-coded."""
import torch


def _dev(a, dev):
    if torch.is_tensor(a): return a.to(dev)
    if isinstance(a, (tuple, list)): return type(a)(_dev(v, dev) for v in a)
    if isinstance(a, dict): return {k: _dev(v, dev) for k, v in a.items()}
    return a


def pack(r, dev="cuda"):
    P, suf = len(r["prefix_ids"]), r["suffix_ids"]
    ids = list(r["prefix_ids"]) + [t for s in suf for t in s]
    pos = list(range(P)) + [P + j for s in suf for j in range(len(s))]
    L = len(ids)
    mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
    mask[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev))
    spans, o = [], P
    for s in suf:
        n = len(s)
        mask[o:o + n, :P] = True
        mask[o:o + n, o:o + n] = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev))
        spans.append((o, o + n))
        o += n
    return torch.tensor([ids]), torch.tensor([pos]), mask[None, None], spans


class Capturer:
    def __init__(self, model, depths):
        self.base, self.depths, self.cap, self.plan = model.model, set(depths), {}, None
        self.lm_calls = 0
        model.lm_head.register_forward_hook(lambda *a: setattr(self, "lm_calls", self.lm_calls + 1))
        for i, l in enumerate(self.base.layers, 1):
            l.register_forward_pre_hook(self._pre, with_kwargs=True)
            l.register_forward_hook(self._post(i), with_kwargs=True)
        self.base.embed_tokens.register_forward_hook(lambda m, a, o: o.to("cuda"))
        self.base.norm.to("cuda")

    def _pre(self, mod, args, kwargs):
        mod.to("cuda")
        return _dev(args, "cuda"), _dev(kwargs, "cuda")

    def _post(self, i):
        def post(mod, args, kwargs, o):
            mod.to("cpu")
            if i in self.depths:
                x = (o[0] if isinstance(o, tuple) else o)[0]
                q, spans = self.plan
                self.cap[i] = (x[q].float().cpu(), torch.stack([x[a:b].float().mean(0) for a, b in spans]).cpu())
            return o
        return post

    @torch.inference_mode()
    def run(self, ids, query, spans, position_ids=None, mask=None):
        self.cap, self.plan = {}, (query, spans)
        kw = {} if mask is None else dict(position_ids=position_ids.to("cuda"), attention_mask=mask)
        self.base(input_ids=ids, use_cache=False, **kw)
        torch.cuda.synchronize()
        return self.cap

    def packed(self, r):
        ids, pos, mask, spans = pack(r)
        return self.run(ids, r["query_pos"], spans, pos, mask), (ids, spans)

    def plain(self, r, i):
        P, s = len(r["prefix_ids"]), r["suffix_ids"][i]
        ids = torch.tensor([list(r["prefix_ids"]) + list(s)])
        return self.run(ids, r["query_pos"], [(P, P + len(s))])
