"""[EL-ARCH-1 local run] The capture function (context text + option texts -> hq [K,H], ho [N,K,H] fp16) for ANY backbone profile. It is the closure that kaggle/evidence_ladder_laya_t6_step_v1.py builds
inline for the 4B model, parametrized by the profile (depths, cue, option prefix) and by the existing capture/capture.py recipe classes: NO second capture implementation.
  resident profile -> capture.capture_resident.ResidentCapturer (gated so its hooks are silent during generation)
  stream profile   -> capture.capture.Capturer (layer weights stream through the GPU; the model lives on the CPU)
Representation (unchanged): depth d = residual stream after decoder layer d; query = hidden at the last prefix token (after the cue); option = fp32 mean over the option's own tokens (independent
suffix of the shared prefix); the forward pass stops after the deepest captured layer; lm_head never runs. torch / transformers are imported lazily."""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from evidence_ladder.models import backbone as BB

LIM = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8)      # same limits as the Kaggle kernels


def load_model(profile: BB.Profile, model_dir=None):
    """-> (model, tokenizer, device_name). resident: fp16 on the GPU. stream: fp16 on the CPU, config cut to the deepest captured layer (the layers beyond it are never run)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    d = str(model_dir or profile.model_dir())
    tok = AutoTokenizer.from_pretrained(d)
    cfg = AutoConfig.from_pretrained(d)
    if profile.local_device_policy == "stream":
        n = max(profile.depths)
        cfg.num_hidden_layers = n
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = list(cfg.layer_types)[:n]
        model = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float16).eval().to("cuda")
    return model, tok, torch.cuda.get_device_name(0)


def make_capturer(profile: BB.Profile, model):
    import torch
    if profile.local_device_policy == "stream":
        from capture.capture import Capturer
        return Capturer(model, profile.depths)
    from capture.capture_resident import ResidentCapturer

    class GatedCapturer(ResidentCapturer):
        """Hooks record only while capturing, so generation with the same resident model is unaffected."""
        active = False

        def _post(self, i):
            inner = super()._post(i)

            def post(mod, args, kwargs, o):
                return inner(mod, args, kwargs, o) if self.active else o
            return post

        def run(self, *a, **k):
            self.active = True
            try:
                return super().run(*a, **k)
            finally:
                self.active = False
    return GatedCapturer(model, profile.depths)


def make_capture_fn(profile: BB.Profile, model, tok, capturer=None) -> Callable[[str, Sequence[str]], Tuple[Any, Any, Dict[str, Any]]]:
    import time
    import torch
    C = capturer or make_capturer(profile, model)
    DEP = list(profile.depths)

    def capture(context_text: str, options: Sequence[str]):
        options = list(options)
        pre = tok.encode(context_text + profile.query_cue, add_special_tokens=False)
        suf = [tok.encode(profile.option_prefix + o, add_special_tokens=False) for o in options]
        info = dict(prefix_tokens=len(pre), option_tokens=[len(s) for s in suf], profile_id=profile.id)
        if not options or len(options) > LIM["max_options"] or len(pre) > LIM["max_prefix_tokens"] or any(len(s) > LIM["max_option_tokens"] or not s for s in suf) \
                or len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]:
            return None, None, dict(info, status="excluded_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        full = model.model.layers
        model.model.layers = torch.nn.ModuleList(list(full)[:max(DEP)])
        t = time.time()
        cuda = torch.cuda.is_available()
        if cuda: torch.cuda.reset_peak_memory_stats()
        try:
            cap, _ = C.packed(rec)
            hq = torch.stack([cap[d][0] for d in DEP])
            ho = torch.stack([cap[d][1] for d in DEP], 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, None, dict(info, status="skipped_oom")
        finally:
            model.model.layers = full
        hq, ho = hq.half().cpu(), ho.half().cpu()
        if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()):
            return None, None, dict(info, status="nonfinite_fp16")
        feat = hashlib.sha256(hq.contiguous().view(torch.uint8).numpy().tobytes() + ho.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return hq, ho, dict(info, status="ok", mode="packed", seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None, depth_keys=sorted(cap),
                            lm_head_calls=C.lm_calls, features_sha256=feat)
    return capture
