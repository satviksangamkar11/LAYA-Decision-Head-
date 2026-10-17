"""[EL-ARCH-1 local run] REAL-MODEL checks of the exposed Qwen3-0.6B hidden states (needs the local GPU and models/qwen3-0.6b; run once, not part of the CPU suites).
  python evidence_ladder/models/tests/backbone_gpu_tests.py <out.json>      (create-only)
Checks: shapes [3,1024] / [N,3,1024] at depths 14/19/23, fp16 finite, lm_head never run, determinism (two captures bit-identical), option-order independence (rows follow the options, cosine >= 0.9999),
packed pass == plain separate forward (the recipe's own equivalence), and that the provenance guard / LiveScorer refuse these states. Every output is labelled DIAGNOSTIC."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evidence_ladder.models import backbone as BB, laya_runtime as RT, profile_capture as PC  # noqa: E402

LABEL = "qwen3-0.6b states: DIAGNOSTIC; NOT consumable by the frozen LAYA head; data for a future adapter"
CTX = "[user]\nFix the slugify function so the tests pass.\n\n[assistant]\nI will read slug.py first."
OPTS = ['{"name":"Read","arguments":{"file_path":"slug.py"}}', '{"name":"Bash","arguments":{"command":"python -m pytest -q"}}', '{"name":"Edit","arguments":{"file_path":"slug.py","new_string":"x","old_string":"y"}}']


def main(out_path):
    P = BB.get("qwen3-0.6b")
    model, tok, dev = PC.load_model(P)
    C = PC.make_capturer(P, model)
    cap = PC.make_capture_fn(P, model, tok, C)
    res = {"label": LABEL, "device": dev, "profile": P.id, "profile_sha256": P.sha256, "checks": {}}
    hq, ho, info = cap(CTX, OPTS)
    ck = res["checks"]
    ck["shapes"] = {"hq": list(hq.shape), "ho": list(ho.shape), "dtype": str(hq.dtype), "depth_keys": info["depth_keys"], "ok": tuple(hq.shape) == (3, 1024) and tuple(ho.shape) == (3, 3, 1024) and info["depth_keys"] == [14, 19, 23]}
    ck["finite_fp16"] = bool(torch.isfinite(hq).all() and torch.isfinite(ho).all() and hq.dtype == torch.float16)
    ck["lm_head_never_run"] = {"calls": C.lm_calls, "ok": C.lm_calls == 0}
    hq2, ho2, info2 = cap(CTX, OPTS)
    ck["deterministic"] = {"bit_identical": bool(torch.equal(hq, hq2) and torch.equal(ho, ho2)), "features_sha256": info["features_sha256"], "features_sha256_repeat": info2["features_sha256"]}
    perm = [2, 0, 1]
    hq3, ho3, _ = cap(CTX, [OPTS[i] for i in perm])
    cos = min(F.cosine_similarity(ho3[k].double(), ho[perm[k]].double(), dim=-1).min().item() for k in range(3))
    ck["option_order_independence"] = {"min_cosine_rows_follow_options": cos, "max_abs": max((ho3[k].float() - ho[perm[k]].float()).abs().max().item() for k in range(3)),
                                      "query_identical_or_close": F.cosine_similarity(hq3.double(), hq.double(), dim=-1).min().item(), "ok": cos >= 0.9999}
    # packed == plain (a separate [prefix + option] sequence)
    pre = tok.encode(CTX + P.query_cue, add_special_tokens=False); suf = [tok.encode(P.option_prefix + o, add_special_tokens=False) for o in OPTS]
    rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
    full = model.model.layers; model.model.layers = torch.nn.ModuleList(list(full)[:max(P.depths)])
    try:
        capp, _ = C.packed(rec); worst = 1.0
        for i in range(len(OPTS)):
            cp = C.plain(rec, i)
            for d in P.depths:
                worst = min(worst, F.cosine_similarity(cp[d][1][0].double(), capp[d][1][i].double(), dim=-1).item(), F.cosine_similarity(cp[d][0].double(), capp[d][0].double(), dim=-1).item())
    finally:
        model.model.layers = full
    ck["packed_equals_plain"] = {"min_cosine": worst, "ok": worst >= 0.9999}
    # the guard refuses these states
    prov = RT.make_provenance(device=dev, context_text=CTX, options=OPTS, profile=P)
    msgs = {}
    for name, fn in (("check_metadata", lambda: RT.check_metadata(prov)), ("check_tensors", lambda: RT.check_tensors(hq, ho, 3, P)), ("LiveScorer", lambda: RT.LiveScorer(cap, None, dev, P))):
        try:
            fn(); msgs[name] = "NOT REFUSED"
        except RT.ProvenanceError as e:
            msgs[name] = str(e)[:160]
    ck["laya_guard_refuses"] = {"messages": msgs, "ok": all("DIAGNOSTIC" in m for m in msgs.values())}
    res["overall"] = "PASS" if ck["shapes"]["ok"] and ck["finite_fp16"] and ck["lm_head_never_run"]["ok"] and ck["deterministic"]["bit_identical"] and ck["option_order_independence"]["ok"] \
        and ck["packed_equals_plain"]["ok"] and ck["laya_guard_refuses"]["ok"] else "FAIL"
    res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2); res["versions"] = {"torch": torch.__version__}; res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"overall": res["overall"], "checks": {k: (v if not isinstance(v, dict) else v.get("ok", v.get("bit_identical"))) for k, v in ck.items()}}))
    return 0 if res["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
