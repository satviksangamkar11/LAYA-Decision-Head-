"""Diff-focused capture v2 for the EDITED-ONLY control (user GO, chat 2026-10-13). Same code as diff_capture_v1; differences: reads requests_edited.jsonl,
and the reproduction check compares, for every option text that was also captured in the earlier 19 pools (same context), the whole-option and
query vectors with the earlier features (options are captured independently, so they must match). Edit sites are anchored on the first EDITED
candidate (the original is not in these pools). Kaggle T4. No labels are read.

v1 description follows.

Same model revision, tokenisation, prefixes, layers [18, 24, 30] and forward passes as the existing capture (kaggle level12 worker, Level 1): packed
when the combined batch fits max_total_tokens, otherwise one forward pass per option with the individual guard (prefix + one option must fit).
In the SAME forward pass, besides the whole-option mean (the existing feature, used only for validation), it records for every option the mean over
its edit-site tokens +- 8 and +- 32 tokens. Edit sites are label-free: each option's token ids are aligned to the pool's first option (hash order)
with difflib; the pool's sites are every reference position where any option differs; every option is represented over ALL sites of its pool.
Also stored for the lexical baseline: each option's own token ids at the site positions (no window).

Inputs (mounted): laya-mutation-score-v2 (requests_level1, level12_config, capture code), laya-mutation-features-v2 (existing B / A features).
Output /kaggle/working/diff_features.pt and diff_meta.json; exits 0 only after every validation check passed.
"""
import difflib
import glob
import gzip
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

IN, OUT = "/kaggle/input", "/kaggle/working"
WINDOWS = (8, 32)
find = lambda pat: sorted(glob.glob(os.path.join(IN, "**", pat), recursive=True))


def load_requests():
    p = find("requests_edited.jsonl*")
    assert len(p) == 1, p
    return [json.loads(l) for l in (gzip.open if p[0].endswith(".gz") else open)(p[0], "rt", encoding="utf-8")]


def sites_for_pool(suffixes):
    """-> per option, sorted positions (relative to the option start) of its tokens at the pool's edit sites. Label-free."""
    ref = suffixes[0]
    ops = [difflib.SequenceMatcher(None, ref, s, autojunk=False).get_opcodes() for s in suffixes]
    site_ref = set()
    for o in ops:
        for tag, i1, i2, j1, j2 in o:
            if tag != "equal":
                site_ref.update(range(i1, i2) if i2 > i1 else [max(i1 - 1, 0), min(i1, len(ref) - 1)])
    out = []
    for s, o in zip(suffixes, ops):
        pos = set()
        for tag, i1, i2, j1, j2 in o:
            hit = [p for p in range(i1, i2) if p in site_ref] if i2 > i1 else ([i1] if (i1 in site_ref or i1 - 1 in site_ref) else [])
            if not hit:
                continue
            if tag == "equal":
                pos.update(j1 + (p - i1) for p in hit)
            elif j2 > j1:
                pos.update(range(j1, j2))
            else:   # deletion in this option: its neighbours stand for the site
                pos.update(p for p in (j1 - 1, j1) if 0 <= p < len(s))
        out.append(sorted(pos))
    return out


def windowed(pos, n, w):
    s = set()
    for p in pos:
        s.update(range(max(0, p - w), min(n, p + w + 1)))
    return sorted(s)


def main():
    t0 = time.time()
    code = os.path.dirname(find("level12_config.json")[0])
    cfg = json.load(open(os.path.join(code, "level12_config.json")))
    proto, lim = cfg["protocol"], cfg["limits"]
    dep = proto["depths"]
    pkg = os.path.join(OUT, "pkg", "capture")
    os.makedirs(pkg, exist_ok=True)
    for n in ("capture.py", "capture_resident.py"):
        shutil.copy(os.path.join(code, n), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, os.path.join(OUT, "pkg"))
    from capture.capture_resident import ResidentCapturer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    qd = snapshot_download("Qwen/Qwen3-4B-Thinking-2507", revision=proto["revision"], local_dir="/tmp/qwen")
    tok = AutoTokenizer.from_pretrained(qd)
    model = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float16).eval()
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(dep)])
    C = ResidentCapturer(model.to("cuda"), dep)

    extra = {}   # depth -> {feature name: [n_options, H]} filled by the extra hook in the same forward

    def hook(i):
        def post(mod, args, kwargs, o):
            if i in dep and C.extra_plan is not None:
                x = (o[0] if isinstance(o, tuple) else o)[0]
                extra[i] = {name: torch.stack([x[idx.to(x.device)].float().mean(0) for idx in per_opt]).cpu() for name, per_opt in C.extra_plan.items()}
            return o
        return post
    C.extra_plan = None
    for i, l in enumerate(model.model.layers, 1):
        l.register_forward_hook(hook(i), with_kwargs=True)

    reqs = load_requests()
    ref_feats = {}   # (variant, decision_id, option text) -> (hq, ho) from the earlier 19-pool capture
    q = find("requests_level1.jsonl*")
    old = {r["decision_id"]: r for r in (json.loads(l) for l in (gzip.open if q[0].endswith(".gz") else open)(q[0], "rt", encoding="utf-8"))}
    for v in "BA":
        for x in torch.load(find("level1/%s.pt" % v)[0], map_location="cpu", weights_only=False)["records"]:
            for j, o in enumerate(old[x["decision_id"]]["options"]):
                ref_feats[(v, x["decision_id"], o)] = (x["hq"], x["ho"][j], x["context_sha256"])
    meta = dict(protocol="diff_capture_v1", depths=dep, windows=WINDOWS, revision=proto["revision"], device=torch.cuda.get_device_name(0), torch=torch.__version__,
                checks={}, rows=[])
    recs = {"B": [], "A": []}

    def capture(r, v):
        prefix = (r["context_text"] if v == "B" else cfg["neutral_prefix"]) + proto["query_cue"]
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = [tok.encode(proto["option_prefix"] + o, add_special_tokens=False) for o in r["options"]]
        assert all(len(pre) + len(s) <= lim["max_total_tokens"] and len(s) <= lim["max_option_tokens"] for s in suf), "individual guard"
        assert len(pre) <= lim["max_prefix_tokens"]
        sites = sites_for_pool(suf)
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        packed = len(pre) + sum(map(len, suf)) <= lim["max_total_tokens"]
        rel = {"whole": [list(range(len(s))) for s in suf]}
        for w in WINDOWS:
            rel["w%d" % w] = [windowed(p, len(s), w) for p, s in zip(sites, suf)]
        rel["site"] = sites
        t = time.time()
        if packed:
            P, starts, o = len(pre), [], len(pre)
            for s in suf:
                starts.append(o)
                o += len(s)
            C.extra_plan = {k: [torch.tensor([starts[j] + p for p in idx[j]]) for j in range(len(suf))] for k, idx in rel.items()}
            cap, _ = C.packed(rec)
            hq = torch.stack([cap[d][0] for d in dep])
            f = {k: torch.stack([extra[d][k] for d in dep], 1) for k in rel}
            ho_existing = torch.stack([cap[d][1] for d in dep], 1)
        else:
            per = []
            for j in range(len(suf)):
                P = len(pre)
                C.extra_plan = {k: [torch.tensor([P + p for p in idx[j]])] for k, idx in rel.items()}
                cap = C.plain(rec, j)
                per.append(({k: torch.stack([extra[d][k][0] for d in dep]) for k in rel}, cap))
            hq = torch.stack([per[0][1][d][0] for d in dep])
            f = {k: torch.stack([p[0][k] for p in per]) for k in rel}
            ho_existing = torch.stack([torch.stack([p[1][d][1][0] for d in dep]) for p in per])
        C.extra_plan = None
        out = dict(decision_id=r["decision_id"], variant=v, hq=hq.half(), **{"ho_" + k: f[k].half() for k in rel if k != "site"},
                   site_token_ids=[[s[p] for p in idx] for s, idx in zip(suf, sites)], n_sites_per_option=[len(x) for x in sites],
                   option_sha256=r["option_sha256"], order_sha256=r["order_sha256"], context_sha256=r["context_sha256"])
        info = dict(decision_id=r["decision_id"], variant=v, mode="packed" if packed else "plain_per_option", prefix_tokens=len(pre),
                    option_tokens=[len(s) for s in suf], sites=[len(x) for x in sites], seconds=round(time.time() - t, 2))
        return out, info, ho_existing

    # ---- validation (any failure stops before the bulk capture is written)
    checks = meta["checks"]
    worst, ok_sha, ok_span, ok_fin, matched = 1.0, True, True, True, 0
    for v in "BA":
        for r in reqs:
            out, info, ho_existing = capture(r, v)
            c = 1.0
            for j, o in enumerate(r["options"]):
                ref = ref_feats.get((v, r["decision_id"], o))
                if ref is None:
                    continue
                matched += 1
                ok_sha &= ref[2] == r["context_sha256"]
                c = min(c, F.cosine_similarity(out["ho_whole"][j].double().flatten(), ref[1].double().flatten(), 0).item(),
                        F.cosine_similarity(out["hq"].double().flatten(), ref[0].double().flatten(), 0).item())
            worst = min(worst, c)
            ok_span &= all(n > 0 for n in out["n_sites_per_option"]) and all(len(x) > 0 for x in out["site_token_ids"])
            ok_fin &= all(torch.isfinite(t.float()).all().item() for k, t in out.items() if k.startswith("ho_") or k == "hq")
            info["whole_vs_existing_cos"] = round(c, 6)
            recs[v].append(out)
            meta["rows"].append(info)
            print("CAP", json.dumps(info), flush=True)
    rep = []
    for r in reqs[:2]:
        a, _, _ = capture(r, "B")
        b = next(x for x in recs["B"] if x["decision_id"] == r["decision_id"])
        rep.append(all(torch.equal(a[k], b[k]) for k in a if k.startswith("ho_") or k == "hq"))
    checks.update(options_matched_to_earlier_capture=matched, hashes_identical=bool(ok_sha) and matched > 0, whole_option_cos_min=round(worst, 6), whole_option_reproduces=worst >= 0.999, spans_nonempty=bool(ok_span),
                  finite=bool(ok_fin), repeat_identical=all(rep), plain_pools=sorted({x["decision_id"] for x in meta["rows"] if x["mode"] == "plain_per_option"}),
                  key_file_mounted=bool(find("*PRIVATE_KEY*") or find("labels*.json")))
    meta["protocol"] = "diff_capture_v2_edited_only"
    passed = checks["hashes_identical"] and checks["whole_option_reproduces"] and checks["spans_nonempty"] and checks["finite"] and checks["repeat_identical"] \
        and not checks["key_file_mounted"]
    checks["all_passed"] = bool(passed)
    meta["seconds"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(OUT, "diff_meta.json"), "w"), indent=1)
    print("CHECKS", json.dumps(checks), flush=True)
    if not passed:
        print("STOP: a validation check failed; no features written", flush=True)
        return
    torch.save(dict(records=recs, meta=meta), os.path.join(OUT, "diff_features.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
