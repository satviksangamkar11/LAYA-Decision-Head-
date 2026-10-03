"""Verification of the Kaggle T4 Phase-2 capture after download (draft amendment e: e1 device lock, e2 completeness, e3 cross-session reproducibility, e4 cross-device cosine).
    PYTHONPATH=. .venv/Scripts/python.exe tools/verify_phase2_capture.py [--t4 data/v3/capture_t4] [--ref3050 data/v3/capture] [--smoke tmp/t4smoke/out/smoke_speed] [--kaggle_manifest PATH]
Create-only: writes results/raw/phase2-capture-verify-<timestamp>.json and refuses to overwrite. No threshold is applied beyond the already registered ones
(plain_cos_min 0.999 is used only as a reference line for the informational cross-device cosine; the repeat check is bit identity)."""
import os, sys, json, glob, hashlib, argparse, datetime
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--t4", default="data/v3/capture_t4")
ap.add_argument("--ref3050", default="data/v3/capture")
ap.add_argument("--smoke", default="tmp/t4smoke/out/smoke_speed")
ap.add_argument("--kaggle_manifest", default=None)
args = ap.parse_args()
out = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "roles": {}}
km = json.load(open(args.kaggle_manifest)) if args.kaggle_manifest and os.path.exists(args.kaggle_manifest) else None
sidecar_keys = ("device", "dtype", "resident", "torch", "transformers", "revision", "gate_every", "plain_every", "depths")
for role in ("train", "cal"):
    ids = json.load(open(f"data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"]
    expect = {"%05d-%s.pt" % (n, hashlib.sha256(d.encode()).hexdigest()[:8]): d for n, d in enumerate(ids)}
    d0 = os.path.join(args.t4, role)
    have = {os.path.basename(f) for f in glob.glob(d0 + "/*.pt")}
    r = dict(manifest=len(ids), files=len(have), missing=sorted(set(expect) - have)[:5], n_missing=len(set(expect) - have), extra=sorted(have - set(expect))[:5], n_extra=len(have - set(expect)))
    side = [json.load(open(f)) for f in sorted(glob.glob(d0 + "/capture-device-*.json"))]
    r["sidecars"] = len(side)
    r["sidecars_identical_on_device_lock_fields"] = len({json.dumps({k: s[k] for k in sidecar_keys}, sort_keys=True) for s in side}) == 1
    r["sidecar_lock"] = {k: side[0][k] for k in sidecar_keys} if side else None
    bad_load, nonfinite, big, wrong_id, hashes = [], 0, 0, 0, {}
    cos = {}
    below, tot, pool_mismatch, ref_compared = 0, 0, 0, 0
    for fn in sorted(have & set(expect)):
        p = os.path.join(d0, fn)
        hashes[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        try:
            a = torch.load(p)
        except Exception as e:
            bad_load.append((fn, repr(e)[:80])); continue
        if a["decision_id"] != expect[fn] or a["role"] != role:
            wrong_id += 1
        q, c = a["query"].float(), a["cands"].float()
        if not (torch.isfinite(q).all() and torch.isfinite(c).all()):
            nonfinite += 1
        if max(q.abs().max().item(), c.abs().max().item()) >= 60000:
            big += 1
        rp = os.path.join(args.ref3050, role, fn)
        if os.path.exists(rp):
            b = torch.load(rp)
            ref_compared += 1
            if a["pool_hash"] != b["pool_hash"] or a["state_hash"] != b["state_hash"] or a["n_prefix_tokens"] != b["n_prefix_tokens"]:
                pool_mismatch += 1
            for kind, x, y in (("query", a["query"].double(), b["query"].double()), ("cands", a["cands"].double(), b["cands"].double())):
                cs = F.cosine_similarity(x, y, dim=-1)
                for dj, dep in enumerate(a["depths"]):
                    cd = cs[dj] if kind == "query" else cs[:, dj]
                    w = cos.setdefault(f"{kind}@{dep}", [1.0, 0.0, 0])
                    w[0] = min(w[0], cd.min().item()); w[1] += cd.mean().item(); w[2] += 1
                    if kind == "cands":
                        below += int((cd < 0.999).sum()); tot += cd.numel()
    r.update(unreadable=bad_load, nonfinite_files=nonfinite, files_with_abs_value_ge_60000=big, wrong_decision_id_or_role=wrong_id)
    r["e4_cross_device_vs_3050"] = dict(files_compared=ref_compared, pool_state_prefix_mismatches=pool_mismatch,
                                         per_depth_min_mean={k: [round(v[0], 6), round(v[1] / v[2], 6)] for k, v in sorted(cos.items())},
                                         option_vectors_below_0p999=f"{below} of {tot}", note="informational; no cross-device threshold is registered")
    if km:
        rec = {e["file"]: e["sha256"] for e in km["files"][role]}
        r["kaggle_manifest_sha256"] = dict(entries=len(rec), mismatches=sum(1 for k, v in hashes.items() if rec.get(k) != v), absent=len(set(hashes) - set(rec)))
    else:
        r["kaggle_manifest_sha256"] = "manifest not provided"
    r["sha256_of_all_files"] = hashlib.sha256("".join(f"{k}:{hashes[k]}\n" for k in sorted(hashes)).encode()).hexdigest()
    out["roles"][role] = r
# e3: smoke files (notebook v5, same decisions, same device class) vs Phase-2 files
e3 = dict(compared=0, bit_identical=0, differing=[])
for f in sorted(glob.glob(os.path.join(args.smoke, "train", "*.pt"))):
    g = os.path.join(args.t4, "train", os.path.basename(f))
    if not os.path.exists(g):
        continue
    a, b = torch.load(f), torch.load(g)
    e3["compared"] += 1
    if torch.equal(a["query"], b["query"]) and torch.equal(a["cands"], b["cands"]):
        e3["bit_identical"] += 1
    else:
        qd = (a["query"].float() - b["query"].float()).abs().max().item(); cd = (a["cands"].float() - b["cands"].float()).abs().max().item()
        cq = F.cosine_similarity(a["cands"].double(), b["cands"].double(), dim=-1).min().item()
        e3["differing"].append(dict(file=os.path.basename(f), max_abs_query=qd, max_abs_cands=cd, min_cos_cands=round(cq, 6)))
out["e3_cross_session_reproducibility"] = e3
p = f"results/raw/phase2-capture-verify-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
assert not os.path.exists(p)
json.dump(out, open(p, "w"), indent=1)
print(json.dumps(out, indent=1)[:6000]); print("wrote", p)
