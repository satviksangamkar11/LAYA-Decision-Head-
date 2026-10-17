"""[EL-ARCH-1 C1] PREPARED, NOT PUSHED, NOT RUN. Kaggle script (no Qwen, no generation; runs on the CPU or any GPU kernel) that scores the arm-E pools with the two comparison scorers on the
SAME stored features LAYA scored: the mandatory linear probe (refit from the 334 stored TRAIN pools, evidence_ladder.models.probe.refit_probe) and the random-head floor control
(evidence_ladder.models.random_head, seeds 11/12/13, nothing trained, nothing loaded except the optional TRAIN feature-scaler statistics of final-s0).
Inputs (Kaggle datasets): the laya_gen bundle (BUNDLE_MANIFEST.json bundle == evidence_ladder_laya_gen_v1, which must also ship evidence_ladder/models/{probe,random_head}.py) and the
features directory written by evidence_ladder_laya_gen_v1 (features/<task>.pt: hq [3,H], ho [N,3,H], options, context_sha256, option_sha256). No adapter, head or encoder is changed.
Outputs: /kaggle/working/probe_random_scores.json (per task: probe and random_head scores per distinct option, option_sha256, role, fingerprints), RUN_SUMMARY.json (verdict, probe
fingerprint + order hash, VAL top-1 of both scorers as informational controls). Always exits 0 with RUN_SUMMARY.json; any gate failure -> verdict GATE_FAILED and no scores are written."""
import glob, hashlib, json, os, sys, time, traceback

OUT = "/kaggle/working"; T0 = time.time()
RS = {"kernel": "evidence_ladder_probe_random_head_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), **kw)
    with open(OUT + "/RUN_SUMMARY.json", "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)
def sha_file(p):
    x = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
    return x.hexdigest()

try:
    import torch
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_laya_gen_v1"), None)
    if not mp: finish("FAILED_NO_BUNDLE")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or sha_file(os.path.join(BUNDLE, f)) != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad)
    sys.path.insert(0, BUNDLE)
    from evidence_ladder.models import laya_forward as LF, probe as PR, random_head as RH
    G = lambda p: os.path.join(BUNDLE, p)
    # ---- stored TRAIN pools (v2 + v3 captures, labelled by the v3 label set) and the VAL pools, exactly as the laya_gen parity gate loads them
    FE = {n: {r["decision_id"]: r for r in torch.load(G("gate/feat_%s_B.pt" % n), map_location="cpu", weights_only=False)["records"]} for n in ("v2", "v3", "val")}
    RQ = {n: {d["decision_id"]: d for d in (json.loads(l) for l in open(G("gate/req_%s.jsonl" % n), encoding="utf-8"))} for n in ("v2", "v3", "val")}
    LAB = json.load(open(G("gate/labels.json"), encoding="utf-8")); REC = json.load(open(G("gate/recorded.json"), encoding="utf-8"))
    def row(did, sr, rq, lab):
        ix = {o: i for i, o in enumerate(rq["options"])}
        return dict(decision_id=did, hq=sr["hq"], ho=torch.stack([sr["ho"][ix[o]] for o in lab["options"]]), ref=lab["reference_index"], pos=torch.tensor([s == "positive" for s in lab["status"]]))
    train_rows = [row(d, sr, RQ[n][d], LAB["v3"][d]) for n in ("v2", "v3") for d, sr in FE[n].items() if d in LAB["v3"]]
    val_rows = [row(d, FE["val"][d], RQ["val"][d], LAB["val"][d]) for d in REC["decision_id"]]
    fit = PR.refit_probe(train_rows, expect_pools=PR.TRAIN_POOLS)      # raises unless exactly 334 pools
    RS["probe"] = {k: fit[k] for k in ("fingerprint", "order_sha256", "n_pools", "n_options", "C", "seed", "role")}
    probe_hit = [LF.hit(LF.probe_decision(fit["probe"], r["ho"]), r["ref"]) for r in val_rows]
    differing = [d for d, h, rec in zip(REC["decision_id"], probe_hit, REC["probe_real"]) if abs(h - rec) > 1e-9]
    RS["probe_gate"] = dict(val_top1=round(sum(probe_hit) / len(probe_hit), 4), recorded_top1=REC["probe_top1"], pools_differing=len(differing), differing=differing)
    if differing: finish("GATE_FAILED", note="the refit probe does not reproduce the recorded VAL probe hits")
    scaler_src = G("models/final-s0.pt")
    heads = {"identity_scaler": RH.build_random_heads(), "train_scaler": RH.build_random_heads(scaler_from=scaler_src)}
    RS["random_head"] = {"role": RH.ROLE, "seeds": list(RH.RANDOM_SEEDS), "state_sha256": {v: {n: RH.state_sha256(h) for n, h in hs.items()} for v, hs in heads.items()},
                         "old_control_not_this_one": RH.OLD_CONTROL_LABEL,
                         "val_top1": {v: round(sum(LF.hit(RH.random_head_scores(hs, r["hq"], r["ho"])["score"], r["ref"]) for r in val_rows) / len(val_rows), 4) for v, hs in heads.items()}}
    # ---- score the arm-E pools from the stored laya_gen features
    feats = sorted(glob.glob("/kaggle/input/**/features/*.pt", recursive=True)); out = {}
    for fp in feats:
        d = torch.load(fp, map_location="cpu", weights_only=False); hq, ho = d["hq"], d["ho"]
        out[d["decision_id"]] = dict(features_file=os.path.basename(fp), features_sha256=sha_file(fp), context_sha256=d["context_sha256"], option_sha256=d["option_sha256"],
                                     probe=PR.probe_scores(fit, ho), random_head={v: RH.random_head_scores(hs, hq, ho) for v, hs in heads.items()},
                                     note="scores are per DISTINCT option in the option order of the features file; pool-dependent logits; comparison only, never a selector")
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "/probe_random_scores.json", "w", encoding="utf-8") as f: json.dump(out, f, indent=1)
    RS["pools_scored"] = len(out); finish("OK" if out else "OK_NO_POOLS")
except SystemExit: raise
except BaseException:
    RS["traceback"] = traceback.format_exc(); finish("FAILED_EXCEPTION")
