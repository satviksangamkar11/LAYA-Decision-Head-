"""[LAYA-INT-1 · Aim: integration, not performance] Integration kernel v1 (Kaggle Tesla T4, fp16, internet OFF): T1 real capture, T2 training parity, T3 adapter inputs,
T4 LAYA forward, T7 provenance rejection (GPU part). T5 (controller) and the pure-logic part of T7 run in the test kernel; T6 (loop) is staged separately.
No training, no generation, no candidate code executed. Thresholds: configs/laya_head_gate_v1.toml (T2, T4 reproduction) and configs/laya_int1_thresholds_v1.toml, both written before any run.
Receipts (create-only): /kaggle/working/receipts/T*.json + README.md; scores for T5: /kaggle/working/T4_scores.json. Always exits 0 with RUN_SUMMARY.json."""
import glob, hashlib, json, os, random, sys, time, traceback

T0 = time.time(); OUT = "/kaggle/working"; REC = OUT + "/receipts"; os.makedirs(REC, exist_ok=True)
TAG = "[LAYA-INT-1 · Aim: integration, not performance]"
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
LIM = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8)
DEP = [18, 24, 30]; CUE = "\n\nNext action:"; OPT_PREFIX = " "
RS = {"kernel": "evidence_ladder_laya_integration_v1", "tag": TAG, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}}
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def save():
    with open(OUT + "/RUN_STATE.json", "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
def finish(verdict, **kw):
    RS["verdict"] = verdict; RS["total_seconds"] = round(time.time() - T0, 1); RS.update(kw); save()
    with open(OUT + "/RUN_SUMMARY.json", "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)
def sha_file(p):
    x = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
    return x.hexdigest()
def receipt(name, passed, body):
    body = dict(body, test=name, passed=bool(passed), tag=TAG, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = "%s/%s.json" % (REC, name)
    assert not os.path.exists(path), "create-only: " + path
    with open(path, "w", encoding="utf-8") as f: json.dump(body, f, indent=1, default=str)
    RS["tests"][name] = bool(passed); save(); log("RECEIPT", name, "PASS" if passed else "FAIL")

try:
    import torch, transformers
    import torch.nn.functional as F
    RS["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available(): finish("FAILED_NO_CUDA")
    DEVNAME = torch.cuda.get_device_name(0); RS["gpu"] = DEVNAME; log("GPU", DEVNAME, RS["versions"])
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_laya_int1_v1"), None)
    if not mp: finish("FAILED_NO_BUNDLE")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or sha_file(os.path.join(BUNDLE, f)) != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad)
    sys.path.insert(0, BUNDLE)
    import tomllib
    from evidence_ladder.models import laya_forward as LF, laya_runtime as RT
    G = lambda p: os.path.join(BUNDLE, p)
    TH = tomllib.load(open(G("configs/laya_head_gate_v1.toml"), "rb")); TI = tomllib.load(open(G("configs/laya_int1_thresholds_v1.toml"), "rb"))
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs: finish("FAILED_NO_MODEL")
    md = dirs[0]; ok = {f: sha_file(os.path.join(md, f)) == h for f, h in QWEN_SHA.items()}; RS["qwen_sha256_ok"] = ok
    if not all(ok.values()): finish("FAILED_MODEL_SHA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float16).eval().to("cuda")     # exactly the TRAIN capture loader call (kaggle/level12_t4.py)
    from capture.capture_resident import ResidentCapturer
    import transformers.integrations.sdpa_attention as SDPA
    class GatedCapturer(ResidentCapturer):      # the training capturer; hooks active only inside run(); the sm_75 GQA switch scoped to capture
        active = False
        def _post(self, i):
            inner = super()._post(i)
            def post(mod, args, kwargs, o): return inner(mod, args, kwargs, o) if self.active else o
            return post
        def run(self, *a, **k):
            orig = SDPA.use_gqa_in_sdpa; self.active = True
            if self.no_gqa: SDPA.use_gqa_in_sdpa = lambda *x, **y: False
            try: return super().run(*a, **k)
            finally: self.active = False; SDPA.use_gqa_in_sdpa = orig
    C = GatedCapturer(model, DEP, no_gqa=False); C.no_gqa = torch.cuda.get_device_capability(0)[0] < 8

    def capture(context_text, options, plain_check=False):
        """THE runtime capture: (hq [3,H] fp16 cpu, ho [N,3,H] fp16 cpu, info). Training recipe; nothing clipped."""
        pre = tok.encode(context_text + CUE, add_special_tokens=False); suf = [tok.encode(OPT_PREFIX + o, add_special_tokens=False) for o in options]
        info = dict(prefix_tokens=len(pre), option_tokens=[len(s) for s in suf])
        if not options or len(options) > LIM["max_options"]: return None, None, dict(info, status="excluded_options_count")
        if len(pre) > LIM["max_prefix_tokens"]: return None, None, dict(info, status="excluded_prefix_over_limit")
        if any(len(s) > LIM["max_option_tokens"] or not s for s in suf): return None, None, dict(info, status="excluded_option_over_limit")
        if len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]: return None, None, dict(info, status="excluded_total_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        full = model.model.layers; model.model.layers = torch.nn.ModuleList(list(full)[:max(DEP)])
        t = time.time(); mode = "packed"; torch.cuda.reset_peak_memory_stats()
        try:
            try:
                cap, _ = C.packed(rec); keys = sorted(cap); hq = torch.stack([cap[d][0] for d in DEP]); ho = torch.stack([cap[d][1] for d in DEP], 1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); mode = "plain_per_option"
                parts = [C.plain(rec, i) for i in range(len(suf))]; keys = sorted(parts[0])
                hq = torch.stack([parts[0][d][0] for d in DEP]); ho = torch.stack([torch.stack([p[d][1][0] for d in DEP]) for p in parts])
            pl = None
            if plain_check:   # independent evidence for the pooling semantics: a plain [prefix + option_0] forward must give the same hq and ho[0]
                p0 = C.plain(rec, 0); pl = min(min(F.cosine_similarity(p0[d][0].double(), cap[d][0].double(), 0).item() for d in DEP),
                                               min(F.cosine_similarity(p0[d][1][0].double(), cap[d][1][0].double(), 0).item() for d in DEP))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return None, None, dict(info, status="skipped_oom")
        finally: model.model.layers = full
        hq, ho = hq.half().cpu(), ho.half().cpu()
        ok_ = bool(torch.isfinite(hq).all() and torch.isfinite(ho).all())
        return (hq, ho, dict(info, status="ok" if ok_ else "nonfinite_fp16", mode=mode, seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                             depth_keys=keys, plain_vs_packed_cos_min=pl, lm_head_calls=C.lm_calls)) if ok_ else (None, None, dict(info, status="nonfinite_fp16"))

    # ---- data
    PRE = G("models/pretrained_31.safetensors"); ADP = {n: G("models/%s.pt" % n) for n in LF.SEEDS}
    FE = {n: {r["decision_id"]: r for r in torch.load(G("gate/feat_%s_B.pt" % n), map_location="cpu", weights_only=False)["records"]} for n in ("v2", "v3", "val")}
    RQ = {n: {d["decision_id"]: d for d in (json.loads(l) for l in open(G("gate/req_%s.jsonl" % n), encoding="utf-8"))} for n in ("v2", "v3", "val")}
    LAB = json.load(open(G("gate/labels.json"), encoding="utf-8")); REC_ = json.load(open(G("gate/recorded.json"), encoding="utf-8"))
    def find(did): return next(n for n in ("v2", "v3", "val") if did in FE[n])
    def row(did, hq, ho, req, lab):
        ix = {o: i for i, o in enumerate(req["options"])}
        return dict(decision_id=did, hq=hq, ho=torch.stack([ho[ix[o]] for o in lab["options"]]), ref=lab["reference_index"], pos=torch.tensor([s == "positive" for s in lab["status"]]),
                    conf=torch.zeros(len(lab["status"]), dtype=torch.bool), trajectory=lab["trajectory"], options=lab["options"], split="val")
    VAL_IDS = list(REC_["decision_id"]); val_rows = [row(d, FE["val"][d]["hq"], FE["val"][d]["ho"], RQ["val"][d], LAB["val"][d]) for d in VAL_IDS]

    # ================================================================== T3 part 1: weights load (hashes + counts) and heads through BOTH paths
    from decision_head import fable_features as ff, fable_train as ft
    heads = {n: LF.build_head(PRE, ADP[n]) for n in LF.SEEDS}
    train_heads = {}
    for n in LF.SEEDS:
        st = torch.load(ADP[n], map_location="cpu", weights_only=False)
        h, mapped = ft.build_head(ff.PROTOCOL, PRE, seed=st["seed"]); assert not h.load_state_dict(st["state"], strict=False).unexpected_keys; train_heads[n] = h.eval()
    adapter_params = {n: sum(p.numel() for k, p in heads[n].named_parameters() if ft.group_of(k) == "adapter") for n in LF.SEEDS}
    pre_params = sum(p.numel() for k, p in heads["final-s0"].named_parameters() if ft.group_of(k) == "pretrained")
    same_state = {n: all(torch.equal(heads[n].state_dict()[k], train_heads[n].state_dict()[k]) for k in heads[n].state_dict()) for n in LF.SEEDS}
    wt = {"adapter_sha256_bundle": {n: sha_file(ADP[n]) for n in LF.SEEDS}, "adapter_sha256_manifest": man["adapter_sha256"], "pretrained_source_sha256": man["pretrained_source_sha256"],
          "adapter_trainable_params": adapter_params, "pretrained_tensor_params": pre_params,
          "expected_adapter_params": 5246982, "scorer_head_state_equals_training_build_head": same_state}
    wt_ok = all(wt["adapter_sha256_bundle"][n] == wt["adapter_sha256_manifest"][n] for n in LF.SEEDS) and all(v == 5246982 for v in adapter_params.values()) and all(same_state.values())
    log("WEIGHTS", json.dumps(wt, default=str)[:400])

    # ================================================================== T1 real capture + T2 training parity (recapture the fixed sample)
    ids = list(TH["sample"]["train_ids"]) + sorted(VAL_IDS); A_rows, recap, t1_first = [], {}, None
    for k, did in enumerate(ids):
        n = find(did); sr = FE[n][did]; rq = RQ[n][did]
        hq, ho, info = capture(rq["context_text"], rq["options"], plain_check=(k < 3))
        if hq is None: A_rows.append(dict(decision_id=did, status=info["status"])); continue
        if t1_first is None: t1_first = dict(info, decision_id=did)
        recap[did] = (hq, ho)
        cmp_ = RT.compare_features((hq, ho), (sr["hq"], sr["ho"]), {"min_cosine": TH["a_feature_recapture"]["min_cosine"], "max_abs_diff": TH["a_feature_recapture"]["max_abs_diff"]})
        A_rows.append(dict(decision_id=did, set=n, status="ok", cos_min=round(cmp_["cos_min"], 7), max_abs_diff=cmp_["max_abs_diff"], passed=cmp_["passed"], seconds=info["seconds"], mode=info["mode"],
                           prefix_tokens=info["prefix_tokens"], identical=bool(torch.equal(hq, sr["hq"]) and torch.equal(ho, sr["ho"])), plain_vs_packed_cos_min=info["plain_vs_packed_cos_min"]))
    okrows = [r for r in A_rows if r["status"] == "ok"]
    t1 = TI["t1_real_capture"]
    h0, o0 = recap[ids[0]]
    t1_checks = {"device_ok": DEVNAME.startswith(t1["device_prefix"]), "dtype": str(h0.dtype), "hq_shape": list(h0.shape), "ho_shape": list(o0.shape), "depth_keys_captured": t1_first["depth_keys"],
                 "lm_head_calls": t1_first["lm_head_calls"], "all_finite": len(okrows) == len(ids),
                 "plain_vs_packed_cos_min_first3": [r["plain_vs_packed_cos_min"] for r in okrows[:3]]}
    t1_ok = (t1_checks["device_ok"] and t1_checks["dtype"] == "torch.float16" and t1_checks["hq_shape"] == t1["hq_shape"] and list(o0.shape[1:]) == t1["ho_shape_tail"]
             and t1_checks["depth_keys_captured"] == DEP and t1_checks["lm_head_calls"] == 0 and t1_checks["all_finite"] and all((x or 0) >= 0.999 for x in t1_checks["plain_vs_packed_cos_min_first3"]))
    receipt("T1_real_capture", t1_ok, {"device": DEVNAME, "versions": RS["versions"], "qwen_repo": LF.__dict__.get("QWEN_REPO", "Qwen/Qwen3-4B-Thinking-2507"), "qwen_revision": RT.QWEN_REVISION,
            "qwen_sha256_ok": ok, "checks": t1_checks, "timing_seconds_per_decision": {"median": sorted(r["seconds"] for r in okrows)[len(okrows) // 2], "max": max(r["seconds"] for r in okrows)},
            "peak_gib_first": t1_first["peak_gib"], "first_decision": t1_first, "thresholds": t1,
            "capture_provenance": RT.make_provenance(device=DEVNAME, context_text=RQ[find(ids[0])][ids[0]]["context_text"], options=RQ[find(ids[0])][ids[0]]["options"])})
    a = TH["a_feature_recapture"]
    t2_ok = len(A_rows) == len(ids) and all(r["status"] == "ok" and r["passed"] for r in A_rows)
    receipt("T2_training_parity", t2_ok, {"thresholds_file": "configs/laya_head_gate_v1.toml [a_feature_recapture]", "thresholds": a, "device": DEVNAME, "versions": RS["versions"], "n": len(A_rows),
            "worst_cos": min([r["cos_min"] for r in okrows] or [None]), "worst_max_abs": max([r["max_abs_diff"] for r in okrows] or [None]), "identical_records": sum(r["identical"] for r in okrows), "rows": A_rows,
            "reference": "stored fp16 features of the 12 TRAIN pools of local-capture-parity-20261012T110722Z and the 53 canonical-VAL pools (Tesla T4 fp16, torch 2.11.0+cu128, transformers 5.16.1 per their meta.json)",
            "capture_provenance": "live capture by this kernel; code hashes " + json.dumps(RT.CAPTURE_CODE_SHA256)})

    # ================================================================== T3 adapter inputs (scorer path == training path)
    t3 = TI["t3_adapter_inputs"]; per_pool = []; worst = 0.0; swap_worst = 0.0; shape_ok = True; order_ok = True
    for r in val_rows:
        n_ = len(r["pos"])
        hq_b, ho_b, mask_b, pos_b, conf_b, ref_b = ft.collate([r])        # the TRAINING collate
        shape_ok &= tuple(hq_b.shape) == (1, 3, 2560) and tuple(ho_b.shape) == (1, n_, 3, 2560) and bool(mask_b.all()) and tuple(mask_b.shape) == (1, n_)
        order_ok &= bool(torch.equal(ho_b[0], r["ho"].float())) and bool(torch.equal(hq_b[0], r["hq"].float()))
        for nme in LF.SEEDS:
            mine = LF.head_logits(heads[nme], r["hq"], r["ho"])
            lg = ft.logits_of(train_heads[nme], [r])[0][0][0, :n_].tolist()           # the TRAINING logits_of
            worst = max(worst, max(abs(a_ - b_) for a_, b_ in zip(mine, lg)))
        if n_ >= 2:    # option i's logit follows option i's pooled vector
            sw = list(range(n_)); sw[0], sw[1] = sw[1], sw[0]
            m1 = LF.head_logits(heads["final-s0"], r["hq"], r["ho"]); m2 = LF.head_logits(heads["final-s0"], r["hq"], r["ho"][sw])
            swap_worst = max(swap_worst, max(abs(m2[i] - m1[sw[i]]) for i in range(n_)))
    t3_ok = wt_ok and shape_ok and order_ok and worst <= t3["max_abs_logit_diff"] and swap_worst <= t3["order_swap_max_abs_diff"]
    receipt("T3_adapter", t3_ok, {"thresholds": t3, "weights": wt, "weights_ok": wt_ok, "collate_shapes_ok": shape_ok, "collate_equals_scorer_tensors": order_ok, "pools": len(val_rows),
            "max_abs_logit_diff_scorer_vs_training_path": worst, "order_swap_max_abs_diff": swap_worst,
            "assertions": "hq [1,3,2560] = last-prefix-token state at depths 18/24/30; ho [1,N,3,2560] = per-option mean states, ho[i] belongs to options[i]; mask all True; prim=0 (choice); same tensors enter ft.logits_of and LF.head_logits",
            "capture_provenance": "stored VAL features: Tesla T4 fp16 captures of results/raw/val-capture-run-20261012T111837Z (not re-captured in this test; see T2)"})

    # ================================================================== T4 LAYA: valid scores, recorded reproduction, permutation stability, controls
    def hit_mean(picks_by_seed, ref_by): return [sum(float(picks_by_seed[n][d] == ref_by[d]) for n in LF.SEEDS) / 3 for d in VAL_IDS]
    ref = {r["decision_id"]: r["ref"] for r in val_rows}; logits_st = {n: {} for n in LF.SEEDS}
    finite = True; shapes_ok = True
    for r in val_rows:
        for n in LF.SEEDS:
            lg = LF.head_logits(heads[n], r["hq"], r["ho"]); logits_st[n][r["decision_id"]] = lg
            finite &= all(map(lambda x: x == x and abs(x) != float("inf"), lg)); shapes_ok &= len(lg) == len(r["pos"])
    picks_st = {n: {d: max(range(len(v)), key=v.__getitem__) for d, v in logits_st[n].items()} for n in LF.SEEDS}
    seed_top1 = {n: sum(float(picks_st[n][d] == ref[d]) for d in VAL_IDS) / len(VAL_IDS) for n in LF.SEEDS}
    hh = hit_mean(picks_st, ref); diff_b = [d for d, h_, rc in zip(VAL_IDS, hh, REC_["head_real"]) if abs(h_ - rc) > TH["b_stored_feature_forward"]["hit_tolerance"]]
    seed_ok = all(abs(seed_top1[n] - REC_["seed_top1"][n]) < 5e-5 for n in LF.SEEDS)
    # recaptured features through the same heads
    picks_re = {n: {} for n in LF.SEEDS}; rrows = {}
    for r in val_rows:
        if r["decision_id"] in recap:
            rr = row(r["decision_id"], recap[r["decision_id"]][0], recap[r["decision_id"]][1], RQ["val"][r["decision_id"]], LAB["val"][r["decision_id"]]); rrows[r["decision_id"]] = rr
            for n in LF.SEEDS:
                lg = LF.head_logits(heads[n], rr["hq"], rr["ho"]); picks_re[n][r["decision_id"]] = max(range(len(lg)), key=lg.__getitem__)
    tot = len(VAL_IDS) * 3; same = sum(int(picks_re[n].get(d) == picks_st[n][d]) for n in LF.SEEDS for d in VAL_IDS)
    c = TH["c_recaptured_feature_forward"]; re_top1 = sum(sum(float(picks_re[n][d] == ref[d]) for n in LF.SEEDS) / 3 for d in VAL_IDS) / len(VAL_IDS)
    # permutation stability
    pt = TI["t4_head"]; perm_worst = 0.0; rank_bad = 0; trials = 0
    for r in val_rows:
        n_ = len(r["pos"]); rng = random.Random(hash(r["decision_id"]) % 1000 if False else sum(map(ord, r["decision_id"])))
        perms = [list(reversed(range(n_)))] + [rng.sample(range(n_), n_) for _ in range(pt["permutation_trials"] - 1)]
        for nme in LF.SEEDS:
            base = logits_st[nme][r["decision_id"]]
            for pm in perms:
                lg = LF.head_logits(heads[nme], r["hq"], r["ho"][pm]); trials += 1
                d_ = max(abs(lg[i] - base[pm[i]]) for i in range(n_)); perm_worst = max(perm_worst, d_)
                tol = pt["permutation_max_abs_logit_diff"]
                o1 = sorted(range(n_), key=lambda i: -base[i]); o2 = [pm[i] for i in sorted(range(n_), key=lambda i: -lg[i])]
                if o1 != o2 and any(abs(base[o1[k]] - base[o1[k + 1]]) >= 2 * tol for k in range(n_ - 1)): rank_bad += int(any(o1[k] != o2[k] and abs(base[o1[k]] - base[o2[k]]) >= 2 * tol for k in range(n_)))
    # controls (reported, NOT gates): probe refit on the 334 TRAIN pools and a random head
    train_rows = [row(d, FE[n][d]["hq"], FE[n][d]["ho"], RQ[n][d], LAB["v3"][d]) for n in ("v2", "v3") for d in FE[n] if d in LAB["v3"]]
    PROBE = LF.fit_probe(train_rows, C=TH["d_probe"]["probe_C"])
    rand_head = LF.build_head(PRE, ADP["final-s0"], random_head_seed=0)
    probe_hit = [LF.hit(LF.probe_decision(PROBE, r["ho"]), r["ref"]) for r in val_rows]; rand_hit = [LF.hit(LF.head_logits(rand_head, r["hq"], r["ho"]), r["ref"]) for r in val_rows]
    bpass = len(diff_b) <= TH["b_stored_feature_forward"]["max_pools_differing"] and seed_ok
    cpass = same / tot >= c["min_same_top_choice"] and abs(re_top1 - REC_["head_top1"]) <= c["max_head_top1_delta"]
    t4_ok = finite and shapes_ok and bpass and cpass and perm_worst <= pt["permutation_max_abs_logit_diff"] and rank_bad == 0
    receipt("T4_laya", t4_ok, {"thresholds": {"gate": TH["b_stored_feature_forward"], "recaptured": c, "t4": pt}, "finite": finite, "shapes_ok": shapes_ok, "pools": len(VAL_IDS),
            "stored_features": {"head_top1_mean_of_seeds": round(sum(hh) / len(hh), 4), "recorded": REC_["head_top1"], "per_seed_top1": seed_top1, "recorded_per_seed": REC_["seed_top1"],
                                "pools_differing_from_recorded": diff_b, "pass": bpass},
            "recaptured_features": {"same_top_choice_fraction": round(same / tot, 4), "head_top1": round(re_top1, 4), "pass": cpass},
            "permutation": {"trials": trials, "max_abs_logit_diff": perm_worst, "ranking_violations": rank_bad},
            "controls_not_gates__label": "probe and random head are reported controls only; the random head is a MISMATCH CONTROL (trained adapter + untrained head), NOT a valid performance comparison", "controls_not_gates": {"probe_top1_refit_C0.01": round(sum(probe_hit) / len(probe_hit), 4), "probe_recorded_top1": REC_["probe_top1"], "probe_pools_differing_from_recorded":
                                   sum(abs(h_ - rc) > 1e-9 for h_, rc in zip(probe_hit, REC_["probe_real"])), "random_head_top1": round(sum(rand_hit) / len(rand_hit), 4)},
            "capture_provenance": "stored VAL features (Tesla T4 fp16) and their fresh T2 recaptures"})
    # save real scores for T5 (VAL option texts are development data already used on Kaggle)
    probe_sc = {r["decision_id"]: LF.probe_decision(PROBE, r["ho"]) for r in val_rows}; rnd_sc = {r["decision_id"]: LF.head_logits(rand_head, r["hq"], r["ho"]) for r in val_rows}
    json.dump({"tag": TAG, "pools": [{"decision_id": r["decision_id"], "options": r["options"], "ref": r["ref"],
               "laya_per_seed_logits": {n: logits_st[n][r["decision_id"]] for n in LF.SEEDS}, "laya_score": [sum(logits_st[n][r["decision_id"]][i] for n in LF.SEEDS) / 3 for i in range(len(r["pos"]))],
               "probe_score": probe_sc[r["decision_id"]], "random_head_score": rnd_sc[r["decision_id"]], "origin": "stored_features_verified_by_recapture"} for r in val_rows]},
              open(OUT + "/T4_scores.json", "w", encoding="utf-8"))

    # ================================================================== T7 provenance rejection, verified by the runtime path
    FS = RT.FeatureScorer(heads, probe=PROBE, random_head=rand_head)
    by_n = {}
    for r in val_rows: by_n.setdefault(len(r["pos"]), []).append(r)
    small = sorted(val_rows, key=lambda r: len(RQ["val"][r["decision_id"]]["context_text"]))[:6]
    def mk(r_, **over):
        rq = RQ["val"][r_["decision_id"]]; sr = FE["val"][r_["decision_id"]]
        d = dict(decision_id=r_["decision_id"], context_text=rq["context_text"], options=rq["options"], hq=sr["hq"].clone(), ho=sr["ho"].clone(),
                 provenance=RT.make_provenance(device=DEVNAME, context_text=rq["context_text"], options=rq["options"]))
        d.update(over); return d
    genuine = [mk(r_) for r_ in small]
    def attempt(batch, **kw):
        try:
            RT.score_stored(batch, capture_fn=capture, scorer=FS, sample_n=len(batch), seed=0, **kw); return {"refused": False}
        except RT.ProvenanceError as e: return {"refused": True, "message": str(e)[:300]}
    cases = {}
    cases["control_genuine_accepted"] = attempt(genuine)
    g = lambda i: dict(genuine[i], provenance=dict(genuine[i]["provenance"]))
    def with_prov(**ch):
        b = [dict(x) for x in genuine]; b[0] = dict(b[0], provenance=dict(b[0]["provenance"], **ch)); return b
    cases["wrong_model_revision"] = attempt(with_prov(model_revision="0" * 40))
    cases["wrong_model_repo"] = attempt(with_prov(model_repo="Qwen/Qwen3-4B"))
    cases["wrong_dtype_metadata"] = attempt(with_prov(dtype="bfloat16"))
    b = [dict(x) for x in genuine]; b[0] = dict(b[0], hq=b[0]["hq"].float(), ho=b[0]["ho"].float()); cases["wrong_dtype_tensor_float32"] = attempt(b)
    cases["wrong_depths"] = attempt(with_prov(depths=[12, 18, 24]))
    cases["wrong_recipe_id"] = attempt(with_prov(recipe_id="other_recipe_v9"))
    cases["wrong_device"] = attempt(with_prov(device="NVIDIA GeForce RTX 3050 6GB Laptop GPU"))
    cases["wrong_capture_code_hash"] = attempt(with_prov(capture_code_sha256={"capture/capture.py": "0" * 64, "capture/capture_resident.py": "1" * 64}))
    sd = float(genuine[0]["ho"].float().std())
    fab = lambda x: (torch.randn(x.shape, generator=torch.Generator().manual_seed(7)) * float(x.float().std())).half()
    b = [dict(x) for x in genuine]; b[0] = dict(b[0], hq=fab(b[0]["hq"]), ho=fab(b[0]["ho"])); cases["fabricated_random_tensors_valid_metadata"] = attempt(b)
    twin = next(((x, y) for n_, rs in by_n.items() for x in rs for y in rs if x is not y and x["decision_id"] in {s["decision_id"] for s in small} and y["decision_id"] not in {s["decision_id"] for s in small}), None)
    if twin is None: twin = (small[0], next(y for y in val_rows if len(y["pos"]) == len(small[0]["pos"]) and y is not small[0]))
    b = [dict(x) for x in genuine]; i0 = [s["decision_id"] for s in small].index(twin[0]["decision_id"])
    b[i0] = dict(b[i0], hq=FE["val"][twin[1]["decision_id"]]["hq"].clone(), ho=FE["val"][twin[1]["decision_id"]]["ho"].clone()); cases["real_features_of_another_text_valid_metadata"] = attempt(b)
    b = [dict(x) for x in genuine]; b[0] = dict(b[0], ho=(b[0]["ho"].float() + 0.05 * sd * torch.randn(b[0]["ho"].shape, generator=torch.Generator().manual_seed(3))).half()); cases["perturbed_5pct_valid_metadata"] = attempt(b)
    b = [dict(x) for x in genuine]; b[0] = dict(b[0], hq="not a tensor"); cases["text_instead_of_tensor"] = attempt(b)
    try: RT.LiveBackend(RT.LiveScorer(capture, FS, DEVNAME)).score("next_action", ["candidate A", "candidate B"]); cases["bare_text_candidates_live"] = {"refused": False}
    except RT.ProvenanceError as e: cases["bare_text_candidates_live"] = {"refused": True, "message": str(e)[:200]}
    # replay verification re-captures 100% of the records in the batch for this milestone (sample_n = len): ONE fabricated record at ANY position must be caught
    positional = []
    for pos in range(len(genuine)):
        bp = [dict(x) for x in genuine]; bp[pos] = dict(bp[pos], hq=fab(bp[pos]["hq"]), ho=fab(bp[pos]["ho"]))
        res_ = attempt(bp); positional.append({"position": pos, "decision_id": genuine[pos]["decision_id"], "refused": res_["refused"], "message": res_.get("message", "")[:160]})
    default_sampling = RT.score_stored(genuine, capture_fn=capture, scorer=FS, seed=1)      # option for LARGE batches only: a random subset (>=25%, min 3); not used for any verdict
    # live mode: capture + score from TEXT in one call, equals the stored-feature forward
    live = RT.LiveScorer(capture, FS, DEVNAME); live_rows = []
    for r_ in small[:3]:
        out = live.score_texts(RQ["val"][r_["decision_id"]]["context_text"], RQ["val"][r_["decision_id"]]["options"])
        ref_s = FS._score(FE["val"][r_["decision_id"]]["hq"], FE["val"][r_["decision_id"]]["ho"])["laya"]["score"]
        live_rows.append({"decision_id": r_["decision_id"], "origin": out["origin"], "max_abs_diff_vs_stored_forward": max(abs(a_ - b_) for a_, b_ in zip(out["scores"]["laya"]["score"], ref_s)), "seconds": out["seconds"]})
    expect_refused = [k for k in cases if k != "control_genuine_accepted"]
    meta_msgs = ("provenance", "capture code", "depths", "captured on", "features must be", "no provenance")
    t7_ok = (not cases["control_genuine_accepted"]["refused"] and all(cases[k]["refused"] for k in expect_refused)
             and all("fresh capture" in cases[k]["message"] for k in ("fabricated_random_tensors_valid_metadata", "real_features_of_another_text_valid_metadata", "perturbed_5pct_valid_metadata"))
             and all(p_["refused"] and "fresh capture" in p_["message"] and p_["decision_id"] in p_["message"] for p_ in positional) and len(positional) == len(genuine)
             and all(l["max_abs_diff_vs_stored_forward"] <= 1e-3 for l in live_rows))
    receipt("T7_provenance_rejection", t7_ok, {"thresholds": TI["t7_provenance_rejection"], "cases": cases, "replay_verification": {"records_recaptured": "100% of the batch (sample_n = len(batch)) in every case above", "single_fabricated_record_at_every_position": positional,
            "optional_default_sampling_for_large_batches": {"batch": default_sampling["batch"], "sampled": default_sampling["sampled"], "note": "random subset >= 25% (min 3); NOT used for any verdict in this milestone; a fabricated record outside the subset would not be caught"}},
            "live_mode": live_rows, "verified_by": "runtime re-capture on the frozen Qwen (laya_runtime.score_stored / LiveScorer), not by stored metadata",
            "capture_provenance": "live captures by this kernel on " + DEVNAME})
    # ================================================================== T2b: the v1-pinned alternative implementation (tools/fable_capture.py) vs the stored training features
    import importlib.util
    spec = importlib.util.spec_from_file_location("fable_capture_alt", G("tools/fable_capture.py")); FC = importlib.util.module_from_spec(spec); spec.loader.exec_module(FC)
    alt = FC.Capturer(model.model, DEP, torch.device("cuda"), 8); alt_rows = []
    for did in ["fable5:bash-dotlink:2", "fable5:py-klondike:1", "fable5:go-dashfeed:4"]:
        n_ = find(did); sr = FE[n_][did]; rq = RQ[n_][did]
        item, exc = FC.plan_one(tok, dict(decision_id=did, context_text=rq["context_text"], options=rq["options"]), FC.LIMITS)
        if item is None: alt_rows.append({"decision_id": did, "status": "excluded", "detail": exc}); continue
        hq_a, ho_a = alt.capture(item); hq_a, ho_a = hq_a.half(), ho_a.half()
        c_ = RT.compare_features((hq_a, ho_a), (sr["hq"], sr["ho"]), {"min_cosine": TH["a_feature_recapture"]["min_cosine"], "max_abs_diff": TH["a_feature_recapture"]["max_abs_diff"]})
        alt_rows.append({"decision_id": did, "status": "ok", "cos_min": c_["cos_min"], "max_abs_diff": c_["max_abs_diff"], "within_T2_thresholds": c_["passed"], "bit_identical": bool(torch.equal(hq_a, sr["hq"]) and torch.equal(ho_a, sr["ho"]))})
    receipt("T2b_alternative_implementation", all(r_["status"] == "ok" and r_["within_T2_thresholds"] for r_ in alt_rows), {"purpose": "evidence for CORRECTION_NOTE.md: tools/fable_capture.py (v1 pin) vs the stored training features",
            "implementation_sha256": sha_file(G("tools/fable_capture.py")), "rows": alt_rows, "thresholds": TH["a_feature_recapture"],
            "note": "informational for the pin correction; the verdict on the scorer's capture path is T2 (capture/capture*.py, bit-identical)"})
    open(REC + "/README.md", "w", encoding="utf-8").write("%s\n\nReceipts of kernel evidence_ladder_laya_integration_v1 (%s). Device %s, %s.\nT1 real capture, T2 training parity, T3 adapter, T4 LAYA, T7 provenance rejection (GPU part). "
                                                       "T5/T6 are separate. PERFORMANCE: NOT TESTED.\n" % (TAG, RS["started_utc"], DEVNAME, json.dumps(RS["versions"])))
    finish("DONE", tests=RS["tests"])
except BaseException as e:
    RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc(); finish("FAILED")
