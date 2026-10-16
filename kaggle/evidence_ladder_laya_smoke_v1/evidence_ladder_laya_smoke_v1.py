"""[LAYA-INT-1 · Aim: integration, not performance] SMALL SMOKE RUN (Kaggle Tesla T4, fp16, internet OFF). 3 stored TRAIN pools (+3 stored VAL pools for the recorded outputs).
S1 load frozen Qwen · S2 capture 18/24/30 with the training recipe vs stored features · S3 3 frozen adapters + pretrained head (sha256) · S4 real scores per option per adapter ·
S5 recorded outputs · S6 scores into the Evidence-Ladder controller (L0 only, never overrides L5/L3). Thresholds: configs/laya_smoke_thresholds_v1.toml (frozen before the run).
STOPS at the first failed stage. No training, no generation, no candidate code executed. Output: /kaggle/working/receipts/*.json, SMOKE_SUMMARY.json, RUN_SUMMARY.json. Exits 0 always."""
import glob, hashlib, json, os, sys, time, traceback

T0 = time.time(); OUT = "/kaggle/working"; REC = OUT + "/receipts"; os.makedirs(REC, exist_ok=True)
TAG = "[LAYA-INT-1 · Aim: integration, not performance]"
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
LIM = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8)
DEP = [18, 24, 30]; CUE = "\n\nNext action:"; OPT_PREFIX = " "
RS = {"kernel": "evidence_ladder_laya_smoke_v1", "tag": TAG, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}, "gpu_mem_gib": {}}
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def save():
    with open(OUT + "/RUN_STATE.json", "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
def finish(verdict, **kw):
    RS["verdict"] = verdict; RS["total_seconds"] = round(time.time() - T0, 1); RS.update(kw); save()
    for n in ("RUN_SUMMARY.json", "SMOKE_SUMMARY.json"):
        with open(OUT + "/" + n, "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)
def sha_file(p):
    x = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
    return x.hexdigest()
def stage(name, passed, body):
    import torch
    RS["gpu_mem_gib"][name] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    body = dict(body, stage=name, passed=bool(passed), tag=TAG, seconds_since_start=round(time.time() - T0, 1), gpu_peak_gib=RS["gpu_mem_gib"][name], created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = "%s/%s.json" % (REC, name); assert not os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f: json.dump(body, f, indent=1, default=str)
    RS["stages"][name] = bool(passed); save(); log("STAGE", name, "PASS" if passed else "FAIL")
    if not passed: finish("SMOKE_FAIL", failed_stage=name)

try:
    import torch, transformers
    import torch.nn.functional as F
    RS["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available(): finish("SMOKE_FAIL", failed_stage="no_cuda")
    DEVNAME = torch.cuda.get_device_name(0); RS["gpu"] = DEVNAME
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_laya_int1_v1"), None)
    if not mp: finish("SMOKE_FAIL", failed_stage="no_bundle")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or sha_file(os.path.join(BUNDLE, f)) != h]
    if bad: finish("SMOKE_FAIL", failed_stage="bundle_hash", bad=bad)
    sys.path.insert(0, BUNDLE)
    import tomllib
    from evidence_ladder.models import laya_forward as LF, laya_runtime as RT, laya as LAYA
    G = lambda p: os.path.join(BUNDLE, p)
    TS = tomllib.load(open(G("configs/laya_smoke_thresholds_v1.toml"), "rb")); RS["thresholds_sha256"] = sha_file(G("configs/laya_smoke_thresholds_v1.toml"))
    BL = json.load(open(G("results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json"), encoding="utf-8"))

    # ---------------------------------------------------------------- S1 load frozen Qwen
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs: finish("SMOKE_FAIL", failed_stage="S1_no_model_dir")
    md = dirs[0]; t = time.time(); ok = {f: sha_file(os.path.join(md, f)) == h for f, h in QWEN_SHA.items()}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float16).eval().to("cuda")      # the TRAIN capture loader call (kaggle/level12_t4.py)
    pdt = {str(p.dtype) for p in model.parameters()}
    s1_ok = all(ok.values()) and DEVNAME.startswith(TS["s1_load"]["device_prefix"]) and pdt == {"torch.float16"} and man["qwen_revision"] == TS["s1_load"]["model_revision"]
    stage("S1_load_qwen", s1_ok, {"model_dir": md, "pinned_revision": man["qwen_revision"], "shard_sha256_ok": ok, "param_dtypes": sorted(pdt), "device": DEVNAME, "versions": RS["versions"],
                                  "load_seconds": round(time.time() - t, 1), "n_layers": len(model.model.layers), "hidden": model.config.hidden_size, "thresholds": TS["s1_load"],
                                  "note": "the revision is established by the sha256 of every weight shard and tokenizer.json equalling the pin of tools/fable_capture.py for revision 768f209d"})
    from capture.capture_resident import ResidentCapturer
    import transformers.integrations.sdpa_attention as SDPA
    class GatedCapturer(ResidentCapturer):
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
    def capture(context_text, options):
        pre = tok.encode(context_text + CUE, add_special_tokens=False); suf = [tok.encode(OPT_PREFIX + o, add_special_tokens=False) for o in options]
        info = dict(prefix_tokens=len(pre), option_tokens=[len(s) for s in suf])
        if not options or len(options) > LIM["max_options"] or len(pre) > LIM["max_prefix_tokens"] or any(len(s) > LIM["max_option_tokens"] or not s for s in suf) \
                or len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]: return None, None, dict(info, status="excluded_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        full = model.model.layers; model.model.layers = torch.nn.ModuleList(list(full)[:max(DEP)]); t = time.time(); torch.cuda.reset_peak_memory_stats()
        try:
            cap, _ = C.packed(rec); keys = sorted(cap); hq = torch.stack([cap[d][0] for d in DEP]); ho = torch.stack([cap[d][1] for d in DEP], 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return None, None, dict(info, status="skipped_oom")
        finally: model.model.layers = full
        hq, ho = hq.half().cpu(), ho.half().cpu()
        if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()): return None, None, dict(info, status="nonfinite_fp16")
        return hq, ho, dict(info, status="ok", mode="packed", seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), depth_keys=keys, lm_head_calls=C.lm_calls)

    # ---------------------------------------------------------------- data (stored features, request texts, labels)
    FE = {n: {r["decision_id"]: r for r in torch.load(G("gate/feat_%s_B.pt" % n), map_location="cpu", weights_only=False)["records"]} for n in ("v2", "v3", "val")}
    RQ = {n: {d["decision_id"]: d for d in (json.loads(l) for l in open(G("gate/req_%s.jsonl" % n), encoding="utf-8"))} for n in ("v2", "v3", "val")}
    LAB = json.load(open(G("gate/labels.json"), encoding="utf-8")); REC_ = json.load(open(G("gate/recorded.json"), encoding="utf-8"))
    find = lambda did: next(n for n in ("v2", "v3", "val") if did in FE[n])
    def ordered(did, hq, ho, lab_key):
        rq = RQ[find(did)][did]; lab = LAB[lab_key][did]; ix = {o: i for i, o in enumerate(rq["options"])}
        return dict(decision_id=did, hq=hq, ho=torch.stack([ho[ix[o]] for o in lab["options"]]), options=lab["options"], ref=lab["reference_index"])

    # ---------------------------------------------------------------- S2 capture parity (TRAIN pools)
    s2t = TS["s2_capture_parity"]; rows, recap = [], {}
    for did in TS["sample"]["train_ids"]:
        n = find(did); sr = FE[n][did]; rq = RQ[n][did]
        assert sr["option_sha256"] == rq["option_sha256"] and sr["context_sha256"] == rq["context_sha256"], "stored record and request disagree: " + did
        hq, ho, info = capture(rq["context_text"], rq["options"])
        if hq is None: rows.append(dict(decision_id=did, status=info["status"], info=info)); continue
        recap[did] = (hq, ho); cm = RT.compare_features((hq, ho), (sr["hq"], sr["ho"]), {"min_cosine": s2t["min_cosine"], "max_abs_diff": s2t["max_abs_diff"]})
        per_depth = [{"depth": DEP[k], "hq_cos": F.cosine_similarity(hq[k].double(), sr["hq"][k].double(), 0).item(), "ho_cos_min": F.cosine_similarity(ho[:, k].double(), sr["ho"][:, k].double(), -1).min().item(),
                      "max_abs": max((hq[k].float() - sr["hq"][k].float()).abs().max().item(), (ho[:, k].float() - sr["ho"][:, k].float()).abs().max().item())} for k in range(3)]
        rows.append(dict(decision_id=did, status="ok", n_options=ho.shape[0], hq_shape=list(hq.shape), ho_shape=list(ho.shape), dtype=str(hq.dtype), device=DEVNAME, cos_min=cm["cos_min"], max_abs_diff=cm["max_abs_diff"],
                         passed=cm["passed"], bit_identical=bool(torch.equal(hq, sr["hq"]) and torch.equal(ho, sr["ho"])), per_depth=per_depth, info=info))
    s2_ok = len(rows) == len(TS["sample"]["train_ids"]) and all(r["status"] == "ok" and r["passed"] for r in rows) and all(r.get("info", {}).get("depth_keys") == DEP for r in rows)
    stage("S2_capture_parity", s2_ok, {"thresholds": s2t, "rows": rows, "reference": "stored fp16 training features (Tesla T4, torch 2.11.0+cu128, transformers 5.16.1: level12/v3 capture runs)",
                                       "capture_provenance": RT.make_provenance(device=DEVNAME, context_text=RQ[find(TS["sample"]["train_ids"][0])][TS["sample"]["train_ids"][0]]["context_text"],
                                                                                options=RQ[find(TS["sample"]["train_ids"][0])][TS["sample"]["train_ids"][0]]["options"])})

    # ---------------------------------------------------------------- S3 weights
    PRE = G("models/pretrained_31.safetensors"); ADP = {n: G("models/%s.pt" % n) for n in LF.SEEDS}
    from safetensors import safe_open
    with safe_open(PRE, "pt") as f:
        t31 = {k: hashlib.sha256(f.get_tensor(k).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k in f.keys()}
    heads = {n: LF.build_head(PRE, ADP[n]) for n in LF.SEEDS}
    from decision_head import fable_train as ft
    n_ad = {n: sum(p.numel() for k, p in heads[n].named_parameters() if ft.group_of(k) == "adapter") for n in LF.SEEDS}
    sha_ad = {n: sha_file(ADP[n]) for n in LF.SEEDS}
    s3_ok = (all(sha_ad[n] == BL["adapters"][n]["sha256"] for n in LF.SEEDS) and all(v == TS["s3_weights"]["adapter_trainable_params"] for v in n_ad.values()) and len(t31) == 31
             and t31 == man["pretrained_31_tensor_sha256"] and man["pretrained_source_sha256"] == BL["pretrained_head"]["sha256"])
    stage("S3_weights", s3_ok, {"adapter_sha256": sha_ad, "manifest_sha256": {n: BL["adapters"][n]["sha256"] for n in LF.SEEDS}, "adapter_trainable_params": n_ad, "pretrained_tensors": len(t31),
                                "pretrained_tensors_match_pack_time_hashes": t31 == man["pretrained_31_tensor_sha256"], "pretrained_source_sha256": man["pretrained_source_sha256"],
                                "pretrained_source_note": "the 842 MB source file was sha256-verified against the MANIFEST at pack time; here the 31 byte-copied tensors are verified against the hashes recorded then"})

    # ---------------------------------------------------------------- S4 real scores per option per adapter (recaptured features) + S5a vs stored features
    s5 = TS["s5_recorded"]; s4_rows, s5a_rows = [], []
    for did in TS["sample"]["train_ids"]:
        rr = ordered(did, recap[did][0], recap[did][1], "v3"); sr_ = FE[find(did)][did]; st = ordered(did, sr_["hq"], sr_["ho"], "v3")
        per = {}
        for n in LF.SEEDS:
            a = LF.head_logits(heads[n], rr["hq"], rr["ho"]); b = LF.head_logits(heads[n], st["hq"], st["ho"])
            per[n] = {"scores_recaptured": a, "scores_stored": b, "finite": all(x == x and abs(x) < 1e30 for x in a), "n_scores": len(a), "n_options": len(rr["options"]),
                      "top_recaptured": max(range(len(a)), key=a.__getitem__), "top_stored": max(range(len(b)), key=b.__getitem__), "max_abs_diff": max(abs(x - y) for x, y in zip(a, b)), "recorded_reference_index": rr["ref"]}
        s4_rows.append({"decision_id": did, "options_sha256": [LF.sha256_text(o) for o in rr["options"]], "per_adapter": per})
    s4_ok = all(v["finite"] and v["n_scores"] == v["n_options"] for r in s4_rows for v in r["per_adapter"].values())
    stage("S4_laya_scores", s4_ok, {"rows": s4_rows, "note": "one score per option, each adapter separately, scores are logits of the pretrained head (no NONE class)",
                                    "capture_provenance": "S2 live recaptures on " + DEVNAME})
    s5a_ok = all(v["top_recaptured"] == v["top_stored"] and v["max_abs_diff"] <= s5["a_max_abs_logit_diff"] for r in s4_rows for v in r["per_adapter"].values())
    # S5b: stored VAL features reproduce the recorded per-pool mean hit of the first 3 recorded VAL pools
    vids = REC_["decision_id"][:3]; b_rows = []
    for k, did in enumerate(vids):
        r = ordered(did, FE["val"][did]["hq"], FE["val"][did]["ho"], "val"); picks = {}
        for n in LF.SEEDS:
            lg = LF.head_logits(heads[n], r["hq"], r["ho"]); picks[n] = max(range(len(lg)), key=lg.__getitem__)
        hit = sum(float(picks[n] == r["ref"]) for n in LF.SEEDS) / 3
        b_rows.append({"decision_id": did, "picks": picks, "ref": r["ref"], "mean_hit": hit, "recorded_mean_hit": REC_["head_real"][k], "match": abs(hit - REC_["head_real"][k]) <= s5["b_hit_tolerance"]})
    s5_ok = s5a_ok and all(r["match"] for r in b_rows)
    stage("S5_recorded", s5_ok, {"a_recaptured_vs_stored_training_features": {"same_top_choice_and_max_abs_logit_diff_ok": s5a_ok, "threshold": s5["a_max_abs_logit_diff"],
                                 "rows": [{"decision_id": r["decision_id"], "per_adapter": {n: {"max_abs_diff": v["max_abs_diff"], "same_top": v["top_recaptured"] == v["top_stored"]} for n, v in r["per_adapter"].items()}} for r in s4_rows]},
                                 "b_recorded_val_outputs": b_rows, "source": REC_["source"], "caveat": "recorded per-pool scorer outputs exist only for VAL pools; TRAIN pools were fitted by the final adapters"})

    # ---------------------------------------------------------------- S6 controller (real scorer interface -> real journal -> real invariant check)
    from evidence_ladder.controller.journal import EvidenceJournal
    from evidence_ladder.controller.next_action import decide_next_action
    from evidence_ladder.controller import invariants as INV
    FS = RT.FeatureScorer(heads); live = RT.LiveScorer(capture, FS, DEVNAME)
    sc = LAYA.LayaScorer(authorized=True, backend=RT.LiveBackend(live, "laya"))        # the existing evidence_ladder.models.laya interface; needs the baseline MANIFEST + capture-code hashes
    s6 = {"scorer_def_status": sc.scorer_def["status"], "scorer_disabled": sc.disabled, "scorer_disabled_reason": sc.meta.disabled_reason, "capture_code_report": sc.meta.capture_report}
    if sc.disabled: stage("S6_controller", False, dict(s6, error="scorer disabled by its compatibility check"))
    did = TS["sample"]["train_ids"][0]; rq = RQ[find(did)][did]
    scores = sc.score("next_action", {"context_text": rq["context_text"], "options": rq["options"]})       # capture + 3 adapters + head inside this call
    prov = sc._backend.last["provenance"]
    n = len(scores); top = max(range(n), key=scores.__getitem__); low = min(range(n), key=scores.__getitem__)
    J = EvidenceJournal(); texts = rq["options"]; res = {}
    res["a_no_trusted_evidence"] = decide_next_action(J, "S6a", texts, scores, None, scorer_id=sc.scorer_id, scorer_provenance=prov)
    tr = [{} for _ in range(n)]; tr[top] = {"L5": "fail"}
    res["b_top_laya_fails_L5"] = decide_next_action(J, "S6b", texts, scores, tr, scorer_id=sc.scorer_id, scorer_provenance=prov)
    tr = [{} for _ in range(n)]; tr[low] = {"L5": "pass", "L3": "pass"}
    res["c_lowest_laya_has_trusted_passes"] = decide_next_action(J, "S6c", texts, scores, tr, scorer_id=sc.scorer_id, scorer_provenance=prov)
    res["d_scorer_unavailable"] = decide_next_action(J, "S6d", texts, None, None)
    expect_a = sorted(range(n), key=lambda i: (-scores[i], i)); i2 = INV.check_i2(J)
    ok_a = res["a_no_trusted_evidence"]["selected_index"] == top and [int(x.split("#")[1]) for x in res["a_no_trusted_evidence"]["order"]] == [i for i in expect_a]
    ok_b = res["b_top_laya_fails_L5"]["selected_index"] != top and res["b_top_laya_fails_L5"]["blocked"] == ["S6b#%d" % top]
    ok_c = res["c_lowest_laya_has_trusted_passes"]["selected_index"] == low
    ok_d = res["d_scorer_unavailable"]["selected_index"] == 0 and not any(r["approved"] for r in res.values())
    try: sc.score("next_action", ["candidate A", "candidate B"]); refusal = "NOT REFUSED"
    except RT.ProvenanceError as e: refusal = "refused: " + str(e)[:160]
    ok_d = ok_d and refusal.startswith("refused")
    stage("S6_controller", ok_a and ok_b and ok_c and ok_d and i2 == "", dict(s6, scores_via_LayaScorer_next_action=scores, laya_top=top, laya_lowest=low, decisions=res, check_i2=i2 or "ok",
          journal_hash=J.journal_hash(), journal_entries=J.entries(), checks={"a_order_follows_L0": ok_a, "b_top_laya_never_selected_when_L5_fails": ok_b, "c_trusted_evidence_outranks_L0": ok_c, "d_approved_false_always": ok_d},
          scorer_provenance=prov, scorer_origin=sc._backend.last["origin"], text_only_refusal=refusal))
    open(REC + "/README.md", "w", encoding="utf-8").write("%s\n\nSMALL SMOKE RUN receipts, kernel evidence_ladder_laya_smoke_v1, %s, %s. SMOKE PASS = the connection works in the tested cases. Not integration PASS. "
                                                       "PERFORMANCE: NOT TESTED.\n" % (TAG, DEVNAME, json.dumps(RS["versions"])))
    finish("SMOKE_PASS", stages=RS["stages"])
except BaseException as e:
    RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc(); finish("SMOKE_FAIL", failed_stage="exception")
