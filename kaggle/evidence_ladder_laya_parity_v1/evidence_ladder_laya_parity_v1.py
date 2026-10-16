"""Evidence-Ladder arm-E kernel v1: GENERATE + CAPTURE + SCORE in one run (Kaggle Tesla T4, HF transformers, fp16, internet OFF). No candidate code is executed; no secrets; public prompts only.
TEMPLATE: tools/pack_el_laya_gen_v1.py copies it to kaggle/evidence_ladder_laya_gen_v1/ (the bundle with code, adapters, gate data is a separate Kaggle dataset).

Order of work (every phase writes to /kaggle/working before the next starts):
  1. bundle + Qwen shard integrity, model load (fp16, same loader call as the TRAIN captures)
  2. PARITY GATE (configs/laya_head_gate_v1.toml, pre-registered): recapture 12 stored TRAIN + 53 stored VAL decisions with the TRAINING recipe and compare with the stored
     features; run frozen adapters + head on stored and recaptured features against the recorded VAL hits; refit the linear probe on the 334 TRAIN pools and reproduce
     its recorded VAL hits. Any failure -> RUN_SUMMARY.json verdict GATE_FAILED and NOTHING is generated.
  3. generation: 4 samples per task (sample_idx 0-1 qwen_sample_v2, 2-3 qwen_sample_v2b), T0.6/top_p0.95/top_k20, 8192 new tokens, left-padded batch of 3 with OOM split,
     appended + fsynced to completions.jsonl (schema of evidence_ladder_gpu_gen_v1 plus sample fields).
  4. when all samples of a task exist: ONE packed capture of the task's distinct candidate programs (code only) as options of one shared prefix, exactly the training
     recipe; features saved (features/<task>.pt + sha256); scores from the frozen adapters+head (3 seeds), the linear probe and the random-head control are appended to
     pools.jsonl and sample_scores.jsonl. Scores are L0 and EXPLORATORY (out-of-distribution): they are never an approval.
Env: EL_TASK_SHARD=i/n (default 0/1; tasks are sharded by Python/Java pair), EL_BUDGET_H (stop starting batches, default 10.5), EL_MAXNEW, EL_BATCH, EL_K (default 4).
Always exits 0 with RUN_SUMMARY.json."""
import glob, hashlib, json, os, sys, time, traceback

T0 = time.time(); OUT = "/kaggle/working"
BUDGET_H = float(os.environ.get("EL_BUDGET_H", "10.5")); MAXNEW = int(os.environ.get("EL_MAXNEW", "8192")); BATCH = int(os.environ.get("EL_BATCH", "3"))
MODE = "parity"   # the packer writes kaggle/evidence_ladder_laya_parity_v1.py with MODE = "parity": the gate only, no generation
K = int(os.environ.get("EL_K", "4")); SHARD = os.environ.get("EL_TASK_SHARD", "0/1")
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
LIM = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8)   # the TRAIN capture limits (kaggle/level12_t4.py)
DEP = [18, 24, 30]; CUE = "\n\nNext action:"; OPT_PREFIX = " "
RS = {"kernel": "evidence_ladder_laya_gen_v1", "shard": SHARD, "k": K, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phases": {}, "notes": [],
      "samples_done": 0, "pools_done": 0, "oom_events": []}
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
def append_fsync(path, objs):
    with open(path, "a", encoding="utf-8") as f:
        for o in objs: f.write(json.dumps(o, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())

try:
    import torch, transformers
    import torch.nn.functional as F
    RS["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available(): finish("FAILED_NO_CUDA")
    RS["gpu"] = torch.cuda.get_device_name(0); RS["gpus"] = torch.cuda.device_count(); log("GPU", RS["gpu"], RS["versions"])
    shard_i, shard_n = (int(x) for x in SHARD.split("/")); assert 0 <= shard_i < shard_n
    # ---- bundle + integrity
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_laya_gen_v1"), None)
    if not mp: finish("FAILED_NO_BUNDLE")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8")); t = time.time()
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or sha_file(os.path.join(BUNDLE, f)) != h]
    RS["phases"]["bundle_hash"] = round(time.time() - t, 1)
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad)
    sys.path.insert(0, BUNDLE)
    from evidence_ladder.models import qwen as Q, qwen_v2b as V2B, laya_forward as LF
    from evidence_ladder.common.interfaces import sha256_text
    import tomllib
    assert Q.SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, Q.SAMPLING
    assert Q.QWEN_REVISION == man["qwen_revision"] and Q.TEMPLATE_HASHES["qwen_sample_v2"] == man["template_hash_qwen_sample_v2"] and V2B.TEMPLATE_HASH == man["template_hash_qwen_sample_v2b"]
    assert len(V2B.TEMPLATES_BY_SAMPLE) == K == 4
    TASKS = {}
    for line in open(os.path.join(BUNDLE, "tasks_public.jsonl"), encoding="utf-8"):
        d = json.loads(line); TASKS[d["task_id"]] = d
    assert len(TASKS) == 80, len(TASKS)
    def prompt_of(tid, s):
        tk = TASKS[tid]
        return V2B.render(V2B.TEMPLATES_BY_SAMPLE[s], task_prompt=tk["prompt"], examples=Q.format_examples(tk["public_tests"]), solution_format=Q.SOLUTION_FORMATS[tk["language"]], language=tk["language"])
    for tid, h in man["smoke_prompt_sha256"].items(): assert prompt_of(tid, 0)[1] == h, "prompt hash mismatch " + tid
    SYSTEM = open(os.path.join(BUNDLE, "gate/system_block.txt"), encoding="utf-8").read()
    assert sha256_text(SYSTEM) == LF.SYSTEM_BLOCK_SHA256
    # ---- model dir + shard sha256
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs: finish("FAILED_NO_MODEL")
    md = dirs[0]; RS["model_dir"] = md; t = time.time()
    ok = {f: sha_file(os.path.join(md, f)) == h for f, h in QWEN_SHA.items()}
    RS["sha256_ok"] = ok; RS["phases"]["sha256"] = round(time.time() - t, 1); save()
    if not all(ok.values()): finish("FAILED_MODEL_SHA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = time.time(); tok = AutoTokenizer.from_pretrained(md); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float16).eval().to("cuda")   # same call as kaggle/level12_t4.py (dtype=float16, then .to cuda)
    RS["phases"]["load"] = round(time.time() - t, 1); save()
    eos = model.generation_config.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])

    # ---- the TRAINING capture recipe (capture/capture_resident.py, level12_t4.py): hooks live only inside capture(); gqa patch scoped to capture()
    from capture.capture_resident import ResidentCapturer
    import transformers.integrations.sdpa_attention as SDPA
    class GatedCapturer(ResidentCapturer):
        active = False
        def _post(self, i):
            inner = super()._post(i)
            def post(mod, args, kwargs, o):
                return inner(mod, args, kwargs, o) if self.active else o
            return post
        def run(self, *a, **k):
            orig = SDPA.use_gqa_in_sdpa; self.active = True
            if self.no_gqa: SDPA.use_gqa_in_sdpa = lambda *x, **y: False
            try: return super().run(*a, **k)
            finally: self.active = False; SDPA.use_gqa_in_sdpa = orig
    C = GatedCapturer(model, DEP, no_gqa=False); C.no_gqa = torch.cuda.get_device_capability(0)[0] < 8
    RS["capture_no_gqa_patch"] = bool(C.no_gqa)

    def capture(context_text, options):
        """-> (hq [3,H] fp16 cpu, ho [N,3,H] fp16 cpu, info). info.status != 'ok' -> (None, None, info). Nothing is clipped."""
        pre = tok.encode(context_text + CUE, add_special_tokens=False); suf = [tok.encode(OPT_PREFIX + o, add_special_tokens=False) for o in options]
        info = dict(prefix_tokens=len(pre), option_tokens=[len(s) for s in suf])
        if not options or len(options) > LIM["max_options"]: return None, None, dict(info, status="excluded_options_count")
        if len(pre) > LIM["max_prefix_tokens"]: return None, None, dict(info, status="excluded_prefix_over_limit")
        if any(len(s) > LIM["max_option_tokens"] or not s for s in suf): return None, None, dict(info, status="excluded_option_over_limit")
        if len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]: return None, None, dict(info, status="excluded_total_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        full = model.model.layers; model.model.layers = torch.nn.ModuleList(list(full)[:max(DEP)])   # TRAIN captures ran the 30-layer truncated model
        t = time.time(); mode = "packed"
        try:
            try:
                cap, _ = C.packed(rec); hq = torch.stack([cap[d][0] for d in DEP]); ho = torch.stack([cap[d][1] for d in DEP], 1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); mode = "plain_per_option"
                parts = [C.plain(rec, i) for i in range(len(suf))]
                hq = torch.stack([parts[0][d][0] for d in DEP]); ho = torch.stack([torch.stack([p[d][1][0] for d in DEP]) for p in parts])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return None, None, dict(info, status="skipped_oom")
        finally: model.model.layers = full
        hq, ho = hq.half().cpu(), ho.half().cpu()
        if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()): return None, None, dict(info, status="nonfinite_fp16")
        return hq, ho, dict(info, status="ok", mode=mode, seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))

    # ---- heads + data for the gate and the scorers
    G = lambda p: os.path.join(BUNDLE, p)
    PRE = G("models/pretrained_31.safetensors")
    heads = {n: LF.build_head(PRE, G("models/%s.pt" % n)) for n in LF.SEEDS}
    rand_head = LF.build_head(PRE, G("models/final-s0.pt"), random_head_seed=0)   # CONTROL: trained adapter s0, random-initialised head
    def load_feats(name):
        return {r["decision_id"]: r for r in torch.load(G("gate/feat_%s_B.pt" % name), map_location="cpu", weights_only=False)["records"]}
    def load_reqs(name):
        return {d["decision_id"]: d for d in (json.loads(l) for l in open(G("gate/req_%s.jsonl" % name), encoding="utf-8"))}
    FE = {n: load_feats(n) for n in ("v2", "v3", "val")}; RQ = {n: load_reqs(n) for n in ("v2", "v3", "val")}
    LAB = json.load(open(G("gate/labels.json"), encoding="utf-8")); REC = json.load(open(G("gate/recorded.json"), encoding="utf-8"))
    TH = tomllib.load(open(G("configs/laya_head_gate_v1.toml"), "rb"))
    def find(did):
        for n in ("v2", "v3", "val"):
            if did in FE[n]: return n
    def row(did, hq, ho, req, lab):
        """Reorder ho from the captured (level-1 request) option order to the label-set option order, as tools/align_transfer_v1.rows_of."""
        ix = {o: i for i, o in enumerate(req["options"])}
        ho2 = torch.stack([ho[ix[o]] for o in lab["options"]])
        return dict(decision_id=did, hq=hq, ho=ho2, ref=lab["reference_index"], pos=torch.tensor([s == "positive" for s in lab["status"]]), trajectory=lab["trajectory"])
    def top_picks(hq, ho):
        return {n: int(max(range(ho.shape[0]), key=lambda i, lg=LF.head_logits(h, hq, ho): lg[i])) for n, h in heads.items()}

    # ================================================================== PARITY GATE
    t = time.time(); gate = {"thresholds_file": "configs/laya_head_gate_v1.toml", "thresholds": TH}
    ids = list(TH["sample"]["train_ids"]) + sorted(REC["decision_id"])
    A_rows, recap = [], {}
    for did in ids:
        n = find(did); sr = FE[n][did]; rq = RQ[n][did]
        if sr["option_sha256"] != rq["option_sha256"] or sr["context_sha256"] != rq["context_sha256"]:
            A_rows.append(dict(decision_id=did, status="stored_vs_request_hash_mismatch")); continue
        hq, ho, info = capture(rq["context_text"], rq["options"])
        if hq is None: A_rows.append(dict(decision_id=did, status=info["status"])); continue
        recap[did] = (hq, ho)
        cos = min(F.cosine_similarity(hq.double(), sr["hq"].double(), dim=-1).min().item(), F.cosine_similarity(ho.double(), sr["ho"].double(), dim=-1).min().item())
        mad = max((hq.float() - sr["hq"].float()).abs().max().item(), (ho.float() - sr["ho"].float()).abs().max().item())
        a = TH["a_feature_recapture"]
        A_rows.append(dict(decision_id=did, set=n, status="ok", cos_min=round(cos, 7), max_abs_diff=mad, identical=bool(torch.equal(hq, sr["hq"]) and torch.equal(ho, sr["ho"])),
                           passed=bool(cos >= a["min_cosine"] and mad <= a["max_abs_diff"]), seconds=info["seconds"], mode=info["mode"], prefix_tokens=info["prefix_tokens"]))
    gate["a_rows"] = A_rows
    gate["a_pass"] = len(A_rows) == len(ids) and all(r["status"] == "ok" and r["passed"] for r in A_rows)
    gate["a_summary"] = dict(n=len(A_rows), ok=sum(r["status"] == "ok" for r in A_rows), worst_cos=min([r["cos_min"] for r in A_rows if r["status"] == "ok"] or [None]),
                             worst_max_abs=max([r["max_abs_diff"] for r in A_rows if r["status"] == "ok"] or [None]), identical=sum(bool(r.get("identical")) for r in A_rows))
    log("GATE A", json.dumps(gate["a_summary"]))
    # B: stored VAL features -> frozen adapters+head -> recorded per-pool head hit
    val_rows, val_rows_re, picks_st, picks_re = [], {}, {}, {}
    for did in REC["decision_id"]:
        lab = LAB["val"][did]; rq = RQ["val"][did]; sr = FE["val"][did]
        r = row(did, sr["hq"], sr["ho"], rq, lab); val_rows.append(r); picks_st[did] = top_picks(r["hq"], r["ho"])
        if did in recap: val_rows_re[did] = row(did, recap[did][0], recap[did][1], rq, lab); picks_re[did] = top_picks(val_rows_re[did]["hq"], val_rows_re[did]["ho"])
    ref = {r["decision_id"]: r["ref"] for r in val_rows}
    head_hit = [sum(float(picks_st[d][n] == ref[d]) for n in LF.SEEDS) / 3 for d in REC["decision_id"]]
    diff_b = [d for d, h, rec in zip(REC["decision_id"], head_hit, REC["head_real"]) if abs(h - rec) > TH["b_stored_feature_forward"]["hit_tolerance"]]
    seed_top1 = {n: sum(float(picks_st[d][n] == ref[d]) for d in REC["decision_id"]) / len(REC["decision_id"]) for n in LF.SEEDS}
    seed_ok = all(abs(seed_top1[n] - REC["seed_top1"][n]) < 5e-5 for n in LF.SEEDS)
    gate["b"] = dict(n=len(head_hit), top1=round(sum(head_hit) / len(head_hit), 4), recorded_top1=REC["head_top1"], pools_differing=len(diff_b), differing=diff_b,
                     per_seed_top1={n: round(v, 4) for n, v in seed_top1.items()}, recorded_per_seed_top1=REC["seed_top1"], per_seed_equal=seed_ok)
    gate["b_pass"] = len(diff_b) <= TH["b_stored_feature_forward"]["max_pools_differing"] and seed_ok
    # C: recaptured features -> same picks
    same = tot = 0
    for d in picks_re:
        for n in LF.SEEDS: tot += 1; same += int(picks_re[d][n] == picks_st[d][n])
    re_hit = [sum(float(picks_re[d][n] == ref[d]) for n in LF.SEEDS) / 3 for d in REC["decision_id"] if d in picks_re]
    c = TH["c_recaptured_feature_forward"]
    re_seed = {n: round(sum(float(picks_re[d][n] == ref[d]) for d in picks_re) / max(len(picks_re), 1), 4) for n in LF.SEEDS}
    gate["c"] = dict(per_seed_top1_recaptured=re_seed, pairs=tot, same_top_choice=round(same / max(tot, 1), 4), recaptured_top1=round(sum(re_hit) / max(len(re_hit), 1), 4), pools=len(re_hit))
    gate["c_pass"] = bool(tot and tot == 3 * len(REC["decision_id"]) and same / tot >= c["min_same_top_choice"] and abs(gate["c"]["recaptured_top1"] - REC["head_top1"]) <= c["max_head_top1_delta"])
    log("GATE B", json.dumps(gate["b"]), "C", json.dumps(gate["c"]))
    # D: probe refit on the TRAIN pools (v2 + v3 captures), recorded VAL probe hits
    train_rows = []
    for n in ("v2", "v3"):
        for did, sr in FE[n].items():
            if did in LAB["v3"]: train_rows.append(row(did, sr["hq"], sr["ho"], RQ[n][did], LAB["v3"][did]))
    gate["d_train_pools"] = len(train_rows)
    PROBE = LF.fit_probe(train_rows, C=TH["d_probe"]["probe_C"])
    probe_hit = [LF.hit(LF.probe_decision(PROBE, r["ho"]), r["ref"]) for r in val_rows]
    diff_d = [d for d, h, rec in zip(REC["decision_id"], probe_hit, REC["probe_real"]) if abs(h - rec) > 1e-9]
    gate["d"] = dict(n=len(probe_hit), top1=round(sum(probe_hit) / len(probe_hit), 4), recorded_probe_top1=REC["probe_top1"], pools_differing=len(diff_d), differing=diff_d)
    gate["d_pass"] = len(diff_d) <= TH["d_probe"]["max_pools_differing"] and len(train_rows) == REC["train_pools"]
    rh = [LF.hit(LF.head_logits(rand_head, r["hq"], r["ho"]), r["ref"]) for r in val_rows]
    gate["random_head_control_val_top1"] = round(sum(rh) / len(rh), 4)   # informational, not a criterion
    log("GATE D", json.dumps(gate["d"]), "random-head VAL top1", gate["random_head_control_val_top1"])
    gate["pass"] = bool(gate["a_pass"] and gate["b_pass"] and gate["c_pass"] and gate["d_pass"]); gate["seconds"] = round(time.time() - t, 1)
    RS["gate"] = {k: v for k, v in gate.items() if k != "thresholds"}; RS["phases"]["gate"] = gate["seconds"]
    with open(OUT + "/GATE.json", "w", encoding="utf-8") as f: json.dump(gate, f, indent=1, default=str)
    save()
    if not gate["pass"]: finish("GATE_FAILED", gate_criteria={k: gate[k] for k in ("a_pass", "b_pass", "c_pass", "d_pass")})
    if MODE == "parity": finish("GATE_PASSED_PARITY_ONLY", gate_criteria={k: gate[k] for k in ("a_pass", "b_pass", "c_pass", "d_pass")})
    del FE, RQ, val_rows, val_rows_re, train_rows, recap
    torch.cuda.empty_cache()

    # ================================================================== GENERATION + POOL SCORING
    py = sorted(x for x in TASKS if TASKS[x]["language"] == "python"); jv = sorted(x for x in TASKS if TASKS[x]["language"] == "java")
    pairs = list(zip(py, jv))[shard_i::shard_n]
    order = [x for pr in pairs for x in pr] + (py[len(jv):] + jv[len(py):] if shard_i == 0 else [])
    RS["shard_tasks"] = order; items = [(tid, s) for tid in order for s in range(K)]
    CP, PP, SP = OUT + "/completions.jsonl", OUT + "/pools.jsonl", OUT + "/sample_scores.jsonl"; os.makedirs(OUT + "/features", exist_ok=True)
    def read_jsonl(p):
        out = []
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                try: out.append(json.loads(ln))
                except Exception: pass
        return out
    comps = {(r["task_id"], r["sample_idx"]): r for r in read_jsonl(CP)}; pooled = {r["task_id"] for r in read_jsonl(PP)}
    todo = [it for it in items if it not in comps]; RS["todo"] = len(todo); RS["items"] = len(items); save()

    def gen_batch(sub, ordinal):
        chats = [tok.apply_chat_template([{"role": "user", "content": prompt_of(i, s)[0]}], tokenize=False, add_generation_prompt=True) for i, s in sub]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device); n_in = enc["input_ids"].shape[1]
        plen = [int(m.sum()) for m in enc["attention_mask"]]; seed = 1000 + ordinal
        torch.manual_seed(seed); torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(): ids = model.generate(**enc, do_sample=True, max_new_tokens=MAXNEW, pad_token_id=tok.eos_token_id, **Q.SAMPLING)
        torch.cuda.synchronize(); wall = time.time() - t; new = ids[:, n_in:]; recs = []
        for k, (tid, s) in enumerate(sub):
            seq = new[k].tolist(); n = next((j + 1 for j, v in enumerate(seq) if v in eos), len(seq)); hit = n >= MAXNEW and seq[n - 1] not in eos
            text = tok.decode(seq[:n], skip_special_tokens=True); code, st = Q.extract_code(text, TASKS[tid]["language"]); tmpl = V2B.TEMPLATES_BY_SAMPLE[s]
            recs.append(dict(task_id=tid, language=TASKS[tid]["language"], sample_idx=s, stage="larger", template=tmpl, prompt_sha256=prompt_of(tid, s)[1],
                             model=Q.QWEN_REPO, model_revision=Q.QWEN_REVISION, tokens_prompt=plen[k], tokens_generated=n, max_new_tokens=MAXNEW, finish_reason="length" if hit else "stop",
                             hit_max_new_tokens=bool(hit), thinking_closed="</think>" in text, extract_status=st, text=text, text_sha256=sha256_text(text),
                             chunk_wall_seconds=round(wall, 1), chunk_size=len(sub), top_k_applied=True, temperature=0.6, top_p=0.95, engine_seed=seed,
                             backend="gpu_hf_t4", shard=SHARD, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        return recs, wall

    def score_pool(tid, final=False):
        """ONE packed capture of the task's distinct candidate programs (code only), then the three scorers. Always appends a pools.jsonl record (scored or not)."""
        tk = TASKS[tid]; ss = sorted((s for (t_, s) in comps if t_ == tid))
        samples = [comps[(tid, s)] for s in ss]; good = [r for r in samples if r["extract_status"] == "ok"]
        rec = dict(task_id=tid, language=tk["language"], label=LF.EXPLORATORY_LABEL, candidate_summary="code only (final fenced block after </think>), no reasoning, as a Write tool call",
                   decision_rendering="laya_forward.decision_request (training system block + task statement + public examples + solution format; options = distinct candidate programs; no NONE)",
                   samples_present=ss, partial_pool=len(ss) < K, templates={s: comps[(tid, s)]["template"] for s in ss}, status="not_scored")
        sample_rows = [dict(task_id=tid, sample_idx=r["sample_idx"], candidate_source_sha256=None, scores={}) for r in samples]
        if len(good) < 2:
            rec["reason"] = "fewer_than_2_samples_with_code"
        else:
            req = LF.decision_request(SYSTEM, tk, [Q.extract_code(r["text"], tk["language"])[0] for r in good], Q)
            n_opt = len(req["options"])
            if n_opt < 2:
                rec["reason"] = "fewer_than_2_distinct_candidates"
            else:
                hq, ho, info = capture(req["context_text"], req["options"]); rec["capture"] = info
                if hq is None:
                    rec["reason"] = info["status"]
                else:
                    fp = OUT + "/features/%s.pt" % tid; tmp = fp + ".tmp"
                    torch.save(dict(protocol="fable_capture_v1", decision_id=tid, depths=DEP, dtype="float16", options=req["options"], context_sha256=sha256_text(req["context_text"]),
                                    option_sha256=[sha256_text(o) for o in req["options"]], hq=hq, ho=ho), tmp)
                    with open(tmp, "rb") as fh: os.fsync(fh.fileno())
                    os.replace(tmp, fp); rec["features_file"] = "features/%s.pt" % tid; rec["features_sha256"] = sha_file(fp)
                    per_seed = {n: LF.head_logits(h, hq, ho) for n, h in heads.items()}
                    laya_mean = [sum(per_seed[n][i] for n in LF.SEEDS) / 3 for i in range(n_opt)]
                    probe = LF.probe_decision(PROBE, ho); rnd = LF.head_logits(rand_head, hq, ho)
                    rec["status"] = "scored"; rec["n_distinct_options"] = n_opt; rec["option_sha256"] = [sha256_text(o) for o in req["options"]]
                    rec["scores"] = {"laya": {"per_seed_logits": per_seed, "score": laya_mean, "note": "score = mean of the three single-adapter logits; each seed listed; not an ensemble model"},
                                     "probe": {"score": probe}, "random_head": {"score": rnd, "note": "trained adapter final-s0 + random-initialised head (control)"}}
                    gi = {id(r): req["index_of_code"][k] for k, r in enumerate(good)}
                    for sr_, r in zip(sample_rows, samples):
                        if id(r) in gi:
                            j = gi[id(r)]
                            sr_["option_index"] = j
                            sr_["scores"] = {"laya": {"status": "ok", "score": laya_mean[j]}, "probe": {"status": "ok", "score": probe[j]}, "random_head": {"status": "ok", "score": rnd[j]}}
        for sr_, r in zip(sample_rows, samples):
            c_, st_ = Q.extract_code(r["text"], tk["language"])
            sr_["candidate_source_sha256"] = sha256_text(c_) if st_ == "ok" else None; sr_["template"] = r["template"]; sr_["extract_status"] = st_
            if not sr_["scores"]: sr_["scores"] = {k: {"status": "not_scored", "reason": rec.get("reason", "no_code")} for k in ("laya", "probe", "random_head")}
        rec["samples"] = sample_rows
        append_fsync(PP, [rec]); append_fsync(SP, sample_rows); pooled.add(tid); RS["pools_done"] += 1; RS.setdefault("pool_status", {})[tid] = rec["status"]; save()
        log("POOL", tid, rec["status"], rec.get("reason", ""), rec.get("capture", {}).get("seconds"))

    def pools_ready(final=False):
        for tid in order:
            if tid in pooled: continue
            n = sum(1 for s in range(K) if (tid, s) in comps)
            if n == K or (final and n >= 1): score_pool(tid, final)

    def write(recs):
        append_fsync(CP, recs)
        for r in recs: comps[(r["task_id"], r["sample_idx"])] = r
        RS["samples_done"] += len(recs); save()
    def run(sub, ordinal):             # OOM -> split this batch only (3 -> 2+1 -> 1)
        try:
            recs, wall = gen_batch(sub, ordinal); write(recs); log("batch", sub, "wall %.0fs tokens %s" % (wall, [r["tokens_generated"] for r in recs]))
            RS.setdefault("batch_log", []).append({"items": sub, "wall_s": round(wall, 1), "tokens": [r["tokens_generated"] for r in recs]})
        except torch.OutOfMemoryError:
            RS["oom_events"].append({"items": sub, "t": round(time.time() - T0)}); log("OOM", sub)
            import gc; gc.collect(); torch.cuda.empty_cache()
            if len(sub) == 1: RS.setdefault("failed_items", []).append(sub[0]); save(); return
            h = 2 if len(sub) > 2 else 1; run(sub[:h], ordinal); run(sub[h:], ordinal + h)
    pools_ready()   # tasks already complete from an earlier session
    stopped = False
    for i in range(0, len(todo), BATCH):
        if (time.time() - T0) / 3600 > BUDGET_H: stopped = True; RS["notes"].append("time guard stopped new batches before item %d of %d" % (i, len(todo))); break
        run(todo[i:i + BATCH], items.index(todo[i])); pools_ready(); RS["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    pools_ready(final=True)
    RS["seconds_per_sample_generation"] = round(sum(b["wall_s"] for b in RS.get("batch_log", [])) / max(RS["samples_done"], 1), 1)
    n = len(comps)
    finish("OK" if n >= len(items) else ("PARTIAL_TIME_GUARD" if stopped else "PARTIAL"), completions=n, expected=len(items))
except BaseException as e:
    RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc(); finish("FAILED")
