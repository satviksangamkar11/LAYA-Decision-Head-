"""[LAYA-INT-1 · Aim: integration, not performance] T6 step kernel (Kaggle Tesla T4, fp16, internet OFF). For each step request: Qwen3-4B-Thinking-2507 PROPOSES 3-5 next actions
(greedy, thinking closed, or REPLAY of a saved raw output), the options are captured WITH THE TRAINING RECIPE from the frozen Qwen in the same process, and the 3 frozen adapters + pretrained
head (plus the probe and random-head controls) score them. Nothing is executed here; the PC runs the allowed action in the WSL sandbox between steps.
The request is embedded below by tools/laya_int1_t6.py (placeholder replaced at pack time). Output: /kaggle/working/t6_step_results.json (create-only per run)."""
import glob, hashlib, json, os, sys, time, traceback

T0 = time.time(); OUT = "/kaggle/working"
REQUEST = json.loads(r'''{"run_id": "e2e/verify_s1", "mode": "replay", "steps": [{"key": "ex1_slug:1", "step": 1, "context_text": "[system]\n[system]\nYou are an autonomous coding agent running on the user's machine.\n\nYou work in the current directory of a real repository. Use your tools to read, search, create, and edit files and to run shell commands (tests, builds, type-checkers). Some tools are deferred: only their names are known until you load their schemas with ToolSearch. Depending on the environment, some tools may be unavailable or without network access; if a tool call fails, adapt and continue with what is available. Never guess at the contents of a file you have not read, and never claim a fix works without running the verification yourself.\n\nMethod:\n1. Reproduce first. Read the relevant code and run the failing command before changing anything.\n2. Form a hypothesis about the root cause; make the smallest edit that tests it.\n3. Verify. Rerun the tests/build after every meaningful change. If it still fails, re-read the output carefully — do not repeat the same edit.\n4. Fix the cause, not the symptom.\n\nRules:\n- Never modify tests to make them pass, unless the user explicitly says the tests are wrong.\n- Keep edits minimal and consistent with the existing code style.\n- End with a brief summary: the root cause, what you changed, and proof that it passes.\n\n[user]\nThe function slugify in slug.py returns slugs with leading or trailing hyphens, so test_slug.py fails. Fix slug.py so that `python3 test_slug.py` passes. Do not modify test_slug.py.", "saved_raw_output": "[{\"name\": \"Read\", \"arguments\": {\"file_path\": \"slug.py\"}}, {\"name\": \"Read\", \"arguments\": {\"file_path\": \"test_slug.py\"}}, {\"name\": \"Bash\", \"arguments\": {\"command\": \"python3 test_slug.py\"}}, {\"name\": \"Edit\", \"arguments\": {\"file_path\": \"slug.py\", \"old_string\": \"def slugify(text):\", \"new_string\": \"def slugify(text):\\n    # Remove leading/trailing hyphens\\n    text = text.strip('-')\\n    # ... rest of the function\"}}]"}, {"key": "ex2_clamp:1", "step": 1, "context_text": "[system]\n[system]\nYou are an autonomous coding agent running on the user's machine.\n\nYou work in the current directory of a real repository. Use your tools to read, search, create, and edit files and to run shell commands (tests, builds, type-checkers). Some tools are deferred: only their names are known until you load their schemas with ToolSearch. Depending on the environment, some tools may be unavailable or without network access; if a tool call fails, adapt and continue with what is available. Never guess at the contents of a file you have not read, and never claim a fix works without running the verification yourself.\n\nMethod:\n1. Reproduce first. Read the relevant code and run the failing command before changing anything.\n2. Form a hypothesis about the root cause; make the smallest edit that tests it.\n3. Verify. Rerun the tests/build after every meaningful change. If it still fails, re-read the output carefully — do not repeat the same edit.\n4. Fix the cause, not the symptom.\n\nRules:\n- Never modify tests to make them pass, unless the user explicitly says the tests are wrong.\n- Keep edits minimal and consistent with the existing code style.\n- End with a brief summary: the root cause, what you changed, and proof that it passes.\n\n[user]\nThe function clamp in clamp.py should limit x to the range [lo, hi] but returns wrong values, so test_clamp.py fails. Fix clamp.py so that `python3 test_clamp.py` passes. Do not modify test_clamp.py.", "saved_raw_output": "[{\"name\": \"Read\", \"arguments\": {\"file_path\": \"clamp.py\"}}, {\"name\": \"Read\", \"arguments\": {\"file_path\": \"test_clamp.py\"}}, {\"name\": \"Bash\", \"arguments\": {\"command\": \"python3 test_clamp.py\"}}, {\"name\": \"Edit\", \"arguments\": {\"file_path\": \"clamp.py\", \"old_string\": \"def clamp(x, lo, hi):\", \"new_string\": \"def clamp(x, lo, hi):\\n    return max(lo, min(x, hi))\"}}]"}, {"key": "ex3_wordcount:1", "step": 1, "context_text": "[system]\n[system]\nYou are an autonomous coding agent running on the user's machine.\n\nYou work in the current directory of a real repository. Use your tools to read, search, create, and edit files and to run shell commands (tests, builds, type-checkers). Some tools are deferred: only their names are known until you load their schemas with ToolSearch. Depending on the environment, some tools may be unavailable or without network access; if a tool call fails, adapt and continue with what is available. Never guess at the contents of a file you have not read, and never claim a fix works without running the verification yourself.\n\nMethod:\n1. Reproduce first. Read the relevant code and run the failing command before changing anything.\n2. Form a hypothesis about the root cause; make the smallest edit that tests it.\n3. Verify. Rerun the tests/build after every meaningful change. If it still fails, re-read the output carefully — do not repeat the same edit.\n4. Fix the cause, not the symptom.\n\nRules:\n- Never modify tests to make them pass, unless the user explicitly says the tests are wrong.\n- Keep edits minimal and consistent with the existing code style.\n- End with a brief summary: the root cause, what you changed, and proof that it passes.\n\n[user]\nThe function count_words in wc.py miscounts words when the text has repeated spaces or newlines, so test_wc.py fails. Fix wc.py so that `python3 test_wc.py` passes. Do not modify test_wc.py.", "saved_raw_output": "[{\"name\": \"Read\", \"arguments\": {\"file_path\": \"wc.py\"}}, {\"name\": \"Read\", \"arguments\": {\"file_path\": \"test_wc.py\"}}, {\"name\": \"Bash\", \"arguments\": {\"command\": \"python3 test_wc.py\"}}, {\"name\": \"Edit\", \"arguments\": {\"file_path\": \"wc.py\", \"old_string\": \"words = text.split()\", \"new_string\": \"words = re.split(r'\\\\s+', text)\"}}]\n</think>\n\nWe are given a problem: the function `count_words` in `wc.py` miscounts words when the text has repeated spaces or newlines, so `test_wc.py` fails.\n\nThe agent must propose exactly 4 DISTINCT candidate next actions for the agent, as a JSON list of objects.\n\nThe candidate actions should be the next steps the agent would take to debug and fix the problem.\n\nLet's think step by step:\n\n1. The agent should first reproduce the problem by reading the failing code and running the test.\n   - So, the first candidate action is to read `wc.py` to understand the current implementation.\n   - The second candidate action is to read `test_wc.py` to see what test is failing (though the problem says the test fails, so we want to know the exact test case).\n   - The third candidate action is to run the test to see the failure (so we can see the output and understand the problem).\n   - The fourth candidate action is to make a minimal edit to the `wc.py` function to fix the problem. The problem says the issue is with repeated spaces and newlines, so the fix likely involves splitting the text in a way that handles multiple whitespace characters (like using `re.split` with a regex that matches one or more whitespace).\n\nHowever, note the problem says: \"Propose exactly 4 DISTINCT candidate next actions for the agent\".\n\nWe must output a JSON list of 4 distinct actions.\n\nLet's consider the tools we have:\n\n- Read: for a file_path\n- Edit: for a file_path, old_string, new_string\n- Bash: for a command\n\nThe problem says: \"Do not modify test_wc.py\", so we won't propose an edit to test_wc.py.\n\nThe agent's method says:\n1. Reproduce first: so read the code and run the failing command.\n\nSo the first actions should be to read the files and run the test.\n\nBut note: the problem says the test fails, so we want to see the failure.\n\nWe are to propose the next actions (the very next steps the agent would take).\n\nGiven the above, the 4 candidate actions could be:\n\n1. Read the `wc.py` file to see the current implementation of `count_words`.\n2. Read the `test_wc.py` file to see what test is failing (so we know the exact input that causes the problem).\n3. Run the test with `python3 test_wc.py` to see the failure output (this is critical to understand the problem).\n4. Then, a candidate edit to `wc.py` that uses a regex split to handle multiple whitespace (this is the fix we expect).\n\nBut note: the problem says the agent must not guess at the contents of a file it hasn't read. So the agent would first read the files.\n\nAlso, the problem says: \"Never claim a fix works without running the verification\". So the agent would run the test after"}], "max_new_tokens": 700, "with_controls": true}''')
TAG = "[LAYA-INT-1 · Aim: integration, not performance]"
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
LIM = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8)
DEP = [18, 24, 30]; CUE = "\n\nNext action:"; OPT_PREFIX = " "
INSTR = ("Propose exactly 4 DISTINCT candidate next actions for the agent, as ONE JSON list. Each item is an object {\"name\": <tool>, \"arguments\": {...}} with tool one of "
         "Read {\"file_path\"}, Edit {\"file_path\",\"old_string\",\"new_string\"} or Bash {\"command\"}. Output only the JSON list.")
TOOLS = {"Read": {"file_path"}, "Edit": {"file_path", "old_string", "new_string"}, "Bash": {"command"}}
RS = {"kernel": "evidence_ladder_laya_t6_step_v1", "tag": TAG, "run_id": REQUEST.get("run_id"), "mode": REQUEST["mode"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def save(name="t6_step_results.json"):
    with open(OUT + "/" + name, "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
def finish(verdict, **kw):
    RS["verdict"] = verdict; RS["total_seconds"] = round(time.time() - T0, 1); RS.update(kw); save(); save("RUN_SUMMARY.json"); log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)
def sha_file(p):
    x = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
    return x.hexdigest()

try:
    import torch, transformers
    RS["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available(): finish("FAILED_NO_CUDA")
    DEVNAME = torch.cuda.get_device_name(0); RS["gpu"] = DEVNAME
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_laya_int1_v1"), None)
    if not mp: finish("FAILED_NO_BUNDLE")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or sha_file(os.path.join(BUNDLE, f)) != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad)
    sys.path.insert(0, BUNDLE)
    from evidence_ladder.models import laya_forward as LF, laya_runtime as RT
    from decision_head import fable_pool as FP
    G = lambda p: os.path.join(BUNDLE, p)
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs: finish("FAILED_NO_MODEL")
    md = dirs[0]; ok = {f: sha_file(os.path.join(md, f)) == h for f, h in QWEN_SHA.items()}; RS["qwen_sha256_ok"] = ok
    if not all(ok.values()): finish("FAILED_MODEL_SHA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float16).eval().to("cuda")
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
    LAST = {}
    def capture(context_text, options):
        pre = tok.encode(context_text + CUE, add_special_tokens=False); suf = [tok.encode(OPT_PREFIX + o, add_special_tokens=False) for o in options]
        info = dict(prefix_tokens=len(pre), option_tokens=[len(s) for s in suf])
        if not options or len(options) > LIM["max_options"] or len(pre) > LIM["max_prefix_tokens"] or any(len(s) > LIM["max_option_tokens"] or not s for s in suf) \
                or len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]: return None, None, dict(info, status="excluded_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        full = model.model.layers; model.model.layers = torch.nn.ModuleList(list(full)[:max(DEP)]); t = time.time(); torch.cuda.reset_peak_memory_stats()
        try:
            cap, _ = C.packed(rec); hq = torch.stack([cap[d][0] for d in DEP]); ho = torch.stack([cap[d][1] for d in DEP], 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return None, None, dict(info, status="skipped_oom")
        finally: model.model.layers = full
        hq, ho = hq.half().cpu(), ho.half().cpu()
        if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()): return None, None, dict(info, status="nonfinite_fp16")
        LAST["feat_sha256"] = hashlib.sha256(hq.contiguous().view(torch.uint8).numpy().tobytes() + ho.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return hq, ho, dict(info, status="ok", mode="packed", seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), depth_keys=sorted(cap))
    PRE = G("models/pretrained_31.safetensors")
    heads = {n: LF.build_head(PRE, G("models/%s.pt" % n)) for n in LF.SEEDS}
    probe = rand_head = None
    if REQUEST.get("with_controls", True):
        LAB = json.load(open(G("gate/labels.json"), encoding="utf-8"))
        rows = []
        for nm in ("v2", "v3"):
            fe = {r["decision_id"]: r for r in torch.load(G("gate/feat_%s_B.pt" % nm), map_location="cpu", weights_only=False)["records"]}
            rq = {d["decision_id"]: d for d in (json.loads(l) for l in open(G("gate/req_%s.jsonl" % nm), encoding="utf-8"))}
            for did, sr in fe.items():
                if did in LAB["v3"]:
                    lab = LAB["v3"][did]; ix = {o: i for i, o in enumerate(rq[did]["options"])}
                    rows.append(dict(ho=torch.stack([sr["ho"][ix[o]] for o in lab["options"]]), pos=torch.tensor([s == "positive" for s in lab["status"]])))
        probe = LF.fit_probe(rows, C=0.01); rand_head = LF.build_head(PRE, G("models/final-s0.pt"), random_head_seed=0); RS["probe_train_pools"] = len(rows)
    FS = RT.FeatureScorer(heads, probe=probe, random_head=rand_head); live = RT.LiveScorer(capture, FS, DEVNAME)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    def propose(state_text):
        msgs = [{"role": "user", "content": state_text + "\n\n" + INSTR}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not p.rstrip().endswith("</think>"): p = p + "\n</think>\n\n"            # thinking closed empty (Thinking-2507 always opens a block): direct answer
        PREFILL = '[{"name": "'; p = p + PREFILL      # answer prefilled with the start of the first JSON action (attempts 1-2: prose / "[" alone produced 0 proposals)
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda"); t = time.time()
        with torch.no_grad(): ids = model.generate(**enc, do_sample=False, max_new_tokens=int(REQUEST.get("max_new_tokens", 700)), pad_token_id=tok.eos_token_id)
        new = ids[0, enc["input_ids"].shape[1]:].tolist()
        return PREFILL + tok.decode(new, skip_special_tokens=True), {"prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": len(new), "seconds": round(time.time() - t, 1), "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}

    def parse(raw):
        i = raw.find("["); items, dropped = [], 0
        try: arr, _ = json.JSONDecoder().raw_decode(raw[i:]) if i >= 0 else ([], 0)
        except ValueError: arr = []
        for a in arr if isinstance(arr, list) else []:
            if isinstance(a, dict) and a.get("name") in TOOLS and isinstance(a.get("arguments"), dict) and set(a["arguments"]) == TOOLS[a["name"]] and all(isinstance(v, str) for v in a["arguments"].values()):
                s = FP.render_call(a["name"], a["arguments"])
                if s not in items: items.append(s)
                else: dropped += 1
            else: dropped += 1
        return items[:5], dropped

    for st in REQUEST["steps"]:
        k = st.get("key", str(st["step"])); r = {"step": st["step"], "context_sha256": LF.sha256_text(st["context_text"])}
        if REQUEST["mode"] == "replay" and st.get("saved_raw_output") is not None:
            raw, gi = st["saved_raw_output"], {"replayed": True}
        else:
            raw, gi = propose(st.get("state_text") or st["context_text"])
        opts, dropped = parse(raw); r.update(raw_output=raw, raw_output_sha256=LF.sha256_text(raw), generation=gi, options=opts, option_sha256=[LF.sha256_text(o) for o in opts], dropped_proposals=dropped)
        if len(opts) < 3:
            r["status"] = "too_few_proposals"; RS["steps"][k] = r; save(); log("STEP", k, "too few proposals"); continue
        out = live.score_texts(st["context_text"], opts)
        r.update(status="ok", scores=out["scores"], provenance=out["provenance"], origin=out["origin"], capture=out["capture"], features_sha256=LAST["feat_sha256"], score_seconds=out["seconds"])
        RS["steps"][k] = r; save(); log("STEP", k, "options", len(opts), "top", max(range(len(opts)), key=out["scores"]["laya"]["score"].__getitem__))
    finish("DONE")
except BaseException as e:
    RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc(); finish("FAILED")
