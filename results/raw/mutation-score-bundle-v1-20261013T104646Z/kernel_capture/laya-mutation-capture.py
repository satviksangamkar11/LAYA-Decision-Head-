"""Kaggle T4 kernel, ONE session, TWO separate experiments (user plan 2026-10-11). No labels are in the input; nothing is trained.

LEVEL 1 - frozen-Qwen feature capture for the frozen full-action pools (fable_capture_v1 representation: hq = residual after layer d
at the last prefix token, ho = fp32 mean over each option's own tokens, fp16 storage, options as independent suffixes). Variants:
  B real pre-decision context          S context of ANOTHER STATE of the same trajectory (same project and author, other evidence)
  X context of another trajectory      A neutral one-line prefix (options only)
Nothing is truncated; a request over a limit is excluded with a reason; packed vs plain parity on the first 5 and the 5 longest.

LEVEL 2 - candidate generation by the SAME frozen Qwen for about 100 trajectory-separated decisions: the request is the existing
blind next-turn prompt (full pre-decision context, tool list; Fable's next action is not in it), N samples per decision, raw text
kept. Decoding, recall, validity and sandbox outcomes are computed later on the user's machine. The sample is not used to tune anything.
Two GPUs: Level 1 runs on GPU 0, then one Level 2 worker per GPU. A wall-clock budget stops Level 2 cleanly; finished decisions are
saved as they complete.

Outputs: /kaggle/working/out/level1/{B,S,X,A}.pt + meta.json ; /kaggle/working/out/level2/gen-<worker>.jsonl + meta-<worker>.json
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

ROOT_IN, OUT = "/kaggle/input", "/kaggle/working/out"
VERSION = "2026-10-12a"   # the dataset copy (level12_worker.py) must carry the same string
RUN_LEVEL1 = True          # False: Level 2 only (Level 1 already captured by kernel version 2)
RUN_LEVEL2 = False         # False: Level 1 only
VARIANTS = "BA"            # which context variants Level 1 captures; "A" = the short no-context capture (user GO 2026-10-12 for Stage 1)
L2_PARITY = None           # None: every Level 2 row; 0 or 1: only rows with that index parity (to finish a half another run lost)
T0 = float(os.environ.get("L2_T0") or time.time())   # workers share the kernel's start time, so the wall budget is global


def find(pred):
    return [os.path.join(r, f) for r, _, fs in os.walk(ROOT_IN, followlinks=True) for f in fs if pred(os.path.join(r, f))]


def load_rows(stem):
    p = [x for x in find(lambda q: ".jsonl" in q) if stem in x][0]
    return [json.loads(l) for l in (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8")]


CODE = os.path.dirname(find(lambda q: q.endswith("level12_config.json"))[0])
CFG = json.load(open(os.path.join(CODE, "level12_config.json")))
PROTO, LIM = CFG["protocol"], CFG["limits"]
DEP = PROTO["depths"]


def qwen_dir():
    q = [os.path.dirname(c) for c in find(lambda p: p.endswith("config.json")) if os.path.exists(os.path.join(os.path.dirname(c), "model.safetensors.index.json"))]
    if q:
        return q[0]
    from huggingface_hub import snapshot_download   # pinned revision, inside Kaggle
    return snapshot_download("Qwen/Qwen3-4B-Thinking-2507", revision=PROTO["revision"], local_dir="/tmp/qwen")


# ------------------------------------------------------------------ LEVEL 1
def level1():
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pkg = "/kaggle/working/pkg/capture"
    os.makedirs(pkg, exist_ok=True)
    for n in ("capture.py", "capture_resident.py"):
        shutil.copy(os.path.join(CODE, n), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, "/kaggle/working/pkg")
    from capture.capture_resident import ResidentCapturer
    reqs = load_rows("requests_level1")
    ctx_of = {r["decision_id"]: r["context_text"] for r in reqs}
    qd = qwen_dir()
    tok = AutoTokenizer.from_pretrained(qd)
    model = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float16).eval()
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(DEP)])
    C = ResidentCapturer(model.to("cuda"), DEP)

    def prefix(r, v):
        return {"B": r["context_text"], "S": r["same_trajectory_context"], "X": ctx_of[r["other_trajectory_from"]], "A": CFG["neutral_prefix"]}[v] + PROTO["query_cue"]

    def capture(r, v, parity=False):
        if v == "S" and not r.get("same_trajectory_context"):
            return None, dict(decision_id=r["decision_id"], variant=v, status="not_available")
        pre = tok.encode(prefix(r, v), add_special_tokens=False)
        suf = [tok.encode(PROTO["option_prefix"] + o, add_special_tokens=False) for o in r["options"]]
        base = dict(decision_id=r["decision_id"], variant=v, prefix_tokens=len(pre), option_tokens=[len(s) for s in suf])
        if len(suf) > LIM["max_options"] or len(pre) > LIM["max_prefix_tokens"] or any(len(s) > LIM["max_option_tokens"] or not s for s in suf) \
                or len(pre) + sum(map(len, suf)) > LIM["max_total_tokens"]:
            return None, dict(base, status="excluded_over_limit")
        rec = dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1)
        torch.cuda.reset_peak_memory_stats()
        t, mode, cap = time.time(), "packed", None
        try:
            cap, _ = C.packed(rec)
            hq = torch.stack([cap[d][0] for d in DEP])
            ho = torch.stack([cap[d][1] for d in DEP], 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mode = "plain_per_option"
            try:
                parts = [C.plain(rec, i) for i in range(len(suf))]
                hq = torch.stack([parts[0][d][0] for d in DEP])
                ho = torch.stack([torch.stack([p[d][1][0] for d in DEP]) for p in parts])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None, dict(base, status="skipped_oom")
        info = dict(base, status="ok", mode=mode, seconds=round(time.time() - t, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                    hq_shape=list(hq.shape), ho_shape=list(ho.shape), max_abs=float(max(hq.abs().max(), ho.abs().max())),
                    finite_fp16=bool(torch.isfinite(hq.half()).all() and torch.isfinite(ho.half()).all()))
        if parity and cap is not None:
            cap2, _ = C.packed(rec)
            info["repeat_identical"] = all(torch.equal(cap[d][0], cap2[d][0]) and torch.equal(cap[d][1], cap2[d][1]) for d in DEP)
            cos = 1.0
            for i in range(len(suf)):
                cp = C.plain(rec, i)
                for d in DEP:
                    cos = min(cos, F.cosine_similarity(cp[d][1][0].double(), cap[d][1][i].double(), 0).item(), F.cosine_similarity(cp[d][0].double(), cap[d][0].double(), 0).item())
            info["plain_cos_min"], info["plain_ok"] = round(cos, 6), cos >= 0.999
        return dict(decision_id=r["decision_id"], context_sha256=r["context_sha256"], option_sha256=r["option_sha256"], order_sha256=r["order_sha256"],
                    hq=hq.half(), ho=ho.half()), info

    VAR = os.environ.get("L12_VARIANTS") or "BSXA"   # set by the parent from its VARIANTS constant
    out = os.path.join(OUT, "level1")
    os.makedirs(out)
    meta = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__, protocol_id=PROTO["id"], depths=DEP, limits=LIM,
                dtype="float16", pools=len(reqs), rows=[])
    check = {r["decision_id"] for r in reqs[:5]} | {r["decision_id"] for r in sorted(reqs, key=lambda r: -len(r["context_text"]))[:5]}
    # in-run validation sample first: abort the bulk capture on a shape, parity, memory or finiteness failure
    for r in [x for x in reqs if x["decision_id"] in check]:
        _, info = capture(r, VAR[0], parity=True)
        print("VALIDATE", json.dumps(info), flush=True)
        meta.setdefault("validation", []).append(info)
        if info["status"] != "ok" or not info.get("plain_ok", True) or not info.get("repeat_identical", True) or not info["finite_fp16"]:
            meta["aborted"] = "validation sample failed"
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
            raise SystemExit("LEVEL 1 ABORTED: validation sample failed")
    for v in VAR:
        recs = []
        for i, r in enumerate(reqs):
            rec, info = capture(r, v)
            meta["rows"].append(info)
            if rec:
                recs.append(rec)
            if i % 25 == 0:
                print(v, i, json.dumps(info), flush=True)
        torch.save(dict(variant=v, protocol=PROTO, limits=LIM, records=recs), os.path.join(out, v + ".pt"))
        print("SAVED", v, len(recs), flush=True)
    meta["lm_head_calls"] = C.lm_calls
    meta["summary"] = {v: {s: sum(1 for x in meta["rows"] if x["variant"] == v and x["status"] == s) for s in ("ok", "excluded_over_limit", "skipped_oom", "not_available")}
                       for v in VAR}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("LEVEL1 DONE", json.dumps(meta["summary"]), flush=True)
    del C, model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ LEVEL 2
def level2_worker(worker, n_workers):
    """Restartable: finished decisions are read back from the worker's file; a decision that crashed the process once is retried one
    sample at a time, and after a second crash it is recorded as `crashed` and skipped (2026-10-11: version 2 lost a whole worker to one
    CUDA error at its first generation; the parent process had kept Level 1's model on GPU 0)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    G = CFG["level2"]
    par = os.environ.get("L2_PARITY")
    rows = [r for i, r in enumerate(load_rows("requests_level2")) if par in (None, "") or i % 2 == int(par)]
    rows = [r for i, r in enumerate(rows) if i % n_workers == worker]
    out = os.path.join(OUT, "level2")
    os.makedirs(out, exist_ok=True)
    gen, att, mp = (os.path.join(out, n % worker) for n in ("gen-%d.jsonl", "attempts-%d.jsonl", "meta-%d.json"))
    done = {json.loads(l)["decision_id"] for l in open(gen, encoding="utf-8")} if os.path.exists(gen) else set()
    tried = [json.loads(l)["decision_id"] for l in open(att, encoding="utf-8")] if os.path.exists(att) else []
    meta = json.load(open(mp)) if os.path.exists(mp) else dict(worker=worker, settings=G, parity=par, decisions=len(rows), done=0, restarts=-1, rows=[])
    meta["restarts"] += 1
    qd = qwen_dir()
    tok = AutoTokenizer.from_pretrained(qd)
    model = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float16).eval().to("cuda")
    meta.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__)
    # speed probe (2026-10-11: version 2 needed 0.25-0.70 s per step, growing with prompt length, as if the key-value cache were not used)
    pi = tok("def f(x): return x; " * 300, return_tensors="pt", add_special_tokens=False).to("cuda")
    for uc in (True, False):
        tp = time.time()
        with torch.no_grad():
            po = model.generate(**pi, do_sample=False, max_new_tokens=24, min_new_tokens=24, use_cache=uc, pad_token_id=tok.eos_token_id)
        meta["probe_use_cache_%s" % uc] = dict(prompt_tokens=int(pi["input_ids"].shape[1]), new_tokens=int(po.shape[1] - pi["input_ids"].shape[1]), seconds=round(time.time() - tp, 2))
        print("PROBE", worker, "use_cache", uc, meta["probe_use_cache_%s" % uc], flush=True)
    deadline = T0 + G["wall_budget_seconds"]
    ends = set(tok.convert_tokens_to_ids(["<|im_end|>"])) | {tok.eos_token_id}
    for r in rows:
        did = r["decision_id"]
        if did in done:
            continue
        if time.time() > deadline:
            meta["stopped"] = "wall-clock budget reached; remaining decisions not generated"
            break
        n_try = tried.count(did)
        if n_try >= 2:
            meta["rows"].append(dict(decision_id=did, status="crashed_twice_skipped"))
            done.add(did)
            continue
        with open(att, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(decision_id=did, attempt=n_try + 1)) + "\n")
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        n_in = ids["input_ids"].shape[1]
        if n_in > G["max_prompt_tokens"]:
            meta["rows"].append(dict(decision_id=did, status="excluded_prompt_over_limit", prompt_tokens=n_in))
            continue
        kw = dict(do_sample=True, temperature=G["temperature"], top_p=G["top_p"], top_k=G["top_k"], max_new_tokens=G["max_new_tokens"], pad_token_id=tok.eos_token_id, use_cache=True)
        t, seqs, mode = time.time(), [], "batched"
        torch.manual_seed(G["seed"] + int(r["seed_offset"]))
        try:
            if n_try:
                raise torch.cuda.OutOfMemoryError("retry after a crash: one sample at a time")
            with torch.no_grad():
                seqs = [x[n_in:] for x in model.generate(**ids, num_return_sequences=G["samples"], **kw)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            seqs, mode = [], "one_at_a_time"
            for _ in range(G["samples"]):
                try:
                    with torch.no_grad():
                        seqs.append(model.generate(**ids, num_return_sequences=1, **kw)[0][n_in:])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
        samples = []
        for s_ in seqs:
            keep = [int(x) for x in s_]
            cut = next((j for j, x in enumerate(keep) if x in ends), None)   # in a batch, a finished sequence is padded with eos after its end
            samples.append(dict(text=tok.decode(keep if cut is None else keep[:cut], skip_special_tokens=True), new_tokens=len(keep) if cut is None else cut + 1,
                                finished=cut is not None))
        row = dict(decision_id=did, status="ok" if len(samples) == G["samples"] else "partial_oom", mode=mode, prompt_tokens=n_in, seconds=round(time.time() - t, 1), samples=samples)
        with open(gen, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta["rows"].append({k: v for k, v in row.items() if k != "samples"} | dict(new_tokens=[x["new_tokens"] for x in samples], finished=[x["finished"] for x in samples]))
        meta["done"] += 1
        print("L2", worker, meta["done"], len(rows), mode, row["seconds"], [x["new_tokens"] for x in samples], flush=True)
        json.dump(meta, open(mp, "w"), indent=1)
    json.dump(meta, open(mp, "w"), indent=1)
    print("LEVEL2 WORKER DONE", worker, meta["done"], flush=True)


if __name__ == "__main__":
    if os.environ.get("L12_STAGE") == "level1":
        level1()
    elif os.environ.get("L2_WORKER") is not None:
        level2_worker(int(os.environ["L2_WORKER"]), int(os.environ["L2_WORKERS"]))
    else:   # the parent never touches CUDA: every stage is its own process, started from the copy of this file in the dataset
        os.makedirs(OUT, exist_ok=False)
        me = os.path.join(CODE, "level12_worker.py")
        assert ('VERSION = "%s"' % VERSION) in open(me, encoding="utf-8").read(), "the dataset copy of the script is not this version"
        if RUN_LEVEL1:
            print("level 1 exit code", subprocess.run([sys.executable, me], env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", L12_STAGE="level1", L12_VARIANTS=VARIANTS)).returncode, flush=True)
        if not RUN_LEVEL2:
            raise SystemExit(0)
        n = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))
        env = lambda g: dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), L2_WORKER=str(g), L2_WORKERS=str(n), L2_T0=str(T0), L2_PARITY="" if L2_PARITY is None else str(L2_PARITY))
        procs, restarts = {g: subprocess.Popen([sys.executable, me], env=env(g)) for g in range(n)}, 0
        while procs:
            time.sleep(5)
            for g, p in list(procs.items()):
                code = p.poll()
                if code is None:
                    continue
                del procs[g]
                print("worker", g, "exit code", code, flush=True)
                if code != 0 and restarts < 40 and time.time() < T0 + CFG["level2"]["wall_budget_seconds"]:
                    restarts += 1
                    procs[g] = subprocess.Popen([sys.executable, me], env=env(g))
        print("LEVEL2 ALL WORKERS FINISHED, restarts", restarts, flush=True)
