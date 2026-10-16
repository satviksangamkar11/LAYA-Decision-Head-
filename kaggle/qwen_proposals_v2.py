#!/usr/bin/env python3
"""Qwen proposal generation for the 45,500 captured decisions (exploratory, no gate). ONE file: Kaggle script kernels upload a single code file.

Context given to Qwen = row['state_text'] only (never the candidates, never the gold). N=4 samples per decision, one request per decision (n=4 shares one prefill).
  mode nothink       : Qwen3-30B-A3B (Kaggle Model qwen-lm/qwen-3/transformers/30b-a3b/1), enable_thinking=False, max_tokens 128, vllm-tpu on TPU v5e-8
  mode think_capped  : Qwen3-4B-Thinking-2507 (dataset qwen3-4b-thinking-2507), thinking cap 384 tokens + batched forced answer (128), vLLM fp16 on a T4, 2 worker processes
Install recipe and vLLM-on-T4 settings are UNVERIFIED until the smoke runs; if vLLM fails to start the script falls back to batched HF transformers (think_capped only).
All speeds are measured in --smoke and printed; nothing here is a measured claim. Token budget formula: decisions x 4 x max_tokens.

  python qwen_proposals_v2.py --selftest [--rows rows.jsonl]     # CPU, no model
  python qwen_proposals_v2.py --smoke --mode nothink|think_capped  # 20 decisions per source
  python qwen_proposals_v2.py --mode nothink|think_capped          # full run, shards of 2000 decisions, resumable
"""
import argparse, glob, gzip, hashlib, json, os, shutil, subprocess, sys, time

SHARD_DECISIONS, BATCH_DECISIONS, N_SAMPLES = 2000, 32, 4
INSTR = ('Reply with exactly ONE next action as a JSON object {"name": <tool>, "arguments": {...}} '
         'in the same format as the history above. No other text.')
ROWS_PATH = "/kaggle/input/laya-head-rows-v1/rows.jsonl"
STATES_DIR = "/kaggle/input/laya-head-states-v1"
RESUME_GLOB = "/kaggle/input/*proposals*/**/shard-*.jsonl.gz"
OUT_DIR = "/kaggle/working/proposals"
VENV = "/tmp/vt"


def log(*a): print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- prompt / parsing
def build_prompt(state_text, tok, enable_thinking):
    msgs = [{"role": "user", "content": state_text + "\n\n" + INSTR}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def canon(text):
    """Canonical JSON of an action string; None if it is not a {"name","arguments"} object."""
    try:
        o = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(o, dict) or "name" not in o or "arguments" not in o:
        return None
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def parse_action(raw):
    """First balanced {...} after an optional </think>; returns (canonical_or_raw_json, ok)."""
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    s = raw.find("{")
    if s < 0:
        return "", False
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(s, len(raw)):
        c = raw[i]
        if esc: esc = False; continue
        if in_str:
            if c == "\\": esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end < 0:
        return "", False
    js = raw[s:end + 1]
    c = canon(js)
    return (c, True) if c else (js, False)


# ---------------------------------------------------------------- engines: generate(prompts, n, max_tokens, temperature, top_p) -> [(prompt_len, [(text, n_tokens, finish_reason)] * n)]
class VLLMEngine:
    def __init__(self, path, dtype, tp, gpu=True):
        from vllm import LLM
        kw = dict(model=path, dtype=dtype, tensor_parallel_size=tp, max_model_len=6144)
        if gpu: kw["gpu_memory_utilization"] = 0.88
        self.llm = LLM(**kw)

    def generate(self, prompts, n, max_tokens, temperature, top_p):
        from vllm import SamplingParams
        ros = self.llm.generate(prompts, SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p))
        return [(len(r.prompt_token_ids), [(o.text, len(o.token_ids), o.finish_reason) for o in r.outputs]) for r in ros]


class HFEngine:
    def __init__(self, path, tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.tok = torch, tok
        tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()

    def generate(self, prompts, n, max_tokens, temperature, top_p):
        torch, tok, out = self.torch, self.tok, []
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i:i + 8], return_tensors="pt", padding=True).to("cuda")
            L = enc["input_ids"].shape[1]
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                        top_p=top_p, num_return_sequences=n, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = g[:, L:]
            for b in range(len(prompts[i:i + 8])):
                outs = []
                for k in range(n):
                    ids = new[b * n + k].tolist()
                    stopped = tok.eos_token_id in ids
                    if stopped: ids = ids[:ids.index(tok.eos_token_id)]
                    outs.append((tok.decode(ids, skip_special_tokens=False), len(ids), "stop" if stopped else "length"))
                out.append((int((enc["attention_mask"][b]).sum()), outs))
        return out


# ---------------------------------------------------------------- one batch of decisions -> sample rows
CAP = [None]       # --think-cap override, set in main()
THINK_MODES = ("think_capped", "think30")      # think30 = Qwen3-30B-A3B with enable_thinking=True (the user's plan); think_capped = Qwen3-4B-Thinking-2507


def run_batch(engine, tok, rows, mode, cap=None):
    think = mode in THINK_MODES
    prompts = [build_prompt(r["state_text"], tok, think) for r in rows]
    t0 = time.time()
    mt = cap or CAP[0] or {"think_capped": 384, "think30": 1024}.get(mode, 128)
    res = engine.generate(prompts, N_SAMPLES, mt, 0.6 if think else 0.7, 0.95 if think else 0.9)
    pre = sum(p for p, _ in res)
    gen = sum(nt for _, outs in res for _, nt, _ in outs)
    samples = [[dict(text=t, ntok=nt, fr=fr, forced=False) for t, nt, fr in outs] for _, outs in res]
    forced = 0
    if think:
        need = [(i, j) for i, s in enumerate(samples) for j, x in enumerate(s) if "</think>" not in x["text"]]
        if need:
            p2 = [prompts[i] + samples[i][j]["text"] + "\n</think>\n\n" for i, j in need]
            for (i, j), (_, outs) in zip(need, engine.generate(p2, 1, 128, 0.3, 0.9)):
                t, nt, _ = outs[0]
                x = samples[i][j]
                x["text"] += "\n</think>\n\n" + t; x["ntok"] += nt; x["forced"] = True
                gen += nt; forced += 1
    out = []
    for r, ss in zip(rows, samples):
        for k, x in enumerate(ss):
            a, ok = parse_action(x["text"])
            has = "</think>" in x["text"]
            out.append(dict(decision_id=r["decision_id"], source=r["source"], split=r.get("split"), sample_idx=k, action_text=a, parse_ok=ok,
                            answer_text=(x["text"].split("</think>")[-1] if has else x["text"]).strip()[:1500],
                            thinking_head=(x["text"].split("</think>")[0] if has else "")[:200],
                            out_tokens=x["ntok"], finish_reason=x["fr"], forced_answer=x["forced"]))
    return out, dict(seconds=time.time() - t0, prefill_tokens=pre, generated_tokens=gen, forced_calls=forced, decisions=len(rows))


# ---------------------------------------------------------------- data loading
def find_model(kind):
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        p = c.lower()
        if kind == "30b" and "30b-a3b" in p and "fp8" not in p and "2507" not in p and "base" not in p: return os.path.dirname(c)
        if kind == "4b" and "4b-thinking-2507" in p: return os.path.dirname(c)
    return None


def states_ids(files):
    import torch
    ids = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        ids += [it["decision_id"] for it in d["items"]]
    return ids


def load_rows(ids, rows_path=ROWS_PATH):
    want, got = set(ids), {}
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["decision_id"] in want:
                got[r["decision_id"]] = dict(decision_id=r["decision_id"], source=r["source"], split=r.get("split"), state_text=r["state_text"],
                                             gold=r["candidates"][0]["text"])
    return got


# ---------------------------------------------------------------- smoke / full
def smoke(engine, tok, mode, rows_by_id, ids):
    by_src = {}
    for d in ids:
        r = rows_by_id.get(d)
        if r and len(by_src.setdefault(r["source"], [])) < 20: by_src[r["source"]].append(d)
    log("smoke decisions per source:", {s: len(v) for s, v in by_src.items()}, "(expected nebius, swesmith, fable5)")
    rows = [rows_by_id[d] for v in by_src.values() for d in v]
    out, m = run_batch(engine, tok, rows, mode)
    for r in rows[::12][:5]:
        print("\nDECISION", r["decision_id"][:60], "\n  state_text tail:", r["state_text"][-300:].replace("\n", " | "))
        for s in [x for x in out if x["decision_id"] == r["decision_id"]]:
            print("  proposal %d ok=%s fin=%s: %s" % (s["sample_idx"], s["parse_ok"], s["finish_reason"], s["action_text"][:110] or s["answer_text"][:110]))
        print("  GOLD:", r["gold"][:110])
    print("\n== SMOKE METRICS (measured) mode=%s" % mode)
    for s in by_src:
        xs = [x for x in out if x["source"] == s]
        ok = [x for x in xs if x["parse_ok"]]
        hit = sum(1 for x in ok if x["action_text"] == canon(rows_by_id[x["decision_id"]]["gold"]))
        any_hit = len({x["decision_id"] for x in ok if x["action_text"] == canon(rows_by_id[x["decision_id"]]["gold"])})
        print("  %-8s samples %d parse_ok %.1f%% | sample==gold %d | decisions with a gold hit %d/%d" % (s, len(xs), 100 * len(ok) / max(len(xs), 1), hit, any_hit, len(by_src[s])))
    sec = m["seconds"] / m["decisions"]
    print("  seconds %.1f | decisions %d | %.2f s/decision | prefill tokens %d | generated tokens %d (%.0f gen tok/s, blended with prefill) | forced answers %d"
          % (m["seconds"], m["decisions"], sec, m["prefill_tokens"], m["generated_tokens"], m["generated_tokens"] / m["seconds"], m["forced_calls"]))
    h = sec * 45500 / 3600
    print("  EXTRAPOLATION for 45,500 decisions on this device: %.1f h (%d session(s) of 12 h); from %d decisions, unverified" % (h, int(h // 12) + 1, m["decisions"]))
    json.dump(dict(mode=mode, metrics=m, per_source={s: len(v) for s, v in by_src.items()}, sample_rows=out[:40]), open("/kaggle/working/smoke_result.json", "w"), indent=1)


def write_shard(path, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def full(engine, tok, mode, rows_by_id, ids, worker, workers, max_hours):
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in glob.glob(RESUME_GLOB, recursive=True):          # shards from an earlier session, mounted as a dataset
        d = os.path.join(OUT_DIR, os.path.basename(f))
        if not os.path.exists(d): shutil.copy2(f, d)
    t0, n_shards = time.time(), (len(ids) + SHARD_DECISIONS - 1) // SHARD_DECISIONS
    for k in range(n_shards):
        if k % workers != worker: continue
        path = os.path.join(OUT_DIR, "shard-%04d.jsonl.gz" % k)
        if os.path.exists(path): log("shard", k, "exists, skip"); continue
        if (time.time() - t0) / 3600 > max_hours: log("time guard hit, stopping cleanly before shard", k); break
        rows, acc = [rows_by_id[d] for d in ids[k * SHARD_DECISIONS:(k + 1) * SHARD_DECISIONS] if d in rows_by_id], []
        for b in range(0, len(rows), BATCH_DECISIONS):
            out, m = run_batch(engine, tok, rows[b:b + BATCH_DECISIONS], mode)
            acc += out
        write_shard(path, acc)
        log("shard", k, "decisions", len(rows), "samples", len(acc), "parse_ok %.1f%%" % (100 * sum(x["parse_ok"] for x in acc) / max(len(acc), 1)),
            "| %.2f s/decision so far" % ((time.time() - t0) / max(1, sum(1 for _ in range(k + 1)))))
    man = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sorted(glob.glob(OUT_DIR + "/shard-*.jsonl.gz"))}
    json.dump(dict(mode=mode, worker=worker, shards=man, decisions_total=len(ids)), open(os.path.join(OUT_DIR, "MANIFEST-w%d.json" % worker), "w"), indent=1)


# ---------------------------------------------------------------- bootstrap + main
def bootstrap(mode):
    if os.path.exists(VENV + "/bin/python") and "--in-venv" in sys.argv: return
    def sh(c): log("$", c); subprocess.run(c, shell=True, check=True)
    sh("pip install -q uv && uv venv %s --python 3.12" % VENV)
    pk = "vllm-tpu==0.28.0" if mode in ("nothink", "think30") else "vllm"
    sh("uv pip install --python %s/bin/python %s transformers huggingface_hub" % (VENV, pk))
    os.execv(VENV + "/bin/python", [VENV + "/bin/python", __file__] + sys.argv[1:] + ["--in-venv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nothink", "think30", "think_capped"], default="think30")
    ap.add_argument("--think-cap", type=int, default=None, help="max generated tokens (thinking + answer) per sample; think30 default 1024")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--rows")
    ap.add_argument("--worker", type=int, default=None); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=None); ap.add_argument("--in-venv", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest(a.rows)
    bootstrap(a.mode)
    if a.mode == "think_capped" and not a.smoke and a.worker is None:         # parent: one worker process per T4
        ps = [subprocess.Popen([sys.executable, __file__, "--mode", a.mode, "--worker", str(i), "--workers", "2", "--in-venv"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in range(2)]
        for p in ps: p.wait()
        return
    from transformers import AutoTokenizer
    tpu = a.mode in ("nothink", "think30")
    path = find_model("30b" if tpu else "4b")
    assert path, "model not found under /kaggle/input"
    log("model", path); tok = AutoTokenizer.from_pretrained(path)
    files = sorted(glob.glob(STATES_DIR + "/shard-*.pt")); assert len(files) == 91, len(files)
    ids = states_ids(files[::15] if a.smoke else files)
    rows_by_id = load_rows(ids); log("decisions", len(ids), "rows", len(rows_by_id))
    try:
        if tpu:
            import jax
            engine = VLLMEngine(path, "bfloat16", jax.device_count(), gpu=False)
        else:
            engine = VLLMEngine(path, "float16", 1)
    except Exception as e:
        log("vLLM failed:", repr(e)[:300])
        if tpu: raise
        log("falling back to batched HF transformers"); engine = HFEngine(path, tok)
    CAP[0] = a.think_cap
    if a.smoke: smoke(engine, tok, a.mode, rows_by_id, ids)
    else: full(engine, tok, a.mode, rows_by_id, ids, a.worker or 0, a.workers, a.max_hours or (8.75 if tpu else 11.0))


# ---------------------------------------------------------------- selftest (CPU, no model)
class FakeTok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        return "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s" % (msgs[0]["content"], "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


class FakeEngine:
    """Truncated thinking for odd decision ordinals, complete otherwise; echoes forced-answer calls."""
    def __init__(self): self.calls = []
    def generate(self, prompts, n, max_tokens, temperature, top_p):
        self.calls.append((len(prompts), n, max_tokens)); out = []
        for i, p in enumerate(prompts):
            if p.rstrip().endswith("</think>"):  # forced answer
                out.append((len(p), [('{"name":"finish","arguments":{"message":"x"}}', 9, "stop")] * n)); continue
            if max_tokens == 128:   # nothink
                outs = [('{"name":"execute_bash","arguments":{"command":"ls %d"}}' % k, 12, "stop") for k in range(n)]
            elif i % 2:             # thinking cut off
                outs = [("Let me think about", 384, "length")] * n
            else:
                outs = [('thinking...\n</think>\n\n{"name":"execute_bash","arguments":{"command":"ls"}}', 50, "stop")] * n
            out.append((len(p), outs))
        return out


def selftest(rows_path):
    for raw, ok in [('</think>\n{"name":"a","arguments":{"x":"}{"}}', True), ('{"name":"a","arguments":{}}', True), ('{"name":"bad"', False), ('no json', False), ('</think>{"name":"a"}', False)]:
        assert parse_action(raw)[1] == ok, raw
    rows = [dict(decision_id="d%d" % i, source="nebius", split="train", state_text="TASK: t\nhistory %d" % i, gold='{"name": "execute_bash", "arguments": {"command": "ls"}}') for i in range(4)]
    if rows_path and os.path.exists(rows_path):
        real = []
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(real) < 60 and r["source"] in ("nebius",) or len(real) < 60 and r["source"] in ("fable5", "swesmith"):
                    real.append(r)
        # leak test on REAL rows: the prompt must not contain any candidate except where it already occurs inside state_text
        tok, hits = FakeTok(), 0
        for r in real:
            p = build_prompt(r["state_text"], tok, True)
            assert p.count(r["state_text"]) == 1 and p.endswith("<think>\n")
            for c in r["candidates"]:
                if c["text"] in p and c["text"] not in r["state_text"]: raise AssertionError("candidate leaked into prompt: " + c["id"])
                hits += c["text"] in r["state_text"]
        log("leak test OK on %d real rows; candidates already inside state_text (not added by the prompt): %d" % (len(real), hits))
        rows = [dict(decision_id=r["decision_id"], source=r["source"], split=r["split"], state_text=r["state_text"], gold=r["candidates"][0]["text"]) for r in real[:4]]
    for mode in ("think_capped", "think30", "nothink"):
        eng = FakeEngine()
        out, m = run_batch(eng, FakeTok(), rows, mode)
        assert len(out) == len(rows) * N_SAMPLES, (mode, len(out))                                  # 4 samples per decision, all kept
        assert all(x["parse_ok"] for x in out), [x for x in out if not x["parse_ok"]][:2]
        if mode in THINK_MODES:
            assert m["forced_calls"] == 2 * N_SAMPLES and eng.calls[1] == (2 * N_SAMPLES, 1, 128), (m, eng.calls)   # truncated ones forced in ONE batched call
            assert sum(x["forced_answer"] for x in out) == 2 * N_SAMPLES
        else:
            assert m["forced_calls"] == 0 and len(eng.calls) == 1
        log("run_batch OK", mode, "samples", len(out), "forced", m["forced_calls"])
    assert canon('{"arguments": {}, "name": "a"}') == canon('{"name":"a","arguments":{}}')
    log("SELFTEST PASS")


if __name__ == "__main__":
    main()
