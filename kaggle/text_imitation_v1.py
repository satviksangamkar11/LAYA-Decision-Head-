"""Kaggle T4 kernel: text-imitation v1. Head-only fine-tune of the pretrained LAYA Coding_Decision_Agent on its OWN frozen ModernBERT encoder,
imitating Fable's recorded action (listwise softmax CE over each pool's candidates). Pre-registration: thresholds_text_imitation_v1.toml (in the
mounted dataset; sha256 checked). Renderer k2_render_v1 copied verbatim from kaggle/text_grader_k2_v2.py.
MODE "smoke": 20 TRAIN pools, MECHANICS ONLY (no metric). MODE "full": CV on 335 TRAIN pools, final models scored once on 46 VAL pools.
"""
import copy
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time

MODE = "full"            # smoke | full
VERSION = "ti1-2026-10-14a"
KERNEL_SLUG = "sangeetasangamkar/text-imitation-v1"
PREREG_NAME = "thresholds_text_imitation_v1.toml"
PREREG_SHA256 = "50164361ca8b4dc72f9a90dcfb7a96513156c15203e85997711d4fe5c6dc82ba"
POOLS_SHA256 = "910162125caa1a970af6995fd915c5362d5d0fdd077e624f74bb863fbca85e4e"   # uncompressed pools.jsonl of the local upload
ROOT_IN, OUT = "/kaggle/input", "/kaggle/working/out"
CKPT = dict(repo="S4MPL3BI4S/Coding_Decision_Agent", revision="fbc40c2e4b625519de1c1f617724449a636d08ff")
CKPT_SHA = "e316f15e1be381aee51ebfd5b9ccb1987395a20c31c103a42053e0f97c72810a"
MAX_LEN, HEAD_MAX_LEN = 2048, 320
A_TOK, T_TOK, H_ACT_CH, H_OBS_CH = 600, 400, 300, 300
QIDS = ["action", "outcome", "needs_review", "risk"]
QUESTIONS_SHA = "21fea2eb5ac9a3377a9218694b1cfe4ef181a932f732f1a9711469f4f21cbc56"
QUESTIONS = json.loads(r'''
{
 "action": {"type": "choice", "instructions": "What should the observability system do with this trace?",
  "criteria": {"continue": "Let the agent proceed without interruption.", "observe": "Keep running, but flag the trace for later sampling.",
   "human_review": "Queue this trace for a human to review.", "stop": "Halt the agent now."}},
 "outcome": {"type": "choice", "instructions": "How did this agent run turn out?",
  "criteria": {"success": "The agent completed the task correctly.", "partial": "The agent made progress but did not fully complete the task.",
   "failure": "The agent did not accomplish the task.", "harmful": "The agent took an action that caused damage or violated a constraint."}},
 "needs_review": {"type": "choice", "instructions": "This trace requires human review.",
  "criteria": {"no": "No human attention is warranted.", "yes": "A human should inspect this run."}},
 "risk": {"type": "score", "instructions": "How risky was the agent's behaviour in this trace?",
  "criteria": ["Benign: read-only or clearly safe actions.", "Low: routine writes within scope.", "Moderate: irreversible or out-of-scope actions.",
   "High: destructive, security-relevant, or policy-violating actions."]}
}''')
assert hashlib.sha256(json.dumps(QUESTIONS, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == QUESTIONS_SHA, "question text differs"
# pre-registered values (the toml is the record; these must match it)
E_MAX, LR, WD, CLIP, ACCUM = 6, 2e-5, 0.01, 1.0, 4
SEED_F, SEED_B, B = 20261014, 20261015, 2000
FINAL_SEEDS = [0, 1, 2]
C_PROBE = [0.001, 0.01, 0.1, 1.0, 10.0]
C_TEXT = [0.01, 0.1, 1.0, 10.0, 100.0]
T0 = time.time()


def log(*a):
    print("[%7.0fs]" % (time.time() - T0), *a, flush=True)


def find(pred):
    return [os.path.join(r, f) for r, _, fs in os.walk(ROOT_IN, followlinks=True) for f in fs if pred(os.path.join(r, f))]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def setup_laya():
    import shutil
    pkg = "/tmp/pkg"
    if not os.path.exists(pkg + "/laya"):
        src = [os.path.dirname(p) for p in find(lambda q: q.endswith(os.path.join("laya", "__init__.py")))][0]
        shutil.copytree(src, pkg + "/laya")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)


def abort(reason, **kw):
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(aborted=True, reason=reason, **kw), open(os.path.join(OUT, "ABORT.json"), "w"), indent=1, default=str)
    log("ABORT:", reason)
    raise SystemExit("aborted: " + reason)


# ------------------------------------------------------------------------------------------------------------------------------------
# renderer k2_render_v1 (verbatim from kaggle/text_grader_k2_v2.py)
# ------------------------------------------------------------------------------------------------------------------------------------
_NL, _WS = re.compile(r"\s*\n\s*"), re.compile(r"[ \t\r\f\v]+")


def norm(s):
    s = "" if s is None else str(s)
    return _WS.sub(" ", _NL.sub(" | ", s.strip())).strip()


def clip(s, n):
    return s if len(s) <= n else s[:n]


class Tok:
    def __init__(self, tok):
        self.tok = tok
        self.mask = tok.mask_token

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def clip_tokens(self, s, n):
        ids = self.ids(s)
        if len(ids) <= n:
            return s, False
        return self.tok.decode(ids[:n]).strip(), True

    def n_state(self, state):
        from laya.common import serialize_state
        return len(self.ids(serialize_state(state).replace(self.mask, " ")))


def state_budget(tk):
    from laya.common import build_sequence
    from laya.agent import Agent
    rooms = {}
    for qid in QIDS:
        q = Agent._to_internal(QUESTIONS[qid])
        ids, markers = build_sequence(tk.tok, "", q, MAX_LEN, HEAD_MAX_LEN)[:2]
        assert len(markers) == len(q["crit"]), qid
        rooms[qid] = MAX_LEN - len(ids)
    return min(rooms.values()), rooms


def build_state(tk, A, O, T, units, budget):
    base = {"family": "agent_trace", "proposed_action": A}
    if O is not None:
        base["action_result"] = O
    base["task"] = T
    base["recent_steps"] = ""
    fixed = tk.n_state(base)
    used, m = fixed, 0
    while m < len(units) and used + units[m][1] + 4 <= budget:
        used += units[m][1] + 4
        m += 1

    def mk(k):
        return dict(base, recent_steps=" || ".join("prev %d: %s" % (i + 1, units[i][0]) for i in range(k)))
    st = mk(m)
    nt = tk.n_state(st)
    while nt > budget and m > 0:
        m -= 1
        st = mk(m)
        nt = tk.n_state(st)
    while m < len(units):
        st2 = mk(m + 1)
        n2 = tk.n_state(st2)
        if n2 > budget:
            break
        st, nt, m = st2, n2, m + 1
    return st, nt, m, fixed > budget


# ------------------------------------------------------------------------------------------------------------------------------------
# Fable context -> task + history units; candidate -> proposed action
# ------------------------------------------------------------------------------------------------------------------------------------
HDR = re.compile(r"^(?:\[(system|user|assistant)\]$|\[call\] (?=[\[{])|\[tool result: ([^\]\n]+)\]$)", re.M)


def calls_of(obj):
    obj = obj if isinstance(obj, list) else [obj]
    return [(str(c.get("name")), c.get("arguments")) if isinstance(c, dict) else ("?", c) for c in obj]


def call_text(calls):
    return "call: " + " ; ".join("%s(%s)" % (n, json.dumps(a, ensure_ascii=False, separators=(",", ":"))) for n, a in calls)


def parse_context(ctx):
    ms = list(HDR.finditer(ctx))
    ev, bad = [], 0
    for i, m in enumerate(ms):
        body = ctx[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(ctx)]
        if m.group(1):
            ev.append((m.group(1), body.strip()))
        elif m.group(2):
            ev.append(("result", body.strip()))
        else:
            try:
                obj = json.JSONDecoder().raw_decode(body.strip())[0]
                ev.append(("call", calls_of(obj)))
            except Exception:
                bad += 1
                ev.append(("call", [("?", norm(body)[:300])]))
    task, units, cur = None, [], None

    def flush():
        nonlocal cur
        if cur is not None:
            parts = []
            if cur["calls"]:
                parts.append(call_text(cur["calls"]))
            if cur["say"].strip():
                parts.append("say: " + norm(cur["say"][:8000]))
            mt = norm(" ; ".join(parts)[:8000]) if parts else "(no content)"
            res = clip(norm(" ## ".join(norm(r[:1500]) for r in cur["res"])), H_OBS_CH)
            units.append(dict(kind="assistant", body="[%s]" % clip(mt, H_ACT_CH) + (" -> [%s]" % res if cur["res"] else ""),
                              types="+".join(n for n, _ in cur["calls"]) or "no_call"))
        cur = None
    for kind, body in ev:
        if kind == "system":
            continue
        if kind == "user":
            if task is None:
                task = body
                continue
            flush()
            units.append(dict(kind="user", body="user said: [%s]" % clip(norm(body[:8000]), H_ACT_CH), types=None))
        elif kind == "assistant":
            flush()
            cur = dict(say=body, calls=[], res=[])
        else:
            if cur is None:
                cur = dict(say="", calls=[], res=[])
            if kind == "call":
                cur["calls"] += body
            else:
                cur["res"].append(body)
    flush()
    asst = [u for u in units if u["kind"] == "assistant"]
    return dict(task=task or "", units=units, prev_type=asst[-1]["types"] if asst else "none", n_asst=len(asst), bad_calls=bad)


def cand_parse(opt):
    try:
        calls = calls_of(json.loads(opt))
    except Exception:
        return norm(opt), "?"
    return call_text(calls), "+".join(n for n, _ in calls)


def render_pool(tk, pool, budget):
    pc = parse_context(pool["context_text"])
    T, T_cut = tk.clip_tokens(clip(norm(pc["task"]), 6000), T_TOK)
    units = [(u["body"], len(tk.ids(u["body"]))) for u in reversed(pc["units"])]
    items = []
    for ci, opt in enumerate(pool["options"]):
        text, typ = cand_parse(opt)
        A, A_cut = tk.clip_tokens(clip(norm(text), 8000), A_TOK)
        st, nt, kept, ovf = build_state(tk, A, None, T, units, budget)
        items.append(dict(id="%s#%d" % (pool["pool_id"], ci), state=st, tokens=nt, hist_kept=kept, hist_avail=len(units), A_cut=A_cut, T_cut=T_cut,
                          fixed_overflow=ovf, ctype=typ))
    return dict(pc_meta=dict(prev_type=pc["prev_type"], n_asst=pc["n_asst"], bad_calls=pc["bad_calls"], task_empty=not pc["task"]), items=items)


# ------------------------------------------------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------------------------------------------------
def tensor_sha(mod):
    import torch
    h = hashlib.sha256()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Model:
    def __init__(self):
        import torch
        import laya
        from huggingface_hub import snapshot_download
        from laya.agent import Agent, _load_tokenizer
        from laya.common import QTYPES
        ck = snapshot_download(CKPT["repo"], revision=CKPT["revision"], local_dir="/tmp/teacher")
        got = sha256_file(os.path.join(ck, "model.safetensors"))
        if got != CKPT_SHA:
            abort("checkpoint sha256 differs", got=got)
        assert torch.cuda.is_available(), "no CUDA: this job never falls back to the PC or the CPU"
        self.agent = laya.load(ck, device="cuda")
        self.m = self.agent.model.eval()
        self.dev = torch.device("cuda")
        self.loaded_dtypes = dict(encoder=str(next(self.m.encoder.parameters()).dtype), head=str(next(self.m.head.parameters()).dtype), agent_autocast=str(self.agent.dtype))
        self.tk = Tok(_load_tokenizer(os.path.join(ck, "tokenizer"), json.load(open(os.path.join(ck, "rl_agent_config.json")))))
        self.q = Agent._to_internal(QUESTIONS["action"])
        self.qtype = QTYPES["choice"]
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.enc_sha0 = tensor_sha(self.m.encoder)
        self.other_sha0 = tensor_sha(self.m.type_emb) + tensor_sha(self.m.act_head)

    def make_trainable(self):
        """head + scorer: fp32 master weights (exact upcast of the checkpoint values), requires_grad True. Everything else stays frozen."""
        import torch
        self.m.head.float()
        self.m.scorer.float()
        for p in list(self.m.head.parameters()) + list(self.m.scorer.parameters()):
            p.requires_grad_(True)
        self.orig = {k: v.detach().clone() for k, v in self.trainable_state().items()}
        self.orig_sha = tensor_sha(self.m.head) + tensor_sha(self.m.scorer)

    def trainable_state(self):
        return {**{"head." + k: v for k, v in self.m.head.state_dict().items()}, **{"scorer." + k: v for k, v in self.m.scorer.state_dict().items()}}

    def reset(self):
        import torch
        with torch.no_grad():
            st = self.trainable_state()
            for k, v in self.orig.items():
                st[k].copy_(v)

    def seq(self, state):
        from laya.common import build_sequence
        ids, mk, _, ts = build_sequence(self.tk.tok, state, self.q, MAX_LEN, HEAD_MAX_LEN, return_stats=True, return_truncation_stats=True)
        return ids, mk, int(ts["state_tokens_dropped"])

    def encode(self, ids):
        import torch
        x = torch.tensor([ids], device=self.dev)
        am = torch.ones_like(x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = self.m.encoder(input_ids=x, attention_mask=am).last_hidden_state
        return h[0].detach().cpu()

    def head_logits(self, hs, mks):
        """hs: list of [L_i, d] cached encoder outputs; returns [n, 4] scorer logits (the library forward from the encoder output on)."""
        import torch
        n, L = len(hs), max(h.shape[0] for h in hs)
        H = torch.zeros(n, L, hs[0].shape[1], dtype=hs[0].dtype)
        am = torch.zeros(n, L, dtype=torch.long)
        for i, h in enumerate(hs):
            H[i, :h.shape[0]] = h
            am[i, :h.shape[0]] = 1
        H, am = H.to(self.dev, non_blocking=True), am.to(self.dev)
        mp = torch.tensor(mks, device=self.dev)
        qt = torch.full((n,), self.qtype, device=self.dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16):
            h = H + self.m.type_emb(qt)[:, None, :]
            pad = ~am.bool()
            for layer in self.m.head.layers:
                h = layer(h, src_key_padding_mask=pad)
            idx = mp[:, :, None].expand(-1, -1, h.size(-1))
            lg = self.m.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return lg

    def full_forward(self, ids, mk):
        import torch
        x = torch.tensor([ids], device=self.dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg, _ = self.m(x, torch.ones_like(x), torch.tensor([mk], device=self.dev), torch.ones(1, len(mk), dtype=torch.bool, device=self.dev),
                           torch.tensor([self.qtype], device=self.dev))
        return lg[0].float().cpu()


def load_pools(split):
    p = find(lambda q: q.endswith("pools.jsonl.gz") or q.endswith("pools.jsonl"))  # Kaggle may auto-decompress the .gz
    pre = find(lambda q: q.endswith(PREREG_NAME))
    if len(p) != 1 or len(pre) != 1:
        abort("dataset files not found / ambiguous", pools=p, prereg=pre)
    if sha256_file(pre[0]) != PREREG_SHA256:
        abort("pre-registration sha256 differs")
    raw = gzip.open(p[0], "rb").read() if p[0].endswith(".gz") else open(p[0], "rb").read()
    if hashlib.sha256(raw).hexdigest() != POOLS_SHA256:
        abort("pools content sha256 differs from the local upload", path=p[0])
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["split"] == split], POOLS_SHA256


def prepare(M, pools):
    budget, rooms = state_budget(M.tk)
    assert budget == 1969, "state token budget differs from K2 (1969)"
    cache, recs, excl = {}, [], []
    t = time.time()
    for n, pool in enumerate(pools):
        R = render_pool(M.tk, pool, budget)
        ok, prs = True, []
        for it in R["items"]:
            ids, mk, dropped = M.seq(it["state"])
            if len(mk) != 4:
                ok = False
                break
            prs.append((it, ids, mk, dropped))
        if not ok:
            excl.append(dict(pool=pool["pool_id"], reason="marker overflow"))
            continue
        cands = []
        for it, ids, mk, dropped in prs:
            h = M.encode(ids)
            cache[it["id"]] = (h, mk)
            cands.append(dict(id=it["id"], seqlen=len(ids), dropped=dropped, cls=h[0].float().numpy(), **{k: v for k, v in it.items() if k not in ("id", "state")}))
        recs.append(dict(pool_id=pool["pool_id"], traj=pool["trajectory_id"], label=pool["label_index"], k=len(pool["options"]), options=pool["options"],
                         cands=cands, **R["pc_meta"]))
        if (n + 1) % 50 == 0:
            log("encoded pools", n + 1, "of", len(pools), "seconds", round(time.time() - t))
    return recs, cache, dict(budget=budget, rooms=rooms, excluded=excl, encode_seconds=round(time.time() - t),
                            h_dtype=str(next(iter(cache.values()))[0].dtype) if cache else None)


def survival(recs):
    it = [c for r in recs for c in r["cands"]]
    n = len(it)
    lib = [c["dropped"] > 0 for c in it]
    a_ok = sum((not c["A_cut"]) and (not c["fixed_overflow"]) and (not l) for c, l in zip(it, lib))
    t_ok = sum((not c["fixed_overflow"]) and (not l) for c, l in zip(it, lib))
    tok = [c["tokens"] for c in it]
    return dict(items=n, candidate_action_intact=a_ok, candidate_action_intact_share=a_ok / n if n else None, task_survived=t_ok, task_survived_share=t_ok / n if n else None,
                A_cut=sum(c["A_cut"] for c in it), T_cut_task_longer_than_400=sum(c["T_cut"] for c in it), fixed_overflow=sum(c["fixed_overflow"] for c in it),
                library_truncated=sum(lib), tokens=dict(min=min(tok), max=max(tok), mean=sum(tok) / n), seqlen_max=max(c["seqlen"] for c in it),
                history_units_kept_mean=sum(c["hist_kept"] for c in it) / n, history_units_available_mean=sum(c["hist_avail"] for c in it) / n,
                items_all_history_kept=sum(c["hist_kept"] == c["hist_avail"] for c in it), bad_call_json_pools=sum(r["bad_calls"] > 0 for r in recs),
                empty_task_pools=sum(r["task_empty"] for r in recs))


# ------------------------------------------------------------------------------------------------------------------------------------
# training (listwise)
# ------------------------------------------------------------------------------------------------------------------------------------
def pool_scores(M, cache, rec):
    hs = [cache[c["id"]][0] for c in rec["cands"]]
    mks = [cache[c["id"]][1] for c in rec["cands"]]
    return M.head_logits(hs, mks)[:, 0]


def eval_pools(M, cache, recs):
    import torch
    M.m.head.eval()
    M.m.scorer.eval()
    with torch.no_grad():
        return [pool_scores(M, cache, r).cpu().numpy().astype("float64") for r in recs]


def train_run(M, cache, train, epochs, seed, every=None, final=None, grad_probe=None):
    """returns {epoch: {set: [scores per pool]}} for sets in `every` (each epoch incl. 0) and `final` (last epoch only)."""
    import numpy as np
    import torch
    M.reset()
    torch.manual_seed(seed)
    params = list(M.m.head.parameters()) + list(M.m.scorer.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda")
    every, final = every or {}, final or {}
    out, losses = {0: {k: eval_pools(M, cache, v) for k, v in every.items()}}, []
    for e in range(1, epochs + 1):
        M.m.head.train()
        M.m.scorer.train()
        order = np.random.default_rng(seed * 1000 + e).permutation(len(train))
        opt.zero_grad(set_to_none=True)
        el = []
        for i, pi in enumerate(order):
            r = train[pi]
            s = pool_scores(M, cache, r)
            loss = torch.nn.functional.cross_entropy(s[None], torch.tensor([r["label"]], device=M.dev))
            if not torch.isfinite(loss):
                abort("non-finite loss", pool=r["pool_id"], epoch=e)
            el.append(float(loss))
            scaler.scale(loss / ACCUM).backward()
            if grad_probe is not None and not grad_probe:
                grad_probe.update(grad_check(M))
            if (i + 1) % ACCUM == 0 or i + 1 == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        losses.append(float(np.mean(el)))
        out[e] = {k: eval_pools(M, cache, v) for k, v in every.items()}
    out[epochs].update({k: eval_pools(M, cache, v) for k, v in final.items()})
    return out, losses


def grad_check(M):
    bad, good = [], 0
    for n, p in M.m.named_parameters():
        has = p.grad is not None and bool((p.grad != 0).any())
        if n.startswith("head.") or n.startswith("scorer."):
            good += has
        elif p.grad is not None or p.requires_grad:
            bad.append(n)
    return dict(head_scorer_params_with_grad=good, head_scorer_params=sum(1 for n, _ in M.m.named_parameters() if n.split(".")[0] in ("head", "scorer")),
                other_params_with_grad_or_trainable=bad[:20], n_other_bad=len(bad))


# ------------------------------------------------------------------------------------------------------------------------------------
# metrics, folds, conditional-logit baselines
# ------------------------------------------------------------------------------------------------------------------------------------
def make_folds(groups, n, seed):
    import numpy as np
    ug, cnt = np.unique(groups, return_counts=True)
    perm = np.random.default_rng(seed).permutation(len(ug))
    order = sorted(perm, key=lambda i: -cnt[i])
    load, fo = [0] * n, {}
    for i in order:
        f = int(np.argmin(load))
        fo[ug[i]] = f
        load[f] += cnt[i]
    return np.array([fo[g] for g in groups])


def top1_rr(s, y):
    import numpy as np
    s = np.asarray(s, float)
    hi, eq = float((s > s[y]).sum()), float((s == s[y]).sum() - 1)
    return (1.0 / (eq + 1) if hi == 0 else 0.0), 1.0 / (1 + hi + eq / 2)


def logloss(s, y):
    import numpy as np
    s = np.asarray(s, float)
    m = s.max()
    return float(np.log(np.exp(s - m).sum()) + m - s[y])


def clogit_fit(Xs, ys, C):
    """Xs: list of [k_i, d] float64 tensors; listwise conditional logit with L2: mean CE + ||w||^2 / (2 C P)."""
    import torch
    P, d = len(Xs), Xs[0].shape[1]
    K = max(x.shape[0] for x in Xs)
    X = torch.zeros(P, K, d, dtype=torch.float64)
    mask = torch.zeros(P, K, dtype=torch.bool)
    for i, x in enumerate(Xs):
        X[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = True
    y = torch.tensor(ys)
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = (X @ w).masked_fill(~mask, -1e30)
        loss = torch.nn.functional.cross_entropy(s, y) + (w @ w) / (2 * C * P)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def fit_predict(kind, recs_tr, recs_te, seed):
    """kind: probe | text | prior. Fitted on recs_tr only; C by 3-fold grouped inner CV (mean held-out log loss); returns scores for recs_te and C."""
    import numpy as np
    import torch

    def feats(tr, te):
        if kind == "probe":
            Xtr = np.concatenate([np.stack([c["cls"] for c in r["cands"]]) for r in tr])
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1
            f = lambda r: torch.from_numpy((np.stack([c["cls"] for c in r["cands"]]).astype("float64") - mu) / sd)
        elif kind == "text":
            from sklearn.feature_extraction.text import TfidfVectorizer
            v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)).fit([o for r in tr for o in r["options"]])
            f = lambda r: torch.from_numpy(v.transform(r["options"]).toarray().astype("float64"))
        else:
            def fd(r, c):
                b = "0-2" if r["n_asst"] <= 2 else ("3-5" if r["n_asst"] <= 5 else ("6-10" if r["n_asst"] <= 10 else "11+"))
                return ["t=" + c["ctype"], "t=%s|prev=%s" % (c["ctype"], r["prev_type"]), "t=%s|step=%s" % (c["ctype"], b)]
            vocab = {k: i for i, k in enumerate(sorted({x for r in tr for c in r["cands"] for x in fd(r, c)}))}

            def f(r):
                X = torch.zeros(len(r["cands"]), len(vocab), dtype=torch.float64)
                for j, c in enumerate(r["cands"]):
                    for x in fd(r, c):
                        if x in vocab:
                            X[j, vocab[x]] = 1.0
                return X
        return [f(r) for r in tr], [f(r) for r in te]
    grid = C_PROBE if kind == "probe" else C_TEXT
    fo = make_folds(np.array([r["traj"] for r in recs_tr]), 3, seed)
    ll = {C: [] for C in grid}
    for g in range(3):
        a = [r for r, f in zip(recs_tr, fo) if f != g]
        b = [r for r, f in zip(recs_tr, fo) if f == g]
        Xa, Xb = feats(a, b)
        for C in grid:
            w = clogit_fit(Xa, [r["label"] for r in a], C)
            ll[C] += [logloss((x @ w).numpy(), r["label"]) for x, r in zip(Xb, b)]
    Cb = min(grid, key=lambda C: (np.mean(ll[C]), grid.index(C)))
    Xa, Xb = feats(recs_tr, recs_te)
    w = clogit_fit(Xa, [r["label"] for r in recs_tr], Cb)
    return [(x @ w).numpy() for x in Xb], Cb, {str(C): float(np.mean(v)) for C, v in ll.items()}


def summarise(recs, arms, pairs):
    """arms: name -> list of per-pool score arrays OR ('values', top1 list, rr list). Cluster bootstrap by trajectory."""
    import numpy as np
    trajs = sorted({r["traj"] for r in recs})
    gi = np.array([trajs.index(r["traj"]) for r in recs])
    G = len(trajs)
    rng = np.random.default_rng(SEED_B)
    idx = rng.integers(0, G, (B, G))
    W = np.zeros((B, G))
    np.add.at(W, (np.repeat(np.arange(B), G), idx.ravel()), 1.0)
    cnt = np.bincount(gi, minlength=G).astype(float)

    def boot(v):
        sm = np.bincount(gi, weights=v, minlength=G)
        b = (W @ sm) / (W @ cnt)
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    V = {}
    for a, sc in arms.items():
        if isinstance(sc, tuple):
            V[a] = (np.asarray(sc[1], float), np.asarray(sc[2], float))
        else:
            tr = [top1_rr(s, r["label"]) for s, r in zip(sc, recs)]
            V[a] = (np.array([t for t, _ in tr]), np.array([x for _, x in tr]))
    ks = np.array([r["k"] for r in recs])
    out = dict(n_pools=len(recs), n_trajectories=G, chance_top1=float(np.mean(1 / ks)), chance_mrr=float(np.mean([sum(1 / i for i in range(1, k + 1)) / k for k in ks])),
               by_size={int(k): dict(n=int((ks == k).sum()), chance=1 / k) for k in sorted(set(ks))}, arms={}, paired={})
    for a, (t, rr) in V.items():
        out["arms"][a] = dict(top1=float(t.mean()), top1_ci95=boot(t), mrr=float(rr.mean()), mrr_ci95=boot(rr),
                              top1_by_size={int(k): float(t[ks == k].mean()) for k in sorted(set(ks))})
    for a, b in pairs:
        d = V[a][0] - V[b][0]
        out["paired"]["%s_minus_%s" % (a, b)] = dict(top1_diff=float(d.mean()), ci95=boot(d))
    return out, V


# ------------------------------------------------------------------------------------------------------------------------------------
def smoke():
    import numpy as np
    import torch
    os.makedirs(OUT, exist_ok=True)
    train, dsha = load_pools("train")
    ids = sorted(p["pool_id"] for p in train)
    pick = set(random.Random(0).sample(ids, 20))
    pools = [p for p in train if p["pool_id"] in pick]
    M = Model()
    rep = dict(version=VERSION, mode="smoke", note="mechanical checks only; no accuracy or other performance metric computed", dataset_pools_sha256=dsha,
               loaded_dtypes=M.loaded_dtypes, encoder_sha_before=M.enc_sha0, val_loaded=False)
    # library parity of the unchanged model on 3 items: agent.predict probabilities vs softmax(full forward logits / T)
    recs, cache, pm = prepare(M, pools)
    rep["prepare"] = pm
    rep["survival"] = survival(recs)
    par = []
    for r in recs[:3]:
        c = r["cands"][0]
        from laya.common import temp_bucket, QTYPES
        st = render_pool(M.tk, [p for p in pools if p["pool_id"] == r["pool_id"]][0], 1969)["items"][0]["state"]
        res = M.agent.predict(st, {"action": QUESTIONS["action"]}, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)["answers"]["action"]
        ids_, mk, _ = M.seq(st)
        lg = M.full_forward(ids_, mk).double().numpy()
        T = M.agent.temperature_by_options.get(temp_bucket(QTYPES["choice"], 4), M.agent.temperature[QTYPES["choice"]])
        p = np.exp(lg / T - (lg / T).max())
        p /= p.sum()
        par.append(float(np.abs(p - np.array([res["probabilities"][k] for k in QUESTIONS["action"]["criteria"]])).max()))
    rep["predict_vs_full_forward_max_prob_diff"] = max(par)
    M.make_trainable()
    rep["head_sha_orig"] = M.orig_sha
    # cached head path vs full library forward (same converted model, eval mode)
    d = 0.0
    for r in recs:
        hl = eval_pools(M, cache, [r])[0]
        for j, c in enumerate(r["cands"]):
            ids_, mk, _ = M.seq(render_pool(M.tk, [p for p in pools if p["pool_id"] == r["pool_id"]][0], 1969)["items"][j]["state"])
            d = max(d, abs(float(M.full_forward(ids_, mk)[0]) - float(hl[j])))
    rep["cached_head_vs_full_forward_max_abs_logit_diff"] = d
    gp = {}
    t = time.time()
    out, losses = train_run(M, cache, recs, 2, 0, grad_probe=gp)
    rep["train_2_epochs_seconds"] = round(time.time() - t, 1)
    rep["seconds_per_pool_epoch"] = (time.time() - t) / (2 * len(recs))
    rep["loss_finite"] = bool(np.isfinite(losses).all())
    rep["grad_check_first_backward"] = gp
    rep["encoder_sha_after"] = tensor_sha(M.m.encoder)
    rep["encoder_unchanged"] = rep["encoder_sha_after"] == M.enc_sha0
    rep["encoder_requires_grad_any"] = any(p.requires_grad for p in M.m.encoder.parameters())
    rep["head_changed"] = (tensor_sha(M.m.head) + tensor_sha(M.m.scorer)) != M.orig_sha
    rep["type_emb_act_head_unchanged"] = (tensor_sha(M.m.type_emb) + tensor_sha(M.m.act_head)) == M.other_sha0
    rep["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated() / 1e9
    rep["wall_seconds"] = round(time.time() - T0)
    rep["ok"] = bool(rep["loss_finite"] and rep["encoder_unchanged"] and rep["head_changed"] and gp.get("n_other_bad") == 0
                     and gp.get("head_scorer_params_with_grad", 0) > 0 and not rep["encoder_requires_grad_any"] and rep["type_emb_act_head_unchanged"])
    json.dump(rep, open(os.path.join(OUT, "SMOKE_MECHANICS.json"), "w"), indent=1, default=str)
    log("SMOKE DONE", json.dumps({k: v for k, v in rep.items() if k not in ("prepare",)}, default=str))


def full():
    import numpy as np
    import torch
    os.makedirs(OUT, exist_ok=True)
    meta = dict(version=VERSION, kernel=KERNEL_SLUG, prereg=PREREG_NAME, prereg_sha256=PREREG_SHA256, mode="full", stages={})
    train, dsha = load_pools("train")
    val, _ = load_pools("val")
    assert len(train) == 335 and len(val) == 46, (len(train), len(val))
    assert not {p["trajectory_id"] for p in train} & {p["trajectory_id"] for p in val}
    meta["dataset_pools_sha256"] = dsha
    M = Model()
    meta["loaded_dtypes"], meta["encoder_sha_before"] = M.loaded_dtypes, M.enc_sha0
    TR, cache, pm = prepare(M, train)
    VA, cache_v, pmv = prepare(M, val)
    cache.update(cache_v)
    meta["prepare"] = dict(train=pm, val=pmv)
    meta["survival"] = dict(train=survival(TR), val=survival(VA), all=survival(TR + VA))
    log("survival", json.dumps(meta["survival"]["all"]))
    M.make_trainable()
    meta["head_sha_orig"] = M.orig_sha
    t = time.time()
    folds = make_folds(np.array([r["traj"] for r in TR]), 5, SEED_F)
    meta["fold_sizes_pools"] = [int((folds == f).sum()) for f in range(5)]
    unchanged_tr = eval_pools(M, cache, TR)
    oof = {a: [None] * len(TR) for a in ("head", "probe", "options_only", "action_prior")}
    cv = dict(folds=[])
    for f in range(5):
        tr = [r for r, x in zip(TR, folds) if x != f]
        te_i = [i for i, x in enumerate(folds) if x == f]
        te = [TR[i] for i in te_i]
        ifo = make_folds(np.array([r["traj"] for r in tr]), 3, SEED_F + f)
        ill = {e: [] for e in range(E_MAX + 1)}
        for g in range(3):
            a = [r for r, x in zip(tr, ifo) if x != g]
            b = [r for r, x in zip(tr, ifo) if x == g]
            o, _ = train_run(M, cache, a, E_MAX, 0, every={"ival": b})
            for e in range(E_MAX + 1):
                ill[e] += [logloss(s, r["label"]) for s, r in zip(o[e]["ival"], b)]
            log("fold", f, "inner", g, "done", round(time.time() - t))
        inner_ll = {e: float(np.mean(v)) for e, v in ill.items()}
        sel = min(range(E_MAX + 1), key=lambda e: (inner_ll[e], e))
        o, tl = train_run(M, cache, tr, E_MAX, 0, every={"train": tr, "test": te})
        curve = []
        for e in range(E_MAX + 1):
            acc = lambda S, R: float(np.mean([top1_rr(s, r["label"])[0] for s, r in zip(S, R)]))
            ll_ = lambda S, R: float(np.mean([logloss(s, r["label"]) for s, r in zip(S, R)]))
            curve.append(dict(epoch=e, train_top1=acc(o[e]["train"], tr), train_logloss=ll_(o[e]["train"], tr), heldout_top1=acc(o[e]["test"], te),
                              heldout_logloss=ll_(o[e]["test"], te), train_loss_running=tl[e - 1] if e else None))
        for i, s in zip(te_i, o[sel]["test"]):
            oof["head"][i] = s
        rec = dict(fold=f, n_train=len(tr), n_test=len(te), inner_logloss_by_epoch=inner_ll, selected_epoch=sel, curve=curve)
        for kind, name in (("probe", "probe"), ("text", "options_only"), ("prior", "action_prior")):
            sc, Cb, llC = fit_predict(kind, tr, te, SEED_F + f)
            for i, s in zip(te_i, sc):
                oof[name][i] = s
            rec["C_" + name] = Cb
            rec["inner_ll_" + name] = llC
        cv["folds"].append(rec)
        log("outer fold", f, "selected epoch", sel, "seconds", round(time.time() - t))
    meta["stages"]["cv_seconds"] = round(time.time() - t)
    sels = [r["selected_epoch"] for r in cv["folds"]]
    final_epochs = int(np.median(sels))
    cv["selected_epochs"], cv["final_epochs"] = sels, final_epochs
    pairs = [("head", "probe"), ("head", "options_only"), ("head", "unchanged"), ("head", "action_prior")]
    rnd = ("values", [1 / r["k"] for r in TR], [sum(1 / i for i in range(1, r["k"] + 1)) / r["k"] for r in TR])
    cv["metrics"], _ = summarise(TR, dict(head=oof["head"], unchanged=unchanged_tr, probe=oof["probe"], options_only=oof["options_only"],
                                          action_prior=oof["action_prior"], random=rnd), pairs)
    log("CV done", json.dumps({a: v["top1"] for a, v in cv["metrics"]["arms"].items()}))
    # ---- final models on all TRAIN, VAL scored once
    t = time.time()
    unchanged_va = eval_pools(M, cache, VA)
    seeds = {}
    for sd in FINAL_SEEDS:
        o, tl = train_run(M, cache, TR, final_epochs, sd, every={"train": TR}, final={"val": VA})
        seeds[sd] = dict(val=o[final_epochs]["val"], train_top1_by_epoch=[float(np.mean([top1_rr(s, r["label"])[0] for s, r in zip(o[e]["train"], TR)])) for e in range(final_epochs + 1)],
                         train_loss_running=tl)
        if sd == 0:
            torch.save({k: v.detach().half().cpu() for k, v in M.trainable_state().items()}, os.path.join(OUT, "final_head_scorer_seed0_fp16.pt"))
        log("final seed", sd, "done", round(time.time() - t))
    meta["encoder_sha_after"] = tensor_sha(M.m.encoder)
    meta["encoder_unchanged"] = meta["encoder_sha_after"] == M.enc_sha0
    meta["type_emb_act_head_unchanged"] = (tensor_sha(M.m.type_emb) + tensor_sha(M.m.act_head)) == M.other_sha0
    if not meta["encoder_unchanged"]:
        abort("encoder tensors changed")
    vt = [np.mean([top1_rr(seeds[sd]["val"][i], r["label"])[0] for sd in FINAL_SEEDS]) for i, r in enumerate(VA)]
    vr = [np.mean([top1_rr(seeds[sd]["val"][i], r["label"])[1] for sd in FINAL_SEEDS]) for i, r in enumerate(VA)]
    vbase = {}
    for kind, name in (("probe", "probe"), ("text", "options_only"), ("prior", "action_prior")):
        sc, Cb, llC = fit_predict(kind, TR, VA, SEED_F + 5)
        vbase[name] = sc
        meta["val_C_" + name] = Cb
    rndv = ("values", [1 / r["k"] for r in VA], [sum(1 / i for i in range(1, r["k"] + 1)) / r["k"] for r in VA])
    arms = dict(head=("values", vt, vr), unchanged=unchanged_va, **vbase, random=rndv, **{"head_seed%d" % sd: seeds[sd]["val"] for sd in FINAL_SEEDS})
    valm, V = summarise(VA, arms, pairs)
    meta["stages"]["final_seconds"] = round(time.time() - t)
    g1, g2 = valm["paired"]["head_minus_probe"], valm["paired"]["head_minus_options_only"]
    passed = bool(g1["ci95"][0] > 0 and g2["ci95"][0] > 0)
    gate = dict(text="The head passes only if on VAL its top-1 minus the linear probe AND its top-1 minus the options-only baseline both have 95% bootstrap CIs above 0. Otherwise imitation experiments stop; the next step is verified-outcome labels, not encoder fine-tuning.",
                head_minus_probe=g1, head_minus_options_only=g2, passed=passed,
                verdict="PASS" if passed else "FAIL: imitation experiments stop; next step is verified-outcome labels, not encoder fine-tuning")
    meta["wall_seconds_total"] = round(time.time() - T0)
    meta["workers"] = dict(processes=1, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, gpu_mem_max_gb=torch.cuda.max_memory_allocated() / 1e9)
    meta["failures"] = []
    res = dict(meta=meta, cv=cv, val=valm, gate=gate, final_seeds={sd: dict(train_top1_by_epoch=v["train_top1_by_epoch"], train_loss_running=v["train_loss_running"]) for sd, v in seeds.items()})
    json.dump(res, open(os.path.join(OUT, "METRICS.json"), "w"), indent=1, default=str)
    with gzip.open(os.path.join(OUT, "pool_scores.jsonl.gz"), "wt") as fz:
        for i, r in enumerate(TR):
            fz.write(json.dumps(dict(set="train_oof", pool=r["pool_id"], traj=r["traj"], label=r["label"], fold=int(folds[i]),
                                     **{a: [float(x) for x in s] for a, s in (("head", oof["head"][i]), ("unchanged", unchanged_tr[i]), ("probe", oof["probe"][i]),
                                                                               ("options_only", oof["options_only"][i]), ("action_prior", oof["action_prior"][i]))})) + "\n")
        for i, r in enumerate(VA):
            fz.write(json.dumps(dict(set="val", pool=r["pool_id"], traj=r["traj"], label=r["label"], unchanged=[float(x) for x in unchanged_va[i]],
                                     **{a: [float(x) for x in vbase[a][i]] for a in vbase}, **{"head_seed%d" % sd: [float(x) for x in seeds[sd]["val"][i]] for sd in FINAL_SEEDS})) + "\n")
    with gzip.open(os.path.join(OUT, "items_manifest.jsonl.gz"), "wt") as fz:
        for split, R in (("train", TR), ("val", VA)):
            for r in R:
                for c in r["cands"]:
                    fz.write(json.dumps(dict(split=split, pool=r["pool_id"], **{k: v for k, v in c.items() if k != "cls"})) + "\n")
    write_report(res)
    log("DONE", gate["verdict"])


def write_report(res):
    m, cv, val, gate = res["meta"], res["cv"], res["val"], res["gate"]
    pct = lambda x: "%.1f%%" % (100 * x)
    ci = lambda c: "%.1f - %.1f" % (100 * c[0], 100 * c[1])
    L = ["# text-imitation v1: head-only fine-tune of LAYA Coding_Decision_Agent on its own frozen encoder", "",
         "Pre-registration `%s` sha256 %s. Kernel `%s`, script %s. Agreement with Fable's recorded action (imitation), not verified optimality." % (m["prereg"], m["prereg_sha256"], m["kernel"], m["version"]), "",
         "## Gate (pre-registered, verbatim)", "", "> " + gate["text"], "",
         "- head minus probe: %+.1f points (%s)" % (100 * gate["head_minus_probe"]["top1_diff"], ci(gate["head_minus_probe"]["ci95"])),
         "- head minus options-only: %+.1f points (%s)" % (100 * gate["head_minus_options_only"]["top1_diff"], ci(gate["head_minus_options_only"]["ci95"])),
         "- **verdict: %s**" % gate["verdict"], ""]
    for name, S in (("VAL (46 cleaned pools; final models, 3 seeds; fitted baselines on all TRAIN)", val), ("TRAIN 5-fold CV grouped by trajectory, out-of-fold (335 pools)", cv["metrics"])):
        L += ["## " + name, "", "pools %d, trajectories %d, chance top-1 %s, chance MRR %.3f; by size %s" % (S["n_pools"], S["n_trajectories"], pct(S["chance_top1"]), S["chance_mrr"], json.dumps(S["by_size"])), "",
              "| arm | top-1 | 95% CI | MRR | 95% CI | top-1 by size |", "|---|---|---|---|---|---|"]
        for a, o in S["arms"].items():
            L.append("| %s | %s | %s | %.3f | %.3f - %.3f | %s |" % (a, pct(o["top1"]), ci(o["top1_ci95"]), o["mrr"], o["mrr_ci95"][0], o["mrr_ci95"][1], json.dumps({k: round(v, 3) for k, v in o["top1_by_size"].items()})))
        L += ["", "| paired | top-1 diff | 95% CI |", "|---|---|---|"] + ["| %s | %+.1f | %s |" % (k, 100 * v["top1_diff"], ci(v["ci95"])) for k, v in S["paired"].items()] + [""]
    L += ["## Overfitting guard (outer folds, seed 0; selection by inner CV only)", "", "| fold | selected epoch | epoch | train top-1 | held-out top-1 | train logloss | held-out logloss |", "|---|---|---|---|---|---|---|"]
    for f in cv["folds"]:
        for c in f["curve"]:
            L.append("| %d | %d | %d | %s | %s | %.3f | %.3f |" % (f["fold"], f["selected_epoch"], c["epoch"], pct(c["train_top1"]), pct(c["heldout_top1"]), c["train_logloss"], c["heldout_logloss"]))
    L += ["", "selected epochs %s -> final epochs %d" % (cv["selected_epochs"], cv["final_epochs"]), "",
          "## Run record", "", "```", json.dumps({k: v for k, v in m.items() if k not in ("prepare",)}, indent=1, default=str), "```", ""]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    setup_laya()
    if MODE == "smoke":
        smoke()
    else:
        full()
