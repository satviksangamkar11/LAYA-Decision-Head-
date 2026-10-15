"""Kaggle T4 x2 kernel: outcome-head v1. Train the pretrained LAYA Coding_Decision_Agent head (frozen ORIGINAL encoder) to predict whether a
coding-agent run ends resolved, from the pre-action state + the step's action (Nebius SWE-rebench-openhands runs, test-executed labels).
Pre-registration: thresholds_outcome_head_v1.toml (mounted dataset; sha256 checked). Renderer k2_render_v1 and the model wrapper are copied
from kaggle/text_imitation_v1.py (which copied them from kaggle/text_grader_k2_v2.py).
MODE "smoke": mechanics only (no performance metric). MODE "full": everything; test scored once.
GPU 0 trains; GPU 1 (subprocess, STAGE=passA) computes the unchanged-head scores and frozen CLS features of every item in parallel.
"""
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time

MODE = "full"            # smoke | full
VERSION = "oh1-2026-10-15a"
KERNEL_SLUG = "sangeetasangamkar/outcome-head-v1"
PREREG_NAME = "thresholds_outcome_head_v1.toml"
PREREG_SHA256 = "e8209a04c6632a36c305387bc9f255f7d6e8e872e3d677b2a439c35dde8bc8bb"
DATA_SHA256 = "487d1b6946863cfccbde70f486c9e413e6d2e591f25233b88934721a8c218652"
ROOT_IN, OUT, WORK = "/kaggle/input", "/kaggle/working/out", "/kaggle/working/tmp"
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
DEPTHS, MIN_STEPS = [3, 10, 20], 21
SPLIT_SEED, VAL_REPOS, TEST_REPOS, VAL_MAX_TASKS, TEST_MAX_TASKS, TRAIN_MAX_RUNS = 20261015, 86, 172, 150, 350, 4000
LR_GRID, E_MAX, WD, CLIP, ACCUM, WARM = [1e-4, 3e-4, 1e-3], 5, 0.01, 1.0, 32, 0.10
OVERFIT_N, OVERFIT_STEPS, OVERFIT_DROP = 32, 60, 0.5
C_TEXT, C_PROBE = [0.01, 0.1, 1.0, 10.0], [0.001, 0.01, 0.1, 1.0]
SEED_B, B = 20261016, 2000
VAL_EVAL_MAX = 2000
T0 = time.time()
SELF = os.path.abspath(globals()["__file__"]) if globals().get("__file__") else None


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
# renderer k2_render_v1 (verbatim)
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
# Nebius OpenHands runs -> steps (category rule verbatim from the branching kernel laya-nebius-branch.py)
# ------------------------------------------------------------------------------------------------------------------------------------
DROP_ARGS = {"is_input", "timeout", "security_risk", "thought"}
ISSUE = re.compile(r"<issue_description>(.*?)</issue_description>", re.S)


def category(name, args):
    if name == "finish":
        return "stop"
    if name == "task_tracker":
        return "plan"
    if name == "str_replace_editor":
        return "read" if args.get("command") == "view" else "edit"
    if name != "execute_bash":
        return "other"
    c = (args.get("command") or "").strip()
    first = c.split()[0] if c.split() else ""
    if re.search(r"\bsed\s+-i|(^|[^2&])>\s*[\w./]|\btee\b|\bpatch\b|git apply", c):
        return "edit"
    if re.search(r"\b(pytest|unittest|tox|nosetests)\b|python\S*\s+\S*test|go test|cargo test|npm (run )?test|make test", c):
        return "test"
    if first in ("git",):
        return "vcs"
    if re.search(r"\b(pip|pip3|apt|apt-get|conda|npm install|yarn add|poetry)\b", c):
        return "env"
    if first in ("grep", "rg", "find", "ls", "tree", "ag", "locate") or re.match(r"(cd \S+ && )?(grep|rg|find|ls)\b", c):
        return "search"
    if first in ("cat", "head", "tail", "less", "more", "nl", "wc") or re.match(r"(cd \S+ && )?(cat|head|tail|sed -n)\b", c):
        return "read"
    if re.search(r"\b(python|python3|node|go run|make|bash|sh|\./)", c):
        return "run"
    return "other"


def parse_run(msgs):
    """-> (task text, steps [(action text, category, observation)]) or None. Steps = non-think tool calls in order."""
    if len(msgs) < 2 or msgs[1]["role"] != "user":
        return None
    u = msgs[1]["content"] or ""
    m = ISSUE.search(u)
    task = (m.group(1) if m else u).strip()
    outputs = {x.get("tool_call_id"): (x.get("content") or "") for x in msgs if x["role"] == "tool"}
    steps = []
    for x in msgs[2:]:
        if x["role"] != "assistant":
            continue
        for c in x.get("tool_calls") or []:
            name = c["function"]["name"]
            if name == "think":
                continue
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                args = {"_raw": c["function"]["arguments"]}
            a = {k: v for k, v in args.items() if k not in DROP_ARGS} if isinstance(args, dict) else {"_raw": str(args)}
            steps.append(("call: %s(%s)" % (name, json.dumps(a, ensure_ascii=False, separators=(",", ":"))), category(name, a), outputs.get(c.get("id"), "")))
    return task, steps


def render_item(tk, task, steps, d, budget):
    """pre-action state at depth d: proposed action = steps[d]; history = steps[:d] newest first; nothing at or after d's observation."""
    T, T_cut = tk.clip_tokens(clip(norm(task), 6000), T_TOK)
    units = []
    for a, _, o in reversed(steps[:d]):
        body = "[%s] -> [%s]" % (clip(norm(a[:8000]), H_ACT_CH), clip(norm(o[:1500]), H_OBS_CH))
        units.append((body, len(tk.ids(body))))
    A, A_cut = tk.clip_tokens(clip(norm(steps[d][0]), 8000), A_TOK)
    st, nt, kept, ovf = build_state(tk, A, None, T, units, budget)
    return st, dict(tokens=nt, hist_kept=kept, hist_avail=len(units), A_cut=A_cut, T_cut=T_cut, fixed_overflow=ovf)


# ------------------------------------------------------------------------------------------------------------------------------------
# data + split
# ------------------------------------------------------------------------------------------------------------------------------------
def load_runs():
    import pyarrow.parquet as pq
    p = find(lambda q: q.endswith("nebius_mixed.parquet"))
    ov = find(lambda q: q.endswith("laya_pretraining_overlap_repos.json"))
    pre = find(lambda q: q.endswith(PREREG_NAME))
    if len(p) != 1 or len(ov) != 1 or len(pre) != 1:
        abort("dataset files not found / ambiguous", parquet=p, overlap=ov, prereg=pre)
    if sha256_file(pre[0]) != PREREG_SHA256:
        abort("pre-registration sha256 differs")
    if sha256_file(p[0]) != DATA_SHA256:
        abort("parquet sha256 differs")
    overlap = {r.replace("/", "__").lower() for r in json.load(open(ov[0]))["repos"]} | {r.lower() for r in json.load(open(ov[0]))["repos"]}
    f = pq.ParquetFile(p[0])
    runs = []
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["trajectory_id", "instance_id", "repo", "trajectory", "resolved"])
        for tid, iid, repo, tr, res in zip(*[t.column(c).to_pylist() for c in ("trajectory_id", "instance_id", "repo", "trajectory", "resolved")]):
            pr = parse_run(tr)
            runs.append(dict(tid=tid, iid=iid, repo=repo, y=int(bool(res)), parsed=pr))
    return runs, overlap


def make_split(runs, overlap):
    rng = random.Random(SPLIT_SEED)
    repos = sorted({r["repo"] for r in runs})
    is_ov = {x: (x.lower() in overlap or x.replace("/", "__").lower() in overlap) for x in repos}
    non = [x for x in repos if not is_ov[x]]
    rng.shuffle(non)
    val_r, test_r = set(non[:VAL_REPOS]), set(non[VAL_REPOS:VAL_REPOS + TEST_REPOS])
    role = {x: ("val" if x in val_r else "test" if x in test_r else "train") for x in repos}
    excl = dict(unparsed={0: 0, 1: 0}, too_short={0: 0, 1: 0})
    ok = []
    for r in runs:
        if r["parsed"] is None:
            excl["unparsed"][r["y"]] += 1
        elif len(r["parsed"][1]) < MIN_STEPS:
            excl["too_short"][r["y"]] += 1
        else:
            ok.append(r)
    by = {"train": [], "val": [], "test": []}
    for r in ok:
        by[role[r["repo"]]].append(r)

    def cap_tasks(rs, n):
        tasks = sorted({r["iid"] for r in rs})
        keep = set(random.Random(SPLIT_SEED + n).sample(tasks, min(n, len(tasks))))
        return [r for r in rs if r["iid"] in keep]
    by["val"] = cap_tasks(by["val"], VAL_MAX_TASKS)
    by["test"] = cap_tasks(by["test"], TEST_MAX_TASKS)
    tr = sorted(by["train"], key=lambda r: r["tid"])
    by["train"] = random.Random(SPLIT_SEED + 1).sample(tr, min(TRAIN_MAX_RUNS, len(tr)))
    assert not ({r["repo"] for r in by["test"]} & {r["repo"] for r in by["train"]}), "repo leak train/test"
    assert not any(is_ov[r["repo"]] for r in by["val"] + by["test"]), "overlap repo outside train"
    info = dict(repos=len(repos), overlap_repos=sum(is_ov.values()), excluded_runs=excl,
                **{s: dict(runs=len(v), tasks=len({r["iid"] for r in v}), repos=len({r["repo"] for r in v}), resolved_share=sum(r["y"] for r in v) / max(1, len(v)))
                   for s, v in by.items()})
    return by, info


# ------------------------------------------------------------------------------------------------------------------------------------
# model (wrapper from text_imitation_v1.py; question = outcome)
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
        assert torch.cuda.is_available(), "no CUDA: this job never falls back to the CPU"
        self.agent = laya.load(ck, device="cuda")
        self.m = self.agent.model.eval()
        self.dev = torch.device("cuda")
        self.tk = Tok(_load_tokenizer(os.path.join(ck, "tokenizer"), json.load(open(os.path.join(ck, "rl_agent_config.json")))))
        self.q = Agent._to_internal(QUESTIONS["outcome"])
        self.qtype = QTYPES["choice"]
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.enc_sha0 = tensor_sha(self.m.encoder)
        self.other_sha0 = tensor_sha(self.m.type_emb) + tensor_sha(self.m.act_head)

    def make_trainable(self):
        self.m.head.float()
        self.m.scorer.float()
        for p in list(self.m.head.parameters()) + list(self.m.scorer.parameters()):
            p.requires_grad_(True)
        self.orig = {k: v.detach().clone() for k, v in self.trainable_state().items()}
        self.orig_sha = tensor_sha(self.m.head) + tensor_sha(self.m.scorer)

    def trainable_state(self):
        return {**{"head." + k: v for k, v in self.m.head.state_dict().items()}, **{"scorer." + k: v for k, v in self.m.scorer.state_dict().items()}}

    def load_state(self, st):
        import torch
        with torch.no_grad():
            cur = self.trainable_state()
            for k, v in st.items():
                cur[k].copy_(v)

    def seq(self, state):
        from laya.common import build_sequence
        ids, mk, _, ts = build_sequence(self.tk.tok, state, self.q, MAX_LEN, HEAD_MAX_LEN, return_stats=True, return_truncation_stats=True)
        return ids, mk, int(ts["state_tokens_dropped"])

    def encode(self, ids):
        import torch
        x = torch.tensor([ids], device=self.dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = self.m.encoder(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        return h[0]

    def head_logits(self, h, mk):
        """h: [L, d] encoder output on the device; returns [4] scorer logits (library forward from the encoder output on)."""
        import torch
        H = h[None]
        am = torch.ones(1, h.shape[0], dtype=torch.long, device=self.dev)
        mp = torch.tensor([mk], device=self.dev)
        qt = torch.full((1,), self.qtype, device=self.dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16):
            x = H + self.m.type_emb(qt)[:, None, :]
            pad = ~am.bool()
            for layer in self.m.head.layers:
                x = layer(x, src_key_padding_mask=pad)
            idx = mp[:, :, None].expand(-1, -1, x.size(-1))
            lg = self.m.scorer(torch.gather(x, 1, idx)).squeeze(-1).float()
        return lg[0]

    def full_forward(self, ids, mk):
        import torch
        x = torch.tensor([ids], device=self.dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg, _ = self.m(x, torch.ones_like(x), torch.tensor([mk], device=self.dev), torch.ones(1, len(mk), dtype=torch.bool, device=self.dev),
                           torch.tensor([self.qtype], device=self.dev))
        return lg[0].float().cpu()


def score_of(lg):
    """logit(success vs rest) and log p(success), log p(not success)."""
    import torch
    lse_all = torch.logsumexp(lg, 0)
    lse_rest = torch.logsumexp(lg[1:], 0)
    return lg[0] - lse_rest, lg[0] - lse_all, lse_rest - lse_all


# ------------------------------------------------------------------------------------------------------------------------------------
# items
# ------------------------------------------------------------------------------------------------------------------------------------
def build_items(M, by, budget):
    items, t = [], time.time()
    for split in ("train", "val", "test"):
        for r in by[split]:
            task, steps = r["parsed"]
            for d in DEPTHS:
                st, meta = render_item(M.tk, task, steps, d, budget)
                ids, mk, dropped = M.seq(st)
                if len(mk) != 4:
                    continue
                from laya.common import serialize_state
                items.append(dict(id="%s@%d" % (r["tid"], d), split=split, tid=r["tid"], iid=r["iid"], repo=r["repo"], y=r["y"], depth=d,
                                  cat=steps[d][1], prev=steps[d - 1][1], ids=ids, mk=mk, dropped=dropped, text=serialize_state(st), **meta))
        log("rendered", split, sum(1 for i in items if i["split"] == split), "items; seconds", round(time.time() - t))
    return items


def survival(items):
    n = len(items)
    lib = [i["dropped"] > 0 for i in items]
    return dict(items=n, action_intact_share=sum((not i["A_cut"]) and (not i["fixed_overflow"]) and (not l) for i, l in zip(items, lib)) / n,
                task_survived_share=sum((not i["fixed_overflow"]) and (not l) for i, l in zip(items, lib)) / n,
                A_cut=sum(i["A_cut"] for i in items), T_cut=sum(i["T_cut"] for i in items), library_truncated=sum(lib),
                tokens_mean=sum(i["tokens"] for i in items) / n, seqlen_max=max(len(i["ids"]) for i in items),
                history_kept_mean=sum(i["hist_kept"] for i in items) / n, history_avail_mean=sum(i["hist_avail"] for i in items) / n)


# ------------------------------------------------------------------------------------------------------------------------------------
# pass A on GPU 1: unchanged-head score + frozen CLS for every item
# ------------------------------------------------------------------------------------------------------------------------------------
def pass_a():
    import numpy as np
    import torch
    setup_laya()
    M = Model()
    seqs = pickle.load(open(os.path.join(WORK, "seqs.pkl"), "rb"))
    sc, cls, t = np.zeros(len(seqs), "float32"), np.zeros((len(seqs), 1024), "float16"), time.time()
    with torch.no_grad():
        for k, (ids, mk) in enumerate(seqs):
            h = M.encode(ids)
            cls[k] = h[0].float().cpu().numpy()
            sc[k] = float(score_of(M.head_logits(h, mk))[0])
            if (k + 1) % 2000 == 0:
                log("passA", k + 1, "of", len(seqs), "seconds", round(time.time() - t))
    np.savez(os.path.join(WORK, "passA.npz"), score=sc, cls=cls)
    json.dump(dict(items=len(seqs), seconds=round(time.time() - t), encoder_sha=tensor_sha(M.m.encoder), enc_sha_ok=tensor_sha(M.m.encoder) == M.enc_sha0,
                   nonfinite=int((~np.isfinite(sc)).sum())), open(os.path.join(WORK, "passA.json"), "w"))
    log("passA done")


# ------------------------------------------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------------------------------------------
def item_loss(M, it, w1):
    import torch
    lg = M.head_logits(M.encode(it["ids"]), it["mk"])
    s, lp, lq = score_of(lg)
    loss = -(w1 * it["y"] * lp + (1 - it["y"]) * lq)
    return loss, s


def eval_scores(M, its):
    import numpy as np
    import torch
    M.m.head.eval()
    M.m.scorer.eval()
    out = np.zeros(len(its))
    with torch.no_grad():
        for k, it in enumerate(its):
            out[k] = float(score_of(M.head_logits(M.encode(it["ids"]), it["mk"]))[0])
    return out


def overfit_check(M, train_items):
    """pre-registered lr rule: 32 fixed train items, 60 full-batch AdamW updates per lr from the original weights."""
    import torch
    sub = random.Random(SPLIT_SEED + 7).sample(train_items, OVERFIT_N)
    w1 = 1.0
    H = [(M.encode(i["ids"]).detach(), i["mk"], i["y"]) for i in sub]
    res = {}
    for lr in LR_GRID:
        M.load_state(M.orig)
        params = list(M.m.head.parameters()) + list(M.m.scorer.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
        M.m.head.train()
        M.m.scorer.train()
        first = last = None
        for step in range(OVERFIT_STEPS):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for h, mk, y in H:
                _, lp, lq = score_of(M.head_logits(h, mk))
                loss = -(w1 * y * lp + (1 - y) * lq) / len(H)
                loss.backward()
                tot += float(loss)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            first = tot if first is None else first
            last = tot
        res[str(lr)] = dict(loss_first=first, loss_last=last, drop=1 - last / first if first else None)
        log("overfit lr", lr, res[str(lr)])
    M.load_state(M.orig)
    passing = [lr for lr in LR_GRID if res[str(lr)]["drop"] is not None and res[str(lr)]["drop"] > OVERFIT_DROP]
    return (min(passing) if passing else None), res


def train(M, tr, va, lr, w1, epochs):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    M.load_state(M.orig)
    torch.manual_seed(0)
    params = list(M.m.head.parameters()) + list(M.m.scorer.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD, betas=(0.9, 0.999), eps=1e-8)
    steps_total = epochs * ((len(tr) + ACCUM - 1) // ACCUM)
    warm = max(1, int(WARM * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps_total - s) / max(1, steps_total - warm)))
    scaler = torch.amp.GradScaler("cuda")
    vsub = va if len(va) <= VAL_EVAL_MAX else random.Random(SPLIT_SEED + 3).sample(va, VAL_EVAL_MAX)
    yv = np.array([i["y"] for i in vsub])
    hist, best = [], (-1.0, None, 0)
    for e in range(1, epochs + 1):
        M.m.head.train()
        M.m.scorer.train()
        order = np.random.default_rng(1000 + e).permutation(len(tr))
        opt.zero_grad(set_to_none=True)
        el, t = [], time.time()
        for k, pi in enumerate(order):
            loss, _ = item_loss(M, tr[pi], w1)
            if not torch.isfinite(loss):
                abort("non-finite loss", item=tr[pi]["id"], epoch=e)
            el.append(float(loss))
            scaler.scale(loss / ACCUM).backward()
            if (k + 1) % ACCUM == 0 or k + 1 == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if (k + 1) % 2000 == 0:
                log("epoch", e, "item", k + 1, "of", len(order), "mean loss", round(float(np.mean(el[-2000:])), 4), "seconds", round(time.time() - t))
        sv = eval_scores(M, vsub)
        auc = float(roc_auc_score(yv, sv))
        hist.append(dict(epoch=e, train_loss=float(np.mean(el)), val_auroc=auc, seconds=round(time.time() - t)))
        log("epoch", e, json.dumps(hist[-1]))
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in M.trainable_state().items()}, e)
    M.load_state(best[1])
    return hist, best[2], best[0]


# ------------------------------------------------------------------------------------------------------------------------------------
# baselines and metrics
# ------------------------------------------------------------------------------------------------------------------------------------
def fit_lr(Xtr, ytr, Xva, yva, Xte, grid):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    best = None
    for C in grid:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        a = roc_auc_score(yva, m.decision_function(Xva))
        if best is None or a > best[0]:
            best = (a, C, m)
    return best[2].decision_function(Xva), best[2].decision_function(Xte), best[1]


def baselines(items, A):
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    S = {s: [k for k, i in enumerate(items) if i["split"] == s] for s in ("train", "val", "test")}
    y = np.array([i["y"] for i in items])
    out = {}
    oh = OneHotEncoder(handle_unknown="ignore").fit([[i["depth"]] for i in items])
    X = oh.transform([[i["depth"]] for i in items])
    out["step_index"] = fit_lr(X[S["train"]], y[S["train"]], X[S["val"]], y[S["val"]], X[S["test"]], [1.0])
    oh2 = OneHotEncoder(handle_unknown="ignore").fit([[items[k]["cat"], items[k]["prev"]] for k in S["train"]])
    X = oh2.transform([[i["cat"], i["prev"]] for i in items])
    out["action_type"] = fit_lr(X[S["train"]], y[S["train"]], X[S["val"]], y[S["val"]], X[S["test"]], [1.0])
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200000).fit([items[k]["text"] for k in S["train"]])
    X = sparse.csr_matrix(v.transform([i["text"] for i in items]))
    out["tfidf"] = fit_lr(X[S["train"]], y[S["train"]], X[S["val"]], y[S["val"]], X[S["test"]], C_TEXT)
    sc = StandardScaler().fit(A["cls"][S["train"]].astype("float32"))
    X = sc.transform(A["cls"].astype("float32"))
    out["probe"] = fit_lr(X[S["train"]], y[S["train"]], X[S["val"]], y[S["val"]], X[S["test"]], C_PROBE)
    return out


def metrics(te_items, val_items, arms_val, arms_te):
    """arms_*: name -> score array aligned with val_items / te_items. AUROC, accuracy at val threshold, within-task matched-depth pair AUROC,
    cluster bootstrap over test repos; paired differences for the gate."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y = np.array([i["y"] for i in te_items])
    yv = np.array([i["y"] for i in val_items])
    repos = sorted({i["repo"] for i in te_items})
    ri = np.array([repos.index(i["repo"]) for i in te_items])
    G = len(repos)
    rng = np.random.default_rng(SEED_B)
    W = np.zeros((B, G))
    np.add.at(W, (np.repeat(np.arange(B), G), rng.integers(0, G, (B, G)).ravel()), 1.0)
    groups = {}
    for k, i in enumerate(te_items):
        groups.setdefault((i["iid"], i["depth"]), []).append(k)
    gl = [(g, [k for k in ks if y[k] == 1], [k for k in ks if y[k] == 0]) for g, ks in groups.items()]
    gl = [(g, p, n) for g, p, n in gl if p and n]
    g_repo = np.array([ri[p[0]] for _, p, _ in gl])

    def pair_stats(s):
        wins = np.array([sum((s[a] > s[b]) + 0.5 * (s[a] == s[b]) for a in p for b in n) for _, p, n in gl], float)
        cnt = np.array([len(p) * len(n) for _, p, n in gl], float)
        return wins, cnt

    def thr(sv):
        cands = np.unique(sv)
        accs = [((sv >= c) == yv).mean() for c in cands]
        return cands[int(np.argmax(accs))]
    res, boots = {}, {}
    for a in arms_te:
        s = np.asarray(arms_te[a], float)
        wins, cnt = pair_stats(s)
        th = thr(np.asarray(arms_val[a], float))
        acc = ((s >= th) == y).astype(float)
        bauc = []
        for b in range(B):
            w = W[b][ri]
            m = w > 0
            bauc.append(roc_auc_score(y[m], s[m], sample_weight=w[m]) if len(set(y[m])) == 2 else np.nan)
        wr = W[:, g_repo]
        bpair = (wr @ wins) / (wr @ cnt)
        bacc = (W @ np.bincount(ri, weights=acc, minlength=G)) / (W @ np.bincount(ri, minlength=G))
        boots[a] = dict(auc=np.array(bauc), pair=bpair)
        res[a] = dict(auroc=float(roc_auc_score(y, s)), auroc_ci95=list(np.nanpercentile(bauc, [2.5, 97.5])),
                      pair_auroc=float(wins.sum() / cnt.sum()), pair_auroc_ci95=list(np.percentile(bpair, [2.5, 97.5])),
                      accuracy=float(acc.mean()), accuracy_ci95=list(np.percentile(bacc, [2.5, 97.5])), threshold_from_val=float(th))
    diffs = {}
    for b_ in ("unchanged_head", "step_index", "tfidf", "probe", "action_type"):
        if "trained_head" in boots and b_ in boots:
            da = boots["trained_head"]["auc"] - boots[b_]["auc"]
            dp = boots["trained_head"]["pair"] - boots[b_]["pair"]
            diffs["trained_head_minus_" + b_] = dict(auroc_diff=res["trained_head"]["auroc"] - res[b_]["auroc"], auroc_diff_ci95=list(np.nanpercentile(da, [2.5, 97.5])),
                                                     pair_diff=res["trained_head"]["pair_auroc"] - res[b_]["pair_auroc"], pair_diff_ci95=list(np.percentile(dp, [2.5, 97.5])))
    gate = all(diffs.get("trained_head_minus_" + b_, {}).get("auroc_diff_ci95", [0])[0] > 0 and diffs["trained_head_minus_" + b_]["pair_diff_ci95"][0] > 0
               for b_ in ("unchanged_head", "step_index"))
    return dict(n_test_items=len(te_items), n_test_repos=G, n_test_pairs=int(sum(len(p) * len(n) for _, p, n in gl)), n_pair_groups=len(gl),
                test_resolved_share=float(y.mean()), arms=res, differences=diffs, gate_pass=bool(gate))


# ------------------------------------------------------------------------------------------------------------------------------------
def main():
    import numpy as np
    import torch
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    setup_laya()
    meta = dict(version=VERSION, kernel=KERNEL_SLUG, mode=MODE, prereg_sha256=PREREG_SHA256, stages={})
    runs, overlap = load_runs()
    by, meta["split"] = make_split(runs, overlap)
    del runs
    log("split", json.dumps(meta["split"]))
    if MODE == "smoke":
        by = {s: random.Random(5).sample(v, min(len(v), n)) for (s, v), n in zip(by.items(), (40, 10, 10))}
    M = Model()
    budget, rooms = state_budget(M.tk)
    assert budget == 1969, "state token budget differs from K2 (1969)"
    items = build_items(M, by, budget)
    meta["survival"] = survival(items)
    meta["items"] = {s: sum(1 for i in items if i["split"] == s) for s in ("train", "val", "test")}
    log("survival", json.dumps(meta["survival"]))
    # leakage assertion: the step's own observation never appears in its rendered state (pre-action)
    leak = 0
    for i in items[:500]:
        r = next(x for x in by[i["split"]] if x["tid"] == i["tid"])
        o = norm(r["parsed"][1][i["depth"]][2])[:120]
        leak += bool(len(o) >= 40 and o in i["text"] and o not in norm(" ".join(s[2] for s in r["parsed"][1][:i["depth"]])))
    meta["own_observation_leak_in_first_500"] = leak
    if leak:
        abort("own observation found in pre-action state", n=leak)
    pickle.dump([(i["ids"], i["mk"]) for i in items], open(os.path.join(WORK, "seqs.pkl"), "wb"))
    nga = torch.cuda.device_count()
    if not SELF:
        abort("script path unknown (run as a script kernel)")
    pa = subprocess.Popen([sys.executable, SELF], env=dict(os.environ, OH_STAGE="passA", CUDA_VISIBLE_DEVICES="1" if nga > 1 else "0"))
    # library parity on 3 items (unchanged model): agent.predict probabilities vs softmax(full forward / T)
    from laya.common import temp_bucket, QTYPES
    par = []
    for i in items[:3]:
        r = next(x for x in by[i["split"]] if x["tid"] == i["tid"])
        st, _ = render_item(M.tk, r["parsed"][0], r["parsed"][1], i["depth"], budget)
        res = M.agent.predict(st, {"outcome": QUESTIONS["outcome"]}, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)["answers"]["outcome"]
        lg = M.full_forward(i["ids"], i["mk"]).double().numpy()
        T = M.agent.temperature_by_options.get(temp_bucket(QTYPES["choice"], 4), M.agent.temperature[QTYPES["choice"]])
        p = np.exp(lg / T - (lg / T).max())
        p /= p.sum()
        par.append(float(np.abs(p - np.array([res["probabilities"][k] for k in QUESTIONS["outcome"]["criteria"]])).max()))
    meta["predict_vs_full_forward_max_prob_diff"] = max(par)
    M.make_trainable()
    with torch.no_grad():
        d = max(abs(float(M.head_logits(M.encode(i["ids"]), i["mk"])[0]) - float(M.full_forward(i["ids"], i["mk"])[0])) for i in items[:5])
    meta["head_path_vs_full_forward_max_abs_logit_diff"] = d
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    te = [i for i in items if i["split"] == "test"]
    lr, meta["overfit"] = overfit_check(M, tr)
    meta["lr_selected"] = lr
    n1 = sum(i["y"] for i in tr)
    w1 = (len(tr) - n1) / max(1, n1)
    meta["w1"] = w1
    if lr is None:
        pa.wait()
        json.dump(meta, open(os.path.join(OUT, "META.json"), "w"), indent=1, default=str)
        abort("no lr passed the overfit check", overfit=meta["overfit"])
    epochs = 1 if MODE == "smoke" else E_MAX
    t = time.time()
    meta["train_history"], meta["best_epoch"], meta["best_val_auroc"] = train(M, tr, va, lr, w1, epochs)
    meta["stages"]["train_seconds"] = round(time.time() - t)
    meta["encoder_unchanged"] = tensor_sha(M.m.encoder) == M.enc_sha0
    meta["type_emb_act_head_unchanged"] = (tensor_sha(M.m.type_emb) + tensor_sha(M.m.act_head)) == M.other_sha0
    meta["encoder_requires_grad_any"] = any(p.requires_grad for p in M.m.encoder.parameters())
    meta["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if not (meta["encoder_unchanged"] and meta["type_emb_act_head_unchanged"] and not meta["encoder_requires_grad_any"]):
        abort("frozen-part check failed", **{k: meta[k] for k in ("encoder_unchanged", "type_emb_act_head_unchanged", "encoder_requires_grad_any")})
    if MODE == "smoke":
        pa.wait()
        meta["passA"] = json.load(open(os.path.join(WORK, "passA.json")))
        meta["note"] = "smoke: mechanics only, no performance metric computed"
        meta["ok"] = bool(meta["predict_vs_full_forward_max_prob_diff"] < 1e-3 and d < 0.05 and pa.returncode == 0 and meta["passA"]["enc_sha_ok"])
        json.dump(meta, open(os.path.join(OUT, "SMOKE_MECHANICS.json"), "w"), indent=1, default=str)
        log("SMOKE DONE", json.dumps({k: meta[k] for k in ("ok", "items", "lr_selected", "predict_vs_full_forward_max_prob_diff",
                                                           "head_path_vs_full_forward_max_abs_logit_diff")}, default=str))
        return
    t = time.time()
    s_va, s_te = eval_scores(M, va), eval_scores(M, te)
    meta["stages"]["eval_seconds"] = round(time.time() - t)
    pa.wait()
    if pa.returncode != 0:
        abort("passA failed", rc=pa.returncode)
    meta["passA"] = json.load(open(os.path.join(WORK, "passA.json")))
    A = dict(np.load(os.path.join(WORK, "passA.npz")))
    idx = {s: [k for k, i in enumerate(items) if i["split"] == s] for s in ("val", "test")}
    bl = baselines(items, A)
    arms_val = dict(trained_head=s_va, unchanged_head=A["score"][idx["val"]], **{a: v[0] for a, v in bl.items()})
    arms_te = dict(trained_head=s_te, unchanged_head=A["score"][idx["test"]], **{a: v[1] for a, v in bl.items()})
    meta["baseline_C"] = {a: v[2] for a, v in bl.items()}
    res = metrics(te, va, arms_val, arms_te)
    res["meta"] = meta
    json.dump(res, open(os.path.join(OUT, "METRICS.json"), "w"), indent=1, default=str)
    with gzip.open(os.path.join(OUT, "test_scores.jsonl.gz"), "wt") as f:
        for k, i in enumerate(te):
            f.write(json.dumps(dict(id=i["id"], repo=i["repo"], iid=i["iid"], depth=i["depth"], y=i["y"], **{a: float(v[k]) for a, v in arms_te.items()})) + "\n")
    torch.save({k: v.half().cpu() for k, v in M.trainable_state().items()}, os.path.join(OUT, "trained_head_scorer_fp16.pt"))
    log("FULL DONE gate_pass", res["gate_pass"], json.dumps({a: (round(v["auroc"], 4), round(v["pair_auroc"], 4)) for a, v in res["arms"].items()}))


if __name__ == "__main__":
    if os.environ.get("OH_STAGE") == "passA":
        pass_a()
    else:
        main()
