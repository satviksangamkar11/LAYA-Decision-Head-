"""Kaggle/T4 variant of tools/capture_pool_v3.py (a COPY, made because the original is frozen while the RTX 3050 TRAIN run is going; merge the flags back afterwards).
Differences, all off by default so behaviour equals the original: --model, --dtype fp16|bf16, --resident (weights stay in VRAM), --shard i/n (disjoint decision positions per worker,
file names are by manifest position so workers can share one outdir), --gate_every / --plain_every (the registered cadence is 50 / 200; the smoke test may only make it STRICTER),
a capture-device JSON provenance sidecar, peak_vram_gb in the log. Gates, thresholds and the record format are unchanged.
Resumable full-pool capture for A5-v3 (frozen Qwen3-4B-Thinking-2507, depths 18/24/30, forward pass truncated after layer 30, lm_head never run).
Rules and hard gates: configs/a5_v3_spec.toml [capture_v3] as amended by [amendment_2026_10_02_d]; thresholds: thresholds.toml [capture_v3_gates] and [capture_one_record].plain_cos_min. A failed gate ABORTS the run.
    PYTHONPATH=. .venv/Scripts/python.exe tools/capture_pool_v3.py ROLE [--limit K] [--ids id1,id2] [--outdir DIR]
ROLE is cal, train or locked. One file per decision; existing files are skipped, so the run resumes where it stopped. The locked role logs every access."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import argparse, hashlib, json, random, time, tomllib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.capture_resident import ResidentCapturer
from capture.pool import SCHEMA, render_pool
from capture.render import RenderOverflow
from compiler import a5v2, state_builder
from compiler.firewall import load_cal_calibration_only, load_dev, log_locked_access

ap = argparse.ArgumentParser()
ap.add_argument("role", choices=["cal", "train", "locked"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ids", default=None, help="comma separated decision ids (smoke tests only)")
ap.add_argument("--outdir", default="data/v3/capture")
ap.add_argument("--model", default=r"D:\local model\models\qwen3-4b-thinking-2507")
ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
ap.add_argument("--resident", action="store_true")
ap.add_argument("--shard", default=None, help="i/n: this worker handles manifest positions p with p % n == i")
ap.add_argument("--require_device", default=None, help="substring that torch.cuda.get_device_name(0) must contain (e.g. T4); aborts before loading the model otherwise")
ap.add_argument("--gate_every", type=int, default=50)
ap.add_argument("--plain_every", type=int, default=200)
args = ap.parse_args()
if args.require_device:
    assert torch.cuda.is_available() and args.require_device in torch.cuda.get_device_name(0), f"wrong device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'} (need {args.require_device})"
ROOT = Path(__file__).resolve().parents[1]
spec =tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["capture_v3"]
TH = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))
one, G = TH["capture_one_record"], TH["capture_v3_gates"]          # gates v2: [amendment_2026_10_02_d] retired indep_cos_min
D, PLAIN, SHAPE = spec["depths"], one["plain_cos_min"], G["shape_change_cos_min"]
P = args.model
assert args.gate_every <= 50 and args.plain_every <= 200, "the registered gate cadence may only be made stricter"
SH_I, SH_N = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
DT = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
man = json.load(open(ROOT / f"data/v3/manifest-{args.role}.json", encoding="utf-8"))
ids = args.ids.split(",") if args.ids else man["decision_ids"]
if args.limit:
    ids = ids[:args.limit]
want = set(ids)
if args.role == "train":
    recs = {r["decision_id"]: r for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if r["decision_id"] in want}
    raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
elif args.role == "cal":
    recs = {r["decision_id"]: r for r in load_cal_calibration_only(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if r["decision_id"] in want}
    raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
else:
    log_locked_access(f"capture_pool_v3 locked role: pool capture of {len(ids)} manifest decisions; forward passes only, no accuracy computed")
    recs = {}
    for l in open(ROOT / "data/decisions/locked-v3.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["decision_id"] in want:
            recs[r["decision_id"]] = r
    raw = a5v2.load_raw(ROOT / "data/locked_v3/fresh-locked.jsonl")
assert set(ids) <= set(recs), "manifest ids missing from the decision records"
outdir = ROOT / args.outdir / args.role
outdir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=DT).eval()
model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(D)])   # the forward pass stops after the deepest captured layer
if args.resident:
    model = model.to("cuda")
C = ResidentCapturer(model, D) if args.resident else Capturer(model, D)
outdir.mkdir(parents=True, exist_ok=True)
json.dump(dict(device=torch.cuda.get_device_name(0), dtype=args.dtype, resident=args.resident, torch=torch.__version__, transformers=__import__("transformers").__version__,
               model=str(P), revision=rev, shard=args.shard, gate_every=args.gate_every, plain_every=args.plain_every, depths=D),
          open(outdir / f"capture-device-{SH_I}of{SH_N}.json", "w"), indent=1)
(ROOT / "data/v3").mkdir(parents=True, exist_ok=True)
log = open(ROOT / "data/v3" / f"capture-log-{args.role}{'' if args.outdir == 'data/v3/capture' else '-smoke'}{'' if SH_N == 1 else f'-w{SH_I}of{SH_N}'}.jsonl", "a", encoding="utf-8")


def cos(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def gate_content_swap(R, q, cands, did):
    """Gate A (proves independence): same sequence shape, other candidates' tokens replaced (deterministic shuffle, same lengths); the target candidate vector at every
    depth and the query vector must be bit-identical. Targets: random candidates plus the NONE option."""
    suf, n = R["suffix_ids"], len(R["suffix_ids"])
    targets = sorted(set(random.Random(a5v2.seed_of(did, "swap")).sample(range(n), min(G["content_swap_targets"], n))) | {n - 1})
    worst = 0.0
    for i in targets:
        flat = [t for j, s_ in enumerate(suf) if j != i for t in s_]
        random.Random(a5v2.seed_of(did, "swapshuffle", i)).shuffle(flat)
        new, o = [], 0
        for j, s_ in enumerate(suf):
            if j == i:
                new.append(s_)
            else:
                new.append(flat[o:o + len(s_)])
                o += len(s_)
        assert [len(x) for x in new] == [len(x) for x in suf] and new[i] == suf[i], "content swap changed the shape"
        cap2, _ = C.packed(dict(prefix_ids=R["prefix_ids"], query_pos=R["query_pos"], suffix_ids=new))
        for dj, d in enumerate(D):
            dc, dq = (cap2[d][1][i] - cands[i, dj]).abs().max().item(), (cap2[d][0] - q[dj]).abs().max().item()
            worst = max(worst, dc, dq)
            assert dc <= G["content_swap_max_abs_diff"] and dq <= G["content_swap_max_abs_diff"], f"content-swap independence broken at depth {d} target {i}: candidate diff {dc}, query diff {dq}"
    return dict(targets=len(targets), max_abs_diff=worst)


def gate_shape_change(R, q, cands):
    """Gate B (numerical sanity, NOT an independence proof): half-pool recapture in reversed order, cosine >= shape_change_cos_min for every candidate and the query."""
    P1 = len(R["suffix_ids"])
    perm = list(range(P1))[::-1]
    lo = 1.0
    for part in (perm[:P1 // 2], perm[P1 // 2:]):
        cap2, _ = C.packed(dict(prefix_ids=R["prefix_ids"], query_pos=R["query_pos"], suffix_ids=[R["suffix_ids"][i] for i in part]))
        for dj, d in enumerate(D):
            c = cos(cap2[d][1], cands[part][:, dj])
            lo = min(lo, c.min().item(), cos(cap2[d][0], q[dj]).item())
            assert c.min().item() >= SHAPE, f"shape-change numerical drift above the registered bound at depth {d}: min cosine {c.min().item():.6f}"
            assert cos(cap2[d][0], q[dj]).item() >= SHAPE, f"query vector drifted with the pool composition at depth {d}"
    return dict(min_cosine=round(lo, 6))


def gate_plain(R, q, cands, did):
    picks = random.Random(a5v2.seed_of(did, "plain")).sample(range(len(R["suffix_ids"])), 5)
    for i in picks:
        cp = C.plain(R, i)
        for dj, d in enumerate(D):
            assert cos(cp[d][1][0], cands[i, dj]).item() >= PLAIN, f"plain-forward equivalence failed at depth {d} candidate {i}"
            assert cos(cp[d][0], q[dj]).item() >= PLAIN, f"plain-forward query mismatch at depth {d}"


def gate_repeat(R, cap):
    cap2, _ = C.packed(R)
    for d in D:
        assert (cap2[d][0] - cap[d][0]).abs().max().item() == 0.0 and (cap2[d][1] - cap[d][1]).abs().max().item() == 0.0, f"identical recapture differs at depth {d}"


t_start, done = time.time(), 0
for n, did in enumerate(ids):
    f = outdir / ("%05d-%s.pt" % (n, hashlib.sha256(did.encode()).hexdigest()[:8]))
    if n % SH_N != SH_I or f.exists():
        continue
    rec = recs[did]
    tid, t = did.rsplit(":", 1)
    ev = raw[tid]["trajectory"]
    t0 = time.time()
    try:
        st = state_builder.build_state(ev, int(t), 32000)
        assert st["state_hash"] == rec["state_hash"], "state rebuild differs from the compiled record"
        R = render_pool(rec, ev, st, tok, rev)
    except RenderOverflow as e:
        log.write(json.dumps({"n": n, "decision_id": did, "skipped": str(e)}) + "\n"); log.flush()
        continue
    cap, (tid_ids, spans) = C.packed(R)
    assert [tid_ids[0, a:b].tolist() for a, b in spans] == R["suffix_ids"], "span check failed"
    q = torch.stack([cap[d][0] for d in D])
    cands = torch.stack([cap[d][1] for d in D], 1)
    assert torch.isfinite(q).all() and torch.isfinite(cands).all(), "non-finite hidden state"
    assert max(q.abs().max().item(), cands.abs().max().item()) < 60000, "value outside the fp16 storage range"
    gates, detail = [], {}
    if n < 3 or n % args.gate_every == 0:
        detail["content_swap"] = gate_content_swap(R, q, cands, did); detail["shape_change"] = gate_shape_change(R, q, cands); gates += ["content_swap", "shape_change"]
    if n == 0 or n % args.plain_every == 0:
        gate_plain(R, q, cands, did); gate_repeat(R, cap); gates += ["plain", "repeat"]
    torch.save(dict(schema=SCHEMA, decision_id=did, role=args.role, texts=R["texts"], pool_hash=R["pool_hash"], prompt_hash=R["prompt_hash"], state_hash=R["state_hash"],
                    depths=D, query=q.half(), cands=cands.half(), n_prefix_tokens=R["n_prefix_tokens"], pool_size=R["pool_size"], tokenizer_revision=rev), f)
    done += 1
    log.write(json.dumps({"n": n, "decision_id": did, "seconds": round(time.time() - t0, 2), "prefix_tokens": R["n_prefix_tokens"], "suffix_tokens": R["n_suffix_tokens"],
                          "pool": R["pool_size"], "gates": gates, "gate_detail": detail,
                          "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n"); log.flush()
    if done % 20 == 0:
        print(f"{args.role} {n + 1}/{len(ids)} captured={done} elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)
assert C.lm_calls == 0, "lm_head was run"
print("DONE", args.role, "captured", done, "of", len(ids), "minutes", round((time.time() - t_start) / 60, 1), "lm_head_calls", C.lm_calls)
