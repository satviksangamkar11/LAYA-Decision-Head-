"""Frozen-Qwen feature capture for the Fable head (protocol ff.PROTOCOL = fable_capture_v1). Device-agnostic (cpu | xla | cuda).

What one record is (see decision_head/fable_features.py): for a request {context_text, options}
  prefix  = tokens(context_text + query_cue)               (add_special_tokens=False)
  suffix_i= tokens(option_prefix + option_i)               (independent suffix of the SAME prefix)
  hq      = raw residual stream after decoder layer d (d in depths) at the LAST prefix token          -> [K,H]
  ho[i]   = fp32 mean of that stream over option i's own suffix tokens                                  -> [N,K,H]
Both are saved as float16 tensors. One packed forward per decision: ids [prefix, suffix_1..suffix_N], block attention mask
(each suffix sees the whole prefix and its own earlier tokens only), position ids restarting at len(prefix) for every suffix
(`capture.capture.pack`, the proven packed design), padded to a size bucket (padded rows attend only to themselves, real rows
never attend to padding). The forward stops after the deepest captured layer: no final norm, no LM head, no generation, no grad.
Input is the request file ONLY (no private labels / gold / provenance are ever opened).

Nothing is clipped: a decision whose prefix / an option / the total exceeds the limits, or whose capture fails or overflows
float16, goes to the capture-exclusion sidecar with a reason. Every requested id ends up captured XOR excluded.

  PYTHONPATH=. .venv/Scripts/python.exe tools/fable_capture.py preflight --requests R.jsonl.gz [--out report.json]    (tokenizer only)
  ... run    --requests R --out DIR --device cpu|xla|cuda [--dtype bfloat16] [--limit N] [--chunk-range a:b]       (loads the model)
  ... parity --requests R --device ... [--n 3]       (packed+padded vs separate plain forwards, bucket invariance; loads the model)
  ... verify --requests R --out DIR                  (coverage + record checks + measured timing; no model)
Nothing here installs packages or downloads: the model and tokenizer load with local_files_only from --model-dir.
A real run (`run`, `parity`) first sha256-verifies the weight shards against the pin (about 8 GB read) and records those hashes in the
binding; `preflight` is tokenizer-only and checks only the small files and the shard SIZES (it says so in its report).
Concurrency: one writer per chunk. The binding file is created atomically (hard link of a fully written temp file), so processes may
start together, but they must use DISJOINT --chunk-range values on the same output directory; overlapping ranges are not supported.
"""
import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture.capture import pack
from decision_head import fable_features as ff
from decision_head.fable_pool import file_sha, jdump, read_gz, sha

MODEL_DIR = ROOT / "models/qwen3-4b-thinking-2507"
PIN = dict(  # models/qwen3-4b-thinking-2507 at revision ff.PROTOCOL["revision"]; shard sha256 from results/raw/qwen3-4b-thinking-2507.manifest.json
    repo="Qwen/Qwen3-4B-Thinking-2507",
    files={"config.json": "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
           "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
           "tokenizer_config.json": "60adcfc35f5f251e19a196ec977b4ca0fcfb534852911b3196d68d5e3b146aa0",
           "model.safetensors.index.json": "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c"},
    shards={"model-00001-of-00003.safetensors": (3957900840, "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a"),
            "model-00002-of-00003.safetensors": (3987450520, "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f"),
            "model-00003-of-00003.safetensors": (99630640, "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953")})
LIMITS = dict(max_prefix_tokens=24000, max_option_tokens=4096, max_total_tokens=24576, max_options=8, shard_size=50,
              buckets=[1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576])
# PROVISIONAL parity tolerances, fixed before any real run: max|packed - plain| / max|plain|
PARITY_REL = {"float32": 1e-4, "bfloat16": 5e-2, "float16": 2e-2}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class CaptureMismatch(Exception):
    """The output directory / inputs belong to a different protocol, request file, limits, model, device or dtype."""


class CaptureAborted(Exception):
    pass


class _Stop(Exception):
    """Raised by the deepest hook: the forward is not run past the last captured layer."""


# ------------------------------------------------------------------ model / metadata
def model_fingerprint(model_dir, deep=False, pin=PIN):
    d = Path(model_dir)
    fp = {n: file_sha(d / n) for n in pin["files"]}
    fp["shard_sizes"] = {n: (d / n).stat().st_size for n in pin["shards"]}
    if deep:
        fp["shard_sha256"] = {n: file_sha(d / n) for n in pin["shards"]}
    return fp


def check_model_dir(model_dir, deep=False, pin=PIN):
    """Pinned local model: refuse a different revision's files. deep=False checks the small files and the shard SIZES only (same-size
    changed weights pass); deep=True also verifies the sha256 of the weight contents. -> fingerprint."""
    fp = model_fingerprint(model_dir, deep, pin)
    bad = [n for n, h in pin["files"].items() if fp[n] != h] + [n for n, (sz, _) in pin["shards"].items() if fp["shard_sizes"][n] != sz]
    if deep:
        bad += [n for n, (_, h) in pin["shards"].items() if fp["shard_sha256"][n] != h]
    if bad:
        raise CaptureMismatch("model dir %s differs from the pinned revision %s: %s" % (model_dir, ff.PROTOCOL["revision"], bad))
    return fp


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        return torch.device("cuda")
    if name == "xla":
        import torch_xla  # lazy: only an XLA run needs it; never installed from here
        try:
            return torch_xla.device()
        except AttributeError:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
    raise ValueError("device must be cpu | xla | cuda")


def runtime_meta(device_name, dtype_name, attn=None):
    meta = dict(device=device_name, dtype=dtype_name, attn_implementation=attn, torch=torch.__version__, python=platform.python_version(), platform=platform.platform(),
                transformers=None, torch_xla=None)
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    if device_name == "xla":
        import torch_xla
        meta["torch_xla"] = getattr(torch_xla, "__version__", "unknown")
    if device_name == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def resolved_attn(base):
    return getattr(getattr(base, "config", None), "_attn_implementation", None)


def load_model(model_dir, device_name, dtype_name, attn="sdpa"):
    """Frozen eval-mode Qwen3 BASE model (no LM head), local files only, weights sha256-verified against the pin (deep).
    -> (tokenizer, base, fingerprint incl. shard_sha256). The resolved attention backend must be sdpa."""
    fp = check_model_dir(model_dir, deep=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    kw = dict(local_files_only=True, attn_implementation=attn)
    try:
        base = AutoModel.from_pretrained(str(model_dir), dtype=DTYPES[dtype_name], **kw)
    except TypeError:  # older transformers spells it torch_dtype
        base = AutoModel.from_pretrained(str(model_dir), torch_dtype=DTYPES[dtype_name], **kw)
    base.eval().requires_grad_(False)
    if resolved_attn(base) != "sdpa":
        raise CaptureMismatch("resolved attention backend is %r, not 'sdpa'" % (resolved_attn(base),))
    return tok, base.to(get_device(device_name)), fp


# ------------------------------------------------------------------ planning (tokenisation + bounds, no model)
def bucket_of(n_tokens, buckets):
    return next(b for b in buckets if n_tokens <= b)


def tokenize_request(tok, req, protocol=ff.PROTOCOL):
    prefix = list(tok.encode(req["context_text"] + protocol["query_cue"], add_special_tokens=False))
    return prefix, [list(tok.encode(protocol["option_prefix"] + o, add_special_tokens=False)) for o in req["options"]]


def plan_one(tok, req, limits, protocol=ff.PROTOCOL):
    """-> (item, None) or (None, exclusion). Bounds are enforced, never clipped."""
    did = req["decision_id"]
    ex = lambda reason, **detail: (None, dict(decision_id=did, kind="bound", reason=reason, detail=detail))
    if not req["options"]:
        return ex("no_options")
    if len(req["options"]) > limits["max_options"]:
        return ex("too_many_options", n=len(req["options"]), limit=limits["max_options"])
    prefix, suf = tokenize_request(tok, req, protocol)
    if not prefix or len(prefix) > limits["max_prefix_tokens"]:
        return ex("prefix_over_budget" if prefix else "empty_prefix", tokens=len(prefix), limit=limits["max_prefix_tokens"])
    for i, s in enumerate(suf):
        if not s or len(s) > limits["max_option_tokens"]:
            return ex("option_over_budget" if s else "empty_option", option=i, tokens=len(s), limit=limits["max_option_tokens"])
    total = len(prefix) + sum(map(len, suf))
    if total > limits["max_total_tokens"]:
        return ex("total_over_budget", tokens=total, limit=limits["max_total_tokens"])
    return dict(decision_id=did, prefix_ids=prefix, suffix_ids=suf, n_tokens=total, bucket=bucket_of(total, limits["buckets"]), req=req), None


# ------------------------------------------------------------------ the capturer
def pool_rows(x, W):
    """x [B,H] (any float dtype), W [R,B] fp32 weights -> [R,H] fp32. Elementwise fp32 multiply-add per row (not a matmul, so no
    reduced-precision matmul path on any device); static shapes for a given bucket."""
    xf = x.float()
    return torch.stack([(W[r, :, None] * xf).sum(0) for r in range(W.shape[0])])


class Capturer:
    """Forward hooks on base.layers[d-1] (d in depths) pool the residual stream on-device; the deepest hook stops the forward."""

    def __init__(self, base, depths, device, max_options):
        self.base, self.depths, self.device, self.R = base, list(depths), device, 1 + max_options
        assert max(self.depths) <= len(base.layers), "model has fewer layers than the deepest captured depth"
        impl = resolved_attn(base)
        if impl is not None and impl != "sdpa":  # eager adds a 4D mask to the scores: a boolean mask would be added as 0/1, not applied
            raise CaptureMismatch("attention backend %r is not supported for the boolean block mask; load the model with attn_implementation='sdpa'" % (impl,))
        self.cap, self._W, self.seen_buckets, self.forwards = {}, None, set(), 0
        for d in self.depths:
            base.layers[d - 1].register_forward_hook(self._hook(d))

    def _hook(self, d):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            self.cap[d] = pool_rows(x[0], self._W)
            if d == max(self.depths):
                raise _Stop
        return hook

    @torch.inference_mode()
    def _forward(self, ids, pos, mask, W):
        self.cap, self._W = {}, W.to(self.device)
        self.forwards += 1
        kw = {} if mask is None else dict(attention_mask=mask.to(self.device), position_ids=pos.to(self.device))
        try:
            self.base(input_ids=ids.to(self.device), use_cache=False, **kw)
        except _Stop:
            pass
        else:
            raise RuntimeError("the forward finished without reaching the deepest captured layer")
        return torch.stack([self.cap[d] for d in self.depths]).cpu()  # [K,R,H] float32

    def capture(self, item, bucket=None):
        """Production path: one packed, bucket-padded forward. -> (hq [K,H], ho [N,K,H]) float32."""
        bucket = bucket or item["bucket"]
        P, N = len(item["prefix_ids"]), len(item["suffix_ids"])
        ids, pos, mask, spans = pack(dict(prefix_ids=item["prefix_ids"], suffix_ids=item["suffix_ids"]), "cpu")
        L = ids.shape[1]
        assert L <= bucket, ("sequence longer than its bucket", L, bucket)
        m = torch.zeros(1, 1, bucket, bucket, dtype=torch.bool)
        m[:, :, :L, :L] = mask
        pad = torch.arange(L, bucket)
        m[0, 0, pad, pad] = True  # padded rows see themselves only; no real row sees them
        W = torch.zeros(self.R, bucket)
        W[0, P - 1] = 1.0
        for i, (a, b) in enumerate(spans):
            W[1 + i, a:b] = 1.0 / (b - a)
        out = self._forward(F.pad(ids, (0, bucket - L)), F.pad(pos, (0, bucket - L)), m, W)
        return out[:, 0], out[:, 1:1 + N].permute(1, 0, 2).contiguous()

    def capture_plain(self, item, i):
        """Reference path for parity checks: a separate unpadded [prefix + suffix_i] sequence, default causal attention.
        -> (hq [K,H], ho_i [K,H])."""
        P, s = len(item["prefix_ids"]), item["suffix_ids"][i]
        W = torch.zeros(self.R, P + len(s))
        W[0, P - 1] = 1.0
        W[1, P:] = 1.0 / len(s)
        out = self._forward(torch.tensor([list(item["prefix_ids"]) + list(s)]), None, None, W)
        return out[:, 0], out[:, 1]


# ------------------------------------------------------------------ run: binding, chunks, resume, coverage
def validate_limits(limits):
    """Positive integers only (a zero / negative bound or shard size would silently select or drop data)."""
    for k in ("max_prefix_tokens", "max_option_tokens", "max_total_tokens", "max_options", "shard_size"):
        if not (isinstance(limits.get(k), int) and not isinstance(limits[k], bool) and limits[k] >= 1):
            raise ValueError("limit %s must be a positive integer, got %r" % (k, limits.get(k)))
    b = limits.get("buckets")
    if not (isinstance(b, list) and b and all(isinstance(x, int) and x >= 1 for x in b) and b == sorted(set(b))):
        raise ValueError("buckets must be a strictly increasing list of positive integers")
    if limits["max_total_tokens"] > b[-1]:
        raise ValueError("max_total_tokens %d exceeds the largest bucket %d" % (limits["max_total_tokens"], b[-1]))
    return limits


def validate_chunk_range(chunk_range, n_chunks):
    """None, or (lo, hi) with 0 <= lo < hi <= n_chunks (negative indices would select unintended chunks)."""
    if chunk_range is None:
        return 0, n_chunks
    lo, hi = chunk_range
    if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi <= n_chunks):
        raise ValueError("chunk range %r must satisfy 0 <= a < b <= %d (the number of chunks)" % (tuple(chunk_range), n_chunks))
    return lo, hi


def _read_requests(requests_path):
    want = ff.expected_hashes(requests_path)  # rejects duplicate ids and stored hashes that do not match the texts
    return list(read_gz(requests_path)), want


def make_binding(protocol, requests_path, ids, limits, fingerprint, device_name, dtype_name, attn):
    return dict(protocol_sha256=ff.protocol_sha(protocol), requests_sha256=file_sha(requests_path), request_ids=ids, limits=limits, model=fingerprint,
                device=device_name, dtype=dtype_name, attn_implementation=attn)


def _create_atomic(path, text):
    """Create-only AND atomic: the content is fully written and fsynced to a temp file, then hard-linked to `path` (fails with
    FileExistsError if it exists). A crash leaves either no `path` or a complete one, never a truncated one."""
    path = Path(path)
    tmp = path.with_name("%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def bind_dir(out_dir, binding):
    """Create-only binding file; an existing one must be identical (resume only into the SAME inputs/config/device/dtype)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "capture-binding.json"
    try:
        _create_atomic(path, json.dumps(binding))
        return
    except FileExistsError:
        pass
    try:
        old = json.load(open(path, encoding="utf-8"))
    except ValueError as e:
        raise CaptureMismatch("existing %s is unreadable (%s); move it aside only if no run is using this directory" % (path, e))
    diff = sorted(k for k in set(old) | set(binding) if old.get(k) != binding.get(k))
    if diff:
        raise CaptureMismatch("output dir %s is bound to different inputs; changed: %s" % (out_dir, ", ".join(diff)))


def _names(idx):
    return "features-%04d.pt" % idx, "features-%04d.done.json" % idx, "capture-excluded-%04d.json" % idx


def _quarantine(out_dir, idx, why):
    q = Path(out_dir) / "interrupted"
    q.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S") + "-%d" % time.time_ns()
    moved = []
    for n in _names(idx):
        p = Path(out_dir) / n
        if p.exists():
            shutil.move(str(p), str(q / (n + "." + stamp)))
            moved.append(n)
    return dict(chunk=idx, reason=why, moved=moved)


def _chunk_state(out_dir, idx, chunk_ids, want, requests_path, protocol, binding_sha):
    """-> ("done", marker) | ("absent", None) | ("bad", reason). A done chunk is re-verified record by record."""
    pt, done, side = (Path(out_dir) / n for n in _names(idx))
    if not done.exists():
        return ("absent", None) if not (pt.exists() or side.exists()) else ("bad", "files without completion marker")
    try:
        mk = json.load(open(done, encoding="utf-8"))
        if mk.get("binding_sha256") != binding_sha or mk.get("chunk_ids") != chunk_ids:
            return "bad", "marker is bound to another binding / chunk"
        cap, exc = mk["decision_ids"], mk["excluded_ids"]
        if sorted(cap + exc) != sorted(chunk_ids) or set(cap) & set(exc):
            return "bad", "captured + excluded != chunk ids"
        sc = json.load(open(side, encoding="utf-8"))
        if [e["decision_id"] for e in sc["excluded"]] != exc:
            return "bad", "exclusion sidecar disagrees with the marker"
        rep, _ = ff.check_features(requests_path, out_dir, protocol, ids=cap, shards=[pt], want=want)
        if not rep["ok"]:
            return "bad", "records failed verification: " + json.dumps({k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in rep.items() if v})[:300]
        return "done", mk
    except Exception as e:  # noqa: BLE001 - any unreadable artefact is not trusted
        return "bad", "unreadable chunk: %r" % (e,)


def timing_summary(entries):
    """Measured seconds per captured decision. The first call per bucket in a process (compile / warm-up on XLA) is reported
    apart from the steady calls."""
    first = [e for e in entries if e["first_of_bucket"]]
    steady = [e for e in entries if not e["first_of_bucket"]]
    s = [e["seconds"] for e in steady]
    by_bucket = {}
    for e in steady:
        by_bucket.setdefault(str(e["bucket"]), []).append(e["seconds"])
    return dict(n=len(entries), first_call_seconds={str(e["bucket"]): round(e["seconds"], 4) for e in first},
                n_steady=len(s), steady_mean_s=round(statistics.fmean(s), 4) if s else None, steady_median_s=round(statistics.median(s), 4) if s else None,
                steady_max_s=round(max(s), 4) if s else None, steady_mean_s_by_bucket={b: round(statistics.fmean(v), 4) for b, v in sorted(by_bucket.items(), key=lambda kv: int(kv[0]))},
                total_s=round(sum(e["seconds"] for e in entries), 3), tokenize_total_s=round(sum(e["tokenize_seconds"] for e in entries), 3))


def run_capture(requests_path, out_dir, tok, capturer, *, protocol=ff.PROTOCOL, limits=LIMITS, fingerprint=None, device_name="cpu", dtype_name="float32",
                attn=None, meta=None, startup_seconds=None, ids=None, limit=None, chunk_range=None, clock=time.perf_counter, max_fail_streak=5, synthetic=False):
    """Capture `requests_path` into `out_dir` chunk by chunk (resumable). -> run report dict (also written create-only)."""
    validate_limits(limits)
    if limit is not None and not (isinstance(limit, int) and limit >= 1):
        raise ValueError("limit must be a positive integer")
    if not synthetic:  # a real run needs the verified weight contents in the binding, not just file sizes
        sh = (fingerprint or {}).get("shard_sha256")
        if not sh or any(sh.get(n) != h for n, (_, h) in PIN["shards"].items()):
            raise CaptureMismatch("a real run needs the sha256 of every weight shard verified against the pin (deep fingerprint)")
    reqs, want = _read_requests(requests_path)
    by_id = {r["decision_id"]: r for r in reqs}
    sel = [r["decision_id"] for r in reqs] if ids is None else list(ids)
    assert len(set(sel)) == len(sel) and all(i in by_id for i in sel), "selected ids must be unique request ids"
    sel = sel[:limit] if limit else sel
    S = limits["shard_size"]
    chunks = [sel[i:i + S] for i in range(0, len(sel), S)]
    lo, hi = validate_chunk_range(chunk_range, len(chunks))  # before anything is written
    binding = make_binding(protocol, requests_path, sel, limits, fingerprint, device_name, dtype_name, attn)
    bind_dir(out_dir, binding)
    bsha = sha(jdump(binding))
    rep = dict(out_dir=str(out_dir), synthetic=synthetic, meta=meta, startup_seconds=startup_seconds, n_selected=len(sel), n_chunks=len(chunks), chunk_range=[lo, hi],
               chunks_skipped_verified=[], chunks_recaptured=[], chunks_written=[], captured=0, excluded=0, excluded_by_reason={})
    t_run = clock()
    capturer.seen_buckets = set()
    fail_streak = 0
    for ci in range(lo, min(hi, len(chunks))):
        chunk = chunks[ci]
        state, info = _chunk_state(out_dir, ci, chunk, want, requests_path, protocol, bsha)
        if state == "done":
            rep["chunks_skipped_verified"].append(ci)
            continue
        if state == "bad":
            rep["chunks_recaptured"].append(_quarantine(out_dir, ci, info))
        records, excluded, timing = [], [], []
        for did in chunk:
            t0 = clock()
            item, ex = plan_one(tok, by_id[did], limits, protocol)
            tok_s = clock() - t0
            if ex:
                excluded.append(ex)
                continue
            first = item["bucket"] not in capturer.seen_buckets
            t1 = clock()
            try:
                hq, ho = capturer.capture(item)
                capturer.seen_buckets.add(item["bucket"])
                if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()):
                    raise ArithmeticError("nonfinite")
            except ArithmeticError:
                excluded.append(dict(decision_id=did, kind="numeric", reason="nonfinite_hidden", detail={}))
                continue
            except Exception as e:  # noqa: BLE001 - a per-decision runtime failure is recorded, not hidden; BaseException (Ctrl-C, crash) propagates
                fail_streak += 1
                if fail_streak >= max_fail_streak:
                    raise CaptureAborted("%d consecutive capture failures, last: %r" % (fail_streak, e)) from e
                excluded.append(dict(decision_id=did, kind="failure", reason="capture_failed", detail=dict(error=type(e).__name__, message=str(e)[:300], bucket=item["bucket"])))
                continue
            fail_streak = 0
            hq16, ho16 = hq.half(), ho.half()
            if not (torch.isfinite(hq16).all() and torch.isfinite(ho16).all()):
                excluded.append(dict(decision_id=did, kind="numeric", reason="fp16_overflow", detail=dict(max_abs=float(max(hq.abs().max(), ho.abs().max())))))
                continue
            h = want[did]
            records.append(dict(decision_id=did, context_sha256=h["context_sha256"], option_sha256=h["option_sha256"], order_sha256=h["order_sha256"], hq=hq16, ho=ho16))
            timing.append(dict(decision_id=did, n_tokens=item["n_tokens"], bucket=item["bucket"], n_options=len(item["suffix_ids"]), seconds=clock() - t1, tokenize_seconds=tok_s,
                               first_of_bucket=first, max_abs=float(max(hq.abs().max(), ho.abs().max()))))
        extra = dict(chunk_index=ci, chunk_ids=chunk, excluded_ids=[e["decision_id"] for e in excluded], binding_sha256=bsha, requests_sha256=binding["requests_sha256"],
                     limits=limits, meta=meta, synthetic=synthetic, timing=timing)
        ff.write_feature_shard(out_dir, ci, records, protocol, marker_extra=extra,
                               sidecars={_names(ci)[2]: json.dumps(dict(chunk=ci, chunk_ids=chunk, excluded=excluded))})
        rep["chunks_written"].append(ci)
        rep["captured"] += len(records)
        rep["excluded"] += len(excluded)
        for e in excluded:
            rep["excluded_by_reason"][e["reason"]] = rep["excluded_by_reason"].get(e["reason"], 0) + 1
    rep["run_seconds"] = clock() - t_run
    with open(Path(out_dir) / ("capture-run-%s-%d.json" % (time.strftime("%Y%m%dT%H%M%S"), time.time_ns())), "x", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def capture_coverage(requests_path, out_dir, protocol=ff.PROTOCOL):
    """Every selected id is captured XOR excluded, chunk by chunk, with records re-verified against the request texts.
    ok=False while chunks are pending. Excluded ids must be removed from the pool before training (assemble fails on them by design)."""
    out_dir = Path(out_dir)
    binding = json.load(open(out_dir / "capture-binding.json", encoding="utf-8"))
    _, want = _read_requests(requests_path)
    if file_sha(requests_path) != binding["requests_sha256"]:
        raise CaptureMismatch("the request file differs from the one this capture was bound to")
    bsha, sel, S = sha(jdump(binding)), binding["request_ids"], binding["limits"]["shard_size"]
    chunks = [sel[i:i + S] for i in range(0, len(sel), S)]
    captured, excluded, pending, bad, timing, metas, synth = [], [], [], [], [], [], set()
    for ci, chunk in enumerate(chunks):
        st, info = _chunk_state(out_dir, ci, chunk, want, requests_path, protocol, bsha)
        if st == "done":
            captured += info["decision_ids"]
            excluded += json.load(open(out_dir / _names(ci)[2], encoding="utf-8"))["excluded"]
            timing += info["timing"]
            metas.append(info["meta"])
            synth.add(bool(info.get("synthetic")))
        elif st == "absent":
            pending.append(ci)
        else:
            bad.append(dict(chunk=ci, reason=info))
    cap_set, exc_ids = set(captured), [e["decision_id"] for e in excluded]
    exact = len(captured) == len(cap_set) and len(exc_ids) == len(set(exc_ids)) and not (cap_set & set(exc_ids)) and \
        (pending or bad or cap_set | set(exc_ids) == set(sel))
    by_reason = {}
    for e in excluded:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    return dict(selected=len(sel), captured=len(captured), excluded=len(excluded), excluded_by_reason=by_reason, excluded_ids=exc_ids, excluded_records=excluded, captured_ids=captured,
                pending_chunks=pending, bad_chunks=bad, ok=bool(exact and not pending and not bad), exact_partition=bool(exact),
                devices=sorted({json.dumps(m, sort_keys=True) for m in metas if m}), timing=timing_summary(timing) if timing else None,
                synthetic_flags=sorted(synth))


# ------------------------------------------------------------------ preflight / parity (CLI)
def preflight(requests_path, tok, limits=LIMITS, model_dir=None, protocol=ff.PROTOCOL):
    """Tokenizer-only and lightweight: the model check covers the small pinned files and the weight shard SIZES, NOT weight contents."""
    validate_limits(limits)
    reqs, _ = _read_requests(requests_path)
    t0 = time.perf_counter()
    ex, totals, buckets, by_reason = [], [], {}, {}
    for r in reqs:
        item, e = plan_one(tok, r, limits, protocol)
        if e:
            ex.append(e["decision_id"])
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        else:
            totals.append(item["n_tokens"])
            buckets[item["bucket"]] = buckets.get(item["bucket"], 0) + 1
    q = lambda p: sorted(totals)[min(len(totals) - 1, int(p * len(totals)))] if totals else None
    return dict(requests=len(reqs), capturable=len(totals), excluded=len(ex), excluded_by_reason=by_reason, limits=limits, tokens_total_min_median_p90_p99_max=[q(0), q(.5), q(.9), q(.99), q(1.0)],
                tokens_sum=sum(totals), bucket_histogram=dict(sorted(buckets.items())), distinct_padded_shapes=len(buckets), chunks=math.ceil(len(reqs) / limits["shard_size"]),
                model_pin_light_ok=bool(model_dir and check_model_dir(model_dir)), weights_content_verified=False, tokenize_seconds=round(time.perf_counter() - t0, 2),
                notes=["LIGHT model check only: small pinned files + weight shard sizes; same-size changed weights would pass here. `run` and `parity` verify the shard sha256 (deep).",
                       "distinct_padded_shapes = XLA programs to compile (one per bucket used); steady time per decision is NOT estimated here",
                       "float16 storage can overflow on Qwen outlier dimensions: such decisions are excluded as fp16_overflow, see run reports"])


def parity(capturer, tok, requests_path, n, dtype_name, limits=LIMITS, protocol=ff.PROTOCOL):
    """Packed + bucket-padded production path vs separate plain forwards (the reference), vs the next bucket (padding invariance),
    and the query across the independent references. Every comparison uses the declared dtype tolerance (relative to the max magnitude)."""
    validate_limits(limits)
    reqs, _ = _read_requests(requests_path)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
    res = []
    for r in reqs:
        item, e = plan_one(tok, r, limits, protocol)
        if e or item["n_tokens"] > 4096:
            continue  # keep the check short; long sequences are covered by the timing batch
        hq, ho = capturer.capture(item)
        nxt = next((b for b in limits["buckets"] if b > item["bucket"]), None)
        pad_hq, pad_ho = capturer.capture(item, nxt) if nxt else (hq, ho)
        ref = [capturer.capture_plain(item, i) for i in range(len(item["suffix_ids"]))]
        rh = torch.stack([x[1] for x in ref])
        res.append(dict(decision_id=item["decision_id"], n_tokens=item["n_tokens"], bucket=item["bucket"], query_rel=rel(hq, ref[0][0]),
                        query_across_references_rel=max(rel(x[0], ref[0][0]) for x in ref), options_rel=rel(ho, rh), padding_rel=max(rel(pad_hq, hq), rel(pad_ho, ho))))
        if len(res) >= n:
            break
    tol = PARITY_REL[dtype_name]
    keys = ("query_rel", "query_across_references_rel", "options_rel", "padding_rel")
    return dict(n=len(res), requested=n, tolerance_rel=tol, results=res, passed=len(res) == n and all(max(x[k] for k in keys) <= tol for x in res),
                note="tolerance fixed in code before any real run; passed requires n results, each comparison within it; a mechanics parity check of packing/padding, not an accuracy result")


def _posint(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer, got %s" % s)
    return v


def _chunk_range(s):
    try:
        lo, hi = (int(x) for x in s.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("chunk range must look like a:b")
    if not 0 <= lo < hi:
        raise argparse.ArgumentTypeError("chunk range needs 0 <= a < b, got %s" % s)
    return lo, hi


def _cli(argv=None):
    """-> process exit code (0 success; 1 failed check / incomplete; 2 bad arguments)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run", "parity", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--requests", required=True)
        if name in ("preflight", "run", "parity"):
            p.add_argument("--model-dir", default=str(MODEL_DIR))
            for k in ("max_prefix_tokens", "max_option_tokens", "max_total_tokens", "max_options", "shard_size"):
                p.add_argument("--" + k.replace("_", "-"), type=_posint)
        if name in ("run", "parity"):
            p.add_argument("--device", required=True, choices=["cpu", "xla", "cuda"])
            p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
            p.add_argument("--attn", default="sdpa", choices=["sdpa"], help="attention backend (the packed block mask is boolean; only sdpa applies it correctly)")
        if name in ("run", "verify"):
            p.add_argument("--out", required=True)
        if name == "preflight":
            p.add_argument("--out")
        if name == "run":
            p.add_argument("--limit", type=_posint)
            p.add_argument("--chunk-range", type=_chunk_range, help="a:b chunk indices, 0 <= a < b <= number of chunks; one process per range")
        if name == "parity":
            p.add_argument("--n", type=_posint, default=3)
    a = ap.parse_args(argv)
    limits = dict(LIMITS, **{k: getattr(a, k) for k in ("max_prefix_tokens", "max_option_tokens", "max_total_tokens", "max_options", "shard_size") if getattr(a, k, None)})
    try:
        validate_limits(limits)
    except ValueError as e:
        ap.error(str(e))
    if a.cmd == "verify":
        rep = capture_coverage(a.requests, a.out)
        print(json.dumps({k: v for k, v in rep.items() if k != "captured_ids"}, indent=1))
        return 0 if rep["ok"] else 1
    if a.cmd == "preflight":
        from transformers import AutoTokenizer
        rep = preflight(a.requests, AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True), limits, a.model_dir)
        if a.out:
            with open(a.out, "x", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "run":  # validate the range against the number of chunks BEFORE the 8 GB hash + model load
        n_req = len(ff.expected_hashes(a.requests))
        n_sel = min(n_req, a.limit) if a.limit else n_req
        try:
            validate_chunk_range(a.chunk_range, math.ceil(n_sel / limits["shard_size"]))
        except ValueError as e:
            ap.error(str(e))
    t0 = time.perf_counter()
    tok, base, fp = load_model(a.model_dir, a.device, a.dtype, a.attn)
    attn = resolved_attn(base)
    capturer = Capturer(base, ff.PROTOCOL["depths"], get_device(a.device), limits["max_options"])
    startup = time.perf_counter() - t0  # imports + weight sha256 + model load + device init (compile time shows up as the first call per bucket)
    meta = runtime_meta(a.device, a.dtype, attn)
    if a.cmd == "parity":
        rep = parity(capturer, tok, a.requests, a.n, a.dtype, limits)
        print(json.dumps(dict(meta=meta, startup_seconds=startup, **rep), indent=1))
        return 0 if rep["passed"] else 1
    rep = run_capture(a.requests, a.out, tok, capturer, limits=limits, fingerprint=fp, device_name=a.device, dtype_name=a.dtype, attn=attn, meta=meta,
                      startup_seconds=startup, limit=a.limit, chunk_range=a.chunk_range)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
