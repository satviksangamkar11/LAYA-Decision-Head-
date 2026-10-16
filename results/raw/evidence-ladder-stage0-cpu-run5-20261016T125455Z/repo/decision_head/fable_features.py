"""Cached frozen-Qwen feature contract for the Fable head (fable_features_v1).

A capture is valid for a request only if the protocol AND the hashes of its context, its options and their presented
order are identical. Old vectors from other pools, context-free option encodings and trainable-adapter outputs
(hidden width != the backbone's) are rejected, never substituted. This module does not run Qwen; the capture job
that writes these shards is pending.

Shard layout: features-NNNN.pt  = {"protocol", "protocol_sha256", "records": [rec]}
              features-NNNN.done.json = completion marker {file_sha256, decision_ids, protocol_sha256}
rec = {decision_id, context_sha256, option_sha256 [N], order_sha256, hq [K,H] fp16, ho [N,K,H] fp16}
hq = hidden state at the last prefix position (float16 tensors, protocol.feature_dtype; other dtypes are rejected) (after the query cue), ho[i] = mean over option i's own tokens when it is
encoded as an independent suffix of the shared prefix; K = len(depths) frozen Qwen layers.
"""
import json
import os
from pathlib import Path

import torch

from decision_head.fable_pool import file_sha, jdump, read_gz, request_hashes, sha

PROTOCOL = dict(
    id="fable_capture_v1", model="Qwen3-4B-Thinking-2507", revision="768f209d9ea81521153ed38c47d515654e938aea",
    depths=[18, 24, 30], hidden=2560, feature_dtype="float16", query_cue="\n\nNext action:", option_prefix=" ",
    suffix_mode="shared_prefix_independent_suffix", query_position="last_prefix_token", option_pool="mean_over_option_tokens",
    renderer="fable_render_v1", context_policy="fable_ctx_v1", dtype_note="record device/dtype/torch/transformers in the capture sidecar; one device class for every split")
PROTOCOL_SHA256 = sha(jdump(PROTOCOL))


class FeatureMismatch(Exception):
    pass


def protocol_sha(protocol):
    return sha(jdump(protocol))


def expected_hashes(requests_path):
    """decision_id -> hashes RECOMPUTED from the request texts (stored hashes are checked against them)."""
    out = {}
    for r in read_gz(requests_path):
        if r["decision_id"] in out:
            raise FeatureMismatch("duplicate request id %s" % r["decision_id"])
        h = request_hashes(r["context_text"], r["options"])
        if h != {k: r[k] for k in h}:
            raise FeatureMismatch("request %s: stored hashes do not match its texts" % r["decision_id"])
        out[r["decision_id"]] = dict(h, n=len(r["options"]))
    return out


def _atomic_write(path, writer, mode="wb"):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        writer(f)
    os.replace(tmp, path)


def write_feature_shard(directory, index, records, protocol=PROTOCOL, marker_extra=None, sidecars=None):
    """Create-only and atomic: sidecars (name -> text), then the shard, then the completion marker, each written to a temp
    file and renamed. The marker is last, so a crash leaves at most unmarked files (never trusted, see check_features).
    `marker_extra` adds fields to the marker (never the reserved ones); sidecar sha256s are recorded in the marker."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    pt, done = d / ("features-%04d.pt" % index), d / ("features-%04d.done.json" % index)
    if pt.exists() or done.exists():
        raise FileExistsError(pt)
    reserved = {"file_sha256", "decision_ids", "protocol_sha256", "sidecars"}
    assert not reserved & set(marker_extra or {}), "reserved marker field"
    side = {}
    for name, text in (sidecars or {}).items():
        _atomic_write(d / name, lambda f, t=text: f.write(t), "w")
        side[name] = file_sha(d / name)
    _atomic_write(pt, lambda f: torch.save(dict(protocol=protocol, protocol_sha256=protocol_sha(protocol), records=records), f))
    marker = dict(marker_extra or {}, file_sha256=file_sha(pt), decision_ids=[r["decision_id"] for r in records], protocol_sha256=protocol_sha(protocol), sidecars=side)
    _atomic_write(done, lambda f: json.dump(marker, f), "w")


def _shards(directory):
    return sorted(Path(directory).glob("features-*.pt"))


def check_features(requests_path, directory, protocol=PROTOCOL, ids=None, shards=None, want=None):
    """-> report dict. ok only when every requested id (default: all) has exactly one matching, finite, complete record.
    `shards` limits the scan to specific shard files (per-shard verification on resume)."""
    want = expected_hashes(requests_path) if want is None else want  # `want` = precomputed expected_hashes (avoids rehashing per shard)
    if ids is not None:
        want = {i: want[i] for i in ids}
    psha, K, H = protocol_sha(protocol), len(protocol["depths"]), protocol["hidden"]
    rep = dict(requested=len(want), matched=0, missing=[], changed={}, bad_dtype=[], bad_shape=[], nonfinite=[], duplicate=[], protocol_mismatch=[],
               unmarked_shards=[], extra=0)
    got, recs = set(), {}
    for pt in (_shards(directory) if shards is None else [Path(x) for x in shards]):
        done = pt.with_suffix("").with_suffix(".done.json")
        marker = json.load(open(done, encoding="utf-8")) if done.exists() else {}
        if marker.get("file_sha256") != file_sha(pt) or any(not (Path(directory) / n).exists() or file_sha(Path(directory) / n) != h for n, h in marker.get("sidecars", {}).items()):
            rep["unmarked_shards"].append(pt.name)  # incomplete or altered after completion: not trusted
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        if marker.get("decision_ids") != [r["decision_id"] for r in blob["records"]] or marker.get("protocol_sha256") != blob.get("protocol_sha256"):
            rep["unmarked_shards"].append(pt.name)  # marker does not describe this shard
            continue
        if blob.get("protocol_sha256") != psha or protocol_sha(blob.get("protocol", {})) != psha:
            rep["protocol_mismatch"].append(pt.name)
            continue
        for r in blob["records"]:
            did = r["decision_id"]
            if did in got:
                rep["duplicate"].append(did)
                continue
            got.add(did)
            if did not in want:
                rep["extra"] += 1
                continue
            w = want[did]
            diff = [k for k in ("context_sha256", "option_sha256", "order_sha256") if r.get(k) != w[k]]
            if diff:
                rep["changed"][did] = diff
            elif not all(isinstance(r.get(k), torch.Tensor) and str(r[k].dtype) == "torch." + protocol["feature_dtype"] for k in ("hq", "ho")):
                rep["bad_dtype"].append(did)  # captures must be real tensors of exactly the protocol's floating dtype
            elif tuple(r["hq"].shape) != (K, H) or tuple(r["ho"].shape) != (w["n"], K, H):
                rep["bad_shape"].append(did)
            elif not (torch.isfinite(r["hq"]).all() and torch.isfinite(r["ho"]).all()):
                rep["nonfinite"].append(did)
            else:
                rep["matched"] += 1
                recs[did] = r
    rep["missing"] = sorted(set(want) - got)
    rep["ok"] = rep["matched"] == len(want) and not any(rep[k] for k in ("missing", "changed", "bad_dtype", "bad_shape", "nonfinite", "duplicate", "protocol_mismatch", "unmarked_shards"))
    return rep, recs


def load_features(requests_path, directory, protocol=PROTOCOL, ids=None):
    """Strict: raises FeatureMismatch unless every requested decision has a matching capture."""
    rep, recs = check_features(requests_path, directory, protocol, ids)
    if not rep["ok"]:
        raise FeatureMismatch({k: (v if not isinstance(v, (list, dict)) else (list(v)[:5], len(v))) for k, v in rep.items() if v})
    return recs
