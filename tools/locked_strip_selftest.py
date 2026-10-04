"""Self-test of a label-stripped LOCKED bundle (amendment e10: "proven by the bundle self-test on the stripped file first"). Tokenizer only: no model, no GPU, no accuracy.
    cd BUNDLE && PYTHONPATH=. <repo>/.venv/Scripts/python.exe <repo>/tools/locked_strip_selftest.py --full-root <repo>
Checks, for EVERY LOCKED decision in the manifest, using the bundle's own code (capture/, compiler/, state/ imported from the bundle because PYTHONPATH=. comes first):
  1. the stripped decision row has exactly the keys {decision_id, state_hash}; the stripped trajectory has exactly {trajectory_id, trajectory}; no label-like key exists anywhere inside the events;
  2. the runner's own data path works on the stripped files alone: state rebuilt from the events reproduces state_hash; render_pool(stripped row, events, ...) succeeds (a missing key would raise KeyError);
  3. render_pool on the stripped row and on the FULL row return identical dicts (texts, prefix and suffix ids, pool_hash, prompt_hash, state_hash, token counts), so no stripped field influences the capture input;
  4. the events in the bundle equal the events in the full file; every manifest decision is present; rows_sha256 and texts_sha256 match MANIFEST.json; the texts file equals a recomputation from the stripped events.
It reads the full LOCKED files once (logged to results/raw/locked-access-log.jsonl of the repo) and writes a create-only result results/raw/locked-strip-selftest-<time>.json. Exit code 1 on any difference."""
import argparse, datetime, hashlib, json, os, sys
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
from pathlib import Path
from transformers import AutoTokenizer
from capture.pool import NONE_TEXT, render_pool
from capture.render import RenderOverflow
from compiler import a5v2, state_builder
from compiler.candidates import distractor_pool

ap = argparse.ArgumentParser()
ap.add_argument("--full-root", required=True, help="the repository root holding the FULL locked files and results/raw")
ap.add_argument("--model", default=r"D:\local model\models\qwen3-4b-thinking-2507")
args = ap.parse_args()
B, F = Path.cwd(), Path(args.full_root)
assert (B / "MANIFEST.json").exists() and B.resolve() != F.resolve(), "run this from the BUNDLE directory"
meta = json.load(open(B / "MANIFEST.json"))
assert meta["contains_labels"] is False and meta["role"] == "locked"
ids = json.load(open(B / "data/v3/manifest-locked.json", encoding="utf-8"))["decision_ids"]
rev = json.load(open(B / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
with open(F / "results/raw/locked-access-log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "purpose": "locked_strip_selftest: compare the label-stripped bundle with the full LOCKED files by hashes of rendered capture inputs; no model outputs, no accuracies"}) + "\n")
LABELISH = {"resolved", "model_patch", "gen_tests_correct", "pred_passes_gen_tests", "label_index", "label_side", "exit_status", "next_action_kind", "tier", "tier_reason", "provenance", "instance_id", "repo", "true_target_unseen_in_state"}


def keys_deep(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k); keys_deep(v, acc)
    elif isinstance(o, list):
        for v in o:
            keys_deep(v, acc)
    return acc


srows = {}
for l in open(B / "data/decisions/locked-v3.jsonl", encoding="utf-8"):
    r = json.loads(l); srows[r["decision_id"]] = r
strajs = {}
for l in open(B / "data/locked_v3/fresh-locked.jsonl", encoding="utf-8"):
    r = json.loads(l); strajs[r["trajectory_id"]] = r
frows = {}
want = set(ids)
for l in open(F / "data/decisions/locked-v3.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["decision_id"] in want:
        frows[r["decision_id"]] = r
fraw = a5v2.load_raw(F / "data/locked_v3/fresh-locked.jsonl")
res = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), bundle=str(B), decisions=len(ids), failures=[])
fail = res["failures"]
if set(srows) != want or len(srows) != len(ids):
    fail.append("stripped rows differ from the manifest ids")
if {tuple(sorted(r)) for r in srows.values()} != {("decision_id", "state_hash")}:
    fail.append("a stripped row has keys other than decision_id and state_hash")
if {tuple(sorted(r)) for r in strajs.values()} != {("trajectory", "trajectory_id")}:
    fail.append("a stripped trajectory has keys other than trajectory_id and trajectory")
deep = set()
for t in strajs.values():
    keys_deep(t["trajectory"], deep)
res["label_like_keys_inside_events"] = sorted(deep & LABELISH)
if deep & LABELISH:
    fail.append(f"label-like keys inside the events: {sorted(deep & LABELISH)}")
tok = AutoTokenizer.from_pretrained(args.model)
n_ok = n_skip = 0
texts = set()
for n, did in enumerate(ids):
    tid, t = did.rsplit(":", 1)
    ev_s, ev_f = strajs[tid]["trajectory"], fraw[tid]["trajectory"]
    if ev_s != ev_f:
        fail.append(f"{did}: events differ between bundle and full file"); continue
    st = state_builder.build_state(ev_s, int(t), 32000)
    if st["state_hash"] != srows[did]["state_hash"] or st["state_hash"] != frows[did]["state_hash"]:
        fail.append(f"{did}: state rebuild differs from state_hash"); continue
    texts |= {p[0] for p in distractor_pool(ev_s, int(t))}
    try:
        Rs = render_pool(srows[did], ev_s, st, tok, rev)
    except RenderOverflow:
        try:
            render_pool(frows[did], ev_f, st, tok, rev); fail.append(f"{did}: overflow only on the stripped row")
        except RenderOverflow:
            n_skip += 1
        continue
    except KeyError as e:
        fail.append(f"{did}: render_pool needs the stripped key {e}"); continue
    Rf = render_pool(frows[did], ev_f, st, tok, rev)
    if Rs != Rf:
        fail.append(f"{did}: rendered capture input differs between stripped and full row")
    else:
        n_ok += 1
    if n % 400 == 0:
        print(n, "/", len(ids), flush=True)
res.update(rendered_identical=n_ok, render_overflow_skipped_identically=n_skip)
tv = json.load(open(F / "kaggle_bundle/textemb/data/texts_trainval.json", encoding="utf-8"))
recomputed = sorted(texts - set(tv) - {NONE_TEXT})
btexts = json.load(open(B / "data/texts_locked.json", encoding="utf-8"))
if recomputed != btexts or hashlib.sha256(json.dumps(btexts).encode()).hexdigest() != meta["texts_sha256"]:
    fail.append("data/texts_locked.json differs from a recomputation or from MANIFEST.json")
if hashlib.sha256((B / "data/decisions/locked-v3.jsonl").read_bytes()).hexdigest() != meta["rows_sha256"]:
    fail.append("rows_sha256 differs from MANIFEST.json")
bad = [f["path"] for f in meta["files"] if hashlib.sha256((B / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
if bad:
    fail.append(f"files differ from MANIFEST.json: {bad[:5]}")
res.update(locked_only_texts=len(btexts), rows_sha256=meta["rows_sha256"], texts_sha256=meta["texts_sha256"], verdict="PASS" if not fail else "FAIL")
p = F / f"results/raw/locked-strip-selftest-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
assert not p.exists()
json.dump(res, open(p, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1), "failures:", fail[:10])
print("wrote", p)
sys.exit(0 if not fail else 1)
