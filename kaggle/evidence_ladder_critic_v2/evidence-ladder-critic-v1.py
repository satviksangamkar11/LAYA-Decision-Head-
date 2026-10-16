"""Evidence-Ladder critic kernel, BATCH version (Step 4; new version of kernel evidence-ladder-critic-v1 so its attached Kaggle Secrets carry over).
Internet on, NVIDIA keys only as Kaggle Secrets NVIDIA_API_KEYS_1..n, never runs candidate code. The ONLY dataset it mounts is the critic packet dataset
(evidence-ladder-critic-packets-v1: blinded packets + a hash-checked copy of the exact code). Reads limits only from configs/critic_concurrency_v1.toml.
Output /kaggle/working/critic_out/: responses.jsonl, MANIFEST.json (sha256), rejected.json, progress.jsonl (fsynced, at-least-once), broker_ledger.jsonl.
No key value is ever printed or written (labels key_1..key_n only)."""
import hashlib, json, sys, time
from pathlib import Path

OUT = Path("/kaggle/working")
RES = {"kernel": "evidence_ladder_critic_batch", "ok": False}


def finish(code=0):
    (OUT / "RUN.json").write_text(json.dumps(RES, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(RES, sort_keys=True))
    sys.exit(code)


t0 = time.time()
cands = sorted(Path("/kaggle/input").glob("**/PACKETS_MANIFEST.json"))
if len(cands) != 1:
    RES["error"] = "expected exactly one packet dataset, found %d" % len(cands)
    finish(2)
src = cands[0].parent
try:
    cm = json.loads((src / "CODE_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [f for f, h in cm["files"].items() if not (src / f).is_file() or hashlib.sha256((src / f).read_bytes()).hexdigest() != h]
except Exception as e:
    RES["error"] = "code manifest unreadable: " + type(e).__name__
    finish(2)
RES["code_files"] = len(cm["files"])
if bad:
    RES["error"] = "code hash mismatch: %r" % bad[:3]
    finish(2)
sys.path.insert(0, str(src / "code"))
from evidence_ladder.critic_batch import batch, config, transport     # noqa: E402  (code came from the hash-checked dataset)
from evidence_ladder.models.kaggle_keys import load_kaggle_keys       # noqa: E402

try:
    cc = config.load_concurrency(src / "code" / "configs" / "critic_concurrency_v1.toml")   # refuses on any disagreement with broker.LIMITS
    keys = load_kaggle_keys()
except Exception as e:
    RES["error"] = "setup failed: " + type(e).__name__                  # message dropped: it could carry secret text
    finish(2)
RES.update(n_keys=len(keys), labels=list(keys), concurrency_sha256=cc["sha256"], limits=cc["limits"], ramp=cc["ramp"])
runner = batch.BatchRunner(src, OUT / "critic_out", transport.make_transport(keys), list(keys), cc)
try:
    res = runner.run()
except Exception as e:
    RES["error"] = "run failed: " + type(e).__name__
    finish(1)
RES.update(ok=True, summary=res["summary"], recovered=res["recovered"], states=res["manifest"]["states"], rejected=len(res["manifest"]["rejected"]),
           responses_sha256=res["manifest"]["responses_sha256"], seconds=round(time.time() - t0, 1))
finish(0)
