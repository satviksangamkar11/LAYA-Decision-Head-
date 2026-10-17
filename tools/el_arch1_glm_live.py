"""[EL-ARCH-1 D1] LIVE GLM STEP for the acceptance run. THE COORDINATOR RUNS THIS IN THE USER'S TERMINAL; it is never run by an agent.

  & 'D:\\local model\\.venv\\Scripts\\python.exe' 'D:\\local model\\tools\\el_arch1_glm_live.py' '<acceptance out dir>'

What it does: reads the blinded critic packets the acceptance driver wrote (<out>/glm/critic_packets, already validated by critic_batch/validator.py), sends each packet ONCE (plus the
broker's bounded retries) to z-ai/glm-5.3-flash with reasoning_effort max, STREAMED, through the project key system (tools/laya_int1_stepE.load_keys: .secrets/nvidia_api_key_<n>.txt,
n >= 2, read into memory, labels key_1..key_k; values are never printed, logged or written) and the critic_batch BatchRunner (<= 39 sends/key/minute, ramp, cooldown on 429, 402/quota stops
everything). Answers are fsynced to <out>/glm/critic_batch/{progress.jsonl,broker_ledger.jsonl,responses.jsonl}; then re-run the acceptance driver, which ingests them through the controller's
critic stage (experiments run in the WSL sandbox; model evidence stays diagnostic). Exit 0 = every packet reached a terminal state; 1 = a leak was found or a packet is unresolved.
No key file readable -> explicit error and ZERO calls. Create-only summary: <out>/glm/GLM_LIVE_SUMMARY_<UTC>.json."""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{16,}|\bsk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}")


def main(out: Path) -> int:
    import laya_int1_stepE as SE                       # only for load_keys(): importing it reads nothing and calls nothing
    from evidence_ladder.critic_batch import batch as BT
    from evidence_ladder.critic_batch.config import load_concurrency
    from evidence_ladder.critic_batch.transport import make_transport
    from evidence_ladder.critic_batch import accounting as ACC
    from evidence_ladder.models import experiments as EXP, glm
    packets = out / "glm" / "critic_packets"
    if not (packets / "PACKETS_MANIFEST.json").is_file():
        print("ERROR: no packets under %s (run the acceptance driver first); zero calls" % packets); return 1
    keys = SE.load_keys()
    if not keys:
        print("ERROR: no readable .secrets/nvidia_api_key_<n>.txt (n >= 2); zero calls made"); return 1
    cfg = glm.GlmConfig(reasoning_params=dict(BT.REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)
    runner = BT.BatchRunner(packets, out / "glm" / "critic_batch", make_transport(keys, read_timeout_s=120.0, total_timeout_s=1800.0), list(keys), load_concurrency(), max_retries=1, glm_config=cfg)
    sent = runner.run()
    acc = ACC.from_broker(runner.broker)
    summary = {"tag": "[EL-ARCH-1 D1] live GLM step", "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "model": glm.GLM_MODEL, "key_labels": list(keys), "send_summary": sent["summary"],
               "recovered": sent["recovered"], "states": sent["manifest"]["states"], "stop_all_critic_calls": sent["manifest"]["stop_all_critic_calls"], "glm_accounting": acc}
    leaks = [str(p.relative_to(out)) for p in sorted((out / "glm").rglob("*")) if p.is_file() and (KEY_RE.search(p.read_text("utf-8", "replace")) or any(k in p.read_text("utf-8", "replace") for k in keys.values()))]
    summary["secret_scan_hits"] = leaks
    with open(out / "glm" / ("GLM_LIVE_SUMMARY_%s.json" % summary["utc"]), "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in ("states", "stop_all_critic_calls", "secret_scan_hits")}, default=str))
    unresolved = sent["manifest"]["states"].get("not_completed", 0)
    return 1 if leaks or unresolved or sent["manifest"]["stop_all_critic_calls"] else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
