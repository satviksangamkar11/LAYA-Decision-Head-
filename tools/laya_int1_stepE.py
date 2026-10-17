"""[LAYA-INT-1 · Aim: integration, not performance] STEP E: the GLM-5.3-Flash critic on the three validated end-state packets, through the REAL controller.

  --mock   fake transport, no keys, no network (tests the whole path; still replays the controller in the WSL sandbox)
  --live   at most 3 initial requests (one per packet) through PcSender + StagedCriticBroker (critic_batch config: ramp from 2/key, cap 10, <=39/min/key, 60 s cooldown on 429),
           z-ai/glm-5.3-flash, reasoning_effort max, STREAMED (critic_batch/transport.py). Keys: .secrets/nvidia_api_key_<n>.txt (n >= 2), read INTO MEMORY here, labels key_1..key_k; values are never
           printed, logged or written; no key file readable -> explicit error and ZERO calls; 402/quota -> stop everything and report.

What it does: reuses the 3 packets of e2e_pass2/E_critic_receipt.json (requests read from run1/critic_ledger.jsonl and checked against their ids), replays the controller to the same end states
(Kaggle outputs are CACHED: nothing is pushed to Kaggle, no Qwen), sends, ingests, runs SUPPORTED experiments in the WSL sandbox (evidence_ladder.models.experiments: one shared allow-list),
REJECTS unsupported/unsafe ones explicitly (journal entry + reason), records diagnostic (L0, model_derived) evidence, runs assert_invariants I1-I8 and writes a journal EXTENSION that
references the untouched originals. Create-only output: results/raw/laya-integration-20261016T184816Z/stepE_<UTC>/ (stepE_mock_<UTC>/ with --mock).
  & 'D:\\local model\\.venv\\Scripts\\python.exe' 'D:\\local model\\tools\\laya_int1_stepE.py' --live"""
import hashlib, json, re, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
TAG = "[LAYA-INT-1 · Aim: integration, not performance] STEP E"
BASE = ROOT / "results/raw/laya-integration-20261016T184816Z"
E2E_DIR = BASE / "e2e_pass2"
NAMES = ["ex1_slug", "ex2_clamp", "ex3_wordcount"]
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{16,}|\bsk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}")


def sha(b): return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()
def write_new(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f: f.write(text)
def jdump(o): return json.dumps(o, indent=1, ensure_ascii=False, default=str, sort_keys=True)
def lf_sha(p):
    b = Path(p).read_bytes()
    return sha(b.replace(b"\r\n", b"\n"))


def original_hashes():
    fs = [E2E_DIR / "E2E_SUMMARY.json", E2E_DIR / "E_critic_receipt.json", E2E_DIR / "run1/critic_ledger.jsonl"] + [E2E_DIR / "run1" / n / f for n in NAMES for f in ("journal.json", "result.json", "steps.json")]
    return {p.relative_to(BASE).as_posix(): lf_sha(p) for p in fs}


def load_keys():
    """-> {label: key}. In-memory only. Same key system as tools/judge_qual_v1.py: .secrets/nvidia_api_key_<n>.txt, n >= 2, labelled key_1..key_k."""
    fs = [p for p in (ROOT / ".secrets").glob("nvidia_api_key_*.txt") if re.fullmatch(r"nvidia_api_key_(\d+)\.txt", p.name) and int(p.name[15:-4]) >= 2]
    keys = {}
    for p in sorted(fs, key=lambda p: int(p.name[15:-4])):
        try: v = p.read_text(encoding="utf-8-sig").strip()
        except OSError: continue
        if v: keys["key_%d" % (len(keys) + 1)] = v
    return keys


def saved_requests():
    rec = json.loads((E2E_DIR / "E_critic_receipt.json").read_text("utf-8"))
    built = {}
    for line in (E2E_DIR / "run1/critic_ledger.jsonl").read_text("utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("event") == "packet_built": built[ev["packet_id"]] = ev["request"]
    from evidence_ladder.models.broker import packet_id_of
    out = {}
    for n in NAMES:
        pid = rec["packets"][n]["packet_id"]
        req = built[pid]
        assert packet_id_of(req) == pid, "saved request does not hash to its packet id: " + n
        assert sha(json.dumps(req, sort_keys=True)) == rec["packets"][n]["request_sha256"], "request_sha256 differs from the receipt: " + n
        out[n] = (pid, req)
    return out


# ------------------------------------------------------------------ replay of the originals (cached Kaggle outputs, WSL sandbox; no Qwen, no Kaggle)
def cached_round(tag):
    d = E2E_DIR / "_kaggle" / tag; res = d / "output" / "t6_step_results.json"
    if not res.exists(): raise RuntimeError("cached Kaggle round missing: %s (step E never pushes to Kaggle)" % tag)
    return json.loads(res.read_text("utf-8")), json.loads((d / "run.json").read_text("utf-8"))["run_id"]


def replay(work: Path):
    import laya_int1_e2e as E2E
    runs, backend = E2E.build_runs(work, "replay")
    for step in (1, 2):
        active = [n for n, r in runs.items() if not r.decided]
        if not active: break
        items = [{"key": "%s:%d" % (n, step), "step": step, "context_text": runs[n].context_text()} for n in active]
        live, live_id = cached_round("live_s%d" % step); ver, ver_id = cached_round("verify_s%d" % step)
        for it in items:
            k = it["key"]; lv, vr = live["steps"].get(k, {}), ver["steps"].get(k, {})
            if lv.get("status") == "ok" and vr.get("status") == "ok": backend.add(lv, vr, live_run=live_id, verify_run=ver_id, verify_mode=ver["mode"])
        for it in items:
            n = it["key"].split(":")[0]; lv = live["steps"].get(it["key"], {})
            meta = dict(lv.get("generation", {}), kaggle_seconds=live.get("total_seconds", 0) / max(len(items), 1))
            runs[n].step(lv.get("options", []), meta)
    for n, r in runs.items():        # the replay must reach the SAME decisions as the originals
        orig = json.loads((E2E_DIR / "run1" / n / "steps.json").read_text("utf-8"))
        mine = [{"step": s["step"], "decision_id": s["decision_id"], "chosen_index": s["chosen_index"], "tests_pass": s["tests_pass"], "label": s["label"]} for s in r.steps]
        assert mine == orig, "replay of %s differs from the original steps.json" % n
    return runs


# ------------------------------------------------------------------ mock transport
def mock_transport(pids):
    """Canned GLM answers (stream-shaped bodies). ex1: one transport failure then a function_call; ex2: the exact test command; ex3: an UNSAFE command (must be rejected, never run)."""
    calls = {}

    def body(obj, tin=900, tout=400):
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(obj), "reasoning_content": "mock reasoning"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": tin, "completion_tokens": tout}}
    answers = {
        pids["ex1_slug"]: {"experiment": {"kind": "function_call", "payload": 'slugify("  A--B  ")', "expected_by_candidate": {}}, "implications": "mock: a-b expected"},
        pids["ex2_clamp"]: {"experiment": {"kind": "command", "payload": "python3 test_clamp.py", "expected_by_candidate": {}}, "implications": "mock: the test command runs"},
        pids["ex3_wordcount"]: {"experiment": {"kind": "command", "payload": "python3 wc.py; cat ../../.secrets/x", "expected_by_candidate": {}}, "implications": "mock: unsafe"}}

    def transport(packet, key_label):
        pid = packet["packet_id"]; calls[pid] = calls.get(pid, 0) + 1
        if pid == pids["ex1_slug"] and calls[pid] == 1: return None, {"error": "mock transport failure"}
        return 200, body(answers[pid])
    transport.calls = calls
    return transport


def main():
    mock, live = "--mock" in sys.argv, "--live" in sys.argv
    if mock == live: sys.exit("exactly one of --mock / --live")
    keys = {}
    if live:
        keys = load_keys()
        if not keys: sys.exit("ERROR: no readable .secrets/nvidia_api_key_<n>.txt (n >= 2); zero calls made")
    from evidence_ladder.critic_batch import batch as BT, validator as V
    from evidence_ladder.critic_batch.config import load_concurrency
    from evidence_ladder.critic_batch.transport import make_transport
    from evidence_ladder.models import experiments as EXP, glm
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = BASE / (("stepE_mock_" if mock else "stepE_") + stamp); out.mkdir(parents=True)
    before = original_hashes()
    saved = saved_requests()
    pids = {n: saved[n][0] for n in NAMES}
    for n in NAMES: V.validate_packet({"packet_id": pids[n], "request": saved[n][1]})       # blinding / allow-list validator on the reused packets
    runs = replay(out)
    for n, r in runs.items():
        assert r.critic_request() == saved[n][1], "replayed end state of %s does not rebuild the saved packet" % n
    pre_labels = {n: r.steps[-1]["label"] for n, r in runs.items()}
    # --- send (or mock-send) through PcSender + StagedCriticBroker via BatchRunner
    man = V.write_packets(out / "critic_packets", {pids[n]: saved[n][1] for n in NAMES})
    labels = list(keys) if live else ["key_1", "key_2", "key_3"]
    transport = make_transport(keys, read_timeout_s=120.0, total_timeout_s=1800.0) if live else mock_transport(pids)
    cfg = glm.GlmConfig(reasoning_params=dict(BT.REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)
    runner = BT.BatchRunner(out / "critic_packets", out / "critic_batch", transport, labels, load_concurrency(), max_retries=1, glm_config=cfg)
    sent = runner.run()
    broker = runner.broker
    # --- controller ingest + experiments + invariants
    results = {}
    for n, r in runs.items():
        r.ctx.critic = broker
        state = r.critic_stage()
        res = r.finish()
        d = out / "extension" / n
        write_new(d / "journal.json", jdump(r.journal.entries()))
        orig = json.loads((E2E_DIR / "run1" / n / "result.json").read_text("utf-8"))
        pid = pids[n]
        ev = [e for e in r.journal.evidence(r.last_state) if e["check_id"] == "critic_experiment_v1"]
        results[n] = {"packet_id": pid, "critic_stage": state, "experiment_decision": r.critic_result, "label_before_critic": pre_labels[n], "label_after": res.answer_label,
                      "critic_changed_label": pre_labels[n] != res.answer_label,
                      "critic_evidence_records": [{"id": e["evidence_id"], "status": e["status"], "independence": e["independence"], "check_id": e["check_id"]} for e in ev],
                      "invariants_checked": r.invariants_checked, "journal_hash_extension": res.journal_hash, "stop_cause": res.stop_cause,
                      "original_receipts": {"journal_hash_original": orig["journal_hash"], "answer_label_original": orig["answer_label"],
                                            "critic_original_state": "unavailable (key system blocked), see E_critic_receipt.json"},
                      "broker": {"state": broker.packets[pid]["state"], "reason": broker.packets[pid]["reason"], "attempts_failed": broker.packets[pid]["attempts"]},
                      "send_response_ledger": [e for e in runner.ledger.events if e.get("packet_id") == pid],
                      "parsed_critic_result": broker.packets[pid]["response"]}
        write_new(d / "step_e_receipt.json", jdump(results[n]))
        write_new(d / "critic_packet.json", jdump({"packet_id": pid, "request": saved[n][1]}))
        write_new(d / "result.json", jdump({"answer_label": res.answer_label, "stop_cause": res.stop_cause, "journal_hash": res.journal_hash, "invariants": r.invariants_checked, "budget_record": res.budget_record}))
    sends = [e for e in runner.ledger.events if e["event"] == "send"]
    summary = {"tag": TAG, "mode": "mock" if mock else "live", "model": glm.GLM_MODEL, "reasoning": cfg.reasoning_params, "allowed_kinds": list(EXP.SUPPORTED_KINDS), "packets": man["packet_ids"],
               "send_summary": sent["summary"], "recovered": sent["recovered"], "bundle": {k: sent["manifest"][k] for k in ("count", "states", "responses_sha256", "stop_all_critic_calls")},
               "initial_requests": sum(1 for e in sends if e.get("attempt") == 1), "sends_total": len(sends),
               "examples": {n: {"critic_stage": v["critic_stage"], "label_before_critic": v["label_before_critic"], "label_after": v["label_after"], "critic_changed_label": v["critic_changed_label"],
                                "invariants_checked": v["invariants_checked"], "broker": v["broker"], "experiment": (v["experiment_decision"] or {}).get("state"),
                                "reason": (v["experiment_decision"] or {}).get("reason")} for n, v in results.items()},
               "invariants_I1_I8_all_pass": all(len(v["invariants_checked"]) == 8 for v in results.values()), "critic_changed_any_label": any(v["critic_changed_label"] for v in results.values()),
               "originals_unchanged": original_hashes() == before, "originals_sha256": before, "stop_all_critic_calls": broker.stop_all_critic_calls}
    write_new(out / "STEPE_SUMMARY.json", jdump(summary))
    shutil.rmtree(out / "replay" / "_sandbox", ignore_errors=True)
    leaks = []       # no key value and no credential-shaped text may appear in any output (labels only)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            t = p.read_text("utf-8", "replace")
            if KEY_RE.search(t) or any(k in t for k in keys.values()): leaks.append(p.relative_to(out).as_posix())
    lines = ["%s  %s" % (lf_sha(p), p.relative_to(out).as_posix()) for p in sorted(out.rglob("*")) if p.is_file()]
    write_new(out / "FINAL_SHA256SUMS.txt", "\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("mode", "send_summary", "initial_requests", "sends_total", "invariants_I1_I8_all_pass", "critic_changed_any_label", "originals_unchanged", "stop_all_critic_calls", "examples")}, indent=1, default=str))
    print("secret scan leaks:", leaks or "none"); print("out:", out)
    sys.exit(1 if leaks or not summary["originals_unchanged"] or not summary["invariants_I1_I8_all_pass"] or summary["stop_all_critic_calls"] else 0)


if __name__ == "__main__":
    main()
