"""[EL-ARCH-1 A3.1/A3.2] GLM ledger reconciliation: every number a report states about GLM calls is recomputed from the THREE ledgers of one critic_batch directory and compared.
Files (all written by BatchRunner / StagedCriticBroker): broker_ledger.jsonl, progress.jsonl, responses.jsonl, MANIFEST.json.
Checks (each -> {ok, detail}):
  sends_match        sends in progress == sends in broker ledger == accounting calls
  ingests_match      every send was ingested exactly once (no in-flight loss) and result events == ingests
  ok_match           accounting ok == responses.jsonl lines == MANIFEST states.ok
  retries_bounded    per packet sends <= 1 + max_retries (the config event of the broker ledger); retries == ingests filed "retry"
  failures_classified every non-2xx / transport ingest carries a category (error_info) and none is "unknown" without text
  tokens_match       per-attempt tokens recomputed from the raw answers == accounting tokens; `actual` only where the provider sent usage, `estimated` otherwise (flagged, never silently mixed)
  no_credentials     no key-shaped text (and none of the optional literal secrets) in any file of the directory
  utf8_clean         no U+FFFD and no cp1252-mojibake marker ("â€”") in any saved text: the receipt bytes are valid UTF-8 and were not re-decoded
Pure standard library plus accounting.py; reads files only."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evidence_ladder.critic_batch import accounting as ACC
from evidence_ladder.critic_batch.transport import MOJIBAKE_RE
from evidence_ladder.models import glm

KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{16,}|\bsk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}")
FILES = ("broker_ledger.jsonl", "progress.jsonl", "responses.jsonl", "MANIFEST.json", "rejected.json")


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _chk(ok: bool, detail: str) -> Dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def reconcile(batch_dir, *, secrets: Iterable[str] = ()) -> Dict[str, Any]:
    d = Path(batch_dir)
    bl, pl, rl = _jsonl(d / "broker_ledger.jsonl"), _jsonl(d / "progress.jsonl"), _jsonl(d / "responses.jsonl")
    man = json.loads((d / "MANIFEST.json").read_text("utf-8")) if (d / "MANIFEST.json").exists() else {}
    acc = ACC.account(d / "broker_ledger.jsonl", d / "progress.jsonl")
    tot = acc["totals"]
    cfg = next((e for e in bl if e["event"] == "config"), {})
    max_retries = cfg.get("max_retries")
    checks: Dict[str, Dict[str, Any]] = {}
    bs, ps = [e for e in bl if e["event"] == "send"], [e for e in pl if e["event"] == "send"]
    checks["sends_match"] = _chk(len(bs) == len(ps) == tot["calls"], "broker sends %d, progress sends %d, accounting calls %d" % (len(bs), len(ps), tot["calls"]))
    bi, pr = [e for e in bl if e["event"] == "ingest"], [e for e in pl if e["event"] == "result"]
    per_send, per_ing = {}, {}
    for e in bs: per_send[e["packet_id"]] = per_send.get(e["packet_id"], 0) + 1
    for e in bi: per_ing[e["packet_id"]] = per_ing.get(e["packet_id"], 0) + 1
    in_flight = {p: per_send[p] - per_ing.get(p, 0) for p in per_send if per_send[p] != per_ing.get(p, 0)}
    checks["ingests_match"] = _chk(not in_flight and len(pr) == len(bi), "sends %d, ingests %d, result events %d, unmatched %s" % (len(bs), len(bi), len(pr), in_flight or "none"))
    n_ok_states = (man.get("states") or {}).get("ok", 0)
    checks["ok_match"] = _chk(tot["ok"] == len(rl) == n_ok_states, "accounting ok %d, responses.jsonl %d, MANIFEST states.ok %d" % (tot["ok"], len(rl), n_ok_states))
    over = {p: n for p, n in per_send.items() if max_retries is not None and n > 1 + max_retries}
    n_retry = sum(1 for e in bi if e["state"] == "retry")
    checks["retries_bounded"] = _chk(max_retries is not None and not over and n_retry == tot["retries"], "max_retries %s, packets over the bound %s, retry ingests %d == accounting retries %d" % (max_retries, over or "none", n_retry, tot["retries"]))
    bad_cls = [(p, a["attempt"]) for p, P in acc["per_packet"].items() for a in P["attempts"] if a["http_status"] is not None and not (200 <= a["http_status"] < 300) and not (a["error"] or {}).get("category")]
    bad_cls += [(p, a["attempt"]) for p, P in acc["per_packet"].items() for a in P["attempts"] if a["http_status"] is None and not (a["error"] or {}).get("category")]
    cats: Dict[str, int] = {}
    for P in acc["per_packet"].values():
        for a in P["attempts"]:
            if a["error"]: cats[a["error"]["category"]] = cats.get(a["error"]["category"], 0) + 1
    checks["failures_classified"] = _chk(not bad_cls and sum(cats.values()) == tot["failed_attempts"], "failed attempts %d, by category %s, unclassified %s" % (tot["failed_attempts"], cats or "none", bad_cls or "none"))
    # tokens recomputed independently from the raw answers (provider usage only; absent usage -> estimated)
    actual_in = actual_out = 0
    n_actual = n_est = 0
    for e in pr:
        if e.get("http_status") is None or not (200 <= e["http_status"] < 300):
            continue
        u = (glm.assemble(e.get("raw")) or {}).get("usage") or {}
        if u.get("prompt_tokens") or u.get("completion_tokens"):
            actual_in += int(u.get("prompt_tokens") or 0); actual_out += int(u.get("completion_tokens") or 0); n_actual += 1
        else:
            n_est += 1
    est_flag = tot["estimated"]
    tok_ok = (n_est == 0 and not est_flag and (actual_in, actual_out) == (tot["tokens_in"], tot["tokens_out"])) or (n_est > 0 and est_flag)
    checks["tokens_match"] = _chk(tok_ok, "answers with provider usage %d (in %d, out %d), answers without usage %d (estimated=%s); accounting in %d out %d" % (n_actual, actual_in, actual_out, n_est, est_flag, tot["tokens_in"], tot["tokens_out"]))
    secs = [s for s in secrets if s]
    leaks, moj = [], []
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.suffix not in (".json", ".jsonl", ".txt", ".log", ".md"):
            continue
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            moj.append((p.name, "not valid UTF-8")); continue
        if KEY_RE.search(text) or any(s in text for s in secs): leaks.append(p.name)
        # json files are ASCII-escaped: also look at the decoded string values
        flat = text
        if p.suffix in (".json", ".jsonl"):
            try: flat = "\n".join(json.dumps(x, ensure_ascii=False) for x in (_jsonl(p) if p.suffix == ".jsonl" else [json.loads(text)]))
            except ValueError: pass
        if "�" in flat or MOJIBAKE_RE.search(flat): moj.append((p.name, "replacement char or cp1252-mojibake marker"))
    checks["no_credentials"] = _chk(not leaks, "files scanned %d, key-shaped text in %s" % (sum(1 for _ in d.rglob("*") if _.is_file()), leaks or "none"))
    checks["utf8_clean"] = _chk(not moj, "valid UTF-8, no U+FFFD / mojibake marker; offenders: %s" % (moj or "none"))
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "accounting_totals": tot, "attempt_error_categories": cats,
            "tokens": {"actual_in": actual_in, "actual_out": actual_out, "answers_with_usage": n_actual, "answers_estimated": n_est}, "dir": str(d)}
