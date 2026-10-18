"""Reference implementation of the Evidence-Ladder v1 contracts (S1-S5) as plain code. Pure logic: no models, no network.
It is the controller's decision core, small enough to test exhaustively; the full controller will import it."""
import hashlib, json, shutil, tempfile
from pathlib import Path

SCHEMAS = json.loads((Path(__file__).resolve().parent.parent / "schemas_v1.json").read_text(encoding="utf-8"))
AUTHORITATIVE_SOURCES = {"environment_observation", "existing_test", "user_test", "certified_check"}
TRUSTED_RANK_SOURCES = AUTHORITATIVE_SOURCES
NOT_PASS = {"fail", "unknown", "timeout", "error", "missing_evidence", "invalid_spec"}


def validate(def_name, obj):
    from jsonschema import Draft202012Validator
    Draft202012Validator({"$ref": "#/$defs/" + def_name, "$defs": SCHEMAS["$defs"]}).validate(obj)


def sha256(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


# ---------------- S1 requirements ----------------
def check_requirement(req, request_text):
    """An 'explicit' requirement must quote the request exactly; otherwise it becomes a derived hypothesis that needs validation (never dropped, never promoted)."""
    r = dict(req)
    if r["kind"] == "explicit":
        s = r["source"]
        if request_text[s.get("char_start", 0):s.get("char_end", 0)] != s.get("quote"):
            r.update(kind="derived_hypothesis", status="needs_validation", source={"type": "inference"}, rationale="quote does not match the request")
    validate("requirement", r)
    return r


def can_gate(req):
    return req["kind"] in ("explicit", "contract") or req["status"] == "validated"


def contradictions(reqs):
    return [(a["requirement_id"], b) for a in reqs for b in a.get("conflicts_with", []) if a["priority"] in ("hard", "functional")]


# ---------------- S2 evidence ----------------
class Registry:
    def __init__(self, checks):
        for c in checks:
            validate("check_def", c)
        self.checks = {c["check_id"]: c for c in checks}
        self.hash = sha256(json.dumps(checks, sort_keys=True))

    def runtime_lookup(self, check_id):
        c = self.checks[check_id]  # KeyError for unknown checks
        if c["grader_only"] or not c["runtime_visible"]:
            raise PermissionError("grader-only check is not reachable from runtime code")
        return c

    def accept(self, ev):
        """Record admission: the check must be registered; schema must validate."""
        if ev["check_id"] not in self.checks:
            raise KeyError("unregistered check " + ev["check_id"])
        validate("evidence_record", ev)
        return ev


def applies(ev, req, cand_id, artifact_hash, env_rev, registry):
    c = registry.checks[ev["check_id"]]
    return (ev["candidate_id"] == cand_id and ev["artifact_hash"] == artifact_hash and ev["environment_revision"] == env_rev
            and ev["requirement_id"] == req["requirement_id"] and ev["requirement_version"] == req["version"] and ev["check_version"] == c["version"])


def authoritative(ev, registry):
    c = registry.checks[ev["check_id"]]
    return (ev["source"] in AUTHORITATIVE_SOURCES and ev["independence"] != "model_derived"
            and c["trust_class"] in ("authoritative", "authoritative_in_scope") and ev["requirement_id"] in c["covers"])


def row_label(req, evs, cand_id, artifact_hash, env_rev, registry):
    app = [e for e in evs if applies(e, req, cand_id, artifact_hash, env_rev, registry)]
    trusted = [e for e in app if authoritative(e, registry)]
    if any(e["status"] == "fail" for e in trusted):
        return "FAILED"
    if {e["status"] for e in trusted} >= {"pass", "fail"}:
        return "CONFLICTED"
    if any(e["status"] == "pass" for e in trusted):
        return "VERIFIED"
    soft = [e for e in app if e["status"] == "pass"]
    if soft and not any(e["status"] == "fail" for e in app):
        return "SUPPORTED"
    return "UNVERIFIED"


def conflicted(req, evs, cand_id, artifact_hash, env_rev, registry):
    t = [e for e in evs if applies(e, req, cand_id, artifact_hash, env_rev, registry) and authoritative(e, registry)]
    return {e["status"] for e in t} >= {"pass", "fail"}


def answer_label(rows):
    labels = [l for l in rows if l != "NOT_APPLICABLE"]
    if any(l in ("FAILED", "CONFLICTED") for l in labels):
        return "BLOCKED"
    if labels and all(l == "VERIFIED" for l in labels):
        return "VERIFIED"
    if labels and all(l in ("VERIFIED", "SUPPORTED") for l in labels):
        return "SUPPORTED"
    return "UNVERIFIED"


def decide(reqs, evs, cand_id, artifact_hash, env_rev, registry, risk="sandboxed", user_confirmed=False, decision_type="commit"):
    gating = [r for r in reqs if can_gate(r) and r["priority"] in ("hard", "functional")]
    rows, labels = [], []
    for r in gating:
        lab = row_label(r, evs, cand_id, artifact_hash, env_rev, registry)
        labels.append(lab)
        rows.append({"requirement_id": r["requirement_id"], "label": lab, "evidence_ids": [e["evidence_id"] for e in evs if e["requirement_id"] == r["requirement_id"]]})
    ans = answer_label(labels)
    ok = ans == "VERIFIED" and (risk != "irreversible" or user_confirmed)
    d = {"decision_id": "d-" + cand_id, "decision_type": decision_type, "candidate_id": cand_id, "artifact_hash": artifact_hash, "requirement_rows": rows,
         "answer_label": ans, "next_step": "commit" if ok else ("block" if ans == "BLOCKED" else "gather_evidence"), "authorized_to_execute": ok, "risk": risk, "derived_by": "controller"}
    validate("decision_record", d)
    return d


# ---------------- S3 scorers ----------------
def scorer_allowed(scorer_def, decision_type, exploratory=False):
    validate("scorer_def", scorer_def)
    return scorer_def["status"] != "disabled" and (decision_type in scorer_def["decision_types_validated"] or exploratory)


def assert_frozen(before_hash, after_hash):
    if before_hash != after_hash:
        raise RuntimeError("registry or scorer changed during the run (I7)")


# ---------------- S5 ranking and stopping ----------------
def rank(cands):
    """cands: dicts with id, trusted_passes, validated_passes, cluster, scorer, deepconf. Lexicographic: a learned score never reorders across trusted evidence."""
    return sorted(cands, key=lambda c: (-c["trusted_passes"], -c["validated_passes"], -c["cluster"], -c.get("scorer", 0.0), -c.get("deepconf", 0.0), c["id"]))


def stop_cause(rounds, budget_used, budget_cap, decided=False, ambiguous_unresolvable=False, contradictory=False, tool_failure=False, api_down=False):
    """rounds: list of {'failure_sigs': [..], 'progress': bool, 'regression': bool, 'lineage': id}. First matching cause wins."""
    if decided: return "decided"
    if contradictory: return "contradictory_requirements"
    if ambiguous_unresolvable: return "missing_information"
    if tool_failure: return "tool_failure"
    if api_down: return "api_unavailable"
    if budget_used >= budget_cap: return "budget"
    by = {}
    for r in rounds:
        by.setdefault(r["lineage"], []).append(r)
    for L in by.values():
        for a, b in zip(L, L[1:]):
            if set(a["failure_sigs"]) & set(b["failure_sigs"]) and b["failure_sigs"]:
                return "same_failure_twice"
        for i, r in enumerate(L):
            if r.get("regression") and i + 1 < len(L) and L[i + 1].get("regression"):
                return "unrepaired_regression"
    if len(rounds) >= 2 and not rounds[-1]["progress"] and not rounds[-2]["progress"]:
        return "no_progress_2"
    return None


# ---------------- S4 transactional commit ----------------
def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def commit_patch(base_dir, patch_fn, verify_fn):
    """Apply patch_fn to a copy of base_dir, verify there, then replace base only if verification passes. Any failure leaves base byte-identical."""
    base = Path(base_dir)
    work = Path(tempfile.mkdtemp(prefix="wt-")) / "wt"
    shutil.copytree(base, work)
    try:
        patch_fn(work)
        if not verify_fn(work):
            raise RuntimeError("verification failed on the worktree")
    except BaseException:
        shutil.rmtree(work.parent, ignore_errors=True)
        return False
    for p in list(base.iterdir()):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in work.iterdir():
        shutil.move(str(p), base / p.name)
    shutil.rmtree(work.parent, ignore_errors=True)
    return True
