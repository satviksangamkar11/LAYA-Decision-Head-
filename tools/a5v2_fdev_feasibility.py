"""F-dev CPU feasibility of the FROZEN sampler a5v2-odds-1 on fresh trajectories. Criteria: configs/a5_v3_spec.toml [fdev_feasibility] (registered before F-dev was compiled).
Old data only through compiler.firewall.load_dev (TRAIN). F-dev is read directly (it is allowed). The locked fresh file is never opened. Output create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/a5v2_fdev_feasibility.py
Steps: (1) rebuild old train features, regenerate the audit's out-of-fold sets and check they equal the registered audit examples; (2) fit and save the frozen odds
models on ALL old train covered rows; (3) build F-dev covered decisions and generate their sets with the frozen models; (4) attackers trained on the old sets, scored on F-dev."""
import collections, hashlib, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
from compiler import a5v2
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
spec3 = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["fdev_feasibility"]
audit = json.load(open(ROOT / "results/raw/a5v2-train-cv-audit-20261001-194503.json", encoding="utf-8"))
raw_old = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
raw_fdev = a5v2.load_raw(ROOT / "data/raw/swerebench-fresh-fdev.jsonl")

# ---- old train ----
old_all, old_decs = 0, []
for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
    old_all += 1
    if not r["candidate_coverage"]:
        continue
    d, why = a5v2.build_decision(r, raw_old[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
    if d:
        old_decs.append(d)
old_covered_share = 3907 / old_all
old_repeat_share = sum(d["stratum"] == "repeat" for d in old_decs) / len(old_decs)
oof = a5v2.oof_sets(old_decs)
ex = audit["reproducibility_example"]
repro = {k: (oof[k]["order"] == v["order"] and oof[k]["seed"] == v["seed"]) for k, v in ex.items()}
frozen = a5v2.fit_frozen(old_decs)
(ROOT / "data/a5v2").mkdir(parents=True, exist_ok=True)
frozen_hash = a5v2.save_frozen(frozen, ROOT / "data/a5v2/odds-a5v2-odds-1.pt")
v1_old = {d["did"]: a5v2.v1_set(d) for d in old_decs}
old_v1 = [d for d in old_decs if v1_old[d["did"]]["true_pos"] is not None]

# ---- F-dev ----
fd_all = fd_cov = 0
fd_decs, excl = [], collections.Counter()
for l in open(ROOT / "data/decisions/fdev-v3.jsonl", encoding="utf-8"):
    r = json.loads(l)
    fd_all += 1
    if not r["candidate_coverage"]:
        excl["uncovered (diagnostic only)"] += 1
        continue
    fd_cov += 1
    d, why = a5v2.build_decision(r, raw_fdev[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
    if d:
        fd_decs.append(d)
    else:
        excl[why] += 1
fd_v2 = {d["did"]: a5v2.generate(d, frozen[d["stratum"]]) for d in fd_decs}
fd_v1 = {d["did"]: a5v2.v1_set(d) for d in fd_decs}
fd_v1_ds = [d for d in fd_decs if fd_v1[d["did"]]["true_pos"] is not None]

res = dict(frozen_odds_sha256=frozen_hash, reproduction_of_registered_audit_examples=repro, old=dict(all_train_decisions=old_all, covered_usable=len(old_decs), covered_share=round(old_covered_share, 4), repeat_share=round(old_repeat_share, 4)),
           fdev=dict(decisions=fd_all, covered=fd_cov, usable=len(fd_decs), covered_share=round(fd_cov / fd_all, 4), repeat_share=round(sum(d["stratum"] == "repeat" for d in fd_decs) / max(1, len(fd_decs)), 4),
                     trajectories=len({d["tid"] for d in fd_decs}), exclusions=dict(excl), strata=dict(collections.Counter(d["stratum"] for d in fd_decs))))
res["retention"] = round(len(fd_decs) / max(1, fd_cov), 4)
out = {}
for which in ("A", "B"):
    net2 = a5v2.fit_attacker(old_decs, oof, which)
    out[f"v2_{which}"] = a5v2.score_attacker(net2, fd_decs, fd_v2, which)
    net1 = a5v2.fit_attacker(old_v1, {d["did"]: v1_old[d["did"]] for d in old_v1}, which)
    out[f"v1_{which}"] = a5v2.score_attacker(net1, fd_v1_ds, fd_v1, which)
res["attackers_trained_on_old_scored_on_fdev"] = out
res["old_grouped_cv_reference"] = dict(v2_A=audit["A5_v2_sets"]["attacker_A_8_features"]["overall"]["acc"], v2_B=audit["A5_v2_sets"]["attacker_B_all_features_plus_hashed_tokens"]["overall"]["acc"],
                                      v1_A=audit["A5_v1_sets_same_audit"]["attacker_A_8_features"]["overall"]["acc"], v1_B=audit["A5_v1_sets_same_audit"]["attacker_B_all_features_plus_hashed_tokens"]["overall"]["acc"])
res["fdev_v2_within_decision_auc"] = a5v2.auc_by_feature(fd_decs, fd_v2)
ref = res["old_grouped_cv_reference"]
chk = dict(
    enough_covered=len(fd_decs) >= spec3["min_covered_fdev_decisions"],
    retention=res["retention"] >= spec3["retention_min"],
    covered_share=abs(res["fdev"]["covered_share"] - old_covered_share) <= spec3["covered_share_tolerance"],
    repeat_share=abs(res["fdev"]["repeat_share"] - old_repeat_share) <= spec3["repeat_share_tolerance"],
    attacker_A_transfer=abs(out["v2_A"]["overall"]["acc"] - ref["v2_A"]) <= spec3["attacker_transfer_tolerance"],
    attacker_B_transfer=abs(out["v2_B"]["overall"]["acc"] - ref["v2_B"]) <= spec3["attacker_transfer_tolerance"],
    audit_reproduced=all(repro.values()))
res["criteria"] = chk
res["verdict"] = "SAMPLER USABLE ON FRESH DATA" if all(chk.values()) else "NOT USABLE: " + ", ".join(k for k, v in chk.items() if not v)
body = json.dumps(res, indent=1)
p = ROOT / ("results/raw/fdev-feasibility-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    f.write(body)
Path(str(p)[:-5] + ".sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  " + p.name + "\n")
print(json.dumps({k: res[k] for k in ("reproduction_of_registered_audit_examples", "old", "fdev", "retention", "attackers_trained_on_old_scored_on_fdev", "old_grouped_cv_reference", "criteria", "verdict")}, indent=1))
print("->", p)
