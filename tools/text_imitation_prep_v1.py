"""Build the upload of text-imitation v1 (no model, no compute): dataset v3 TRAIN pools + 46 cleaned VAL pools, [system] block stripped,
secret scan (patterns of tools/secret_scan_parquet_v1.py + JWT) with masking, sha256 checks against the frozen sets.

  python tools/text_imitation_prep_v1.py <results dir>
Writes <results dir>/upload/{pools.jsonl.gz, thresholds_text_imitation_v1.toml, dataset-metadata.json} and <results dir>/SCAN_REPORT.json.
"""
import ast
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import sys

V3 = "results/raw/laya-fullaction-set-v3-20261012T055420Z"
VAL = "results/raw/laya-fullaction-clean-v4-20261012T130553Z/val_v1_clean"
SHA = {V3 + "/FROZEN_SET.json": "725cd39dd27f0232de9ebd700a91042cec993505e8fedd8117aa024ceed4d8f4",
       V3 + "/requests.jsonl.gz": "f8addc405ba5eb6f87eddd8ea4bba5504ddda24c34dfa454048e02bf0b237c27",
       V3 + "/private_labels.jsonl.gz": "87b1424757a6e0bd2dd2cbe5cd14f19ed0e2c4f649524c6bfb7f5b0e17b61539",
       VAL + "/requests.jsonl.gz": "be0a7757581a7e7128e9c1bcfffe4b44dba2b3302f1e3f8da34e732aea165005",
       VAL + "/private_labels.jsonl.gz": "8aedf21446c01721bd2ad4754c02684bae8eb77bba0fcb998ec635ee16437d93"}
PREREG = "configs/thresholds_text_imitation_v1.toml"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def patterns():
    tree = ast.parse(open("tools/secret_scan_parquet_v1.py", encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PATTERNS":
            p = ast.literal_eval(n.value)
            p["jwt"] = r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            return p
    raise SystemExit("PATTERNS not found")


def strip_system(ctx):
    m = re.search(r"^\[user\]$", ctx, re.M)
    assert ctx.startswith("[system]\n") and m, "unexpected context start"
    return ctx[m.start():]


def main(out):
    os.makedirs(out + "/upload", exist_ok=False)
    for p, h in SHA.items():
        assert sha(p) == h, p
    P = patterns()
    RX = {k: re.compile(v) for k, v in P.items()}
    hits = collections.Counter()
    hit_pools = collections.defaultdict(set)

    def scan(text, field, pid):
        for k, rx in RX.items():
            text, n = rx.subn("[REDACTED_SECRET]", text)
            if n:
                hits[(k, field)] += n
                hit_pools[k].add(pid)
        return text
    rows, chars = [], 0
    for d, split in ((V3, "train"), (VAL, "val")):
        req = {json.loads(l)["decision_id"]: json.loads(l) for l in gzip.open(d + "/requests.jsonl.gz", "rt", encoding="utf-8")}
        for l in gzip.open(d + "/private_labels.jsonl.gz", "rt", encoding="utf-8"):
            lab = json.loads(l)
            assert lab["split"] == split and lab["origin"][lab["reference_index"]] == "reference"
            r = req[lab["decision_id"]]
            ctx = strip_system(r["context_text"])
            chars += len(ctx) + sum(len(o) for o in r["options"])
            rows.append(dict(pool_id=lab["decision_id"], trajectory_id=lab["trajectory_id"], split=split,
                             context_text=scan(ctx, "context", lab["decision_id"]),
                             options=[scan(o, "option", lab["decision_id"]) for o in r["options"]], label_index=lab["reference_index"]))
    tr = {r["trajectory_id"] for r in rows if r["split"] == "train"}
    va = {r["trajectory_id"] for r in rows if r["split"] == "val"}
    assert not tr & va
    n = collections.Counter(r["split"] for r in rows)
    assert n == {"train": 335, "val": 46}, n
    # re-scan the masked text: must be clean
    residual = sum(len(rx.findall(r["context_text"] + "\n".join(r["options"]))) for r in rows for rx in RX.values())
    assert residual == 0
    with gzip.open(out + "/upload/pools.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(PREREG, out + "/upload/")
    json.dump({"title": "laya-text-imitation-v1", "id": "sangeetasangamkar/laya-text-imitation-v1", "licenses": [{"name": "other"}]},
              open(out + "/upload/dataset-metadata.json", "w"), indent=1)
    rep = dict(patterns=P, characters_scanned=chars, pools=dict(n), trajectories=dict(train=len(tr), val=len(va)),
               hits={"%s|%s" % k: v for k, v in sorted(hits.items())}, pools_with_hits={k: sorted(v) for k, v in hit_pools.items()},
               total_hits=sum(hits.values()), masked_with="[REDACTED_SECRET]", residual_after_mask=residual,
               inputs_sha256=SHA, prereg_sha256=sha(PREREG), upload_sha256={f: sha(out + "/upload/" + f) for f in os.listdir(out + "/upload")},
               fields=["pool_id", "trajectory_id", "split", "context_text (no [system] block)", "options", "label_index"],
               not_uploaded="CAL and TEST: not read, not uploaded; label metadata other than label_index; [system] block")
    json.dump(rep, open(out + "/SCAN_REPORT.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("patterns", "pools_with_hits")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
