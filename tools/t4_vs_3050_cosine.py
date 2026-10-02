"""Cross-device check: Kaggle T4 fp16 capture vs the RTX 3050 bf16 files, same decisions (file names are by manifest position, so they match).
    PYTHONPATH=. .venv/Scripts/python.exe tools/t4_vs_3050_cosine.py DIR_WITH_T4_PT_FILES [DIR_3050=data/v3/capture/train]
Reports per-depth min/mean cosine over query and option vectors. No threshold is registered for this yet (amendment (e) is pending), so 0.999
(capture_one_record.plain_cos_min) is shown only as a reference line, not as a gate."""
import sys, json, glob, os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
import torch.nn.functional as F

t4, ref = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data/v3/capture/train")
rows, worst = [], {}
for f in sorted(glob.glob(os.path.join(t4, "*.pt"))):
    a, b = torch.load(f), torch.load(os.path.join(ref, os.path.basename(f)))
    assert a["decision_id"] == b["decision_id"] and a["pool_hash"] == b["pool_hash"], f
    for kind in ("query", "cands"):
        x, y = a[kind].double(), b[kind].double()          # query [D,H]; cands [P,D,H]
        c = F.cosine_similarity(x, y, dim=-1)
        for dj, d in enumerate(a["depths"]):
            cd = c[dj] if kind == "query" else c[:, dj]
            w = worst.setdefault((kind, d), [1.0, 0.0, 0])
            w[0] = min(w[0], cd.min().item()); w[1] += cd.mean().item(); w[2] += 1
    rows.append(os.path.basename(f))
out = {f"{k}@{d}": dict(min_cos=round(v[0], 6), mean_cos=round(v[1] / v[2], 6)) for (k, d), v in sorted(worst.items())}
print(json.dumps(dict(files=len(rows), reference_plain_cos_min=0.999, per_depth=out), indent=1))
