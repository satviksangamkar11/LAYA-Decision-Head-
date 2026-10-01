import json, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
info = HfApi().model_info(repo, files_metadata=True)
rows = []
for s in info.siblings:
    rows.append({"name": s.rfilename, "size": s.size, "sha256": (s.lfs.sha256 if s.lfs else None)})
out = {"repo": repo, "revision": info.sha, "files": rows}
json.dump(out, open(sys.argv[2], "w"), indent=1)
tot = sum(r["size"] or 0 for r in rows if not r["name"].startswith(("original/","metal/")))
print(repo, info.sha[:10], len(rows), "files; root checkpoint bytes:", tot, "=", round(tot/2**30,2), "GiB")
