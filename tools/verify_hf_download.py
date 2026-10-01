"""Verify a Hugging Face model directory against the official manifest. Read-only.

Checks (all must pass for exit code 0):
  1. every manifest file (minus --exclude prefixes) exists with the exact byte size
  2. no leftover .incomplete download parts
  3. --deep: each .safetensors header parses and its tensor data ends exactly at EOF
  4. --deep: index.json weight_map and shard headers agree (no missing, no extra tensors)
  5. --deep: sha256 of every LFS file equals the manifest sha256

Usage: python verify_hf_download.py MANIFEST.json LOCAL_DIR [--deep] [--exclude original/ metal/]
Writes results/raw/verify-<dir>-<mode>-<time>.json (create-only).
"""
import argparse, hashlib, json, struct, sys, time
from pathlib import Path


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    return header, 8 + n + end


def sha256(path, label):
    h, done, size, t0 = hashlib.sha256(), 0, path.stat().st_size, time.time()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            done += len(chunk)
            if done % (1 << 30) < (8 << 20):
                print(f"  {label}: {done/2**30:.1f}/{size/2**30:.1f} GiB  {time.time()-t0:.0f}s", flush=True)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("local")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    man = json.load(open(a.manifest)); root = Path(a.local)
    want = [r for r in man["files"] if not r["name"].startswith(tuple(a.exclude))]
    rep = {"repo": man["repo"], "revision": man["revision"], "mode": "deep" if a.deep else "quick",
           "missing": [], "wrong_size": [], "incomplete_parts": [], "bad_header": [],
           "index_missing_tensors": [], "index_extra_tensors": [], "bad_sha256": [], "ok_files": 0}

    for r in want:
        p = root / r["name"]
        if not p.exists():
            rep["missing"].append(r["name"])
        elif r["size"] is not None and p.stat().st_size != r["size"]:
            rep["wrong_size"].append({"file": r["name"], "have": p.stat().st_size, "want": r["size"]})
        else:
            rep["ok_files"] += 1
    dl = root / ".cache" / "huggingface" / "download"
    if dl.exists():
        rep["incomplete_parts"] = sorted(x.name for x in dl.glob("*.incomplete"))

    if a.deep:
        present = [r for r in want if (root / r["name"]).exists() and r["name"] not in {w["file"] for w in rep["wrong_size"]}]
        shard_tensors = {}
        for r in present:
            if r["name"].endswith(".safetensors"):
                try:
                    hdr, expect = st_header(root / r["name"])
                    if expect != (root / r["name"]).stat().st_size:
                        rep["bad_header"].append({"file": r["name"], "header_says": expect, "size": (root / r["name"]).stat().st_size})
                    shard_tensors[r["name"]] = {k for k in hdr if k != "__metadata__"}
                except Exception as e:
                    rep["bad_header"].append({"file": r["name"], "error": repr(e)})
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            wm = json.load(open(idx))["weight_map"]
            for t, shard in wm.items():
                if shard not in shard_tensors or t not in shard_tensors[shard]:
                    rep["index_missing_tensors"].append(f"{t} -> {shard}")
            indexed = set(wm)
            for shard, ts in shard_tensors.items():
                rep["index_extra_tensors"] += [f"{t} in {shard}" for t in ts if t not in indexed]
        for r in present:
            if r["sha256"]:
                got = sha256(root / r["name"], r["name"])
                if got != r["sha256"]:
                    rep["bad_sha256"].append({"file": r["name"], "got": got, "want": r["sha256"]})

    problems = [k for k in ("missing", "wrong_size", "incomplete_parts", "bad_header",
                            "index_missing_tensors", "index_extra_tensors", "bad_sha256") if rep[k]]
    rep["verdict"] = "COMPLETE AND VERIFIED" if not problems else "INCOMPLETE: " + ", ".join(problems)
    out = Path(__file__).resolve().parent.parent / "results" / "raw" / \
        f"verify-{root.name}-{rep['mode']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:12]) for k, v in rep.items()}, indent=1))
    print("wrote", out)
    sys.exit(0 if not problems else 1)


main()
