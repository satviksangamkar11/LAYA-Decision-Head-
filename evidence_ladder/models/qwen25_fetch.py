"""[Qwen2.5-Coder-1.5B local run] Fetch one pinned Hugging Face snapshot into models/<name>/ and write MANIFEST_pinned.json in the same shape as models/qwen3-0.6b (repo, revision, license, files{sha256,bytes,
lfs_sha256_matches}). Files are streamed to disk by huggingface_hub (nothing is loaded into RAM) and hashed in 16 MB chunks. Create-only: refuses a non-empty target dir.
    PYTHONPATH=. .venv/Scripts/python.exe evidence_ladder/models/qwen25_fetch.py
User approval for this download (~3.1 GB, Apache-2.0): given in chat, 2026-10-18."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
TARGET = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-coder-1.5b-instruct"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download
    if TARGET.exists() and any(TARGET.iterdir()):
        sys.exit("target exists and is not empty (create-only): %s" % TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION, (info.sha, REVISION)
    lic = info.card_data.license if info.card_data else None
    files = {}
    for s in sorted(info.siblings, key=lambda x: x.rfilename):
        path = Path(hf_hub_download(REPO, s.rfilename, revision=REVISION, local_dir=str(TARGET)))
        digest = sha256_file(path)
        lfs = s.lfs.sha256 if getattr(s, "lfs", None) else None
        files[s.rfilename] = {"sha256": digest, "bytes": path.stat().st_size, "lfs_sha256_matches": (digest == lfs) if lfs else None}
        assert files[s.rfilename]["bytes"] == s.size, (s.rfilename, path.stat().st_size, s.size)
        if lfs:
            assert digest == lfs, "LFS sha256 mismatch for %s" % s.rfilename
        print(s.rfilename, files[s.rfilename]["bytes"], files[s.rfilename]["lfs_sha256_matches"], flush=True)
    with open(TARGET / "MANIFEST_pinned.json", "x", encoding="utf-8") as f:
        json.dump({"repo": REPO, "revision": REVISION, "license": lic, "files": files}, f, indent=1)
    print("DONE", len(files), "files, license", lic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
