"""Secret scan shared by the Kaggle bundle builders (same patterns and audited allowlist as tools/make_kaggle_bundle.py; that script keeps its own copy and is not changed).
scan_tree(root) -> (hits, files, audited_seen). A hit means: do not upload."""
import hashlib, re
from pathlib import Path
PATTERNS = {"huggingface token": r"hf_[A-Za-z0-9]{30,}", "github token": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}", "aws key": r"AKIA[0-9A-Z]{16}",
            "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}", "openai-style key": r"sk-[A-Za-z0-9_-]{32,}",
            "google key": r"AIza[0-9A-Za-z_-]{35}", "kaggle key": r'"key"\s*:\s*"[0-9a-f]{32}"', "generic secret assignment": r"(?i)(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]"}
BADNAMES = re.compile(r"(^|/)(\.env.*|id_rsa.*|.*\.pem|.*\.p12|kaggle\.json|credentials.*|\.netrc|.*\.kdbx)$", re.I)
AUDITED = {"27ad5f3557d72992eedde7fb48d5476b2a801d2c204b65fc13804b6303eff9cb": "elastic/elastic-otel-python", "5cb73eef6081c5418a0d84d16698b55c3410a413f2fcb288f3fb68ea6c1ff810": "yt-dlp/yt-dlp",
           "d6cbb053abf2933889a0ccbf6ac244623a63a2e3397e991dde09266bdaa932d1": "tornadoweb/tornado"}


def scan_tree(root):
    root, hits, files, seen = Path(root), [], [], set()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if BADNAMES.search(rel):
            hits.append((rel, 0, "forbidden file name"))
        for i, l in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name, pat in PATTERNS.items():
                for m in re.finditer(pat, l):
                    if name == "generic secret assignment":
                        v = re.search(r"['\"]([A-Za-z0-9_\-/+=]{16,})['\"]", m.group(0)).group(1)
                        h = hashlib.sha256(v.encode()).hexdigest()
                        if h in AUDITED:
                            seen.add(h)
                            continue
                    hits.append((rel, i, name))
        files.append(dict(path=rel, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return hits, files, seen
