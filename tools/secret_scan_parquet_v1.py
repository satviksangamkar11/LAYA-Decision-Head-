"""Credential scan of a trajectory parquet before upload (user, chat 2026-10-13). Streams batches; reports pattern counts and masked samples only.

  python tools/secret_scan_parquet_v1.py <parquet> <out json>
"""
import collections
import json
import re
import sys

import pyarrow.parquet as pq

PATTERNS = {
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "github_token": r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}",
    "openai_style_key": r"\bsk-(proj-)?[A-Za-z0-9_-]{32,}",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}",
    "nvidia_key": r"\bnvapi-[A-Za-z0-9_-]{20,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "private_key_block": r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
    "stripe_live_key": r"\b(sk|rk)_live_[0-9a-zA-Z]{20,}",
}
RX = {k: re.compile(v) for k, v in PATTERNS.items()}


def mask(s):
    return s[:6] + "..." + s[-2:] if len(s) > 10 else "***"


def main(path, out):
    f = pq.ParquetFile(path)
    hits, samples, runs, chars = collections.Counter(), collections.defaultdict(list), 0, 0
    for b in f.iter_batches(batch_size=200, columns=["trajectory_id", "trajectory"]):
        for tid, msgs in zip(b.column("trajectory_id").to_pylist(), b.column("trajectory").to_pylist()):
            runs += 1
            for m in msgs:
                texts = [m.get("content") or ""] + [c["function"]["arguments"] or "" for c in (m.get("tool_calls") or [])]
                for t in texts:
                    chars += len(t)
                    for k, rx in RX.items():
                        for mm in rx.finditer(t):
                            hits[k] += 1
                            if len(samples[k]) < 5:
                                samples[k].append(dict(trajectory_id=tid, role=m.get("role"), masked=mask(mm.group(0)),
                                                       context=re.sub(re.escape(mm.group(0)), mask(mm.group(0)), t[max(0, mm.start() - 60):mm.end() + 60])))
    rep = dict(file=path, runs=runs, characters=chars, patterns=PATTERNS, hits=dict(hits), samples=samples)
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(runs=runs, characters=chars, hits=dict(hits)), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
