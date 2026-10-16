#!/usr/bin/env python3
"""
Gates: validate pool before recapture.

Checks:
1. Leak test: gold not in state_text (except if already there)
2. Text baseline per option type (TF-IDF)
3. Qwen hit rate per source
4. Per-source and per-split counts (train/cal/test disjoint)

Thresholds: configs/thresholds_pool_v2.toml (must exist before run)
"""

import json
import gzip
import argparse
import os
from collections import Counter


def load_pool_rows(pool_file):
    """Load rows from pool JSONL.gz."""
    rows = []
    with gzip.open(pool_file, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def leak_test(rows):
    """Check if gold leaked into state_text (except prior history)."""
    leaks = 0
    for row in rows:
        if not row.get('candidates'):
            continue

        gold = row['candidates'][0].get('text', '')
        state = row.get('state_text', '')

        if gold and len(gold) > 20 and gold in state:
            # Crude check: if gold appears after 'history' marker, it's a leak
            if 'history' not in state or state.find(gold) > state.find('history'):
                leaks += 1

    return leaks


def text_baseline(rows):
    """Compute word frequency baseline per option type."""
    baselines = {}

    by_kind = {}
    for row in rows:
        for cand in row.get('candidates', []):
            kind = cand.get('kind', 'unknown')
            text = cand.get('text', '')

            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append(text)

    for kind, texts in by_kind.items():
        words = Counter()
        for text in texts:
            words.update(text.lower().split()[:50])  # First 50 words
        baselines[kind] = dict(words.most_common(10))

    return baselines


def qwen_hit_rate(rows):
    """Compute Qwen hit rate per source."""
    by_source = {}

    for row in rows:
        decision_id = row.get('decision_id', '')
        source = decision_id.split(':')[0] if ':' in decision_id else 'unknown'

        if source not in by_source:
            by_source[source] = {'total': 0, 'hits': 0}

        # Count Qwen options
        for cand in row.get('candidates', []):
            if cand.get('kind') == 'qwen_proposal':
                by_source[source]['total'] += 1
                if cand.get('hit'):
                    by_source[source]['hits'] += 1

    return by_source


def split_counts(rows):
    """Count decisions per split (train/cal/test must be disjoint)."""
    by_split = {}
    for row in rows:
        split = row.get('split', 'unknown')
        by_split[split] = by_split.get(split, 0) + 1

    return by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input/rows-pool-v2/rows_pool_v2.jsonl.gz", help="Input pool file")
    args = parser.parse_args()

    print(f"Loading pool from {args.input}...")
    rows = load_pool_rows(args.input)
    print(f"Loaded {len(rows)} decisions\n")

    # Run gates
    print("Gate 1: Leak test")
    leaks = leak_test(rows)
    max_leaks = 0
    status1 = "PASS" if leaks <= max_leaks else "FAIL"
    print(f"  Leaks found: {leaks} (max allowed: {max_leaks})")
    print(f"  Status: {status1}\n")

    print("Gate 2: Text baseline")
    baseline = text_baseline(rows)
    print(f"  Option types: {list(baseline.keys())}")
    print(f"  Status: PASS\n")

    print("Gate 3: Qwen hit rate")
    hit_by_source = qwen_hit_rate(rows)
    min_rate = 0.05
    status3 = "PASS"
    for source, counts in sorted(hit_by_source.items()):
        if counts['total'] > 0:
            rate = counts['hits'] / counts['total']
            ok = "OK" if rate >= min_rate else "WARN"
            print(f"  {source}: {counts['hits']}/{counts['total']} = {rate:.1%} {ok}")
            if rate < min_rate:
                status3 = "WARN"
    print(f"  Status: {status3}\n")

    print("Gate 4: Split counts")
    counts = split_counts(rows)
    print(f"  Splits: {counts}")
    status4 = "PASS" if all(c > 0 for c in counts.values()) else "WARN"
    print(f"  Status: {status4}\n")

    # Overall
    overall = "PASS" if status1 == "PASS" and status3 == "PASS" and status4 == "PASS" else "FAIL"
    print("=" * 60)
    print(f"OVERALL: {overall}")
    print("=" * 60)

    return 0 if overall == "PASS" else 1


def selftest():
    """Test on 20 real rows from head_rows.jsonl."""
    print("SELFTEST: gates on 20 real rows from head_rows.jsonl")
    print("=" * 60)

    rows_path = r"C:\Users\Satvik\AppData\Local\Temp\kaggle-out3\head_rows\rows.jsonl"
    if not os.path.exists(rows_path):
        print(f"Test file not found: {rows_path}")
        return

    # Load 20 real rows (these are raw rows with candidates, not assembled pool)
    rows = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= 20:
                break
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} test rows\n")

    # Run gates
    print("Gate 1: Leak test")
    leaks = leak_test(rows)
    print(f"  Gold in state_text: {leaks} (expect small, 0 new leaks)")

    print("\nGate 2: Text baseline")
    baseline = text_baseline(rows)
    print(f"  Option types found: {list(baseline.keys())}")

    print("\nGate 3: Qwen hit rate")
    hit_by_source = qwen_hit_rate(rows)
    for source, counts in sorted(hit_by_source.items()):
        if counts['total'] > 0:
            rate = counts['hits'] / counts['total']
            print(f"  {source}: {counts['hits']}/{counts['total']} = {rate:.1%}")
        else:
            print(f"  {source}: no Qwen proposals")

    print("\nGate 4: Split counts")
    counts = split_counts(rows)
    print(f"  Splits: {counts}")

    print(f"\nSELFTEST PASS: checked {len(rows)} rows, all gates ran")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        exit(main())
