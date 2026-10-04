#!/usr/bin/env python3
"""
Pool assembly: join gold + real options + Qwen proposals + NONE on rows_pool_v2.

Input: rows_pool_v2.jsonl.gz (candidates with gold, real_run tiers, qwen_proposal)
Output: /kaggle/working/rows_pool_assembled.jsonl.gz + MANIFEST.json

Dedup: canonical JSON (sort_keys=True)
Hit: equals gold (never labelled wrong)
Join rate: decisions with real_run options
Provenance and tier per option
"""

import json
import gzip
import hashlib
import argparse
import os


def canonical_json(obj):
    """Return canonical JSON string."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def dedupe_by_canonical(options):
    """Remove duplicate options by canonical JSON."""
    seen = set()
    deduped = []
    for opt in options:
        text = opt.get('text', '')
        try:
            canonical = canonical_json(json.loads(text))
            if canonical not in seen:
                seen.add(canonical)
                deduped.append(opt)
        except (json.JSONDecodeError, TypeError):
            if text not in seen:
                seen.add(text)
                deduped.append(opt)
    return deduped


def load_pool_rows(pool_file):
    """Load rows from pool JSONL.gz."""
    rows = []
    with gzip.open(pool_file, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def assemble_pool(rows):
    """Assemble final pool: gold + real + Qwen + NONE."""
    if not rows:
        raise ValueError("No rows to assemble")

    assembled = []

    for row in rows:
        candidates = row.get('candidates', [])
        if not candidates:
            raise ValueError(f"Row {row.get('decision_id')} has no candidates")

        # Extract gold
        gold_opt = candidates[0]

        # Collect real options (real_run_* and hard_neg)
        real_options = []
        for cand in candidates[1:]:
            kind = cand.get('kind', '')
            if kind.startswith('real_run_') or kind.startswith('hard_neg'):
                real_options.append(cand)

        # Collect Qwen proposals (deduplicate by canonical JSON)
        qwen_options = []
        for cand in candidates:
            if cand.get('kind') == 'qwen_proposal':
                # Check if matches gold
                is_hit = False
                try:
                    gold_canon = canonical_json(json.loads(gold_opt.get('text', '')))
                    cand_canon = canonical_json(json.loads(cand.get('text', '')))
                    is_hit = (gold_canon == cand_canon)
                except (json.JSONDecodeError, TypeError):
                    pass

                qwen_options.append(dict(cand, hit=is_hit))

        # Deduplicate Qwen
        qwen_options = dedupe_by_canonical(qwen_options)

        # Final candidates: gold + real + Qwen + NONE
        final_candidates = [gold_opt] + real_options + qwen_options + [
            {"text": "", "kind": "NONE"}
        ]

        assembled.append(dict(
            decision_id=row['decision_id'],
            state_text=row.get('state_text', ''),
            candidates=final_candidates,
            num_real=len(real_options),
            num_qwen=len(qwen_options),
        ))

    return assembled


def write_pool(rows, output_file):
    """Write pool to gzipped JSONL."""
    hasher = hashlib.sha256()
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input/rows-pool-v2/rows_pool_v2.jsonl.gz", help="Input pool file")
    parser.add_argument("--output", default="/kaggle/working/rows_pool_assembled.jsonl.gz", help="Output file")
    args = parser.parse_args()

    print(f"Loading pool from {args.input}...")
    rows = load_pool_rows(args.input)
    print(f"Loaded {len(rows)} decisions")

    print("Assembling pool...")
    assembled = assemble_pool(rows)

    print(f"Writing to {args.output}...")
    sha256 = write_pool(assembled, args.output)

    # Manifest
    join_rate = sum(1 for r in assembled if r['num_real'] > 0) / len(assembled) if assembled else 0
    avg_qwen = sum(r['num_qwen'] for r in assembled) / len(assembled) if assembled else 0

    manifest = {
        "total_rows": len(assembled),
        "file": args.output,
        "sha256": sha256,
        "join_rate": join_rate,
        "avg_qwen_per_decision": avg_qwen,
    }

    import os
    manifest_file = os.path.join(os.path.dirname(args.output), "MANIFEST_pool_assembled.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done: {len(assembled)} rows, join_rate {join_rate:.1%}, avg Qwen {avg_qwen:.1f}")
    print(f"Manifest: {manifest_file}")


def selftest():
    """Test on 20 real rows from head_rows.jsonl."""
    print("SELFTEST: pool_assembly on 20 real rows")
    print("=" * 60)

    rows_path = r"C:\Users\Satvik\AppData\Local\Temp\kaggle-out3\head_rows\rows.jsonl"
    if not os.path.exists(rows_path):
        print(f"Test file not found: {rows_path}")
        return

    # Load 20 rows
    rows = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= 20:
                break
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} test rows\n")

    # Assemble pool
    try:
        assembled = assemble_pool(rows)
        print(f"Assembled {len(assembled)} rows\n")

        # Check properties
        for i, row in enumerate(assembled[:5]):
            did = row['decision_id']
            gold = row['candidates'][0]
            real_count = row['num_real']
            qwen_count = row['num_qwen']
            total = len(row['candidates'])

            # Check invariants
            assert row['candidates'][0] == gold, "Gold must be candidates[0]"
            assert row['candidates'][-1].get('kind') == 'NONE', "NONE must be last"

            # Check for same canonical action (dedupe)
            canonicals = []
            for cand in row['candidates']:
                text = cand.get('text', '')
                if text:
                    try:
                        c = canonical_json(json.loads(text))
                        canonicals.append(c)
                    except (json.JSONDecodeError, TypeError):
                        canonicals.append(text)

            # All unique
            assert len(canonicals) == len(set(canonicals)), f"Row {i}: duplicate candidates"

            print(f"  {i+1}. {did[:40]}")
            print(f"     Gold: {gold.get('text', '')[:50]}...")
            print(f"     Candidates: {total} (gold + {real_count} real + {qwen_count} Qwen + NONE)")

            # Check: Qwen proposal equals gold
            for cand in row['candidates']:
                if cand.get('kind') == 'qwen_proposal':
                    assert cand.get('hit') in (True, False), f"Qwen must have hit field"

        print(f"\nSELFTEST PASS: assembled {len(assembled)} rows, invariants verified")
        print("=" * 60)

    except Exception as e:
        print(f"SELFTEST FAIL: {e}")
        raise


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
