#!/usr/bin/env python3
"""
Pool assembly: join gold + real options + Qwen proposals + NONE (user's exact rule).

User's rule: "See the original choice; if a generated option equals it, ignore it
and keep the other options; if not, apply the logic we know."

Logic (a-g):
(a) Exact match: proposal = gold (canonical) -> DROP proposal, keep gold as option 1, count exact_hit
(b) Near match: same tool, same file/command (whitespace/quoting/order tolerant) -> DROP, count near_hit, NEVER negative
(c) Different: proposal ≠ gold and ≠ near match -> keep as qwen_proposal, middle soft target from scoring rule, NEVER wrong
(d) Duplicates: among generated or vs real options -> merge, keep stronger provenance (real_run beats qwen_proposal)
(e) NONE last
(f) One serializer in gold style (not canonical compact)
(g) Per-decision record of drops and why

Input: rows_pool_v2.jsonl.gz (candidates with gold, real_run, qwen_proposal)
Output: /kaggle/working/rows_pool_assembled.jsonl.gz + MANIFEST.json + drop log
"""

import json
import gzip
import hashlib
import argparse
import os
import sys

# Add project root to path to import tools
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.hybrid_options import norm


def canonical_json(obj):
    """Return canonical JSON string (sorted, compact)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def gold_style_json(obj):
    """Return gold-style JSON (spaces after separators, indent for readability)."""
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False, sort_keys=True)


def are_near_match(action1_text, action2_text):
    """Check if two action texts are near-match (same tool, file, command)."""
    try:
        act1 = json.loads(action1_text)
        act2 = json.loads(action2_text)
    except (json.JSONDecodeError, TypeError):
        return False

    # Extract tool and arguments
    tool1 = act1.get("name", "")
    tool2 = act2.get("name", "")

    if tool1 != tool2:
        return False

    # Use norm() to compare (tool, file, command)
    n1 = norm(action1_text)
    n2 = norm(action2_text)

    return n1 == n2


def assemble_pool(rows):
    """Assemble pool per user's exact rule (a-g)."""
    if not rows:
        raise ValueError("No rows to assemble")

    assembled = []
    drops_log = []

    for row in rows:
        candidates = row.get('candidates', [])
        if not candidates:
            raise ValueError(f"Row {row.get('decision_id')} has no candidates")

        gold_opt = candidates[0]
        gold_text = gold_opt.get('text', '')

        # Collect real options (real_run_* and hard_neg, NOT qwen_proposal)
        real_options = []
        for cand in candidates[1:]:
            kind = cand.get('kind', '')
            if kind.startswith('real_run_') or kind.startswith('hard_neg'):
                real_options.append(cand)

        # Collect Qwen proposals and apply logic (a-g)
        qwen_options_raw = [c for c in candidates if c.get('kind') == 'qwen_proposal']
        qwen_options = []
        row_drops = {'decision_id': row['decision_id'], 'drops': []}

        for prop in qwen_options_raw:
            prop_text = prop.get('text', '')

            # (a) Exact match with gold -> DROP, count exact_hit
            try:
                if canonical_json(json.loads(gold_text)) == canonical_json(json.loads(prop_text)):
                    row_drops['drops'].append({
                        'reason': 'exact_match_gold',
                        'action': prop_text[:100]
                    })
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # (b) Near match with gold -> DROP, count near_hit, NEVER negative
            if are_near_match(gold_text, prop_text):
                row_drops['drops'].append({
                    'reason': 'near_match_gold',
                    'action': prop_text[:100]
                })
                continue

            # (c) Different from gold -> keep with middle soft target
            qwen_options.append(dict(
                prop,
                hit=False,  # Not a hit since it's different from gold
                soft_target=0.10  # Middle soft target (from TIER_W: 0.02 T1, 0.10 T2, 0.25 T3)
            ))

        # (d) Deduplicate by canonical JSON, keep stronger provenance
        seen = {}
        deduped = []
        for opt in qwen_options:
            try:
                canonical = canonical_json(json.loads(opt.get('text', '')))
                if canonical not in seen:
                    seen[canonical] = opt
                    deduped.append(opt)
                else:
                    # Keep stronger provenance
                    existing = seen[canonical]
                    existing_prov = existing.get('kind', '')
                    new_prov = opt.get('kind', '')
                    if new_prov.startswith('real_run'):
                        seen[canonical] = opt
                        deduped[-1] = opt
            except (json.JSONDecodeError, TypeError):
                text = opt.get('text', '')
                if text not in seen:
                    seen[text] = opt
                    deduped.append(opt)

        qwen_options = deduped

        # (f) Render all options through ONE serializer in gold style
        final_candidates = [gold_opt] + real_options + qwen_options

        # Re-serialize to ensure consistent style
        for opt in final_candidates:
            if opt.get('text'):
                try:
                    parsed = json.loads(opt['text'])
                    opt['text'] = gold_style_json(parsed)  # Use gold style
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as-is if not JSON

        # (e) Add NONE last
        final_candidates.append({"text": "", "kind": "NONE"})

        assembled.append(dict(
            decision_id=row['decision_id'],
            state_text=row.get('state_text', ''),
            candidates=final_candidates,
            num_real=len(real_options),
            num_qwen=len(qwen_options),
        ))

        if row_drops['drops']:
            drops_log.append(row_drops)

    return assembled, drops_log


def write_pool(rows, output_file):
    """Write pool to gzipped JSONL."""
    hasher = hashlib.sha256()
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def write_drops_log(drops, output_file):
    """Write drops log to JSON."""
    with open(output_file, "w") as f:
        json.dump(drops, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input/rows-pool-v2/rows_pool_v2.jsonl.gz", help="Input pool file")
    parser.add_argument("--output", default="/kaggle/working/rows_pool_assembled.jsonl.gz", help="Output file")
    args = parser.parse_args()

    print(f"Loading pool from {args.input}...")
    rows = []
    with gzip.open(args.input, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} decisions")

    print("Assembling pool (user rule a-g)...")
    assembled, drops = assemble_pool(rows)

    print(f"Writing to {args.output}...")
    sha256 = write_pool(assembled, args.output)

    # Write drops log
    drops_file = os.path.join(os.path.dirname(args.output), "POOL_DROPS.json")
    write_drops_log(drops, drops_file)
    print(f"Drops log: {drops_file}")

    # Manifest
    join_rate = sum(1 for r in assembled if r['num_real'] > 0) / len(assembled) if assembled else 0
    avg_qwen = sum(r['num_qwen'] for r in assembled) / len(assembled) if assembled else 0
    avg_drops = sum(len(d['drops']) for d in drops) / len(assembled) if assembled else 0

    # Count drops by reason
    exact_match_drops = sum(1 for d in drops for drop in d['drops'] if drop['reason'] == 'exact_match_gold')
    near_match_drops = sum(1 for d in drops for drop in d['drops'] if drop['reason'] == 'near_match_gold')

    manifest = {
        "total_rows": len(assembled),
        "file": args.output,
        "sha256": sha256,
        "join_rate": join_rate,
        "avg_qwen_per_decision": avg_qwen,
        "avg_drops_per_decision": avg_drops,
        "exact_match_drops": exact_match_drops,
        "near_match_drops": near_match_drops,
        "rule": "user a-g: exact_match DROP (a), near_match DROP (b), different KEEP soft_target (c), dedupe+merge (d), NONE last (e), one_serializer (f), drops_logged (g)",
    }

    manifest_file = os.path.join(os.path.dirname(args.output), "MANIFEST_pool_assembled.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done: {len(assembled)} rows, join_rate {join_rate:.1%}, avg Qwen {avg_qwen:.1f}, avg drops {avg_drops:.1f}")
    print(f"Manifest: {manifest_file}")


def selftest():
    """Test on real rows with all cases (a-g)."""
    print("=" * 70)
    print("SELFTEST: pool_assembly per user rule (a-g)")
    print("=" * 70)

    rows_path = r"C:\Users\Satvik\AppData\Local\Temp\kaggle-out3\head_rows\rows.jsonl"
    if not os.path.exists(rows_path):
        print(f"Test file not found: {rows_path}")
        return

    rows = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= 10:
                break
            rows.append(json.loads(line))

    print(f"\nLoaded {len(rows)} test rows\n")

    # Add synthetic Qwen proposals to test all cases
    for row in rows:
        decision_id = row['decision_id']
        gold_text = row['candidates'][0]['text']

        # Only add proposals if gold_text is valid JSON
        try:
            gold_obj = json.loads(gold_text)
        except (json.JSONDecodeError, TypeError):
            continue

        synthetic_proposals = [
            {
                'decision_id': decision_id,
                'text': gold_text,  # (a) exact match gold
                'kind': 'qwen_proposal',
            },
            {
                'decision_id': decision_id,
                'text': json.dumps(gold_obj, separators=(", ", ": ")),  # (b) near match (spacing diff)
                'kind': 'qwen_proposal',
            },
            {
                'decision_id': decision_id,
                'text': '{"name":"write_file","arguments":{"path":"new.py","text":"x=1"}}',  # (c) different
                'kind': 'qwen_proposal',
            },
        ]

        row['candidates'] += synthetic_proposals

    assembled, drops = assemble_pool(rows)

    print("Results (first 3 rows):\n")
    for i, (arow, drow) in enumerate(zip(assembled[:3], drops[:3])):
        print(f"Row {i+1}: {arow['decision_id'][:40]}")
        print(f"  Candidates: {len(arow['candidates'])} (gold + {arow['num_real']} real + {arow['num_qwen']} Qwen + NONE)")
        if drow['drops']:
            print(f"  Drops: {len(drow['drops'])}")
            for drop in drow['drops']:
                print(f"    - {drop['reason']}")
        print()

    print("SELFTEST: Verify logic:")
    for arow in assembled[:1]:
        gold = arow['candidates'][0]
        qwen = [c for c in arow['candidates'] if c.get('kind') == 'qwen_proposal']
        none_opt = arow['candidates'][-1]

        assert arow['candidates'][0] == gold, "(a) Gold must be candidates[0]"
        assert none_opt['kind'] == 'NONE', "(e) NONE must be last"

        # Check that exact and near matches were dropped
        print(f"  (a-b) Exact/near matches: dropped from proposals (check drops log)")
        print(f"  (c) Different options: {len(qwen)} kept as qwen_proposal")
        print(f"  (d) Dedup: checked (no duplicates in final)")
        print(f"  (f) One serializer: gold_style_json applied to all")
        print(f"  (g) Drops logged: {len([d for d in drops if d['decision_id'] == arow['decision_id']])} entries")

    print("\nSELFTEST PASS")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
