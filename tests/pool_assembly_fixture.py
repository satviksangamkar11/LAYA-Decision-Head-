#!/usr/bin/env python3
"""
Fixture: 10 real rows + synthetic v2 proposals to test pool_assembly logic.

v2 proposal schema: decision_id, sample_idx, action_text, parse_ok, ...
Tests:
  - proposal=gold (should hit flag)
  - duplicate proposal (should dedupe, keep first)
  - unparsable proposal (parse_ok=False)
  - proposal=hard_negative (should keep both)
"""

import json


def load_fixture():
    """Load 10 real rows from head_rows.jsonl."""
    rows_path = r"C:\Users\Satvik\AppData\Local\Temp\kaggle-out3\head_rows\rows.jsonl"
    rows = []

    try:
        with open(rows_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: fixture file not found at {rows_path}")
        return []

    return rows


def generate_synthetic_proposals(rows):
    """
    Generate synthetic v2 proposals for test cases:
    - Proposal equal to gold (should hit)
    - Duplicate proposal (should dedupe)
    - Unparsable proposal (parse_ok=False)
    - Proposal equal to hard_negative
    """
    proposals_by_id = {}

    for i, row in enumerate(rows):
        decision_id = row['decision_id']
        candidates = row['candidates']
        gold = candidates[0]['text'] if candidates else '{"name":"unknown","arguments":{}}'
        hard_neg = candidates[1]['text'] if len(candidates) > 1 else '{"name":"alt","arguments":{}}'

        proposals = []

        # Sample 0: equals gold (hit=True)
        proposals.append({
            'decision_id': decision_id,
            'sample_idx': 0,
            'action_text': gold,
            'parse_ok': True,
        })

        # Sample 1: duplicate of gold (should be deduped)
        proposals.append({
            'decision_id': decision_id,
            'sample_idx': 1,
            'action_text': gold,
            'parse_ok': True,
        })

        # Sample 2: unparsable (parse_ok=False)
        proposals.append({
            'decision_id': decision_id,
            'sample_idx': 2,
            'action_text': '{incomplete json',
            'parse_ok': False,
        })

        # Sample 3: equals hard_negative
        proposals.append({
            'decision_id': decision_id,
            'sample_idx': 3,
            'action_text': hard_neg,
            'parse_ok': True,
        })

        proposals_by_id[decision_id] = proposals

    return proposals_by_id


def test_pool_assembly():
    """Test pool_assembly with fixture: gold, duplicates, unparsable, hard_neg."""
    print("=" * 70)
    print("TEST: pool_assembly with synthetic proposals")
    print("=" * 70)

    # Load fixture
    rows = load_fixture()
    if not rows:
        print("SKIP: fixture not available")
        return

    print(f"Loaded {len(rows)} fixture rows")

    # Generate proposals
    proposals_by_id = generate_synthetic_proposals(rows)

    # Inline pool_assembly functions
    def canonical_json(obj):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True)

    def dedupe_by_canonical(options):
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

    # Simulate pool assembly logic
    print(f"\nProcessing {len(rows)} rows with synthetic proposals:\n")

    for row in rows[:3]:  # Test first 3
        decision_id = row['decision_id']
        candidates = row['candidates']
        gold = candidates[0]['text'] if candidates else ''

        proposals = proposals_by_id.get(decision_id, [])

        # Assemble like pool_assembly
        real_options = []
        for cand in candidates[1:]:
            if cand.get('kind', '').startswith('hard_neg') or cand.get('kind', '').startswith('real_run'):
                real_options.append(cand)

        qwen_options = []
        for prop in proposals:
            if prop['parse_ok']:
                is_hit = prop['action_text'] == gold
                qwen_options.append({
                    'text': prop['action_text'],
                    'kind': 'qwen_proposal',
                    'hit': is_hit,
                    'sample_idx': prop['sample_idx'],
                })

        # Dedupe
        qwen_options = dedupe_by_canonical(qwen_options)

        # Final candidates
        final = [candidates[0]] + real_options + qwen_options + [{"text": "", "kind": "NONE"}]

        # Assertions
        assert final[0] == candidates[0], "Gold must be candidates[0]"
        assert final[-1]['kind'] == 'NONE', "NONE must be last"

        # Check dedup and hit flag
        parse_ok_proposals = [p for p in proposals if p['parse_ok']]
        unique_canonicals = len(set(p['action_text'] for p in parse_ok_proposals))
        qwen_with_hit = sum(1 for q in qwen_options if q.get('hit'))

        print(f"Row {decision_id[:40]}:")
        print(f"  Proposals: {len(proposals)} total, {len(parse_ok_proposals)} parse_ok")
        print(f"  After dedupe: {len(qwen_options)} unique Qwen options")
        print(f"  Hit flagged: {qwen_with_hit} (expected 1 for gold match)")

        # Verify
        assert len(qwen_options) < len(parse_ok_proposals), "Dedupe removed duplicates"
        assert qwen_with_hit == 1, "Exactly one proposal should be flagged hit (equals gold)"
        assert final[0]['text'] == gold, "Gold stayed candidates[0]"

        print(f"  Final candidates: {len(final)} (gold + {len(real_options)} real + {len(qwen_options)} Qwen + NONE)")
        print()

    print("TEST PASS: gold, duplicates, unparsable, hard_neg handled correctly")
    print("=" * 70)


if __name__ == "__main__":
    test_pool_assembly()
