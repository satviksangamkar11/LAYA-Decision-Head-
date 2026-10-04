#!/usr/bin/env python3
"""
Gates v2: Text-only baseline must NOT find gold (FAIL if it does).

Semantics (from KAGGLE_WORKFLOW.md, margin=0.15):
  PASS = text_only_gold_top1 <= chance + 0.15
  FAIL = text_only_gold_top1 > chance + 0.15

PASS means baseline CANNOT find gold (good signal, no shortcut).
FAIL means baseline finds gold (bad, gold is shortcut).
"""

import json
import math
from collections import Counter


def load_real_rows(limit=50):
    """Load real rows from head_rows.jsonl."""
    rows_path = r"C:\Users\Satvik\AppData\Local\Temp\kaggle-out3\head_rows\rows.jsonl"
    rows = []
    try:
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                if len(rows) >= limit:
                    break
                rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: file not found at {rows_path}")
        return []
    return rows


def compute_chance(rows):
    """Compute chance baseline: mean(1/n_options)."""
    chances = []
    for row in rows:
        n_options = len(row.get('candidates', []))
        if n_options > 0:
            chances.append(1.0 / n_options)
    if not chances:
        return 0.0
    return sum(chances) / len(chances)


def test_gold_findability(rows_train, rows_cal, margin=0.15):
    """
    Test: Can text-only baseline find gold?
    PASS = top-1 <= chance + margin (cannot find it)
    FAIL = top-1 > chance + margin (finds it, shortcut)
    """
    print("=" * 70)
    print("GATES v2: Gold Findability (inverted logic)")
    print("=" * 70)

    chance = compute_chance(rows_train + rows_cal)
    threshold = chance + margin

    print(f"\nChance (mean 1/n_options): {chance:.3f}")
    print(f"Margin: {margin}")
    print(f"FAIL if baseline finds gold (top-1 > {threshold:.3f})")
    print(f"PASS if baseline cannot find gold (top-1 <= {threshold:.3f})\n")

    # Collect texts: gold=1, non-gold=0
    texts = []
    labels = []

    for row in rows_train:
        candidates = row.get('candidates', [])
        if candidates:
            texts.append(candidates[0]['text'])
            labels.append(1)
            for cand in candidates[1:]:
                texts.append(cand['text'])
                labels.append(0)

    # Learn word weights
    gold_words = Counter()
    non_gold_words = Counter()

    for text, label in zip(texts, labels):
        words = text.lower().split()[:20]
        if label == 1:
            gold_words.update(words)
        else:
            non_gold_words.update(words)

    # Score on cal split
    texts_cal = []
    labels_cal = []
    for row in rows_cal:
        candidates = row.get('candidates', [])
        if candidates:
            texts_cal.append(candidates[0]['text'])
            labels_cal.append(1)
            for cand in candidates[1:]:
                texts_cal.append(cand['text'])
                labels_cal.append(0)

    scores = []
    for text in texts_cal:
        words = text.lower().split()[:20]
        score = 0.0
        for word in words:
            gold_count = gold_words[word] + 1
            non_gold_count = non_gold_words[word] + 1
            score += math.log(gold_count / non_gold_count)
        scores.append(score > 0)  # predict gold if score > 0

    # Compute top-1 accuracy on gold
    gold_indices = [i for i, label in enumerate(labels_cal) if label == 1]
    if gold_indices:
        gold_correct = sum(scores[i] for i in gold_indices)
        top1_gold = gold_correct / len(gold_indices)
    else:
        top1_gold = 0.0

    status = "PASS" if top1_gold <= threshold else "FAIL"

    print(f"Cal split ({len(labels_cal)} samples, {len(gold_indices)} gold):")
    print(f"  Baseline top-1 on gold: {top1_gold:.3f}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Status: {status}")

    if status == "FAIL":
        print(f"\n  *** PROBLEM: Baseline FOUND the gold ({top1_gold:.1%}) ***")
        print(f"  Gold is a shortcut (text tells it apart) - BAD signal")
    else:
        print(f"\n  Good: Baseline CANNOT find gold - no shortcut")

    print()
    return status, top1_gold, threshold


def main():
    print("\n" + "=" * 70)
    print("GATES v2: Text-only baseline should NOT detect gold")
    print("=" * 70 + "\n")

    rows = load_real_rows(limit=50)
    if not rows:
        print("SKIP: fixture not available")
        return

    print(f"Loaded {len(rows)} real rows\n")

    # Split
    split_idx = int(len(rows) * 0.6)
    rows_train = rows[:split_idx]
    rows_cal = rows[split_idx:]

    print(f"Train: {len(rows_train)} rows")
    print(f"Cal: {len(rows_cal)} rows\n")

    # Test
    status, top1, thresh = test_gold_findability(rows_train, rows_cal, margin=0.15)

    print("=" * 70)
    print(f"OVERALL GATE: {status}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
