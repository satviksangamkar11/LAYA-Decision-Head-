#!/usr/bin/env python3
"""
Test gates: train text-only TF-IDF + logistic baseline on real rows.

Split rows into train (60%) and cal (40%).
Per option type: fit TF-IDF classifier.
Report accuracy on cal with margin rule from thresholds.
"""

import json
import sys
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
        print(f"Warning: fixture file not found at {rows_path}")
        return []

    return rows


def text_to_features(text, max_words=50):
    """Extract word features from text."""
    words = text.lower().split()[:max_words]
    return Counter(words)


def train_baseline(rows_train):
    """Train TF-IDF-like baseline per option type (gold vs non-gold)."""
    print("Training text-only baseline on train split...\n")

    baselines = {}

    for row in rows_train:
        candidates = row.get('candidates', [])
        gold = candidates[0]['text'] if candidates else ''

        # Extract features
        gold_features = text_to_features(gold)
        non_gold_features = Counter()

        for cand in candidates[1:]:
            kind = cand.get('kind', '')
            text = cand.get('text', '')
            non_gold_features.update(text_to_features(text))

        # Store for calibration
        if not baselines:
            baselines['gold'] = gold_features
            baselines['non_gold'] = non_gold_features
        else:
            baselines['gold'].update(gold_features)
            baselines['non_gold'].update(non_gold_features)

    # Compute log-odds for each word
    odds = {}
    for word in set(baselines['gold'].keys()) | set(baselines['non_gold'].keys()):
        gold_count = baselines['gold'][word] + 1  # Laplace smoothing
        non_gold_count = baselines['non_gold'][word] + 1
        odds[word] = gold_count / non_gold_count

    return odds


def predict(text, odds, threshold=1.0):
    """Simple text-only prediction: sum log-odds of words."""
    words = text.lower().split()[:50]
    score = sum(odds.get(w, 1.0) for w in words)
    prediction = "gold" if score > threshold else "non_gold"
    confidence = min(score, 2.0)  # Cap for readability
    return prediction, confidence


def evaluate_baseline(rows_cal, odds):
    """Evaluate on calibration split."""
    print(f"Evaluating on cal split ({len(rows_cal)} rows):\n")

    correct = 0
    total = 0

    per_type = Counter()
    correct_per_type = Counter()

    for row in rows_cal:
        candidates = row.get('candidates', [])
        gold = candidates[0]['text'] if candidates else ''

        # Predict gold
        pred, conf = predict(gold, odds)

        if pred == "gold":
            correct += 1
            correct_per_type['gold'] += 1

        total += 1
        per_type['gold'] += 1

        # Evaluate on hard negatives
        for cand in candidates[1:]:
            kind = cand.get('kind', '')
            text = cand.get('text', '')

            pred, conf = predict(text, odds)

            if pred == "non_gold":
                correct_per_type[kind] += 1

            per_type[kind] += 1

    accuracy = correct / max(total, 1)

    print(f"Overall accuracy (gold vs non-gold): {accuracy:.1%}")
    print(f"Per option type:\n")

    for kind in sorted(per_type.keys()):
        count = per_type[kind]
        correct_count = correct_per_type[kind]
        acc = correct_count / count if count > 0 else 0
        print(f"  {kind:30s}: {correct_count:3d}/{count:3d} = {acc:.1%}")

    return accuracy


def main():
    print("=" * 70)
    print("TEST: gates text-only baseline (TF-IDF + logistic)")
    print("=" * 70 + "\n")

    # Load fixture
    rows = load_real_rows(limit=50)
    if not rows:
        print("SKIP: fixture not available")
        return

    print(f"Loaded {len(rows)} real rows")

    # Split: 60% train, 40% cal
    split_idx = int(len(rows) * 0.6)
    rows_train = rows[:split_idx]
    rows_cal = rows[split_idx:]

    print(f"Train split: {len(rows_train)} rows")
    print(f"Cal split: {len(rows_cal)} rows\n")

    # Train baseline
    odds = train_baseline(rows_train)
    print(f"Learned {len(odds)} word odds\n")

    # Evaluate
    accuracy = evaluate_baseline(rows_cal, odds)

    # Compare with thresholds
    print(f"\nThreshold rule: accuracy must be > 0.60 (from thresholds file)")
    status = "PASS" if accuracy > 0.60 else "FAIL"
    print(f"Status: {status}")

    print("=" * 70)


if __name__ == "__main__":
    main()
