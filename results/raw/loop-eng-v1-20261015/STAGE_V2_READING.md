# Loop engineering v2 (A-C, bounded): result (2026-10-15)

Kernel `sangeetasangamkar/laya-loop-eng-run-v2` v1 (CPU), pre-registration `configs/thresholds_loop_eng_v2.toml` (sha256 55be94eb...).
Reproduction gate passed (0.7826 / 0.8043 / 0.7609). 46 cleaned VAL pools, 32 trajectories. Probe recomputed on the same 46: 0.8913 (41/46);
the 0.943 figure belongs to the 53 raw pools. Options-only order verified identical to pack5 (labels and option counts equal for all 46).
Exploratory (VAL was examined before); N = 46, so one pool = 2.2 points.

## 1. Oracle ceiling over the probe (judged first)
- Probe errors (5): go-switchcase:1, go-switchcase:3, java-canneryline:1, py-alert-rules:1, ts-k8s-ssa-configmap:1.
- LAYA ensemble correct where the probe is wrong: **0**. Probe correct where the LAYA ensemble is wrong: 4. oracle({laya_ens, probe}) = 41 = the probe alone.
  So no choice between the two argmax picks can beat the probe (ceiling 89.1%).
- Individual seeds are complementary on 2 probe errors (java-canneryline:1 fixed by seed 1; ts-k8s-ssa-configmap:1 by seeds 0 and 2): ceiling with
  {3 seeds, ensemble, probe} = 43/46 (93.5%); the ensemble average drops both. Two probe errors are never fixed by any scorer.

## 2. Net wrong->right minus right->wrong (all 46 pools)
| rule | correct | vs LAYA ens | vs probe (CI by trajectory) |
|---|---|---|---|
| probe alone | 41 | +4 | 0 |
| C_avg (0.5/0.5 probabilities) | **42** | +5 / -0 | +1 / -0, net +1 [0.000, 0.067]; exact McNemar p = 1.0 |
| C_laya_first t=0.30 | 41 | +4 / -0 | 0 |
| C_laya_first t=0.10 / 0.20 | 38 | +1 / -0 | -3 |
| C_probe_first t=0.10 / 0.20 / 0.30 | 40 | +3 / -0 | -1 |
| S_unanimous_else_probe | 40 | +3 / -0 | -1 |
| S_vote | 36 | +1 / -2 | -5 [-0.222, -0.021] |
The single gain over the probe (C_avg) is py-alert-rules:1: all three LAYA seeds pick option 2, the probe picks 1, the average picks option 0, which
is the recorded action: neither scorer's argmax was right (that is why 42 > oracle 41).

## 3. Controls through the same rules
- C_avg with the NO-CONTEXT LAYA in LAYA's place: also 42, also +1 vs the probe: the same gain without context.
- C_avg with options-only in LAYA's place: 43, +2 vs the probe (CI [0.000, 0.102]).
- Both replaced by no-context versions: 37.
Pre-registered reading: improvement only if the CI of the net gain vs the probe is above 0 AND exceeds the control's gain. **Not met** (CI lower bound
0.000; controls match or exceed the gain).

## 4. Context evidence
Removing context costs LAYA ens 3 pools (37 -> 34) and the probe 5 pools (41 -> 36), so both read some context; but the combination gain does not
depend on it (control rows above).

## Reading
A-C does not show that this loop improves decisions: one pool, no statistical support, reproduced by controls that do not use context. The
decide-or-defer cascades never beat the probe (best equals it). Headroom exists only in individual seeds (43/46), and exploiting it needs a
per-example selector that cannot be tuned or validated on 46 pools without a larger development set.
