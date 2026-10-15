# Loop engineering v1, stage 2 (L2 order voting): not run, because it cannot change any score (2026-10-15)

Found before building the stage-2 kernel (no run, no result seen):
1. The decision head has no positional encoding (head.py: "permutation-equivariant by construction"); reordering rows changes nothing.
2. The Qwen capture used for these pools (results/raw/level12-bundle-20261012T104251Z/kernel_val/laya-val-capture-run.py, protocol of
   fable_capture_v1) feeds every option as an INDEPENDENT suffix after the context: hq = the last context token (before any option), ho = the mean
   over that option's own tokens, each option attending only to the context. The original run checked packed vs plain per-option capture
   (cosine >= 0.999).
Hence a re-capture in another option order yields the same hq and ho for every candidate, and the order-averaged probabilities equal the
single-pass probabilities: L2 is a no-op for this system by construction. The pre-registered L2 is recorded as NOT RUN (no information to gain);
the pre-registration file is unchanged.
