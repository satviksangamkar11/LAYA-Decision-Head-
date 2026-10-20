# Branch arch-v2-design-2026-10-20: COMPLETE (experimental record, 2026-10-20)

No further development on this branch. Not merged into main. Kept as the record of: the V2 design study (not adopted), P1 offline grading, trusted verifier v2/v3,
the P2 generation comparison (closed: Kaggle 13 tasks + local Python; local Java not repeated), and the WSLC infrastructure failure.

Decisions (user, 2026-10-20): close P2 with Kaggle's Java results (legacy runner: no local MXC/WSLC parity claim); keep the original architecture (V1.5); fresh resampling stays
the baseline; failure-aware resampling is not integrated; V2 migration not justified; next work P3 (runtime verification) on branch v1.5-p3-verification, cut from this branch's head
(it carries the verifier v3 fix and the grading tools).

Open infrastructure issue: MXC WSLC fails every call with `WslcListSessionImages ... HRESULT 0x80070057` since 2026-10-20 00:40 UTC (fresh store too; old store kept in
models/wslc-store-suspect/). Revisit when local Java validation is needed (likely fix: reboot / WSL service restart). Evidence: results/raw/p2-generation-compare-java-v3-* (CRASHED.txt).
