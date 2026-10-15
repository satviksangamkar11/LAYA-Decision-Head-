# Sandbox pilot v2 (late decision points, narrowed to one task): HOLD (2026-10-15)

Saved by the main session from the agent's final report (the agent could not write REPORT.md).

- Pre-registration: `configs/thresholds_sandbox_pilot_v2.toml` (user narrowing folded in before any execution), `configs/fidelity_v4.toml`
  (v3 + masking of test-duration tokens on test-runner commands), `SANDBOX_LABEL_PILOT_CONTRACT_v2.md`. Code: `tools/sandbox_pilot_prep_v2.py`,
  `tools/sandbox_pilot_v2.py`.
- Qualified late states: 0 of 2. NVIDIA calls: 0. Candidates, rollouts, labels: none.

| task | actions / decision position | initial tests (x2) | prefix replay | B/C hash | oracle |
|---|---|---|---|---|---|
| starlette_exporter-89 | 71 / 61 | ok | fails at step 5 of 59 (A, B, C identical) | equal | not run |
| python-storage-1357 | 86 / 76 | ok | fails at step 13 of 69 (A, B, C identical) | equal | not run |

Causes (not repository-state differences):
- starlette: the recorded run used a 1,000-column tty (pytest separator lines exactly 1,000 characters; `collected` vs our `collecting ... collected`);
  our sandbox prints 80 columns. Fidelity v3/v4 keep those lines strict. Every late prefix contains several pytest runs (14 and 9).
- python-storage: OpenHands clipped a long editor `view` at 18,850 characters with `<response clipped><NOTE>...`; our emulation does not clip.
- Not measured: whether the tests already pass at the decision point; both runs' last 10 actions look like post-fix verification and cleanup,
  so qualified states would likely give "both 3/3" labels.
- v1's slot-T failure was a wrong-type answer (GLM returned a read action twice), not unparsable output.

Possible continuation (needs the user's approval; post hoc, must be flagged): `configs/fidelity_v5.toml` with environment parity
(COLUMNS=1000 under a pseudo-terminal; emulate OpenHands view clipping from the recorded observation), then re-qualify in a new results folder.
Cleanup done (v2 state copies removed, fstrim; v1 setup environments kept). Secret scan: no hits. No Kaggle, no training, no commit.
