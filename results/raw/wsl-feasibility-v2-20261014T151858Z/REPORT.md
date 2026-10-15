# WSL feasibility check v2 (2026-10-14): 1 of 2 QUALIFIED

Rule: `SELECTION_v2.md` (written before any v2 fetch, draw or run). Tools: `tools/wsl_feasibility_select_v2.py`, `tools/wsl_feasibility_run_v2.py`.
v1 folder untouched.

## Selection
Same universe as v1 (1,741 mixed; repo exclusions RootSE 39, CodeTraceBench 2, LAYA 862; 838 eligible; seed 20261014). 15 ids walked
(13 reused from v1, 2 fetched one row each): v1 checks rejected 10 (Docker/system packages via apt pre_install 6, other 4), X1 heavy/over-80
rejected 1 (`iterative__datachain-206`, nvidia/CUDA wheels), X2 network prefix rejected 0. Tasks: `googleapis__python-storage-1357`,
`pvlib__pvanalytics-128`; reserves `stephenhillier__starlette_exporter-89`, `dwavesystems__dwave-system-373`.

## Results
| task | setup | initial tests x2 (net off) | prefix steps matched (A/B/C) | hash B==C | verdict |
|---|---|---|---|---|---|
| googleapis__python-storage-1357 | OK, 513 s | F2P 1/1 fail, P2P 20/20 pass, both runs | 5/6 each | equal (A too) | NOT QUALIFIED |
| pvlib__pvanalytics-128 | impossible: no uv build of exact Python 3.7.16 | - | - | - | replaced by reserve 1 |
| stephenhillier__starlette_exporter-89 | OK, 202 s | F2P 1/1 fail, P2P 20/20 pass, both runs | 4/4 each (all exact) | equal (A too) | QUALIFIED |

googleapis reason: step 6 `ls -la tests/` differs only in owner/group (`root root` recorded vs `replay systemd-journal`) and mtime (recorded
image build time vs clone time); file names, modes, sizes and link counts are equal. Under compare_v2 this is a MISMATCH and the rule was not
changed. Reproducing it would need a new versioned rule (e.g. replay as in-namespace uid/gid 0 and file mtimes set to the recorded time, or a
metadata-insensitive `ls` comparison). User decision.

Network: setup only; the socket + DNS probe failed inside the sandbox for both executed tasks. Disk guard: C: 8.40 GiB free before (PowerShell),
lowest runner reading 8.30 GiB, 8.35 GiB after; no abort; task dir peak 0.11 GiB. Cleanup: each task's `/var/lib/fable-runs/feas-v2/<id>` and
`runs/*` removed, `fstrim -av` after each (`CLEANUP.log`). Code fixes: none needed (`FIXES.md`).
No GLM/NVIDIA calls, no candidates, no continuations, no training, no Kaggle.
