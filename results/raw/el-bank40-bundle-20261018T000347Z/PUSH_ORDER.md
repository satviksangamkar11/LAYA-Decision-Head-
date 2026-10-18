# EL-BANK40 PUSH ORDER (exact; the coordinator runs these, this driver runs none)

thinking policy in the bundle: `{"policy": "prompt_brief", "brief_instruction": "Think BRIEFLY: reason in a few short sentences only, do not re-verify the same point, and give the final answer as soon as you have it.", "ceiling": 8192}`; commit: 90cc3e528bd4b9e64c84273d14a079eba4bce05f

1. Create/refresh the bundle dataset (private): `kaggle datasets create -p "D:\local model\results\raw\el-bank40-bundle-20261018T000347Z\dataset"` (first time) or `kaggle datasets version -p "D:\local model\results\raw\el-bank40-bundle-20261018T000347Z\dataset" -m bank40 bundle`
2. Push the GPU kernel (T4 x2, internet off): `kaggle kernels push -p "D:\local model\results\raw\el-bank40-bundle-20261018T000347Z\kernel_pass1"`; if your CLI supports it add `--accelerator NvidiaTeslaT4`; check on the Kaggle page that the session shows 2 GPUs.
3. When COMPLETE: `kaggle kernels output sangeetasangamkar/el-bank40-run-pass1-v1 -p <pass1_out_dir>`
4. If RUN_SUMMARY.json says PARTIAL_TIME_GUARD / PARTIAL: `python tools/el_bank40_driver.py pass1-datasets <pass1_out_dir> <ds_dir>`, upload `<ds_dir>/pass1_out` as dataset `el-bank40-pass1-out-v1`, add it to `dataset_sources` of the pass-1 kernel (resume) and push again; finished tasks (DONE) are skipped. Repeat until done.
5. Critic: `python tools/el_bank40_driver.py pass1-datasets <pass1_out_dir> <ds_dir>` then `python tools/pack_evidence_ladder_critic_batch_v1.py <ds_dir>/critic_packets`; upload that dataset and push the EXISTING critic kernel (kaggle/evidence_ladder_critic_v2, internet on, GPU off, Secret NVIDIA_API_KEYS added by the user). Download its output.
6. `python tools/el_bank40_driver.py critic-dataset <critic_output_dir> <ds_dir2>`; upload `<ds_dir2>/critic_responses` as dataset `el-bank40-critic-responses-v1` and `<ds_dir>/pass1_out` as `el-bank40-pass1-out-v1`.
7. Push the pass-2 kernel (CPU; jail on): `kaggle kernels push -p "D:\local model\results\raw\el-bank40-bundle-20261018T000347Z\kernel_pass2"`; download its output into <pass2_out_dir>.
8. Results: `python tools/el_bank40_driver.py compile <pass1_out_dir> --pass2 <pass2_out_dir> --out <results dir>` -> BANK40_RESULTS.md/json.

If the jail cannot start on Kaggle the kernel stops with verdict JAIL_UNAVAILABLE and executes nothing (no unsandboxed fallback). Never print or commit key values.
