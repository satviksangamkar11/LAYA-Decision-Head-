"""[EL-ARCH-1 B3] Kaggle T4 generation entry (Qwen3-4B-Thinking-2507, fp16, internet OFF): generation ONLY, no code is executed, no secrets. TEMPLATE: the coordinator replaces
__GEN_REQUEST_JSON__ with {"authorized": true, "jobs": [...]} (jobs as in evidence_ladder/models/gen_entry.py, built on the PC with gen_entry.make_job) and pushes it with the code bundle
(dataset bundle name "evidence_ladder_el_arch1_v1" holding BUNDLE_MANIFEST.json {"bundle","files":{path:sha256},"qwen_revision"} + the evidence_ladder package + specs/evidence_ladder_v1 +
configs/el_arch1_limits_v1.toml) and the Qwen dataset.
Output: /kaggle/working/gen_records.jsonl (append-only, fsynced, resumable) + RUN_SUMMARY.json (always exits 0). The records feed ReplayGenerationBackend on the PC."""
import glob, hashlib, json, os, sys, time, traceback

REQUEST = json.loads(r'''__GEN_REQUEST_JSON__''')
T0 = time.time(); OUT = "/kaggle/working"
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
RS = {"kernel": "el_arch1_generate_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def finish(verdict, **kw):
    RS["verdict"] = verdict; RS["total_seconds"] = round(time.time() - T0, 1); RS.update(kw)
    with open(OUT + "/RUN_SUMMARY.json", "w", encoding="utf-8") as f: json.dump(RS, f, indent=1, default=str)
    print("VERDICT", verdict, flush=True); sys.stdout.flush(); os._exit(0)


try:
    if REQUEST.get("authorized") is not True: finish("REFUSED_NOT_AUTHORIZED")
    os.environ["EVIDENCE_LADDER_ALLOW_INFERENCE"] = "1"      # set only after the request itself carried authorized=true
    import torch, transformers
    RS["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__}
    if not torch.cuda.is_available(): finish("FAILED_NO_CUDA")
    RS["gpu"] = torch.cuda.get_device_name(0)
    mp = next((p for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)) if json.load(open(p, encoding="utf-8")).get("bundle") == "evidence_ladder_el_arch1_v1"), None)
    if not mp: finish("FAILED_NO_BUNDLE")
    BUNDLE = os.path.dirname(mp); man = json.load(open(mp, encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(BUNDLE, f)) or hashlib.sha256(open(os.path.join(BUNDLE, f), "rb").read()).hexdigest() != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad)
    sys.path.insert(0, BUNDLE)
    from evidence_ladder.models import qwen as Q, gen_entry as GE
    assert Q.QWEN_REVISION == man.get("qwen_revision", Q.QWEN_REVISION)
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs: finish("FAILED_NO_MODEL")
    backend = Q.HFQwenBackend(authorized=True, model_dir=dirs[0], log_path=OUT + "/gen_log.jsonl", verify_files=QWEN_SHA)      # sha256-verifies the shards on first load
    summary = GE.run_jobs(backend, REQUEST["jobs"], OUT + "/gen_records.jsonl")
    finish("DONE", summary=summary, snapshot_verified=backend.snapshot_verified)
except BaseException as e:      # noqa: BLE001
    RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc(); finish("FAILED")
