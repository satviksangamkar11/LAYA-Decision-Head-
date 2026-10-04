"""Kaggle TPU v5e-8 smoke: download Qwen/Qwen3-30B-A3B inside Kaggle and answer ONE question with vllm-tpu. Exploratory, no gate, nothing from this machine."""
import os, sys, json, time, shutil, subprocess

T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def sh(cmd, check=True):
    log("$", cmd); p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout: print(line.rstrip()[:400], flush=True)
    p.wait()
    if check and p.returncode: raise SystemExit("FAILED (%d): %s" % (p.returncode, cmd))
    return p

log("python", sys.version.split()[0]); sh("uname -a; df -h /tmp /kaggle/working | cat; free -g | head -2; ls /dev | grep -i -E 'accel|vfio|tpu' | head", check=False)
import glob
WEIGHTS, MOUNTED = None, False
for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
    if "30b" in c.lower() and "a3b" in c.lower() and "fp8" not in c.lower() and "base" not in c.lower() and "2507" not in c.lower():
        WEIGHTS, MOUNTED = os.path.dirname(c), True; log("using mounted Kaggle model", WEIGHTS); break
if not WEIGHTS:
    for d in ("/tmp/q30", "/dev/shm/q30", "/kaggle/temp/q30"):
        try:
            os.makedirs(d, exist_ok=True); free = shutil.disk_usage(os.path.dirname(d)).free / 1e9; log("disk free for", d, "%.0f GB" % free)
            if free > 75 and WEIGHTS is None: WEIGHTS = d
        except Exception as e: log("skip", d, e)
    assert WEIGHTS, "no mounted model and no directory with 75 GB free for the 61 GB bf16 weights"
log("weights dir", WEIGHTS, "mounted" if MOUNTED else "will download from Hugging Face")

sh("pip install -q uv")
sh("uv venv /tmp/vt --python 3.12")
sh("uv pip install --python /tmp/vt/bin/python vllm-tpu==0.28.0 huggingface_hub hf_transfer 2>&1")

RUN = r'''
import os, sys, time, json
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
from huggingface_hub import snapshot_download
if os.path.exists(os.path.join(sys.argv[1], "config.json")):
    path = sys.argv[1]; log("using mounted model, no download")
else:
    log("downloading Qwen/Qwen3-30B-A3B")
    path = snapshot_download("Qwen/Qwen3-30B-A3B", local_dir=sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
log("downloaded", sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e9, "GB")
import jax
log("jax devices", jax.device_count(), [d.device_kind for d in jax.devices()][:8])
from vllm import LLM, SamplingParams
llm = LLM(model=path, tensor_parallel_size=jax.device_count(), max_model_len=2048, max_num_seqs=4, dtype="bfloat16")
log("engine ready")
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
sp = SamplingParams(temperature=0.0, max_tokens=64)
llm.generate([prompt], sp)                        # warm-up (XLA compile)
t = time.time(); out = llm.generate([prompt], sp)[0]; dt = time.time() - t
text = out.outputs[0].text; n = len(out.outputs[0].token_ids)
log("ANSWER:", repr(text)); log("tokens", n, "in %.2fs = %.1f tok/s" % (dt, n / dt))
t = time.time(); outs = llm.generate([prompt] * 4, SamplingParams(temperature=0.7, max_tokens=64)); dt4 = time.time() - t
log("4 parallel: %.1f tok/s" % (sum(len(o.outputs[0].token_ids) for o in outs) / dt4))
json.dump(dict(model="Qwen/Qwen3-30B-A3B", devices=jax.device_count(), answer=text, tokens=n, seconds=dt, tok_per_s_single=n / dt, tok_per_s_4_parallel=sum(len(o.outputs[0].token_ids) for o in outs) / dt4, vllm_tpu="0.28.0", seconds_total=time.time() - T0), open("/kaggle/working/tpu30b_result.json", "w"), indent=1)
log("DONE")
'''
open("/kaggle/working/run30b.py", "w").write(RUN)
sh("/tmp/vt/bin/python -u /kaggle/working/run30b.py %s 2>&1" % WEIGHTS)
log("finished")
