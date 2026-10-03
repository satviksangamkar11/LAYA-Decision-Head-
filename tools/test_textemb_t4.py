"""Local self-test of tools/capture_textemb_v3_t4.py on a tiny random Qwen3 + the real tokenizer (CPU, fp32): the script runs end to end, every text is embedded once, the registered batch-vs-alone gate
(cosine >= 0.9999) passes on every shard, shapes/dtypes are right, lm_head is never called, and the output equals a direct standalone embedding computed independently here.
    PYTHONPATH=. .venv/Scripts/python.exe tools/test_textemb_t4.py"""
import os, sys, json, subprocess, tempfile, shutil
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
from pathlib import Path
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from capture.render import _ids

tokdir = r"D:\local model\models\qwen3-4b-thinking-2507"
tok = AutoTokenizer.from_pretrained(tokdir)
torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
tmp = Path(tempfile.mkdtemp())
m = Qwen3ForCausalLM(cfg).eval()
m.save_pretrained(tmp / "model"); tok.save_pretrained(tmp / "model")
texts = sorted({"open file src/a.py", "run pytest -x tests/test_a.py", "edit src/a.py: replace foo with bar", "none of the above", "search the repo for 'class Foo'", "git diff", "x" * 300, "ls", "finish and submit patch"} | {f"inspect module m{i} " + "word " * (i % 9) for i in range(60)})
json.dump(texts, open(tmp / "texts.json", "w", encoding="utf-8"))
env = dict(os.environ, PYTHONPATH=".")
r = subprocess.run([sys.executable, "tools/capture_textemb_v3_t4.py", "--texts", str(tmp / "texts.json"), "--outdir", str(tmp / "out"), "--model", str(tmp / "model"), "--device", "cpu", "--dtype", "fp32"],
                   capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1])
print(r.stdout[-600:], r.stderr[-600:])
assert r.returncode == 0
d = {}
for f in sorted((tmp / "out").glob("shard-*.pt")):
    d.update(torch.load(f))
assert sorted(d) == texts and all(v.shape == (3, 64) and v.dtype == torch.float16 for v in d.values())
# independent standalone reference for 5 texts
sd = m.model
sd.layers = torch.nn.ModuleList(list(sd.layers)[:30])
store = {}
for i, l in enumerate(sd.layers, 1):
    if i in (18, 24, 30):
        l.register_forward_hook(lambda mod, a, k, o, i=i: store.__setitem__(i, o[0] if isinstance(o, tuple) else o), with_kwargs=True)
pre = _ids(tok, "Next action:")
for x in texts[:5] + ["ls"]:
    s = pre + _ids(tok, " " + x)
    with torch.inference_mode():
        sd(input_ids=torch.tensor([s]), use_cache=False)
    ref = torch.stack([store[dp][0, len(pre):].float().mean(0) for dp in (18, 24, 30)])
    cs = torch.nn.functional.cosine_similarity(ref.double(), d[x].float().double(), dim=-1).min().item()
    assert cs > 0.9999, (x, cs)
assert "worst_gate_cosine" in r.stdout and "lm_head_calls 0" in r.stdout
shutil.rmtree(tmp)
print("PASS: T4 text-embedding script embeds every text once, passes the 0.9999 batch-vs-alone gate on every shard, matches an independent standalone reference, lm_head never called")
