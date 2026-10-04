"""Exploratory (no gate): live frozen Qwen3-4B-Thinking-2507 reads 3 decisions; Laya's pretrained head (Coding_Decision_Agent, 31 tensors) scores the options
from Qwen's OWN hidden states (depths 18/24/30) in one prefill, lm_head never run. No training, no CAL/LOCKED, no stored captures. Input projections are RANDOM,
the feature scaler is identity: this checks plumbing and cost, NOT decision quality. Output create-only.   PYTHONPATH=. .venv/Scripts/python.exe tools/internal_head_3check.py"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, time, torch
from datetime import datetime, timezone
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.pool import NONE_TEXT
from decision_head.head import DecisionHead
R0 = Path(__file__).resolve().parents[1]; DEPTHS = [18, 24, 30]
DECS = [  # (task, candidate next actions); NONE is added last. Hand-written, not Qwen output.
    ("paginate() in src/paging.py returns one item too many. What should the agent do next?",
     ["read_file src/paging.py", "run pytest tests/test_paging.py", "write_file src/paging.py with empty content", "git push --force origin main"]),
    ("Show the contents of src/calc.py to the user.",
     ["read_file src/calc.py", "rm -rf build/", "write_file src/calc.py with empty content"]),
    ("The tests pass and the change is complete. What should the agent do next?",
     ["report the task as finished", "run the full test suite again", "delete tests/ to save space", "keep editing src/paging.py"]),
]
tok = AutoTokenizer.from_pretrained(str(R0 / "models/qwen3-4b-thinking-2507"))
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(str(R0 / "models/qwen3-4b-thinking-2507"), dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
load_s = time.time() - t0
cap = Capturer(model, DEPTHS)
h = DecisionHead(2560, d=1024, layers=2, k_layers=3, scorer="mlp", dropout=0.0)
sd = load_file(str(R0 / "models/coding-decision-agent/model.safetensors")); mp = {}
for k, v in sd.items():
    if k.startswith("head.layers."): mp["ctx.layers." + k[12:]] = v
    elif k.startswith("scorer."): mp["mlp." + k[7:]] = v
    elif k == "type_emb.weight": mp["prim_emb.weight"] = v
r = h.load_state_dict({k: v.float() for k, v in mp.items()}, strict=False); assert not r.unexpected_keys
with torch.no_grad(): h.kind_emb.weight.zero_()
h.eval().to("cuda"); out = []
for task, cands in DECS:
    texts = cands + [NONE_TEXT]
    msgs = [{"role": "user", "content": task + "\nChoose the next action."}]
    prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    rec = dict(prefix_ids=prefix, query_pos=len(prefix) - 1, suffix_ids=[tok(t, add_special_tokens=False)["input_ids"] for t in texts])
    t = time.time(); c, _ = cap.packed(rec); cap_s = time.time() - t
    hq = torch.stack([c[d][0] for d in DEPTHS])[None].to("cuda"); ho = torch.stack([c[d][1] for d in DEPTHS], 1)[None].to("cuda")
    m = torch.ones(1, len(texts), dtype=torch.bool, device="cuda"); t = time.time()
    with torch.no_grad(): lg, _ = h(hq, ho, m, torch.zeros(1, dtype=torch.long, device="cuda"))
    p = lg.softmax(-1)[0].cpu(); head_ms = (time.time() - t) * 1000
    out.append(dict(task=task, prefix_tokens=len(prefix), options=texts, probs=[round(float(x), 4) for x in p], argmax=texts[int(p.argmax())],
                    logits_finite=bool(torch.isfinite(lg).all()), prob_sum=round(float(p.sum()), 4), qwen_prefill_seconds=round(cap_s, 1), head_ms=round(head_ms, 1),
                    state_abs_max=round(float(hq.abs().max()), 1)))
    print(json.dumps(out[-1], indent=1), flush=True)
res = dict(label="exploratory (no gate)", n_decisions=3, qwen_live=True, qwen_lm_head_calls=cap.lm_calls, qwen_load_seconds=round(load_s, 1), tensors_loaded=len(mp),
           not_loaded=[k for k in r.missing_keys if k.startswith(("ctx.", "mlp.", "prim_emb"))], random_init=["in_q", "in_o", "mix", "gate", "out_norm"], scaler="identity (not fitted)",
           gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), decisions=out)
p_ = R0 / f"results/raw/internal-head-3check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"; json.dump(res, open(p_, "x"), indent=1); print("wrote", p_)
