"""Structural load test, no weights loaded into RAM.

Builds the model on the meta device from config.json (transformers' own class), then compares
every parameter name / shape against the safetensors headers on disk. Catches missing or extra
tensors and shape mismatches before any runtime test.

Usage: python check_loadable.py MODEL_DIR
"""
import json, struct, sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

root = Path(sys.argv[1])
cfg = AutoConfig.from_pretrained(root)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}

disk, dtypes = {}, Counter()
for f in sorted(root.glob("*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__":
            disk[k] = tuple(v["shape"])
            dtypes[v["dtype"]] += 1

missing = sorted(set(expected) - set(disk))
extra = sorted(set(disk) - set(expected))
shape_bad = [(k, expected[k], disk[k]) for k in set(expected) & set(disk) if expected[k] != disk[k]]
print("architecture:", cfg.architectures, "| layers:", getattr(cfg, "num_hidden_layers", None))
print("quantization_config:", getattr(cfg, "quantization_config", None))
print("expected tensors:", len(expected), "| on disk:", len(disk), "| dtypes on disk:", dict(dtypes))
print("missing from disk:", len(missing), missing[:8])
print("extra on disk:", len(extra), extra[:8])
print("shape mismatches:", len(shape_bad), shape_bad[:8])
ok = not (missing or shape_bad)
print("VERDICT:", "STRUCTURALLY LOADABLE" if ok else "PROBLEM FOUND")
sys.exit(0 if ok else 1)
