"""Pack the head-swap experiment (kaggle/head_swap_gpu.py) for Kaggle: file work only, NO upload and NO launch (held for the user's review).

  PYTHONPATH=. python tools/head_swap_pack_v1.py <out dir>

upload/heads.pt: the 31 transferred decision tensors of each of the four on-disk LAYA checkpoints, mapped exactly as decision_head/fable_train.py
build_head maps them (head.layers.* -> ctx.layers.*, scorer.* -> mlp.*, type_emb.weight -> prim_emb.weight), kept in their stored dtype; the three
Fable-trained adapters of laya-baseline-v1 (for the zero-training swap). The Fable / VAL / native features are NOT repacked: the kernel mounts the
existing private dataset laya-head-exp5-v1 (pack5.pt, exp4lib.py, head.py). No CAL / TEST, no texts.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
SLUG, DATA5 = "sangeetasangamkar/laya-head-swap-v1", "sangeetasangamkar/laya-head-exp5-v1"
CKPT = dict(coding="models/coding-decision-agent/model.safetensors", typed="models/laya-typed-decisions/model.safetensors",
            code="models/laya-code/model.safetensors", conductor="models/laya-conductor/model.safetensors")
ADAPTERS = ["results/raw/laya-finetune-all-v1-20261012T080145Z/adapters/final-s%d.pt" % s for s in (0, 1, 2)]
SKIPPED = {"act_head.0.weight", "act_head.0.bias", "act_head.2.weight", "act_head.2.bias", "temperature"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mapped(path):
    sd, out, other = load_file(str(ROOT / path)), {}, set()
    for k, v in sd.items():
        if k.startswith("head.layers."):
            out["ctx.layers." + k[len("head.layers."):]] = v
        elif k.startswith("scorer."):
            out["mlp." + k[len("scorer."):]] = v
        elif k == "type_emb.weight":
            out["prim_emb.weight"] = v
        elif not k.startswith("encoder"):
            other.add(k)
    assert len(out) == 31 and other == SKIPPED, (path, len(out), other)
    return out


def main(out_arg):
    out = ROOT / out_arg
    (out / "upload").mkdir(parents=True)
    (out / "kernel").mkdir()
    heads = {n: dict(tensors=mapped(p)) for n, p in CKPT.items()}
    shapes = {n: {k: tuple(v.shape) for k, v in h["tensors"].items()} for n, h in heads.items()}
    assert all(s == shapes["coding"] for s in shapes.values()), "shapes differ between checkpoints"
    assert sha(CKPT["coding"]) == "e316f15e1be381aee51ebfd5b9ccb1987395a20c31c103a42053e0f97c72810a"
    ads = [torch.load(ROOT / p, map_location="cpu", weights_only=False)["state"] for p in ADAPTERS]
    sources = dict(checkpoints={n: dict(path=p, sha256=sha(ROOT / p)) for n, p in CKPT.items()}, baseline_adapters=[dict(path=p, sha256=sha(ROOT / p)) for p in ADAPTERS],
                   not_loaded=sorted(SKIPPED))
    torch.save(dict(heads=heads, baseline_adapters=ads, sources=sources), out / "upload/heads.pt")
    shutil.copy(ROOT / "kaggle/head_swap_gpu.py", out / "kernel/laya-head-swap-run.py")
    (out / "upload/dataset-metadata.json").write_text(json.dumps(dict(title="laya-head-swap-v1", id=SLUG, licenses=[dict(name="other")]), indent=1), "utf-8")
    (out / "kernel/kernel-metadata.json").write_text(json.dumps(dict(
        id="sangeetasangamkar/laya-head-swap-run", title="laya-head-swap-run", code_file="laya-head-swap-run.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=False, dataset_sources=[DATA5, SLUG], kernel_sources=[], competition_sources=[], model_sources=[],
        machine_shape="NvidiaTeslaT4"), indent=1), "utf-8")
    man = dict(status="PREPARED, NOT UPLOADED, NOT LAUNCHED (held for the user's review)", sources=sources,
               tensor_dtype={n: str(next(iter(h["tensors"].values())).dtype) for n, h in heads.items()}, contains_cal_or_test=False, contains_texts=False,
               reused_dataset=dict(slug=DATA5, local=str(Path("results/raw/head-exp5-bundle-v1-20261012T203417Z/upload")), pack5_sha256=
                                   "0aec2b00f85ad90c5a64d2cb02b1fc5faba86869409e024b663664fb4ba0b456"),
               script_sha256=sha(ROOT / "kaggle/head_swap_gpu.py"),
               files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted((out / "upload").iterdir())})
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), "utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
