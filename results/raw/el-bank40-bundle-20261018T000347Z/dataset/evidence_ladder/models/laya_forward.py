"""LAYA forward pass for arm E: frozen Qwen3-4B states -> frozen Fable adapter -> pretrained LAYA head, plus the linear probe and a random-head control.

Everything here is L0: it returns SCORES, never an approval, and callers must not average these with the critic (architecture section 7).
No training, no adapter change, no encoder change. Heavy imports (torch, sklearn, decision_head.head) are lazy so the pure parts (rendering, backend lookup) import anywhere.

Decision rendering for stage-1 coding (EXPLORATORY, OUT-OF-DISTRIBUTION: the adapters learnt Fable next-action choice in agent loops, not whole-program choice):
  context = the training system block (verbatim, hash-pinned) + "\n\n[user]\n" + task statement + public examples + required solution format
  cue     = decision_head.fable_features.PROTOCOL["query_cue"]  ("\n\nNext action:"), appended by the capture, not here
  options = the task's DISTINCT candidate programs (code only, never the reasoning), each as a Write tool call in the training option format
            (compact JSON, name then arguments with sorted keys). The trained recipe has no NONE option (pools of 3-5 real actions), so none is added;
            a task with fewer than 2 distinct candidates is not scored. One packed capture per task: the options are independent suffixes of one shared prefix.
Scores are logits inside that pool (pool-dependent, softmax-equivariant); the head has no positional encoding so option order does not matter.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

SYSTEM_BLOCK_SHA256 = "9bf0352e5debcef4997f0fe2b514a408b072c61e83a87249c7aeb4fdaa5d7250"   # the single system block of all 337 TRAIN contexts (1242 chars)
QUERY_CUE = "\n\nNext action:"
SOLUTION_PATH = {"python": "solution.py", "java": "Solution.java"}
EXPLORATORY_LABEL = "exploratory, out-of-distribution: next-action head applied to whole-program candidates; agreement with Fable was never measured here"
SEEDS = ("final-s0", "final-s1", "final-s2")
from evidence_ladder.models import backbone as BB   # noqa: E402

_P4 = BB.get()      # the LAYA head + adapters read THIS backbone's states (profile qwen3-4b-thinking-2507); depths / hidden / cue come from the profile
DEPTHS, HIDDEN, QUERY_CUE = _P4.depths, _P4.hidden_size, _P4.query_cue


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def candidate_option(code: str, language: str) -> str:
    """One candidate program as a Write tool call, byte-format of the TRAIN options (compact, ensure_ascii=False, arguments sorted)."""
    return json.dumps({"name": "Write", "arguments": {"content": code, "file_path": SOLUTION_PATH[language]}}, separators=(",", ":"), ensure_ascii=False)


def render_context(system_block: str, task_prompt: str, examples: str, solution_format: str) -> str:
    if sha256_text(system_block) != SYSTEM_BLOCK_SHA256:
        raise ValueError("system block differs from the one the adapters were trained with")
    return "%s\n\n[user]\n%s\n\nPublic examples (input and expected output):\n%s\n\nRequired solution format:\n%s" % (system_block, task_prompt, examples, solution_format)


def decision_request(system_block: str, task: Mapping[str, Any], codes: Sequence[str], qwen) -> Dict[str, Any]:
    """-> {context_text, options (one per DISTINCT code, first-seen order), index_of_code: for each input code the option index}."""
    lang = task["language"]
    uniq: List[str] = []
    idx = []
    for c in codes:
        if c not in uniq:
            uniq.append(c)
        idx.append(uniq.index(c))
    return {"context_text": render_context(system_block, task["prompt"], qwen.format_examples(task["public_tests"]), qwen.SOLUTION_FORMATS[lang]),
            "options": [candidate_option(c, lang) for c in uniq], "index_of_code": idx}


# ------------------------------------------------------------------ heads (torch)
def build_head(pretrained_path, adapter_path: Optional[str] = None, *, random_head_seed: Optional[int] = None):
    """Mirrors decision_head.fable_train.build_head (same constructor, same 31-tensor mapping, kind_emb zero) so the training modules are not needed.
    adapter_path: a final-s*.pt (state of every non-pretrained tensor + scaler buffers). random_head_seed: the control, the 31 pretrained tensors are NOT loaded
    (fresh seeded init) but the trained adapter is."""
    import torch
    from safetensors.torch import load_file
    from decision_head.head import DecisionHead
    with torch.random.fork_rng():
        torch.manual_seed(0 if random_head_seed is None else random_head_seed)
        head = DecisionHead(HIDDEN, d=1024, layers=2, k_layers=len(DEPTHS), scorer="mlp", dropout=0.0)
    if random_head_seed is None:
        mapped = {}
        for k, v in load_file(str(pretrained_path)).items():
            if k.startswith("head.layers."):
                mapped["ctx.layers." + k[len("head.layers."):]] = v.float()
            elif k.startswith("scorer."):
                mapped["mlp." + k[len("scorer."):]] = v.float()
            elif k == "type_emb.weight":
                mapped["prim_emb.weight"] = v.float()
        r = head.load_state_dict(mapped, strict=False)
        assert not r.unexpected_keys and len(mapped) == 31, (r.unexpected_keys, len(mapped))
    with torch.no_grad():
        head.kind_emb.weight.zero_()
    if adapter_path is not None:
        st = torch.load(adapter_path, map_location="cpu", weights_only=False)
        assert not head.load_state_dict(st["state"], strict=False).unexpected_keys
    return head.eval()


def head_logits(head, hq, ho):
    """hq [K,H], ho [N,K,H] (fp16 or fp32 tensors), N >= 2 -> list of N float logits (prim = choice, as the trainer's logits_of). CPU, fp32, no grad."""
    import torch
    with torch.no_grad():
        n = ho.shape[0]
        lg, _ = head(hq.float()[None], ho.float()[None], torch.ones(1, n, dtype=torch.bool), torch.zeros(1, dtype=torch.long))
    return [float(x) for x in lg[0, :n]]


# ------------------------------------------------------------------ linear probe (the mandatory baseline)
def fit_probe(rows: Sequence[Mapping[str, Any]], C: float = 0.01, random_state: Optional[int] = None):
    """Recipe of kaggle/head_exp4_gpu.py / tools/fullaction_pilot_v1.py probe_scores with C pinned: logistic regression on the standardised raw option states
    (3 depths x 2560 flattened), class_weight balanced, max_iter 2000. rows: {ho: tensor [N,K,H], pos: bool tensor [N]}. -> (scaler, model)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([r["ho"].float().reshape(len(r["pos"]), -1).numpy() for r in rows])
    y = np.concatenate([r["pos"].numpy().astype(int) for r in rows])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=random_state).fit(sc.transform(X), y)


# [EL-ARCH-1 C1] every scorer output is labelled with its role
ROLES = {"laya": "scorer_under_test", "probe": "mandatory_baseline_linear_probe", "random_head": "untrained_random_init_floor_control"}


def probe_decision(probe, ho) -> List[float]:
    """ho [N,K,H] -> N decision-function values."""
    sc, m = probe
    return [float(x) for x in m.decision_function(sc.transform(ho.float().reshape(ho.shape[0], -1).numpy()))]


def hit(scores: Sequence[float], ref: int) -> float:
    """Tie-aware top-1 hit, the same rule as tools/fullaction_pilot_v1.hit."""
    s = [float(x) for x in scores]
    top = [i for i, v in enumerate(s) if v >= max(s) - 1e-12]
    return float(ref in top) / len(top)


# ------------------------------------------------------------------ the ladder-side scorer backend (no torch, no model)
class SavedScoreBackend:
    """Scores a Kaggle kernel already computed, keyed by sha256 of the candidate text. Only records whose `origin` says the runtime itself produced the features
    (live capture in the scoring process, or stored features verified by re-capture) are loaded; anything else is counted in `.refused`. A source with no saved
    score raises KeyError (scorer unavailable, never 0)."""
    ORIGINS = ("live_capture_same_process", "stored_features_verified_by_recapture")

    def __init__(self, records: Sequence[Mapping[str, Any]], kind: str):
        assert kind in ("laya", "probe", "random_head"), kind
        self.kind, self._by, self.refused = kind, {}, 0
        for r in records:
            sc = (r.get("scores") or {}).get(kind)
            if r.get("origin") not in self.ORIGINS:
                self.refused += 1
                continue
            if sc and sc.get("status") == "ok" and r.get("candidate_source_sha256"):
                self._by[r["candidate_source_sha256"]] = float(sc["score"])

    def score(self, decision_type: str, pool: Sequence[Any]) -> List[float]:
        return [self._by[sha256_text(p if isinstance(p, str) else str(p))] for p in pool]
