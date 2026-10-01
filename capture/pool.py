"""Full-pool capture records (schema capture-record-v2). Model-independent, like capture/render.py: the tokenizer is a parameter.
One record = the shared prefix (state + question + answer cue) plus one independent suffix for EVERY pool candidate and for the mandatory NONE option.
Candidate vectors do not depend on the other candidates (verified, and asserted again by tools/capture_pool_v3.py), so any candidate set can be sampled offline.
Rules: configs/a5_v3_spec.toml [capture_v3]."""
import hashlib
import json

from capture.render import QUESTION, RenderOverflow, _ids
from compiler.candidates import distractor_pool

SCHEMA = "capture-record-v2"
NONE_TEXT = "None of these options, or the evidence is insufficient"


def render_pool(rec, ev, state, tok, tokenizer_revision, max_prefix_tokens=12000):
    t = int(rec["decision_id"].rsplit(":", 1)[1])
    pool = distractor_pool(ev, t)
    texts = [p[0] for p in pool] + [NONE_TEXT]
    prefix = _ids(tok, state["text"]) + _ids(tok, QUESTION)
    if len(prefix) > max_prefix_tokens:
        raise RenderOverflow(f"{rec['decision_id']}: prefix {len(prefix)} tokens > {max_prefix_tokens}")
    suffix = [_ids(tok, " " + x) for x in texts]
    out = {"schema": SCHEMA, "decision_id": rec["decision_id"], "texts": texts, "prefix_ids": prefix, "query_pos": len(prefix) - 1, "suffix_ids": suffix,
           "state_hash": state["state_hash"], "tokenizer_revision": tokenizer_revision, "n_prefix_tokens": len(prefix), "n_suffix_tokens": sum(map(len, suffix)),
           "pool_size": len(pool)}
    out["pool_hash"] = hashlib.sha256("\n".join(texts[:-1]).encode()).hexdigest()[:16]
    out["prompt_hash"] = hashlib.sha256(json.dumps([SCHEMA, tokenizer_revision, prefix, suffix], sort_keys=True).encode()).hexdigest()
    return out
