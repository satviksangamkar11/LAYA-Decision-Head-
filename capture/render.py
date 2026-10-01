"""Render one compiled decision into a capture-ready record (schema capture-record-v1). Model-independent code:
the tokenizer is a parameter, no constant names a model, a layer, a hidden size or a token id.

Layout (design B, DECISION_HEAD.md section 3): a shared prefix (state + question + answer cue) and one independent suffix per
candidate, so each option's hidden states depend only on the prefix, never on the other options or their order.
Segments are tokenised separately so every position is exact. Raises instead of truncating.

Frozen schema (change = new schema name, never edit in place):
  schema, decision_id, decision_type, order (candidate ids as presented), label_index, prefix_ids, query_pos (= last prefix token),
  suffix_ids (one list per candidate), option_spans ([start, end) inside each suffix), mask, meta (head_meta rows, never `source`),
  state_hash, manifest_hash, prompt_hash, layout_version, tokenizer_revision, n_prefix_tokens, n_suffix_tokens
Provenance (repo, trajectory_id, tier, label) is stored beside the record under `provenance`, never tokenised.
"""
import hashlib
import json
import random

from state.schema import Candidate

SCHEMA = "capture-record-v1"
LAYOUT = "plain-v1"
QUESTION = ("\n\nQuestion: Which action should the agent take next?\n"
            "Choose the single best next action from the options that follow. "
            "If none fits or the evidence is insufficient, choose the none option.\nAnswer:")


class RenderOverflow(Exception):
    pass


def _ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def render(rec, state, tok, tokenizer_revision, max_prefix_tokens=12000, order=None):
    """rec: a data/decisions record; state: build_state() output for it. `order` (indices) re-presents the candidates."""
    cands = rec["candidates"]
    idx = list(range(len(cands))) if order is None else list(order)
    assert sorted(idx) == list(range(len(cands))), "order must be a permutation"
    assert sum(c["id"] == "NONE" for c in cands) == 1, "every decision carries exactly one NONE"
    prefix = _ids(tok, state["text"]) + _ids(tok, QUESTION)
    if len(prefix) > max_prefix_tokens:
        raise RenderOverflow(f"{rec['decision_id']}: prefix {len(prefix)} tokens > {max_prefix_tokens}")
    suffix = [_ids(tok, " " + cands[i]["text"]) for i in idx]
    meta = [Candidate(candidate_id=cands[i]["id"], text=cands[i]["text"], action_type=cands[i]["action_type"], tool=cands[i]["tool"],
                      safety_class=cands[i]["safety_class"], executable=cands[i]["id"] != "NONE").head_meta() for i in idx]
    out = {"schema": SCHEMA, "decision_id": rec["decision_id"], "decision_type": rec["decision_type"],
           "order": [cands[i]["id"] for i in idx], "label_index": idx.index(rec["label_index"]),
           "prefix_ids": prefix, "query_pos": len(prefix) - 1, "suffix_ids": suffix,
           "option_spans": [[0, len(s)] for s in suffix], "mask": [1] * len(idx), "meta": meta,
           "state_hash": state["state_hash"], "manifest_hash": state["manifest_hash"], "layout_version": LAYOUT,
           "tokenizer_revision": tokenizer_revision, "n_prefix_tokens": len(prefix), "n_suffix_tokens": sum(map(len, suffix))}
    out["prompt_hash"] = hashlib.sha256(json.dumps([LAYOUT, tokenizer_revision, prefix, suffix], sort_keys=True).encode()).hexdigest()
    out["provenance"] = dict(rec["provenance"], tier=rec.get("tier"), split=rec.get("split"))
    return out


def selfcheck(path_dec="data/decisions/action-swerebench-sample-v2.jsonl", path_raw="data/raw/swerebench-sample-20261001.jsonl",
              tok_dir=r"D:\local model\models\qwen3-4b-thinking-2507", manifest="results/raw/qwen3-4b-thinking-2507.manifest.json"):
    from transformers import AutoTokenizer
    from compiler import state_builder
    tok = AutoTokenizer.from_pretrained(tok_dir)
    rev = json.load(open(manifest))["revision"]
    raw = {}
    for l in open(path_raw, encoding="utf-8"):
        r = json.loads(l)
        raw[r["trajectory_id"]] = r
    recs = []
    for l in open(path_dec, encoding="utf-8"):
        r = json.loads(l)
        recs.append(r)
        if len(recs) >= 400:
            break
    rec = max((r for r in recs if r["cut_stage"] == "none"), key=lambda r: r["t"])  # a real decision with a long history
    ev = raw[rec["provenance"]["trajectory_id"]]["trajectory"]
    st = state_builder.build_state(ev, rec["t"], 32000)
    assert st["state_hash"] == rec["state_hash"], "state rebuild differs from the compiled record"
    a = render(rec, st, tok, rev)
    b = render(rec, state_builder.build_state(ev, rec["t"], 32000), tok, rev)
    assert a == b and a["prompt_hash"] == b["prompt_hash"], "re-render is not identical"
    # spans decode to the option text; query position is the answer cue
    for i, s in enumerate(a["suffix_ids"]):
        assert tok.decode(s).strip() == rec["candidates"][[c["id"] for c in rec["candidates"]].index(a["order"][i])]["text"].strip()
    assert tok.decode(a["prefix_ids"][a["query_pos"]]).strip() == ":" and a["mask"] == [1] * len(a["order"])
    # order independence: shuffled presentation gives the same suffix tokens per candidate and a correct label index
    perm = list(range(len(rec["candidates"])))
    random.Random(0).shuffle(perm)
    c = render(rec, st, tok, rev, order=perm)
    by_id = {cid: s for cid, s in zip(a["order"], a["suffix_ids"])}
    assert all(by_id[cid] == s for cid, s in zip(c["order"], c["suffix_ids"])), "suffix depends on order"
    assert c["prefix_ids"] == a["prefix_ids"] and c["order"][c["label_index"]] == a["order"][a["label_index"]]
    assert all(len(m) == 12 for m in a["meta"]) and "source" not in json.dumps(a["meta"])
    # overflow raises instead of truncating
    try:
        render(rec, st, tok, rev, max_prefix_tokens=10)
        raise AssertionError("overflow did not raise")
    except RenderOverflow:
        pass
    print("OK", rec["decision_id"], "prefix tokens", a["n_prefix_tokens"], "suffix tokens", a["n_suffix_tokens"],
          "options", len(a["order"]), "prompt_hash", a["prompt_hash"][:16])
    return a


if __name__ == "__main__":
    selfcheck()
