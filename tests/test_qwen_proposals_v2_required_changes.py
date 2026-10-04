#!/usr/bin/env python3
"""
Failing test for qwen_proposals_v2.py: demonstrates required features that are missing.

This test shows:
1. Full raw text not stored
2. Logprobs not captured
3. exact_hit and near_hit flags missing
4. Style leak (canonical JSON vs gold style)
5. Tool validation missing
6. Diversity tracking missing
7. Reproducibility fields missing in manifest
8. Per-source hit rate missing in final manifest
9. No documentation of what Qwen doesn't see
10. No split consistency assertion

Status: These are intentionally failing to drive implementation.
Run with: python tests/test_qwen_proposals_v2_required_changes.py
"""

import json


def test_1_full_raw_text_stored():
    """FAIL: Raw text is truncated to 200 chars (thinking) and 1500 chars (answer)."""
    output = {
        'thinking_head': 'x' * 200,
        'answer_text': 'x' * 1500,
        'out_tokens': 100,
    }
    # Should have:
    assert 'raw_text' in output, "MISSING: full raw_text field (should be gzipped or stored)"
    assert len(output.get('raw_text', '')) >= len(output['thinking_head']) + len(output['answer_text']), \
        "MISSING: raw_text should contain full thinking + answer, not truncated"


def test_2_logprobs_captured():
    """FAIL: No cumulative logprob or mean token logprob stored."""
    output = {
        'action_text': '{"name":"x"}',
        'out_tokens': 50,
    }
    # Should have:
    assert 'cumulative_logprob' in output, \
        "MISSING: cumulative_logprob (from vLLM CompletionOutput.cumulative_logprob)"
    assert 'mean_token_logprob' in output, \
        "MISSING: mean_token_logprob (cumulative_logprob / out_tokens)"
    # For soft target training (log-odds scoring):
    if output.get('cumulative_logprob') is not None:
        expected_mean = output['cumulative_logprob'] / max(output.get('out_tokens', 1), 1)
        assert abs(output.get('mean_token_logprob', 0) - expected_mean) < 0.01, \
            "WRONG: mean_token_logprob should equal cumulative / token count"


def test_3_exact_hit_and_near_hit():
    """FAIL: Only has action_text == gold check (too strict, no near_hit)."""
    gold = '{"name": "execute_bash", "arguments": {"command": "ls"}}'

    proposal_exact = '{"name":"execute_bash","arguments":{"command":"ls"}}'
    proposal_near = '{"name": "execute_bash", "arguments": {"command":"ls"}}'  # Different spacing
    proposal_wrong = '{"name":"execute_bash","arguments":{"command":"cat"}}'  # Different command

    samples = [
        {'action_text': proposal_exact, 'raw_text': 'exact match'},
        {'action_text': proposal_near, 'raw_text': 'near match (spacing)'},
        {'action_text': proposal_wrong, 'raw_text': 'wrong option'},
    ]

    # Should have exact_hit and near_hit flags:
    for sample in samples:
        assert 'exact_hit' in sample, f"MISSING: exact_hit flag in {sample['action_text'][:50]}"
        assert 'near_hit' in sample, f"MISSING: near_hit flag in {sample['action_text'][:50]}"

    # Verify: exact match should have exact_hit=True
    assert samples[0].get('exact_hit') == True, "exact match should have exact_hit=True"

    # Verify: near match (whitespace only) should have near_hit=True, not low score
    # (Requires norm() from tools/hybrid_options.py)
    if samples[1].get('near_hit'):
        print("OK: near match detected (exact whitespace difference)")
    else:
        raise AssertionError("WRONG: near match not detected; should use norm() for tolerance")

    # Verify: wrong option should have both False
    assert samples[2].get('exact_hit') == False, "wrong option should have exact_hit=False"
    assert samples[2].get('near_hit') == False, "wrong option should have near_hit=False"


def test_4_style_not_leaked():
    """FAIL: action_text is canonical compact JSON, gold is default style."""
    # Gold (from rows.jsonl): original style
    gold_original_style = '{"name": "execute_bash", "arguments": {"command": "ls"}}'

    # Current proposal output: canonical compact
    proposal_canonical = '{"name":"execute_bash","arguments":{"command":"ls"}}'

    # TF-IDF baseline can separate by format alone (spaces after ':' and ',')
    # This is a STYLE LEAK: the gold is distinguishable from proposals by spacing

    # Must fix: serialize ALL options through ONE serializer
    # Either:
    # (a) Return proposals in gold's original style (spaces, key order)
    # (b) Re-serialize gold to canonical style before assembly
    # (c) Have pool_assembly normalize all options to one style

    # TEST: They should be indistinguishable by format
    # If text-only classifier finds gold 100%, that's the shortcut
    format_diff = (gold_original_style != proposal_canonical)
    assert not format_diff, \
        "STYLE LEAK: gold has spaces, proposals are compact. " \
        "TF-IDF can separate by spacing alone. Use ONE serializer."


def test_5_tool_validation():
    """FAIL: No validation of tool names or arguments."""
    task_tools = ['execute_bash', 'write_file', 'read_file']

    valid_proposal = '{"name":"execute_bash","arguments":{"command":"ls"}}'
    invalid_tool = '{"name":"invalid_tool","arguments":{"x":"y"}}'
    invalid_args = '{"name":"execute_bash","arguments":{}}'  # Missing required 'command'

    samples = [
        {'action_text': valid_proposal},
        {'action_text': invalid_tool},
        {'action_text': invalid_args},
    ]

    # Should have: validated tool names and arguments
    for sample in samples:
        assert 'valid_tool' in sample or 'tool_error' in sample, \
            "MISSING: validation flag (valid_tool bool or tool_error message)"

    # Should drop invalid tools before returning
    # (Count drops per task, per source)
    raise AssertionError(
        "MISSING: Tool validation. Must validate tool name against task's tool list "
        "and arguments against tool schema. Drop invalid deterministically. "
        "(nebius has 'tools' column; swesmith and fable have own tool sets)"
    )


def test_6_diversity_tracked():
    """FAIL: No diversity metric; 4 samples often repeat."""
    samples = [
        '{"name":"execute_bash","arguments":{"command":"ls"}}',
        '{"name":"execute_bash","arguments":{"command":"ls"}}',
        '{"name":"write_file","arguments":{"path":"f.txt","text":"x"}}',
        '{"name":"write_file","arguments":{"path":"f.txt","text":"x"}}',
    ]

    # Should have: diversity count (how many distinct by canonical form)
    # If fewer than 4 distinct, should sample more and keep top 4 distinct

    distinct_canonicals = len(set(samples))  # Should be 2, not 4

    raise AssertionError(
        f"MISSING: diversity tracking. Got {distinct_canonicals} distinct of {len(samples)} samples. "
        "Must sample extra and keep 4 distinct by canonical form, recording diversity per decision."
    )


def test_7_reproducibility_manifest():
    """FAIL: Manifest missing seed, model hash, vllm version, prompt hash."""
    manifest = {
        'mode': 'think30',
        'worker': 0,
        'decisions_total': 45500,
    }

    # Should have:
    required_fields = ['seed', 'model_revision', 'vllm_version', 'prompt_hash', 'prompt_template']
    missing = [f for f in required_fields if f not in manifest]

    assert not missing, f"MISSING in manifest: {missing}"


def test_8_per_source_hit_rate_in_manifest():
    """FAIL: Final manifest (not just smoke) missing per-source hit rate."""
    manifest = {
        'mode': 'think30',
        'decisions_total': 45500,
    }

    # Should have: per-source breakdown of hit rates
    # {"nebius": {"samples": 20000, "parse_ok": 0.95, "hits": 0.12, "decisions_with_hit": 2400}, ...}

    assert 'per_source_stats' in manifest, \
        "MISSING: per_source_stats in final manifest (not just smoke)"


def test_9_context_fidelity_documented():
    """FAIL: Not documented what Qwen doesn't see."""
    # From header: "state_text only (never the candidates, never the gold)"
    # But should explicitly document:
    # - Tool list (not in state_text)
    # - Earlier steps (only tail of history)
    # - Decision_id, source, split (not in state_text)

    state_text_docstring = """
    state_text = tail of the trajectory (max 5,000 chars)
    Does NOT include:
    - Tool list
    - Earlier steps (only keeps tail of history)
    - The full context (only recent history)
    Qwen must infer tool names from the history examples.
    """

    raise AssertionError(
        "MISSING: documentation of what Qwen does not see. "
        "Must document: tool list, earlier steps, full context. "
        "Say it in the output manifest and/or header."
    )


def test_10_splits_identical():
    """FAIL: No assertion that train/cal/test splits are produced identically."""
    # All splits should use the same:
    # - mode (think30, think_capped, nothink)
    # - max_tokens cap
    # - temperature
    # - top_p
    # - seed (if any)
    # - same model checkpoint

    # Should have: assertion or explicit test that verifies this

    raise AssertionError(
        "MISSING: assertion that train, cal, and test splits are produced identically. "
        "Same mode, cap, temperature, top_p, seed, model checkpoint. "
        "Nothing from test used to tune generation parameters."
    )


def main():
    """Run all tests; each will fail and show what's missing."""
    tests = [
        ("1. Full raw text stored", test_1_full_raw_text_stored),
        ("2. Logprobs captured", test_2_logprobs_captured),
        ("3. exact_hit and near_hit", test_3_exact_hit_and_near_hit),
        ("4. Style not leaked", test_4_style_not_leaked),
        ("5. Tool validation", test_5_tool_validation),
        ("6. Diversity tracked", test_6_diversity_tracked),
        ("7. Reproducibility manifest", test_7_reproducibility_manifest),
        ("8. Per-source hit rate", test_8_per_source_hit_rate_in_manifest),
        ("9. Context fidelity documented", test_9_context_fidelity_documented),
        ("10. Splits identical", test_10_splits_identical),
    ]

    failed = []
    for name, test in tests:
        try:
            test()
            print("PASS: " + name)
        except AssertionError as e:
            print("FAIL: " + name + ": " + str(e)[:100])
            failed.append(name)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print(f"\nFailing tests (needed for production):")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
