"""Build prompts for Qwen proposal generation from decision rows."""

def build_prompt(row, tokenizer):
    """
    Build a chat template prompt from a decision row.

    Args:
        row: dict with keys decision_id, state_text, candidates, ...
        tokenizer: HF tokenizer with apply_chat_template method

    Returns:
        str: formatted prompt (already processed by apply_chat_template)

    Asserts:
        - Only uses row['state_text'], never reads row['candidates']
        - Result is ready to pass to model.generate() or llm.generate()
    """
    state_text = row.get('state_text', '')

    # The instruction to append (system context + user message)
    instruction = (
        'Reply with exactly ONE next action as a JSON object '
        '{"name": <tool>, "arguments": {...}} in the same format as the history above. '
        'No other text, no <think> tags.'
    )

    # Build messages: only state_text + instruction, NO candidates
    messages = [
        {
            "role": "user",
            "content": f"{state_text}\n\n{instruction}"
        }
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


def test_no_candidate_leak():
    """
    Test that candidate texts do not appear in the prompt.

    Load 50 real rows and check:
    (a) Gold candidate text is not in prompt unless already in state_text
    (b) Hard_neg and NONE candidate texts are not in prompt
    """
    import json
    from transformers import AutoTokenizer

    rows_path = "C:/Users/Satvik/AppData/Local/Temp/kaggle-out3/head_rows/rows.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")

    gold_leaked_but_in_state = 0
    gold_leaked_new = 0
    hard_neg_leaked = 0
    total_rows = 0

    with open(rows_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            row = json.loads(line)
            total_rows += 1

            prompt = build_prompt(row, tokenizer)

            # Check gold candidate
            gold_text = row['candidates'][0].get('text', '')
            if gold_text and len(gold_text) > 10:
                if gold_text in prompt:
                    # Is it already in state_text?
                    if gold_text in row.get('state_text', ''):
                        gold_leaked_but_in_state += 1
                    else:
                        gold_leaked_new += 1
                        print(f"  LEAK (new): {row.get('decision_id')}")

            # Check hard_neg candidates
            for cand in row['candidates'][1:]:
                if cand.get('kind', '').startswith('hard_neg') or cand.get('kind') == 'NONE':
                    cand_text = cand.get('text', '')
                    if cand_text and len(cand_text) > 10:
                        if cand_text in prompt:
                            if cand_text not in row.get('state_text', ''):
                                hard_neg_leaked += 1
                                print(f"  hard_neg leaked: {row.get('decision_id')}, kind={cand.get('kind')}")

    print("=" * 60)
    print(f"TEST: Candidate leak check on {total_rows} rows")
    print("=" * 60)
    print(f"Gold leaked (already in state_text): {gold_leaked_but_in_state} (OK)")
    print(f"Gold leaked (NEW, not in state): {gold_leaked_new} (FAIL if > 0)")
    print(f"Hard_neg/NONE leaked: {hard_neg_leaked} (FAIL if > 0)")
    print()
    print(f"RESULT: {'PASS' if gold_leaked_new == 0 and hard_neg_leaked == 0 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    test_no_candidate_leak()
