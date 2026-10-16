"""Parse action JSON from Qwen proposal text."""

import json
import re


def parse_action(raw):
    """
    Extract and parse the first balanced {...} JSON object from raw text.

    Args:
        raw: str, raw Qwen output (may include <think>...</think> and other text)

    Returns:
        tuple: (parsed_text_canonical, ok)
            parsed_text_canonical: str, the canonical JSON string of the parsed action
            ok: bool, True if parsing succeeded and required keys present
    """
    # Strip <think>...</think> if present
    if "<think>" in raw:
        # Find the closing </think>
        think_end = raw.find("</think>")
        if think_end >= 0:
            raw = raw[think_end + 8:]  # Skip past </think>

    # Find first balanced {...} by brace matching
    first_brace = raw.find("{")
    if first_brace < 0:
        return ("", False)

    # Brace matching: find the closing }
    depth = 0
    in_string = False
    escape_next = False
    json_end = -1

    for i in range(first_brace, len(raw)):
        c = raw[i]

        if escape_next:
            escape_next = False
            continue

        if c == "\\":
            escape_next = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    json_end = i
                    break

    if json_end < 0:
        return ("", False)

    json_str = raw[first_brace:json_end + 1]

    # Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return (json_str, False)

    # Check required keys: name and arguments
    if not isinstance(parsed, dict):
        return (json_str, False)

    if "name" not in parsed or "arguments" not in parsed:
        return (json_str, False)

    # Return canonical form
    canonical = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    return (canonical, True)


def test_parse_action():
    """Test parse_action on 8 hand-made strings."""
    test_cases = [
        # (raw_text, expected_ok, description)
        ('{"name": "execute_bash", "arguments": {"command": "ls"}}', True, "Valid JSON"),
        ('<think>Planning...\n</think>\n{"name": "execute_bash", "arguments": {"command": "ls"}}', True, "With <think> tags"),
        ('Some text before\n{"name": "file_edit", "arguments": {"path": "a.py"}}\nSome text after', True, "JSON in the middle"),
        ('{"name": "broken", "arguments": }', False, "Broken JSON (trailing })"),
        ('{"name": "missing_args"}', False, "Missing 'arguments' key"),
        ('{malformed json', False, "Not balanced braces"),
        ('{"name": "execute_bash", "arguments": {"command": "echo \\"quoted\\""}}', True, "JSON with escaped quotes"),
        ('No JSON here at all', False, "No JSON"),
    ]

    print("=" * 60)
    print("TEST: parse_action on 8 hand-made strings")
    print("=" * 60)

    passed = 0
    for raw, expected_ok, desc in test_cases:
        canonical, ok = parse_action(raw)
        status = "PASS" if ok == expected_ok else "FAIL"
        if ok == expected_ok:
            passed += 1
        print(f"{status}: {desc}")
        if ok:
            print(f"       Parsed: {canonical[:60]}")
        print()

    print("=" * 60)
    print(f"Result: {passed}/{len(test_cases)} passed")
    print("=" * 60)


if __name__ == "__main__":
    test_parse_action()
