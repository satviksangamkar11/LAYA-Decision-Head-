"""Second versioned sampling template for architecture stage 3 (>= 2 prompt strategies). qwen_sample_v1 / v2 in qwen.py are untouched.
v2b = v2 with the same task, examples and solution-format blocks; only the closing instruction differs (a different solving strategy)."""
from evidence_ladder.common.interfaces import sha256_text
from evidence_ladder.models import qwen as Q

TEMPLATE_ID = "qwen_sample_v2b"
TEMPLATE = (
    "{task_prompt}\n\n"
    "Public examples (input and expected output):\n{examples}\n\n"
    "Required solution format:\n{solution_format}\n\n"
    "In your reasoning, first list the edge cases (empty or minimal inputs, boundary values, duplicates, large inputs) and work out a simple brute-force "
    "way to check the public examples by hand. Then derive the efficient solution and check it against both. "
    "Then give your final complete solution in {language} as ONE fenced code block. The code block must be the last thing in your answer."
)
TEMPLATE_HASH = sha256_text(TEMPLATE)
TEMPLATES_BY_SAMPLE = ("qwen_sample_v2", "qwen_sample_v2", TEMPLATE_ID, TEMPLATE_ID)   # sample_idx 0-1 -> v2, 2-3 -> v2b


def render(template_id: str, **fields):
    """-> (prompt, sha256). v2 is delegated to qwen.render so its hash equals the earlier FC-3 / GPU-gen runs."""
    if template_id == "qwen_sample_v2":
        return Q.render(template_id, **fields)
    assert template_id == TEMPLATE_ID, template_id
    need = {"task_prompt", "examples", "solution_format", "language"}
    if set(fields) != need:
        raise ValueError("template %s needs fields %s, got %s" % (template_id, sorted(need), sorted(fields)))
    p = TEMPLATE.format(**fields)
    return p, sha256_text(p)
