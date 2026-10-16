# Solution format and public tests

For every task write one file named `P1-01.py` ... `P1-40.py`, containing:

```python
def solve(data):
    # return a JSON-serializable Python value
    pass
```

The `data` input matches the task's specified dictionary. Example inputs and outputs are in `tasks.jsonl` and `public_cases.json`. Do not put grading references or protected cases in the worker notebook. You may implement a worker wrapper that selects one task, gives the model its prompt and PUBLIC tests, and saves candidate code under `submissions/<arm>/P1-XX.py`.

Arms:
- A: Qwen + tools, first attempt.
- B: Qwen with extra attempts under matched generated-token budget.
- C: Qwen with public-test execution feedback and repair.
- D0: C + Evidence-Ladder controller, no GLM.
- D1: D0 + GLM-5.3-Flash critic that proposes executable discriminating tests.
- E: D1 + existing adapter/LAYA as OUT-OF-DOMAIN transfer test ONLY. E is not an acceptance gate for LAYA.

Critical: the kit DOES NOT include Qwen/GLM/adapter model inference. It supplies question inputs, observable outputs, grader and comparison tools, NOT a working implementation of arms C/D0/D1/E. Do not label a simple repeated Qwen run as an Evidence-Ladder arm.

All output folders MUST be paired by task. Log budget and critic usage per arm separately. Keep failures and timeouts, not just successful submissions.
