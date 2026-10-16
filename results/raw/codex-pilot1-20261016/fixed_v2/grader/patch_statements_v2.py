"""v2: clarify ambiguous statements (QUESTIONS.md and tasks.jsonl prompts). Run once; trusted tooling."""
import json, re, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
ADD = {
 "P1-10": " Unicode whitespace means characters for which Python's str.isspace() is true.",
 "P1-14": " The empty string may appear as a word; all empty strings form one group.",
 "P1-21": " Characters such as [ ] . \ ^ + ( ) are ordinary literals, not special.",
 "P1-23": " Nesting may be deeper than two levels, and k may have several digits or be 0.",
 "P1-26": " Operands may be negative; division truncates toward zero also for negative operands.",
 "P1-27": " Order is numeric by node id (node 10 comes after node 9).",
 "P1-28": " An empty grid is [] or a grid whose rows are empty ([[]]).",
}
q = root/"worker"/"QUESTIONS.md"; lines = q.read_text(encoding="utf-8").split("\n"); cur=None; n=0
for i,l in enumerate(lines):
    m = re.match(r"## (P1-\d\d) ", l)
    if m: cur = m.group(1); continue
    if cur in ADD and l.startswith("Input") and ADD[cur] not in l:
        lines[i] = l + ADD[cur]; n += 1
q.write_text("\n".join(lines), encoding="utf-8")
t = root/"worker"/"tasks.jsonl"; out=[]; k=0
for l in t.read_text(encoding="utf-8").splitlines():
    if not l.strip(): continue
    d = json.loads(l)
    if d["task_id"] in ADD and ADD[d["task_id"]] not in d["prompt"]:
        d["prompt"] += ADD[d["task_id"]]; k += 1
    out.append(json.dumps(d, ensure_ascii=False))
t.write_text("\n".join(out)+"\n", encoding="utf-8")
print({"questions_md_patched": n, "tasks_jsonl_patched": k})
