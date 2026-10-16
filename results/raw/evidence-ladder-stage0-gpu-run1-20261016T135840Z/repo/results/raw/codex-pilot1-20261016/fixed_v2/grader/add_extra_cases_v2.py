"""v2: (1) recompute EVERY existing protected and public expected value from the (fixed) oracles and report mismatches;
(2) append targeted cases for the gaps found in the 2026-10-16 review (results/raw/codex-pilot1-20261016/review/).
Expected values always come from the trusted oracles. Writes protected_cases.json (original kept as protected_cases.orig.json).
Trusted tooling: no candidate code is executed here."""
import json, shutil, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from oracles import ORACLES

P = here / "protected_cases.json"
orig = here / "protected_cases.orig.json"
if not orig.exists():
    shutil.copy(P, orig)
cases = json.loads(orig.read_text(encoding="utf-8"))
by = {t["task_id"]: t["cases"] for t in cases}

# ---- (1) recompute all existing expected values ----------------------------------------------------------
mismatch = []
for t in cases:
    key = t["task_id"][-2:]
    for i, c in enumerate(t["cases"]):
        got = ORACLES[key](json.loads(json.dumps(c["input"])))
        if got != c["expected"] or type(got) is not type(c["expected"]):
            mismatch.append((t["task_id"], i, c["expected"], got))
pub = json.loads((here.parent / "worker" / "public_cases.json").read_text(encoding="utf-8"))
pub_mismatch = []
for t in pub:
    key = t["task_id"][-2:]
    for i, c in enumerate(t["cases"]):
        got = ORACLES[key](json.loads(json.dumps(c["input"])))
        if got != c["expected"] or type(got) is not type(c["expected"]):
            pub_mismatch.append((t["task_id"], i, c["expected"], got))

# ---- (2) targeted extra inputs ---------------------------------------------------------------------------
E = {}
E["P1-01"] = [{"text": s} for s in ["İ", "İstanbul", "Iİiı", "Kelvin", "café àéî", "AEIOUaeiou", "xyz\n\t"]]
E["P1-05"] = [{"text": s} for s in ["Kelvin K", "KELVIN k", "İstanbul ISTANBUL", "café é", "A-a_a1a", "x" * 3 + "ß"]]
E["P1-10"] = [{"text": s} for s in [
    " a  b　c ", "a\x1cb", " \t\n\r\x0b\x0cx", "​z", "a\u0085b", "  ", "a   b", "x   y ", "  mid  dle  "]]
E["P1-11"] = [{"text": s} for s in ["()", "([]{})", "a(b[c]{d})e", "{[()]}", "((()))", "()[]{}", "([)]", "((", "}{", "(]", "[({})]", "", "x", "(a)(b)[c]{d}", "{[}]", "(((((((())))))))", "([{}])([{}])", ")("]]
E["P1-13"] = [{"text": s} for s in ["aAbB", "abcabcbb", "Aa", "aa", "abBA", "pwwkew", "tmmzuxt", "dvdf", "abba", "a", "AaBbCc", "abcdeafgh"]]
E["P1-14"] = [{"words": w} for w in [
    ["", "a", ""], ["listen", "silent", "enlist", "google", "gogole", "a", "b", "a"], ["ab", "ba", "abc", "cba", "bca", "ab"],
    ["", "", ""], ["aab", "aba", "baa", "abb"], ["a"], [], ["abc", "bca", "cab", "abd", "bad", "dab"]]]
E["P1-19"] = [{"s": s, "t": t} for s, t in [
    ("ADOBECODEBANC", "ABC"), ("bbaa", "ab"), ("abab", "ab"), ("baab", "ab"), ("aab", "aab"), ("abcbac", "abc"), ("xabcabcx", "cab"),
    ("a", "aa"), ("aa", "aa"), ("abcabc", "cba"), ("cabwefgewcwaefgcf", "cae"), ("bba", "ab"), ("abca", "aa"), ("zzabzz", "ba")]]
E["P1-20"] = [
    {"capacity": 2, "ops": [["put", 1, 1], ["put", 2, 2], ["put", 1, 10], ["put", 3, 3], ["get", 1], ["get", 2], ["get", 3]]},
    {"capacity": 2, "ops": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["get", 1], ["get", 3]]},
    {"capacity": 1, "ops": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["get", 2]]},
    {"capacity": 0, "ops": [["put", 1, 1], ["get", 1]]},
    {"capacity": 2, "ops": [["put", 1, 1], ["put", 2, 2], ["put", 2, 20], ["put", 3, 3], ["get", 1], ["get", 2], ["get", 3]]},
    {"capacity": 3, "ops": [["put", 1, 1], ["put", 2, 2], ["put", 3, 3], ["get", 1], ["put", 4, 4], ["get", 2], ["get", 3], ["get", 1], ["get", 4]]},
    {"capacity": 2, "ops": [["get", 5], ["put", 5, 0], ["get", 5], ["put", 5, -1], ["get", 5]]}]
E["P1-21"] = [{"text": a, "pattern": b} for a, b in [
    ("a[b]", "a[b]"), ("ab", "a[b]"), ("a.b", "a.b"), ("axb", "a.b"), ("a\\b", "a\\b"), ("a+b", "a+b"), ("aab", "a+b"), ("a^b", "a^b"),
    ("(x)", "(x)"), ("x", "(x)"), ("abcdef", "a*f"), ("", "*"), ("", "?"), ("aaa", "a*a"), ("abc", "*?*"), ("ab", "a**b"), ("a", "[a]"),
    ("[a]", "[a]"), ("a]", "a]"), ("x.y", "x?y"), ("abc", "a?c"), ("ac", "a?c"), ("abbbc", "a*c"), ("abc", "a*b"), ("mississippi", "m*iss*pi"), ("mississippi", "m*iss?pi")]]
E["P1-23"] = [{"text": s} for s in [
    "2[a2[b2[c]]]", "3[a]2[bc]", "10[a]", "0[abc]d", "2[2[2[2[a]]]]", "a2[b]c", "12[x]", "1[2[3[a]]]", "2[ab3[c]]d", "3[2[a1[b]]]", "", "abc", "2[0[x]y]", "1[1[1[1[z]]]]"]]
E["P1-26"] = [{"tokens": t} for t in [
    ["7", "-3", "/"], ["-7", "3", "/"], ["-7", "-3", "/"], ["-7", "2", "/"], ["0", "-5", "/"], ["3", "-4", "*", "5", "+"],
    ["2", "1", "+", "3", "*"], ["4", "13", "5", "/", "+"], ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],
    ["-1"], ["5", "-8", "-"], ["-9", "-2", "/"], ["1", "-2", "/"], ["-6", "4", "/"]]]
E["P1-27"] = [
    {"n": 12, "edges": [[10, 2]]}, {"n": 12, "edges": [[11, 0], [10, 1]]}, {"n": 11, "edges": [[10, 0]]}, {"n": 0, "edges": []},
    {"n": 3, "edges": [[1, 1]]}, {"n": 4, "edges": [[0, 1], [0, 1], [1, 2], [2, 3], [3, 1]]}, {"n": 13, "edges": [[12, 3], [10, 3], [3, 11]]},
    {"n": 5, "edges": [[4, 3], [3, 2], [2, 1], [1, 0]]}, {"n": 12, "edges": [[9, 10], [10, 2], [2, 11]]}]
E["P1-28"] = [{"grid": g} for g in [
    [[]], [], [[0]], [[1]], [[0, 0], [0, 0]], [[0, 0, 0]], [[0], [0], [0]], [[0, 1], [1, 0]], [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], [[0] * 6 for _ in range(6)],
    [[0, 0, 0], [1, 1, 0], [0, 0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 1]],
    [[0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]], [[0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]]]]
E["P1-34"] = [{"text": a, "pattern": b} for a, b in [
    ("", "a*"), ("", "a*b*c*"), ("aaa", "a*a"), ("ab", ".*"), ("aab", "c*a*b"), ("mississippi", "mis*is*p*."), ("mississippi", "mis*is*ip*."),
    ("abc", "a.c"), ("", ".*"), ("a", "ab*"), ("ab", ".*c"), ("aaa", "ab*a*c*a"), ("a", ".*..a*"), ("ab", ".*.."), ("abcd", "d*"),
    ("aaa", "a*a*a*"), ("bbbba", ".*a*a"), ("aa", "a"), ("aaa", "ab*ac*a"), ("", ""), ("a", ""), ("", "a")]]

added = {}
for tid, inputs in E.items():
    key = tid[-2:]
    have = {json.dumps(c["input"], sort_keys=True) for c in by[tid]}
    n = 0
    for inp in inputs:
        s = json.dumps(inp, sort_keys=True)
        if s in have:
            continue
        have.add(s)
        by[tid].append({"input": inp, "expected": ORACLES[key](json.loads(s))})
        n += 1
    added[tid] = n
out = [{"task_id": t["task_id"], "cases": by[t["task_id"]]} for t in cases]
P.write_text(json.dumps(out, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
print(json.dumps({"existing_mismatches_vs_oracle": mismatch, "public_mismatches_vs_oracle": pub_mismatch, "cases_added": added,
                  "total_cases": sum(len(t["cases"]) for t in out)}, indent=1))
