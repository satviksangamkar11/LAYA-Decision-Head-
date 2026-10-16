# H2 review: Pilot 1 tasks P1-21 to P1-40 (read-only)

Scope: `worker/QUESTIONS.md`, `worker/tasks.jsonl`, `worker/public_cases.json`, `grader/oracles.py` (p21 to p40), `grader/protected_cases.json` (tasks 21 to 40).
Method: reading and Grep slices only. No code was executed. Oracle behaviour was checked by hand-tracing one or more small cases per task. Protected expected values were spot-checked by hand (about 25 checks), not all re-derived. Public cases were compared with `tasks.jsonl`.

## Summary table

| Task | Verdict | Issues |
|---|---|---|
| P1-21 Glob matcher | WEAK | `fnmatchcase` and an unescaped regex translation both pass all 44 cases. No `[` or `.` appears in any case. 37 of 40 protected expect false. |
| P1-22 Max nonoverlapping intervals | OK | None found. |
| P1-23 Decode nested repetition | WEAK | Every case has nesting depth 2 or less. No adjacent top-level groups. A two-pass expander passes all 44. |
| P1-24 Spreadsheet column number | OK | None found. |
| P1-25 Search rotated array | OK | Linear scan also passes (statement does not require log time). Note only. |
| P1-26 RPN | WEAK | No negative divisor in protected, and no case with both operands negative. A sign rule that ignores the divisor's sign passes all 44. |
| P1-27 Lex-smallest topo order | WEAK | Max node id is 9 everywhere, so a string-keyed heap passes. "Lexicographically" is unresolved for ids of 10 or more. |
| P1-28 Grid shortest walk | WEAK | About 32 of 40 protected expect -1 (my count from the listing). Zero-column grid untested. |
| P1-29 Kth largest | OK | None found. |
| P1-30 Days until warmer | OK | None found. Equal-temperature case (81, 81) separates strict from non-strict. |
| P1-31 Levenshtein | OK | None found. |
| P1-32 LIS strict | OK | None found. Public [7,7,7] separates strict from non-strict. |
| P1-33 Word break | OK | None found. |
| P1-34 Restricted regex | WEAK | Only about two protected cases expect true. No empty-text case where a star-only pattern should match. |
| P1-35 Trapped rainwater | OK | None found. |
| P1-36 Dijkstra | OK | None found. |
| P1-37 Min-cost right/down | OK | None found. |
| P1-38 Equal partition | OK | None found. Even-sum-only shortcut fails protected [5,17]. |
| P1-39 Max product subarray | OK | None found. |
| P1-40 SCC | OK | None found. Weak-connectivity counting fails the protected cases. |

BROKEN: none. Statement and oracle agree for all 20 tasks in this scope. No oracle bug was found by hand-tracing.
WEAK: P1-21, P1-23, P1-26, P1-27, P1-28, P1-34.

## Details

### P1-21 Glob matcher: WEAK
1. Statement is clear. "all other chars literal" implies case-sensitive matching, which is reasonable.
2. Oracle is the standard DP over pattern characters. Traced "a?" vs "ab", and the star rule `nd[j]=nd[j-1] or dp[j]`, by hand. Correct.
3. All inputs are legal.
4. Coverage gaps: no `[`, `]`, `.`, or `\` in any pattern or text, although the statement says they are literal. Text never contains `*` or `?`. Only 3 of 40 protected cases expect true (`bc`/`*c*`, `b`/`?`, `bceecbeabea`/`b*a`). Repeated `**` is covered.
5. Passing wrong programs: `fnmatch.fnmatchcase(text, pattern)`. It matches the same glob syntax but treats `[...]` as a set, which violates the statement, and no case exposes that. Also `re.fullmatch(pattern.replace('?','.').replace('*','.*'), text)` without escaping, since no pattern contains `.` or other regex metacharacters.

### P1-22 Max nonoverlapping intervals: OK
1. "touching endpoints allowed" is clear.
2. Oracle sorts by (end, start) and takes `a>=end`. Correct for half-open intervals. Hand-traced `[[1,2],[2,3],[1,3]]` = 2 and the protected `[[-19,-11],[8,17],[20,29],[-9,-4],[-4,4],[2,4],[-12,-8]]` = 5.
3. Start<end held in every case I looked at. I did not check all 40 blocks line by line.
4. Covers touching, duplicates, negatives, empty.
5. None found.

### P1-23 Decode nested repetition: WEAK
1. Clear enough: nested `k[...]`, lowercase atoms, valid syntax.
2. Oracle parser is correct. Hand-traced `2[a2[b]]` = `abbabb`. Hand-traced the `k=0` case.
3. All 40 protected inputs are legal. Expansions are small. Two protected lines were omitted by the display. I read their inputs with a separate Grep: `15[a8[ccdda]]e` (about 616 chars expanded by hand) and `13[a8[ceeac]]` (about 533 chars). Both are well under 2000. I did not read their expected strings.
4. Coverage gaps: maximum nesting depth is 2 everywhere (public and protected). No depth-3 case. No adjacent top-level groups with no literal between them. k of 0 and multi-digit k are covered.
5. Passing wrong program: a non-recursive expander that applies `re.sub(r'(\d+)\[([a-z]*)\]', ...)` exactly twice. It handles depth 2 and passes all 44 cases, but fails depth 3, which the statement allows.
Check: `11[c2[]]ce` expects twelve `c` followed by `e`. A Grep count confirmed the `c{12}e` form, which matches the oracle semantics.

### P1-24 Spreadsheet column number: OK
1. Clear.
2. Oracle is base-26 accumulation. Hand-traced `AA` = 27 and `ZIXU` = 463705.
3. Legal.
4. Public and protected cover single letters, two letters, and up to five letters.
5. None found.

### P1-25 Search rotated distinct sorted array: OK
1. "rotated strictly increasing" is clear. Unrotated arrays are covered as rotations by 0.
2. Oracle is the standard rotated binary search. Hand-traced `[4,5,1,2,3]` target 2 = 3, and `[-7,8]` target 8 = 1.
3. Spot-checked the rotations in about 12 protected arrays. All are legal rotations.
4. Covers empty, single, not-present, first and last index, negatives.
5. A linear scan (`nums.index` when present) is also correct. The statement does not require log time. This is a note, not a defect.

### P1-26 Reverse Polish notation: WEAK
1. Clear.
2. Oracle division `(abs(x)//abs(b)) * sign` truncates toward zero. Correct. Hand-traced `7,-3,/` = -2.
3. Legal. No zero divisor appears.
4. Gaps: the only negative divisor is public `7,-3,/`. Protected has none. No case has both operands negative. Divisions in protected all have a positive divisor.
5. Passing wrong program: `sign = -1 if (x < 0 or b < 0) else 1; result = sign * (abs(x)//abs(b))`. It gets `7,-3,/` right and `-18,14,/` right, but gives `-2` for `-7,-3,/` where the correct answer is `2`. No protected case exposes this.

### P1-27 Lexicographically smallest topological order: WEAK
1. "lexicographically smallest" is standard for integer sequences (numeric comparison). A string-order reading is also possible but unlikely.
2. Oracle is Kahn with a min-heap. Correct. Duplicate edges are deduplicated and counted once. Hand-traced `n=5, [[1,0]]` = `[1,0,2,3,4]`, and `n=8, [[0,5],[0,1],[3,5],[3,4],[4,2]]` = `[0,1,3,4,2,5,6,7]`. Self-loops leave an in-degree that never reaches zero, so the result is `[]`. That fits "[] on a cycle".
3. Legal. Node ids checked are in range 0..n-1.
4. Gap: every node id in public and protected is at most 9 (largest protected n is 10). So numeric and string order never differ. Self-loops are not named in the statement but are covered by the cycle rule.
5. Passing wrong program: Kahn with a heap of `str(node)` values. It passes all 44 because single-digit ids sort the same both ways.

### P1-28 Grid shortest walk: WEAK
1. Clear, including "blocked endpoint".
2. Oracle is BFS with endpoint and empty-grid checks. Correct. Hand-traced `[[0,1],[0,0]]` = 2.
3. Legal. Rectangular grids throughout (spot-checked).
4. Skew: by my count from the listing, 8 of 40 protected expect a non-negative answer and about 32 expect -1. Zero-column grid such as `[[]]` is untested (oracle returns -1; statement's "empty grid" arguably covers it).
5. No all-44 passing wrong program found quickly. Blocked-start and blocked-end cases are covered.

### P1-29 Kth largest with duplicates: OK
1. Clear.
2. `sorted(nums, reverse=True)[k-1]` is correct. Hand-traced `[5,5,1]` k=2 = 5.
3. Legal (k within 1..len checked on the cases I read).
4. Duplicates and negatives covered.
5. None found.

### P1-30 Days until warmer temperature: OK
1. "STRICTLY warmer" is clear.
2. Monotonic stack pops on `x > a[top]`, which is strict. Correct. Hand-traced `[1,2,3]` = `[1,1,0]` and `[66,45,81,...]` first entries.
3. Legal.
4. Equal-temperature cases are present and separate strict from non-strict.
5. None found.

### P1-31 Levenshtein: OK
1. Clear.
2. Standard DP with the three moves. Hand-traced `cde` to `cbac` = 3. Hand-checked `b` to `abbbdaaedd` = 9 by reasoning.
3. Legal.
4. Covers empty strings on both sides, and substitutions.
5. None found.

### P1-32 Longest strictly increasing subsequence: OK
1. Clear.
2. `bisect_left` patience method is correct for strict increase.
3. Legal.
4. Public `[7,7,7]` separates strict from non-strict. Negatives covered.
5. None found.

### P1-33 Word break: OK
1. Clear.
2. DP over prefix positions, correct. Hand-checked `"ccbcccdc"` with its word list (true).
3. Legal (words nonempty).
4. Covers empty text, empty dictionary, reuse of words, and the `aaaa` case.
5. None found.

### P1-34 Simplified regex match: WEAK
1. Clear.
2. Standard DP. Correct. Public `mis*is*p*.` on `mississippi` = false matches the known result.
3. Legal. No pattern starts with `*`. No `**`.
4. Gaps: only about two protected cases expect true (`ebabaac`/`..*ca*` and `ccb`/`.*.*`). No empty-text case with a star-only pattern expecting true. A `+`-style star bug is caught by `ebabaac`, so that part is covered.
5. No all-44 passing program found. This is a coverage gap, not a found bug.

### P1-35 Trapped rainwater: OK
1. Clear.
2. Two-pointer algorithm, correct. Hand-traced `[15,12,17,17]` = 3, and `[1,12,9,4,10,15,19,19,14,0,13,14]` = 28.
3. Legal.
4. Empty, single, monotone, and large cases.
5. None found.

### P1-36 Dijkstra: OK
1. Clear. Unreachable gives null.
2. Stale-entry check `if k != dist[u]` is correct. Hand-traced `[[0,1,2],[1,2,3],[0,2,10]]` = `[0,2,5]` and `n=6, [[1,3,13],[2,3,6],[1,5,3],[5,1,3],[3,0,6],[1,5,12]], source 1` = `[19,0,null,13,null,3]`.
3. Legal. Weights nonnegative. Self-loops and zero weights present.
4. Covers unreachable, zero weights, multi-edges, self-loops.
5. BFS ignoring weights fails public. None other found.

### P1-37 Min-cost right/down grid: OK
1. Clear.
2. DP is correct. Hand-traced `[[4,2,0,5,4],[2,3,5,2,5],[3,5,4,4,4]]` = 21 (the protected expected value), and `[[0,5,-1],[3,3,0]]` = 6.
3. Legal. Only -1 and nonnegative values appear.
4. Covers obstacles at start and end, single cell, obstacles forcing unreachability.
5. None found.

### P1-38 Equal subset partition: OK
1. Clear.
2. Subset-sum set with bound `s//2` is correct. Hand-traced `[1,5,11,5]` (true, 11 reached) and `[7,0,18,15,10,15,9,16,6,10]` (true, 53 = 18+15+10+10).
3. Legal.
4. Odd sums, zeros, empty covered.
5. Even-sum-only shortcut fails protected `[5,17]`, which is false. Not a passing program.

### P1-39 Maximum product subarray: OK
1. Clear.
2. Min/max tracking is correct. Hand-traced `[5,-2,-6,4,1,-2,7,3]` = 1008 (the protected expected value), `[3,0,4,2,-4]` = 8, and `[1,-7,1,-4,0,3,1,1,-4,7,6]` = 42.
3. Legal. Nonempty.
4. Covers zeros, all-negative, even negative count.
5. Plain Kadane on sums fails public `[-2,3,-4]` = 24.

### P1-40 Strongly connected components: OK
1. Clear.
2. Kosaraju with reversed-finish order. Correct. Hand-traced public `n=3, [[0,1],[1,0],[1,2]]` = 2.
3. Legal (checked node ids in range on the cases I read).
4. Covers isolated nodes, self-loops, duplicate edges, n=0 and n=1.
5. Counting weakly connected components fails protected `n=2, [[0,1],[0,1],[0,0],[0,0]]` (expected 2).

## Public cases
`worker/public_cases.json` matches the `public_tests` in `tasks.jsonl` for all 20 tasks in this scope. Every public case is inside the stated domain.

## Protected cases: method and limits
- Read through Grep slices of the single-line JSON. Blocks for all 20 tasks were read.
- Exact per-block counts were verified only for P1-21 (40 entries). Other blocks look like about 40 each, which is what `protected_count` says. Not counted exhaustively.
- Two P1-23 lines were not displayed in full. Their inputs were read separately; expected strings were not read.
- No protected case was found outside its stated input domain in the places I looked.

## Checklist
- [x] Read QUESTIONS.md and tasks.jsonl for P1-21 to P1-40
- [x] Read oracles.py p21 to p40
- [x] Compared public_cases.json with tasks.jsonl (all 20 match)
- [x] Read protected_cases.json for all 20 tasks (Grep slices)
- [x] Hand-traced at least one oracle case per task
- [x] Checked legal-domain preconditions in the protected cases I read
- [ ] Not done: executing the oracles or the protected cases (not allowed in this review)
- [ ] Not done: exact count of every protected block (only P1-21 verified)
- [ ] Not done: full hand re-derivation of every protected expected value (about 25 spot checks)
- [x] Report written to review/H2_tasks_21_40.md
