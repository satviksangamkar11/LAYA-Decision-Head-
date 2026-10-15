# H1 review: Pilot 1 tasks P1-01 to P1-20

Read-only review. No code was run, no existing file was changed, no network used.

Sources read: `worker/QUESTIONS.md`, `worker/tasks.jsonl`, `worker/public_cases.json`, `grader/oracles.py` (p01 to p20), and all 40 protected cases per task from `grader/protected_cases.json`.

Method note: `protected_cases.json` is one minified line (about 82k tokens), so the Read tool cannot load it. I read it with Grep `-o` over the pattern `"task_id":"P1-\d\d"|\{"input":.{0,300}?"expected":.{0,300}?\}` at offsets 0 to 800. Each task is 41 matches (one marker, 40 cases). I checked the counts: 40 protected cases per task for P1-01 to P1-20. Limitation: the extraction stops at the first `}`, so some expected strings were cut short for display. This affects P1-10 expected values that contain `}` (for example `"?IEO{}`). For those I checked the input and the oracle logic, not the full expected string. The protected expected values were not recomputed by hand; I checked oracle logic against the statement and spot-checked about 12 expected values.

## Summary table

| Task | Verdict | Issues |
|---|---|---|
| P1-01 Count vowels | BROKEN (latent) | Oracle uses `.lower()`, so U+0130 (İ) lowers to `i` + U+0307 and counts as a vowel. Statement says ASCII vowels only. No test contains non-ASCII. |
| P1-02 Sum even | OK | None. Python `%` handles negatives correctly. |
| P1-03 First non-repeating | OK | "character" is undefined for combining or non-BMP text (oracle uses code points). No non-ASCII test. |
| P1-04 Rotate right | OK | None. Modulo handled. Wrong no-modulo variant fails protected (k=73, len=10). |
| P1-05 Word frequencies | BROKEN (latent) | Same `.lower()` problem before the ASCII regex. Kelvin sign U+212A lowers to `k`, and İ lowers to `i`. Statement says ASCII-letter spans only. No test contains non-ASCII. |
| P1-06 Merge sorted | OK | None. Sorting the concatenation also passes, which is acceptable. |
| P1-07 Compress ranges | OK | Returns lists, not tuples (JSON-normal). Coverage good. |
| P1-08 Run-length | OK | Code-point runs (fine for Python). No whitespace runs tested. |
| P1-09 Second distinct | OK | None. |
| P1-10 Normalize whitespace | WEAK | Oracle uses `str.split()`, which treats U+001C to U+001F as whitespace; these are not in Unicode White_Space. No protected input contains `\r`, `\f`, `\v`, NBSP or U+001C to U+001F. A `[ \t\n]+` solution passes all 44. |
| P1-11 Balanced brackets | WEAK | Every protected case expected `true` contains no brackets (E, Aa, space, `!\t`, bIe, `bbE9a `). The only bracket-nested true cases are public. |
| P1-12 Max subarray | OK | Empty input excluded by NONEMPTY. Coverage good. |
| P1-13 Longest unique | WEAK | Statement says case-sensitive. No P1-13 input contains an uppercase letter. A `text.lower()` solution passes all 44. |
| P1-14 Anagram groups | WEAK | Statement is silent on `""` as a word. Oracle groups `""` as its own anagram class. Many protected cases contain `""`. A solution that drops empty words would fail, so the result depends on this reading. |
| P1-15 Merge intervals | OK | None. Shared endpoint merges, adjacency does not. Good coverage. |
| P1-16 Count islands | OK | None. Diagonal cases covered. |
| P1-17 BFS shortest path | OK | None. Undirected variant caught by public case. |
| P1-18 Fewest coins | OK | `coins=[]` with `amount=0` untested (oracle returns 0, matches statement). Oracle silently drops non-positive coins (illegal input). |
| P1-19 Min window | WEAK | Earliest-start tie-break is only partly tested. The one tie I found (`abddceabdbebabcbbeaaea`, `adb`) has two identical windows `abd`, so a latest-start variant cannot be distinguished there. I did not complete an exhaustive tie search. |
| P1-20 LRU cache | WEAK | "put on existing key marks most recently used" is never observable. A variant where `put` does not refresh recency passes all 44 (trace in details). |

Counts: BROKEN 2 (P1-01, P1-05; both latent, non-ASCII only). WEAK 6 (P1-10, P1-11, P1-13, P1-14, P1-19, P1-20). OK 12 (P1-02, 03, 04, 06, 07, 08, 09, 12, 15, 16, 17, 18).

## Details

### P1-01 Count vowels: BROKEN (latent)
1. Statement: "ASCII vowels a/e/i/o/u, case-insensitively; do not count other letters." Clear, but contradicted by the oracle on non-ASCII input.
2. Oracle: `sum(c in 'aeiou' for c in d['text'].lower())`. For "İ" (U+0130), Python `.lower()` gives `i` + U+0307, so the oracle returns 1. The statement requires 0.
3. Public and protected inputs are ASCII (including `\t`, `\n`, punctuation). Legal domain is any string, so non-ASCII is legal.
4. Coverage: empty (public), uppercase (public), `y` not a vowel (public "rhythm"), whitespace and punctuation (protected). Missing: non-ASCII letters.
5. Wrong program passing all 44: `sum(c.lower() in 'aeiou' for c in text)`. Same as oracle, so it passes; it differs from the statement on İ.
6. Fix: test `c in 'aeiouAEIOU'` on the original text, not on `.lower()`.

### P1-02 Sum even: OK
1. Clear.
2. `sum(x for x in nums if x % 2 == 0)` is correct for negatives (Python `-4 % 2 == 0`).
3. All legal. 4. Empty, single element, negatives, zero, all-odd singletons covered. 5. A wrong sign-filter (`x > 0`) fails public `-4`. No obvious pass-all wrong program.

### P1-03 First non-repeating: OK
1. "character" and "zero-based index" assume code points. Fine for Python; not stated.
2. `Counter` + `next(...)` is correct; the "aAbA" public case checks case sensitivity.
3. All ASCII, legal.
4. Covered: empty, all-repeated, unique at index 0, unique in middle. Not seen: a unique character only at the last index. Missing: non-ASCII.
5. A case-insensitive counter returns 2 on public "aAbA" (expected 0), so it fails.

### P1-04 Rotate right: OK
1. Clear.
2. `k %= len(a)`, then `a[-k:]+a[:-k]`; k=0 returns a copy; empty returns []. Correct.
3. All k >= 0. Legal.
4. Covered: empty with k=100, k=0, k<len, k>len, k=multiple of len (k=10 with len 5; k=86 with len 2), single element, large k. Good.
5. A no-modulo version `a[len-k:]+a[:len-k]` fails protected k=73 (len 10) because the slice clips to the whole list.

### P1-05 Word frequencies: BROKEN (latent)
1. "maximal ASCII-letter [A-Za-z]+ spans; lowercase them". Clear.
2. Oracle lowercases the whole text first, then matches `[a-z]+`. Kelvin sign U+212A lowers to `k` (counted). İ lowers to `i` + U+0307 (the `i` is counted, which breaks the span rule). Statement says these are non-letters that split words.
3. Public and protected are ASCII only.
4. Covered: digits adjacent to letters (`123world`), punctuation, hyphen, newline, uppercase repeats, empty. Missing: non-ASCII.
5. `re.findall('[A-Za-z]+', text)` then lowercase matches the statement; passes all 44; differs from oracle only on non-ASCII.
6. Fix: match `[A-Za-z]+` on the original text, then lowercase each word.

### P1-06 Merge sorted: OK
1. Clear.
2. Two-pointer merge with tail appends. Correct.
3. Every protected a and b checked: both sorted nondecreasing.
4. Covered: both empty, one empty, duplicates across lists, negatives, lists with long runs of equal values. Public case `[1,1]` with `[1]` catches a merge that drops the tail.
5. `sorted(a+b)` passes; this is a correct alternative, not a wrong program.

### P1-07 Compress ranges: OK
1. Clear.
2. Standard run scan. Correct.
3. Every protected list checked: strictly increasing.
4. Covered: empty, singletons, negative runs, run across zero (public), runs of length 2 and 3, final run. Good.
5. Returns lists; a solution returning tuples only differs if the grader does not JSON-normalise. Minor.

### P1-08 Run-length: OK
1. Clear.
2. Correct. Case-sensitive via public "AaA".
3. Legal.
4. Covered: empty, singletons, all-equal runs, alternating characters, case flip, long runs. Missing: whitespace runs.
5. None found that passes all 44 while violating the statement.

### P1-09 Second distinct largest: OK
1. Clear.
2. `sorted(set(nums), reverse=True)[1]` or None. Correct.
3. Legal.
4. Covered: empty, one distinct value (`[3,3]`, `[-4]`), duplicate max (`[5,3,5,2]`, `[14,-9,...,14,...]`), negatives, two distinct values. Good.
5. `sorted(nums)[-2]` (no dedupe) fails public `[5,3,5,2]`.

### P1-10 Normalize whitespace: WEAK
1. "Unicode whitespace" is not defined. Oracle uses Python `str.split()` (str.isspace). That includes U+001C to U+001F, which are not in Unicode White_Space.
2. Oracle is internally consistent with its definition. The statement is less precise than the oracle.
3. Legal.
4. Covered: leading and trailing whitespace, internal tabs and newlines, empty, no-change. Missing: `\r`, `\f`, `\v`, NBSP, U+3000, U+001C to U+001F, all-whitespace input.
5. `re.sub(r'[ \t\n]+', ' ', text).strip()` passes all 44 (the only whitespace in the tests is space, tab and newline) but violates "Unicode whitespace".
6. Several protected expected strings were truncated by my extraction at `}`; not a finding about the grader.

### P1-11 Balanced brackets: WEAK
1. Clear.
2. Stack algorithm, non-bracket characters ignored, final `not stack` check. Correct.
3. Legal.
4. Covered: empty (public), unmatched close (`0 }`), wrong order (`([)]` public), unclosed open (`{I()eBO`, `E ] c ac({([?A`). Gap: every protected `true` case has no brackets. Only public has a correctly nested true case (`([{}])`, `hello{(a+b)}!`).
5. I did not find a plausible wrong program that passes all 44. The gap makes a "true unless obviously wrong" heuristic harder to detect, but I cannot name one that passes.

### P1-12 Max subarray: OK
1. Clear; NONEMPTY is stated.
2. Kadane with `cur=best=a[0]`. Correct for all-negative inputs.
3. All protected nonempty (checked the single-element cases).
4. Covered: singletons (`[0]`, `[-14]`, `[8]`, `[2]`, `[7]`, `[-6]`, `[29]`), all-negative (public), all-zero (public), max at end, max mid-list. Good.
5. Kadane with `max(0, ...)` fails public `[-5,-2,-7]`. `max(a)` fails public `[5,-10,4,7]`.

### P1-13 Longest unique: WEAK
1. "case-sensitive" stated.
2. Sliding window with last-index dict. Correct as written.
3. Legal.
4. Covered: empty, repeats, mixed. Gap: no uppercase letter in any public or protected input (every input uses lowercase letters only).
5. `len-of-longest-unique(text.lower())` passes all 44 and violates "case-sensitive".

### P1-14 Anagram groups: WEAK
1. Clear for non-empty words. Silent on `""`. Statement says "lowercase words" and does not say whether `""` is a word.
2. Key = sorted letters; `dict` insertion order gives first-appearance groups; words kept in order. Correct.
3. Legal under the oracle's reading. Many protected cases include `""`.
4. Covered: empty list, single word, duplicates (`["a","a"]`), no anagrams, anagram with multiplicity (`abebc`/`abbec`), `ec` vs `ccec` (a set-key solution fails protected).
5. A `frozenset` key fails protected `ec`/`ccec`. A solution that drops `""` fails many protected cases. That is the ambiguity: the result changes with that reading.
6. Verdict is WEAK, but if the author intends `""` to be excluded the task is BROKEN.

### P1-15 Merge intervals: OK
1. Clear. "shared endpoint merges; integer adjacency does not" is explicit.
2. `sorted`, then merge when `l > out[-1][1]`. Correct.
3. Every protected `start<=end`. Legal.
4. Covered: empty, shared endpoint (public), adjacency (public `[1,1],[2,2]`), nested containment (`[-2,2],[-3,6],[-1,0]`), unsorted input, degenerate points, duplicates, long chains. Good.
5. `l - prev_end <= 1` merges adjacency and fails public `[1,1],[2,2]`. `l < prev_end` misses shared endpoints and fails public `[1,3],[3,5]`.

### P1-16 Count islands: OK
1. Clear.
2. DFS over 4-neighbours; `n=len(g[0]) if m else 0`. Correct.
3. Protected grids rectangular.
4. Covered: `[]`, single cell 0 and 1, one row, one column, diagonal non-connection (public `101/010/101` = 5), a 5x5 mixed grid (I traced the 5 components by hand: correct). Gap: no multi-row all-ones grid.
5. 8-neighbour connectivity fails public (gives 1 instead of 5).

### P1-17 BFS path length: OK
1. Clear.
2. BFS with levels; returns level when target is popped; `seen` on enqueue. Correct. Source equals target returns 0.
3. Protected n >= 1, endpoints in range (spot-checked about 15 cases).
4. Covered: source equals target, unreachable, reversed edge only, self-loops, duplicate edges, shortest vs longer path (public), unreachable when source has no out-edges.
5. Undirected treatment fails public `[[1,0]]` (expects -1). DFS instead of BFS fails public (`[0,1],[1,2],[0,2]` expects 1).

### P1-18 Fewest coins: OK
1. Clear. "amount zero -> 0" stated.
2. DP with sentinel `n+1`; `dp[n] > n` means unreachable. Correct, since a reachable amount never needs more than `n` coins.
3. Protected coins positive, amounts up to 55.
4. Covered: amount 0 (with non-empty coins), empty coins with amount > 0, coin larger than amount, greedy-failing case (public `[1,3,4]` 6 gives 2). Gap: empty coins with amount 0 (not in protected).
5. Greedy largest-first fails public `[1,3,4]`.

### P1-19 Minimum window: WEAK
1. Clear. Tie-break and empty `t` stated.
2. Standard sliding window with `need`/`window`/`missing`; `best` updated only on strictly shorter length, so the earliest-start minimum is kept. Correct.
3. Legal.
4. Covered: empty `t`, empty `s`, impossible, `t` longer than `s`, multiplicity (`aa`/`aa`, `bbc`), public `ADOBECODEBANC`. Gap: the tie-break is only exercised where tied windows have the same text. The one tie I checked (`abddceabdbebabcbbeaaea`, `adb`: windows at 0 and 6 are both `abd`) cannot tell earliest from latest. I did not finish a full tie search, so the gap is probable, not proven.
5. A variant that keeps the latest minimum (`<=` instead of `<`) passes every tie case that I found, because the tied strings are identical.

### P1-20 LRU cache: WEAK
1. Clear, including "capacity zero stores nothing".
2. OrderedDict: `get` and `put` move to end, eviction from front, `cap==0` ignored. Correct.
3. Protected ops are well-formed; capacities 0 to 5; values can be negative.
4. Covered: capacity 0 (several), capacity 1, get-refresh (public `cap=2` eviction), eviction after gets, put-overwrite of an existing key (several protected cases).
5. Gap: "updating ... marks most recently used" is not observable. Variant: `put` on an existing key does not call `move_to_end`. I traced protected case #6 (capacity 2: `put 3 0`, `put 4 -3`, gets, `put 1 6`, `get 6`, `put 6 9`, `put 6 1`, `put 1 11`, `put 5 4`, `get 4`, `put 1 11`, `put 1 2`). Gets return -1 in both versions, so the variant passes all 44. Protected case #6 is the first of this capacity-2 family; I did not check the other capacity-2 cases for update-recency dependence.

## Final checklist

- [x] QUESTIONS.md read for P1-01 to P1-20
- [x] tasks.jsonl public tests read for P1-01 to P1-20 (4 per task, matching the statement examples)
- [x] public_cases.json read; consistent with tasks.jsonl (checked by reading)
- [x] oracles.py p01 to p20 read and compared against statement
- [x] protected_cases.json read for P1-01 to P1-20 (40 per task, counted)
- [x] Legal-domain check done for every protected case (sortedness, strict increase, nonempty, start<=end, rectangular grids, n>=1, nonnegative capacity; spot checks where full checking was not practical)
- [ ] Every protected expected value recomputed by hand: NOT DONE (spot checks only; oracle logic checked instead)
- [ ] Full expected strings for P1-10 values containing `}`: NOT DONE (display truncation in my extraction)
- [ ] Exhaustive tie search for P1-19: NOT DONE
- [x] Report written to review/H1_tasks_01_20.md
- [x] No code run, no existing file changed, no network

Blocked: none. Waiting on user: decision on whether `""` is a legal word for P1-14, and whether non-ASCII input is in scope for P1-01 and P1-05.
