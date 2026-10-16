# Pilot 1: Forty Python coding tasks

This is an **original, assistant-curated DEVELOPMENT PILOT**. It is **not** a published benchmark, not freshly released after 2025, and not a sealed independent holdout. Difficulty buckets are qualitative labels, not empirically calibrated.

**Submit one Python function** named `solve(data)` per task. Input is a JSON object; the return value must be JSON-serializable. Python standard library is allowed. Do not read or write files, start processes, or access network.

## P1-01 — Count vowels (easy)
Input {"text": string}. Return number of ASCII vowels a/e/i/o/u, case-insensitively; do not count other letters.
Example:
```json
{"input": {"text": ""}, "expected": 0}
```

## P1-02 — Sum even integers (easy)
Input {"nums": list of integers}. Return sum of elements divisible by 2, including negative even integers; empty list gives 0.
Example:
```json
{"input": {"nums": []}, "expected": 0}
```

## P1-03 — First non-repeating character (easy)
Input {"text": string}. Return zero-based index of the first character occurring exactly once (case-sensitive), or -1.
Example:
```json
{"input": {"text": ""}, "expected": -1}
```

## P1-04 — Rotate array right (easy)
Input {"nums": list of integers, "k": nonnegative int}. Rotate right by k positions; preserve elements; empty list returns [].
Example:
```json
{"input": {"nums": [], "k": 100}, "expected": []}
```

## P1-05 — Word frequencies (easy)
Input {"text": string}. Words are maximal ASCII-letter [A-Za-z]+ spans; lowercase them and return JSON object mapping words to counts. Ignore punctuation/digits.
Example:
```json
{"input": {"text": ""}, "expected": {}}
```

## P1-06 — Merge sorted arrays (easy)
Input {"a": sorted integer list, "b": sorted integer list}. Return one nondecreasing merged list, retaining duplicates.
Example:
```json
{"input": {"a": [], "b": []}, "expected": []}
```

## P1-07 — Compress consecutive ranges (easy)
Input {"nums": strictly increasing integer list}. Return maximal consecutive runs as [start,end] pairs, inclusive.
Example:
```json
{"input": {"nums": []}, "expected": []}
```

## P1-08 — Run-length encoding (easy)
Input {"text": string}. Return sequential [character,count] pairs for maximal adjacent equal-character runs; empty string -> [].
Example:
```json
{"input": {"text": ""}, "expected": []}
```

## P1-09 — Second distinct largest (easy)
Input {"nums": list of integers}. Return the second-largest DISTINCT integer, or JSON null if fewer than two distinct values.
Example:
```json
{"input": {"nums": []}, "expected": null}
```

## P1-10 — Normalize whitespace (easy)
Input {"text": string}. Strip leading/trailing whitespace and replace each internal run of Unicode whitespace with one ASCII space.
Example:
```json
{"input": {"text": ""}, "expected": ""}
```

## P1-11 — Balanced brackets (medium)
Input {"text": string}. Check (), [], {} nesting and pairing; ignore all non-bracket characters. Return boolean.
Example:
```json
{"input": {"text": ""}, "expected": true}
```

## P1-12 — Maximum nonempty subarray sum (medium)
Input {"nums": NONEMPTY list of integers}. Return maximum sum of a contiguous nonempty subarray, including all-negative cases.
Example:
```json
{"input": {"nums": [-3]}, "expected": -3}
```

## P1-13 — Longest unique-character substring (medium)
Input {"text": string}. Return length of the longest contiguous substring with no repeated characters; case-sensitive.
Example:
```json
{"input": {"text": ""}, "expected": 0}
```

## P1-14 — Stable anagram groups (medium)
Input {"words": list of lowercase words}. Group anagrams. Groups are ordered by first appearance; words within each group stay in original input order.
Example:
```json
{"input": {"words": []}, "expected": []}
```

## P1-15 — Merge closed intervals (medium)
Input {"intervals": list of [start,end] with start<=end}. Return sorted union as closed intervals. Overlap or shared endpoint merges; mere integer adjacency does not.
Example:
```json
{"input": {"intervals": []}, "expected": []}
```

## P1-16 — Count islands (medium)
Input {"grid": rectangular list of strings containing characters 0 or 1}. Count connected components of 1 cells, using 4-neighbor adjacency. [] -> 0.
Example:
```json
{"input": {"grid": []}, "expected": 0}
```

## P1-17 — Unweighted directed shortest-path length (medium)
Input {"n": positive integer, "edges": [u,v] directed edges, "source": node, "target": node}. Nodes 0..n-1. Return minimum edge count or -1.
Example:
```json
{"input": {"n": 1, "edges": [], "source": 0, "target": 0}, "expected": 0}
```

## P1-18 — Fewest coins (medium)
Input {"coins": positive int list, "amount": nonnegative int}. Coins are unlimited. Return minimum number to reach amount, or -1; amount zero -> 0.
Example:
```json
{"input": {"coins": [1, 3], "amount": 0}, "expected": 0}
```

## P1-19 — Minimum containing window (medium)
Input {"s": string, "t": string}. Return shortest substring of s containing all characters of t with multiplicities, or empty string if impossible. Break ties by earliest start; empty t -> empty string.
Example:
```json
{"input": {"s": "", "t": "A"}, "expected": ""}
```

## P1-20 — LRU cache simulation (medium)
Input {"capacity": nonnegative int, "ops": list of ["put",key,value] or ["get",key] with integer keys/values}. Output the results of get commands; missing -> -1. Updating and successful get mark most recently used. Capacity zero stores nothing.
Example:
```json
{"input": {"capacity": 0, "ops": [["put", 1, 2], ["get", 1]]}, "expected": [-1]}
```

## P1-21 — Glob matcher (medium)
Input {"text": string, "pattern": string}. Return whether entire text matches pattern, where ? matches exactly one character and * matches zero or more characters; all other chars literal.
Example:
```json
{"input": {"text": "", "pattern": "*"}, "expected": true}
```

## P1-22 — Maximum nonoverlapping intervals (medium)
Input {"intervals": list of [start,end] with start<end}. Intervals are half-open [start,end). Return maximum number with pairwise disjoint interiors (touching endpoints allowed).
Example:
```json
{"input": {"intervals": []}, "expected": 0}
```

## P1-23 — Decode nested repetition (medium)
Input {"text": string} formatted as nested k[encoded_text] where k is nonnegative decimal, text atoms are lowercase letters, and bracket syntax is valid. Return fully expanded string; expanded length <= 2000.
Example:
```json
{"input": {"text": ""}, "expected": ""}
```

## P1-24 — Spreadsheet column number (medium)
Input {"label": nonempty uppercase A-Z string}. Return Excel-style base-26 column number: A=1, Z=26, AA=27.
Example:
```json
{"input": {"label": "A"}, "expected": 1}
```

## P1-25 — Search rotated distinct sorted array (medium)
Input {"nums": rotated strictly increasing integer list, "target": int}. Return target index or -1; empty list -> -1.
Example:
```json
{"input": {"nums": [], "target": 3}, "expected": -1}
```

## P1-26 — Reverse Polish notation (medium)
Input {"tokens": valid postfix expression list of decimal integer strings and +,-,*,/}. Return int result. Division truncates toward zero; divisor never zero; result fits signed 32-bit.
Example:
```json
{"input": {"tokens": ["3", "4", "+"]}, "expected": 7}
```

## P1-27 — Lexicographically smallest topological order (medium)
Input {"n": nonnegative int, "edges": directed prerequisite edges [u,v]}. Return lexicographically smallest topological ordering of all nodes 0..n-1, or [] on a cycle. Duplicate edges allowed.
Example:
```json
{"input": {"n": 0, "edges": []}, "expected": []}
```

## P1-28 — Grid shortest walk (medium)
Input {"grid": rectangular matrix of integers 0=open or 1=blocked}. Return min 4-neighbor moves from top-left to bottom-right, or -1 if no path, blocked endpoint, or empty grid.
Example:
```json
{"input": {"grid": []}, "expected": -1}
```

## P1-29 — Kth largest with duplicates (medium)
Input {"nums": nonempty int list, "k": int between 1 and len(nums)}. Return kth largest element counting duplicate occurrences.
Example:
```json
{"input": {"nums": [1], "k": 1}, "expected": 1}
```

## P1-30 — Days until warmer temperature (medium)
Input {"temps": integer list}. For each day return days until first STRICTLY warmer temperature to its right, or 0 if none.
Example:
```json
{"input": {"temps": []}, "expected": []}
```

## P1-31 — Levenshtein edit distance (hard)
Input {"a": string, "b": string}. Return minimum number of unit-cost insertions, deletions and substitutions to turn a into b.
Example:
```json
{"input": {"a": "", "b": ""}, "expected": 0}
```

## P1-32 — Longest strictly increasing subsequence (hard)
Input {"nums": int list}. Return length of a strictly increasing subsequence, not necessarily contiguous; [] -> 0.
Example:
```json
{"input": {"nums": []}, "expected": 0}
```

## P1-33 — Word break feasibility (hard)
Input {"text": string, "words": list of nonempty strings}. Return whether text can be segmented into zero or more dictionary words, reusable unlimited times; empty text -> true.
Example:
```json
{"input": {"text": "", "words": []}, "expected": true}
```

## P1-34 — Simplified full regular expression match (hard)
Input {"text": string, "pattern": valid restricted regex}. Pattern supports . (any one char) and * (zero+ of preceding token). Match the ENTIRE text. Pattern never begins with * or contains consecutive *.
Example:
```json
{"input": {"text": "", "pattern": ""}, "expected": true}
```

## P1-35 — Trapped rainwater (hard)
Input {"heights": list of nonnegative ints}. Return total water units trapped between histogram bars of unit width.
Example:
```json
{"input": {"heights": []}, "expected": 0}
```

## P1-36 — Dijkstra all-node distances (hard)
Input {"n": positive int, "edges": [u,v,weight] directed with NONNEGATIVE int weight, "source": node}. Return list of shortest distances from source, using JSON null for unreachable nodes.
Example:
```json
{"input": {"n": 1, "edges": [], "source": 0}, "expected": [0]}
```

## P1-37 — Minimum-cost right/down grid (hard)
Input {"grid": rectangular matrix of nonnegative cell costs and -1 obstacles}. Move only right or down from top-left to bottom-right. Path cost includes endpoints. Return minimum cost, or null if no path/empty grid.
Example:
```json
{"input": {"grid": []}, "expected": null}
```

## P1-38 — Equal subset partition (hard)
Input {"nums": list of nonnegative integers}. Return whether nums can be split into two disjoint subsets with equal sums; empty list -> true.
Example:
```json
{"input": {"nums": []}, "expected": true}
```

## P1-39 — Maximum product of nonempty subarray (hard)
Input {"nums": NONEMPTY integer list}. Return maximum product of a contiguous, nonempty subarray, allowing negatives and zero.
Example:
```json
{"input": {"nums": [-1]}, "expected": -1}
```

## P1-40 — Strongly connected components (hard)
Input {"n": nonnegative int, "edges": directed [u,v] pairs}. Return number of strongly connected components among nodes 0..n-1; isolated nodes each count one.
Example:
```json
{"input": {"n": 0, "edges": []}, "expected": 0}
```
