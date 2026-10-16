# Pilot 1 Java — 40 NEW coding questions

This is an ORIGINAL development test bank, not a recent third-party benchmark or sealed holdout. Difficulty is a provisional label. All questions differ from the Python Pilot 1 question topics.

Use Java 21. For each question submit one `public final class Solution` with `public static Object solve(Map<String,Object> data)`. The provided support `Json.java` and `Runner.java` handle one JSON input object per line and serialize the returned Java value. Inputs use numbers as `Number`, arrays as `List<?>`, and objects as `Map<?,?>`. Outputs may use Java primitives/wrappers, Strings, Lists or Maps. No external libraries, file I/O, network or subprocesses.

Each problem has four public examples and 40 additional **development grader** cases. The grader is not a secure execution sandbox. Never expose grader files to the model during generation.


## J1-01 — ASCII palindrome filter (easy)

Input {"text": string}. Keep only ASCII letters A-Z/a-z and digits 0-9; compare case-insensitively. Return true iff the filtered string is a palindrome. Empty filtered text is a palindrome.

**Examples**

```json
{"input":{"text":""},"expected":true}
```

```json
{"input":{"text":"A man, a plan, a canal: Panama!"},"expected":true}
```

```json
{"input":{"text":"java 21"},"expected":false}
```

```json
{"input":{"text":"0P0"},"expected":true}
```


## J1-02 — Gregorian leap year (easy)

Input {"year": integer 1..9999}. Return true iff it is a Gregorian leap year (divisible by 400, or divisible by 4 but not 100).

**Examples**

```json
{"input":{"year":2000},"expected":true}
```

```json
{"input":{"year":1900},"expected":false}
```

```json
{"input":{"year":2024},"expected":true}
```

```json
{"input":{"year":2025},"expected":false}
```


## J1-03 — Multiplicative persistence (easy)

Input {"n": nonnegative integer <= 1,000,000,000,000}. Repeatedly replace n with the product of its decimal digits until it is a one-digit number. Return the number of replacements; one-digit inputs yield 0.

**Examples**

```json
{"input":{"n":0},"expected":0}
```

```json
{"input":{"n":39},"expected":3}
```

```json
{"input":{"n":999},"expected":4}
```

```json
{"input":{"n":77},"expected":4}
```


## J1-04 — Integer to Roman numeral (easy)

Input {"n": integer 1..3999}. Return the standard uppercase minimal Roman numeral, using subtractive forms IV, IX, XL, XC, CD and CM.

**Examples**

```json
{"input":{"n":1},"expected":"I"}
```

```json
{"input":{"n":4},"expected":"IV"}
```

```json
{"input":{"n":944},"expected":"CMXLIV"}
```

```json
{"input":{"n":3999},"expected":"MMMCMXCIX"}
```


## J1-05 — 24-hour clock to 12-hour clock (easy)

Input {"time": string HH:MM} where HH is 00..23 and MM is 00..59. Return string "hh:mm AM" or "hh:mm PM"; hh is always two digits (01..12).

**Examples**

```json
{"input":{"time":"00:00"},"expected":"12:00 AM"}
```

```json
{"input":{"time":"12:00"},"expected":"12:00 PM"}
```

```json
{"input":{"time":"23:59"},"expected":"11:59 PM"}
```

```json
{"input":{"time":"09:07"},"expected":"09:07 AM"}
```


## J1-06 — Validate ISBN-10 (easy)

Input {"code": string}. Ignore spaces and hyphens. Return true iff ten characters remain, the first nine are decimal digits, the last is a digit or uppercase X (value 10), and the ISBN-10 weighted checksum sum(10-i)*digit[i] is divisible by 11.

**Examples**

```json
{"input":{"code":"0-306-40615-2"},"expected":true}
```

```json
{"input":{"code":"0-8044-2957-X"},"expected":true}
```

```json
{"input":{"code":"0-306-40615-3"},"expected":false}
```

```json
{"input":{"code":"abc"},"expected":false}
```


## J1-07 — Transpose rectangular matrix (easy)

Input {"matrix": rectangular list of integer lists}. Return its transpose. Input [] returns []; nonempty matrices may contain zero columns. Output lists are in row-major order.

**Examples**

```json
{"input":{"matrix":[]},"expected":[]}
```

```json
{"input":{"matrix":[[1,2],[3,4]]},"expected":[[1,3],[2,4]]}
```

```json
{"input":{"matrix":[[1,2,3]]},"expected":[[1],[2],[3]]}
```

```json
{"input":{"matrix":[[],[]]},"expected":[]}
```


## J1-08 — Parse unsigned binary number (easy)

Input {"bits": nonempty string of 0/1, length 1..60}. Return the nonnegative decimal integer represented by the bits, allowing leading zeros.

**Examples**

```json
{"input":{"bits":"0"},"expected":0}
```

```json
{"input":{"bits":"1"},"expected":1}
```

```json
{"input":{"bits":"00001010"},"expected":10}
```

```json
{"input":{"bits":"111111111111111111111111111111111111111111111111111111111111"},"expected":1152921504606846975}
```


## J1-09 — ASCII Caesar cipher (easy)

Input {"text": string, "shift": integer -1000..1000}. Shift only ASCII letters cyclically by shift; preserve letter case and leave digits, punctuation and all other characters unchanged.

**Examples**

```json
{"input":{"text":"Abc Zz!","shift":2},"expected":"Cde Bb!"}
```

```json
{"input":{"text":"xyz XYZ","shift":-1},"expected":"wxy WXY"}
```

```json
{"input":{"text":"123 äΩ","shift":26},"expected":"123 äΩ"}
```

```json
{"input":{"text":"A","shift":1000},"expected":"M"}
```


## J1-10 — Weekday of ISO date (easy)

Input {"date": valid ISO date YYYY-MM-DD between 1900-01-01 and 2100-12-31}. Return its English uppercase weekday name: MONDAY through SUNDAY.

**Examples**

```json
{"input":{"date":"2000-01-01"},"expected":"SATURDAY"}
```

```json
{"input":{"date":"2024-02-29"},"expected":"THURSDAY"}
```

```json
{"input":{"date":"1900-01-01"},"expected":"MONDAY"}
```

```json
{"input":{"date":"2100-12-31"},"expected":"FRIDAY"}
```


## J1-11 — Parse one CSV record (medium)

Input {"row": one valid CSV record without embedded newlines}. Return list of fields. Comma separates unquoted fields; double quotes protect commas, and inside quoted fields two adjacent double quotes represent one literal double quote. Preserve leading/trailing spaces and empty fields; empty input returns [""].

**Examples**

```json
{"input":{"row":""},"expected":[""]}
```

```json
{"input":{"row":"a,b,c"},"expected":["a","b","c"]}
```

```json
{"input":{"row":"\"a,b\",c"},"expected":["a,b","c"]}
```

```json
{"input":{"row":"\"ab\"\"cd\",,\"\""},"expected":["ab\"cd","",""]}
```


## J1-12 — IPv4 CIDR network address (medium)

Input {"ip": canonical dotted-decimal IPv4 address, "prefix": integer 0..32}. Return normalized network address as "a.b.c.d/prefix" after zeroing host bits.

**Examples**

```json
{"input":{"ip":"10.17.2.9","prefix":8},"expected":"10.0.0.0/8"}
```

```json
{"input":{"ip":"192.168.2.15","prefix":24},"expected":"192.168.2.0/24"}
```

```json
{"input":{"ip":"1.2.3.4","prefix":0},"expected":"0.0.0.0/0"}
```

```json
{"input":{"ip":"255.255.255.255","prefix":32},"expected":"255.255.255.255/32"}
```


## J1-13 — Clockwise spiral matrix traversal (medium)

Input {"matrix": rectangular list of integer rows}. Return entries visited clockwise from top-left: top row, right column, bottom row reversed, left column upward, repeating. Empty matrix or zero columns returns [].

**Examples**

```json
{"input":{"matrix":[]},"expected":[]}
```

```json
{"input":{"matrix":[[1,2,3],[4,5,6]]},"expected":[1,2,3,6,5,4]}
```

```json
{"input":{"matrix":[[3],[2],[1]]},"expected":[3,2,1]}
```

```json
{"input":{"matrix":[[1,2,3],[4,5,6],[7,8,9]]},"expected":[1,2,3,6,9,8,7,4,5]}
```


## J1-14 — Rotate square matrix clockwise (medium)

Input {"matrix": N×N integer matrix, N>=0}. Return a new matrix after rotating by 90 degrees clockwise. [] returns [].

**Examples**

```json
{"input":{"matrix":[]},"expected":[]}
```

```json
{"input":{"matrix":[[1]]},"expected":[[1]]}
```

```json
{"input":{"matrix":[[1,2],[3,4]]},"expected":[[3,1],[4,2]]}
```

```json
{"input":{"matrix":[[1,2,3],[4,5,6],[7,8,9]]},"expected":[[7,4,1],[8,5,2],[9,6,3]]}
```


## J1-15 — Evaluate polynomial in descending coefficient order (medium)

Input {"coeff": integers [a_d,...,a_0], "x": integer}. Return a_d*x^d+...+a_0. Empty coefficient list represents the zero polynomial. Input constraints guarantee a signed 64-bit result and intermediate Horner values.

**Examples**

```json
{"input":{"coeff":[],"x":5},"expected":0}
```

```json
{"input":{"coeff":[5],"x":3},"expected":5}
```

```json
{"input":{"coeff":[1,-2,3],"x":-4},"expected":27}
```

```json
{"input":{"coeff":[0,0,9],"x":0},"expected":9}
```


## J1-16 — Repeatedly remove adjacent K-groups (medium)

Input {"text": lowercase ASCII string, "k": integer 1..5}. Repeatedly remove a group whenever k consecutive equal letters appear; after removal, newly adjacent groups may also vanish. Return remaining text.

**Examples**

```json
{"input":{"text":"","k":3},"expected":""}
```

```json
{"input":{"text":"deeedbbcccbdaa","k":3},"expected":"aa"}
```

```json
{"input":{"text":"aaaaa","k":2},"expected":"a"}
```

```json
{"input":{"text":"abbba","k":1},"expected":""}
```


## J1-17 — Josephus circle winner (medium)

Input {"n": integer 1..10000, "k": integer 1..1000}. People numbered 1..n form a circle; remove every kth person, counting inclusively from person 1 and resuming after each removal. Return the 1-based number of the survivor.

**Examples**

```json
{"input":{"n":1,"k":1},"expected":1}
```

```json
{"input":{"n":7,"k":3},"expected":4}
```

```json
{"input":{"n":5,"k":1},"expected":5}
```

```json
{"input":{"n":10,"k":10},"expected":8}
```


## J1-18 — Count subarrays with target sum (medium)

Input {"nums": list of signed integers, "k": signed integer}. Return the number of NONEMPTY contiguous subarrays whose sum is exactly k. Input length <= 100, result fits signed 64-bit.

**Examples**

```json
{"input":{"nums":[],"k":0},"expected":0}
```

```json
{"input":{"nums":[0,0,0],"k":0},"expected":6}
```

```json
{"input":{"nums":[1,2,-1,2],"k":3},"expected":2}
```

```json
{"input":{"nums":[-1,-1,1],"k":-1},"expected":3}
```


## J1-19 — Count obstacle-avoiding right/down paths (medium)

Input {"grid": rectangular matrix of 0=open and 1=blocked cells, dimensions <=12x12}. From top-left to bottom-right, move only right or down through open cells. Return number of valid paths; empty grid or blocked endpoint gives 0.

**Examples**

```json
{"input":{"grid":[]},"expected":0}
```

```json
{"input":{"grid":[[0]]},"expected":1}
```

```json
{"input":{"grid":[[1]]},"expected":0}
```

```json
{"input":{"grid":[[0,0,0],[0,1,0],[0,0,0]]},"expected":2}
```


## J1-20 — Exact-word and prefix queries (medium)

Input {"words": list of lowercase words, "queries": list of {"type":"exact"|"prefix", "word": lowercase string}}. Return a boolean for each query in order. "exact" matches complete words; "prefix" is true when any stored word starts with the query, including empty prefix if words is nonempty.

**Examples**

```json
{"input":{"words":[],"queries":[{"type":"prefix","word":""}]},"expected":[false]}
```

```json
{"input":{"words":["cat","car"],"queries":[{"type":"prefix","word":"ca"},{"type":"exact","word":"ca"}]},"expected":[true,false]}
```

```json
{"input":{"words":["a","ab"],"queries":[{"type":"exact","word":"a"}]},"expected":[true]}
```

```json
{"input":{"words":["aaa"],"queries":[{"type":"prefix","word":"b"}]},"expected":[false]}
```


## J1-21 — Multiply integer matrices (medium)

Input {"a": R×K integer matrix, "b": K×C integer matrix}. Return R×C matrix product. Constraints: R,K,C >=1 and <=7, each element -10..10.

**Examples**

```json
{"input":{"a":[[1]],"b":[[2]]},"expected":[[2]]}
```

```json
{"input":{"a":[[1,0],[0,1]],"b":[[3,4],[5,6]]},"expected":[[3,4],[5,6]]}
```

```json
{"input":{"a":[[-1,2,0]],"b":[[1],[5],[-9]]},"expected":[[9]]}
```

```json
{"input":{"a":[[0,0]],"b":[[3,4],[5,6]]},"expected":[[0,0]]}
```


## J1-22 — Validate partially filled Sudoku (medium)

Input {"board": list of nine strings, length 9 each, with characters . or 1..9}. Return true iff no digit repeats in any row, column or 3×3 box. Ignore dots; a complete solution is NOT required.

**Examples**

```json
{"input":{"board":[".........",".........",".........",".........",".........",".........",".........",".........","........."]},"expected":true}
```

```json
{"input":{"board":["11.......",".........",".........",".........",".........",".........",".........",".........","........."]},"expected":false}
```

```json
{"input":{"board":["1........","1........",".........",".........",".........",".........",".........",".........","........."]},"expected":false}
```

```json
{"input":{"board":["1........",".1.......",".........",".........",".........",".........",".........",".........","........."]},"expected":false}
```


## J1-23 — Critical path duration in a DAG (medium)

Input {"duration": nonnegative integer duration per node, "edges": list of [u,v] meaning u must finish before v starts}. Nodes are 0-based, graph is a DAG, n<=20. Tasks may run in parallel. Return the earliest time all tasks can finish, or 0 for no tasks.

**Examples**

```json
{"input":{"duration":[],"edges":[]},"expected":0}
```

```json
{"input":{"duration":[5],"edges":[]},"expected":5}
```

```json
{"input":{"duration":[3,4,5],"edges":[[0,1],[1,2]]},"expected":12}
```

```json
{"input":{"duration":[2,5,9],"edges":[[0,2],[1,2]]},"expected":14}
```


## J1-24 — Evaluate plus/minus parentheses expression (medium)

Input {"expression": valid expression made of decimal nonnegative integer literals, +, -, parentheses and spaces, with unary + or - allowed}. Return signed integer mathematical value. No multiplication or division; intermediate results fit signed 64-bit.

**Examples**

```json
{"input":{"expression":"0"},"expected":0}
```

```json
{"input":{"expression":"-(-5+3)"},"expected":2}
```

```json
{"input":{"expression":"1 - (2 - (3-4))"},"expected":-2}
```

```json
{"input":{"expression":" +12-(+3 + -2) "},"expected":11}
```


## J1-25 — Restore possible dotted IPv4 addresses (medium)

Input {"digits": string of 0..12 decimal digits}. Insert three dots to form valid IPv4 addresses; segments must be 0..255 with no leading zeros except "0". Return all valid addresses in ascending lexicographic order, without duplicates.

**Examples**

```json
{"input":{"digits":"0000"},"expected":["0.0.0.0"]}
```

```json
{"input":{"digits":"25525511135"},"expected":["255.255.11.135","255.255.111.35"]}
```

```json
{"input":{"digits":"1111"},"expected":["1.1.1.1"]}
```

```json
{"input":{"digits":"123"},"expected":[]}
```


## J1-26 — Count distinct palindromic substrings (medium)

Input {"text": lowercase ASCII string length <=80}. Return the number of DISTINCT nonempty palindromic contiguous substrings. Equal substrings at different positions count once.

**Examples**

```json
{"input":{"text":""},"expected":0}
```

```json
{"input":{"text":"aaa"},"expected":3}
```

```json
{"input":{"text":"ababa"},"expected":5}
```

```json
{"input":{"text":"abcd"},"expected":4}
```


## J1-27 — Maximum XOR of two elements (medium)

Input {"nums": list of at least two nonnegative integers <=1,000,000}. Return the maximum bitwise XOR across pairs of DISTINCT indices.

**Examples**

```json
{"input":{"nums":[0,0]},"expected":0}
```

```json
{"input":{"nums":[1,2]},"expected":3}
```

```json
{"input":{"nums":[2,8,16,31]},"expected":29}
```

```json
{"input":{"nums":[1023,0,1]},"expected":1023}
```


## J1-28 — Count per-user events in an inclusive time window (medium)

Input {"events": list of {"user": string, "minute": integer}, "start": integer, "end": integer}, with start<=end. Count events for each user whose minute lies in [start,end]. Return JSON object mapping only participating users to counts.

**Examples**

```json
{"input":{"events":[],"start":0,"end":1},"expected":{}}
```

```json
{"input":{"events":[{"user":"a","minute":0}],"start":1,"end":2},"expected":{}}
```

```json
{"input":{"events":[{"user":"a","minute":1},{"user":"b","minute":1}],"start":1,"end":1},"expected":{"a":1,"b":1}}
```

```json
{"input":{"events":[{"user":"z","minute":3},{"user":"z","minute":2}],"start":2,"end":3},"expected":{"z":2}}
```


## J1-29 — Minimum jumps to last index (medium)

Input {"nums": nonempty list of nonnegative integers}, where each number is the maximum forward jump from that index. Start at index 0. Return the smallest jumps to index n-1, or -1 if unreachable; a singleton returns 0.

**Examples**

```json
{"input":{"nums":[0]},"expected":0}
```

```json
{"input":{"nums":[2,3,1,1,4]},"expected":2}
```

```json
{"input":{"nums":[3,2,1,0,4]},"expected":-1}
```

```json
{"input":{"nums":[1,0]},"expected":1}
```


## J1-30 — Area covered by axis-aligned rectangles (medium)

Input {"rectangles": list of [x1,y1,x2,y2] integer coordinates with x1<x2 and y1<y2}. Return the total geometric UNION area; overlaps count once, touching borders do not add overlap. Empty list returns 0.

**Examples**

```json
{"input":{"rectangles":[]},"expected":0}
```

```json
{"input":{"rectangles":[[0,0,2,2]]},"expected":4}
```

```json
{"input":{"rectangles":[[0,0,2,2],[1,1,3,3]]},"expected":7}
```

```json
{"input":{"rectangles":[[0,0,2,1],[2,0,4,1]]},"expected":4}
```


## J1-31 — Multiply arbitrarily long decimal strings (hard)

Input {"a": unsigned decimal string, "b": unsigned decimal string}, each 1..40 digits, leading zeros allowed. Return their exact product as a normalized decimal string with no leading zeros except "0". No floating point.

**Examples**

```json
{"input":{"a":"0","b":"123"},"expected":"0"}
```

```json
{"input":{"a":"0012","b":"003"},"expected":"36"}
```

```json
{"input":{"a":"99","b":"99"},"expected":"9801"}
```

```json
{"input":{"a":"123456789","b":"987654321"},"expected":"121932631112635269"}
```


## J1-32 — Count array inversions (hard)

Input {"nums": integer list}. Return number of index pairs i<j with nums[i]>nums[j]. Equal elements do not count; empty list returns 0.

**Examples**

```json
{"input":{"nums":[]},"expected":0}
```

```json
{"input":{"nums":[1,2,3]},"expected":0}
```

```json
{"input":{"nums":[3,2,1]},"expected":3}
```

```json
{"input":{"nums":[2,2,2]},"expected":0}
```


## J1-33 — Range additions and range sums (hard)

Input {"nums": integer list, "operations": list of ["add",l,r,v] or ["sum",l,r]}; indices 0<=l<=r<len(nums) inclusive. Adds modify the current array; each "sum" appends the current inclusive range sum to output list.

**Examples**

```json
{"input":{"nums":[1,2,3],"operations":[["sum",0,2],["add",0,1,4],["sum",1,2]]},"expected":[6,9]}
```

```json
{"input":{"nums":[0],"operations":[["add",0,0,-1],["sum",0,0]]},"expected":[-1]}
```

```json
{"input":{"nums":[5,5],"operations":[["sum",1,1]]},"expected":[5]}
```

```json
{"input":{"nums":[3,-2,10],"operations":[]},"expected":[]}
```


## J1-34 — Maximum bipartite matching size (hard)

Input {"left": L, "right": R, "edges": list of [u,v] with 0<=u<L and 0<=v<R}, L,R<=12. Return the maximum number of disjoint left-to-right pairs. Duplicate edges are allowed.

**Examples**

```json
{"input":{"left":0,"right":0,"edges":[]},"expected":0}
```

```json
{"input":{"left":2,"right":2,"edges":[[0,0],[1,1]]},"expected":2}
```

```json
{"input":{"left":2,"right":1,"edges":[[0,0],[1,0]]},"expected":1}
```

```json
{"input":{"left":3,"right":3,"edges":[[0,0],[0,1],[1,1],[2,2]]},"expected":3}
```


## J1-35 — Burst balloons for maximum coins (hard)

Input {"nums": list of nonnegative integers 0..9, length <=8}. Each turn pop one balloon; coins gained equal left_neighbor * popped * right_neighbor, taking missing neighbors as 1. Return maximum total coins across possible removal orders; empty list gives 0.

**Examples**

```json
{"input":{"nums":[]},"expected":0}
```

```json
{"input":{"nums":[7]},"expected":7}
```

```json
{"input":{"nums":[3,1,5,8]},"expected":167}
```

```json
{"input":{"nums":[0,0,2]},"expected":2}
```


## J1-36 — Count constrained N-Queens placements (hard)

Input {"n": integer 1..9, "blocked": list of [row,col] cells with 0-based coordinates}. Count placements of n queens on an n×n board so no queens share row, column or diagonal and no queen is in a blocked cell. Duplicate blocked cells do not change the count.

**Examples**

```json
{"input":{"n":1,"blocked":[]},"expected":1}
```

```json
{"input":{"n":2,"blocked":[]},"expected":0}
```

```json
{"input":{"n":4,"blocked":[]},"expected":2}
```

```json
{"input":{"n":8,"blocked":[[0,0]]},"expected":88}
```


## J1-37 — All parenthesized arithmetic results (hard)

Input {"expression": string of 1..6 nonnegative integer operands separated by binary +, - or * with no spaces}. Return the sorted list of integer results for ALL binary parenthesizations, retaining duplicates when multiple parenthesizations yield the same result.

**Examples**

```json
{"input":{"expression":"5"},"expected":[5]}
```

```json
{"input":{"expression":"2-1-1"},"expected":[0,2]}
```

```json
{"input":{"expression":"2*3-4*5"},"expected":[-34,-14,-10,-10,10]}
```

```json
{"input":{"expression":"1+2+3"},"expected":[6,6]}
```


## J1-38 — Count distinct subsequence embeddings (hard)

Input {"source": lowercase string, "target": lowercase string}, length <=20. Return number of index sequences in source whose characters form target in order. Empty target has exactly one embedding.

**Examples**

```json
{"input":{"source":"","target":""},"expected":1}
```

```json
{"input":{"source":"abc","target":""},"expected":1}
```

```json
{"input":{"source":"rabbbit","target":"rabbit"},"expected":3}
```

```json
{"input":{"source":"aaaa","target":"aa"},"expected":6}
```


## J1-39 — Longest palindromic subsequence length (hard)

Input {"text": lowercase string length <=70}. Return maximum length of a not-necessarily-contiguous palindromic subsequence. Empty text returns 0.

**Examples**

```json
{"input":{"text":""},"expected":0}
```

```json
{"input":{"text":"bbbab"},"expected":4}
```

```json
{"input":{"text":"abcd"},"expected":1}
```

```json
{"input":{"text":"agbdba"},"expected":5}
```


## J1-40 — Max-profit nonoverlapping job schedule (hard)

Input {"jobs": list of [start,end,profit] with start<end and nonnegative profit}. Select a subset maximizing sum of profit; jobs may touch (previous end <= next start). Return maximum achievable profit, 0 if none.

**Examples**

```json
{"input":{"jobs":[]},"expected":0}
```

```json
{"input":{"jobs":[[1,2,9]]},"expected":9}
```

```json
{"input":{"jobs":[[1,3,5],[3,5,8],[2,4,10]]},"expected":13}
```

```json
{"input":{"jobs":[[0,2,5],[0,2,8],[2,3,1]]},"expected":9}
```
