"""v2: plausible WRONG solutions (from the 2026-10-16 review and common mistakes). Each must FAIL at least one protected case.
Reports, per file given, which wrong solutions escape (pass every case). Trusted tooling; the wrong programs are tiny and hand-written here.
Usage: python3 wrong_solutions_v2.py <protected_cases.json> <out.json>"""
import collections, copy, fnmatch, heapq, json, re, signal, sys
from pathlib import Path

W = {}
def w(task, name, src):
    W.setdefault(task, []).append((name, src))

# --- easy
w("P1-01", "counts_y_too", "def solve(d): return sum(c in 'aeiouyAEIOUY' for c in d['text'])")
w("P1-01", "str_lower_first", "def solve(d): return sum(c in 'aeiou' for c in d['text'].lower())")
w("P1-05", "str_lower_first", "import re,collections\ndef solve(d): return dict(collections.Counter(re.findall(r'[a-z]+', d['text'].lower())))")
w("P1-05", "keeps_digits", "import re,collections\ndef solve(d): return dict(collections.Counter(re.findall(r'[a-z0-9]+', d['text'].lower())))")
w("P1-05", "split_on_space", "import collections\ndef solve(d): return dict(collections.Counter(d['text'].lower().split()))")
w("P1-10", "ascii_ws_only", "import re\ndef solve(d): return ' '.join(re.split(r'[ \\t\\n]+', d['text'].strip(' \\t\\n'))) if d['text'].strip(' \\t\\n') else ''")
w("P1-10", "ascii_class_sub", "import re\ndef solve(d): return re.sub(r'[ \\t\\r\\n\\f\\v]+', ' ', d['text']).strip()")
w("P1-10", "split_single_space", "def solve(d): return ' '.join(d['text'].strip().split(' '))")
w("P1-11", "counts_only", "def solve(d):\n    t=d['text']; return all(t.count(a)==t.count(b) for a,b in ('()','[]','{}'))")
w("P1-11", "parens_only", "def solve(d):\n    n=0\n    for c in d['text']:\n        if c=='(': n+=1\n        elif c==')':\n            n-=1\n            if n<0: return False\n    return n==0")
w("P1-11", "ignores_type", "def solve(d):\n    st=[]\n    for c in d['text']:\n        if c in '([{': st.append(c)\n        elif c in ')]}':\n            if not st: return False\n            st.pop()\n    return not st")
w("P1-13", "lowercase_first", "def solve(d):\n    s=d['text'].lower(); last={}; st=b=0\n    for i,c in enumerate(s):\n        st=max(st,last.get(c,-1)+1); last[c]=i; b=max(b,i-st+1)\n    return b")
w("P1-13", "distinct_count", "def solve(d): return len(set(d['text']))")
# --- medium
w("P1-14", "drops_empty_words", "def solve(d):\n    g={}\n    for x in d['words']:\n        if x: g.setdefault(''.join(sorted(x)),[]).append(x)\n    return list(g.values())")
w("P1-14", "sorted_groups", "def solve(d):\n    g={}\n    for x in d['words']: g.setdefault(''.join(sorted(x)),[]).append(x)\n    return [g[k] for k in sorted(g)]")
w("P1-14", "sorted_inside", "def solve(d):\n    g={}\n    for x in d['words']: g.setdefault(''.join(sorted(x)),[]).append(x)\n    return [sorted(v) for v in g.values()]")
w("P1-16", "no_up_neighbor", "def solve(d):\n    g=d['grid'];m=len(g);n=len(g[0]) if m else 0;seen=set();c=0\n    for i in range(m):\n        for j in range(n):\n            if g[i][j]!='1' or (i,j) in seen: continue\n            c+=1;st=[(i,j)];seen.add((i,j))\n            while st:\n                x,y=st.pop()\n                for u,v in ((x+1,y),(x,y-1),(x,y+1)):\n                    if 0<=u<m and 0<=v<n and g[u][v]=='1' and (u,v) not in seen: seen.add((u,v));st.append((u,v))\n    return c")
w("P1-16", "wraps_around", "def solve(d):\n    g=d['grid'];m=len(g);n=len(g[0]) if m else 0;seen=set();c=0\n    for i in range(m):\n        for j in range(n):\n            if g[i][j]!='1' or (i,j) in seen: continue\n            c+=1;st=[(i,j)];seen.add((i,j))\n            while st:\n                x,y=st.pop()\n                for u,v in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):\n                    if -1<=u<m and -1<=v<n and g[u%m][v%n]=='1' and (u%m,v%n) not in seen: seen.add((u%m,v%n));st.append((u%m,v%n))\n    return c")
w("P1-16", "counts_ones", "def solve(d): return sum(r.count('1') for r in d['grid'])")
w("P1-18", "greedy_largest_first", "def solve(d):\n    a=d['amount'];n=0\n    for c in sorted(d['coins'],reverse=True):\n        n+=a//c;a%=c\n    return n if a==0 else -1")
w("P1-18", "unbounded_off_by_one", "def solve(d):\n    a=d['amount'];dp=[10**9]*(a+1);dp[0]=0\n    for v in range(1,a+1):\n        for c in d['coins']:\n            if c<v: dp[v]=min(dp[v],dp[v-c]+1)\n    return -1 if dp[a]>=10**9 else dp[a]")
w("P1-19", "latest_start_on_ties", "import collections\ndef solve(d):\n    s,t=d['s'],d['t']\n    if not t: return ''\n    best=None\n    need=collections.Counter(t)\n    for i in range(len(s)):\n        for j in range(i+len(t),len(s)+1):\n            if not need-collections.Counter(s[i:j]):\n                if best is None or j-i<=len(best): best=s[i:j]\n                break\n    return best or ''")
w("P1-19", "ignores_multiplicity", "def solve(d):\n    s,t=d['s'],d['t']\n    if not t: return ''\n    best=None\n    for i in range(len(s)):\n        for j in range(i+1,len(s)+1):\n            if set(t)<=set(s[i:j]):\n                if best is None or j-i<len(best): best=s[i:j]\n                break\n    return best or ''")
w("P1-20", "no_refresh_on_put_update", "def solve(d):\n    cap=d['capacity'];c={};order=[];out=[]\n    for op in d['ops']:\n        if op[0]=='get':\n            k=op[1]\n            if k in c: order.remove(k);order.append(k);out.append(c[k])\n            else: out.append(-1)\n        else:\n            k,v=op[1],op[2]\n            if cap==0: continue\n            if k in c: c[k]=v\n            else:\n                if len(c)>=cap: del c[order.pop(0)]\n                c[k]=v;order.append(k)\n    return out")
w("P1-20", "no_refresh_on_get", "def solve(d):\n    cap=d['capacity'];c={};order=[];out=[]\n    for op in d['ops']:\n        if op[0]=='get': out.append(c.get(op[1],-1))\n        else:\n            k,v=op[1],op[2]\n            if cap==0: continue\n            if k in c: order.remove(k)\n            elif len(c)>=cap: del c[order.pop(0)]\n            c[k]=v;order.append(k)\n    return out")
w("P1-21", "fnmatch", "import fnmatch\ndef solve(d): return fnmatch.fnmatchcase(d['text'],d['pattern'])")
w("P1-21", "unescaped_regex", "import re\ndef solve(d): return re.fullmatch(d['pattern'].replace('*','.*').replace('?','.'), d['text']) is not None")
w("P1-23", "two_passes_only", "import re\ndef solve(d):\n    s=d['text']\n    for _ in range(2): s=re.sub(r'(\\d+)\\[([a-z]*)\\]', lambda m: m.group(2)*int(m.group(1)), s)\n    return s")
w("P1-25", "bisect_ignores_rotation", "import bisect\ndef solve(d):\n    a=d['nums'];i=bisect.bisect_left(a,d['target'])\n    return i if i<len(a) and a[i]==d['target'] else -1")
w("P1-25", "strict_left_compare", "def solve(d):\n    a=d['nums'];k=d['target'];l=0;r=len(a)-1\n    while l<=r:\n        m=(l+r)//2\n        if a[m]==k: return m\n        if a[l]<a[m]:\n            if a[l]<=k<a[m]: r=m-1\n            else: l=m+1\n        else:\n            if a[m]<k<=a[r]: l=m+1\n            else: r=m-1\n    return -1")
w("P1-26", "sign_if_either_negative", "def solve(d):\n    a=[]\n    for t in d['tokens']:\n        if t not in '+-*/' or len(t)>1: a.append(int(t));continue\n        b=a.pop();x=a.pop()\n        if t=='+':a.append(x+b)\n        if t=='-':a.append(x-b)\n        if t=='*':a.append(x*b)\n        if t=='/':a.append((abs(x)//abs(b))*(-1 if (x<0 or b<0) else 1))\n    return a[-1]")
w("P1-26", "floor_division", "def solve(d):\n    a=[]\n    for t in d['tokens']:\n        if t not in '+-*/' or len(t)>1: a.append(int(t));continue\n        b=a.pop();x=a.pop()\n        if t=='+':a.append(x+b)\n        if t=='-':a.append(x-b)\n        if t=='*':a.append(x*b)\n        if t=='/':a.append(x//b)\n    return a[-1]")
w("P1-27", "string_keyed_heap", "import heapq\ndef solve(d):\n    n=d['n'];adj=[set() for _ in range(n)];deg=[0]*n\n    for a,b in d['edges']:\n        if b not in adj[a]: adj[a].add(b);deg[b]+=1\n    h=[str(i) for i in range(n) if deg[i]==0];heapq.heapify(h);out=[]\n    while h:\n        u=int(heapq.heappop(h));out.append(u)\n        for v in adj[u]:\n            deg[v]-=1\n            if deg[v]==0: heapq.heappush(h,str(v))\n    return out if len(out)==n else []")
w("P1-28", "diagonal_moves", "import collections\ndef solve(d):\n    g=d['grid'];m=len(g);n=len(g[0]) if m else 0\n    if not m or not n or g[0][0]!=0 or g[-1][-1]!=0: return -1\n    q=collections.deque([(0,0,0)]);seen={(0,0)}\n    while q:\n        x,y,k=q.popleft()\n        if x==m-1 and y==n-1: return k\n        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1)):\n            u,v=x+dx,y+dy\n            if 0<=u<m and 0<=v<n and g[u][v]==0 and (u,v) not in seen: seen.add((u,v));q.append((u,v,k+1))\n    return -1")
w("P1-28", "ignores_blocked_start", "import collections\ndef solve(d):\n    g=d['grid'];m=len(g);n=len(g[0]) if m else 0\n    if not m or not n: return -1\n    q=collections.deque([(0,0,0)]);seen={(0,0)}\n    while q:\n        x,y,k=q.popleft()\n        if x==m-1 and y==n-1: return k\n        for u,v in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):\n            if 0<=u<m and 0<=v<n and g[u][v]==0 and (u,v) not in seen: seen.add((u,v));q.append((u,v,k+1))\n    return -1")
w("P1-34", "python_re_fullmatch", "import re\ndef solve(d): return re.fullmatch(d['pattern'], d['text']) is not None")
w("P1-34", "star_means_one_or_more", "def solve(d):\n    import functools\n    s,p=d['text'],d['pattern']\n    @functools.lru_cache(None)\n    def f(i,j):\n        if j==len(p): return i==len(s)\n        first=i<len(s) and p[j] in (s[i],'.')\n        if j+1<len(p) and p[j+1]=='*': return first and (f(i+1,j) or f(i+1,j+2))\n        return first and f(i+1,j+1)\n    return f(0,0)")


class TO(BaseException):
    pass
def _al(s, f): raise TO()
signal.signal(signal.SIGALRM, _al)

def run_file(path):
    cases = {t["task_id"]: t["cases"] for t in json.loads(Path(path).read_text(encoding="utf-8"))}
    out = {}
    for task, lst in W.items():
        for name, src in lst:
            ns = {}
            exec(src, ns); fn = ns["solve"]
            bad = 0
            for c in cases[task]:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 2.0)
                    try: got = fn(copy.deepcopy(c["input"]))
                    finally: signal.setitimer(signal.ITIMER_REAL, 0)
                    if got != c["expected"] or type(got) is not type(c["expected"]): bad += 1
                except BaseException: bad += 1
            out[f"{task}:{name}"] = bad
    return out

res = run_file(sys.argv[1])
escaped = sorted(k for k, v in res.items() if v == 0)
Path(sys.argv[2]).write_text(json.dumps({"file": sys.argv[1], "wrong_solutions": len(res), "caught": len(res) - len(escaped), "escaped": escaped, "failing_cases_per_solution": res}, indent=1), encoding="utf-8")
print(json.dumps({"file": Path(sys.argv[1]).name, "wrong_solutions": len(res), "caught": len(res) - len(escaped), "escaped": escaped}, indent=1))
