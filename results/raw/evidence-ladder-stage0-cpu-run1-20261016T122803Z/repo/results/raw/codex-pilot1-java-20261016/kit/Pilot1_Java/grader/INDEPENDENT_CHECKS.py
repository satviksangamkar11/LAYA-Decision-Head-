#!/usr/bin/env python3
"""Independent Python cross-checks for high-risk Java reference tasks.
No agent-generated code executed. This is diagnostic, not a fully independent benchmark.
"""
import collections,csv,io,datetime,ipaddress,functools,itertools,json,pathlib,math, bisect
H=pathlib.Path(__file__).parent
P=json.loads((H/'public_cases_expected.json').read_text())
V=json.loads((H/'protected_cases.json').read_text())

def check(n,d):
 if n==1:
  import re
  v=re.sub('[^a-z0-9]','',d['text'].lower());return v==v[::-1]
 if n==3:
  x=d['n'];steps=0
  while x>=10:
   digits=list(map(int,str(x)))
   x=math.prod(digits);steps+=1
  return steps
 if n==6:
  z=d['code'].replace('-','').replace(' ','')
  if not __import__('re').fullmatch('[0-9]{9}[0-9X]',z):return False
  vals=[int(v) for v in z[:9]]+[10 if z[-1]=='X' else int(z[-1])]
  return sum(a*b for a,b in zip(vals,range(10,0,-1)))%11==0
 if n==7:
  a=d['matrix'];return [list(col) for col in zip(*a)] if a else []
 if n==9:
  out=[]
  for c in d['text']:
   off=ord('A') if 'A'<=c<='Z' else ord('a') if 'a'<=c<='z' else None
   out.append(chr(off+(ord(c)-off+d['shift'])%26) if off is not None else c)
  return ''.join(out)
 if n==13:
  a=[row.copy() for row in d['matrix']];out=[]
  while a and a[0]:
   out.extend(a.pop(0))
   a=[list(col) for col in zip(*a)][::-1] if a else []
  return out
 if n==15:
  x=d['x'];return sum(v*x**(len(d['coeff'])-1-j) for j,v in enumerate(d['coeff']))
 if n==20:
  w=set(d['words']);return [q['word'] in w if q['type']=='exact' else any(z.startswith(q['word']) for z in w) for q in d['queries']]
 if n==2:
  y=d['year'];return (y%4==0) and (y%100!=0 or y%400==0)
 if n==4:
  import random
  num=d['n'];rom={1:'I',4:'IV',5:'V',9:'IX',10:'X',40:'XL',50:'L',90:'XC',100:'C',400:'CD',500:'D',900:'CM',1000:'M'}
  out=''; remaining=num
  for k in sorted(rom,reverse=True):
   count,remaining=divmod(remaining,k);out+=rom[k]*count
  return out
 if n==5:
  h,m=map(int,d['time'].split(':'));return f'{h%12 or 12:02d}:{m:02d} {"AM" if h<12 else "PM"}'
 if n==8:return int(d['bits'],2)
 if n==10:return datetime.date.fromisoformat(d['date']).strftime('%A').upper()
 if n==11:return next(csv.reader([d['row']])) if d['row'] else ['']
 if n==12:return str(ipaddress.ip_network(f"{d['ip']}/{d['prefix']}",strict=False))
 if n==14:
  a=d['matrix'];return [list(z)[::-1] for z in zip(*a)] if a else []
 if n==16:
  s,k=d['text'],d['k']
  if k==1:return ''
  while True:
   parts=[(a,len(list(g))) for a,g in itertools.groupby(s)]
   new=''.join(ch*qty for ch,qty in parts if qty<k)
   # handle any group length >=k and its remainder via repeated consecutive k-deletes
   new=''.join(ch*(qty%k) for ch,qty in parts)
   if new==s:return s
   s=new
 if n==17:
  n,k=d['n'],d['k'];a=list(range(1,n+1));p=0
  while len(a)>1:p=(p+k-1)%len(a);a.pop(p)
  return a[0]
 if n==18:
  v=d['nums'];return sum(sum(v[i:j])==d['k'] for i in range(len(v)) for j in range(i+1,len(v)+1))
 if n==19:
  a=d['grid'];r=len(a);c=len(a[0]) if r else 0
  @functools.lru_cache(None)
  def f(i,j):
   if i>=r or j>=c or a[i][j]:return 0
   if (i,j)==(r-1,c-1):return 1
   return f(i+1,j)+f(i,j+1)
  return f(0,0)
 if n==21:
  A,B=d['a'],d['b'];return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
 if n==22:
  b=d['board'];areas=[]
  areas.extend([list(row) for row in b]);areas.extend([[b[r][c] for r in range(9)] for c in range(9)])
  for i in range(0,9,3):
   for j in range(0,9,3):areas.append([b[r][c] for r in range(i,i+3) for c in range(j,j+3)])
  return all(len(z:=[c for c in area if c!='.'])==len(set(z)) for area in areas)
 if n==23:
  t=d['duration'];pred=[[] for _ in t]
  for a,b in d['edges']:pred[b].append(a)
  @functools.lru_cache(None)
  def f(j):return t[j]+max([f(k) for k in pred[j]] or [0])
  return max([f(j) for j in range(len(t))] or [0])
 if n==24:
  # syntactically restricted arithmetic. Python eval acts as independent parser here, only on fixed fixtures
  return eval(d['expression'],{'__builtins__':{}})
 if n==25:
  s=d['digits'];ans=[]
  for c in itertools.combinations(range(1,len(s)),3):
   z=[s[:c[0]],s[c[0]:c[1]],s[c[1]:c[2]],s[c[2]:]]
   if all(len(p)<=3 and (len(p)==1 or p[0]!='0') and int(p)<=255 for p in z):ans.append('.'.join(z))
  return sorted(ans)
 if n==26:
  s=d['text'];return len({s[i:j] for i in range(len(s)) for j in range(i+1,len(s)+1) if s[i:j]==s[i:j][::-1]})
 if n==27:return max(a^b for a,b in itertools.combinations(d['nums'],2))
 if n==28:return dict(collections.Counter(e['user'] for e in d['events'] if d['start']<=e['minute']<=d['end']))
 if n==29:
  x=d['nums'];v=[len(x)+1]*len(x);v[0]=0
  for i,j in enumerate(x):
   if v[i]>len(x):continue
   for k in range(i+1,min(len(x),i+j+1)):v[k]=min(v[k],v[i]+1)
  return v[-1] if v[-1]<=len(x) else -1
 if n==30:
  rect=d['rectangles']; xs=sorted({x for x1,y1,x2,y2 in rect for x in (x1,x2)});ys=sorted({y for x1,y1,x2,y2 in rect for y in (y1,y2)})
  return sum((x1-x0)*(y1-y0) for x0,x1 in zip(xs,xs[1:]) for y0,y1 in zip(ys,ys[1:]) if any(a<=x0 and b<=y0 and c>=x1 and f>=y1 for a,b,c,f in rect))
 if n==31:return str(int(d['a'])*int(d['b']))
 if n==32:
  a=d['nums'];return sum(a[i]>a[j] for i in range(len(a)) for j in range(i+1,len(a)))
 if n==33:
  a=d['nums'].copy();out=[]
  for op in d['operations']:
   if op[0]=='sum':out.append(sum(a[op[1]:op[2]+1]))
   else:
    for j in range(op[1],op[2]+1):a[j]+=op[3]
  return out
 if n==34:
  L,R=d['left'],d['right'];ed=[{v for u,v in d['edges'] if u==j} for j in range(L)]
  @functools.lru_cache(None)
  def dfs(i,mask):
   if i==L:return 0
   return max([dfs(i+1,mask)]+[1+dfs(i+1,mask|1<<v) for v in ed[i] if not (mask>>v)&1])
  return dfs(0,0)
 if n==35:
  @functools.lru_cache(None)
  def dfs(t):
   if not t:return 0
   return max(dfs(t[:i]+t[i+1:])+(t[i-1] if i>0 else 1)*t[i]*(t[i+1] if i+1<len(t) else 1) for i in range(len(t)))
  return dfs(tuple(d['nums']))
 if n==36:
  n=d['n'];forbid=set(tuple(z) for z in d['blocked'])
  @functools.lru_cache(None)
  def dfs(row,qs):
   if row==n:return 1
   return sum(dfs(row+1,qs+(col,)) for col in range(n) if (row,col) not in forbid and all(q!=col and abs(row-i)!=abs(q-col) for i,q in enumerate(qs)))
  return dfs(0,())
 if n==37:
  s=d['expression'];nums=list(map(int,__import__('re').split('[+*-]',s)));ops=__import__('re').findall('[+*-]',s)
  @functools.lru_cache(None)
  def f(i,j):
   if i==j:return (nums[i],)
   arr=[]
   for k in range(i,j):
    for a in f(i,k):
     for b in f(k+1,j):arr.append({'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b}[ops[k]]())
   return tuple(arr)
  return sorted(f(0,len(nums)-1))
 if n==38:
  s,t=d['source'],d['target'];dp=[1]+[0]*len(t)
  for x in s:
   ndp=dp.copy()
   for i in range(1,len(dp)):
    if x==t[i-1]:ndp[i]+=dp[i-1]
   dp=ndp
  return dp[-1]
 if n==39:
  s=d['text'];t=s[::-1];a=[0]*(len(t)+1)
  for c in s:
   b=[0]*(len(t)+1)
   for j,x in enumerate(t,1):b[j]=a[j-1]+1 if c==x else max(a[j],b[j-1])
   a=b
  return a[-1]
 if n==40:
  jobs=sorted(d['jobs'],key=lambda z:z[0])
  starts=[j[0] for j in jobs]
  @functools.lru_cache(None)
  def f(i):
   if i==len(jobs):return 0
   st,en,v=jobs[i]
   j=bisect.bisect_left(starts,en,i+1)
   return max(f(i+1),v+f(j))
  return f(0)
 return None

checked=0; skipped=[]
for i,(pub,priv) in enumerate(zip(P,V),1):
  if check(i,pub['cases'][0]['input']) is None:
   skipped.append(i);continue
  for c in pub['cases']+priv['cases']:
   got=check(i,c['input']);exp=c['expected']
   assert got==exp,(i,c['input'],got,exp)
   checked+=1
print('PASS:',checked,'independent Python cross-checked case outputs; task IDs skipped:',skipped)
