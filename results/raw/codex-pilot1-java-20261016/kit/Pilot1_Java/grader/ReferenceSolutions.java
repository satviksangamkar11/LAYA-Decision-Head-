import java.util.*;
import java.time.*;
public final class ReferenceSolutions {
 private ReferenceSolutions() {}
 @SuppressWarnings("unchecked") static Map<String,Object> obj(Object o){return (Map<String,Object>)o;}
 static List<Object> li(Object o){return (List<Object>)o;}
 static String str(Object o){return (String)o;}
 static int in(Object o){return ((Number)o).intValue();}
 static long lng(Object o){return ((Number)o).longValue();}
 static int[] ia(Object o){List<Object> l=li(o);int[] a=new int[l.size()];for(int i=0;i<a.length;i++)a[i]=in(l.get(i));return a;}
 static long[] la(Object o){List<Object> l=li(o);long[] a=new long[l.size()];for(int i=0;i<a.length;i++)a[i]=lng(l.get(i));return a;}
 static int[][] mat(Object o){List<Object> l=li(o);int[][] a=new int[l.size()][];for(int i=0;i<a.length;i++)a[i]=ia(l.get(i));return a;}
 static List<Object> array(int[] a){List<Object> l=new ArrayList<>();for(int x:a)l.add(x);return l;}
 static List<Object> matrix(int[][] a){List<Object> l=new ArrayList<>();for(int[] x:a)l.add(array(x));return l;}
 public static Object solve(int n, Map<String,Object> d){return switch(n){
 case 1->pal(str(d.get("text")));
 case 2->leap(in(d.get("year")));
 case 3->persistence(lng(d.get("n")));
 case 4->roman(in(d.get("n")));
 case 5->time12(str(d.get("time")));
 case 6->isbn(str(d.get("code")));
 case 7->transpose(mat(d.get("matrix")));
 case 8->Long.parseLong(str(d.get("bits")),2);
 case 9->caesar(str(d.get("text")),in(d.get("shift")));
 case 10->LocalDate.parse(str(d.get("date"))).getDayOfWeek().toString();
 case 11->csv(str(d.get("row")));
 case 12->network(str(d.get("ip")),in(d.get("prefix")));
 case 13->spiral(mat(d.get("matrix")));
 case 14->rotate(mat(d.get("matrix")));
 case 15->poly(la(d.get("coeff")),lng(d.get("x")));
 case 16->reduce(str(d.get("text")),in(d.get("k")));
 case 17->josephus(in(d.get("n")),in(d.get("k")));
 case 18->sumcount(ia(d.get("nums")),lng(d.get("k")));
 case 19->paths(mat(d.get("grid")));
 case 20->trie(obj(d));
 case 21->multiply(mat(d.get("a")),mat(d.get("b")));
 case 22->sudoku(li(d.get("board")));
 case 23->critical(ia(d.get("duration")),mat(d.get("edges")));
 case 24->new Expr(str(d.get("expression"))).parse();
 case 25->restore(str(d.get("digits")));
 case 26->distinctPal(str(d.get("text")));
 case 27->maxxor(ia(d.get("nums")));
 case 28->aggregate(li(d.get("events")), in(d.get("start")),in(d.get("end")));
 case 29->jumps(ia(d.get("nums")));
 case 30->unionArea(mat(d.get("rectangles")));
 case 31->bigmul(str(d.get("a")),str(d.get("b")));
 case 32->inversions(ia(d.get("nums")));
 case 33->ranges(ia(d.get("nums")),li(d.get("operations")));
 case 34->matching(in(d.get("left")),in(d.get("right")),mat(d.get("edges")));
 case 35->balloons(ia(d.get("nums")));
 case 36->queens(in(d.get("n")),mat(d.get("blocked")));
 case 37->paren(str(d.get("expression")));
 case 38->subseq(str(d.get("source")),str(d.get("target")));
 case 39->lps(str(d.get("text")));
 case 40->jobs(mat(d.get("jobs")));
 default->throw new IllegalArgumentException("Task "+n);
 };}
 static boolean pal(String s){StringBuilder x=new StringBuilder();for(char c:s.toCharArray()){if(c>='A'&&c<='Z')x.append((char)(c+32));else if(c>='a'&&c<='z'||c>='0'&&c<='9')x.append(c);}return x.toString().contentEquals(x.reverse());}
 static boolean leap(int y){return y%400==0||y%4==0&&y%100!=0;}
 static int persistence(long n){int c=0;while(n>=10){long p=1;while(n>0){p*=n%10;n/=10;}n=p;c++;}return c;}
 static String roman(int n){StringBuilder s=new StringBuilder();int[] v={1000,900,500,400,100,90,50,40,10,9,5,4,1};String[] a={"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};for(int i=0;i<v.length;i++)while(n>=v[i]){s.append(a[i]);n-=v[i];}return s.toString();}
 static String time12(String s){int h=Integer.parseInt(s.substring(0,2)),m=Integer.parseInt(s.substring(3));return String.format(Locale.ROOT,"%02d:%02d %s",h%12==0?12:h%12,m,h<12?"AM":"PM");}
 static boolean isbn(String s){s=s.replace("-","").replace(" ","");if(s.length()!=10)return false;int sum=0;for(int i=0;i<10;i++){char c=s.charAt(i);int v=c=='X'&&i==9?10:c>='0'&&c<='9'?c-'0':-1;if(v<0)return false;sum+=(10-i)*v;}return sum%11==0;}
 static List<Object> transpose(int[][] a){int r=a.length,c=r==0?0:a[0].length;int[][] b=new int[c][r];for(int i=0;i<r;i++)for(int j=0;j<c;j++)b[j][i]=a[i][j];return matrix(b);}
 static String caesar(String s,int shift){int k=Math.floorMod(shift,26);StringBuilder b=new StringBuilder();for(char c:s.toCharArray()){if(c>='a'&&c<='z')c=(char)('a'+(c-'a'+k)%26);else if(c>='A'&&c<='Z')c=(char)('A'+(c-'A'+k)%26);b.append(c);}return b.toString();}
 static List<String> csv(String s){List<String> out=new ArrayList<>();StringBuilder b=new StringBuilder();boolean quote=false;for(int i=0;i<s.length();i++){char c=s.charAt(i);if(c=='"'){if(quote&&i+1<s.length()&&s.charAt(i+1)=='"'){b.append('"');i++;}else quote=!quote;}else if(c==','&&!quote){out.add(b.toString());b.setLength(0);}else b.append(c);}out.add(b.toString());return out;}
 static String network(String ip,int p){String[] a=ip.split("\\.");long v=0;for(String x:a)v=(v<<8)|Integer.parseInt(x);long mask=p==0?0:(0xFFFFFFFFL << (32-p))&0xFFFFFFFFL;long n=v&mask;return ((n>>>24)&255)+"."+((n>>>16)&255)+"."+((n>>>8)&255)+"."+(n&255)+"/"+p;}
 static List<Object> spiral(int[][] a){List<Object> out=new ArrayList<>();int r=a.length,c=r==0?0:a[0].length;int t=0,b=r-1,l=0,h=c-1;while(t<=b&&l<=h){for(int j=l;j<=h;j++)out.add(a[t][j]);t++;for(int i=t;i<=b;i++)out.add(a[i][h]);h--;if(t<=b){for(int j=h;j>=l;j--)out.add(a[b][j]);b--;}if(l<=h){for(int i=b;i>=t;i--)out.add(a[i][l]);l++;}}return out;}
 static List<Object> rotate(int[][] a){int n=a.length,b[][]=new int[n][n];for(int i=0;i<n;i++)for(int j=0;j<n;j++)b[j][n-1-i]=a[i][j];return matrix(b);}
 static long poly(long[] a,long x){long s=0;for(long v:a)s=s*x+v;return s;}
 static String reduce(String s,int k){StringBuilder chars=new StringBuilder();int[] counts=new int[s.length()+1];int top=0;for(char c:s.toCharArray()){if(top>0&&chars.charAt(top-1)==c){counts[top-1]++;if(counts[top-1]==k){chars.setLength(--top);}}else{chars.append(c);counts[top++]=1;if(k==1)chars.setLength(--top);}}StringBuilder out=new StringBuilder();for(int i=0;i<top;i++)out.append(String.valueOf(chars.charAt(i)).repeat(counts[i]));return out.toString();}
 static int josephus(int n,int k){int x=0;for(int i=1;i<=n;i++)x=(x+k)%i;return x+1;}
 static long sumcount(int[] a,long k){Map<Long,Long> h=new HashMap<>();h.put(0L,1L);long sum=0,c=0;for(int v:a){sum+=v;c+=h.getOrDefault(sum-k,0L);h.merge(sum,1L,Long::sum);}return c;}
 static long paths(int[][] a){int r=a.length;if(r==0)return 0;int c=a[0].length;if(c==0)return 0;long[] dp=new long[c];dp[0]=a[0][0]==1?0:1;for(int i=0;i<r;i++)for(int j=0;j<c;j++)dp[j]=a[i][j]==1?0:dp[j]+(j>0?dp[j-1]:0);return dp[c-1];}
 static List<Object> trie(Map<String,Object> d){Set<String> words=new HashSet<>();for(Object w:li(d.get("words")))words.add(str(w));List<Object> ans=new ArrayList<>();for(Object q:li(d.get("queries"))){Map<String,Object> x=obj(q);String w=str(x.get("word")),type=str(x.get("type"));if(type.equals("exact"))ans.add(words.contains(w));else {boolean yes=false;for(String z:words)if(z.startsWith(w)){yes=true;break;}ans.add(yes);}}return ans;}
 static List<Object> multiply(int[][] a,int[][] b){int r=a.length,c=r==0?0:a[0].length,z=b.length==0?0:b[0].length;int[][] v=new int[r][z];for(int i=0;i<r;i++)for(int t=0;t<c;t++)if(a[i][t]!=0)for(int j=0;j<z;j++)v[i][j]+=a[i][t]*b[t][j];return matrix(v);}
 static boolean sudoku(List<Object> board){for(int i=0;i<9;i++){Set<Character> row=new HashSet<>(),col=new HashSet<>();for(int j=0;j<9;j++){char a=str(board.get(i)).charAt(j),b=str(board.get(j)).charAt(i);if(a!='.'&&!row.add(a)||b!='.'&&!col.add(b))return false;}}for(int r=0;r<9;r+=3)for(int c=0;c<9;c+=3){Set<Character> s=new HashSet<>();for(int i=r;i<r+3;i++)for(int j=c;j<c+3;j++){char x=str(board.get(i)).charAt(j);if(x!='.'&&!s.add(x))return false;}}return true;}
 static long critical(int[] duration,int[][] edges){int n=duration.length;List<List<Integer>> g=new ArrayList<>();for(int i=0;i<n;i++)g.add(new ArrayList<>());int[] indeg=new int[n];for(int[] e:edges){g.get(e[0]).add(e[1]);indeg[e[1]]++;}ArrayDeque<Integer> q=new ArrayDeque<>();long[] dp=new long[n];for(int i=0;i<n;i++){if(indeg[i]==0)q.add(i);dp[i]=duration[i];}long ans=0;while(!q.isEmpty()){int x=q.remove();ans=Math.max(ans,dp[x]);for(int y:g.get(x)){dp[y]=Math.max(dp[y],dp[x]+duration[y]);if(--indeg[y]==0)q.add(y);}}return ans;}
 static class Expr{String s;int p;Expr(String s){this.s=s;}void ws(){while(p<s.length()&&s.charAt(p)==' ')p++;}long parse(){long n=e();ws();if(p!=s.length())throw new IllegalArgumentException("extra");return n;}long e(){ws();long v=term();while(true){ws();if(p>=s.length()||s.charAt(p)!= '+'&&s.charAt(p)!='-')return v;char op=s.charAt(p++);long b=term();v=op=='+'?v+b:v-b;}}long term(){ws();int sign=1;while(p<s.length()&&(s.charAt(p)=='+'||s.charAt(p)=='-')){if(s.charAt(p)=='-')sign=-sign;p++;ws();}long val;if(s.charAt(p)=='('){p++;val=e();ws();if(p>=s.length()||s.charAt(p++)!=')')throw new IllegalArgumentException("paren");}else{val=0;while(p<s.length()&&Character.isDigit(s.charAt(p)))val=val*10+s.charAt(p++)-'0';}return sign*val;}}
 static List<String> restore(String s){List<String> out=new ArrayList<>();for(int a=1;a<=3;a++)for(int b=1;b<=3;b++)for(int c=1;c<=3;c++){int d=s.length()-a-b-c;if(d<1||d>3)continue;String[] x={s.substring(0,a),s.substring(a,a+b),s.substring(a+b,a+b+c),s.substring(a+b+c)};boolean ok=true;for(String z:x)if(z.length()>1&&z.charAt(0)=='0'||Integer.parseInt(z)>255)ok=false;if(ok)out.add(String.join(".",x));}Collections.sort(out);return out;}
 static int distinctPal(String s){Set<String> seen=new HashSet<>();for(int c=0;c<s.length();c++)for(int i=c,j=c;i>=0&&j<s.length()&&s.charAt(i)==s.charAt(j);i--,j++)seen.add(s.substring(i,j+1));for(int c=1;c<s.length();c++)for(int i=c-1,j=c;i>=0&&j<s.length()&&s.charAt(i)==s.charAt(j);i--,j++)seen.add(s.substring(i,j+1));return seen.size();}
 static int maxxor(int[] a){int best=0;for(int i=0;i<a.length;i++)for(int j=i+1;j<a.length;j++)best=Math.max(best,a[i]^a[j]);return best;}
 static Map<String,Object> aggregate(List<Object> events,int lo,int hi){Map<String,Object> ans=new TreeMap<>();for(Object v:events){Map<String,Object> e=obj(v);int t=in(e.get("minute"));if(lo<=t&&t<=hi){String u=str(e.get("user"));ans.put(u,(int)ans.getOrDefault(u,0)+1);}}return ans;}
 static int jumps(int[] a){if(a.length<=1)return 0;int far=0,end=0,step=0;for(int i=0;i<a.length-1;i++){if(i>far)return -1;far=Math.max(far,i+a[i]);if(i==end){end=far;step++;if(end>=a.length-1)return step;}}return -1;}
 static long unionArea(int[][] rect){Set<Integer> xs=new TreeSet<>(),ys=new TreeSet<>();for(int[] r:rect){xs.add(r[0]);xs.add(r[2]);ys.add(r[1]);ys.add(r[3]);}Integer[] x=xs.toArray(new Integer[0]),y=ys.toArray(new Integer[0]);long v=0;for(int i=0;i+1<x.length;i++)for(int j=0;j+1<y.length;j++){for(int[] r:rect)if(r[0]<=x[i]&&x[i+1]<=r[2]&&r[1]<=y[j]&&y[j+1]<=r[3]){v+=(long)(x[i+1]-x[i])*(y[j+1]-y[j]);break;}}return v;}
 static String bigmul(String a,String b){int[] out=new int[a.length()+b.length()];for(int i=a.length()-1;i>=0;i--)for(int j=b.length()-1;j>=0;j--){int t=(a.charAt(i)-'0')*(b.charAt(j)-'0')+out[i+j+1];out[i+j+1]=t%10;out[i+j]+=t/10;}StringBuilder sb=new StringBuilder();int i=0;while(i<out.length&&out[i]==0)i++;for(;i<out.length;i++)sb.append(out[i]);return sb.length()==0?"0":sb.toString();}
 static long inversions(int[] nums){int[] a=nums.clone(),buf=new int[a.length];return merge(a,buf,0,a.length);}static long merge(int[] a,int[] b,int lo,int hi){if(hi-lo<=1)return 0;int m=(hi+lo)/2;long inv=merge(a,b,lo,m)+merge(a,b,m,hi);int i=lo,j=m,k=lo;while(i<m||j<hi){if(j==hi||i<m&&a[i]<=a[j])b[k++]=a[i++];else{b[k++]=a[j++];inv+=m-i;}}System.arraycopy(b,lo,a,lo,hi-lo);return inv;}
 static class BIT{long[] b;BIT(int n){b=new long[n+3];}void add(int i,long x){for(i++;i<b.length;i+=i&-i)b[i]+=x;}long pref(int i){long s=0;for(i++;i>0;i-=i&-i)s+=b[i];return s;}long range(int l,int r){return pref(r)-(l>0?pref(l-1):0);}}
 static List<Object> ranges(int[] a,List<Object> ops){int n=a.length;BIT b1=new BIT(n+1),b2=new BIT(n+1);for(int i=0;i<n;i++)bitRange(b1,b2,i,i,a[i]);List<Object> ans=new ArrayList<>();for(Object v:ops){List<Object> q=li(v);String type=str(q.get(0));int l=in(q.get(1)),r=in(q.get(2));if(type.equals("add"))bitRange(b1,b2,l,r,lng(q.get(3)));else ans.add(bitPref(b1,b2,r)-bitPref(b1,b2,l-1));}return ans;}
 static void bitRange(BIT a,BIT b,int l,int r,long val){a.add(l,val);a.add(r+1,-val);b.add(l,-val*l);b.add(r+1,val*(r+1));}static long bitPref(BIT a,BIT b,int i){if(i<0)return 0;return (i+1)*a.pref(i)+b.pref(i);}
 static int matching(int left,int right,int[][] edges){List<List<Integer>> g=new ArrayList<>();for(int i=0;i<left;i++)g.add(new ArrayList<>());for(int[] e:edges)g.get(e[0]).add(e[1]);int[] match=new int[right];Arrays.fill(match,-1);int count=0;for(int i=0;i<left;i++)if(aug(i,g,match,new boolean[right]))count++;return count;}
 static boolean aug(int v,List<List<Integer>> g,int[] mt,boolean[] seen){for(int w:g.get(v))if(!seen[w]){seen[w]=true;if(mt[w]<0||aug(mt[w],g,mt,seen)){mt[w]=v;return true;}}return false;}
 static long balloons(int[] a){int n=a.length;long[][] dp=new long[n+2][n+2];int[] b=new int[n+2];b[0]=b[n+1]=1;System.arraycopy(a,0,b,1,n);for(int len=1;len<=n;len++)for(int l=1;l+len-1<=n;l++){int r=l+len-1;for(int k=l;k<=r;k++)dp[l][r]=Math.max(dp[l][r],dp[l][k-1]+dp[k+1][r]+(long)b[l-1]*b[k]*b[r+1]);}return dp[1][n];}
 static long queens(int n,int[][] blocked){int[] forbid=new int[n];for(int[] rc:blocked)forbid[rc[0]]|=1<<rc[1];return queensDfs(0,n,0,0,0,forbid);}static long queensDfs(int row,int n,int cols,int d1,int d2,int[] forbid){if(row==n)return 1;int avail=((1<<n)-1)&~(cols|d1|d2|forbid[row]);long ans=0;while(avail!=0){int p=avail&-avail;avail-=p;ans+=queensDfs(row+1,n,cols|p,(d1|p)<<1,(d2|p)>>1,forbid);}return ans;}
 static List<Object> paren(String s){List<Long> v=results(s);Collections.sort(v);return new ArrayList<>(v);}static List<Long> results(String s){List<Long> ans=new ArrayList<>();for(int i=0;i<s.length();i++){char op=s.charAt(i);if(op=='+'||op=='-'||op=='*'){for(long a:results(s.substring(0,i)))for(long b:results(s.substring(i+1))){ans.add(op=='+'?a+b:op=='-'?a-b:a*b);}}}if(ans.isEmpty())ans.add(Long.parseLong(s));return ans;}
 static long subseq(String s,String t){long[] dp=new long[t.length()+1];dp[0]=1;for(char c:s.toCharArray())for(int j=t.length()-1;j>=0;j--)if(c==t.charAt(j))dp[j+1]+=dp[j];return dp[t.length()];}
 static int lps(String s){int n=s.length();if(n==0)return 0;int[][] dp=new int[n][n];for(int i=n-1;i>=0;i--){dp[i][i]=1;for(int j=i+1;j<n;j++)dp[i][j]=s.charAt(i)==s.charAt(j)?2+(j-i>1?dp[i+1][j-1]:0):Math.max(dp[i+1][j],dp[i][j-1]);}return dp[0][n-1];}
 static long jobs(int[][] a){Arrays.sort(a,Comparator.comparingInt(x->x[1]));int n=a.length;long[] dp=new long[n+1];for(int i=0;i<n;i++){int j=0;while(j<i&&a[j][1]<=a[i][0])j++;dp[i+1]=Math.max(dp[i],dp[j]+a[i][2]);}return dp[n];}
}
