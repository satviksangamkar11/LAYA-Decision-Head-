import java.util.*;
import java.math.*;
public final class Json {
  private Json(){}
  public static Object parse(String input) { P p=new P(input); Object o=p.value();p.ws();if(p.i!=input.length())throw new IllegalArgumentException("Trailing JSON");return o; }
  private static final class P {
    final String s; int i=0; P(String s){this.s=s;}
    void ws(){while(i<s.length()&&Character.isWhitespace(s.charAt(i)))i++;}
    char next(){return i<s.length()?s.charAt(i):'\0';}
    void eat(char c){ws();if(next()!=c)throw new IllegalArgumentException("Expected "+c+" at "+i);i++;}
    Object value(){ws();char c=next();if(c=='"')return string(); if(c=='{')return map();if(c=='[')return array();if(c=='t'&&s.startsWith("true",i)){i+=4;return true;}if(c=='f'&&s.startsWith("false",i)){i+=5;return false;}if(c=='n'&&s.startsWith("null",i)){i+=4;return null;}return number();}
    Object number(){int st=i; if(next()=='-')i++;while(i<s.length()&&Character.isDigit(next()))i++;if(next()=='.'){i++;while(i<s.length()&&Character.isDigit(next()))i++;}if(next()=='e'||next()=='E'){i++;if(next()=='+'||next()=='-')i++;while(i<s.length()&&Character.isDigit(next()))i++;}if(i==st)throw new IllegalArgumentException("Invalid number at "+i);String t=s.substring(st,i);return (t.contains(".")||t.contains("e")||t.contains("E"))?new BigDecimal(t):Long.valueOf(t);}
    String string(){eat('"');StringBuilder b=new StringBuilder();while(true){if(i>=s.length())throw new IllegalArgumentException("Incomplete string");char c=s.charAt(i++);if(c=='"')break;if(c=='\\'){if(i>=s.length())throw new IllegalArgumentException("Incomplete escape");c=s.charAt(i++);switch(c){case 'n':b.append('\n');break;case 'r':b.append('\r');break;case 't':b.append('\t');break;case 'b':b.append('\b');break;case 'f':b.append('\f');break;case 'u':if(i+4>s.length())throw new IllegalArgumentException("Bad unicode");b.append((char)Integer.parseInt(s.substring(i,i+4),16));i+=4;break;default:b.append(c);}}else{b.append(c);}}return b.toString();}
    Object array(){eat('[');List<Object> a=new ArrayList<>();ws();if(next()==']'){i++;return a;}while(true){a.add(value());ws();char c=next();if(c==']'){i++;return a;}eat(',');}}
    Object map(){eat('{');Map<String,Object> m=new LinkedHashMap<>();ws();if(next()=='}'){i++;return m;}while(true){String k=string();eat(':');m.put(k,value());ws();if(next()=='}'){i++;return m;}eat(',');}}
  }
  public static String stringify(Object o){StringBuilder b=new StringBuilder();add(b,o);return b.toString();}
  private static void add(StringBuilder b,Object o){if(o==null){b.append("null");return;}if(o instanceof String s){b.append('"');for(int i=0;i<s.length();i++){char c=s.charAt(i);switch(c){case '"':b.append("\\\"");break;case '\\':b.append("\\\\");break;case '\n':b.append("\\n");break;case '\r':b.append("\\r");break;case '\t':b.append("\\t");break;case '\b':b.append("\\b");break;case '\f':b.append("\\f");break;default:if(c<32){b.append(String.format("\\u%04x",(int)c));}else b.append(c);}}b.append('"');return;}
   if(o instanceof Number||o instanceof Boolean){b.append(o);return;}
   if(o instanceof Map<?,?> m){b.append('{');boolean first=true;for(var e:m.entrySet()){if(!first)b.append(',');first=false;add(b,String.valueOf(e.getKey()));b.append(':');add(b,e.getValue());}b.append('}');return;}
   if(o instanceof Iterable<?> it){b.append('[');boolean first=true;for(Object e:it){if(!first)b.append(',');first=false;add(b,e);}b.append(']');return;}
   if(o.getClass().isArray()){b.append('[');int n=java.lang.reflect.Array.getLength(o);for(int i=0;i<n;i++){if(i>0)b.append(',');add(b,java.lang.reflect.Array.get(o,i));}b.append(']');return;}
   throw new IllegalArgumentException("Unsupported output: "+o.getClass());
  }
}
