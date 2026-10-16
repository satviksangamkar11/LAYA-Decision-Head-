import java.util.*;import java.io.*;
public final class Runner {
  @SuppressWarnings("unchecked") public static void main(String[] args) throws Exception {
    try(BufferedReader br = new BufferedReader(new InputStreamReader(System.in,java.nio.charset.StandardCharsets.UTF_8))){
      String line;while((line=br.readLine())!=null){
        try {Object data=Json.parse(line);Object result=Solution.solve((Map<String,Object>)data); System.out.println(Json.stringify(result));}
        catch(Throwable e){System.err.println("Candidate failed: "+e);System.exit(2);}
      }
    }
  }
}
