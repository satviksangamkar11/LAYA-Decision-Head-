import java.util.*;import java.io.*;
public final class ReferenceRunner {
  @SuppressWarnings("unchecked") public static void main(String[] args) throws Exception {
    try(BufferedReader br=new BufferedReader(new InputStreamReader(System.in, java.nio.charset.StandardCharsets.UTF_8))){
      String line; while((line=br.readLine())!=null){Map<String,Object> req=(Map<String,Object>)Json.parse(line);
      Object output=ReferenceSolutions.solve(((Number)req.get("task_number")).intValue(), (Map<String,Object>)req.get("input"));
      System.out.println(Json.stringify(output));}
    }
  }
}
