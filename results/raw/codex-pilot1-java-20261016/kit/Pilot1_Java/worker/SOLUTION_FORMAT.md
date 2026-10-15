# Java 21 solution interface

For each task return a **single complete Java source file** defining this exact public class and method:

```java
import java.util.*;
public final class Solution {
    public static Object solve(Map<String,Object> data) {
        // your implementation
        return null;
    }
}
```

The grader supplies `Json.java` and `Runner.java`. `Runner` reads newline-delimited JSON on standard input and calls `Solution.solve(data)` once per input. A task's submission is named `J1-01.java`, `J1-02.java`, ... `J1-40.java` in the submitted folder; the grader temporarily copies each submission to `Solution.java` before compilation.

Parser representation: JSON numeric values are Java `Number` (cast via `((Number)x).intValue()` or `.longValue()`); JSON arrays are `List<?>` and objects are `Map<?,?>`. Return only JSON-serializable types. Java 21 standard library allowed. No network, file I/O, subprocesses, reflective access to private grader internals or third-party dependencies.

**Strict separation:** Worker receives this folder only, never `grader/`. Do not expose any protected development or sealed evaluation cases during generation. Do not compile arbitrary untrusted submissions on a machine containing secrets or hidden tests.
