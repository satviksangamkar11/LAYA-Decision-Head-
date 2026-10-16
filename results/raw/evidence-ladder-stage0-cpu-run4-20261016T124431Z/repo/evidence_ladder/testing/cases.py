"""Fixed case table shared by the recorder (WSL) and the replay suite (Kaggle). Pure data. Sources have NO trailing newline: the Qwen code extractor
joins fence lines with "\n", so a generation carrying these sources reproduces them byte for byte (hash keys match)."""
PY_TASK, JAVA_TASK = "P1-01", "J1-02"

PY_REF = "def solve(d): return sum(c in 'aeiouAEIOU' for c in d['text'])"   # equals oracle p01 (recorder asserts it)
_JPRE = "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ int y=((Number)d.get(\"year\")).intValue(); "
JAVA_REF = _JPRE + "return (y%400==0)||(y%4==0&&y%100!=0); } }"
JAVA_WRONG = _JPRE + "return y%4==0; } }"

# name -> (task_id, language, source, note)
CASES = {
    "py_ref": (PY_TASK, "python", PY_REF, "correct (oracle p01)"),
    "py_wrong_y": (PY_TASK, "python", "def solve(d): return sum(c in 'aeiouyAEIOUY' for c in d['text'])", "known wrong: counts y"),
    "py_wrong_y_v2": (PY_TASK, "python", "def solve(d): return sum(c in 'aeiouyAEIOUY' for c in d['text'])\n# variant 2", "same failure, different source"),
    "py_lower_first": (PY_TASK, "python", "def solve(d): return sum(c in 'aeiou' for c in d['text'].lower())", "passes public, fails protected"),
    "py_const99": (PY_TASK, "python", "def solve(d): return 99", "known wrong: constant"),
    "py_syntax": (PY_TASK, "python", "def solve(d) return 1", "syntax error"),
    "py_timeout": (PY_TASK, "python", "def solve(d):\n    while True: pass", "never returns"),
    "py_crash": (PY_TASK, "python", "def solve(d): raise ValueError('x')", "runtime error"),
    "java_ref": (JAVA_TASK, "java", JAVA_REF, "correct (hand-written leap-year rule)"),
    "java_wrong": (JAVA_TASK, "java", JAVA_WRONG, "known wrong: no century rule"),
    "java_wrong_v2": (JAVA_TASK, "java", JAVA_WRONG + "\n// variant 2", "same failure, different source"),
    "java_wrong_v3": (JAVA_TASK, "java", JAVA_WRONG + "\n// variant 3", "same failure, different source"),
    "java_compile_error": (JAVA_TASK, "java", "public final class Solution { broken", "compile error"),
}
EXPERIMENTS = {
    "python": [("function_call", '{"text": "hello world"}'), ("stdin_input", '{"text": "yellow"}'), ("function_call", "{}")],
    "java": [("function_call", '{"year": 2000}'), ("stdin_input", '{"year": 1900}'), ("function_call", "{}")],
}
