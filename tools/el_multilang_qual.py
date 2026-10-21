"""[MULTILANG 2026-10-21] Qualification of the trusted verifier for C, C++, JavaScript, Go and Rust on one sandbox backend. Sibling of tools/el_kaggle_port_qual.py (which qualifies python).
Per language: the toolchain probe inside the backend (a language whose probe fails is reported `unavailable` with the exact reason, never silently moved to another backend), then 14 checks through
ProviderExecutionBackend(trusted=True) -> TrustedVerifier -> trusted_native (engine.judge_each verdicts) and LcbRunner (LiveCodeBench stdin comparison):
  honest_pass, wrong_answer_fails, forged_frames_never_pass, claim_then_crash_never_passes, hang_is_killed, compile_error_is_candidate_outcome, fresh_process_per_input,
  socket_denied (a host-side listener must never see a connection), host_paths_unreadable (canary files outside the workspace), expected_never_staged (a spy on every sandbox call scans the staged files
  and the command for the expected canary), large_output_never_silently_truncated, lcb_stdin_honest_pass, lcb_stdin_wrong_fails, lcb_stdin_whitespace_rules.
Usage:  python tools/el_multilang_qual.py <out dir> --backend wsl|mxc|wslc|kaggle-jail [--languages c,cpp,javascript,go,rust]
Writes <out dir>/QUAL_<backend>.json ; exit 0 only if every language that was runnable is fully qualified (an `unavailable` language does not fail the run, it is reported)."""
import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_ladder.execution import config as K, engine, trusted_native as N                # noqa: E402
from evidence_ladder.execution.lcb_runner import LcbRunner                                       # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend, ProviderEngineError   # noqa: E402

LANGS = ("c", "cpp", "javascript", "go", "rust")
EXT = {"c": ".c", "cpp": ".cpp", "javascript": ".js", "go": ".go", "rust": ".rs"}
CANARY = "EXPECTED-CANARY-77"
CASES = [{"input": "1 2\n", "expected": "3\n"}, {"input": "3 4\n", "expected": "7\n"}, {"input": "10 20\n", "expected": "30\n"}]

# ------------------------------------------------------------------ candidate programs: stdin "a b" -> stdout; each body replaces {BODY}
C_HEAD = ('#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <unistd.h>\n#include <sys/socket.h>\n#include <netinet/in.h>\n#include <arpa/inet.h>\n'
          'int main(void){ long a,b; if(scanf("%ld %ld",&a,&b)!=2) return 2;\n{BODY}\nreturn 0; }\n')
C_BODY = {
    "honest": 'printf("%ld\\n",a+b);',
    "wrong": 'printf("%ld\\n",a+b+1);',
    "forged": 'printf("@@TV@@ V %ld\\n@@LCB@@ %ld\\n%ld\\n",a+b,a+b,a+b+1);',
    "crash": 'printf("%ld\\n",a+b); fflush(stdout); _exit(3);',
    "hang": 'for(;;){ }',
    "socket": ('int s=socket(AF_INET,SOCK_STREAM,0); int ok=0; if(s>=0){ struct sockaddr_in x; memset(&x,0,sizeof x); x.sin_family=AF_INET; x.sin_port=htons({PORT}); inet_pton(AF_INET,"127.0.0.1",&x.sin_addr);'
               ' ok=(connect(s,(struct sockaddr*)&x,sizeof x)==0); } puts(ok?"ALLOWED":"denied");'),
    "paths": ('const char *P[]={{PATHS}}; int n=0; for(int i=0;i<{NPATHS};i++){ FILE*f=fopen(P[i],"r"); if(f){ n++; fclose(f);} } puts(n?"READABLE":"denied");'),
    "state": 'long c=0; FILE*f=fopen("state.txt","r"); if(f){ fscanf(f,"%ld",&c); fclose(f);} c++; f=fopen("state.txt","w"); if(f){ fprintf(f,"%ld",c); fclose(f);} printf("%ld\\n",c);',
    "canary": 'printf("nothing\\n");',
    "big": 'for(int i=0;i<10000;i++) putchar(\'x\'); putchar(\'\\n\');',
}
GO_BODY = {
    "honest": ('"fmt"', 'var a,b int64; fmt.Scan(&a,&b); fmt.Println(a+b)'),
    "wrong": ('"fmt"', 'var a,b int64; fmt.Scan(&a,&b); fmt.Println(a+b+1)'),
    "forged": ('"fmt"', 'var a,b int64; fmt.Scan(&a,&b); fmt.Printf("@@TV@@ V %d\\n@@LCB@@ %d\\n%d\\n",a+b,a+b,a+b+1)'),
    "crash": ('"fmt"\n"os"', 'var a,b int64; fmt.Scan(&a,&b); fmt.Println(a+b); os.Exit(3)'),
    "hang": ('', 'for { }'),
    "socket": ('"fmt"\n"net"\n"time"', 'c,err:=net.DialTimeout("tcp","127.0.0.1:{PORT}",2*time.Second); if err==nil { c.Close(); fmt.Println("ALLOWED") } else { fmt.Println("denied") }'),
    "paths": ('"fmt"\n"os"', 'P:=[]string{{PATHS}}; n:=0; for _,p:=range P { f,err:=os.Open(p); if err==nil { n++; f.Close() } }; if n>0 { fmt.Println("READABLE") } else { fmt.Println("denied") }'),
    "state": ('"fmt"\n"os"\n"strconv"\n"strings"', 'c:=0; if d,err:=os.ReadFile("state.txt"); err==nil { c,_=strconv.Atoi(strings.TrimSpace(string(d))) }; c++; os.WriteFile("state.txt",[]byte(strconv.Itoa(c)),0644); fmt.Println(c)'),
    "canary": ('"fmt"', 'fmt.Println("nothing")'),
    "big": ('"strings"\n"fmt"', 'fmt.Println(strings.Repeat("x",10000))'),
}
RS_BODY = {
    "honest": 'println!("{}", a+b);',
    "wrong": 'println!("{}", a+b+1);',
    "forged": 'println!("@@TV@@ V {}\\n@@LCB@@ {}\\n{}", a+b, a+b, a+b+1);',
    "crash": 'println!("{}", a+b); std::process::exit(3);',
    "hang": 'loop { std::hint::black_box(0); }',
    "socket": ('let ok = std::net::TcpStream::connect_timeout(&"127.0.0.1:{PORT}".parse().unwrap(), std::time::Duration::from_secs(2)).is_ok(); println!("{}", if ok {"ALLOWED"} else {"denied"});'),
    "paths": 'let p = [{PATHS}]; let n = p.iter().filter(|x| std::fs::File::open(x).is_ok()).count(); println!("{}", if n>0 {"READABLE"} else {"denied"});',
    "state": ('let c: i64 = std::fs::read_to_string("state.txt").ok().and_then(|t| t.trim().parse().ok()).unwrap_or(0) + 1; let _ = std::fs::write("state.txt", c.to_string()); println!("{}", c);'),
    "canary": 'println!("nothing");',
    "big": 'println!("{}", "x".repeat(10000));',
}
RS_HEAD = ('use std::io::Read;\nfn main(){ let mut s=String::new(); std::io::stdin().read_to_string(&mut s).unwrap(); let v: Vec<i64> = s.split_whitespace().map(|x| x.parse().unwrap()).collect();\n'
           'let (a,b)=(v[0],v[1]);\n{BODY}\n}\n')
JS_HEAD = 'const fs=require("fs"); const [a,b]=fs.readFileSync(0,"utf8").trim().split(/\\s+/).map(Number);\n{BODY}\n'
JS_BODY = {
    "honest": 'console.log(a+b);',
    "wrong": 'console.log(a+b+1);',
    "forged": 'console.log("@@TV@@ V "+(a+b)+"\\n@@LCB@@ "+(a+b)+"\\n"+(a+b+1));',
    "crash": 'fs.writeSync(1,(a+b)+"\\n"); process.exit(3);',
    "hang": 'for(;;){}',
    "socket": ('const net=require("net"); const c=net.connect({port:{PORT},host:"127.0.0.1"}); c.on("connect",()=>{console.log("ALLOWED");c.destroy();}); c.on("error",()=>console.log("denied")); setTimeout(()=>{console.log("denied");process.exit(0)},3000).unref();'),
    "paths": 'const P=[{PATHS}]; let n=0; for(const p of P){ try{ fs.openSync(p,"r"); n++; }catch(e){} } console.log(n?"READABLE":"denied");',
    "state": 'let c=0; try{ c=parseInt(fs.readFileSync("state.txt","utf8")) }catch(e){} c++; try{ fs.writeFileSync("state.txt",String(c)) }catch(e){} console.log(c);',
    "canary": 'console.log("nothing");',
    "big": 'console.log("x".repeat(10000));',
}


def program(lang: str, kind: str, port: int = 0, paths=()) -> str:
    sub = {"{PORT}": str(port), "{PATHS}": ",".join(json.dumps(p, ensure_ascii=True) for p in paths), "{NPATHS}": str(len(paths))}

    def fill(t):
        for k, v in sub.items():
            t = t.replace(k, v)
        return t
    if lang in ("c", "cpp"):
        return fill(C_HEAD.replace("{BODY}", C_BODY[kind]))
    if lang == "go":
        imp, body = GO_BODY[kind]
        return fill("package main\nimport (\n%s\n)\nfunc main(){\n%s\n}\n" % (imp, body))
    if lang == "rust":
        return fill(RS_HEAD.replace("{BODY}", RS_BODY[kind]))
    return fill(JS_HEAD.replace("{BODY}", JS_BODY[kind]))


# ------------------------------------------------------------------ backends
class Spy:
    """wraps a sandbox's .run: scans every staged file and the command for the expected canary (the supervisor must never stage expected values)"""
    def __init__(self, inner):
        self.inner, self.hits, self.calls, self.env_label = inner, [], 0, getattr(inner, "env_label", "")

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.calls += 1
        if CANARY in cmd:
            self.hits.append("cmd")
        for f in Path(workspace).rglob("*"):
            if f.is_file() and CANARY.encode() in f.read_bytes():
                self.hits.append(f.name)
        return self.inner.run(workspace, cmd, persist=persist, timeout=timeout)

    def provenance(self):
        return self.inner.provenance() if hasattr(self.inner, "provenance") else {}

    def close(self):
        c = getattr(self.inner, "close", None)
        if c:
            c()


def make_backend(name: str, scratch: Path):
    """-> (sandbox, capture_limit, note). Raises on unavailability; the caller reports it (no fallback)."""
    if name == "wsl":
        from evidence_ladder.controller.wsl_sandbox import WslSandbox, available
        if not available():
            raise RuntimeError("wsl does not answer")
        return WslSandbox(scratch, []), 6000, "WSL2 Ubuntu-24.04 systemd + bubblewrap (reference engine); toolchains from /opt/toolchains and /usr"
    if name == "mxc":
        from evidence_ladder.controller.sandbox_provider import make_sandbox
        node = shutil.which("node")
        return make_sandbox(scratch, [], backend="mxc", tool_dirs=[str(Path(node).parent)] if node else []), 6000, "Windows MXC processcontainer; node from the Windows host"
    if name == "wslc":
        from evidence_ladder.controller.sandbox_provider import make_sandbox
        return make_sandbox(scratch, [], backend="wslc"), 6000, "MXC WSLC pinned local image"
    if name == "bare":      # NEGATIVE CONTROL, never a qualification: no isolation at all. socket_denied and host_paths_unreadable MUST fail here, which proves those checks can detect a leak.
        import signal
        import subprocess

        class BarePort:
            env_label = "bare-control"

            def run(self, workspace, cmd, persist=False, timeout=60):
                p = subprocess.Popen(["bash", "-c", cmd], cwd=str(workspace), stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                killed = False
                try:
                    out, err = p.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    killed = True
                    os.killpg(p.pid, signal.SIGKILL)
                    out, err = p.communicate()
                return {"exit_code": -9 if killed else p.returncode, "killed": killed, "stdout": out.decode("utf-8", "replace")[-6000:], "stderr": err.decode("utf-8", "replace")[-6000:],
                        "protected_unchanged": True, "pre_tree_sha256": "", "post_tree_sha256": "", "persisted": bool(persist)}
        return BarePort(), 6000, "NO ISOLATION (negative control)"
    if name == "kaggle-jail":
        from evidence_ladder.execution.kaggle_port import KaggleJailPort
        from evidence_ladder.execution import kaggle_jail as KJ
        if not (KJ.ROOTFS / "usr" / "bin" / "python3").exists():
            raise RuntimeError("no kaggle-jail rootfs at %s (build it: python -c \"from evidence_ladder.execution import kaggle_jail as K; K.build_rootfs(toolchains=(...))\")" % KJ.ROOTFS)
        return KaggleJailPort(), 1 << 20, "Kaggle chroot+uid+seccomp jail (rootfs %s)" % KJ.ROOTFS
    raise SystemExit("unknown backend %r" % name)


class Listener:
    def __init__(self):
        self.s = socket.socket()
        self.s.bind(("0.0.0.0", 0))
        self.s.listen(8)
        self.port, self.accepted = self.s.getsockname()[1], 0
        self.s.settimeout(0.2)
        self.stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self.stop:
            try:
                c, _ = self.s.accept()
                self.accepted += 1
                c.close()
            except OSError:
                pass

    def control(self) -> bool:
        """positive control: a plain unsandboxed connection from this very process reaches the listener"""
        before = self.accepted
        c = socket.create_connection(("127.0.0.1", self.port), timeout=3)
        c.close()
        time.sleep(0.5)
        return self.accepted > before

    def close(self):
        self.stop = True
        self.s.close()


def wsl_form(p: str) -> str:
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/") if os.name == "nt" and len(p) > 2 and p[1] == ":" else p


# ------------------------------------------------------------------ the suite for one language
def qualify_language(lang: str, be: ProviderExecutionBackend, spy: Spy, listener: Listener, canary_paths, cfg, capture_limit: int):
    tv = be.trusted_verifier
    tv.capture_limit = capture_limit
    checks = {}

    def verdict(src: str, cases=CASES, per_input=None):
        tv.per_input_timeout = float(per_input or cfg.py_timeout)
        t0 = time.time()
        res = be._engine_run(lang, src.encode(), [c["input"] for c in cases])
        v, each = engine.judge_each(lang, res, cases, cfg)
        return v, each, res, time.time() - t0

    def check(name, fn):
        t0 = time.time()
        try:
            r = fn()
            checks[name] = dict(r if isinstance(r, dict) else {"ok": bool(r)}, seconds=round(time.time() - t0, 2))
        except Exception as e:      # noqa: BLE001
            checks[name] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300]), "seconds": round(time.time() - t0, 2)}

    def outcome(v):
        return {"status": v.status, "failure_class": v.failure_class}

    def honest():
        v, each, res, dt = verdict(program(lang, "honest"))
        return {"ok": v.status == "pass" and all(e[0] == "pass" for e in each), **outcome(v)}
    check("honest_pass", honest)

    def wrong():
        v, *_ = verdict(program(lang, "wrong"))
        return {"ok": v.status == "fail" and v.failure_class == "wrong_answer", **outcome(v)}
    check("wrong_answer_fails", wrong)

    def forged():
        v, *_ = verdict(program(lang, "forged"))
        return {"ok": v.status != "pass", **outcome(v)}
    check("forged_frames_never_pass", forged)

    def crash():
        v, each, *_ = verdict(program(lang, "crash"))
        return {"ok": v.status != "pass" and all(e[0] != "pass" for e in each), **outcome(v)}
    check("claim_then_crash_never_passes", crash)

    def hang():
        v, each, res, dt = verdict(program(lang, "hang"), per_input=2.0)
        return {"ok": v.status == "timeout" and dt < 60, **outcome(v), "wall_s": round(dt, 1)}
    check("hang_is_killed", hang)

    def compile_error():
        bad = "this is not a valid program {{{ @@@\n"
        v, *_ = verdict(bad)
        # compiled languages: compile_error; javascript: a SyntaxError exit at run time is a runtime_error. Either is a candidate failure, never a pass and never an infrastructure raise
        return {"ok": v.status != "pass" and v.failure_class in ("compile_error", "runtime_error"), **outcome(v)}
    check("compile_error_is_candidate_outcome", compile_error)

    def fresh():
        v, each, res, dt = verdict(program(lang, "state"), cases=[{"input": "1 1\n", "expected": "1\n"}] * 3)
        return {"ok": v.status == "pass", **outcome(v), "observed": [e[2] for e in each]}
    check("fresh_process_per_input", fresh)

    def sock():
        before = listener.accepted
        v, each, res, dt = verdict(program(lang, "socket", port=listener.port), cases=[{"input": "1 1\n", "expected": "denied\n"}])
        return {"ok": v.status == "pass" and listener.accepted == before, **outcome(v), "listener_connections": listener.accepted - before, "observed": [e[2] for e in each]}
    check("socket_denied", sock)

    def paths():
        v, each, res, dt = verdict(program(lang, "paths", paths=canary_paths), cases=[{"input": "1 1\n", "expected": "denied\n"}])
        return {"ok": v.status == "pass", **outcome(v), "observed": [e[2] for e in each]}
    check("host_paths_unreadable", paths)

    def staged():
        n0 = len(spy.hits)
        v, *_ = verdict(program(lang, "canary"), cases=[{"input": "1 1\n", "expected": CANARY + "\n"}])
        return {"ok": v.status == "fail" and len(spy.hits) == n0 and spy.calls > 0, **outcome(v), "spy_hits": spy.hits[n0:]}
    check("expected_never_staged", staged)

    def big():
        want = "x" * 10000 + "\n"
        v, each, *_ = verdict(program(lang, "big"), cases=[{"input": "1 1\n", "expected": want}])
        # allowed: pass (the port captured everything) or invalid_output (the port may have truncated); never any other pass-like outcome
        return {"ok": (v.status == "pass") or (v.status == "fail" and v.failure_class == "invalid_output"), **outcome(v), "capture_limit": capture_limit}
    check("large_output_never_silently_truncated", big)

    lr = LcbRunner(be.sandbox.run, Path(tempfile.mkdtemp(prefix="mlq-lcb-")))
    lr.capture_limit = capture_limit
    lcb_case = {"input": "2 3\n", "output": "5\n", "testtype": "stdin"}

    def lcb_ok():
        g = lr.grade(program(lang, "honest"), [lcb_case, {"input": "10 20\n", "output": "30", "testtype": "stdin"}], None, language=lang)
        return {"ok": bool(g["pass"]), "passed": g["passed"], "total": g["total"]}
    check("lcb_stdin_honest_pass", lcb_ok)

    def lcb_wrong():
        g = lr.grade(program(lang, "wrong"), [lcb_case], None, language=lang)
        return {"ok": (not g["pass"]) and (g["first_fail"] or {}).get("status") == "wrong_answer"}
    check("lcb_stdin_wrong_fails", lcb_wrong)

    def lcb_ws():
        # LiveCodeBench strips each line: expected "5 " / trailing newline differences must not matter, a different number must
        g1 = lr.grade(program(lang, "honest"), [{"input": "2 3\n", "output": "5  \n\n", "testtype": "stdin"}], None, language=lang)
        g2 = lr.grade(program(lang, "honest"), [{"input": "2 3\n", "output": "6\n", "testtype": "stdin"}], None, language=lang)
        return {"ok": bool(g1["pass"]) and not g2["pass"]}
    check("lcb_stdin_whitespace_rules", lcb_ws)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", required=True, choices=("wsl", "mxc", "wslc", "kaggle-jail", "bare"))
    ap.add_argument("--languages", default=",".join(LANGS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / ("QUAL_%s.json" % a.backend)
    if dest.exists():
        sys.exit("refusing to overwrite %s" % dest)
    langs = [l for l in a.languages.split(",") if l]
    cfg = K.SandboxConfig()
    rec = {"backend": a.backend, "verifier": "trusted_verify_v3 + %s" % N.VERSION, "launcher_sha256": N.LAUNCHER_SHA256, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "languages": {}}
    scratch = Path(tempfile.mkdtemp(prefix="mlq-"))
    try:
        sb, cap, note = make_backend(a.backend, scratch / "sb")
    except Exception as e:      # noqa: BLE001  the backend itself cannot run here: every language is `unavailable` for this backend, nothing falls back
        rec["backend_error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        for l in langs:
            rec["languages"][l] = {"state": "unavailable", "reason": "backend unavailable: " + rec["backend_error"]}
        dest.write_text(json.dumps(rec, indent=1))
        print(json.dumps({l: v["state"] for l, v in rec["languages"].items()}))
        return 0
    spy = Spy(sb)
    be = ProviderExecutionBackend(spy, backend=a.backend, scratch=scratch / "pe", trusted=True)
    rec.update(note=note, capture_limit=cap)
    listener = Listener()
    cdir = Path(tempfile.mkdtemp(prefix="mlq-canary-"))
    (cdir / "canary.txt").write_text("HOSTCANARY\n")
    cp = str(cdir / "canary.txt")
    canary_paths = [cp, wsl_form(cp), "/root", "/kaggle/input", "/proc/1/environ"] if os.name != "nt" else [cp, wsl_form(cp), cp.replace("\\", "/")]
    rec["listener_positive_control"] = listener.control()
    all_ok = True
    probes = be.probe_toolchains(langs)
    for lang in langs:
        p = probes[lang]
        if not p.get("runnable"):
            rec["languages"][lang] = {"state": "unavailable", "status": p.get("status"), "reason": p.get("reason")}
            continue
        try:
            checks = qualify_language(lang, be, spy, listener, canary_paths, cfg, cap)
        except ProviderEngineError as e:
            checks = {"suite": {"ok": False, "error": "infrastructure: " + str(e)[:300]}}
        ok = all(c["ok"] for c in checks.values())
        all_ok &= ok
        rec["languages"][lang] = {"state": "qualified" if ok else "failing", "version": p.get("version"), "checks": checks}
        print(lang, rec["languages"][lang]["state"], {k: v["ok"] for k, v in checks.items() if not v["ok"]}, flush=True)
    rec["spy_expected_hits"] = spy.hits
    rec["sandbox_calls"] = spy.calls
    rec["all_runnable_ok"] = bool(all_ok and not spy.hits)
    listener.close()
    spy.close()
    dest.write_text(json.dumps(rec, indent=1))
    print(json.dumps({l: v["state"] for l, v in rec["languages"].items()}))
    return 0 if rec["all_runnable_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
