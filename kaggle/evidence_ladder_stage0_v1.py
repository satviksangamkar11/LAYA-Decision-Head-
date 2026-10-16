"""Kaggle kernel A, pilot Stage 0: FC-1 (jail feasibility), FC-3 (Qwen throughput, only if a GPU and a mounted Qwen snapshot exist), FC-4 (protected-test separation), then the
execution/red-team suites through the Kaggle jail. Internet OFF, NO secrets, no NVIDIA calls. Input: dataset with MANIFEST.json (bundle evidence_ladder_stage0_v1) + code tree.
Writes /kaggle/working/STAGE0_RESULTS.json after every phase (so partial results survive a crash)."""
import glob, hashlib, json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working"); t0 = time.time()
RES = {"python": sys.version, "platform": platform.platform(), "uid": os.getuid(), "phases": {}}
def save(): (OUT / "STAGE0_RESULTS.json").write_text(json.dumps(RES, indent=1, default=str), encoding="utf-8")
def phase(name):
    def deco(f):
        t = time.time()
        try:
            RES["phases"][name] = f()
        except BaseException as e:
            RES["phases"][name] = {"status": "CRASH", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        RES["phases"][name]["seconds"] = round(time.time() - t, 1); save(); return f
    return deco

man_path = next((p for p in sorted(Path("/kaggle/input").glob("**/MANIFEST.json")) if json.loads(p.read_text(encoding="utf-8")).get("bundle") == "evidence_ladder_stage0_v1"), None)
if man_path is None:
    RES["fatal"] = "bundle not found"; save(); sys.exit("bundle not found")
man = json.loads(man_path.read_text(encoding="utf-8")); src = man_path.parent
bad = [f for f, h in man["files"].items() if not (src / f).is_file() or hashlib.sha256((src / f).read_bytes()).hexdigest() != h]
RES["manifest_files"] = len(man["files"]); RES["hash_mismatches"] = bad
if bad:
    save(); sys.exit("bundle hash mismatch")
REPO = OUT / "repo"
if REPO.exists(): shutil.rmtree(REPO)
shutil.copytree(src, REPO); (REPO / "MANIFEST.json").unlink()
(REPO / "CLAUDE.md").write_text("HOST-CANARY-do-not-read-from-jail\n")      # stand-in host file for the red-team read tests
JAVA_HOME = str(Path(os.path.realpath(shutil.which("java"))).parents[1]) if shutil.which("java") else ""
os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, EL_SECRET_CANARY="canary-do-not-leak-7f3a", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from evidence_ladder.execution import kaggle_jail as KJ, engine, config as K
from evidence_ladder.common.interfaces import Artifact, GraderIsolationError
from evidence_ladder.execution.artifacts import ArtifactStore
from evidence_ladder.execution.backend import SandboxExecutionBackend
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate
from evidence_ladder.execution.taskload import load_tasks
CFG = K.SandboxConfig(py_timeout=6.0, java_run_timeout=8.0)

@phase("FC1_rootfs")
def _():
    r = KJ.build_rootfs(JAVA_HOME); r["status"] = "BUILT"; return r

@phase("FC1_selfcheck")
def _():
    R = KJ.ROOTFS
    rec = {"rootfs_paths_exist": {p: (R / p.lstrip("/")).exists() or (R / p.lstrip("/")).is_symlink() for p in ("usr/bin/python3", "usr/bin/python3.13", "lib64/ld-linux-x86-64.so.2", "usr/bin/prlimit",
           JAVA_HOME + "/bin/javac", JAVA_HOME + "/lib/libjli.so", JAVA_HOME + "/lib/server/libjvm.so", "tmp", "jail", "dev/null")},
           "rootfs_symlink_targets": {t: os.readlink(R / t) for t in ("lib", "lib64", "bin") if (R / t).is_symlink()}}
    d = REPO / "sc"; d.mkdir(exist_ok=True); c = d / "cand.py"; c.write_text("print(1)\ndef solve(x): return 1\n"); r = d / "runner.py"; shutil.copyfile(K.PY_RUNNER, r)
    p = subprocess.run(KJ.bwrap_cmd(c, r), input=b'{"a":1}\n', capture_output=True, timeout=30, env={"PATH": "/usr/bin"})
    rec["trivial_python"] = {"returncode": p.returncode, "stdout": p.stdout.decode()[-300:], "stderr": p.stderr.decode()[-1500:]}
    rec["status"] = "PASS" if p.returncode == 0 and b'"value": 1' in p.stdout else "FAIL"
    lj = R / JAVA_HOME.lstrip("/") / "lib/libjli.so"
    rec["libjli"] = {"lexists": os.path.lexists(lj), "islink": lj.is_symlink(), "readlink": os.readlink(lj) if lj.is_symlink() else None, "exists": lj.exists()}
    w = REPO / "scj"; w.mkdir(exist_ok=True); os.chmod(w, 0o755)
    pj = subprocess.run(KJ.java_jail(w, [JAVA_HOME + "/bin/javac", "-version"], rw=True, jdk=JAVA_HOME), capture_output=True, timeout=60, env={"PATH": "/usr/bin"})
    rec["javac_version_in_jail"] = {"returncode": pj.returncode, "stdout": pj.stdout.decode()[-300:], "stderr": pj.stderr.decode()[-800:]}
    if pj.returncode != 0:
        rec["status"] = "FAIL"
    return rec

PROBE = r'''
import os, sys, socket, resource, ctypes
def a(fn):
    try: return "ALLOWED:" + str(fn())[:60]
    except BaseException as e: return "blocked:" + type(e).__name__ + ":" + str(getattr(e, "errno", ""))
def mnt():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mount(b"none", b"/tmp", b"tmpfs", 0, None) != 0: raise OSError(ctypes.get_errno(), "mount")
def solve(d):
    libc = ctypes.CDLL(None, use_errno=True)
    return {"uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(), "groups": os.getgroups(), "cwd": os.getcwd(), "env": sorted(os.environ), "root_ls": sorted(os.listdir("/")),
      "dev_null_w": a(lambda: open("/dev/null", "w").write("x")), "dev_urandom_r": a(lambda: len(open("/dev/urandom", "rb").read(4))), "proc": a(lambda: os.listdir("/proc")), "kaggle": a(lambda: os.listdir("/kaggle")), "tmp_w": a(lambda: open("/tmp/x", "w").write("1")),
      "usr_w": a(lambda: open("/usr/x", "w")), "etc_w": a(lambda: open("/etc/x", "w")), "jail_w": a(lambda: open("/jail/cand.py", "w")), "host_canary": a(lambda: open("/kaggle/working/repo/CLAUDE.md").read()),
      "setuid0": a(lambda: os.setuid(0)), "chroot_escape": a(lambda: os.chroot("/")), "mknod": a(lambda: os.mknod("/tmp/n", 0o20644, os.makedev(1, 3))), "mount": a(mnt),
      "inet_socket": a(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)), "inet6_socket": a(lambda: socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)),
      "raw_socket": a(lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW)), "netlink_socket": a(lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 0)),
      "unix_socketpair": a(lambda: socket.socketpair()), "ptrace": libc.ptrace(0, 0, 0, 0), "ptrace_errno": ctypes.get_errno(),
      "rlimits": {n: list(resource.getrlimit(getattr(resource, "RLIMIT_" + n))) for n in ("AS", "FSIZE", "NOFILE", "NPROC", "CORE")}}
'''
@phase("FC1_layers_python")
def _():
    r = engine.run_python(PROBE.encode(), [{}], CFG)
    items, term = engine._parse("python", r)
    return {"phase_status": r.phase_status, "term": term, "stderr_tail": r.stderr_tail, "probe": items[0][1] if items else None}

@phase("FC1_layers_java")
def _():
    java = ('import java.util.*; import java.nio.file.*; import java.net.*;\npublic final class Solution { static String a(java.util.concurrent.Callable<Object> c){ try{ return "ALLOWED:"+c.call(); }catch(Throwable e){ return "blocked:"+e.getClass().getSimpleName(); } }\n'
            ' public static Object solve(Map<String,Object> d){ Map<String,Object> m=new LinkedHashMap<>(); m.put("user", System.getProperty("user.name")); m.put("host_canary", a(()->new String(Files.readAllBytes(Paths.get("/kaggle/working/repo/CLAUDE.md")))));'
            ' m.put("socket", a(()->{Socket s=new Socket(); s.connect(new InetSocketAddress("1.1.1.1",80),2000); return 1;})); m.put("write_usr", a(()->Files.write(Paths.get("/usr/x"), new byte[1])));'
            ' m.put("list_kaggle", a(()->Files.list(Paths.get("/kaggle")).count())); m.put("proc", a(()->Files.list(Paths.get("/proc")).count())); return m.toString(); } }\n')
    be = SandboxExecutionBackend(ArtifactStore(OUT / "store_probe"), CFG)
    o = be.run_experiment(Artifact.make("J1-01", "java", java, origin="probe"), {"kind": "function_call", "payload": json.dumps({"text": "a"})})
    return {"outcome": o}

def run_script(cmd, timeout, env=None):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, **(env or {})))
    (OUT / ("stdout_%s.txt" % Path(cmd[1]).stem)).write_text(p.stdout + "\n---STDERR---\n" + p.stderr, encoding="utf-8")
    return {"exit_code": p.returncode, "stdout_tail": p.stdout[-6000:], "stderr_tail": p.stderr[-3000:], "full_stdout_file": "stdout_%s.txt" % Path(cmd[1]).stem}

@phase("FC1_redteam_v2_original_script")
def _():
    # the existing redteam_run_v2.py, logic unchanged: it spawns grade_sandboxed_v2.py; here that script is replaced by a wrapper that swaps in the Kaggle jail
    g = REPO / "results/raw/codex-pilot1-20261016/fixed_v2/grader/sandbox"
    shutil.copyfile(g / "grade_sandboxed_v2.py", g / "grade_sandboxed_v2_orig.py")
    (g / "grade_sandboxed_v2.py").write_text("import sys, importlib.util\nsys.path.insert(0, %r)\nfrom evidence_ladder.execution import kaggle_jail\n"
        "s = importlib.util.spec_from_file_location('orig', %r); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\nm.bwrap_cmd = kaggle_jail.bwrap_cmd\nm.main()\n" % (str(REPO), str(g / "grade_sandboxed_v2_orig.py")))
    out = OUT / "redteam_v2_kaggle.json"
    r = run_script([sys.executable, str(g / "redteam_run_v2.py"), str(out)], 300)
    r["report"] = json.loads(out.read_text()) if out.exists() else None
    r["status"] = "PASS" if r["report"] and r["report"]["all_pass"] else "FAIL"
    r["protected_file_exists_on_host"] = (g.parent / "protected_cases.json").exists()
    shutil.copyfile(g / "grade_sandboxed_v2_orig.py", g / "grade_sandboxed_v2.py")      # restore for the suites below
    return r

@phase("FC1_FC4_execution_suite_through_kaggle_jail")
def _():
    # evidence_ladder/execution/tests/run_tests.py: reference, wrong solutions, compile/timeout classes, hostile python+java, hash tamper, gate, static no-protected-import.
    # Tests that assert bubblewrap namespaces / WSL receipts are expected to fail here: they are listed, not hidden.
    r = run_script([sys.executable, "evidence_ladder/execution/tests/run_tests.py"], 2400)
    try: r["summary"] = json.loads((OUT / r["full_stdout_file"]).read_text().split("\n---STDERR---")[0].strip().splitlines()[-1])
    except Exception: r["summary"] = None
    return r

@phase("FC4_separation_in_kaggle")
def _():
    rec = {}
    gate = SelectionGate(); store = ArtifactStore(OUT / "store_fc4"); gr = ProtectedGrader(gate, store, CFG)
    ref = Artifact.make("P1-01", "python", (K.PY_ROOT / "grader/oracles.py").read_text() + "\n\ndef solve(d):\n    return p01(d)\n", origin="reference")
    try: gr.grade(ref); rec["gate_open_raises"] = False
    except GraderIsolationError: rec["gate_open_raises"] = True
    sent = []; orig_spawn = engine.spawn
    engine.spawn = lambda cmd, stdin_bytes, timeout, max_out: (sent.append(stdin_bytes), orig_spawn(cmd, stdin_bytes, timeout, max_out))[1]
    be = SandboxExecutionBackend(store, CFG); task = {t.task_id: t for t in load_tasks("python")}["P1-01"]
    be.run_public(ref, task); be.run_experiment(ref, {"kind": "function_call", "payload": json.dumps({"text": "abc"})})
    pub_inputs = {json.dumps(c["input"], ensure_ascii=True) for c in task.public_tests} | {json.dumps({"text": "abc"}, ensure_ascii=True)}
    lines = {l for b in sent for l in b.decode().splitlines()}
    rec["selection_stdin_lines"] = len(lines); rec["selection_stdin_only_public_inputs"] = lines <= pub_inputs
    prot = json.loads(K.PY_ROOT.joinpath("grader/protected_cases.json").read_text())
    p01 = next(t for t in prot if (t.get("task_id") or t["id"]) == "P1-01")["cases"]
    prot_in = {json.dumps(c["input"], ensure_ascii=True) for c in p01} - pub_inputs
    rec["protected_inputs_not_public"] = len(prot_in); rec["no_protected_input_in_selection_stdin"] = not (lines & prot_in)
    raw_prot = K.PY_ROOT.joinpath("grader/protected_cases.json").read_text(); marker = raw_prot[len(raw_prot) // 2: len(raw_prot) // 2 + 40]      # 40 raw bytes from the middle of the hidden-case file
    h = len(marker) // 2      # the marker is split in the scanner source so the candidate file itself does not contain it
    scan = ("import os, sys, json\nM = %r + %r\ndef solve(d):\n    hits = []\n    for root, ds, fs in os.walk('/'):\n        for f in fs:\n            try:\n                if M.encode() in open(os.path.join(root, f), 'rb').read(2000000): hits.append(os.path.join(root, f))\n            except OSError: pass\n"
            "    sys.stderr.write('SCAN_HITS ' + json.dumps(hits) + '\\n'); return -1\n" % (marker[:h], marker[h:]))
    o = be.run_public(Artifact.make("P1-01", "python", scan, origin="hostile:scan"), task)
    tail = "\n".join(x.get("stderr_tail", "") for x in o)
    rec["jail_scan_marker"] = marker; rec["jail_scan_hits_line"] = sorted({l for l in tail.splitlines() if l.startswith("SCAN_HITS")})
    rec["jail_fs_scan_ran_and_found_nothing"] = rec["jail_scan_hits_line"] == ["SCAN_HITS []"]
    rec["host_has_marker_file"] = marker in raw_prot
    gate.close(); sent.clear(); g = gr.grade(ref)
    rec["graded_after_close"] = {k: g[k] for k in ("passed", "passed_tests", "total_tests", "failure")}
    glines = {l for b in sent for l in b.decode().splitlines()}
    rec["grading_stdin_lines"] = len(glines)
    rec["grading_stdin_equals_protected_inputs_only"] = glines == {json.dumps(c["input"], ensure_ascii=True) for c in p01}
    rec["grading_stdin_has_expected_key"] = any('"expected"' in l for l in glines)
    rec["hiddengrade_fields"] = sorted(g)
    engine.spawn = orig_spawn
    ok = (rec["gate_open_raises"] and rec["selection_stdin_only_public_inputs"] and rec["no_protected_input_in_selection_stdin"] and rec["host_has_marker_file"]
          and rec["jail_fs_scan_ran_and_found_nothing"] and rec["grading_stdin_equals_protected_inputs_only"] and not rec["grading_stdin_has_expected_key"] and g["passed"])
    rec["status"] = "PASS" if ok else "FAIL"
    return rec

QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}   # HF LFS oids of Qwen/Qwen3-4B-Thinking-2507 @ 768f209d9ea8 (HF tree API)
@phase("FC3_qwen_throughput")
def _():
    import torch
    if not torch.cuda.is_available():
        return {"status": "SKIPPED", "reason": "no CUDA device in this kernel"}
    dirs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "4b-thinking-2507" in c.lower()]
    if not dirs:
        return {"status": "BLOCKED", "reason": "no mounted qwen3-4b-thinking-2507 snapshot"}
    md = dirs[0]; rec = {"model_dir": md, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "gpus": torch.cuda.device_count(), "sha256_ok": {}}
    for f, h in QWEN_SHA.items():
        x = hashlib.sha256()
        with open(Path(md, f), "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
        rec["sha256_ok"][f] = x.hexdigest() == h
    if not all(rec["sha256_ok"].values()):
        rec["status"] = "FAIL"; rec["reason"] = "mounted snapshot does not match the pinned Hugging Face file hashes"; return rec
    os.environ["EVIDENCE_LADDER_ALLOW_INFERENCE"] = "1"
    from evidence_ladder.models import qwen as Q
    from evidence_ladder.common.interfaces import GenerationRequest
    be = Q.HFQwenBackend(authorized=True, model_dir=md, log_path=str(OUT / "qwen_gen_log.jsonl"))
    t = time.time(); model, tok = be._load(); rec["load_seconds"] = round(time.time() - t, 1)
    tasks = {x.task_id: x for x in load_tasks("python")}; tasks.update({x.task_id: x for x in load_tasks("java")})
    MAXNEW = int(os.environ.get("EL_MAXNEW", "8192")); rows = []
    for tid in ("P1-01", "P1-05", "P1-10", "P1-20", "J1-09"):
        tk = tasks[tid]; lang = tk.language
        prompt, _ = Q.render("qwen_sample_v1", task_prompt=tk.prompt, examples=Q.format_examples(tk.public_tests), language=lang)
        req = GenerationRequest(tid, prompt, 1, [1000], MAXNEW, "sample", True, "qwen_sample_v1")
        be.language = lang
        torch.cuda.synchronize(); t = time.time()
        g = be.generate(req)[0]; torch.cuda.synchronize(); wall = time.time() - t
        row = {"task_id": tid, "tokens_prompt": g.tokens_prompt, "tokens_generated": g.tokens_generated, "wall_seconds": round(wall, 2), "tokens_per_s": round(g.tokens_generated / wall, 2),
               "thinking_closed": "</think>" in g.text, "hit_max_new_tokens": g.tokens_generated >= MAXNEW, "extract_status": g.extract_status}
        if g.extract_status == "ok":
            a = Artifact.make(tid, lang, g.code, origin="sample:1000")
            ex = [o for o in SandboxExecutionBackend(ArtifactStore(OUT / "store_fc3"), CFG).run_public(a, tk) if o["check_id"] == "public_examples_v1"]
            row["public_examples_pass"] = sum(o["status"] == "pass" for o in ex); row["public_examples_total"] = len(ex)
        rows.append(row); rec["rows"] = rows; RES["phases"]["FC3_qwen_throughput"] = rec; save()
    try:
        tok.padding_side = "left"; chats = []
        for tid in ("P1-01", "P1-05", "P1-10", "P1-20"):
            tk = tasks[tid]; prompt, _ = Q.render("qwen_sample_v1", task_prompt=tk.prompt, examples=Q.format_examples(tk.public_tests), language=tk.language)
            chats.append(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device); n_in = enc["input_ids"].shape[1]
        torch.manual_seed(1); torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(): ids = model.generate(**enc, do_sample=True, max_new_tokens=2048, pad_token_id=tok.eos_token_id, **Q.SAMPLING)
        torch.cuda.synchronize(); wall = time.time() - t
        ntok = int((ids[:, n_in:] != tok.eos_token_id).sum())
        rec["batch4"] = {"max_new_tokens": 2048, "wall_seconds": round(wall, 2), "generated_tokens_total_nonpad": ntok, "aggregate_tokens_per_s": round(ntok / wall, 2)}
    except BaseException as e:
        rec["batch4"] = {"error": repr(e)[:300]}
    rec["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    tot_t = sum(r["tokens_generated"] for r in rows); tot_s = sum(r["wall_seconds"] for r in rows)
    rec["summary"] = {"problems": len(rows), "tokens_generated_total": tot_t, "wall_total": round(tot_s, 1), "mean_tokens_per_s": round(tot_t / tot_s, 2), "mean_tokens_per_problem": round(tot_t / len(rows), 1),
                      "mean_seconds_per_problem": round(tot_s / len(rows), 1), "truncated_at_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rows), "max_new_tokens": MAXNEW}
    rec["status"] = "MEASURED"; return rec

RES["seconds"] = round(time.time() - t0, 1); save()
print(json.dumps({k: v.get("status") for k, v in RES["phases"].items()}))
