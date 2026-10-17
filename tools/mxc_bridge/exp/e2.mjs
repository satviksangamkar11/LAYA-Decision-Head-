import * as mxc from "@microsoft/mxc-sdk/v1";
import fs from "node:fs"; import path from "node:path";
const py = "C:\\Users\\Satvik\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none";
const ws = fs.mkdtempSync(path.join("D:\\local model\\tools\\mxc_bridge\\exp\\", "ws-"));
fs.writeFileSync(path.join(ws, "probe.py"), String.raw`
import os, socket, sys, json
out = {}
out["env"] = sorted(os.environ.keys())
out["cwd"] = os.getcwd()
def tryread(p):
    try:
        with open(p, "rb") as f: return "READ %d" % len(f.read(50))
    except Exception as e: return "ERR " + type(e).__name__
out["secret"] = tryread(r"D:\local model\.secrets\nvidia_api_key_2.txt")
out["claude_md"] = tryread(r"D:\local model\CLAUDE.md")
out["hosts"] = tryread(r"C:\Windows\System32\drivers\etc\hosts")
out["ssh"] = tryread(r"C:\Users\Satvik\.ssh\id_rsa")
def trywrite(p):
    try:
        open(p,"w").write("x"); return "WROTE"
    except Exception as e: return "ERR " + type(e).__name__
out["write_ws"] = trywrite("out.txt")
out["write_proj"] = trywrite(r"D:\local model\mxc_probe_should_fail.txt")
out["write_c"] = trywrite(r"C:\mxc_probe_should_fail.txt")
try:
    s = socket.create_connection(("1.1.1.1", 80), timeout=5); out["net"] = "CONNECTED"
except Exception as e: out["net"] = "ERR " + type(e).__name__ + str(e)[:40]
try: out["dns"] = socket.gethostbyname("example.com")
except Exception as e: out["dns"] = "ERR " + type(e).__name__
print(json.dumps(out, indent=1))
`);
async function go(label, req) {
  const t0 = performance.now();
  try { const r = await mxc.run(req); console.log("==", label, Math.round(performance.now()-t0), "ms exit", r.exitCode, "timedOut", r.timedOut, "warn", JSON.stringify(r.warnings)); console.log(r.stdout.slice(0,2500)); console.log("STDERR:", r.stderr.slice(0,800)); }
  catch (e) { console.log("==", label, "THROW", e.code, e.message.slice(0,400), JSON.stringify(e.details||{}).slice(0,300)); }
}
const la = path.join(ws, "_la"), tmp = path.join(ws, "_tmp");
fs.mkdirSync(la); fs.mkdirSync(tmp);
const base = { command: `"${py}\\python.exe" probe.py`, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000,
  inheritDefaultEnvironment: false, environment: {SystemRoot: "C:\\Windows", PYTHONHASHSEED: "0", LOCALAPPDATA: la, TEMP: tmp, TMP: tmp},
  filesystem: { readwritePaths: [ws], readonlyPaths: [py] } };
await go("A fs-policy, no network block", base);
await go("B network egress deny", {...base, network: {egress: {default: "deny"}}});
