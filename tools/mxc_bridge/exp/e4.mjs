import * as mxc from "@microsoft/mxc-sdk/v1";
import fs from "node:fs"; import path from "node:path";
const py = "C:\\Users\\Satvik\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none";
const ws = fs.mkdtempSync(path.join("D:\\local model\\tools\\mxc_bridge\\exp\\", "ws-"));
const la = path.join(ws, "_la"), tmp = path.join(ws, "_tmp");
for (const d of [la, tmp]) fs.mkdirSync(d);
fs.writeFileSync(path.join(ws, "t.py"), 'print("hello from python")\nimport sys; print(sys.argv)\n');
fs.writeFileSync(path.join(ws, "a.txt"), "content A\n");
async function go(label, req) {
  const t0 = performance.now();
  try { const r = await mxc.run(req); console.log("==", label, Math.round(performance.now()-t0), "ms exit", r.exitCode, "timedOut", r.timedOut, "warn", JSON.stringify(r.warnings)); console.log(JSON.stringify(r.stdout.slice(0,600))); console.log("STDERR:", JSON.stringify(r.stderr.slice(0,400))); }
  catch (e) { console.log("==", label, "THROW", e.code, e.message.slice(0,300)); }
}
const env = {SystemRoot: "C:\\Windows", PYTHONHASHSEED: "0", LOCALAPPDATA: la, TEMP: tmp, TMP: tmp, PATH: "C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\WindowsPowerShell\\v1.0"};
const mk = (command, extra={}) => ({ command, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000, inheritDefaultEnvironment: false, environment: env,
  filesystem: { readwritePaths: [ws], readonlyPaths: [py] }, ...extra });
const ps = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ";
await go("ps ls/cat", mk(ps + '"ls; cat a.txt; echo hi"'));
await go("ps python", mk(ps + `"& '${py}\\python.exe' t.py x y"`));
await go("direct python argv", mk(`"${py}\\python.exe" t.py x y`));
// timeout and process tree
fs.writeFileSync(path.join(ws, "spin.py"), 'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c","import time,os;open(\'child.pid\',\'w\').write(str(os.getpid()));time.sleep(120)"])\ntime.sleep(120)\n');
await go("timeout 3s with child", mk(`"${py}\\python.exe" spin.py`, {timeoutMs: 3000}));
await new Promise(r => setTimeout(r, 1000));
const pidf = path.join(ws, "child.pid");
if (fs.existsSync(pidf)) { const pid = fs.readFileSync(pidf, "utf8"); try { process.kill(Number(pid), 0); console.log("CHILD STILL ALIVE", pid); } catch { console.log("child gone", pid); } } else console.log("no child pid file");
