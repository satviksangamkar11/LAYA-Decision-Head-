import * as mxc from "@microsoft/mxc-sdk/v1";
import fs from "node:fs"; import path from "node:path";
const py = "C:\\Users\\Satvik\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none";
const git = "C:\\Program Files\\Git";
const ws = fs.mkdtempSync(path.join("D:\\local model\\tools\\mxc_bridge\\exp\\", "ws-"));
const la = path.join(ws, "_la"), tmp = path.join(ws, "_tmp"), shim = path.join(ws, "_shim");
for (const d of [la, tmp, shim]) fs.mkdirSync(d);
fs.writeFileSync(path.join(shim, "python3"), '#!/bin/sh\nexec "/c/Users/Satvik/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe" "$@"\n');
fs.writeFileSync(path.join(ws, "t.py"), 'print("hello from python")\n');
async function go(label, req) {
  const t0 = performance.now();
  try { const r = await mxc.run(req); console.log("==", label, Math.round(performance.now()-t0), "ms exit", r.exitCode, "timedOut", r.timedOut, "warn", JSON.stringify(r.warnings)); console.log(r.stdout.slice(0,1500)); console.log("STDERR:", r.stderr.slice(0,600)); }
  catch (e) { console.log("==", label, "THROW", e.code, e.message.slice(0,400)); }
}
const env = {SystemRoot: "C:\\Windows", PYTHONHASHSEED: "0", LOCALAPPDATA: la, TEMP: tmp, TMP: tmp, PATH: "/usr/bin:/c/WINDOWS/system32", HOME: tmp};
const mk = (cmd, extra={}) => ({ command: `"${git}\\bin\\bash.exe" -c "${cmd}"`, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000,
  inheritDefaultEnvironment: false, environment: env, filesystem: { readwritePaths: [ws], readonlyPaths: [py, git] }, ...extra });
const gitUsr = path.join(git, "usr", "bin") + path.sep;
await go("usr/bin/echo.exe", {command: `"${gitUsr}echo.exe" hi`, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000, inheritDefaultEnvironment:false, environment: env, filesystem:{readwritePaths:[ws], readonlyPaths:[git]}});
await go("usr/bin/bash.exe", {command: `"${gitUsr}bash.exe" -c "echo hi"`, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000, inheritDefaultEnvironment:false, environment: env, filesystem:{readwritePaths:[ws], readonlyPaths:[git]}});
await go("cmd python", {command: `cmd.exe /c ""${py}\\python.exe" t.py"`, containment: {type:"processcontainer"}, workingDirectory: ws, timeoutMs: 30000, inheritDefaultEnvironment:false, environment: env, filesystem:{readwritePaths:[ws], readonlyPaths:[py]}});
