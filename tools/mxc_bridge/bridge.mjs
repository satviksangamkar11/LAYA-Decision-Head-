// LAYA <-> Microsoft MXC bridge (V1 Node SDK, @microsoft/mxc-sdk 1.0.1). Python side: evidence_ladder/controller/mxc_sandbox.py.
// Protocol: one JSON object per line on stdin; one result line per request on stdout, prefixed with "@@MXC@@" (everything else on stdout is ignored).
//   {"op":"probe"}                                   -> {"ok":true,"support":..,"backends":..,"probe":..,"sdk":..}
//   {"op":"run","id":..,"command":..,"cwd":..,"env":{..},"readwrite":[..],"readonly":[..],"timeoutMs":..,"maxOutputBytes":..}
//        -> {"ok":true,"id":..,"exitCode":..,"timedOut":..,"stdout":..,"stderr":..,"outputCapped":bool,"warnings":[..],"wallMs":..}
//   {"op":"quit"}
// A run uses containment processcontainer, NO network block (MXC denies network when none is declared; verified by tests), inheritDefaultEnvironment=false (the host environment is never
// inherited) and the filesystem lists given by the caller. Output is read through spawn() so the bridge can stop the workload when it exceeds maxOutputBytes (run() has no output cap).
// One-shot use: node bridge.mjs --once '<json request>' . Serve use: node bridge.mjs --serve .
import * as mxc from "@microsoft/mxc-sdk/v1";
import { createInterface } from "node:readline";

const MARK = "@@MXC@@";
const out = (o) => process.stdout.write(MARK + JSON.stringify(o) + "\n");

function collect(stream, cap, state, key) {
  const chunks = [];
  let n = 0;
  return new Promise((resolve) => {
    if (!stream) return resolve("");
    stream.on("data", (b) => {
      if (n >= cap) { state.capped = true; return; }
      const take = Math.min(b.length, cap - n);
      chunks.push(b.subarray(0, take)); n += take;
      if (take < b.length) state.capped = true;
      if (state.capped && state.onCap) state.onCap();
    });
    stream.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    stream.on("error", () => resolve(Buffer.concat(chunks).toString("utf8")));
    stream.on("close", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });
}

async function doRun(req) {
  const t0 = performance.now();
  const cap = Number(req.maxOutputBytes || 1048576);
  const request = {
    command: req.command,
    containment: { type: "processcontainer" },
    workingDirectory: req.cwd,
    timeoutMs: req.timeoutMs,
    inheritDefaultEnvironment: false,
    environment: req.env || {},
    filesystem: { readwritePaths: req.readwrite || [], readonlyPaths: req.readonly || [], deniedPaths: req.denied || [], clearPolicyOnExit: true },
  };
  if (!streamingOk()) {
    // Node < 24.21 (or < 26.8) cannot use native stdio on Windows: fall back to the captured run(); the output cap is then applied AFTER the run (the tail is not kept, the head is).
    const r = await mxc.run(request);
    const capped = r.stdout.length > cap || r.stderr.length > cap;
    return { ok: true, id: req.id, exitCode: r.exitCode, timedOut: !!r.timedOut, stdout: r.stdout.slice(0, cap), stderr: r.stderr.slice(0, cap), outputCapped: false, outputTruncatedPostHoc: capped,
             capMode: "post_hoc", warnings: [...(r.warnings || [])], wallMs: Math.round(performance.now() - t0) };
  }
  const state = { capped: false, onCap: null };
  const proc = await mxc.spawn(request);
  state.onCap = () => { try { proc.kill(); } catch (_) { /* already gone */ } };
  const so = collect(proc.standardOutput, cap, state, "stdout");
  const se = collect(proc.standardError, cap, state, "stderr");
  const w = await proc.wait();
  const [stdout, stderr] = await Promise.all([so, se]);
  const warnings = [...(proc.warnings || [])];
  try { proc.dispose(); } catch (_) { /* released by wait() */ }
  return { ok: true, id: req.id, exitCode: w.exitCode, timedOut: !!w.timedOut, stdout, stderr, outputCapped: state.capped, capMode: "streaming", warnings, wallMs: Math.round(performance.now() - t0) };
}

function streamingOk() {
  const [maj, min] = process.versions.node.split(".").map(Number);
  return (maj === 24 && min >= 21) || (maj === 26 && min >= 8) || maj > 26;
}

async function handle(req) {
  try {
    if (req.op === "probe") {
      let probe = null;
      try { probe = mxc.probe(); } catch (e) { probe = { error: String(e.message || e) }; }
      return { ok: true, sdk: mxc.SDK_CONTRACT_VERSION, support: mxc.getPlatformSupport(), backends: mxc.getAvailableBackends(), probe, node: process.version, platform: process.platform, arch: process.arch };
    }
    if (req.op === "run") return await doRun(req);
    return { ok: false, id: req.id, error: { code: "bad_op", message: "unknown op " + req.op } };
  } catch (e) {
    return { ok: false, id: req.id, error: { code: e.code || "bridge_error", message: String(e.message || e).slice(0, 800), details: e.details || null, nativeCode: e.nativeCode ?? null } };
  }
}

const mode = process.argv[2];
if (mode === "--once") {
  out(await handle(JSON.parse(process.argv[3])));
} else if (mode === "--serve") {
  const rl = createInterface({ input: process.stdin });
  let chain = Promise.resolve();
  rl.on("line", (line) => {
    if (!line.trim()) return;
    chain = chain.then(async () => {
      let req;
      try { req = JSON.parse(line); } catch (e) { return out({ ok: false, error: { code: "bad_json", message: String(e.message) } }); }
      if (req.op === "quit") { process.exit(0); }
      out(await handle(req));
    });
  });
  rl.on("close", () => chain.then(() => process.exit(0)));
} else {
  process.stderr.write("usage: node bridge.mjs --once <json> | --serve\n");
  process.exit(2);
}
