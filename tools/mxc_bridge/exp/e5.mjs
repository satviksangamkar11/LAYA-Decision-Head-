import * as mxc from "@microsoft/mxc-sdk/v1";
for (const type of ["wslc", "isolation_session", "windows_sandbox", "hyperlight"]) {
  try {
    const r = await mxc.run({ command: "echo hi", containment: { type }, timeoutMs: 20000 });
    console.log(type, "RAN", JSON.stringify(r).slice(0, 300));
  } catch (e) { console.log(type, "THROW", e.code, "|", e.message.slice(0, 350), "| remediation:", e.remediation, "| native:", e.nativeCode); }
}
try { console.log("validateProvision wslc", JSON.stringify(await mxc.validateProvision({ containment: { type: "wslc" }, command: "echo hi" }))); } catch (e) { console.log("validateProvision wslc THROW", e.code, e.message.slice(0, 300)); }
