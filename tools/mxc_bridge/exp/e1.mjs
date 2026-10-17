import * as mxc from "@microsoft/mxc-sdk/v1";
const t0 = performance.now();
const r = await mxc.run({ command: 'cmd.exe /c echo hello & whoami & set', containment: {type:"processcontainer"}, timeoutMs: 20000 });
console.log(JSON.stringify(r, null, 1).slice(0, 3000), "ms", performance.now()-t0);
