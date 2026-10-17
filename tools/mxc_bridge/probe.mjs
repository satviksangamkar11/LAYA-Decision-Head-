import * as mxc from "@microsoft/mxc-sdk/v1";
console.log(JSON.stringify({exports: Object.keys(mxc).slice(0,60)}));
try { console.log("support", JSON.stringify(mxc.getPlatformSupport(), null, 1)); } catch (e) { console.log("support error", e.message); }
try { console.log("backends", JSON.stringify(mxc.getAvailableBackends(), null, 1)); } catch (e) { console.log("backends error", e.message); }
try { console.log("probe", JSON.stringify(mxc.probe(), null, 1)); } catch (e) { console.log("probe error", e.message); }
