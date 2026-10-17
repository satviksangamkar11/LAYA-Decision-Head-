# EL research B: free sandboxes for model-written code

[EL-RESEARCH-B · Aim: free sandbox survey] Checked 2026-10-18. Tags: **[D]** documented on a page I fetched (see sources), **[V]** vendor or single secondary source, **[I]** my inference, **[U]** not verified. Free limits change; re-check before deciding.

**Key fact for this machine:** Windows 11 Home. Windows Sandbox is not supported on Home [D]. WSL2 installs on Windows 11 with no edition limit shown [D].

## (a) Local / self-hosted

| Option | Isolation | Free limits and licence | Arbitrary repo test cmd | Offline (no network) |
|---|---|---|---|---|
| Docker / Podman | Container, shared host kernel [I] | Podman Apache-2.0 [D]; Linux native, macOS/Windows via VM [D]. Docker Desktop free only under 250 employees AND under $10M revenue [D] | Yes [I] | Yes, `--network none` [I] |
| gVisor (runsc) | Userspace app kernel, "not a syscall filter" [D] | Apache-2.0 [D]; Linux host [I] | Yes, via Docker/containerd [I] | Yes [I] |
| Firecracker | microVM on KVM [D] | Apache-2.0 [D]; tested only on AWS Linux bare metal [D] | Yes [I] | Yes [I] |
| Kata Containers | VM per container [D] | Apache-2.0 [D]; needs VT-x/SVM on Linux [D] | Yes [I] | Yes [I] |
| bubblewrap | Linux user namespaces [D] | LGPL-2.1 per README [D] | Yes, Linux only [I] | Yes [I] |
| nsjail | Namespaces, cgroups, seccomp-bpf [D] | Apache-2.0 [D]; Linux only [D] | Yes, Linux only [I] | Yes [I] |
| firejail | [U] | [U] | [U] | [U] |
| Judge0 CE | Isolated per submission; mechanism stated by a third party only [V] | GPL-3.0 per repo [D]; a Hostinger page says MIT, repo wins [D] | Mostly language runners, not repo test suites [I] | Yes if self-hosted [I] |
| Piston | "Excels at running untrusted code" [V] | MIT [D]. Public API no longer free from 2026-02-15, self-host needed [D] | Language runtimes, not repos [I] | Yes if self-hosted [I] |
| Pyodide / WASM | WebAssembly, no native processes [D] | MPL-2.0 [D]; README makes no security claim [D] | No for native deps or node/go/java [I] | Yes [I] |
| RestrictedPython | None: "not a sandbox system or a secured environment" [D] | Licence [U] | Not a sandbox, do not use for untrusted code [D] | n/a |

**Windows-specific**

| Option | Isolation | Free limits and notes |
|---|---|---|
| WSL2 | Lightweight VM [I] | Free [D]; runs Linux toolchains, so Linux sandboxes (nsjail, bubblewrap, Docker Engine) work inside it [I] |
| Windows Sandbox | Disposable Hyper-V VM [D] | Pro, Enterprise, Education only, NOT Home [D]. Network on by default, can be disabled [D]. One instance at a time [D]. Not usable on this machine |
| Hyper-V | VM | Pro/Enterprise/Education [I]; edition page not fetched [U] |
| Docker Desktop on Windows | Container via WSL2 or Hyper-V [D] | Listed editions are Enterprise, Pro, Education [D]. Home status is ambiguous on the page [D] |

## (b) Hosted free tiers

Every hosted option sends code off this machine. Your project rules require a yes for that [D: CLAUDE.md].

| Option | Isolation | Free limits | Status and licence | Repo test cmd |
|---|---|---|---|---|
| E2B | "Isolated Linux VM" [D]; Firecracker not on the docs page [U] | Hobby: one-time $100 credit [D]; 1 h per session; 20 concurrent [D]; Pro $150/mo + usage [D]. Aggregators say recurring credit [conflict] | Licence [U] | Yes [I] |
| Daytona | Dedicated kernel, isolation tech not named [D] | Tier 1 (email verified): 10 vCPU, 10 or 20 GiB RAM (page conflicts), 30 GiB disk [D]. $200 signup credit [V] | **Repo archived 2026-10-03, no further updates** [D]. AGPL-3.0 [V] | Yes [I] |
| Modal | gVisor [D] | $30/month free credit [D]; 5 min default timeout, 24 h max [D]; $0.0000131 per core-second [D] | Proprietary service [I] | Yes [I]; no network-block option on the page [U] |
| Vercel Sandbox | Firecracker microVM [V, Vercel KB] | Hobby: 5 CPU-hours/mo, 420 GB-hours memory/mo, 10 concurrent, 45 min session, max 4 vCPU/8 GB [D]. Downloads free [D] | Service | Yes [I] |
| GitHub Codespaces / Actions | VM [I] | Codespaces free: 120 compute hours, 15 GB-month storage [D]. Actions: 2,000 min/mo private [D]; standard runners free on public repos, no stated cap [D] | Service | Yes, but not built for hostile code [I] |
| Google Colab | VM | Max 12 h runtime, idle timeouts, limits unpublished [D]. FAQ bans circumventing anti-abuse policies by containerization and multi-account workarounds [D] | Service terms | Policy risk [I] |
| Kaggle notebooks | VM | 30 h/week GPU, 12 h session, 20 min idle [V, third-party only]; official docs fetch returned no content [U] | Service terms | [U] |
| Riza | WebAssembly [D] | 100,000 requests/mo, 30 s per request, outbound HTTP only [D] | Service | No, function-level only [I] |
| CodeSandbox, Replit | [U] (CodeSandbox pricing returned 403) | [U] | [U] | [U] |

## (c) Ranking

**(1) Developer laptop plugin (Windows, macOS, Linux, offline)**
1. **Docker or Podman, `--network none`.** Only option that covers all three OSes [I]. Check Docker Desktop thresholds; Podman avoids the licence question [I].
2. **WSL2 with Linux sandboxes (nsjail, bubblewrap, or Docker Engine inside).** Works on Home [D]. Windows-only path, same Linux code as item 4 [I]. Docker Engine licence [U].
3. **gVisor under Docker on Linux.** Stronger than a plain container [D]. Linux only [I].
4. **nsjail or bubblewrap on Linux/WSL2.** Fast and small; process-level, not a VM [I].
5. **Firecracker or Kata.** Strongest local option, but Linux with KVM or VT-x only [D] and heavy to run [I]. For Python-only checks, Pyodide is a fallback [I].

**(2) Hosted service**
1. **Modal.** gVisor documented [D]; recurring $30/mo credit [D]; 24 h max timeout [D].
2. **Vercel Sandbox.** Recurring free allotment [D]; 45 min session and 10 concurrent on Hobby [D]; Firecracker claim is vendor-only [V].
3. **E2B.** Isolated VM [D]; $100 one-time credit [D]; 1 h session [D]. Confirm recurring vs one-time.
4. **GitHub Codespaces / Actions.** Free quotas documented [D]; for trusted repos, not hostile code [I].
5. **Daytona.** Good docs, but the repo is archived with no further updates [D]. Hosted service status [U]. Ranked last.

## Notes
- No hosted option is documented as offline [I].
- Judge0 GPL-3.0 and AGPL-3.0 (Daytona) need legal review before bundling in a plugin [I].
- Docker Desktop licence: thresholds are from docs.docker.com; the page shows no date [D].
- Sources disagree on E2B and Daytona free tiers. Official pages were preferred where fetched.

## Sources (fetched 2026-10-18)
- https://e2b.dev/pricing ; https://docs.e2b.dev/sandbox
- https://www.daytona.io/docs/en/limits/ ; https://github.com/daytonaio/daytona ; https://daytona.io/docs/
- https://modal.com/pricing ; https://modal.com/docs/guide/sandbox
- https://vercel.com/docs/sandbox/pricing ; https://vercel.com/kb/guide/run-terminal-bench-harbor-benchmarks-vercel-sandbox
- https://docs.github.com/en/billing/concepts/product-billing/github-codespaces ; https://docs.github.com/en/billing/concepts/product-billing/github-actions
- https://research.google.com/colaboratory/faq.html ; https://www.kaggle.com/docs/notebooks (no content returned)
- https://riza.io/pricing
- https://github.com/judge0/judge0 ; https://www.hostinger.com/applications/judge0-ce
- https://github.com/engineer-man/piston
- https://github.com/google/gvisor ; https://github.com/firecracker-microvm/firecracker ; https://github.com/kata-containers/kata-containers
- https://github.com/containers/bubblewrap ; https://github.com/google/nsjail ; https://github.com/containers/podman
- https://github.com/pyodide/pyodide ; https://github.com/zopefoundation/RestrictedPython
- https://docs.docker.com/subscription/desktop-license/ ; https://docs.docker.com/desktop/setup/install/windows-install/
- https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-overview
- https://learn.microsoft.com/en-us/windows/wsl/install

## Checklist
- DONE: local licences and isolation (fetched); hosted limits for E2B, Daytona, Modal, Vercel, GitHub, Colab, Riza; Windows editions (Sandbox, WSL2, Docker Desktop).
- NOT DONE / UNVERIFIED: firejail; Hyper-V edition page; Kaggle official limits; CodeSandbox, Replit; RestrictedPython and Codespaces licences; E2B isolation tech; Daytona hosted status.
- BLOCKED: none.
- WAITING ON USER: none for this research. Any hosted trial that sends code off this machine needs a yes.
