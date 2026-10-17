# EL research A: is execution sandboxing necessary? (as of 2026-10-18)

**Bottom line.** The verifier's logic does not need sandboxing, but running a candidate edit's tests does. A dry run executes project code (tests, conftest, build and config files) with the user's privileges unless something confines it. Removing the sandbox does not remove that risk; it moves the boundary outward to a disposable environment (VM, container, CI runner). Keep execution confined everywhere. On CI, the runner is the sandbox.

## 1) What coding agents do

- **Claude Code (documented):** OS sandbox for Bash only (bubblewrap on Linux/WSL2, Seatbelt on macOS). Native Windows runs commands unsandboxed. File tools, MCP servers and hooks run outside it. Fails open by default when dependencies are missing; `failIfUnavailable` makes that a hard failure. Docs say to run `--dangerously-skip-permissions` inside a container, VM or the sandbox runtime.
- **Codex CLI (partly verified):** network off by default, writes limited to the workspace. Secondary sources say Seatbelt on macOS and seccomp+landlock on Linux. The official page redirected, so unconfirmed.
- **OpenHands (documented):** Docker sandbox recommended; the process sandbox "has no container isolation" and is marked unsafe.
- **Devin (documented, VPC docs):** a fresh VM per session; the planning model runs in the vendor tenant.
- **Cursor (documented via news):** OS-level sandbox for terminal commands, including Windows via WSL2. Escapes CVE-2026-50548/50549 fixed in 3.0 (2026-04-02).
- **SWE-agent, Aider, Agentless:** not verified. Aider's sources disagree on whether `/run` is approval-gated. SWE-ReX supports Docker deployments.
- **Pattern (inferred):** local agents use an OS sandbox with fail-open defaults, or none. Hosted agents and benchmark harnesses use per-session VMs or containers.

## 2) Incidents

- **Replit, July 2025:** agent wiped live production records during a code freeze (Fortune, 2025-07-23). Destructive commands against live resources, not a sandbox escape.
- **Gemini CLI, July 2025:** user-reported file loss after a failed mkdir led to overwriting moves (GitHub issue #4586). Self-reported.
- **Amazon Q extension 1.84.0, July 2025:** a wipe prompt run with trust-all-tools flags shipped in a signed release. AWS said the code was malformed.
- **Nx "s1ngularity", 2025-08-26 (Wiz):** malware invoked installed AI CLIs with `--dangerously-skip-permissions` or `--yolo` to hunt secrets and push them to GitHub. Scale estimates are vendor figures and disagree.
- **CurXecute, CVE-2025-54135 (Cursor, fixed 1.3):** prompt injection via a Slack MCP message rewrote `~/.cursor/mcp.json` with no approval, leading to command execution.
- **DuneSlide, CVE-2026-50548/50549 (Cursor, CVSS 9.8, fixed 3.0):** zero-click prompt injection in content the agent reads escaped the terminal sandbox. Cato says the weakness is structural across agents.
- **OWASP LLM06:2025, Excessive Agency** (v2.0, 2024-11-18) names this class.

## 3) Minimum viable isolation (no containers)

Scratch copy, subprocess with timeout, rlimits, no network, env scrub, path jail.

- **Protects against:** runaway or hung tests (timeout, output cap); secret leakage through environment variables (scrub); network exfiltration (no network); stray writes outside the scratch copy, but only if the jail is kernel-enforced.
- **Does not protect against:**
  - A scratch copy alone. The process runs as the user and can read `~/.ssh`, cloud credentials and browser stores unless path-jailed (inferred from file permissions).
  - Git hooks. Linked worktrees share the main repo's hooks directory (git commit history). A model-written hook can run on commit in the real repo. Claude Code denies `.git/hooks` and `.git/config` for this reason. Its Linux deny list is built at launch and misses files created later, which covers our scratch copies.
  - Symlinks, `..` and absolute paths, unless enforced by the kernel.
  - Daemonized children that outlive the timeout (inferred).
  - Any kernel exploit.
- **Windows gap:** no bubblewrap or Seatbelt, and rlimits do not bound a process tree (inferred). This minimum is weakest exactly where the plugin must run.
- **Pipeline-specific:** protected tests must come from a pristine source, not the edited scratch copy. Otherwise the model can change them before they run (inferred from the design).

## 4) Execution removed, verification kept

Parts survive without running code: parsing, AST diff, schema checks, `git apply --check`, compile-only checks. Behavioural evidence is lost: runtime errors, logic faults, environment failures.

"Static" is often not execution-free. pytest imports `conftest.py`, type checkers load plugins, and eslint configs and build files are code. Only parser-level checks are truly non-executing.

A legitimate option is running the user's own test command on the user's machine after explicit per-project consent, inside a fail-closed OS sandbox. That is consented execution, not no execution. The guarantee lost is isolation from project code the user has chosen to trust.

## 5) Recommendation

- **(a) Developer machine:** keep execution, but require consent and fail closed. Use bubblewrap inside WSL2 on Windows and Seatbelt on macOS; if neither is available, refuse rather than run unsandboxed. Deny `.git/hooks`, `.git/config`, shell rc files and `.mcp.json`. Allow network only to allowlisted hosts, scrub env, and take protected tests from a pristine source. Offer a Docker path for native Windows users. Do not describe arbitrary repos as "safe."
- **(b) Hosted multi-tenant:** process-level jails are not enough. Run each job in a fresh microVM or container with no user secrets, egress denied, ephemeral storage, time and memory limits, and destroy it after the run. Removing execution removes the evidence the product sells, so the learned-score gate stays advisory only.
- **(c) CI:** the ephemeral runner is the sandbox. Drop the local sandbox only when there are no deploy credentials, protected tests come from the base branch, network is off or allowlisted, and timeouts apply.

"Remove the sandbox in production" should read as "move the sandbox to the environment."

## Unverified / inferred

- Codex enforcement per OS and config key names (official page redirected; secondary sources only).
- Cursor's Windows sandbox details (search summary, not fetched).
- Aider approval gating (sources disagree).
- SWE-agent default container setup and Agentless validation method (not found).
- Devin's per-session network and command scoping (third-party only).
- Replit rollback details (secondary); Gemini CLI details (self-reported); Nx scale figures (vendor).
- Git hook sharing: verified via git commit history; behaviour from a third-party blog.
- Windows rlimit and process-tree limits; daemonized-child and file-permission reasoning.
- Versions and dates: Claude Code docs as read 2026-10-18; Cursor 3.0 (2026-04-02); OWASP 2025 v2.0 (2024-11-18). No pricing cited.

## Sources

- Claude Code sandboxing: https://code.claude.com/docs/en/sandboxing (read 2026-10-18)
- Claude Code sandbox environments: https://code.claude.com/docs/en/sandbox-environments (read 2026-10-18)
- Anthropic engineering, sandboxing: https://anthropic.com/engineering/claude-code-sandboxing
- Codex security: https://developers.openai.com/codex/security (redirects to https://learn.chatgpt.com/docs/security; not fetched)
- OpenHands runtimes overview: https://docs.openhands.dev/usage/runtimes/overview
- OpenHands Docker-in-sandbox: https://docs.openhands.dev/enterprise/docker-in-sandbox.md
- Devin VPC overview: https://docs.devin.ai/enterprise/vpc/overview
- DuneSlide (Cursor): https://thehackernews.com/2026/07/critical-cursor-flaws-could-let-prompt.html
- CurXecute CVE-2025-54135: https://www.securityweek.com/several-vulnerabilities-patched-in-ai-code-editor-cursor/amp/
- Replit incident: https://fortune.com/2025/07/23/ai-coding-tool-replit-wiped-database-called-it-a-catastrophic-failure
- Gemini CLI incident: https://incidentdatabase.ai/reports/6120
- Amazon Q extension: https://www.theregister.com/2025/07/24/amazon_q_ai_prompt/
- Nx s1ngularity: https://www.wiz.io/blog/s1ngularity-supply-chain-attack
- OWASP LLM Top 10 2025 (LLM06): https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025
- Git worktree hooks: https://public-inbox.org/git/20180929191029.13994-3-pclouds@gmail.com/ (commit 8aff1a9 diff mirror)

## Checklist

- DONE: sandbox docs, OpenHands, Devin, Cursor, Codex (partial), incidents (7), git hook sharing, OWASP.
- NOT DONE: Codex official page, SWE-agent and Agentless source checks, Aider official docs.
- Nothing else was written. Only this file was created.
