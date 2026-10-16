# WSL toolchain additions (2026-10-13, user: "fix Go / C# / Rust")

Adds to results/raw/wsl-toolchains-20261010/MANIFEST.md. Install root /opt/toolchains (mounted read-only in the sandbox). Checksums verified before extraction.

| Item | Version | Source | Checksum |
|---|---|---|---|
| .NET SDK | 10.0.302 linux-x64 (613 MB) | builds.dotnet.microsoft.com (release-metadata 10.0 releases.json) | sha512 from releases.json, OK |
| Rust (rustc, cargo, std) | 1.96.0 x86_64-unknown-linux-gnu (585 MB) | static.rust-lang.org | .sha256, OK |
| Go module cache | gopkg.in/yaml.v3 v3.0.1 + gopkg.in/check.v1 v0.0.0-20161208181325-20d25e280405 | proxy.golang.org | sum.golang.org at download |
| NuGet fallback folder | Microsoft.NET.Test.Sdk 17.14.1, xunit 2.9.3, xunit.runner.visualstudio 3.1.4 + dependencies (13 packages, 73 MB) | nuget.org via a dummy project (no trace code run) | NuGet package signatures |

Runner environment changes (tools/fable_wsl_runner.py): GOPROXY = file:///opt/toolchains/gomodcache/cache/download, GOSUMDB off; NUGET_PACKAGES /tmp/nuget,
RestoreAdditionalProjectFallbackFolders /opt/toolchains/nuget-packages, NuGetAudit false; /opt/toolchains/bin gains dotnet, cargo, rustc, rustdoc (ahead of
the system .NET 8 and rustc 1.75). New function compare_v2 (fidelity rule v2, formatting only; v1 unchanged). Sandbox re-qualified after the change:
results/raw/fable-wsl-runner-qualification-20261012T233639Z (18 of 18 probes).

Replay results after the repair (canonical TRAIN and VAL only; CAL / TEST never executed):
- Go: go.mod recreated in the replay copy when the recorded output proves the module name. 41 of 41 repaired states MATCH in both runs under rule v1
  (TRAIN 27, VAL 14: 25 + 2 and 13 + 1 from the two runs); 2 states already had go.mod and were not touched. Before: 0 of 14 VAL Go states replayed.
  Runs: go-mod-fidelity-v1-20261012T231723Z (43 states, before the module cache: 38 MATCH), -20261012T232637Z (43 states, yaml.v3 only: 38),
  -20261012T233557Z (the 3 yaml states after adding its module graph: 3 MATCH). -20261012T231107Z was aborted (no rows).
- C#: salesforce-query-cursor:4 and :5 build on .NET 10 and pass 7 of 7 tests like the trace; MISMATCH under rule v1 (relative vs absolute path in one
  line), MATCH under rule v2. csharp-kennelcard:2 has no project file in its workspace: not repairable.
- Rust: rs-sizecast:2 with a recreated Cargo.toml (crate name proven by `Doc-tests rs_sizecast` and the tests' `use rs_sizecast::`): 16 tests pass like the
  trace; MISMATCH under v1 (parallel test order), MATCH under v2. results/raw/cs-rust-fidelity-v1-20261012T233242Z.
