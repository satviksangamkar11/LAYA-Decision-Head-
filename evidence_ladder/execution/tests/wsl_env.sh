#!/bin/bash
# Usage (from Windows): wsl -d Ubuntu-24.04 -u root -e bash "<worktree>/evidence_ladder/execution/tests/wsl_env.sh" python3 evidence_ladder/execution/tests/run_tests.py
export PATH=/opt/toolchains/bin:$PATH
cd "$(dirname "$0")/../../.." || exit 9
exec "$@"
