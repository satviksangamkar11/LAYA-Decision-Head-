#!/bin/bash
# usage: bash tmp/run_qual_wsl.sh <out dir (repo relative)> <backend: kaggle-jail|bare> [languages]
export EL_KJAIL_ROOT=/var/lib/kjail-root
export PYTHONPATH=.
export PATH="/opt/toolchains/bin:$PATH"
OUT="$1"; BE="$2"; LANGS="${3:-c,cpp,javascript,go,rust}"
/usr/bin/python3 tools/el_multilang_qual.py "$OUT" --backend "$BE" --languages "$LANGS"
