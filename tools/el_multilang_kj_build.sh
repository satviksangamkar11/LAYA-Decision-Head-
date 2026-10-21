#!/bin/bash
# builds the Kaggle-jail rootfs on the WSL proxy (root); usage: bash tools/el_multilang_kj_build.sh "c,cpp,javascript,go,rust"
export EL_KJAIL_ROOT=/var/lib/kjail-root
export PYTHONPATH=.
export PATH="/opt/toolchains/bin:$PATH"
TC="${1:-c,cpp,javascript,go,rust}"
/usr/bin/python3 - <<EOF
import json, time
from evidence_ladder.execution import kaggle_jail as K
t0 = time.time()
r = K.build_rootfs(toolchains=tuple(x for x in "$TC".split(",") if x))
r["seconds"] = round(time.time() - t0, 1)
print(json.dumps(r))
EOF
du -sh /var/lib/kjail-root
