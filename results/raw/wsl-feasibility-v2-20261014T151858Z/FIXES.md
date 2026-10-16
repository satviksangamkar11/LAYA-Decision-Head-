# v2 code fixes (SELECTION_v2.md, "Bugs in the untested v1 runner code")

None needed. The v1 test / replay / editor-emulation / hash code ran for the first time in v2 (googleapis__python-storage-1357 and
stephenhillier__starlette_exporter-89) and every stage completed without a code error. The one replay MISMATCH (googleapis step 6, `ls -la tests/`)
is a real output difference (file owner/group and mtime), not a code bug, so it was not "fixed" and the fidelity rule was not changed.

Design changes made BEFORE any v2 run (not bug fixes): disk guard, per-task caches on the /opt/conda mount with UV_LINK_MODE=hardlink and the
download cache removed after setup (v1 shared /var/lib/fable-runs/feas-v1/uvcache, copy mode), PIP_NO_CACHE_DIR=1, cleanup + fstrim after each task.
One edit to my own new v2 code before the first run: a string literal broken across lines by the patch script (syntax), fixed before execution.
