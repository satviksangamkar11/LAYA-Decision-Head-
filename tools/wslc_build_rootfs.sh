#!/bin/bash
# [WSLC backend] Build a MINIMAL, reproducible rootfs tar for MXC WSLC (wslc.imageTarPath) from the already-qualified Ubuntu-24.04 WSL distro. NO download.
# Contents: bash + the coreutils the runners use, the distro's python3.12 (system), the toolchains python 3.13 / JDK 26.0.1 / Go 1.26.4 from /opt/toolchains (same paths as the qualified WSL
# sandbox), the shared libraries they need (ldd closure), a tiny /etc. Run INSIDE the distro as root:
#     wsl -d Ubuntu-24.04 -u root -- bash "/mnt/d/local model/tools/wslc_build_rootfs.sh" "/mnt/d/local model/models/wslc-image/wslc_rootfs_v1.tar"
# The tar is deterministic (sorted names, mtime 0, numeric owner); its sha256 is the image digest recorded in configs/wslc_image_v1.toml.
set -euo pipefail
OUT="$1"
R="$(mktemp -d /tmp/wslc-rootfs.XXXXXX)"
trap 'rm -rf "$R"' EXIT
mkdir -p "$R"/{proc,sys,dev,tmp,work,etc,root,home,run,var/tmp,opt/toolchains,usr/bin,usr/lib,usr/lib64,usr/sbin}
chmod 1777 "$R/tmp" "$R/var/tmp"
ln -s usr/bin "$R/bin"; ln -s usr/lib "$R/lib"; ln -s usr/lib64 "$R/lib64"; ln -s usr/sbin "$R/sbin"

copy_file() {   # copy a file (dereferencing symlinks) to the same path inside $R
    local f="$1"
    [ -e "$f" ] || return 0
    cp -L --parents "$f" "$R" 2>/dev/null || true
}
copy_libs() {   # ldd closure of one ELF file
    { ldd "$1" 2>/dev/null || true; } | awk '/=>/ {print $3} /^[[:space:]]*\/lib/ {print $1}' | while read -r l; do if [ -n "$l" ]; then copy_file "$l"; fi; done || true
}

BINS="bash sh dash ls cat echo mkdir rm cp mv touch head tail sort uniq wc diff env timeout sleep true false sed grep find tr cut tee date pwd basename dirname chmod ln id uname xargs readlink stat test expr printf cmp whoami ps kill mktemp realpath tar gzip which mkfifo seq yes nproc df sha256sum md5sum od wc"
for b in $BINS; do
    p="$(command -v "$b" 2>/dev/null || true)"
    [ -n "$p" ] && [ -f "$p" ] || continue
    p="$(readlink -f "$p")"
    copy_file "$p"; copy_libs "$p"
    [ -e "$R/usr/bin/$b" ] || ln -s "$(basename "$p")" "$R/usr/bin/$b" 2>/dev/null || true
done
# bash must be /usr/bin/bash and /bin/sh -> bash is fine for the runners (dash kept as /usr/bin/dash)
ln -sf bash "$R/usr/bin/sh"
cp -L --parents /usr/lib64/ld-linux-x86-64.so.2 "$R" 2>/dev/null || cp -L --parents /lib64/ld-linux-x86-64.so.2 "$R"

# system python 3.12 (stdlib without tests)
copy_file /usr/bin/python3.12; copy_libs /usr/bin/python3.12
(cd / && find usr/lib/python3.12 -type f ! -path '*/test/*' ! -path '*/idlelib/*' ! -path '*/turtledemo/*' ! -path '*/ensurepip/*' ! -name '*.pyc' | while read -r f; do cp --parents -a "$f" "$R" ; done)
find usr/lib/python3.12/lib-dynload -name '*.so' 2>/dev/null | while read -r so; do copy_libs "/$so"; done || true
for so in /usr/lib/python3.12/lib-dynload/*.so; do copy_libs "$so"; done
ln -sf python3.12 "$R/usr/bin/python3.12-system"

# toolchains (same absolute paths as the qualified WSL sandbox)
for d in jdk-26.0.1 python go1.26.4; do cp -a "/opt/toolchains/$d" "$R/opt/toolchains/$d"; done
mkdir -p "$R/opt/toolchains/bin"
for t in java javac python python3 go gofmt; do [ -L "/opt/toolchains/bin/$t" ] && cp -a "/opt/toolchains/bin/$t" "$R/opt/toolchains/bin/$t"; done
# shared libs of the toolchain binaries
find "$R/opt/toolchains" -type f \( -name '*.so' -o -name '*.so.*' -o -perm -u+x \) | while read -r f; do
    if head -c4 "$f" 2>/dev/null | grep -q 'ELF'; then { ldd "$f" 2>/dev/null || true; } | awk '/=>/ {print $3}' | while read -r l; do case "$l" in /opt/*) ;; *) copy_file "$l";; esac; done || true; fi
done
copy_file /etc/ld.so.cache
if [ -d /usr/lib/locale/C.utf8 ]; then mkdir -p "$R/usr/lib/locale"; cp -a /usr/lib/locale/C.utf8 "$R/usr/lib/locale/C.utf8"; fi

cat > "$R/etc/passwd" <<'EOF'
root:x:0:0:root:/root:/bin/bash
nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
EOF
cat > "$R/etc/group" <<'EOF'
root:x:0:
nogroup:x:65534:
EOF
printf 'hosts: files\n' > "$R/etc/nsswitch.conf"
printf '127.0.0.1 localhost\n' > "$R/etc/hosts"
printf 'wslc-rootfs-v1\n' > "$R/etc/wslc-image-id"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -C "$R" -cf "$OUT" .
sha256sum "$OUT" | awk '{print "sha256:" $1}'
ls -l "$OUT" | awk '{print "bytes " $5}'
du -sh "$R" | awk '{print "tree " $1}'
