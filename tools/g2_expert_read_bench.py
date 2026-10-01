"""G2: expert-slice read speed from the gpt-oss-120b MXFP4 shards. Read-only.

Simulates decode: per token, 36 layers in sequence, 4 distinct experts per layer read in
parallel, 4 reads per expert. Output is create-only under results/raw/.
"""
import ctypes, glob, json, os, platform, random, re, statistics, struct, sys, time
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = r"D:\local model\gpt-oss-120b"
TOKENS, LAYERS, TOPK, NEXP = 12, 36, 4, 128
PAT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj_(blocks|scales)$")
ALIGN = 4096


def layout():
    idx = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            m = PAT.match(k) if k != "__metadata__" else None
            if m:
                a, b = v["data_offsets"]
                assert v["shape"][0] == NEXP and (b - a) % NEXP == 0
                idx.setdefault(int(m.group(1)), []).append((f, 8 + n + a, (b - a) // NEXP))
    assert len(idx) == LAYERS and all(len(v) == 4 for v in idx.values())
    return idx


class Buffered:
    def __init__(self):
        self.fh, self.buf = {}, bytearray(16 << 20)

    def read(self, f, off, size):
        fh = self.fh.get(f) or self.fh.setdefault(f, open(f, "rb", buffering=0))
        fh.seek(off)
        return fh.readinto(memoryview(self.buf)[:size])


k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.CreateFileW.restype = wintypes.HANDLE
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]


class NoBuf:
    """FILE_FLAG_NO_BUFFERING reads: aligned offset, size and buffer."""

    def __init__(self):
        self.h = {}
        raw = ctypes.create_string_buffer((16 << 20) + 2 * ALIGN)
        self.addr = (ctypes.addressof(raw) + ALIGN - 1) & ~(ALIGN - 1)
        self.raw = raw

    def read(self, f, off, size):
        h = self.h.get(f)
        if h is None:
            h = k32.CreateFileW(f, 0x80000000, 1, None, 3, 0x20000000 | 0x10000000, None)
            if h in (None, wintypes.HANDLE(-1).value):
                raise OSError(ctypes.get_last_error())
            self.h[f] = h
        a0 = off & ~(ALIGN - 1)
        n = ((off + size + ALIGN - 1) & ~(ALIGN - 1)) - a0
        assert n <= 16 << 20
        if not k32.SetFilePointerEx(h, a0, None, 0):
            raise OSError(ctypes.get_last_error())
        got = wintypes.DWORD(0)
        if not k32.ReadFile(h, self.addr, n, ctypes.byref(got), None):
            raise OSError(ctypes.get_last_error())
        return got.value


def run(mode, threads, idx, experts, tok0):
    mk = Buffered if mode == "buffered" else NoBuf
    local = {}

    def reader():
        import threading
        t = threading.get_ident()
        return local.get(t) or local.setdefault(t, mk())

    def read_expert(args):
        layer, e = args
        r, tot = reader(), 0
        for f, base, per in idx[layer]:
            tot += r.read(f, base + e * per, per)
        return tot

    times, nbytes = [], 0
    pool = ThreadPoolExecutor(threads)
    for t in range(tok0, tok0 + TOKENS):
        t0 = time.perf_counter()
        for layer in range(LAYERS):
            chosen = [(layer, experts[layer][t * TOPK + j]) for j in range(TOPK)]
            nbytes += sum(pool.map(read_expert, chosen)) if threads > 1 else sum(map(read_expert, chosen))
        times.append(time.perf_counter() - t0)
    pool.shutdown()
    return times, nbytes


def main():
    out = os.path.join("results", "raw", "g2-expert-read-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    idx = layout()
    rng = random.Random(0)
    experts = {L: rng.sample(range(NEXP), NEXP) for L in range(LAYERS)}  # distinct per layer
    per_expert = sum(p for _, _, p in idx[0])
    print("per-expert bytes", per_expert, "per-token bytes", per_expert * TOPK * LAYERS, flush=True)
    res, tok0 = {}, 0
    for mode, th in [("buffered", 1), ("buffered", 4), ("nobuf", 1), ("nobuf", 4)]:
        # every run takes fresh experts (cold cache); 4 runs x 12 tokens x 4 = 192 > 128, so
        # buffered runs use tokens 0-23, unbuffered runs ignore the cache and reuse 0-11.
        t0 = tok0 if mode == "buffered" else 0
        times, nb = run(mode, th, idx, experts, t0)
        if mode == "buffered":
            tok0 += TOKENS
        key = "%s_t%d" % (mode, th)
        res[key] = dict(per_token_s=[round(x, 4) for x in times], p50=round(statistics.median(times), 4),
                        max=round(max(times), 4), gb_per_s=round(nb / sum(times) / 1e9, 3))
        print(key, "p50 %.3f s/token, %.2f GB/s" % (res[key]["p50"], res[key]["gb_per_s"]), flush=True)
    best = min(v["p50"] for v in res.values())
    go, tune = 1.0, 3.0
    verdict = "GO" if best <= go else "TUNE" if best <= tune else "STOP_AND_ASK"
    doc = dict(gate="G2", thresholds="configs/thresholds.toml g2_expert_read v1", host=platform.node(),
               per_expert_bytes=per_expert, tokens_per_run=TOKENS, results=res, best_p50=best, verdict=verdict)
    with open(out, "x") as fh:
        json.dump(doc, fh, indent=1)
    print("verdict", verdict, "->", out)


if __name__ == "__main__":
    main()
