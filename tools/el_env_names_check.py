"""Prints ONLY which key-like variable NAMES exist in this shell (names and counts, never values). Used before the live critic run."""
import os
pre = sorted(k for k in os.environ if k.startswith("NVIDIA_API_KEY"))
print("NVIDIA_API_KEY* names present:", pre, "| count:", len(pre), "| all non-empty:", all(bool(os.environ[k].strip()) for k in pre) if pre else None)
other = sorted(k for k in os.environ if any(w in k.upper() for w in ("NVIDIA", "NVAPI", "NGC", "NIM_", "GLM")) and k not in pre)
print("other NVIDIA/NGC/NIM/GLM-like names:", other)
print("names containing 'KEY' (names only):", sorted(k for k in os.environ if "KEY" in k.upper() and k not in pre and k not in other))
