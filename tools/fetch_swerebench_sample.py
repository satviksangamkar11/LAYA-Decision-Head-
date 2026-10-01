"""Fetch a spread-out sample of nebius/SWE-rebench-openhands-trajectories through the datasets server (read-only, no full download).

Usage: python tools/fetch_swerebench_sample.py OUT.jsonl [chunks=12] [rows_per_chunk=25]
Chunks are taken at evenly spaced offsets through the dataset so repositories vary. Output is create-only; one JSON object per row.
"""
import json, sys, time, urllib.request

DS = "nebius/SWE-rebench-openhands-trajectories"
out, chunks, per = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12, int(sys.argv[3]) if len(sys.argv) > 3 else 25


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # transient server or network error: back off and retry
            time.sleep(5 * (i + 1))
            err = e
    raise err


total = get(f"https://datasets-server.huggingface.co/size?dataset={DS}")["size"]["dataset"]["num_rows"]
n = 0
with open(out, "x", encoding="utf-8") as f:
    for c in range(chunks):
        off = c * (total - per) // max(1, chunks - 1)
        d = get(f"https://datasets-server.huggingface.co/rows?dataset={DS}&config=default&split=train&offset={off}&length={per}")
        for r in d["rows"]:
            row = dict(r["row"], _offset=r["row_idx"])
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
        print(f"chunk {c}: offset {off}, rows so far {n}", flush=True)
print("done", n, "rows of", total)
