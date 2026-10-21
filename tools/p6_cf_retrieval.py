"""[P6 dataset, option C] Selective retrieval from open-r1/codeforces `verifiable_tests` (test split, 40 shards, 9.3 GB) by HTTP range requests:
 index  : read each shard's parquet footer and its `problem_id` column only; map candidate problems -> (shard, row groups); estimate the bytes of the row
          groups that hold them (from parquet metadata, nothing else downloaded).
 fetch  : download only those row groups (all columns) and keep only the candidates' rows -> one local parquet per shard; stops before exceeding the budget.
Every HTTP byte is counted; the total is reported and compared with the allowance. Usage:
  python -I tools/p6_cf_retrieval.py index <candidates.json> <out dir>
  python -I tools/p6_cf_retrieval.py fetch <out dir> <budget MB>"""
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/open-r1/codeforces/resolve/main/verifiable_tests/test-%05d-of-00040.parquet"
NSHARDS = 40
COUNTED = {"bytes": 0, "requests": 0}


class RangeFile:
    """Seekable read-only file over HTTP range requests; counts every byte received."""

    def __init__(self, url):
        self.url, self.pos, self.closed = url, 0, False
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.final = url

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, end)})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:
                raise IOError("server ignored the range request (status %s)" % r.status)
            data = r.read()
        COUNTED["bytes"] += len(data); COUNTED["requests"] += 1
        return data

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        data = self._get(self.pos, end)
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        self.closed = True


def index(cand_path, out):
    want = set(json.loads(Path(cand_path).read_text("utf-8")))
    rows = []
    for k in range(NSHARDS):
        f = RangeFile(BASE % k)
        pf = pq.ParquetFile(f)
        md = pf.metadata
        cols = [md.schema.column(i).name for i in range(md.num_columns)]
        pid_col = cols.index("problem_id")
        for g in range(md.num_row_groups):
            ids = set(pf.read_row_group(g, columns=["problem_id"]).column(0).to_pylist())
            hit = ids & want
            if hit:
                rg = md.row_group(g)
                rows.append({"shard": k, "row_group": g, "problems": sorted(hit), "rows": rg.num_rows, "problems_in_group": len(ids),
                             "group_bytes": sum(rg.column(c).total_compressed_size for c in range(md.num_columns))})
        print("shard", k, "bytes so far", COUNTED["bytes"], flush=True)
    found = sorted({p for r in rows for p in r["problems"]})
    rep = {"candidates": len(want), "found": len(found), "missing": sorted(want - set(found)), "groups": rows,
           "bytes_to_fetch": sum(r["group_bytes"] for r in rows), "index_bytes_downloaded": COUNTED["bytes"], "index_requests": COUNTED["requests"]}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "INDEX.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "groups"}, indent=1))


def fetch(out, budget_mb):
    rep = json.loads((Path(out) / "INDEX.json").read_text("utf-8"))
    want = {p for r in rep["groups"] for p in r["problems"]}
    if rep["bytes_to_fetch"] / 1e6 > budget_mb:
        sys.exit("refusing: %.1f MB needed > %.1f MB allowed" % (rep["bytes_to_fetch"] / 1e6, budget_mb))
    by_shard = {}
    for r in rep["groups"]:
        by_shard.setdefault(r["shard"], []).append(r["row_group"])
    for k, groups in sorted(by_shard.items()):
        pf = pq.ParquetFile(RangeFile(BASE % k))
        t = pa.concat_tables([pf.read_row_group(g) for g in groups])
        t = t.filter(pa.compute.is_in(t.column("problem_id"), value_set=pa.array(sorted(want))))
        pq.write_table(t, Path(out) / ("tests_shard%02d.parquet" % k))
        print("shard", k, "rows kept", t.num_rows, "downloaded so far MB", round(COUNTED["bytes"] / 1e6, 1), flush=True)
    (Path(out) / "FETCH.json").write_text(json.dumps({"downloaded_bytes": COUNTED["bytes"], "requests": COUNTED["requests"], "shards": sorted(by_shard)}, indent=1))
    print("downloaded MB", round(COUNTED["bytes"] / 1e6, 1))


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index(sys.argv[2], sys.argv[3])
    else:
        fetch(sys.argv[2], float(sys.argv[3]))
