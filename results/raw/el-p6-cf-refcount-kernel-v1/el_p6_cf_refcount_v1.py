"""[P6 Codeforces] Counts-only Kaggle CPU scan: per candidate problem, how many ACCEPTED submissions exist per programming language in
open-r1/codeforces-submissions (pinned 07e7bf31…, default config, 36 shards; downloaded and filtered on Kaggle, deleted after counting). Also counts all
submissions (any verdict) per candidate. No code is executed, no source is kept. Writes /kaggle/working/REFCOUNT.json."""
import collections, concurrent.futures as CF, json, os, tempfile, time, urllib.request
from pathlib import Path

REV = "07e7bf31f284b0e90e02dfc7ab675d5821e1c068"
URL = "https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/%s/data/train-%%05d-of-00036.parquet" % REV
CANDS = ["2021/B", "2022/A", "2024/A", "2024/B", "2025/A", "2025/B", "2026/A", "2027/A", "2027/B", "2028/A", "2029/A", "2029/B", "2030/A", "2030/C", "2031/A", "2031/B", "2032/A", "2033/A", "2033/B", "2034/A", "2034/B", "2035/A", "2035/B", "2036/A", "2036/B", "2036/C", "2037/A", "2037/B", "2038/J", "2038/N", "2039/A", "2039/B", "2039/C1", "2040/A", "2040/B", "2041/B", "2042/A", "2042/B", "2043/A", "2043/B", "2044/A", "2044/B", "2046/A", "2047/A", "2048/A", "2048/B", "2050/A", "2050/B", "2050/C", "2051/A", "2051/B", "2051/C", "2051/D", "2053/A", "2053/B", "2055/A", "2055/B", "2056/A", "2057/A", "2057/B", "2059/A", "2060/A", "2060/B", "2060/C", "2060/D", "2061/A", "2062/A", "2062/B", "2062/C", "2063/A", "2063/B", "2065/A", "2065/B", "2065/C1", "2065/D", "2067/A", "2067/B"]
OUT = Path("/kaggle/working")


def one(k, raw):
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    dest = raw / ("s%02d.parquet" % k)
    urllib.request.urlretrieve(URL % k, dest)
    t = pq.read_table(dest, columns=["problem_id", "programmingLanguage", "verdict"])
    t = t.filter(pc.is_in(t.column("problem_id"), value_set=pa.array(CANDS)))
    rows = t.to_pylist()
    dest.unlink()
    return rows


def main():
    raw = Path(tempfile.mkdtemp(dir="/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else None))
    acc = collections.defaultdict(collections.Counter); anyv = collections.Counter(); ids_seen = set()
    with CF.ThreadPoolExecutor(max_workers=6) as ex:
        for rows in ex.map(lambda k: one(k, raw), range(36)):
            for r in rows:
                anyv[r["problem_id"]] += 1; ids_seen.add(r["problem_id"])
                if r["verdict"] == "OK":
                    acc[r["problem_id"]][r["programmingLanguage"]] += 1
    langs = collections.Counter()
    for c in acc.values():
        langs.update(c)
    rep = {"candidates": len(CANDS), "with_any_submission": len(ids_seen), "with_accepted": sum(1 for p in CANDS if acc[p]),
           "accepted_by_language_total": dict(langs.most_common()), "per_problem": {p: {"any": anyv[p], "accepted": dict(acc[p])} for p in CANDS},
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "REFCOUNT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_problem"}, indent=1))


if __name__ == "__main__":
    main()
