"""Read ONE trajectory from the on-disk SWE-rebench-openhands parquet without loading the trajectory column (streams small batches of one row group)."""
import pyarrow.parquet as pq

P = "data/nebius_swe_rebench/trajectories.parquet"


def load(trajectory_id):
    f = pq.ParquetFile(P, pre_buffer=False, buffer_size=1 << 20)
    off = 0
    for g in range(f.metadata.num_row_groups):
        ids = f.read_row_group(g, columns=["trajectory_id"])["trajectory_id"].to_pylist()
        if trajectory_id in ids:
            k = ids.index(trajectory_id)
            seen = 0
            for b in f.iter_batches(batch_size=8, row_groups=[g], columns=["trajectory_id", "instance_id", "resolved", "trajectory"]):
                if seen + b.num_rows > k:
                    return b.slice(k - seen, 1).to_pylist()[0]
                seen += b.num_rows
        off += len(ids)
    raise KeyError(trajectory_id)
