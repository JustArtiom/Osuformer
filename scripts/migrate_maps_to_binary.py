from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.cache.paths import CachePaths
from src.config.loader import load_config


@click.command()
@click.option("--name", required=True, type=str)
@click.option("--config-path", default="config/config.yaml", type=click.Path(path_type=Path))
@click.option("--batch-size", default=256, type=int, help="Rows per streaming batch.")
@click.option("--delete-old/--keep-old", default=False)
def main(name: str, config_path: Path, batch_size: int, delete_old: bool) -> None:
    cfg = load_config(str(config_path))
    paths = CachePaths(root=Path(cfg.paths.cache) / name)
    if not paths.maps.exists():
        raise SystemExit(f"no maps.parquet at {paths.maps}")
    if paths.maps_index.exists() and paths.maps_bin.exists():
        raise SystemExit(f"{paths.maps_index} already exists — aborting (remove it to re-run)")

    pf = pq.ParquetFile(paths.maps)
    total_rows = pf.metadata.num_rows
    print(f"migrating {total_rows} rows from {paths.maps} ({paths.maps.stat().st_size / 1e9:.1f} GB)")
    print(f"writing   → {paths.maps_bin} + {paths.maps_index}")

    all_columns: list[str] = pf.schema_arrow.names
    metadata_columns = [c for c in all_columns if c not in {"event_types", "event_values"}]

    index_rows: list[dict] = []
    next_slot = 0
    skipped_corrupt = 0
    int32_max = np.iinfo(np.int32).max
    int32_min = np.iinfo(np.int32).min

    with open(paths.maps_bin, "wb") as bin_handle:
        with tqdm(total=total_rows, desc="maps") as pbar:
            for batch in pf.iter_batches(batch_size=batch_size):
                types_col = batch.column("event_types").to_pylist()
                values_col = batch.column("event_values").to_pylist()
                metadata = {c: batch.column(c).to_pylist() for c in metadata_columns}
                for i in range(batch.num_rows):
                    types_raw = np.asarray(types_col[i], dtype=np.int64)
                    values_raw = np.asarray(values_col[i], dtype=np.int64)
                    if types_raw.shape != values_raw.shape:
                        skipped_corrupt += 1
                        pbar.update(1)
                        continue
                    if (types_raw > int32_max).any() or (types_raw < int32_min).any():
                        skipped_corrupt += 1
                        pbar.update(1)
                        continue
                    if (values_raw > int32_max).any() or (values_raw < int32_min).any():
                        skipped_corrupt += 1
                        pbar.update(1)
                        continue
                    types = types_raw.astype(np.int32)
                    values = values_raw.astype(np.int32)
                    flat = np.empty(types.shape[0] * 2, dtype=np.int32)
                    flat[0::2] = types
                    flat[1::2] = values
                    bin_handle.write(flat.tobytes(order="C"))
                    row = {c: metadata[c][i] for c in metadata_columns}
                    row["slot_offset"] = int(next_slot)
                    row["n_events"] = int(types.shape[0])
                    index_rows.append(row)
                    next_slot += flat.shape[0]
                    pbar.update(1)

    if skipped_corrupt:
        print(f"skipped {skipped_corrupt} maps with out-of-range event values")

    print("writing index parquet...")
    pq.write_table(pa.Table.from_pylist(index_rows), paths.maps_index)

    bin_size = paths.maps_bin.stat().st_size
    idx_size = paths.maps_index.stat().st_size
    old_size = paths.maps.stat().st_size
    print(f"done: maps.bin = {bin_size / 1e9:.2f} GB  maps_index.parquet = {idx_size / 1e6:.1f} MB  (was {old_size / 1e9:.2f} GB)")

    if delete_old:
        paths.maps.unlink()
        print(f"deleted {paths.maps}")


if __name__ == "__main__":
    main()
