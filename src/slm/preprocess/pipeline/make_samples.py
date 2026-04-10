from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("./artifacts/data/unified_python")
OUTPUT_SIZES = [100, 1000, 10000]


def get_parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.parquet"))


def collect_first_n_rows(folder: Path, n: int) -> pa.Table:
    parquet_files = get_parquet_files(folder)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")

    chunks = []
    rows_collected = 0

    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)

        for row_group_idx in range(pf.num_row_groups):
            table = pf.read_row_group(row_group_idx)

            rows_needed = n - rows_collected
            if rows_needed <= 0:
                break

            if table.num_rows > rows_needed:
                table = table.slice(0, rows_needed)

            chunks.append(table)
            rows_collected += table.num_rows

            if rows_collected >= n:
                break

        if rows_collected >= n:
            break

    if not chunks:
        raise ValueError(f"No rows could be read from parquet files in: {folder}")

    result = pa.concat_tables(chunks)

    if result.num_rows < n:
        print(f"Warning: only found {result.num_rows} rows, less than requested {n}.")

    return result


def save_samples(folder: Path, output_sizes: list[int]) -> None:
    max_n = max(output_sizes)
    full_sample = collect_first_n_rows(folder, max_n)

    for n in output_sizes:
        out_dir = Path(f"sample_{n}")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_table = full_sample.slice(0, min(n, full_sample.num_rows))
        out_path = out_dir / f"sample_{n}.parquet"

        pq.write_table(out_table, out_path)
        print(f"Saved {out_table.num_rows} rows to {out_path}")


if __name__ == "__main__":
    save_samples(INPUT_DIR, OUTPUT_SIZES)
