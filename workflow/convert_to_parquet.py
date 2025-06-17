import sys
import os
import gzip
import json
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
import argparse

CHUNK_SIZE = 500_000  # Number of JSON lines per chunk


def write_chunk(records: List[dict], chunk_index: int, output_prefix: str, dry_run: bool, manifest: dict):
    df = pd.DataFrame(records)
    if df.empty:
        return None

    chunk_path = f"{output_prefix}.part{chunk_index}.parquet"

    if dry_run:
        print(f"[DRY RUN] Would write chunk {chunk_index} with {len(df)} rows to {chunk_path}")
        return None

    table = pa.Table.from_pandas(df)
    pq.write_table(table, chunk_path)
    print(f"✅ Wrote {chunk_path} ({len(df)} rows)")

    print(f"📤 Uploading {chunk_path} via fetch_and_upload.py")
    subprocess.run([sys.executable, "workflow/fetch_and_upload.py", "--upload-only", chunk_path])
    os.remove(chunk_path)
    print(f"🧹 Deleted {chunk_path} after upload")

    manifest[chunk_path] = {
        "last_synced": pd.Timestamp.utcnow().isoformat() + "Z",
        "source_last_modified": "converted"
    }
    return chunk_path


def convert_to_parquet_chunks(input_file: str, output_prefix: str, dry_run: bool = False):
    chunk = []
    chunk_index = 0
    manifest_path = "ol_sync_manifest.json"
    manifest = {}

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    with gzip.open(input_file, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                chunk.append(record)
            except json.JSONDecodeError:
                continue

            if len(chunk) >= CHUNK_SIZE:
                write_chunk(chunk, chunk_index, output_prefix, dry_run, manifest)
                chunk = []
                chunk_index += 1

    if chunk:
        write_chunk(chunk, chunk_index, output_prefix, dry_run, manifest)

    if not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n🌟 Finished {'simulated' if dry_run else ''} conversion into {chunk_index + 1} parquet chunks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input .txt.gz file")
    parser.add_argument("output_prefix", help="Prefix for output .parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Print intended actions without writing or uploading files")
    args = parser.parse_args()

    convert_to_parquet_chunks(args.input_file, args.output_prefix.replace(".parquet", ""), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
