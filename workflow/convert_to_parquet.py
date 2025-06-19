import os
import sys
import gzip
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
import argparse

from fetch_and_upload import upload_with_chunks, load_manifest, save_manifest, login

CHUNK_SIZE = 5_000_000  # Number of JSON lines per chunk

def normalize_record(record):
    if "bio" in record:
        if isinstance(record["bio"], dict):
            record["bio"] = record["bio"].get("value", "")
        elif not isinstance(record["bio"], str):
            record["bio"] = str(record["bio"])
    return record


def write_chunk(records: List[dict], chunk_index: int, output_prefix: str, dry_run: bool, manifest: dict, source_last_modified: str):
    print(f"📦 Attempting to write chunk {chunk_index} with {len(records)} records")
    df = pd.DataFrame(records)
    if df.empty:
        print(f"⚠️ Skipping chunk {chunk_index} — no valid records.")
        return None

    chunk_path = f"{output_prefix}.part{chunk_index}.parquet"

    if dry_run:
        print(f"[DRY RUN] Would write chunk {chunk_index} with {len(df)} rows to {chunk_path}")
        return None

    table = pa.Table.from_pandas(df)
    pq.write_table(table, chunk_path, compression="snappy")
    print(f"✅ Wrote {chunk_path} ({len(df)} rows)")

    login(token=os.environ["HF_TOKEN"])
    import time

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            upload_with_chunks(chunk_path, chunk_path)
            break
        except Exception as e:
            print(f"❌ Upload attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)  # exponential backoff
    os.remove(chunk_path)
    print(f"🧹 Deleted {chunk_path} after upload")

    parent_entry = manifest.setdefault(input_file, {})
    parent_entry.setdefault("converted_chunks", {})
    parent_entry["source_last_modified"] = source_last_modified
    parent_entry["last_synced"] = pd.Timestamp.utcnow().isoformat() + "Z"
    parent_entry["converted_chunks"][chunk_path] = {
        "last_synced": pd.Timestamp.utcnow().isoformat() + "Z",
        "converted": True
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

    source_entry = manifest.get(input_file, {})
    source_last_modified = source_entry.get("source_last_modified", "<unknown>")

    total_lines = 0
    parsed_records = 0

    import time

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            f = gzip.open(input_file, 'rt', encoding='utf-8', errors='ignore')
            break
        except Exception as e:
            print(f"❌ Download/open attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)  # exponential backoff

    with f:
        bad_lines = []
        for i, line in enumerate(f):
            total_lines += 1
            try:
                json_part = line.strip().split('	')[-1]
                record = normalize_record(json.loads(json_part))
                parsed_records += 1
                chunk.append(record)
            except json.JSONDecodeError:
                if len(bad_lines) < 5:
                    bad_lines.append(line.strip())
                continue

            if len(chunk) >= CHUNK_SIZE:
                write_chunk(chunk, chunk_index, output_prefix, dry_run, manifest, source_last_modified)
                chunk = []
                chunk_index += 1

    if chunk:
        write_chunk(chunk, chunk_index, output_prefix, dry_run, manifest, source_last_modified)

    if not dry_run:
        # 🧽 Delete orphaned parquet chunks from HF
        known_chunks = set(manifest.get(input_file, {}).get("converted_chunks", {}).keys())
        actual_chunks = {
            f"{output_prefix}.part{i}.parquet" for i in range(chunk_index + 1)
        }
        orphaned = known_chunks - actual_chunks

        if orphaned:
            from huggingface_hub import HfApi
            api = HfApi()
            for filename in orphaned:
                try:
                    print(f"🗑️ Deleting orphaned chunk from HF: {filename}")
                    api.delete_file(
                        path_in_repo=filename,
                        repo_id="sayshara/ol_dump",
                        repo_type="dataset",
                        revision="main",
                        token=os.environ["HF_TOKEN"]
                    )
                    manifest[input_file]["converted_chunks"].pop(filename, None)
                except Exception as e:
                    print(f"⚠️ Failed to delete {filename}: {e}")

        save_manifest(manifest)

    print(f"📊 Processed {total_lines} lines — parsed {parsed_records} JSON objects.")
    if bad_lines:
        print("🔍 Example malformed lines:")
        for idx, bad in enumerate(bad_lines, 1):
            print(f"  [{idx}] {bad[:200]}...")
    print(f"🌟 Finished {'simulated' if dry_run else ''} conversion into {chunk_index + 1} parquet chunks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input .txt.gz file")
    parser.add_argument("output_prefix", help="Prefix for output .parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Print intended actions without writing or uploading files")
    args = parser.parse_args()

    convert_to_parquet_chunks(args.input_file, args.output_prefix.replace(".parquet", ""), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
