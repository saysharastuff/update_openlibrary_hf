import os
import sys
import gzip
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
import argparse
import time
from huggingface_hub import HfApi

from fetch_and_upload import upload_with_chunks, load_manifest, save_manifest, login

MAX_PARQUET_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # Target max chunk size ~4GB

def normalize_record(record):
    if "bio" in record:
        if isinstance(record["bio"], dict):
            record["bio"] = record["bio"].get("value", "")
        elif not isinstance(record["bio"], str):
            record["bio"] = str(record["bio"])

    if "notes" in record:
        if isinstance(record["notes"], dict):
            record["notes"] = record["notes"].get("value", "")
        elif not isinstance(record["notes"], str):
            record["notes"] = str(record["notes"])

    if "description" in record:
        if isinstance(record["description"], dict):
            record["description"] = record["description"].get("value", "")
        elif not isinstance(record["description"], str):
            record["description"] = str(record["description"])

    return record


def write_chunk(records: List[dict], chunk_index: int, output_prefix: str, dry_run: bool, manifest: dict, source_last_modified: str, input_file: str):
    chunk_path = f"{output_prefix}.parquet" if chunk_index == 0 and chunk_index == 0 else f"{output_prefix}.part{chunk_index}.parquet"

    print(f"📦 Attempting to write chunk {chunk_index} with {len(records)} records")
    if not records:
        print(f"⚠️ Skipping chunk {chunk_index} — no valid records.")
        return None

    schema = pa.Schema.from_pandas(pd.DataFrame(records))
    with pq.ParquetWriter(chunk_path, schema, compression="snappy") as writer:
        for i in range(0, len(records), 100_000):
            batch = records[i:i+100_000]
            df = pd.DataFrame(batch)
            if df.empty:
                continue
            table = pa.Table.from_pandas(df)
            table = table.cast(schema)
            writer.write_table(table)
    print(f"✅ Wrote {chunk_path} ({len(df)} rows)")

    login(token=os.environ["HF_TOKEN"])

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
    final_key = f"{output_prefix}.parquet" if chunk_index == 0 else chunk_path
    parent_entry["converted_chunks"][final_key] = {
        "last_synced": pd.Timestamp.utcnow().isoformat() + "Z",
        "converted": True
    }
    return chunk_path


def convert_to_parquet_chunks(input_file: str, output_prefix: str, dry_run: bool = False):
    chunk_index = 0
    buffer = []
    buffer_limit = 100_000
    current_size = 0
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
            time.sleep(2 ** attempt)

    with f:
        bad_lines = []
        writer = None
        chunk_path = f"{output_prefix}.part{chunk_index}.parquet"
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0:
                print(f"📈 Processed {i:,} lines, {parsed_records:,} parsed so far...")
            total_lines += 1
            try:
                json_part = line.strip().split('	')[-1]
                record = normalize_record(json.loads(json_part))
                parsed_records += 1
                buffer.append(record)
            except json.JSONDecodeError:
                if len(bad_lines) < 5:
                    bad_lines.append(line.strip())
                continue

            if len(buffer) >= buffer_limit:
                if writer is None:
                    df_sample = pd.DataFrame(buffer)
                    schema = pa.Table.from_pandas(df_sample).schema
                    df = pd.DataFrame(buffer)
                for name in schema.names:
                    if name not in df.columns:
                        df[name] = None
                table = pa.Table.from_pandas(df)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(chunk_path, schema, compression="snappy")
                else:
                    missing_fields = [name for name in schema.names if name not in table.schema.names]
                    if missing_fields:
                        print(f"⚠️ Skipping batch — missing fields not in schema: {missing_fields}")
                        buffer.clear()
                        continue
                    missing_fields = [name for name in schema.names if name not in table.schema.names]
                    if missing_fields:
                        print(f"⚠️ Skipping batch — missing fields not in schema: {missing_fields}")
                        buffer.clear()
                        continue
                    table = table.select([name for name in schema.names if name in table.schema.names]).cast(schema)
                writer.write_table(table)
                buffer.clear()
                current_size = os.path.getsize(chunk_path) if os.path.exists(chunk_path) else 0
                if current_size >= MAX_PARQUET_SIZE_BYTES:
                    writer.close()
                    login(token=os.environ["HF_TOKEN"])
                    upload_with_chunks(chunk_path, chunk_path)
                    os.remove(chunk_path)
                    print(f"🧹 Deleted {chunk_path} after upload")
                    parent_entry = manifest.setdefault(input_file, {})
                    if not dry_run:
                        save_manifest(manifest)
                    parent_entry.setdefault("converted_chunks", {})
                    parent_entry["source_last_modified"] = source_last_modified
                    parent_entry["last_synced"] = pd.Timestamp.utcnow().isoformat() + "Z"
                    parent_entry["converted_chunks"][chunk_path] = {
                        "last_synced": pd.Timestamp.utcnow().isoformat() + "Z",
                        "converted": True
                    }
                    chunk_index += 1
                    writer = None
                    chunk_path = f"{output_prefix}.part{chunk_index}.parquet"
        if buffer:
            df = pd.DataFrame(buffer)
            for name in schema.names:
                if name not in df.columns:
                    df[name] = None
            table = pa.Table.from_pandas(df)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(chunk_path, schema, compression="snappy")
            else:
                table = table.select(schema.names).cast(schema)
        if writer and not buffer:
            # no final buffer, flush writer
            writer.close()
        if writer:
            writer.close()
            login(token=os.environ["HF_TOKEN"])
            upload_with_chunks(chunk_path, chunk_path)
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
        

    if not dry_run:
        # 🧽 Delete orphaned parquet chunks from HF
        known_chunks = {
            k for k in manifest.get(input_file, {}).get("converted_chunks", {}).keys()
            if k.endswith(".parquet")
        }
        actual_chunks = {
            f"{output_prefix}.parquet" if i == 0 else f"{output_prefix}.part{i}.parquet" for i in range(chunk_index + 1)
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
