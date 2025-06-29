# openlibrary_pipeline.py
"""OpenLibrary → Hugging Face Hub one‑stop pipeline.

Sub‑commands
------------
fetch   : download the latest raw OpenLibrary dump(s) and archive them on the Hub (branch *backup/raw*)
convert : convert a given dump to snappy‑compressed Parquet and upload it under authors/, editions/, or works/
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, login, upload_file

# ────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ────────────────────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
HF_REPO_ID: str = os.getenv("HF_REPO_ID", "sayshara/openlibrary")
MANIFEST_PATH = "ol_sync_manifest.json"
CHUNK_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB – chunked upload threshold

# Limit Parquet parts to roughly 3 GB of uncompressed JSON data
TARGET_BYTES = 3 * 1024 ** 3
BATCH_ROWS   = 50_000          # Arrow/Pandas rows kept in memory before writing

FILES: Dict[str, str] = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz",
}

# ────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ────────────────────────────────────────────────────────────────

def get_last_modified(url: str) -> str | None:
    for attempt in range(1, 4):
        try:
            r = requests.head(url, allow_redirects=True, timeout=10)
            r.raise_for_status()
            return r.headers.get("Last-Modified")
        except Exception as e:
            print(f"⚠️ HEAD attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return None


def ensure_branch_exists(branch: str = "backup/raw") -> None:
    """Ensure *branch* exists in the dataset repo (idempotent)."""
    HfApi().create_branch(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        branch=branch,
        token=HF_TOKEN,
        exist_ok=True,
    )


def upload_with_chunks(local_path: str | Path, repo_path: str, *, dry: bool = False, branch: str | None = None):
    """Upload *local_path* to the Hub, splitting >5 GB files into 5 GB parts."""
    local_path = str(local_path)
    size = os.path.getsize(local_path)
    revision = branch or ("backup/raw" if local_path.endswith(".txt.gz") else "main")

    if not dry and revision != "main":
        ensure_branch_exists(revision)

    def _single(src: str, dst: str):
        for attempt in range(1, 4):
            try:
                upload_file(
                    path_or_fileobj=src,
                    path_in_repo=dst,
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=revision,
                    token=HF_TOKEN,
                    commit_message="OpenLibrary sync",
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"⚠️ Upload attempt {attempt} failed: {e}")
                time.sleep(2 ** attempt)

    if size <= CHUNK_SIZE_BYTES:
        if not dry:
            _single(local_path, repo_path)
    else:
        with open(local_path, "rb") as f:
            idx = 0
            while chunk := f.read(CHUNK_SIZE_BYTES):
                part = f"{repo_path}.part{idx}" if idx else repo_path
                tmp = Path(f"__tmp_{idx}")
                tmp.write_bytes(chunk)
                if not dry:
                    _single(str(tmp), part)
                tmp.unlink()
                idx += 1

# ────────────────────────────────────────────────────────────────
# MANIFEST UTILITIES
# ────────────────────────────────────────────────────────────────

def load_manifest() -> Dict[str, dict]:
    return json.loads(Path(MANIFEST_PATH).read_text()) if Path(MANIFEST_PATH).exists() else {}


def save_manifest(data: Dict[str, dict]):
    Path(MANIFEST_PATH).write_text(json.dumps(data, indent=2))

# ────────────────────────────────────────────────────────────────
# FETCH COMMAND
# ────────────────────────────────────────────────────────────────

def _download_upload(name: str, url: str, manifest: Dict[str, dict], *, dry: bool, keep: bool):
    lm = get_last_modified(url) if not dry else "<dry>"
    if not dry and manifest.get(name, {}).get("source_last_modified") == lm and Path(name).exists():
        print(f"✅ {name} already up‑to‑date")
        return

    if not Path(name).exists():
        print(f"⬇️ Downloading {name}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(name, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)

    upload_with_chunks(name, name, dry=dry)

    manifest[name] = {
        "last_synced": datetime.utcnow().isoformat() + "Z",
        "source_last_modified": lm,
    }

    if not keep:
        Path(name).unlink(missing_ok=True)


def fetch_cli(args: argparse.Namespace):
    if not args.dry_run:
        login(token=HF_TOKEN)

    manifest = load_manifest()
    targets = [args.only] if args.only else list(FILES.keys())
    for t in targets:
        if t not in FILES:
            print(f"❌ Unknown dump {t}")
            continue
        _download_upload(t, FILES[t], manifest, dry=args.dry_run, keep=args.keep)

    if not args.dry_run:
        save_manifest(manifest)
        upload_with_chunks(MANIFEST_PATH, f"metadata/{MANIFEST_PATH}")

# ────────────────────────────────────────────────────────────────
# CONVERT COMMAND – streaming Parquet writer
# ────────────────────────────────────────────────────────────────

def _normalize(rec: dict) -> dict:
    for k, v in rec.items():
        if isinstance(v, (dict, list)):
            rec[k] = json.dumps(v)
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            rec[k] = str(v)
    return rec


def sniff_schema(path: str) -> pa.Schema:
    """Stream *path* once to determine the Parquet schema."""
    field_types: Dict[str, pa.DataType] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                record_json = line.rsplit("\t", 1)[-1]
                rec = _normalize(json.loads(record_json))
            except json.JSONDecodeError:
                continue
            for k, v in rec.items():
                if v is None:
                    typ = pa.string()
                else:
                    try:
                        typ = pa.scalar(v).type
                    except Exception:
                        typ = pa.string()
                prev = field_types.get(k)
                if prev is None:
                    field_types[k] = typ
                elif prev != typ:
                    field_types[k] = pa.string()

    fields = [pa.field(name, field_types.get(name, pa.string())) for name in sorted(field_types)]
    return pa.schema(fields)


def convert_cli(args: argparse.Namespace):
    login(token=HF_TOKEN)
    cfg = (args.config or re.search(r"ol_dump_(\w+)_latest", args.input_file).group(1)).lower()

    schema = sniff_schema(args.input_file)
    schema_json = {name: str(t) for name, t in zip(schema.names, schema.types)}
    schema_path = f"{cfg}_schema.json"
    Path(schema_path).write_text(json.dumps(schema_json, indent=2))
    if not args.dry_run:
        upload_with_chunks(schema_path, f"metadata/{schema_path}")

    idx = 0                   # part counter
    parsed = 0                # line counter
    writer: pq.ParquetWriter | None = None
    cur_size = 0              # uncompressed bytes in current part
    buf: list[dict] = []

    def out_path(i: int) -> str:
        return f"{cfg}_{i}.parquet"

    def start_writer():
        nonlocal writer
        return pq.ParquetWriter(out_path(idx), schema, compression="snappy")

    def close_and_upload():
        nonlocal writer, cur_size, idx
        if writer:
            writer.close()
            if not args.dry_run:
                upload_with_chunks(out_path(idx), f"{cfg}/{out_path(idx)}")
            os.remove(out_path(idx))
            idx += 1
            writer = None
            cur_size = 0

    def flush_batch():
        nonlocal writer, cur_size, buf
        if not buf:
            return
        df = pd.DataFrame(buf)

        for col in schema.names:
            if col not in df:
                df[col] = None
        df = df[schema.names]
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

        if writer is None:
            writer = start_writer()
        writer.write_table(table)
        cur_size += table.nbytes
        buf.clear()

    with gzip.open(args.input_file, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                record_json = line.rsplit("\t", 1)[-1]
                buf.append(_normalize(json.loads(record_json)))
                parsed += 1
            except json.JSONDecodeError:
                continue

            if len(buf) >= BATCH_ROWS:
                flush_batch()

            if writer and cur_size >= TARGET_BYTES:
                close_and_upload()

        flush_batch()
        close_and_upload()

    print(f"📊 Parsed {parsed:,} lines into {idx} ≤3 GB Parquet part(s)")


# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────

def main():
    top = argparse.ArgumentParser(prog="openlibrary_pipeline")
    sub = top.add_subparsers(dest="cmd", required=True)

    # fetch sub-command
    f = sub.add_parser("fetch")
    f.add_argument("--only")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--keep", action="store_true")

    # convert sub-command
    c = sub.add_parser("convert")
    c.add_argument("input_file")
    c.add_argument("--config")
    c.add_argument("--dry-run", action="store_true")

    ns = top.parse_args()
    if ns.cmd == "fetch":
        fetch_cli(ns)
    else:
        convert_cli(ns)


if __name__ == "__main__":
    main()
