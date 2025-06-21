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
from typing import Dict, List

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
CHUNK_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
TARGET_BYTES = 1 * 1024 ** 3               # 1 GB raw JSON per Parquet chunk

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
    HfApi().create_branch(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        branch=branch,
        token=HF_TOKEN,
        exist_ok=True,  # quietly ignore if already exists
    )


def upload_with_chunks(local_path: str | Path, repo_path: str, *, dry: bool = False, branch: str | None = None):
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
# CONVERT COMMAND (1 GB raw‑JSON threshold)
# ────────────────────────────────────────────────────────────────

def _normalize(rec: dict) -> dict:
    for k, v in rec.items():
        if isinstance(v, (dict, list)):
            rec[k] = json.dumps(v)
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            rec[k] = str(v)
    return rec


def _flush(buf: List[dict], idx: int, cfg: str, dry: bool):
    if not buf:
        return
    tmp_file = f"{cfg}_{idx}.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(buf)), tmp_file, compression="snappy")
    if not dry:
        upload_with_chunks(tmp_file, f"{cfg}/{tmp_file}")
    os.remove(tmp_file)


def convert_cli(args: argparse.Namespace):
    login(token=HF_TOKEN)
    cfg = (args.config or re.search(r"ol_dump_(\w+)_latest", args.input_file).group(1)).lower()

    buf: List[dict] = []
    buf_bytes = 0
    idx = 0
    parsed = 0

    with gzip.open(args.input_file, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                record_json = line.rsplit("\t", 1)[-1]
                buf.append(_normalize(json.loads(record_json)))
                buf_bytes += len(record_json.encode("utf-8"))
                parsed += 1
            except json.JSONDecodeError:
                continue

            if buf_bytes >= TARGET_BYTES:
                _flush(buf, idx, cfg, args.dry_run)
                buf.clear()
                buf_bytes = 0
                idx += 1

        if buf:
            _flush(buf, idx, cfg, args.dry_run)

    print(f"📊 Parsed {parsed:,} lines into {idx + 1} chunk(s) (≤1 GB raw each)")

# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────

def main():
    top = argparse.ArgumentParser(prog="openlibrary_pipeline")
    sub = top.add_subparsers(dest="cmd", required=True)

    # fetch sub‑command
    f = sub.add_parser("fetch", help="Download & archive raw dumps")
    f.add_argument("--only", help="Process only the named dump file in FILES")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--keep", action="store_true", help="Keep local copy after upload")

    # convert sub‑command
    c = sub.add_parser("convert", help="Convert one dump to Parquet & upload")
    c.add_argument("input_file", help="Path to ol_dump_*.txt.gz")
    c.add_argument("--config", help="authors | editions | works (auto‑detected)")
    c.add_argument("--dry-run", action="store_true")

    ns = top.parse_args()
    if ns.cmd == "fetch":
        fetch_cli(ns)
    else:
        convert_cli(ns)


if __name__ == "__main__":
    main()
