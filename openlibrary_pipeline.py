"""Download the latest OpenLibrary raw dumps (.txt.gz) and push
    them to the Hugging Face Hub for archival (branch backup/raw).

Changes vs. original:
• Default HF_REPO_ID is now "sayshara/openlibrary" (overridable via
  env var HF_REPO_ID).
• upload_with_chunks() now preserves the repo_path that is passed in
  (no os.path.basename()), enabling sub‑directories in future calls.
• Minor refactors/typing + strict retry helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests
from huggingface_hub import HfApi, hf_hub_download, login, upload_file
from huggingface_hub.utils import HfHubHTTPError

# --------‑ configuration ----------------------------------------------------
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
HF_REPO_ID: str = os.environ.get("HF_REPO_ID", "sayshara/openlibrary")
MANIFEST_PATH = "ol_sync_manifest.json"
CHUNK_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

FILES: Dict[str, str] = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz",
}

# --------‑ helpers ----------------------------------------------------------

def get_last_modified(url: str) -> str | None:
    """HEAD request with polite retries (3‑exponential‑backoff)."""
    for attempt in range(1, 4):
        try:
            r = requests.head(url, allow_redirects=True, timeout=10)
            r.raise_for_status()
            return r.headers.get("Last-Modified")
        except Exception as e:  # pragma: no cover
            print(f"⚠️ HEAD attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return None  # pragma: no cover


def ensure_branch_exists(branch: str = "backup/raw") -> None:
    api = HfApi()
    branches = api.list_repo_refs(repo_id=HF_REPO_ID, repo_type="dataset")
    if branch not in {b.name for b in branches.branches}:
        print(f"➕ Creating branch '{branch}' from 'main'")
        api.create_branch(repo_id=HF_REPO_ID, repo_type="dataset", branch=branch, token=HF_TOKEN)


def upload_with_chunks(
    local_path: str | Path,
    repo_path: str,
    *,
    dry_run: bool = False,
    branch: str | None = None,
) -> None:
    """Upload *local_path* to *repo_path* in the Hub, preserving sub‑dirs."""

    api = HfApi()
    local_path = str(local_path)
    file_size = os.path.getsize(local_path)

    # decide revision/branch
    revision = branch or ("backup/raw" if local_path.endswith(".txt.gz") else "main")
    if not dry_run and revision != "main":
        ensure_branch_exists(revision)

    def _single_upload(src: str, dest: str) -> None:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                upload_file(
                    path_or_fileobj=src,
                    path_in_repo=dest,  # <— *preserve* folders!
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=revision,
                    commit_message="Upload OpenLibrary data",
                    token=HF_TOKEN,
                )
                break
            except Exception as e:  # pragma: no cover
                print(f"⚠️ Upload attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)

    if file_size <= CHUNK_SIZE_BYTES:
        print(f"📤 Uploading {local_path} → {repo_path} ({file_size / 1e9:.2f} GB)")
        if not dry_run:
            _single_upload(local_path, repo_path)
    else:
        print(f"⚠️ {local_path} is >5 GB — chunking")
        with open(local_path, "rb") as f:
            chunk_idx = 0
            while chunk := f.read(CHUNK_SIZE_BYTES):
                part_name = f"{repo_path}.part{chunk_idx}" if chunk_idx else repo_path
                tmp = Path(f"__tmp_{Path(part_name).name}")
                tmp.write_bytes(chunk)
                print(f"   ↳ chunk {chunk_idx}: {tmp}")
                if not dry_run:
                    _single_upload(str(tmp), part_name)
                tmp.unlink()
                chunk_idx += 1


# ----------‑ manifest helpers ----------------------------------------------

def load_manifest() -> Dict[str, dict]:
    return json.loads(Path(MANIFEST_PATH).read_text()) if Path(MANIFEST_PATH).exists() else {}


def save_manifest(data: Dict[str, dict]):
    Path(MANIFEST_PATH).write_text(json.dumps(data, indent=2))


# ----------‑ driver ---------------------------------------------------------

def handle_download_and_upload(filename: str, url: str, manifest: Dict[str, dict], *, dry_run: bool, keep: bool):
    print(f"🌠 Checking {filename}")
    ol_modified = get_last_modified(url) if not dry_run else "<dry‑run>"
    last_synced = manifest.get(filename, {}).get("source_last_modified")

    already_up_to_date = not dry_run and last_synced == ol_modified and Path(filename).exists()
    if already_up_to_date:
        print(f"✅ {filename} already current (OL header = {ol_modified})")
        return

    # download (if necessary)
    if not Path(filename).exists():
        print(f"⬇️ Downloading {filename}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)

    # upload to Hub (root of backup/raw)
    upload_with_chunks(filename, filename, dry_run=dry_run, branch=None)

    # manifest bookkeeping
    manifest.setdefault(filename, {})
    manifest[filename].update(
        {
            "last_synced": datetime.utcnow().isoformat() + "Z",
            "source_last_modified": ol_modified,
            "converted_chunks": {filename: {"converted": True, "last_synced": datetime.utcnow().isoformat() + "Z"}},
        }
    )

    if not keep:
        Path(filename).unlink(missing_ok=True)


def main_fetch_and_upload(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Sync raw OpenLibrary dumps to the Hub")
    parser.add_argument("--only", help="Process only the named file in FILES")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true", help="Keep local copies after upload")
    args = parser.parse_args(argv)

    if not args.dry_run:
        login(token=HF_TOKEN)

    manifest = load_manifest()

    targets = [args.only] if args.only else list(FILES.keys())
    for name in targets:
        if name not in FILES:
            print(f"❌ Unknown file: {name}")
            continue
        handle_download_and_upload(name, FILES[name], manifest, dry_run=args.dry_run, keep=args.keep)

    if not args.dry_run:
        save_manifest(manifest)
        upload_with_chunks(MANIFEST_PATH, f"metadata/{MANIFEST_PATH}", dry_run=False)
        print("🌟 Raw‑dump sync complete.")


# ---------------------------------------------------------------------------
#  Execute if called directly as original script
# ---------------------------------------------------------------------------
if __name__ == "__main__" and os.path.basename(sys.argv[0]).endswith("fetch_and_upload.py"):
    main_fetch_and_upload()

###############################################################
#  Part 2 — convert_to_parquet.py                             #
###############################################################

"""Convert an OpenLibrary .txt.gz dump to snappy‑compressed Parquet
    in ~3 GB chunks and upload them into sub‑directories that map 1‑to‑1
    to Hugging Face dataset configs (authors/, works/, editions/).

Key updates:
• repo_path for upload is now f"{config}/{basename}", preserving
  the multi‑config directory structure (authors/, works/, editions/).
• Automatically infers *config* (authors|works|editions) from the
  input filename if not supplied by the user.
• Re‑uses the improved upload_with_chunks from Part 1.
"""

# ‑‑ standard libs ————————————————————————————————————————
import gzip
import json as _json
import re
from typing import List

# 3rd‑party
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Re‑use helpers from part 1
from __future__ import annotations

MAX_PARQUET_SIZE_BYTES = 3 * 1024 * 1024 * 1024  # 3 GB

# ----------‑ utilities ------------------------------------------------------

def _normalize_record(rec: dict) -> dict:
    for k, v in rec.items():
        if isinstance(v, (dict, list)):
            try:
                rec[k] = _json.dumps(v)
            except Exception:  # pragma: no cover
                rec[k] = str(v)
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            rec[k] = str(v)
    return rec


# ----------‑ uploader wrapper ----------------------------------------------

def _upload_parquet(local_path: str, config: str):
    """Upload keeping authors/ works/ editions/ folder."""
    repo_path = f"{config}/{os.path.basename(local_path)}"
    upload_with_chunks(local_path, repo_path)


# ----------‑ converter core -------------------------------------------------

def convert_to_parquet_chunks(input_file: str, *, config: str | None = None, dry_run: bool = False):
    # infer config if needed
    if not config:
        m = re.search(r"ol_dump_(\w+)_latest", os.path.basename(input_file))
        config = m.group(1) if m else os.path.splitext(os.path.basename(input_file))[0]
    config = config.lower()

    buffer: list[dict] = []
    total_lines = parsed = 0
    chunk_idx = 0
    chunk_path = lambda idx: f"{config}_{idx}.parquet"  # local temp file name

    with gzip.open(input_file, "rt", encoding="utf‑8", errors="ignore") as fp:
        for line in fp:
            total_lines += 1
            try:
                record = _normalize_record(_json.loads(line.strip().split("\t")[-1]))
                buffer.append(record)
                parsed += 1
            except _json.JSONDecodeError:
                continue

            if len(buffer) >= 100_000:  # write batch
                _flush(buffer, chunk_idx, chunk_path(chunk_idx), config, dry_run)
                buffer.clear()
                chunk_idx += 1

        # flush remainder
        if buffer:
            _flush(buffer, chunk_idx, chunk_path(chunk_idx), config, dry_run)

    print(f"📊 Lines read: {total_lines:,} / JSON ok: {parsed:,}. Chunks: {chunk_idx + 1}.")


def _flush(batch: List[dict], idx: int, local_path: str, config: str, dry_run: bool):
    if not batch:
        return
    df = pd.DataFrame(batch)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, local_path, compression="snappy")
    print(f"✅ Wrote {local_path} ({len(df):,} rows)")
    if not dry_run:
        _upload_parquet(local_path, config)
    os.remove(local_path)


# ----------‑ driver ---------------------------------------------------------

def main_convert(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Convert OL dump to Parquet & upload to the Hub")
    p.add_argument("input_file", help="Path to ol_dump_*.txt.gz")
    p.add_argument("--config", help="authors | works | editions (auto‑detected if omitted)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    login(token=HF_TOKEN)
    convert_to_parquet_chunks(args.input_file, config=args.config, dry_run=args.dry_run)


if __name__ == "__main__" and os.path.basename(sys.argv[0]).endswith("convert_to_parquet.py"):
    main_convert()
