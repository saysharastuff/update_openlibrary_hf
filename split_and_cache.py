import os
import sys
import json
import requests
import gzip
from tqdm import tqdm
from datetime import datetime

HF_REPO_ID = "sayshara/ol_dump"
CACHE_DIR = "gz_cache"
CHUNK_DIR = "chunks"
MANIFEST_PATH = "ol_sync_manifest.json"
LINES_PER_CHUNK = 5_000_000

FILES = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_large_file(path, out_dir, max_lines=LINES_PER_CHUNK):
    ensure_dir(out_dir)
    base = os.path.basename(path).replace('.txt.gz', '')
    i = 0
    line_count = 0
    current_lines = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            current_lines.append(line)
            line_count += 1
            if line_count >= max_lines:
                out_path = os.path.join(out_dir, f"{base}_chunk_{i:03}.txt.gz")
                with gzip.open(out_path, 'wt') as out_f:
                    out_f.writelines(current_lines)
                print(f"✅ Wrote chunk {i:03} for {base}")
                i += 1
                current_lines = []
                line_count = 0
        if current_lines:
            out_path = os.path.join(out_dir, f"{base}_chunk_{i:03}.txt.gz")
            with gzip.open(out_path, 'wt') as out_f:
                out_f.writelines(current_lines)
            print(f"✅ Wrote final chunk {i:03} for {base}")

def get_last_modified(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return r.headers.get("Last-Modified")
    except Exception as e:
        print(f"⚠️  Failed to fetch Last-Modified for {url}: {e}")
        return None

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, 'wb') as file, tqdm(
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {os.path.basename(dest_path)}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            bar.update(size)

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(data):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a filename as an argument.")
        sys.exit(1)

    fname = sys.argv[1]
    url = FILES.get(fname)
    if not url:
        print(f"❌ Unknown file: {fname}")
        sys.exit(1)

    ensure_dir(CACHE_DIR)
    ensure_dir(CHUNK_DIR)
    manifest = load_manifest()

    fpath = os.path.join(CACHE_DIR, fname)
    remote_last_modified = get_last_modified(url)
    local_record = manifest.get(fname, {})
    cached_last_modified = local_record.get("source_last_modified")

    if not os.path.exists(fpath) or remote_last_modified != cached_last_modified:
        print(f"📥 Downloading {fname}")
        download_file(url, fpath)
        manifest[fname] = {
            "last_synced": datetime.utcnow().isoformat() + "Z",
            "source_last_modified": remote_last_modified
        }
    else:
        print(f"✅ Using cached {fname}")

    if os.path.getsize(fpath) > 5 * 1024 * 1024 * 1024:
        print(f"🔪 Splitting {fname}...")
        split_large_file(fpath, CHUNK_DIR)
    else:
        print(f"📦 No split needed. Skipping split for {fname}.")

    save_manifest(manifest)
