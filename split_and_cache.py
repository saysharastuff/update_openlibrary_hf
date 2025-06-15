import os
import requests
import gzip
from tqdm import tqdm

FILES = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
}

CACHE_DIR = "gz_cache"
CHUNK_DIR = "chunks"
LINES_PER_CHUNK = 5_000_000

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

if __name__ == "__main__":
    ensure_dir(CACHE_DIR)
    for fname, url in FILES.items():
        fpath = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"📥 Downloading {fname}")
            download_file(url, fpath)
        else:
            print(f"⚡ Using cached {fname}")
        if os.path.getsize(fpath) > 5 * 1024 * 1024 * 1024:
            print(f"🔪 Splitting {fname}...")
            split_large_file(fpath, CHUNK_DIR)
