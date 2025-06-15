import os
import json
import requests
from datetime import datetime
from huggingface_hub import HfApi, upload_file, login
import pandas as pd
import gzip
from tqdm import tqdm
import sys

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = "sayshara/ol_dump"
MANIFEST_PATH = "ol_sync_manifest.json"
FILES = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
}

def get_last_modified(url):
    r = requests.head(url, allow_redirects=True)
    return r.headers.get("Last-Modified")

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(data):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)

def convert_txtgz_to_parquet(txtgz_path, parquet_path):
    try:
        # Read the .txt.gz file in chunks and write to a Parquet file
        chunk_size = 100000  # Number of rows per chunk
        with gzip.open(txtgz_path, 'rt') as f:
            reader = pd.read_csv(f, sep='\t', index_col=0, chunksize=chunk_size)
            with tqdm(
                desc=f"Converting {os.path.basename(txtgz_path)} to Parquet",
                unit="rows",
                unit_scale=True,
                dynamic_ncols=True,
                leave=True,
                miniters=chunk_size,
                mininterval=0.5
            ) as bar:
                for i, chunk in enumerate(reader):
                    mode = 'wb' if i == 0 else 'a'  # Write mode for first chunk, append mode for others
                    chunk.to_parquet(parquet_path, compression='snappy', index=True, engine='pyarrow', mode=mode)
                    bar.update(len(chunk))
    except Exception as e:
        print(f"❌ Error converting {txtgz_path} to Parquet: {e}")
        raise

def download_file(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure HTTP errors are raised
        total = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
            miniters=1024*100,      # update every 100KB
            mininterval=0.5         # or at least every 0.5 seconds
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
            bar.refresh()
            print("✅ Download complete.")
    except requests.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        raise

def process_file(filename):
    login(token=HF_TOKEN)
    
    manifest = load_manifest()

    url = FILES.get(filename)
    if not url:
        print(f"❌ File {filename} not found in FILES.")
        return

    print(f"\n🌠 Checking {filename}")
    ol_modified = get_last_modified(url)
    last_synced = manifest.get(filename, {}).get("source_last_modified")

    if last_synced == ol_modified:
        print(f"✅ Already up to date (OL: {ol_modified})")
        return

    print(f"🚀 New version detected (OL: {ol_modified}, HF: {last_synced})")
    try:
        # Download
        download_file(url, filename)
    except Exception:
        return  # Skip if download fails

    parquet_path = filename.replace('.txt.gz', '.parquet')

    try:
        print(f"📦 Converting {filename} to Parquet format...")
        convert_txtgz_to_parquet(filename, parquet_path)
    except Exception:
        # Clean up downloaded file if conversion fails
        if os.path.exists(filename):
            os.remove(filename)
        return

    try:
        upload_file(
            path_or_fileobj=parquet_path,
            path_in_repo=parquet_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ Error uploading {parquet_path}: {e}")
        # Clean up files if upload fails
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        return

    manifest[filename] = {
        "last_synced": datetime.utcnow().isoformat() + "Z",
        "source_last_modified": ol_modified
    }

    # Clean up files after successful upload
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(parquet_path):
        os.remove(parquet_path)

    save_manifest(manifest)
    try:
        upload_file(
            path_or_fileobj=MANIFEST_PATH,
            path_in_repo=f"metadata/{MANIFEST_PATH}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ Error uploading manifest: {e}")

    print("\n🌟 Sync complete. Manifest updated and uploaded.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a filename as an argument.")
        sys.exit(1)

    process_file(sys.argv[1])