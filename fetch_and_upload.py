import os
import json
import requests
from datetime import datetime
from huggingface_hub import HfApi, upload_file, login
import pandas as pd
import gzip
import traceback

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
    # Read the .txt.gz file into a DataFrame
    with gzip.open(txtgz_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')  
    # Save as Parquet
    df.to_parquet(parquet_path)

def main():
    login(token=HF_TOKEN)
    api = HfApi()
    manifest = load_manifest()

    for filename, url in FILES.items():
        print(f"\n🌠 Checking {filename}")
        ol_modified = get_last_modified(url)
        last_synced = manifest.get(filename, {}).get("source_last_modified")

        if last_synced == ol_modified:
            print(f"✅ Already up to date (OL: {ol_modified})")
            continue

        print(f"🚀 New version detected (OL: {ol_modified}, HF: {last_synced})")
        try:
            # Download
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            traceback.print_exc()
            continue

        parquet_path = filename.replace('.txt.gz', '.parquet')

        try:
            print(f"📦 Converting {filename} to Parquet format...")
            convert_txtgz_to_parquet(filename, parquet_path)
        except Exception as e:
            print(f"❌ Error converting {filename} to Parquet: {e}")
            traceback.print_exc()
            # Clean up downloaded file if conversion fails
            if os.path.exists(filename):
                os.remove(filename)
            continue

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
            traceback.print_exc()
            # Clean up files if upload fails
            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
            continue


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
        traceback.print_exc()

    print("\n🌟 Sync complete. Manifest updated and uploaded.")

if __name__ == "__main__":
    main()
