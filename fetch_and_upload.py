import os
import json
import requests
from datetime import datetime
from huggingface_hub import HfApi, upload_file, login

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
        with requests.get(url, stream=True) as r:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )

        manifest[filename] = {
            "last_synced": datetime.utcnow().isoformat() + "Z",
            "source_last_modified": ol_modified
        }

        os.remove(filename)

    save_manifest(manifest)

    upload_file(
        path_or_fileobj=MANIFEST_PATH,
        path_in_repo=f"metadata/{MANIFEST_PATH}",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("\n🌟 Sync complete. Manifest updated and uploaded.")

if __name__ == "__main__":
    main()
