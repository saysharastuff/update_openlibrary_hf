import os
import sys
import json
import argparse
import requests
from datetime import datetime
from huggingface_hub import HfApi, upload_file, login

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = "sayshara/ol_dump"
MANIFEST_PATH = "ol_sync_manifest.json"
CHUNK_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

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

def download_file(filename, url):
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def upload_with_chunks(path, repo_path, dry_run=False):
    api = HfApi()
    file_size = os.path.getsize(path)
    if file_size <= CHUNK_SIZE_BYTES:
        print(f"📤 Uploading {path} to {repo_path} ({file_size / 1e9:.2f} GB)")
        if not dry_run:
            upload_file(
                path_or_fileobj=path,
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
    else:
        print(f"⚠️ File {path} > 5GB, uploading in chunks")
        with open(path, "rb") as f:
            chunk_idx = 0
            while True:
                chunk = f.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                chunk_filename = f"{repo_path}.part{chunk_idx}"
                with open(chunk_filename, "wb") as cf:
                    cf.write(chunk)
                print(f"📤 Uploading chunk {chunk_idx}: {chunk_filename}")
                if not dry_run:
                    upload_file(
                        path_or_fileobj=chunk_filename,
                        path_in_repo=chunk_filename,
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                os.remove(chunk_filename)
                chunk_idx += 1

def handle_download_and_upload(filename, url, manifest, dry_run, keep):
    print(f"\n🌠 Checking {filename}")
    ol_modified = get_last_modified(url) if not dry_run else "<dry-run-time>"
    last_synced = manifest.get(filename, {}).get("source_last_modified")

    if not dry_run and last_synced == ol_modified:
        print(f"✅ {filename} already up to date (OL: {ol_modified})")
        return

    print(f"🚀 New version detected (OL: {ol_modified}, HF: {last_synced})")
    print(f"📥 Would download {filename} from {url}") if dry_run else download_file(filename, url)
    if not dry_run:
        if os.path.exists(filename):
            print(f"📝 {filename} exists before upload")
        upload_with_chunks(filename, filename, dry_run=dry_run)
        if os.path.exists(filename) and not keep:
            print(f"🧹 Deleting {filename} after upload")
            os.remove(filename)

    manifest[filename] = {
        "last_synced": datetime.utcnow().isoformat() + "Z",
        "source_last_modified": ol_modified
    }

def handle_upload_only(filename, manifest, dry_run):
    print(f"\n📤 Upload-only mode for {filename}")
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found for upload")
        return
    upload_with_chunks(filename, filename, dry_run=dry_run)
    if not dry_run:
        os.remove(filename)
        manifest[filename] = {
            "last_synced": datetime.utcnow().isoformat() + "Z",
            "source_last_modified": "manual-upload"
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only process the named file")
    parser.add_argument("--upload-only", help="Only upload the named file")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without performing network ops")
    parser.add_argument("--keep", action="store_true", help="Keep downloaded files after upload")
    args = parser.parse_args()

    if not args.dry_run:
        login(token=HF_TOKEN)
    manifest = load_manifest()

    if args.only:
        name = args.only.strip()
        if name in FILES:
            handle_download_and_upload(name, FILES[name], manifest, dry_run=args.dry_run, keep=args.keep)
        else:
            print(f"❌ Unknown file name: {name}")
    elif args.upload_only:
        name = args.upload_only.strip()
        handle_upload_only(name, manifest, dry_run=args.dry_run)
    else:
        for filename, url in FILES.items():
            handle_download_and_upload(filename, url, manifest, dry_run=args.dry_run, keep=args.keep)

    if not args.dry_run:
        save_manifest(manifest)
        upload_file(
            path_or_fileobj=MANIFEST_PATH,
            path_in_repo=f"metadata/{MANIFEST_PATH}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    print("\n🌟 Sync complete." + (" (Dry run mode)" if args.dry_run else " Manifest updated and uploaded."))

if __name__ == "__main__":
    main()
