import os
import requests
import subprocess
from datetime import datetime
from pathlib import Path

# OpenLibrary dump info
DUMP_BASE_URL = "https://ol-dumps.openlibrary.org/latest/"
DUMP_FILES = ["ol_dump_works_latest.txt.gz", "ol_dump_editions_latest.txt.gz", "ol_dump_authors_latest.txt.gz"]
LOCAL_DIR = Path("ol_dumps")
HF_REPO = "sayshara/ol_dump"


def get_remote_file_date(url):
    resp = requests.head(url)
    if 'Last-Modified' in resp.headers:
        return datetime.strptime(resp.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S %Z')
    return None

def get_local_file_date(path):
    if not path.exists():
        return None
    return datetime.utcfromtimestamp(path.stat().st_mtime)

def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def ensure_latest_dumps():
    LOCAL_DIR.mkdir(exist_ok=True)
    updated = False
    for fname in DUMP_FILES:
        url = DUMP_BASE_URL + fname
        local_path = LOCAL_DIR / fname
        remote_date = get_remote_file_date(url)
        local_date = get_local_file_date(local_path)
        if not local_path.exists() or (remote_date and (not local_date or remote_date > local_date)):
            print(f"Downloading {fname}...")
            download_file(url, local_path)
            updated = True
        else:
            print(f"{fname} is up to date.")
    return updated

def upload_to_huggingface():
    # Initialize git lfs and huggingface repo if needed
    if not (LOCAL_DIR / ".git").exists():
        subprocess.run(["git", "init"], cwd=LOCAL_DIR)
        subprocess.run(["git", "lfs", "install"], cwd=LOCAL_DIR)
        subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/{HF_REPO}"], cwd=LOCAL_DIR)
        subprocess.run(["git", "pull", "origin", "main"], cwd=LOCAL_DIR)
    # Add and commit new files
    subprocess.run(["git", "add", "*"], cwd=LOCAL_DIR)
    subprocess.run(["git", "commit", "-m", "Update ol_dumps"], cwd=LOCAL_DIR)
    # Push to HuggingFace
    subprocess.run(["git", "push", "origin", "main"], cwd=LOCAL_DIR)

def main():
    updated = ensure_latest_dumps()
    if updated:
        print("Uploading new dumps to HuggingFace...")
        upload_to_huggingface()
    else:
        print("No updates needed.")

if __name__ == "__main__":
    main()
