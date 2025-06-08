import os
import requests
from huggingface_hub import HfApi, login

HF_REPO_ID = "sayshara/ol_dump"
HF_TOKEN = os.environ["HF_TOKEN"]

FILES = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
}

def get_last_modified(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return r.headers.get("Last-Modified")
    except requests.RequestException as e:
        print(f"⚠️ Failed to get HEAD from {url}: {e}")
        return None


# Authenticate
login(token=HF_TOKEN)
api = HfApi()

print(f"🔗 Syncing files from OpenLibrary to HuggingFace dataset: {HF_REPO_ID}")

for filename, url in FILES.items():
    print(f"🌠 Checking {filename}...")

    # Get timestamps
    ol_timestamp = get_last_modified(url)
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
    hf_timestamp = get_last_modified(hf_url)

    print(f"  🕒 OpenLibrary: {ol_timestamp}")
    print(f"  🕒 HuggingFace : {hf_timestamp}")

    # Only upload if different or missing
    if hf_timestamp is None or ol_timestamp > hf_timestamp:
        print(f"🚀 New version found! Downloading and uploading {filename}...")
        """
        with requests.get(url, stream=True) as r:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        os.remove(filename)
        """
        
    else:
        print(f"✅ {filename} is already up to date.")

print("✨ Sync complete!")
