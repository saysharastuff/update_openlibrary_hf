import os, subprocess, requests

HF_REPO = "datasets/sayshara/ol_dump"
TOKEN = os.environ["HF_TOKEN"]
FILES = {
    "ol_dump_authors_latest.txt.gz": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
    "ol_dump_editions_latest.txt.gz": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "ol_dump_works_latest.txt.gz": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
}

# Clone the Hugging Face dataset repo
print("Cloning Hugging Face repo...")
subprocess.run(["git", "clone", f"https://git:{TOKEN}@huggingface.co/{HF_REPO}", "repo"], check=True)
os.chdir("repo")

# Track with Git LFS once
subprocess.run(["git", "lfs", "track", "*.gz"], check=True)
with open(".gitattributes", "a") as f:
    f.write("\n*.gz filter=lfs diff=lfs merge=lfs -text\n")
subprocess.run(["git", "add", ".gitattributes"], check=True)
subprocess.run(["git", "commit", "-m", "Track .gz with LFS"], check=False)

# Authenticate with Hugging Face
subprocess.run(["huggingface-cli", "login", "--token", os.environ["HF_TOKEN"]], check=True)

# Download, add, commit, and push each file individually
for filename, url in FILES.items():
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Moving {filename} into repo...")
    subprocess.run(["mv", f"../{filename}", "."], check=True)

    subprocess.run(["git", "add", filename], check=True)
    subprocess.run(["git", "commit", "-m", f"Add {filename}"], check=True)
    subprocess.run(["git", "push"], check=True)

    print(f"Removing {filename} to conserve space...")
    os.remove(filename)
