import os, subprocess, requests

HF_REPO = "datasets/your-username/openlibrary-dumps"
DUMP_URL = "https://openlibrary.org/data/ol_dump_latest.txt.gz"
DUMP_FILE = "ol_dump_latest.txt.gz"

print("Downloading dump...")
r = requests.get(DUMP_URL, stream=True)
with open(DUMP_FILE, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Cloning HF repo...")
subprocess.run(["git", "clone", f"https://huggingface.co/{HF_REPO}", "repo"], check=True)
os.chdir("repo")

subprocess.run(["git", "lfs", "track", "*.gz"], check=True)
with open(".gitattributes", "a") as f: f.write("\n*.gz filter=lfs diff=lfs merge=lfs -text\n")

subprocess.run(["mv", f"../{DUMP_FILE}", "."], check=True)
subprocess.run(["git", "add", "."], check=True)
subprocess.run(["git", "commit", "-m", "Automated OpenLibrary upload"], check=True)

subprocess.run(["huggingface-cli", "login", "--token", os.environ["HF_TOKEN"]], check=True)
subprocess.run(["git", "push"], check=True)
