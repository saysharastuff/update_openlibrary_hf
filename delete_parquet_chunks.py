import os
import sys
from huggingface_hub import delete_file

def cleanup_chunks(dump_base, repo_id, hf_token):
    for i in range(1000):  # Max 1000 chunks
        chunk_name = f"parquet/{dump_base}/{dump_base}_chunk_{i:03}.parquet"
        try:
            delete_file(
                path_in_repo=chunk_name,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )
            print(f"🧹 Deleted {chunk_name}")
        except Exception as e:
            if '404' in str(e):
                break  # no more chunks
            print(f"⚠️ Error deleting {chunk_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide the base dump name (e.g., ol_dump_editions_latest)")
        sys.exit(1)

    dump_base = sys.argv[1]
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set.")
        sys.exit(1)

    cleanup_chunks(dump_base, "sayshara/ol_dump", HF_TOKEN)
