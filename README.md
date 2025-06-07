# update_ol_dump

This project contains a Python script to check for the latest OpenLibrary ol_dumps (works, editions, authors), download them if not present, and upload them to the HuggingFace Hub (sayshara/ol_dump) using the huggingface CLI with git lfs.

## Requirements
- Python 3.8+
- huggingface_hub
- git-lfs
- huggingface-cli (from `pip install huggingface_hub`)

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   sudo apt-get install git-lfs
   ```
2. Login to HuggingFace CLI:
   ```bash
   huggingface-cli login
   ```
3. Run the script:
   ```bash
   python update_ol_dump.py
   ```

The script will check for the latest ol_dump files, download them if needed, and upload them to the specified HuggingFace repository.
