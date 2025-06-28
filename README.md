# OpenLibrary Dataset Uploader

This repository automates the process of downloading the latest OpenLibrary dump and uploading it to a Hugging Face dataset repository using GitHub Actions.

## 🚀 Setup Instructions

1. Create a new repository on GitHub and upload these files.
2. Go to **Settings > Secrets and Variables > Actions** and add a new secret:
   - `HF_TOKEN`: your Hugging Face access token.
3. Add a new variable to the **Settings > Secrets and Variables > Actions** section:
   - `HF_REPO_ID`: the Hugging Face dataset repository where the data will be uploaded (e.g., `username/dataset_name`).
3. Trigger the workflow manually or wait for the scheduled run

## ✨ Files

- `openlibrary_pipeline.py`: Downloads the dataset from OpenLibrary and then pushes it to Hugging Face.
- `.github/workflows//process_openlibrary.yml`: GitHub Actions CI job.
