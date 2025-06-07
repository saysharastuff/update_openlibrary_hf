# 📦 Dataset Uploader Template

This repository automates the process of downloading the latest OpenLibrary dump and uploading it to a Hugging Face dataset repository using GitHub Actions.

## 🚀 Setup Instructions

1. Create a new repository on GitHub and upload these files.
2. Go to **Settings > Secrets > Actions** and add a new secret:
   - `HF_TOKEN`: your Hugging Face access token.
3. Trigger the workflow manually or wait for the scheduled run (every Sunday at 5am UTC).

## ✨ Files

- `fetch_and_upload.py`: Downloads and pushes the dataset.
- `.github/workflows/upload.yml`: GitHub Actions CI job.
