#!/usr/bin/env python3
"""
Helper script for GitHub Actions: OpenLibrary dump sync
Commands:
  check-download   - Check remote dumps for updates & download if needed
  split --dump     - Split a specific dump into 500MB chunks
  convert --dump   - Convert chunks for a specific dump to Parquet
  upload           - Upload Parquet files and manifest to Hugging Face
"""
import os
import sys
import json
import argparse
import requests
import subprocess

# Environment-based config
env = os.environ
MANIFEST = env.get('MANIFEST', 'ol_sync_manifest.json')

DEFAULT_DUMPS = [
    {"name": "authors", "url": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz", "file": "ol_dump_authors_latest.txt.gz"},
    {"name": "editions", "url": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz", "file": "ol_dump_editions_latest.txt.gz"},
    {"name": "works", "url": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz", "file": "ol_dump_works_latest.txt.gz"}
]

try:
    DUMPS = json.loads(env.get('DUMPS', json.dumps(DEFAULT_DUMPS)))
    if not isinstance(DUMPS, list):
        raise ValueError("DUMPS must be a JSON array")
except json.JSONDecodeError:
    print("Error: DUMPS environment variable is not valid JSON")
    sys.exit(1)

HF_TOKEN = env.get('HF_TOKEN')
HF_REPO = env.get('HF_REPO')


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, 'r') as f:
            return json.load(f)
    return {}


def save_manifest(mf):
    with open(MANIFEST, 'w') as f:
        json.dump(mf, f, indent=2)


def check_download():
    mf = load_manifest()
    updated_any = False
    for d in DUMPS:
        url = d['url']
        fname = d['file']
        print(f"Checking dump: {fname} from {url}")  # Add logging
        headers = requests.head(url).headers
        last_mod = headers.get('Last-Modified')
        print(f"Last-Modified header: {last_mod}")  # Add logging
        if last_mod is None or mf.get(fname, {}).get('source_last_modified') != last_mod:
            print(f"Downloading {fname}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(fname, 'wb') as out_f:
                for chunk in r.iter_content(chunk_size=8192):
                    out_f.write(chunk)
            mf[fname] = {'source_last_modified': last_mod}
            updated_any = True
        else:
            print(f"No updates for {fname}.")  # Add logging
    save_manifest(mf)
    # GitHub Actions output
    print(f"::set-output name=updated_any::{str(updated_any).lower()}")


def split_dump(dump):
    src = f"ol_dump_{dump}_latest.txt.gz"
    prefix = f"ol_dump_{dump}_"
    cmd = [
        'bash', '-lc',
        f"gzip -dc {src} | split -b 500m - {prefix}"
    ]
    subprocess.check_call(cmd)
    print(f"Split {src} into chunks prefixed {prefix}")


def convert_dump(dump):
    prefix = f"ol_dump_{dump}_"
    files = [f for f in os.listdir('.')
             if f.startswith(prefix) and (f.endswith('.gz') or f.endswith('.txt'))]
    if not files:
        print(f"No chunks found for dump '{dump}'")
        return
    import pyarrow.csv as csv
    import pyarrow.parquet as pq
    import gzip

    for chunk in sorted(files):
        out_file = chunk.rsplit('.', 1)[0] + '.parquet'
        print(f"Converting {chunk} → {out_file}")
        if chunk.endswith('.gz'):
            with gzip.open(chunk, 'rt') as f:
                table = csv.read_csv(
                    f,
                    parse_options=csv.ParseOptions(delimiter='\t')
                )
        else:
            table = csv.read_csv(
                chunk,
                parse_options=csv.ParseOptions(delimiter='\t')
            )
        pq.write_table(table, out_file)
    print(f"Finished converting dump '{dump}'")


def upload_files():
    from huggingface_hub import HfApi
    api = HfApi()
    api.set_access_token(HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type='dataset', exist_ok=True)

    # Upload all .parquet files in cwd
    for fname in sorted(os.listdir('.')):
        if fname.endswith('.parquet'):
            print(f"Uploading {fname} to {HF_REPO}")
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=fname,
                repo_id=HF_REPO
            )
    # Upload manifest under metadata/
    print(f"Uploading manifest {MANIFEST}")
    api.upload_file(
        path_or_fileobj=MANIFEST,
        path_in_repo=f"metadata/{MANIFEST}",
        repo_id=HF_REPO
    )


def main():
    parser = argparse.ArgumentParser(description='OL dump sync utility')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('check-download', help='Check and download dumps')

    sp = sub.add_parser('split', help='Split a dump into chunks')
    sp.add_argument('--dump', required=True, help='Name of the dump to split')

    cp = sub.add_parser('convert', help='Convert dump chunks to Parquet')
    cp.add_argument('--dump', required=True, help='Name of the dump to convert')

    sub.add_parser('upload', help='Upload Parquet and manifest')

    args = parser.parse_args()
    print(f"Executing command: {args.command}")  # Add logging

    if args.command == 'check-download':
        check_download()
    elif args.command == 'split':
        split_dump(args.dump)
    elif args.command == 'convert':
        convert_dump(args.dump)
    elif args.command == 'upload':
        upload_files()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
