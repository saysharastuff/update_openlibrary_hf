#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import subprocess

MANIFEST = os.environ.get('MANIFEST', 'ol_sync_manifest.json')
DUMPS = json.loads(os.environ.get('DUMPS', '[]'))
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_REPO = os.environ.get('HF_REPO')

def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return {}

def save_manifest(mf):
    with open(MANIFEST, 'w') as f:
        json.dump(mf, f, indent=2)

def check_download():
    mf = load_manifest()
    updated_any = False
    for d in DUMPS:
        hdr = requests.head(d['url']).headers.get('Last-Modified')
        if mf.get(d['file'], {}).get('source_last_modified') != hdr:
            print(f"Downloading {d['file']}")
            r = requests.get(d['url'], stream=True)
            with open(d['file'], 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            mf[d['file']] = {'source_last_modified': hdr}
            updated_any = True
    save_manifest(mf)
    # Emit output for GitHub Actions
    print(f\"::set-output name=updated_any::{str(updated_any).lower()}\")
    sys.exit(0)

def split(dump):
    fname = f\"ol_dump_{dump}_latest.txt.gz\"
    prefix = f\"ol_dump_{dump}_\"
    cmd = f\"gzip -dc {fname} | split -b 500m - {prefix}\"
    subprocess.check_call(cmd, shell=True)
    print(f\"Split {fname} into chunks.\")

def convert(dump):
    chunks = [f for f in os.listdir('.') if f.startswith(f\"ol_dump_{dump}_\") and f.endswith(('.gz','.txt'))]
    for chunk in chunks:
        out = chunk.rsplit('.',1)[0] + '.parquet'
        print(f\"Converting {chunk} → {out}\")
        subprocess.check_call([
            'python3','-c',
            (
                'import pyarrow.csv as csv, pyarrow.parquet as pq; '
                'tbl=csv.read_csv(\\''+chunk+'\\', parse_options=csv.ParseOptions(delimiter=\"\\t\")); '
                'pq.write_table(tbl, \\''+out+'\\')'
            )
        ])
    print(f\"Finished Parquet for {dump}\")

def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    api.set_access_token(HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type='dataset', exist_ok=True)
    for p in [f for f in os.listdir('.') if p.endswith('.parquet')]:
        print(f\"Uploading {p}\")
        api.upload_file(path_or_fileobj=p, path_in_repo=p, repo_id=HF_REPO)
    print(\"Uploading manifest\")
    api.upload_file(path_or_fileobj=MANIFEST, path_in_repo=f\"metadata/{MANIFEST}\", repo_id=HF_REPO)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('check-download')
    sp = sub.add_parser('split'); sp.add_argument('--dump', required=True)
    cp = sub.add_parser('convert'); cp.add_argument('--dump', required=True)
    sub.add_parser('upload')
    args = parser.parse_args()
    if args.cmd == 'check-download':
        check_download()
    elif args.cmd == 'split':
        split(args.dump)
    elif args.cmd == 'convert':
        convert(args.dump)
    elif args.cmd == 'upload':
        upload()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
