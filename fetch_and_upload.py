import os
import sys
import gzip
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import login, upload_file
import traceback

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = "sayshara/ol_dump"

def convert_txtgz_to_parquet(txtgz_path, parquet_path):
    chunk_size = 100_000
    try:
        with gzip.open(txtgz_path, 'rt') as f:
            reader = pd.read_csv(f, sep='\t', index_col=0, chunksize=chunk_size)
            parquet_writer = None
            for chunk in tqdm(reader, desc=f"Converting {os.path.basename(txtgz_path)}", dynamic_ncols=True):
                table = pa.Table.from_pandas(chunk)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
                parquet_writer.write_table(table)
            if parquet_writer:
                parquet_writer.close()
    except Exception as e:
        print(f"❌ Error converting {txtgz_path} to Parquet: {e}")
        raise

def process_chunk(chunk_path):
    login(token=HF_TOKEN)

    filename = os.path.basename(chunk_path)
    parquet_name = filename.replace('.txt.gz', '.parquet')
    parquet_path = os.path.join("chunks", parquet_name)

    print(f"📦 Converting {filename} to {parquet_name}")
    convert_txtgz_to_parquet(chunk_path, parquet_path)

    print(f"📤 Uploading {parquet_name} to Hugging Face Hub")
    upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=f"parquet/editions/{parquet_name}",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    os.remove(parquet_path)

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("❌ Please provide a .txt.gz chunk file as an argument.")
            sys.exit(1)
        process_chunk(sys.argv[1])
    except Exception:
        print("Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)