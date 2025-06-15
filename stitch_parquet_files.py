import os
import sys
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

def stitch_parquet_files(input_dir, base_name, output_dir="./"):
    files = sorted(f for f in os.listdir(input_dir) if f.startswith(base_name) and f.endswith(".parquet"))
    if not files:
        print(f"❌ No parquet files found for base: {base_name}")
        return

    print(f"🔍 Found {len(files)} parquet parts for {base_name}")
    tables = []
    for f in tqdm(files, desc=f"Stitching {base_name}"):
        table = pq.read_table(os.path.join(input_dir, f))
        tables.append(table)
    combined = pa.concat_tables(tables)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.full.parquet")
    pq.write_table(combined, output_path, compression='snappy')
    print(f"✅ Combined file written to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide the base dump name, e.g. 'ol_dump_editions_latest'")
        sys.exit(1)

    dump_base = sys.argv[1].replace(".txt.gz", "")
    stitch_parquet_files("stitched_parts", dump_base)
