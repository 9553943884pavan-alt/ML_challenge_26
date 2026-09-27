import pandas as pd
import os
import glob
from pathlib import Path
import time

def process_tsv_to_parquet(input_file: str, output_dir: str):
    """
    Reads a TSV file, optimizes its data types according to Rule 4, 
    and saves it as a highly compressed Parquet file.
    """
    start_time = time.time()
    file_name = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{file_name}.parquet")
    
    print(f"Processing: {input_file}...")
    
    # Read the TSV file
    # We specify string for all columns initially to avoid mixed type warnings
    df = pd.read_csv(input_file, sep="\t", dtype=str)
    
    # Apply Rule 4: Downcast low-cardinality strings to categoricals
    if 'country' in df.columns:
        df['country'] = df['country'].astype('category')
        
    # Write to Parquet using PyArrow compression (Snappy by default)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    
    elapsed = time.time() - start_time
    # Print memory reduction stats
    print(f"Saved to: {output_file}")
    print(f"Time taken: {elapsed:.2f} seconds.")
    print("-" * 50)

def main():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # Ensure processed directories exist
    os.makedirs(processed_dir / "train", exist_ok=True)
    os.makedirs(processed_dir / "test", exist_ok=True)
    
    # Find all TSV files in the raw directory (recursively)
    tsv_files = glob.glob(str(raw_dir / "**" / "*.tsv"), recursive=True)
    
    if not tsv_files:
        print(f"No TSV files found in {raw_dir}")
        return
        
    for tsv_file in tsv_files:
        # Determine if it belongs in train or test processed folder
        if "train" in str(Path(tsv_file).parent):
            out_path = processed_dir / "train"
        elif "test" in str(Path(tsv_file).parent):
            out_path = processed_dir / "test"
        else:
            out_path = processed_dir
            
        process_tsv_to_parquet(tsv_file, out_path)

if __name__ == "__main__":
    main()
