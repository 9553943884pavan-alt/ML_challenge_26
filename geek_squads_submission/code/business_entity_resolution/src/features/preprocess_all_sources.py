import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.text_normalization import apply_advanced_preprocessing

def old_clean_text(df):
    name = df["business_name"].astype(object).fillna("").astype(str).str.lower().str.strip()
    address = df["business_address"].astype(object).fillna("").astype(str).str.lower().str.strip()
    if "country" in df.columns:
        country = df["country"].astype(object).fillna("").astype(str).str.lower().str.strip()
    else:
        country = pd.Series([""] * len(df), index=df.index)
    return name + " | " + address + " | " + country

def process_file(input_path, output_path):
    print(f"Loading {input_path.name}...")
    df = pd.read_parquet(input_path)
    
    print(f"Applying Dual Preprocessing to {input_path.name}...")
    # Lexical for TF-IDF (Heavily normalized, sorted)
    df['text_lexical'] = apply_advanced_preprocessing(df, apply_token_sort=True)
    # Semantic for Cross-Encoder (Lightly normalized, keeps grammar)
    df['text_semantic'] = old_clean_text(df)
    
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def main():
    base_dir = Path(__file__).resolve().parents[2]
    
    # Input directories
    train_dir_in = base_dir / "data" / "processed" / "train"
    test_dir_in = base_dir / "data" / "processed" / "test"
    
    # Output directories
    train_dir_out = base_dir / "data" / "processed" / "train_norm"
    test_dir_out = base_dir / "data" / "processed" / "test_norm"
    
    # Files to process
    files_to_process = [
        (train_dir_in / "train_source1.parquet", train_dir_out / "train_source1.parquet"),
        (train_dir_in / "train_source2.parquet", train_dir_out / "train_source2.parquet"),
        (train_dir_in / "train_source3.parquet", train_dir_out / "train_source3.parquet"),
        (test_dir_in / "test_source1.parquet", test_dir_out / "test_source1.parquet"),
        (test_dir_in / "test_source2.parquet", test_dir_out / "test_source2.parquet"),
        (test_dir_in / "test_source3.parquet", test_dir_out / "test_source3.parquet")
    ]
    
    for in_path, out_path in files_to_process:
        if in_path.exists():
            process_file(in_path, out_path)
        else:
            print(f"Warning: {in_path} does not exist, skipping.")
            
    print("All files processed successfully!")

if __name__ == "__main__":
    main()
