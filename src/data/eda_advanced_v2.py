import pandas as pd
from pathlib import Path

def check_non_ascii(series):
    mask = series.astype(str).apply(lambda x: not x.isascii())
    return mask.mean() * 100

def check_dataset(df, name):
    print(f"\n--- {name} ---")
    print(f"Max Name Length: {df['business_name'].astype(str).str.len().max()}")
    dupes = df.duplicated(subset=['business_name', 'business_address']).sum()
    print(f"Exact Duplicates (Name+Address): {dupes} ({(dupes/len(df))*100:.2f}%)")
    print(f"Non-ASCII Characters: {check_non_ascii(df['business_name']):.2f}%")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    test_dir = base_dir / "data" / "processed" / "test"
    
    print("Loading data for Advanced Trap Detection (S3 and Test Set)...")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    check_dataset(s3, "TRAIN Source 3")
    del s3 # Free memory
    
    t1 = pd.read_parquet(test_dir / "test_source1.parquet")
    check_dataset(t1, "TEST Source 1 (Reference)")
    del t1
    
    t2 = pd.read_parquet(test_dir / "test_source2.parquet")
    check_dataset(t2, "TEST Source 2")
    del t2
    
    t3 = pd.read_parquet(test_dir / "test_source3.parquet")
    check_dataset(t3, "TEST Source 3")

if __name__ == "__main__":
    main()
