import pandas as pd
from pathlib import Path
import re

def check_non_ascii(series):
    mask = series.astype(str).apply(lambda x: not x.isascii())
    return mask.mean() * 100

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("Loading data for Advanced Trap Detection...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet")
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    
    print("\n--- 1. String Truncation Trap ---")
    print("Are they artificially cutting off Source 2 strings at a specific length?")
    print(f"S1 Name Max Length: {s1['business_name'].astype(str).str.len().max()}")
    print(f"S2 Name Max Length: {s2['business_name'].astype(str).str.len().max()}")
    
    print("\n--- 2. Internal Duplicate Trap ---")
    print("Are there businesses with the exact same name/address but different Entity IDs?")
    s1_dupes = s1.duplicated(subset=['business_name', 'business_address']).sum()
    s2_dupes = s2.duplicated(subset=['business_name', 'business_address']).sum()
    print(f"S1 Exact Duplicates: {s1_dupes} ({(s1_dupes/len(s1))*100:.2f}%)")
    print(f"S2 Exact Duplicates: {s2_dupes} ({(s2_dupes/len(s2))*100:.2f}%)")
    
    print("\n--- 3. Hidden Character / Non-ASCII Trap ---")
    print("Are there weird unicode characters ruining the strings?")
    print(f"S1 Name Non-ASCII: {check_non_ascii(s1['business_name']):.2f}%")
    print(f"S2 Name Non-ASCII: {check_non_ascii(s2['business_name']):.2f}%")

if __name__ == "__main__":
    main()
