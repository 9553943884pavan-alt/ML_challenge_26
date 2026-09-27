import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[2]
    test_dir = base_dir / "data" / "processed" / "test"
    
    files = ["test_source1.parquet", "test_source2.parquet", "test_source3.parquet"]
    
    for file_name in files:
        file_path = test_dir / file_name
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        print(f"--- EDA for {file_name} ---")
        df = pd.read_parquet(file_path)
        
        # Shape and Size
        print(f"Shape: {df.shape}")
        
        # Null Value Counts
        print("\nNull Value Counts:")
        print(df.isnull().sum())
        
        # Investigate the "Country" column trick
        print("\nCountry Column Distribution:")
        print(df['country'].value_counts(dropna=False))
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
