import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    files = ["train_source1.parquet", "train_source2.parquet", "train_source3.parquet"]
    
    for file_name in files:
        file_path = train_dir / file_name
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        print(f"--- EDA for {file_name} ---")
        df = pd.read_parquet(file_path)
        
        # Shape and Size
        print(f"Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Null Value Counts
        print("\nNull Value Counts:")
        print(df.isnull().sum())
        
        # Display a few noisy examples to inspect
        print("\nSample Data (First 3 rows):")
        print(df.head(3).to_string())
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
