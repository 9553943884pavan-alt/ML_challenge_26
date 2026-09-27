import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
import time
import gc

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

def clean_text(df):
    name = df['business_name'].astype(str).fillna("").str.lower().str.strip()
    address = df['business_address'].astype(str).fillna("").str.lower().str.strip()
    country = df['country'].astype(str).fillna("").str.lower().str.strip()
    return name + " " + address + " " + country

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("1. Loading Full Training Data...")
    start_time = time.time()
    
    s1 = pd.read_parquet(train_dir / "train_source1.parquet")
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    gt_df = pd.read_parquet(train_dir / "train_ground_truth.parquet")
    
    s1['text'] = clean_text(s1)
    s2['text'] = clean_text(s2)
    s3['text'] = clean_text(s3)
    
    corpus_df = pd.concat([s2, s3], ignore_index=True)
    
    corpus_texts = corpus_df['text'].tolist()
    corpus_ids = corpus_df['entity_id'].values
    
    print(f"Data Loaded. S1 Queries: {len(s1):,}, Corpus Size: {len(corpus_df):,}")
    
    # Free up memory
    del s2, s3
    gc.collect()
    
    print("2. Fitting TF-IDF Vectorizer...")
    # Using float32 to save RAM on 8 Million records
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    
    print("3. Transforming S1 Queries...")
    s1_matrix = vectorizer.transform(s1['text'])
    s1_ids = s1['entity_id'].values
    
    print("4. Computing Similarities in Chunks (Threshold 0.60)...")
    batch_size = 25000
    threshold = 0.60
    
    predictions = {}
    
    num_batches = int(np.ceil(s1_matrix.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, s1_matrix.shape[0])
        
        batch_matrix = s1_matrix[start_idx:end_idx]
        
        # Sparse matrix dot product
        sims = batch_matrix.dot(corpus_matrix.T)
        
        # Extract matches efficiently
        for j in range(sims.shape[0]):
            q_id = s1_ids[start_idx + j]
            row_data = sims.getrow(j)
            
            # Find indices where similarity > threshold
            match_indices = row_data.indices[row_data.data > threshold]
            
            if len(match_indices) > 0:
                predictions[q_id] = [corpus_ids[idx] for idx in match_indices]
            else:
                predictions[q_id] = []
                
        print(f"  Processed batch {i+1}/{num_batches}...")
        
    print(f"\n5. Inference Complete in {(time.time() - start_time)/60:.2f} minutes.")
    
    print("6. Calculating Ground Truth F0.5 Metric...")
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids'])
        if matches and matches != "None":
            gt_dict[row['source1_entity_id']] = matches.split(",")
        else:
            gt_dict[row['source1_entity_id']] = []
            
    f05 = calculate_macro_f05(gt_dict, predictions)
    print(f"\n======================================")
    print(f"FULL DATASET MACRO F0.5 SCORE: {f05:.4f}")
    print(f"======================================")
    
    # Save the score out for the user
    report_file = base_dir / "reports" / "09_full_train_evaluation.txt"
    with open(report_file, "w") as f:
        f.write(f"TF-IDF Full Training Evaluation\nThreshold: {threshold}\nMacro F0.5 Score: {f05:.4f}\n")

if __name__ == "__main__":
    main()
