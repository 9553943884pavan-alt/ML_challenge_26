import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(df):
    if 'business_name' in df.columns:
        name = df['business_name'].astype(str).fillna("").str.lower().str.strip()
        address = df['business_address'].astype(str).fillna("").str.lower().str.strip()
    else:
        name = df['name'].astype(str).fillna("").str.lower().str.strip()
        address = df['short_description'].astype(str).fillna("").str.lower().str.strip()
        
    country = df.get('country', pd.Series([""]*len(df))).astype(str).fillna("").str.lower().str.strip()
    return name + " | " + address + " | " + country

def main():
    base_dir = Path("c:/Users/Pavan/Downloads/ML_Challenge_26")
    train_dir = base_dir / "data" / "processed"
    
    print("1. Loading Data...")
    s1 = pd.read_parquet(train_dir / "train" / "train_source1.parquet")
    s2 = pd.read_parquet(train_dir / "train" / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train" / "train_source3.parquet")
    
    s1['text'] = clean_text(s1)
    s2['text'] = clean_text(s2)
    s3['text'] = clean_text(s3)
    
    gt_df = pd.read_parquet(train_dir / "train" / "train_ground_truth.parquet")
    
    VAL_SAMPLE_SIZE = 10000
    s1 = s1.head(VAL_SAMPLE_SIZE)
    valid_s1_ids = set(s1['entity_id'])
    
    gt_df = gt_df[gt_df['source1_entity_id'].isin(valid_s1_ids)]
    true_match_ids = set()
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids'])
        if matches and matches != "None":
            m_list = matches.split(",")
            gt_dict[row['source1_entity_id']] = m_list
            true_match_ids.update(m_list)
        else:
            gt_dict[row['source1_entity_id']] = []
            
    s2_true = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true = s3[s3['entity_id'].isin(true_match_ids)]
    
    s2_noise = s2.sample(n=100000, random_state=42)
    s3_noise = s3.sample(n=100000, random_state=42)
    
    corpus_df = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True).drop_duplicates(subset=['entity_id'])
    
    corpus_texts = corpus_df['text'].tolist()
    corpus_ids = corpus_df['entity_id'].tolist()
    corpus_id_to_idx = {c_id: i for i, c_id in enumerate(corpus_ids)}
    
    print("2. Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    s1_matrix = vectorizer.transform(s1['text'])
    
    print("3. Sweeping Thresholds...")
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    recalls = {t: 0 for t in thresholds}
    candidates_count = {t: 0 for t in thresholds}
    
    total_gt = sum([len(matches) for matches in gt_dict.values()])
    
    from tqdm import tqdm
    chunk_size = 1000
    for start_idx in tqdm(range(0, s1_matrix.shape[0], chunk_size), desc="Evaluating Chunks"):
        end_idx = min(start_idx + chunk_size, s1_matrix.shape[0])
        chunk_sims = s1_matrix[start_idx:end_idx].dot(corpus_matrix.T).toarray()
        
        for local_i, i in enumerate(range(start_idx, end_idx)):
            q_id = s1.iloc[i]['entity_id']
            sims = chunk_sims[local_i]
            
            actual_matches = gt_dict.get(q_id, [])
            match_indices = [corpus_id_to_idx[m] for m in actual_matches if m in corpus_id_to_idx]
            
            for t in thresholds:
                valid_count = np.sum(sims > t)
                candidates_count[t] += valid_count
                
                for m_idx in match_indices:
                    if sims[m_idx] > t:
                        recalls[t] += 1
                        
    print("\n--- TF-IDF Sweep Results ---")
    for t in thresholds:
        recall = recalls[t] / total_gt if total_gt > 0 else 0
        avg_cands = candidates_count[t] / VAL_SAMPLE_SIZE
        print(f"Threshold: {t:.2f} | Recall: {recall:.4f} | Avg Candidates/Query: {avg_cands:.1f}")

if __name__ == "__main__":
    main()
