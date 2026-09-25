import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
import time
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

def clean_text(df):
    name = df['business_name'].astype(str).fillna("").str.lower().str.strip()
    address = df['business_address'].astype(str).fillna("").str.lower().str.strip()
    country = df['country'].astype(str).fillna("").str.lower().str.strip()
    # Adding a pipe separator helps the Cross-Encoder understand column boundaries
    return name + " | " + address + " | " + country

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("1. Loading Mini-Split (1000 Queries vs 50k Corpus)...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet").head(1000)
    s1['text'] = clean_text(s1)
    
    gt_df = pd.read_parquet(train_dir / "train_ground_truth.parquet")
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1['entity_id'])]
    
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
            
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    
    s2_true = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true = s3[s3['entity_id'].isin(true_match_ids)]
    
    s2_noise = s2.sample(n=25000, random_state=42)
    s3_noise = s3.sample(n=25000, random_state=42)
    
    corpus_df = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True).drop_duplicates(subset=['entity_id'])
    corpus_df['text'] = clean_text(corpus_df)
    corpus_ids = corpus_df['entity_id'].values
    corpus_texts = corpus_df['text'].tolist()
    
    print("2. Stage 1: Running TF-IDF Retriever to get Top-10 Candidates...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    s1_matrix = vectorizer.transform(s1['text'])
    
    candidate_lists = {}
    for i in range(s1_matrix.shape[0]):
        q_id = s1.iloc[i]['entity_id']
        query_vec = s1_matrix[i]
        sims = query_vec.dot(corpus_matrix.T).toarray()[0]
        # Keep Top 10 to pass to Cross-Encoder
        top_10_idx = np.argsort(sims)[::-1][:10]
        candidate_lists[q_id] = [corpus_ids[idx] for idx in top_10_idx]
        
    print("3. Stage 2: Initializing Zero-Shot Cross-Encoder (< 8B Parameters)...")
    # ms-marco-MiniLM-L-6-v2 is an extremely fast, 22 Million parameter model trained on Bing Search queries.
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=128)
    
    print("4. Re-ranking Candidates...")
    corpus_text_lookup = dict(zip(corpus_ids, corpus_texts))
    
    predictions = {}
    
    # We will score all pairs
    start = time.time()
    for idx, row in s1.iterrows():
        q_id = row['entity_id']
        q_text = row['text']
        candidates = candidate_lists[q_id]
        
        if not candidates:
            predictions[q_id] = []
            continue
            
        # Build pairs: (Query, Candidate)
        pairs = [[q_text, corpus_text_lookup[c]] for c in candidates]
        
        # Predict logits
        logits = model.predict(pairs, show_progress_bar=False)
        
        # Convert logits to probabilities via Sigmoid
        probs = sigmoid(logits)
        
        # Threshold (Cross-Encoders usually have high confidence, so 0.8 is a good start)
        threshold = 0.80
        final_preds = [candidates[i] for i in range(len(candidates)) if probs[i] > threshold]
        
        predictions[q_id] = final_preds

    print(f"Cross-Encoder Re-ranking completed in {time.time() - start:.2f}s")
    
    f05 = calculate_macro_f05(gt_dict, predictions)
    print(f"\nFinal Macro F0.5 Score after Cross-Encoder: {f05:.4f}")

if __name__ == "__main__":
    main()
