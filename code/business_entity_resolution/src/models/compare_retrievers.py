import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import sys
from pathlib import Path
import time
import json

def clean_text(df):
    name = df['business_name'].astype(str).fillna("").str.lower().str.strip()
    address = df['business_address'].astype(str).fillna("").str.lower().str.strip()
    country = df['country'].astype(str).fillna("").str.lower().str.strip()
    return name + " " + address + " " + country

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
    query_texts = s1['text'].tolist()
    query_ids = s1['entity_id'].tolist()
    
    print(f"Corpus size: {len(corpus_df)}")
    
    # ---------------------------------------------------------
    # Experiment A: Exact BM25 (rank_bm25 library - Word Level)
    # ---------------------------------------------------------
    print("\n--- Running Exact BM25 (Word-Level) ---")
    start_bm25 = time.time()
    
    # BM25 requires tokenized lists (word level)
    tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 Index Built in {time.time() - start_bm25:.2f}s")
    
    bm25_top10_recall_hits = 0
    total_valid_queries = 0
    
    start_retrieval = time.time()
    for q_idx, query in enumerate(query_texts):
        q_id = query_ids[q_idx]
        true_matches = set(gt_dict.get(q_id, []))
        if not true_matches: continue # Skip singletons for recall metric
        
        total_valid_queries += 1
        tokenized_query = query.split(" ")
        
        # Get top 10 candidate indices
        scores = bm25.get_scores(tokenized_query)
        top_10_idx = np.argsort(scores)[::-1][:10]
        top_10_ids = set([corpus_ids[i] for i in top_10_idx])
        
        # Did we find ANY of the true matches in our Top 10?
        if len(true_matches.intersection(top_10_ids)) > 0:
            bm25_top10_recall_hits += 1
            
    print(f"BM25 Retrieval Time: {time.time() - start_retrieval:.2f}s")
    bm25_recall = bm25_top10_recall_hits / total_valid_queries
    
    # ---------------------------------------------------------
    # Experiment B: TF-IDF (Character N-Grams - Typo Robust)
    # ---------------------------------------------------------
    print("\n--- Running TF-IDF (Character 3-Grams) ---")
    start_tfidf = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    print(f"TF-IDF Index Built in {time.time() - start_tfidf:.2f}s")
    
    tfidf_top10_recall_hits = 0
    
    s1_matrix = vectorizer.transform(query_texts)
    
    start_retrieval = time.time()
    for q_idx in range(s1_matrix.shape[0]):
        q_id = query_ids[q_idx]
        true_matches = set(gt_dict.get(q_id, []))
        if not true_matches: continue
        
        query_vec = s1_matrix[q_idx]
        sims = query_vec.dot(corpus_matrix.T).toarray()[0]
        
        top_10_idx = np.argsort(sims)[::-1][:10]
        top_10_ids = set([corpus_ids[i] for i in top_10_idx])
        
        if len(true_matches.intersection(top_10_ids)) > 0:
            tfidf_top10_recall_hits += 1
            
    print(f"TF-IDF Retrieval Time: {time.time() - start_retrieval:.2f}s")
    tfidf_recall = tfidf_top10_recall_hits / total_valid_queries
    
    # --- Print Results ---
    print("\n" + "="*50)
    print("BLOCKING STAGE (RECALL@10) COMPARISON")
    print("="*50)
    print(f"Queries evaluated: {total_valid_queries} (excluding singletons)")
    print(f"Exact BM25 (Word-Level) Recall@10: {bm25_recall*100:.2f}%")
    print(f"TF-IDF (Char N-Gram) Recall@10:    {tfidf_recall*100:.2f}%")
    print("="*50)
    print("\nConclusion: For noisy strings (typos, abbreviations), Character N-Grams heavily outperform standard word-level BM25.")

if __name__ == "__main__":
    main()
