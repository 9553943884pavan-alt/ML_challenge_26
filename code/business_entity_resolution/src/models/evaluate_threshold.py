import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
import time
import pickle
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

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
    TFIDF_THRESHOLD = 0.40
    CE_THRESHOLD = 0.20
    VAL_SAMPLE_SIZE = 10000 
    
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print(f"1. Loading Validation Split ({VAL_SAMPLE_SIZE} samples)...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet").head(VAL_SAMPLE_SIZE)
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
    
    s2['text'] = clean_text(s2)
    s3['text'] = clean_text(s3)
    
    s2_true = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true = s3[s3['entity_id'].isin(true_match_ids)]
    
    # Large noise sample for corpus representation
    s2_noise = s2.sample(n=100000, random_state=42)
    s3_noise = s3.sample(n=100000, random_state=42)
    
    corpus_df = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True).drop_duplicates(subset=['entity_id'])
    corpus_ids = corpus_df['entity_id'].values
    corpus_texts = corpus_df['text'].tolist()
    corpus_text_lookup = dict(zip(corpus_ids, corpus_texts))
    
    print(f"2. Stage 1: TF-IDF (Threshold > {TFIDF_THRESHOLD})...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    
    print("   -> Fitting corpus...")
    t0 = time.time()
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    print(f"   -> Done fitting corpus in {time.time()-t0:.2f}s")
    
    print("   -> Transforming s1 queries...")
    s1_matrix = vectorizer.transform(s1['text'])
    
    print(f"   -> Computing dot products to find candidates (Threshold > {TFIDF_THRESHOLD})...")
    t0 = time.time()
    
    candidate_lists = {}
    chunk_size = 1000
    for start_idx in tqdm(range(0, s1_matrix.shape[0], chunk_size), desc="TF-IDF Filtering (Chunked)"):
        end_idx = min(start_idx + chunk_size, s1_matrix.shape[0])
        chunk_sims = s1_matrix[start_idx:end_idx].dot(corpus_matrix.T).toarray()
        
        for local_i, i in enumerate(range(start_idx, end_idx)):
            q_id = s1.iloc[i]['entity_id']
            sims = chunk_sims[local_i]
            valid_idx = np.where(sims > TFIDF_THRESHOLD)[0]
            candidate_lists[q_id] = [corpus_ids[idx] for idx in valid_idx]
            
    print(f"   -> Candidate selection done in {time.time()-t0:.2f}s")

    # Evaluate TF-IDF performance
    recalled = 0
    total_gt = sum([len(matches) for matches in gt_dict.values()])
    for q_id, candidates in candidate_lists.items():
        actual_matches = gt_dict.get(q_id, [])
        for m in actual_matches:
            if m in candidates:
                recalled += 1
    recall_at_stage1 = recalled / total_gt if total_gt > 0 else 0
    total_candidates = sum([len(c) for c in candidate_lists.values()])
    avg_candidates = total_candidates / VAL_SAMPLE_SIZE
    print(f"\n--- TF-IDF Stage 1 Stats (Threshold {TFIDF_THRESHOLD}) ---")
    print(f"Recall: {recall_at_stage1:.4f}")
    print(f"Total Candidate Pairs: {total_candidates} (Avg per query: {avg_candidates:.1f})")

    print("\n3. Stage 2: Cross-Encoder Inference...")
    cache_path = base_dir / "data" / "processed" / "cross_encoder_preds_cache_10k_val_tfidf040.pkl"
    
    if cache_path.exists():
        print(f"Loading predictions from cache ({cache_path})...")
        with open(cache_path, 'rb') as f:
            query_candidate_probs = pickle.load(f)
    else:
        model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=128)
        
        # Batch GPU Operations! Construct all pairs first.
        print("   -> Constructing batches...")
        all_pairs = []
        pair_to_q_c = []
        for q_id, candidates in candidate_lists.items():
            q_text = s1[s1['entity_id'] == q_id].iloc[0]['text']
            for c in candidates:
                all_pairs.append([q_text, corpus_text_lookup[c]])
                pair_to_q_c.append((q_id, c))
                
        print(f"   -> Running Neural Net on {len(all_pairs)} pairs...")
        t0 = time.time()
        
        if all_pairs:
            logits = model.predict(all_pairs, batch_size=256, show_progress_bar=True)
            # BAAI/bge-reranker-v2-m3 outputs calibrated similarity scores directly
            probs = logits
            if isinstance(probs, np.ndarray) and probs.ndim > 1:
                probs = probs.flatten()
        else:
            probs = []
            
        print(f"   -> Inference took {time.time() - t0:.2f} seconds")
        
        query_candidate_probs = {q_id: [] for q_id in s1['entity_id']}
        for (q_id, c), prob in zip(pair_to_q_c, probs):
            query_candidate_probs[q_id].append((c, prob))
            
        print("Saving predictions to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(query_candidate_probs, f)

    print("\n4. Evaluation Results...")
    predictions = {}
    for q_id, c_probs in query_candidate_probs.items():
        valid_candidates = [c for c, p in c_probs if p >= CE_THRESHOLD]
        predictions[q_id] = valid_candidates
        
    f05 = calculate_macro_f05(gt_dict, predictions)
    print(f"\n==========================================")
    print(f"Final Validation Results (Sample: {VAL_SAMPLE_SIZE})")
    print(f"TF-IDF Threshold:        {TFIDF_THRESHOLD}")
    print(f"Cross-Encoder Threshold: {CE_THRESHOLD}")
    print(f"Macro F0.5 Score:        {f05:.4f}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
