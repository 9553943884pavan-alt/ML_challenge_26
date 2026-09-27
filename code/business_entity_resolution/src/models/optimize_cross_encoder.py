import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

def clean_text(df):
    name = df['business_name'].astype(str).fillna("").str.lower().str.strip()
    address = df['business_address'].astype(str).fillna("").str.lower().str.strip()
    country = df['country'].astype(str).fillna("").str.lower().str.strip()
    return name + " | " + address + " | " + country

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("1. Loading Mini-Split...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet").head(2500)
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
    
    s2_noise = s2.sample(n=50000, random_state=42)
    s3_noise = s3.sample(n=50000, random_state=42)
    
    corpus_df = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True).drop_duplicates(subset=['entity_id'])
    corpus_df['text'] = clean_text(corpus_df)
    corpus_ids = corpus_df['entity_id'].values
    corpus_texts = corpus_df['text'].tolist()
    corpus_text_lookup = dict(zip(corpus_ids, corpus_texts))
    
    print("2. Stage 1: TF-IDF (Extracting Top 15 Candidates, NO THRESHOLD)...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    s1_matrix = vectorizer.transform(s1['text'])
    # Fast vectorized similarity computation for all queries at once
    all_sims = s1_matrix.dot(corpus_matrix.T).toarray()
    
    candidate_lists = {}
    from tqdm import tqdm
    for i in tqdm(range(all_sims.shape[0]), desc="TF-IDF Candidate Filtering"):
        q_id = s1.iloc[i]['entity_id']
        sims = all_sims[i]
        
        # Rule 2: Abandon Top K logic. Retrieve ALL candidates that pass the 0.60 threshold.
        valid_idx = np.where(sims > 0.60)[0]
        candidate_lists[q_id] = [corpus_ids[idx] for idx in valid_idx]
        
    print("3. Stage 2: Cross-Encoder Inference...")
    from tqdm import tqdm
    import pickle
    
    cache_path = base_dir / "data" / "processed" / "cross_encoder_preds_cache_2500.pkl"
    if cache_path.exists():
        print("Loading predictions from cache...")
        with open(cache_path, 'rb') as f:
            query_candidate_probs = pickle.load(f)
    else:
        model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=128)
        
        # Store predictions so we don't have to re-run Neural Net for every threshold
        query_candidate_probs = {}
        
        for idx, row in tqdm(s1.iterrows(), total=s1.shape[0], desc="Cross-Encoder Inference"):
            q_id = row['entity_id']
            q_text = row['text']
            candidates = candidate_lists.get(q_id, [])
            
            if not candidates:
                query_candidate_probs[q_id] = []
                continue
                
            pairs = [[q_text, corpus_text_lookup[c]] for c in candidates]
            logits = model.predict(pairs, show_progress_bar=False)
            
            # BAAI/bge-reranker-v2-m3 outputs scores that do not need sigmoid
            probs = logits
            
            # In case the model outputs 2D array [[score], [score]]
            if isinstance(probs, np.ndarray) and probs.ndim > 1:
                probs = probs.flatten()
                
            # Store as (candidate_id, probability)
            query_candidate_probs[q_id] = list(zip(candidates, probs))
            
        print("Saving predictions to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(query_candidate_probs, f)

    print("\n4. Sweeping Thresholds to Optimize Cross-Encoder...")
    thresholds_to_test = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    
    best_threshold = 0
    best_f05 = 0
    results = []
    
    print(f"{'Threshold':<15} | {'Macro F0.5 Score':<15}")
    print("-" * 35)
    
    for thresh in thresholds_to_test:
        predictions = {}
        for q_id, cand_probs in query_candidate_probs.items():
            final_preds = [c_id for c_id, prob in cand_probs if prob > thresh]
            predictions[q_id] = final_preds
            
        f05 = calculate_macro_f05(gt_dict, predictions)
        print(f"{thresh:<15.2f} | {f05:<15.4f}")
        results.append((thresh, f05))
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            
    print("-" * 35)
    print(f"Optimal Threshold: {best_threshold:.2f} (F0.5 = {best_f05:.4f})")
    
    # Save Report
    report_file = base_dir / "reports" / "06_cross_encoder_optimization_report.md"
    report_content = f"# Cross-Encoder Threshold Optimization\n\n**Date:** {time.strftime('%Y-%m-%d')}\n**Model:** ms-marco-MiniLM-L-6-v2\n\n## Sweep Results\n| Threshold | Macro F0.5 Score |\n| :--- | :--- |\n"
    for t, f in results:
        report_content += f"| {t:.2f} | {f:.4f} |\n"
        
    report_content += f"\n**Optimal Global Threshold:** `{best_threshold:.2f}` (Yields F0.5 of `{best_f05:.4f}`)\n"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    main()
