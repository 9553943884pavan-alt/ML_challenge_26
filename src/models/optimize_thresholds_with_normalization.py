import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
import json
import time
import pickle

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05
from src.features.text_normalization import apply_advanced_preprocessing

def old_clean_text(df):
    name = df["business_name"].astype(object).fillna("").astype(str).str.lower().str.strip()
    address = df["business_address"].astype(object).fillna("").astype(str).str.lower().str.strip()
    if "country" in df.columns:
        country = df["country"].astype(object).fillna("").astype(str).str.lower().str.strip()
    else:
        country = pd.Series([""] * len(df), index=df.index)
    return name + " | " + address + " | " + country

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    cache_file = base_dir / "data" / "interim" / "optimization_cache_norm.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not cache_file.exists():
        print("Loading Data (2500 Queries)...")
        s1 = pd.read_parquet(train_dir / "train_source1.parquet").head(2500)
        
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
        corpus_ids = corpus_df['entity_id'].values
        
        print("Applying Dual Preprocessing (Lexical vs Semantic)...")
        # Lexical (For TF-IDF: Heavily normalized, sorted)
        s1['text_lexical'] = apply_advanced_preprocessing(s1, apply_token_sort=True)
        corpus_df['text_lexical'] = apply_advanced_preprocessing(corpus_df, apply_token_sort=True)
        
        # Semantic (For Cross-Encoder: Lightly normalized, keeps grammar/order)
        # Using old_clean_text which is just lower() and strip(), preserving structure.
        s1['text_semantic'] = old_clean_text(s1)
        corpus_df['text_semantic'] = old_clean_text(corpus_df)
        
        corpus_texts_lex = corpus_df['text_lexical'].tolist()
        corpus_texts_sem = corpus_df['text_semantic'].tolist()
        
        corpus_lookup_sem = dict(zip(corpus_ids, corpus_texts_sem))
        
        print("Building TF-IDF Matrix on Lexical Text...")
        vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
        corpus_matrix = vectorizer.fit_transform(corpus_texts_lex)
        s1_matrix = vectorizer.transform(s1['text_lexical'])
        
        print("Running TF-IDF Retrieval...")
        ce_candidate_lists = {}
        tfidf_scores_dict = {}
        
        for i in range(s1_matrix.shape[0]):
            q_id = s1.iloc[i]['entity_id']
            query_vec = s1_matrix[i]
            sims = query_vec.dot(corpus_matrix.T).toarray()[0]
            
            # Keep top 20 to give CE a wide net
            top_20_idx = np.argsort(sims)[::-1][:20]
            cands = [corpus_ids[idx] for idx in top_20_idx]
            ce_candidate_lists[q_id] = cands
            
            tfidf_scores_dict[q_id] = {corpus_ids[idx]: float(sims[idx]) for idx in top_20_idx}
    
        print("Running Cross-Encoder on Semantic Text...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=128)
        ce_scores_dict = {}
        
        total_queries = len(s1)
        for idx, row in s1.iterrows():
            if idx % 500 == 0:
                print(f"CE Progress: {idx}/{total_queries}")
                
            q_id = row['entity_id']
            q_text = row['text_semantic']
            candidates = ce_candidate_lists[q_id]
            
            if not candidates:
                ce_scores_dict[q_id] = {}
                continue
                
            pairs = [[q_text, corpus_lookup_sem[c]] for c in candidates]
            probs = sigmoid(model.predict(pairs, show_progress_bar=False))
            
            ce_scores_dict[q_id] = {candidates[i]: float(probs[i]) for i in range(len(candidates))}
            
        print("Caching results to disk...")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "gt_dict": gt_dict,
                "ce_candidate_lists": ce_candidate_lists,
                "tfidf_scores": tfidf_scores_dict,
                "ce_scores": ce_scores_dict
            }, f)
    else:
        print(f"Loading cached results from {cache_file}...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        gt_dict = cache["gt_dict"]
        ce_candidate_lists = cache["ce_candidate_lists"]
        tfidf_scores_dict = cache["tfidf_scores"]
        ce_scores_dict = cache["ce_scores"]

    print("Sweeping Thresholds...")
    
    tfidf_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    ce_thresholds = [0.10, 0.50, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
    
    results = []
    
    for t_thresh in tfidf_thresholds:
        for ce_thresh in ce_thresholds:
            preds = {}
            for q_id, cands in ce_candidate_lists.items():
                final_cands = []
                for c_id in cands:
                    t_score = tfidf_scores_dict[q_id].get(c_id, 0)
                    c_score = ce_scores_dict[q_id].get(c_id, 0)
                    
                    if t_score >= t_thresh and c_score >= ce_thresh:
                        final_cands.append(c_id)
                preds[q_id] = final_cands
                
            f05 = calculate_macro_f05(gt_dict, preds)
            results.append((t_thresh, ce_thresh, f05))
            
    results.sort(key=lambda x: x[2], reverse=True)
    
    report_file = base_dir / "reports" / "12_dual_pipeline_sweep_report.md"
    content = f"# Dual-Pipeline Threshold Optimization (2.5k Queries)\n\n"
    content += f"**Architecture:**\n"
    content += f"- **TF-IDF Retrieval:** Uses Heavily Normalized Lexical Text (Sorted, Stopwords Removed, Suffixes Stripped)\n"
    content += f"- **Cross-Encoder Re-Ranking:** Uses Lightly Normalized Semantic Text (Grammar and Word Order Intact)\n\n"
    content += f"## Top 10 Configurations\n\n"
    content += f"| TF-IDF Threshold | Cross-Encoder Threshold | F0.5 Score |\n"
    content += f"| :--- | :--- | :--- |\n"
    
    for t_thresh, ce_thresh, f05 in results[:10]:
        content += f"| {t_thresh:.2f} | {ce_thresh:.3f} | **{f05:.5f}** |\n"
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\nBest Configuration: TF-IDF > {results[0][0]:.2f}, CE > {results[0][1]:.3f} -> F0.5: {results[0][2]:.5f}")
    print(f"Report written to {report_file}")

if __name__ == "__main__":
    main()
