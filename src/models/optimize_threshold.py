import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
import time

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
    
    print("2. Fitting TF-IDF Retriever...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_df['text'])
    s1_matrix = vectorizer.transform(s1['text'])
    
    from tqdm import tqdm
    print("3. Precomputing Similarity Scores...")
    # Precompute all similarities for the queries
    all_sims = []
    for i in tqdm(range(s1_matrix.shape[0]), desc="Computing similarities"):
        query_vec = s1_matrix[i]
        sims = query_vec.dot(corpus_matrix.T)
        all_sims.append((s1.iloc[i]['entity_id'], sims))
        
    print("4. Sweeping Thresholds to Optimize F0.5...")
    thresholds_to_test = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    best_threshold = 0
    best_f05 = 0
    results = []
    
    print(f"{'Threshold':<15} | {'Macro F0.5 Score':<15}")
    print("-" * 35)
    
    for thresh in thresholds_to_test:
        predictions = {}
        for s1_id, sims in all_sims:
            indices = sims.indices[sims.data > thresh]
            preds = [corpus_ids[idx] for idx in indices]
            predictions[s1_id] = preds
            
        f05 = calculate_macro_f05(gt_dict, predictions)
        print(f"{thresh:<15.2f} | {f05:<15.4f}")
        results.append((thresh, f05))
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            
    print("-" * 35)
    print(f"Optimal Threshold: {best_threshold:.2f} (F0.5 = {best_f05:.4f})")
    
    # Save Report
    report_file = base_dir / "reports" / "05_threshold_optimization_report.md"
    report_content = f"# Threshold Optimization Experiment\n\n**Date:** {time.strftime('%Y-%m-%d')}\n**Model:** TF-IDF Character N-Grams\n\n## Sweep Results\n| Threshold | Macro F0.5 Score |\n| :--- | :--- |\n"
    for t, f in results:
        report_content += f"| {t:.2f} | {f:.4f} |\n"
        
    report_content += f"\n**Optimal Global Threshold:** `{best_threshold:.2f}` (Yields F0.5 of `{best_f05:.4f}`)\n"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    main()
