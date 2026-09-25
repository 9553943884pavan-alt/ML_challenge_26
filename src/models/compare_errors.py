import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
import json
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
    
    print("Loading data for Error Comparison...")
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
    corpus_lookup = dict(zip(corpus_ids, corpus_texts))
    
    print("Running TF-IDF (Threshold 0.60)...")
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    s1_matrix = vectorizer.transform(s1['text'])
    
    tfidf_preds = {}
    ce_candidate_lists = {}
    for i in range(s1_matrix.shape[0]):
        q_id = s1.iloc[i]['entity_id']
        query_vec = s1_matrix[i]
        sims = query_vec.dot(corpus_matrix.T).toarray()[0]
        
        tfidf_idx = np.where(sims > 0.60)[0]
        tfidf_preds[q_id] = set([corpus_ids[idx] for idx in tfidf_idx])
        
        top_15_idx = np.argsort(sims)[::-1][:15]
        ce_candidate_lists[q_id] = [corpus_ids[idx] for idx in top_15_idx]

    print("Running Cross-Encoder (Threshold 0.995)...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=128)
    ce_preds = {}
    
    for idx, row in s1.iterrows():
        q_id = row['entity_id']
        q_text = row['text']
        candidates = ce_candidate_lists[q_id]
        
        if not candidates:
            ce_preds[q_id] = set()
            continue
            
        pairs = [[q_text, corpus_lookup[c]] for c in candidates]
        probs = sigmoid(model.predict(pairs, show_progress_bar=False))
        
        final_preds = set([candidates[i] for i in range(len(candidates)) if probs[i] > 0.995])
        ce_preds[q_id] = final_preds

    print("Analyzing Discrepancies...")
    tfidf_errors = []
    ce_errors = []
    shared_errors = []
    
    for q_id in s1['entity_id']:
        true_set = set(gt_dict.get(q_id, []))
        pred_tfidf = tfidf_preds.get(q_id, set())
        pred_ce = ce_preds.get(q_id, set())
        
        tfidf_wrong = (pred_tfidf != true_set)
        ce_wrong = (pred_ce != true_set)
        
        q_text = s1[s1['entity_id'] == q_id]['text'].values[0]
        
        error_obj = {
            "query": q_text,
            "true_matches": [corpus_lookup.get(x, x) for x in true_set],
            "tfidf_preds": [corpus_lookup.get(x, x) for x in pred_tfidf],
            "ce_preds": [corpus_lookup.get(x, x) for x in pred_ce]
        }
        
        if tfidf_wrong and not ce_wrong:
            tfidf_errors.append(error_obj)
        elif ce_wrong and not tfidf_wrong:
            ce_errors.append(error_obj)
        elif tfidf_wrong and ce_wrong:
            shared_errors.append(error_obj)

    report_file = base_dir / "reports" / "07_error_comparison_report.md"
    report_content = f"# Model Error Comparison Analysis\n\n"
    report_content += f"**Total Queries Evaluated:** 1000\n"
    report_content += f"**Shared Errors (Both Failed):** {len(shared_errors)}\n"
    report_content += f"**TF-IDF Errors (Cross-Encoder Fixed These):** {len(tfidf_errors)}\n"
    report_content += f"**Cross-Encoder Errors (TF-IDF Got These Right):** {len(ce_errors)}\n\n"
    
    def write_error_section(title, errors):
        res = f"## {title} (Showing up to 3 examples)\n"
        for err in errors[:3]:
            res += f"- **Query:** `{err['query']}`\n"
            res += f"  - **True Match:** `{err['true_matches']}`\n"
            res += f"  - **TF-IDF Predicted:** `{err['tfidf_preds']}`\n"
            res += f"  - **Cross-Encoder Predicted:** `{err['ce_preds']}`\n\n"
        return res

    report_content += write_error_section("Where Cross-Encoder Saved Us (TF-IDF Failed)", tfidf_errors)
    report_content += write_error_section("Where Lexical TF-IDF Was Smarter (Cross-Encoder Failed)", ce_errors)
    report_content += write_error_section("The Hardest Cases (Both Failed)", shared_errors)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Error Comparison Report written to {report_file}")
    
    # Dump shared errors to json for deep inspection
    import json
    shared_errors_file = base_dir / "reports" / "shared_errors.json"
    with open(shared_errors_file, "w", encoding="utf-8") as f:
        json.dump(shared_errors, f, indent=4, ensure_ascii=False)
    print(f"Dumped {len(shared_errors)} shared errors to {shared_errors_file}")

if __name__ == "__main__":
    main()
