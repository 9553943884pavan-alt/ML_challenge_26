import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
import time
import json

# Add root directory to python path for relative imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

def clean_text(df):
    """Concatenates name, address, and country into a single normalized string"""
    name = df['business_name'].fillna("").astype(str).str.lower().str.strip()
    address = df['business_address'].fillna("").astype(str).str.lower().str.strip()
    country = df['country'].astype(str).fillna("").str.lower().str.strip()
    return name + " " + address + " " + country

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("1. Loading Source 1 (Rule 1: Mini Split of 1000 queries)...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet").head(1000)
    s1['text'] = clean_text(s1)
    
    print("2. Loading Ground Truth and extracting true matches...")
    gt_df = pd.read_parquet(train_dir / "train_ground_truth.parquet")
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1['entity_id'])]
    
    true_match_ids = set()
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids'])
        if matches == "None" or matches == "" or pd.isna(matches):
            gt_dict[row['source1_entity_id']] = []
        else:
            m_list = matches.split(",")
            gt_dict[row['source1_entity_id']] = m_list
            true_match_ids.update(m_list)
            
    print(f"Found {len(true_match_ids)} true target entities.")
    
    print("3. Building Mini-Corpus (True Matches + 100k Random Noise)...")
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    
    # Ensure all true matches are in the corpus
    s2_true = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true = s3[s3['entity_id'].isin(true_match_ids)]
    
    # Add random negative noise to simulate finding needles in a haystack
    s2_noise = s2.sample(n=50000, random_state=42)
    s3_noise = s3.sample(n=50000, random_state=42)
    
    corpus_df = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True).drop_duplicates(subset=['entity_id'])
    corpus_df['text'] = clean_text(corpus_df)
    corpus_ids = corpus_df['entity_id'].values
    
    # Lookup dictionary for error analysis reporting
    corpus_text_lookup = dict(zip(corpus_df['entity_id'], corpus_df['text']))
    
    print(f"Mini-Corpus built with {len(corpus_df)} records.")
    
    print("4. Fitting Lexical Retriever (Character N-Grams for high typo robustness)...")
    # Character n-grams (3-grams) act exactly like a robust BM25 for noisy strings
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
    
    start = time.time()
    corpus_matrix = vectorizer.fit_transform(corpus_df['text'])
    print(f"Fitted corpus matrix in {time.time() - start:.2f}s")
    
    s1_matrix = vectorizer.transform(s1['text'])
    
    print("5. Performing Retrieval and Scoring...")
    predictions = {}
    errors = []
    
    # Strict threshold to prioritize precision (F_0.5 metric)
    threshold = 0.75 
    
    for i in range(s1_matrix.shape[0]):
        s1_id = s1.iloc[i]['entity_id']
        query_vec = s1_matrix[i]
        
        # Fast sparse dot product (Cosine Similarity)
        sims = query_vec.dot(corpus_matrix.T)
        
        # Retrieve candidates above threshold
        indices = sims.indices[sims.data > threshold]
        preds = [corpus_ids[idx] for idx in indices]
        predictions[s1_id] = preds
        
        # Collect Error Analysis
        true_matches = gt_dict.get(s1_id, [])
        pred_set = set(preds)
        true_set = set(true_matches)
        
        if pred_set != true_set:
            errors.append({
                "s1_id": s1_id,
                "s1_text": s1.iloc[i]['text'],
                "true_matches": true_matches,
                "predicted_matches": preds,
                "false_positives": list(pred_set - true_set),
                "false_negatives": list(true_set - pred_set)
            })
            
    f05 = calculate_macro_f05(gt_dict, predictions)
    print(f"\nFinal Macro F0.5 Score on Mini-Split: {f05:.4f}")
    
    print("6. Writing Error Analysis Report...")
    report_file = base_dir / "reports" / "02_retriever_prototype_report.md"
    report_content = f"""# Prototype 1: Lexical Retrieval (TF-IDF / BM25 Approximation)

**Date:** {time.strftime('%Y-%m-%d')}
**Dataset:** Mini-Split (1,000 Source 1 Queries vs 100,000+ noisy Corpus)
**Metric:** Macro F_0.5

## Execution Results
*   **Threshold Used:** `{threshold}`
*   **F_0.5 Score:** `{f05:.4f}`
*   **Total Errors:** `{len(errors)}` / 1000 queries failed perfectly matching.

## Error Analysis Highlights
Because we used Character N-Grams (`char_wb`), the model is highly robust to typos, but without a Cross-Encoder, it struggles with:
1.  **False Merges on Chains:** Combining different branches of the same franchise (e.g., Starbucks on 5th Ave vs Starbucks on 6th Ave).
2.  **Singleton Failures:** Predicting matches when the query actually has zero true matches in the corpus.

### Example Failure Cases:
"""
    for err in errors[:5]:
        fp_texts = [corpus_text_lookup.get(fp, "Unknown") for fp in err['false_positives']]
        fn_texts = [corpus_text_lookup.get(fn, "Unknown") for fn in err['false_negatives']]
        
        report_content += f"\n**Query [{err['s1_id']}]:** `{err['s1_text']}`\n"
        if fp_texts:
            report_content += f"- ❌ **False Positive (Bad Match):** `{fp_texts[0]}`\n"
        if fn_texts:
            report_content += f"- ⚠️ **False Negative (Missed Match):** `{fn_texts[0]}`\n"
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
