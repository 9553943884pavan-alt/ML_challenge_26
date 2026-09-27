"""
Feature Generation Script for LightGBM Re-Ranker
1. Randomly samples 5k queries from S1 to match overall distribution.
2. Runs HVT retrieval (K=20) to generate candidate pairs.
3. Scores retrieval metrics to verify distribution.
4. Generates 25+ String, Binary, and Statistical features for every pair.
5. Saves the final feature dataset to parquet.
"""

import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pathlib import Path
import sys
import time
from joblib import Parallel, delayed

try:
    import jellyfish
    import rapidfuzz
except ImportError:
    print("Please run: pip install jellyfish rapidfuzz")
    sys.exit(1)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.text_normalization import advanced_clean_name, advanced_clean_address

# ─────────────────────────────────────────────────────────────
#  RETRIEVAL FUNCTIONS
# ─────────────────────────────────────────────────────────────
def recall_at_k(preds, gt_dict, k):
    recalls = []
    for q_id, trues in gt_dict.items():
        if not trues: continue
        pred_top_k = set(preds.get(q_id, [])[:k])
        recalls.append(len(set(trues) & pred_top_k) / len(trues))
    return float(np.mean(recalls)) if recalls else 0.0

def sparse_threshold_with_scores(query_mat, corpus_mat, query_ids, corpus_ids, threshold=0.20, chunk_size=5000):
    preds = {}
    scores = {}
    n_queries = query_mat.shape[0]
    
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    for chunk_idx, start_idx in enumerate(range(0, n_queries, chunk_size)):
        end_idx = min(start_idx + chunk_size, n_queries)
        chunk_mat = query_mat[start_idx:end_idx]
        sim_mat = chunk_mat.dot(corpus_mat.T)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} done ({end_idx:,}/{n_queries:,} queries)", flush=True)
        
        for i in range(sim_mat.shape[0]):
            q_id = query_ids[start_idx + i]
            row = sim_mat.getrow(i)
            
            valid_mask = row.data >= threshold
            valid_idx = row.indices[valid_mask]
            valid_scores = row.data[valid_mask]
            
            # Sort by score descending
            order = np.argsort(valid_scores)[::-1]
            
            preds[q_id] = [corpus_ids[valid_idx[j]] for j in order]
            scores[q_id] = [valid_scores[j] for j in order]
    return preds, scores

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def char_ngram_jaccard(s1, s2, n=3):
    if not s1 or not s2 or len(s1) < n or len(s2) < n: return 0.0
    set1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
    set2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def generate_features(q_id, c_id, q_row, c_row, rank, score, is_match):
    q_name = q_row['name_lex']
    c_name = c_row['name_lex']
    q_addr = q_row['addr_lex']
    c_addr = c_row['addr_lex']
    
    # Safely handle potential NaNs in original columns
    q_country = str(q_row.get('country', '')).lower().strip()
    c_country = str(c_row.get('country', '')).lower().strip()
    
    # 1. Retrieval Metadata
    features = {
        'query_id': q_id,
        'candidate_id': c_id,
        'label': 1 if is_match else 0,
        'hvt_rank': rank,
        'hvt_cosine_score': score,
    }
    
    # 2. String Similarity Features (Name)
    if q_name and c_name:
        features['name_jaro_winkler'] = jellyfish.jaro_winkler_similarity(q_name, c_name)
        features['name_token_sort'] = rapidfuzz.fuzz.token_sort_ratio(q_name, c_name) / 100.0
        features['name_token_set'] = rapidfuzz.fuzz.token_set_ratio(q_name, c_name) / 100.0
        features['name_partial_ratio'] = rapidfuzz.fuzz.partial_ratio(q_name, c_name) / 100.0
        features['name_levenshtein_ratio'] = rapidfuzz.fuzz.ratio(q_name, c_name) / 100.0
        features['name_jaccard'] = get_jaccard(q_name, c_name)
        features['name_3gram_jaccard'] = char_ngram_jaccard(q_name, c_name, n=3)
        features['name_prefix_match'] = 1.0 if q_name[:4] == c_name[:4] and len(q_name) >= 4 else 0.0
    else:
        features['name_jaro_winkler'] = 0.0
        features['name_token_sort'] = 0.0
        features['name_token_set'] = 0.0
        features['name_partial_ratio'] = 0.0
        features['name_levenshtein_ratio'] = 0.0
        features['name_jaccard'] = 0.0
        features['name_3gram_jaccard'] = 0.0
        features['name_prefix_match'] = 0.0
        
    # 3. String Similarity Features (Address)
    if q_addr and c_addr:
        features['addr_jaro_winkler'] = jellyfish.jaro_winkler_similarity(q_addr, c_addr)
        features['addr_token_sort'] = rapidfuzz.fuzz.token_sort_ratio(q_addr, c_addr) / 100.0
        features['addr_jaccard'] = get_jaccard(q_addr, c_addr)
    else:
        features['addr_jaro_winkler'] = 0.0
        features['addr_token_sort'] = 0.0
        features['addr_jaccard'] = 0.0
        
    # 4. Exact Match Binary Features
    features['exact_name_match'] = 1 if q_name == c_name and q_name != "" else 0
    features['exact_addr_match'] = 1 if q_addr == c_addr and q_addr != "" else 0
    features['country_match'] = 1 if q_country == c_country and q_country not in ["", "nan"] else 0
    
    # 5. Statistical / Structural Features
    q_name_len = len(q_name.split())
    c_name_len = len(c_name.split())
    features['name_word_count_diff'] = abs(q_name_len - c_name_len)
    
    q_addr_len = len(q_addr.split())
    c_addr_len = len(c_addr.split())
    features['addr_word_count_diff'] = abs(q_addr_len - c_addr_len)
    
    # NEW FEATURE: Address Numbers Match
    q_digits = set(re.findall(r'\d+', q_addr))
    c_digits = set(re.findall(r'\d+', c_addr))
    
    if not q_digits and not c_digits:
        features['address_numbers_match'] = 1.0
    elif not q_digits or not c_digits:
        features['address_numbers_match'] = 0.5
    else:
        features['address_numbers_match'] = 1.0 if len(q_digits.intersection(c_digits)) > 0 else 0.0
    
    return features

# ─────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train_norm"
    gt_dir    = base_dir / "data" / "processed" / "train"

    # 1. Load Random Representative 100k Sample (1 Lakh)
    print("Loading data and creating 1 Lakh representative sample...")
    s1_full = pd.read_parquet(train_dir / "train_source1.parquet")
    s1 = s1_full.sample(n=100000, random_state=42).copy()
    
    gt_df = pd.read_parquet(gt_dir / "train_ground_truth.parquet")
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1['entity_id'])]

    true_match_ids, gt_dict = set(), {}
    for _, row in gt_df.iterrows():
        m_str = str(row['matched_entity_ids'])
        m_list = m_str.split(",") if m_str and m_str != "None" else []
        gt_dict[row['source1_entity_id']] = m_list
        true_match_ids.update(m_list)

    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")

    # Increase noise proportionately for 100k queries (1/3 of total dataset)
    s2_true  = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true  = s3[s3['entity_id'].isin(true_match_ids)]
    
    # Safely sample up to 400,000 if available
    s2_noise_len = min(400000, len(s2) - len(s2_true))
    s3_noise_len = min(400000, len(s3) - len(s3_true))
    s2_noise = s2[~s2['entity_id'].isin(true_match_ids)].sample(n=s2_noise_len, random_state=42)
    s3_noise = s3[~s3['entity_id'].isin(true_match_ids)].sample(n=s3_noise_len, random_state=42)

    corpus = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True)\
               .drop_duplicates(subset=['entity_id'])

    print(f"Queries: {len(s1):,} | Corpus: {len(corpus):,} | True targets: {len(true_match_ids):,}")

    # 2. Text Normalization
    print("Applying text normalization...")
    for df in [s1, corpus]:
        df['name_lex'] = df['business_name'].apply(lambda x: advanced_clean_name(x, apply_token_sort=True))
        df['addr_lex'] = df['business_address'].apply(lambda x: advanced_clean_address(x, apply_token_sort=True))

    s1_ids = s1['entity_id'].values
    cor_ids = corpus['entity_id'].values

    # 3. HVT Retrieval — TfidfVectorizer with max_df=0.25
    # max_df=0.25 physically drops fat-head tokens (street, road, limited, null)
    # that appear in >25% of docs, making the sparse matrix dramatically leaner.
    print("Fitting HVT TF-IDF (max_df=0.25)...")
    vec_hvt = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=2,
        max_df=0.25
    )
    corpus_hvt = vec_hvt.fit_transform(corpus['name_lex'] + " " + corpus['addr_lex'])
    s1_hvt = vec_hvt.transform(s1['name_lex'] + " " + s1['addr_lex'])
    print(f"Vocabulary size: {len(vec_hvt.vocabulary_):,}")
    print(f"Matrix shapes: Queries {s1_hvt.shape}, Corpus {corpus_hvt.shape}")

    print("Retrieving candidates with score >= 0.28 in chunks (5k query blocks)...")
    optimal_thresh = 0.28
    preds, scores = sparse_threshold_with_scores(s1_hvt, corpus_hvt, s1_ids, cor_ids, threshold=optimal_thresh, chunk_size=5000)
    
    # 4. Verify Metrics on Sample
    r_all = recall_at_k(preds, gt_dict, 100000)
    print(f"\n--- Validation on 10k Sample ---")
    print(f"Final Extraction Recall: {r_all:.5f}")
    if r_all < 0.98:
        print("WARNING: Recall dropped significantly below expected 99% on random sample!")
    else:
        print("SUCCESS: Recall held steady > 98% on random sample!")
        
    # 5. Feature Engineering
    print("\nGenerating Pairwise Features for LightGBM...")
    
    # Create fast lookup dictionaries for rows
    s1_dict = s1.set_index('entity_id').to_dict('index')
    corpus_dict = corpus.set_index('entity_id').to_dict('index')
    
    # ── Build flat list of all (q_id, c_id, rank, score, is_match) tuples ──
    print("Building pair list...")
    pair_args = []
    for q_id, c_list in preds.items():
        if q_id not in s1_dict: continue
        q_row = s1_dict[q_id]
        c_score_list = scores[q_id]
        true_matches = set(gt_dict.get(q_id, []))
        for rank, c_id in enumerate(c_list):
            if c_id not in corpus_dict: continue
            pair_args.append((q_id, c_id, q_row, corpus_dict[c_id], rank+1, c_score_list[rank], c_id in true_matches))
    
    print(f"Total pairs to process: {len(pair_args):,}")
    
    # ── Parallelize across all CPU cores using joblib ──
    print(f"Generating features in parallel (joblib)...")
    t = time.time()
    
    all_features = Parallel(n_jobs=-1, prefer='threads', verbose=5)(
        delayed(generate_features)(q_id, c_id, q_row, c_row, rank, score, is_match)
        for q_id, c_id, q_row, c_row, rank, score, is_match in pair_args
    )
    
    print(f"Feature engineering done in {time.time() - t:.1f}s")
    
    # 6. Save to Parquet
    feat_df = pd.DataFrame(all_features)
    
    # Print basic stats
    positives = feat_df['label'].sum()
    negatives = len(feat_df) - positives
    print(f"\nDataset Distribution:")
    print(f"  Total Pairs: {len(feat_df):,}")
    print(f"  True Matches (1): {positives:,}")
    print(f"  False Positives (0): {negatives:,}")
    print(f"  Imbalance Ratio: 1 positive per {negatives/positives:.1f} negatives")
    
    out_path = base_dir / "data" / "processed" / "pair_features_100k_sample.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    print(f"\nSaved feature dataset to: {out_path}")

if __name__ == "__main__":
    main()
