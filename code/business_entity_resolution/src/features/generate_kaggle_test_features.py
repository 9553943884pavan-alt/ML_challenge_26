"""
Kaggle Test Set Feature Generation Pipeline
===========================================
This script is adapted for the blind Kaggle Test Set.
It performs:
1. Text Normalization on Source 1 (Test) and Corpus (Test Source 2 & 3)
2. HVT TF-IDF Vectorization (max_df=0.02 for memory safety)
3. Chunked Threshold Retrieval (Threshold = 0.28)
4. Joblib Parallel Feature Engineering (Jaro, Jaccard, etc.)
5. Saves test_pair_features.parquet for inference.

NOTE: This script does not generate 'label' or 'is_match' since test ground truth is hidden.
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
#  FEATURE ENGINEERING
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

def generate_features(q_id, c_id, q_row, c_row, rank, score):
    q_name = q_row['name_lex']
    c_name = c_row['name_lex']
    q_addr = q_row['addr_lex']
    c_addr = c_row['addr_lex']
    
    q_country = str(q_row.get('country', '')).lower().strip()
    c_country = str(c_row.get('country', '')).lower().strip()
    
    features = {
        'query_id': q_id,
        'candidate_id': c_id,
        'hvt_rank': rank,
        'hvt_cosine_score': score,
    }
    
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
        for f in ['name_jaro_winkler', 'name_token_sort', 'name_token_set', 'name_partial_ratio', 
                  'name_levenshtein_ratio', 'name_jaccard', 'name_3gram_jaccard', 'name_prefix_match']:
            features[f] = 0.0
            
    if q_addr and c_addr:
        features['addr_jaro_winkler'] = jellyfish.jaro_winkler_similarity(q_addr, c_addr)
        features['addr_token_sort'] = rapidfuzz.fuzz.token_sort_ratio(q_addr, c_addr) / 100.0
        features['addr_jaccard'] = get_jaccard(q_addr, c_addr)
    else:
        for f in ['addr_jaro_winkler', 'addr_token_sort', 'addr_jaccard']:
            features[f] = 0.0
            
    features['exact_name_match'] = 1 if q_name == c_name and q_name != "" else 0
    features['exact_addr_match'] = 1 if q_addr == c_addr and q_addr != "" else 0
    features['country_match'] = 1 if q_country == c_country and q_country not in ["", "nan"] else 0
    
    q_name_len = len(q_name.split())
    c_name_len = len(c_name.split())
    features['name_word_count_diff'] = abs(q_name_len - c_name_len)
    
    q_addr_len = len(q_addr.split())
    c_addr_len = len(c_addr.split())
    features['addr_word_count_diff'] = abs(q_addr_len - c_addr_len)
    
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
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    base_dir = Path(__file__).resolve().parents[2]
    
    # In a true Kaggle notebook environment, you would change these to:
    # data_dir = Path("../input/dataset_name")
    data_dir = base_dir / "data" / "processed" / "test_norm"
    
    print("Loading Kaggle Test Datasets...")
    s1 = pd.read_parquet(data_dir / "test_source1.parquet")
    s2 = pd.read_parquet(data_dir / "test_source2.parquet")
    s3 = pd.read_parquet(data_dir / "test_source3.parquet")
    
    corpus = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Queries (S1): {len(s1):,} | Corpus (S2+S3): {len(corpus):,}")
    
    print("\nApplying text normalization (this may take a few minutes)...")
    for df in [s1, corpus]:
        if 'name_lex' not in df.columns:
            df['name_lex'] = df['business_name'].apply(lambda x: advanced_clean_name(x, apply_token_sort=True))
        if 'addr_lex' not in df.columns:
            df['addr_lex'] = df['business_address'].apply(lambda x: advanced_clean_address(x, apply_token_sort=True))

    s1_ids = s1['entity_id'].values
    cor_ids = corpus['entity_id'].values
    s1_dict = s1.set_index('entity_id').to_dict('index')
    corpus_dict = corpus.set_index('entity_id').to_dict('index')

    print("\nFitting HVT TF-IDF (max_df=0.02)...")
    t0 = time.time()
    vec_hvt = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=2,
        max_df=0.02
    )
    corpus_hvt = vec_hvt.fit_transform(corpus['name_lex'] + " " + corpus['addr_lex'])
    s1_hvt = vec_hvt.transform(s1['name_lex'] + " " + s1['addr_lex'])
    print(f"Vocabulary size: {len(vec_hvt.vocabulary_):,}")
    print(f"TF-IDF fit+transform done in {time.time()-t0:.1f}s")

    print("\nRetrieving candidates with score >= 0.28 in chunks...")
    t0 = time.time()
    threshold = 0.28
    chunk_size = 1000
    
    n_queries = s1_hvt.shape[0]
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    
    preds = {}
    scores = {}
    
    for chunk_idx, start_idx in enumerate(range(0, n_queries, chunk_size)):
        end_idx = min(start_idx + chunk_size, n_queries)
        chunk_mat = s1_hvt[start_idx:end_idx]
        sim_mat = chunk_mat.dot(corpus_hvt.T)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} done", flush=True)
        
        for i in range(sim_mat.shape[0]):
            q_id = s1_ids[start_idx + i]
            row = sim_mat.getrow(i)
            
            valid_mask = row.data >= threshold
            valid_idx = row.indices[valid_mask]
            valid_scores = row.data[valid_mask]
            
            order = np.argsort(valid_scores)[::-1]
            
            preds[q_id] = [cor_ids[valid_idx[j]] for j in order]
            scores[q_id] = [valid_scores[j] for j in order]

    print(f"Retrieval done in {time.time()-t0:.1f}s")

    print("\nBuilding pair list...")
    pair_args = []
    for q_id, c_list in preds.items():
        q_row = s1_dict[q_id]
        c_score_list = scores[q_id]
        for rank, c_id in enumerate(c_list):
            if c_id not in corpus_dict: continue
            pair_args.append((q_id, c_id, q_row, corpus_dict[c_id], rank+1, c_score_list[rank]))
            
    print(f"Total pairs to process: {len(pair_args):,}")
    
    print("\nGenerating features in parallel (joblib)...")
    t0 = time.time()
    all_features = Parallel(n_jobs=-1, prefer='threads', verbose=5)(
        delayed(generate_features)(q_id, c_id, q_row, c_row, rank, score)
        for q_id, c_id, q_row, c_row, rank, score in pair_args
    )
    print(f"Feature engineering done in {time.time() - t0:.1f}s")
    
    feat_df = pd.DataFrame(all_features)
    out_path = base_dir / "data" / "processed" / "test_pair_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    print(f"\nSaved Test feature dataset to: {out_path}")
    print("Ready for LightGBM Inference!")

if __name__ == "__main__":
    main()
