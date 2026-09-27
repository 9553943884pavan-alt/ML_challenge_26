"""
Kaggle End-to-End Test Pipeline (Inference + Submission)
========================================================
Runs inference on the Test Set and generates matching_results.tsv 
and candidate_pairs.tsv exactly as required by the README.

Features a 3-Way Accelerated Retrieval Engine (GPU CuPy -> C++ sparse_dot_topn -> Safe CPU)
to process 1.7 Million queries against 5 Million corpus entries under the 12-hour limit.
Chunk size: 4000, Top-K: 30.
"""

from pathlib import Path
import sys
import time
import os
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import joblib
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import jellyfish
    import rapidfuzz
    import lightgbm as lgb
except ImportError:
    print("Please install required packages: pip install jellyfish rapidfuzz lightgbm")
    sys.exit(1)

import unicodedata
import re

# ─────────────────────────────────────────────────────────────
#  TEXT NORMALIZATION ENGINE (Embedded for Kaggle Standalone)
# ─────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = {
    r'\binc\b\.?': '', r'\bllc\b\.?': '', r'\bl\.l\.c\.\b': '', r'\bltd\b\.?': '', r'\blimited\b': '',
    r'\bcorp\b\.?': '', r'\bcorporation\b': '', r'\bco\b\.?': '', r'\bcompany\b': '', r'\bplc\b\.?': '',
    r'\bgmbh\b': '', r'\bsa\b': '', r'\bnv\b': '', r'\bbv\b': '', r'\bsrl\b': '', r'\bspa\b': ''
}
ADDRESS_ABBREV = {
    r'\bst\b\.?': 'street', r'\bave\b\.?': 'avenue', r'\brd\b\.?': 'road', r'\bblvd\b\.?': 'boulevard',
    r'\bdr\b\.?': 'drive', r'\bln\b\.?': 'lane', r'\bct\b\.?': 'court', r'\bpl\b\.?': 'place',
    r'\bsq\b\.?': 'square', r'\bste\b\.?': 'suite', r'\bapt\b\.?': 'apartment', r'\bpkwy\b\.?': 'parkway',
    r'\bhwy\b\.?': 'highway',
}
BUSINESS_ABBREV = {
    r'\bmfg\b\.?': 'manufacturing', r'\bmgmt\b\.?': 'management', r'\bintl\b\.?': 'international',
    r'\bgrp\b\.?': 'group', r'\bassoc\b\.?': 'associates', r'\btech\b\.?': 'technology',
    r'\bbros\b\.?': 'brothers', r'\bctr\b\.?': 'center',
}
STEMMING_MAP = {
    r'\bservices\b': 'service', r'\bpartners\b': 'partner', r'\bholdings\b': 'holding',
    r'\bstores\b': 'store', r'\bcenters\b': 'center',
}
STOPWORDS = {'the', 'and', '&', 'of', 'for', 'in', 'at', 'on', 'a', 'an'}

def unicode_normalize(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKC', text)

def case_fold_and_whitespace(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text.lower()).strip()

def normalize_punctuation(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text)).strip()

def remove_legal_suffixes(text):
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def expand_business_abbreviations(text):
    for pattern, replacement in BUSINESS_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def apply_stemming(text):
    for pattern, replacement in STEMMING_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_numbers(text):
    return re.sub(r'\b0+(\d+)\b', r'\1', text)

def remove_business_noise(text):
    text = re.sub(r'\(?id:\s*\d+\)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:po box|p\.o\. box|pmb|suite|ste|floor|fl|room|rm|apt)\s*#?\w+\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def expand_address_abbreviations(text):
    for pattern, replacement in ADDRESS_ABBREV.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def remove_stopwords(text):
    return " ".join([t for t in text.split() if t not in STOPWORDS])

def deduplicate_tokens(text):
    seen = set()
    dedup = []
    for t in text.split():
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return " ".join(dedup)

def sort_tokens(text):
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def advanced_clean_name(text, apply_token_sort=False):
    if pd.isna(text) or text is None: return ""
    text = str(text)
    text = unicode_normalize(text)
    text = case_fold_and_whitespace(text)
    text = remove_business_noise(text)
    text = remove_legal_suffixes(text)
    text = expand_business_abbreviations(text)
    text = apply_stemming(text)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = normalize_numbers(text)
    text = remove_stopwords(text)
    text = deduplicate_tokens(text)
    if apply_token_sort: text = sort_tokens(text)
    return text

def advanced_clean_address(text, apply_token_sort=False):
    if pd.isna(text) or text is None: return ""
    text = str(text)
    text = unicode_normalize(text)
    text = case_fold_and_whitespace(text)
    text = remove_business_noise(text)
    text = expand_address_abbreviations(text)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = normalize_numbers(text)
    text = deduplicate_tokens(text)
    if apply_token_sort: text = sort_tokens(text)
    return text

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def get_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def generate_features(q_id, c_id, q_name, c_name, q_addr, c_addr, q_country, c_country, rank, score):
    q_name = q_name if q_name else ""
    c_name = c_name if c_name else ""
    q_addr = q_addr if q_addr else ""
    c_addr = c_addr if c_addr else ""
    
    q_country = str(q_country).lower().strip() if q_country else ""
    c_country = str(c_country).lower().strip() if c_country else ""
    
    features = {
        'query_id': q_id,
        'candidate_id': c_id,
        'hvt_rank': rank,
        'hvt_cosine_score': score,
    }
        
    if q_name and c_name:
        features['name_jaro_winkler'] = jellyfish.jaro_winkler_similarity(q_name, c_name)
        features['name_token_sort'] = rapidfuzz.fuzz.token_sort_ratio(q_name, c_name) / 100.0
        features['name_jaccard'] = get_jaccard(q_name, c_name)
    else:
        for f in ['name_jaro_winkler', 'name_token_sort', 'name_jaccard']:
            features[f] = 0.0
            
    if q_addr and c_addr:
        features['addr_jaro_winkler'] = jellyfish.jaro_winkler_similarity(q_addr, c_addr)
        features['addr_token_sort'] = rapidfuzz.fuzz.token_sort_ratio(q_addr, c_addr) / 100.0
        features['addr_jaccard'] = get_jaccard(q_addr, c_addr)
    else:
        for f in ['addr_jaro_winkler', 'addr_token_sort', 'addr_jaccard']:
            features[f] = 0.0
            
    features['exact_name_match'] = 1 if q_name == c_name and q_name != "" else 0
    features['country_match'] = 1 if q_country == c_country and q_country not in ["", "nan"] else 0
    
    features['name_word_count_diff'] = abs(len(q_name.split()) - len(c_name.split()))
    features['addr_word_count_diff'] = abs(len(q_addr.split()) - len(c_addr.split()))
    
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
    # ── 1. PATH CONFIGURATION ──
    if os.path.exists("/kaggle/input"):
        logger.info("🌍 Detected Kaggle Environment!")
        base_dir = Path("/kaggle/working")
        # Change this path to where your test TSVs are stored on Kaggle
        test_dir = Path("/kaggle/input/datasets/pavan9938/test")
        
        # User defined paths
        vec_path = Path("/kaggle/input/datasets/pavan9938/vector/vec_hvt.joblib") 
        model_path = Path("/kaggle/input/datasets/pavan9938/train-norm/lgb_model.txt")
        thresh_path = Path("/kaggle/input/datasets/pavan9938/train-norm/lgbm_threshold.json")
        
        out_dir = base_dir / "output"
        processed_dir = base_dir 
    else:
        logger.info("💻 Detected Local Environment!")
        base_dir = Path(__file__).resolve().parents[2]
        test_dir = base_dir / "data" / "raw" / "test"
        model_path = base_dir / "data" / "processed" / "kaggle_pipeline" / "lgbm_kaggle.joblib"
        out_dir = base_dir / "output"
        processed_dir = base_dir / "data" / "processed" / "kaggle_pipeline_test"
        
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # ── 2. LOAD TEST DATA ──
    t_load = time.time()
    norm_cache_dir = processed_dir / "normalized_cache_test"
    s1_cache_path = norm_cache_dir / "s1_test_norm.parquet"
    corpus_cache_path = norm_cache_dir / "corpus_test_norm.parquet"
    
    if s1_cache_path.exists() and corpus_cache_path.exists():
        logger.info("✅ Found pre-normalized data in Kaggle working cache! Loading directly...")
        s1_test = pd.read_parquet(s1_cache_path)
        corpus_test = pd.read_parquet(corpus_cache_path)
        logger.info(f"    -> Data loading completed in {time.time() - t_load:.2f}s")
    else:
        logger.info(f"Loading Test Data from {test_dir}...")
        
        s1_test = pd.read_parquet(test_dir / "test_source1.parquet")
        s2_test = pd.read_parquet(test_dir / "test_source2.parquet")
        s3_test = pd.read_parquet(test_dir / "test_source3.parquet")
        
        corpus_test = pd.concat([s2_test, s3_test]).drop_duplicates(subset=['entity_id'])
        
        # EXPLICIT RAM CLEANUP: Free the individual dataframes to prevent RAM doubling
        del s2_test, s3_test
        import gc; gc.collect()
        
        logger.info(f"    -> Data loading completed in {time.time() - t_load:.2f}s")
        
        # ── 3. NORMALIZE TEXT ON THE FLY ──
        t_check = time.time()
        logger.info("\n" + "="*50)
        logger.info(" [PHASE 2/5] TEXT NORMALIZATION")
        logger.info("="*50)
        
        from tqdm import tqdm
        tqdm.pandas(mininterval=10) # Print progress every 10s to keep Kaggle logs clean
        
        for df_name, df in [("Queries (S1)", s1_test), ("Corpus (S2+S3)", corpus_test)]:
            if 'name_lex' not in df.columns:
                logger.info(f"  -> Normalizing Business Names for {df_name} ({len(df):,} rows)...")
                df['name_lex'] = df['business_name'].progress_apply(lambda x: advanced_clean_name(str(x), apply_token_sort=True))
            else:
                logger.info(f"  -> Found pre-normalized 'name_lex' for {df_name}, skipping computation.")
                
            if 'addr_lex' not in df.columns:
                logger.info(f"  -> Normalizing Addresses for {df_name} ({len(df):,} rows)...")
                df['addr_lex'] = df['business_address'].progress_apply(lambda x: advanced_clean_address(str(x), apply_token_sort=True))
            else:
                logger.info(f"  -> Found pre-normalized 'addr_lex' for {df_name}, skipping computation.")
            
        os.makedirs(norm_cache_dir, exist_ok=True)
        s1_test.to_parquet(s1_cache_path, index=False)
        corpus_test.to_parquet(corpus_cache_path, index=False)
        logger.info(f"    -> Saved normalized data to {norm_cache_dir} for faster re-runs!")
        
    logger.info("    -> Data extraction ready for vectorization")

    # ── 4. TF-IDF VECTORIZATION ──
    import joblib
    if not os.path.exists("/kaggle/input"):
        vec_path = model_path.parent / "vec_hvt.joblib"
        
    if not vec_path.exists():
        logger.error(f"Vectorizer not found at {vec_path}! Please train and save it first.")
        sys.exit(1)
        
    logger.info(f"Loading pre-trained Vectorizer from {vec_path}...")
    vec_hvt = joblib.load(vec_path)
    
    logger.info("    -> Extracting C-Arrays and Freeing Pandas DataFrames to save 10GB RAM...")
    s1_ids = s1_test['entity_id'].values.copy()
    s1_names = s1_test['name_lex'].astype(object).fillna('').values
    s1_addrs = s1_test['addr_lex'].astype(object).fillna('').values
    s1_countries = s1_test['country'].astype(object).fillna('').values if 'country' in s1_test.columns else np.array([""] * len(s1_ids))
    
    cor_ids = corpus_test['entity_id'].values.copy()
    cor_names = corpus_test['name_lex'].astype(object).fillna('').values
    cor_addrs = corpus_test['addr_lex'].astype(object).fillna('').values
    cor_countries = corpus_test['country'].astype(object).fillna('').values if 'country' in corpus_test.columns else np.array([""] * len(cor_ids))
    
    del s1_test, corpus_test
    import gc; gc.collect()
    
    t_start = time.time()
    logger.info("[1] Vectorizing Test Corpus...")
    corpus_text = cor_names + " " + cor_addrs
    corpus_hvt = vec_hvt.transform(corpus_text)
    logger.info(f"    -> Corpus vectorization took {time.time() - t_start:.2f}s")
    
    t_start = time.time()
    logger.info("[2] Vectorizing Test Queries...")
    s1_text = s1_names + " " + s1_addrs
    s1_hvt = vec_hvt.transform(s1_text)
    logger.info(f"    -> Query vectorization took {time.time() - t_start:.2f}s")
    
    # ── 5. CHUNKED RETRIEVAL AND FEATURE GENERATION ──
    # Arrays already extracted above
    n_queries = s1_hvt.shape[0]
    chunk_size = 4000 
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    threshold = 0.15
    top_k = 30
    
    # 🚀 GPU ACCELERATION SETUP
    USE_GPU = False
    corpus_hvt_T = corpus_hvt.T
    
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
        logger.info("🟢 GPU (CuPy) detected! Moving Corpus to GPU for Rocket Speed Dot Product...")
        # CuPy requires CSR format explicitly. Passing CSC scrambles internal data!
        corpus_hvt_gpu = cpx_sparse.csr_matrix(corpus_hvt_T.tocsr())
        USE_GPU = True
    except ImportError:
        logger.info("🟡 GPU not found. Falling back to CPU Dot Product with safe sub-batching...")
    
    logger.info(f"[3] Chunked Processing (Size: {chunk_size}, Thresh: {threshold}, Top-K: {top_k})...")
    
    # Dictionary to store candidates for candidate_pairs.tsv
    candidates_dict = defaultdict(list)
    chunk_files = []
    
    for chunk_idx, start_idx in enumerate(range(0, n_queries, chunk_size)):
        t_chunk = time.time()
        end_idx = min(start_idx + chunk_size, n_queries)
        
        pair_args = []
        
        if USE_GPU:
            # GPU Accelerated Path (Super Fast)
            s1_chunk_gpu = cpx_sparse.csr_matrix(s1_hvt[start_idx:end_idx])
            sim_mat_gpu = s1_chunk_gpu.dot(corpus_hvt_gpu)
            sim_mat = sim_mat_gpu.get() # Pull sparse matrix back to CPU RAM
            del s1_chunk_gpu, sim_mat_gpu
            
            for i in range(sim_mat.shape[0]):
                q_idx = start_idx + i
                q_id = s1_ids[q_idx]
                q_name = s1_names[q_idx]
                q_addr = s1_addrs[q_idx]
                q_country = s1_countries[q_idx]
                
                row = sim_mat.getrow(i)
                mask = row.data >= threshold
                valid_idx = row.indices[mask]
                valid_scores = row.data[mask]
                
                if len(valid_scores) > top_k:
                    top_indices = np.argsort(valid_scores)[::-1][:top_k]
                    valid_idx = valid_idx[top_indices]
                    valid_scores = valid_scores[top_indices]
                    
                order = np.argsort(valid_scores)[::-1]
                
                for rank, j in enumerate(order):
                    c_idx = valid_idx[j]
                    candidates_dict[q_id].append(cor_ids[c_idx])
                    pair_args.append((q_id, cor_ids[c_idx], q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j]))
        else:
            # CPU Fallback Path
            is_cpp = False
            try:
                from sparse_dot_topn import sp_matmul_topn
                # sparse_dot_topn requires CSR format, passing CSC will silently fail!
                corpus_hvt_T_csr = corpus_hvt_T.tocsr()
                sim_mat = sp_matmul_topn(s1_hvt[start_idx:end_idx], corpus_hvt_T_csr, top_n=top_k, threshold=threshold, n_threads=-1)
                is_cpp = True
            except ImportError:
                try:
                    from sparse_dot_topn.awesome_cossim_topn import awesome_cossim_topn
                    corpus_hvt_T_csr = corpus_hvt_T.tocsr()
                    sim_mat = awesome_cossim_topn(s1_hvt[start_idx:end_idx], corpus_hvt_T_csr, top_k, threshold, use_threads=True, n_jobs=-1)
                    is_cpp = True
                except ImportError:
                    is_cpp = False
            
            if is_cpp:
                for i in range(sim_mat.shape[0]):
                    q_idx = start_idx + i
                    q_id = s1_ids[q_idx]
                    q_name = s1_names[q_idx]
                    q_addr = s1_addrs[q_idx]
                    q_country = s1_countries[q_idx]
                    
                    row = sim_mat.getrow(i)
                    valid_idx = row.indices
                    valid_scores = row.data
                    
                    order = np.argsort(valid_scores)[::-1]
                    for rank, j in enumerate(order):
                        c_idx = valid_idx[j]
                        candidates_dict[q_id].append(cor_ids[c_idx])
                        pair_args.append((q_id, cor_ids[c_idx], q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j]))
            else:
                # Vanilla Scipy CPU Fallback with safe sub-batching
                sub_batch_size = 1000
                for sub_start in range(start_idx, end_idx, sub_batch_size):
                    sub_end = min(sub_start + sub_batch_size, end_idx)
                    sim_mat = s1_hvt[sub_start:sub_end].dot(corpus_hvt_T)
                    
                    for i in range(sim_mat.shape[0]):
                        q_idx = sub_start + i
                        q_id = s1_ids[q_idx]
                        q_name = s1_names[q_idx]
                        q_addr = s1_addrs[q_idx]
                        q_country = s1_countries[q_idx]
                        
                        row = sim_mat.getrow(i)
                        mask = row.data >= threshold
                        valid_idx = row.indices[mask]
                        valid_scores = row.data[mask]
                        
                        if len(valid_scores) > top_k:
                            top_indices = np.argsort(valid_scores)[::-1][:top_k]
                            valid_idx = valid_idx[top_indices]
                            valid_scores = valid_scores[top_indices]
                            
                        order = np.argsort(valid_scores)[::-1]
                        
                        for rank, j in enumerate(order):
                            c_idx = valid_idx[j]
                            candidates_dict[q_id].append(cor_ids[c_idx])
                            pair_args.append((q_id, cor_ids[c_idx], q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j]))
        
        if not pair_args: continue
        
        feats = Parallel(n_jobs=-1, prefer='threads')(
            delayed(generate_features)(*args) for args in pair_args
        )
        
        chunk_path = processed_dir / f"test_chunk_{chunk_idx:03d}.parquet"
        pd.DataFrame(feats).to_parquet(chunk_path, index=False)
        chunk_files.append(chunk_path)
        
        logger.info(f"  -> Chunk {chunk_idx+1}/{n_chunks} saved to disk ({len(feats):,} pairs) in {time.time()-t_chunk:.1f}s")
        
        # Explicit memory cleanup
        del sim_mat
        del pair_args
        del feats
        import gc
        gc.collect()
        
    # ── 6. LIGHTGBM INFERENCE ──
    logger.info("Loading pre-trained LightGBM model...")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please train it first or update Kaggle path!")
        sys.exit(1)
        
    model = joblib.load(model_path)
    
    logger.info("Running LightGBM inference on Test Chunks...")
    drop_cols = ['query_id', 'candidate_id', 'label']
    
    pred_dfs = []
    for chunk_path in chunk_files:
        df = pd.read_parquet(chunk_path)
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        X_test = df[feature_cols].fillna(0)
        
        out_df = df[['query_id', 'candidate_id']].copy()
        out_df['pred_prob'] = model.predict_proba(X_test)[:, 1]
        
        pred_dfs.append(out_df)
        del df, X_test
        
    test_df = pd.concat(pred_dfs, ignore_index=True)
    del pred_dfs
    import gc; gc.collect()
    
    # ── 7. GENERATE SUBMISSION FILES ──
    logger.info("Generating Final Output Files...")
    
    import json
    thresh_path = model_path.parent / "lgbm_threshold.json"
    if thresh_path.exists():
        with open(thresh_path, "r") as f:
            LGBM_THRESHOLD = json.load(f).get("optimal_threshold", 0.1)
        logger.info(f"Loaded mathematically optimal threshold from training: {LGBM_THRESHOLD}")
    else:
        LGBM_THRESHOLD = 0.1
        logger.warning(f"Threshold file not found at {thresh_path}. Defaulting to {LGBM_THRESHOLD}")
    
    # Filter predictions
    matches = test_df[test_df['pred_prob'] >= LGBM_THRESHOLD]
    submission_dict = defaultdict(list)
    for _, row in matches.iterrows():
        submission_dict[row['query_id']].append(row['candidate_id'])
        
    # Write matching_results.tsv
    sub_records = []
    for q_id in s1_ids:
        matched_str = ",".join(submission_dict.get(q_id, []))
        sub_records.append({'source1_entity_id': q_id, 'matched_entity_ids': matched_str})
        
    sub_df = pd.DataFrame(sub_records)
    sub_path = out_dir / "matching_results.tsv"
    sub_df.to_csv(sub_path, sep='\t', index=False)
    
    # Write candidate_pairs.tsv
    cand_records = []
    for q_id in s1_ids:
        cand_str = ",".join(candidates_dict.get(q_id, []))
        cand_records.append({'source1_entity_id': q_id, 'candidate_entity_ids': cand_str})
        
    cand_df = pd.DataFrame(cand_records)
    cand_path = out_dir / "candidate_pairs.tsv"
    cand_df.to_csv(cand_path, sep='\t', index=False)
    
    logger.info("=====================================================")
    logger.info(f"✅ SUBMISSION FILES SUCCESSFULLY GENERATED IN output/")
    logger.info(f"  -> {sub_path}")
    logger.info(f"  -> {cand_path}")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
