"""
Kaggle End-to-End Pipeline (Train + Validate + Threshold Sweep)
===============================================================
Strictly adheres to all rules, heavily optimized for Kaggle's 30GB RAM limit.
Features a 3-Way Accelerated Retrieval Engine (GPU CuPy -> C++ sparse_dot_topn -> Safe CPU).

Pipeline Steps:
1. Text Normalization (Done ONCE per dataset).
2. HVT TF-IDF Vectorization (max_df=0.15 to preserve High Recall).
3. Chunked Candidate Retrieval & Feature Engineering:
   - Processes 4,000 queries at a time using the 3-Way Engine.
   - Computes dot product -> Thresholds at 0.15 (Top-30) -> Joblib Parallel Features.
   - Saves chunk directly to disk as Parquet to free RAM immediately!
4. LightGBM Training: Splits chunks (80% Train / 20% Validation), trains, and saves the model.
5. Threshold Sweep: Predicts on the 20% held-out chunks, sweeps LightGBM probability 
   thresholds (0.1 to 0.9), and computes global Precision, Recall, and F0.5.
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
import os
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

try:
    import jellyfish
    import rapidfuzz
except ImportError:
    print("Please install required packages: pip install jellyfish rapidfuzz lightgbm")
    sys.exit(1)

import unicodedata
from urllib.parse import urlparse

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

def generate_features(q_id, c_id, q_name, c_name, q_addr, c_addr, q_country, c_country, rank, score, is_match):
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
    if is_match is not None:
        features['label'] = 1 if is_match else 0
        
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
#  CHUNKED PROCESSING ENGINE
# ─────────────────────────────────────────────────────────────
def process_data_in_chunks(s1, corpus, gt_dict, vec_hvt, out_dir, mode="train"):
    """
    Executes TF-IDF Retrieval -> Thresholding -> Feature Extraction -> Disk Save.
    Done in small chunks to ensure RAM never exceeds Kaggle limits.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Check for existing cached chunks (Kaggle Dataset / output re-use)
    existing_chunks = list(Path(out_dir).glob("*.parquet"))
    if len(existing_chunks) > 0:
        logger.info(f"✅ Found {len(existing_chunks)} cached chunks in {out_dir}! Skipping TF-IDF retrieval and using cache.")
        return sorted([Path(f) for f in existing_chunks])
        
    s1_ids = s1['ids']
    s1_names = s1['names']
    s1_addrs = s1['addrs']
    s1_countries = s1['countries']
    
    cor_ids = corpus['ids']
    cor_names = corpus['names']
    cor_addrs = corpus['addrs']
    cor_countries = corpus['countries']
    
    t_start = time.time()
    logger.info(f"[1] Vectorizing Corpus for {mode.upper()}...")
    corpus_text = cor_names + " " + cor_addrs
    
    is_pretrained = getattr(vec_hvt, 'is_pretrained', False)
    if mode == "train" and not is_pretrained:
        corpus_hvt = vec_hvt.fit_transform(corpus_text)
        import joblib
        vec_save_path = Path(out_dir).parent / "vec_hvt.joblib"
        joblib.dump(vec_hvt, vec_save_path)
        logger.info(f"    -> Saved newly fitted TF-IDF Vectorizer to {vec_save_path}")
    else:
        logger.info(f"    -> Using PRE-TRAINED vectorizer (Skipping fit). Transforming only...")
        corpus_hvt = vec_hvt.transform(corpus_text)
        
    logger.info(f"    -> Corpus vectorization took {time.time() - t_start:.2f}s")
    
    t_start = time.time()
    logger.info(f"[2] Vectorizing Queries for {mode.upper()}...")
    s1_text = s1_names + " " + s1_addrs
    s1_hvt = vec_hvt.transform(s1_text)
    logger.info(f"    -> Query vectorization took {time.time() - t_start:.2f}s")
    
    n_queries = s1_hvt.shape[0]
    chunk_size = 4000 
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    threshold = 0.15 
    top_k = 30       # Increased to 30 to maximize recall since Engine is fast
    
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
                trues = set(gt_dict.get(q_id, [])) if gt_dict else set()
                
                for rank, j in enumerate(order):
                    c_idx = valid_idx[j]
                    c_id = cor_ids[c_idx]
                    is_match = (c_id in trues) if gt_dict else None
                    pair_args.append((q_id, c_id, q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j], is_match))
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
                    trues = set(gt_dict.get(q_id, [])) if gt_dict else set()
                    
                    for rank, j in enumerate(order):
                        c_idx = valid_idx[j]
                        c_id = cor_ids[c_idx]
                        is_match = (c_id in trues) if gt_dict else None
                        pair_args.append((q_id, c_id, q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j], is_match))
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
                        trues = set(gt_dict.get(q_id, [])) if gt_dict else set()
                        
                        for rank, j in enumerate(order):
                            c_idx = valid_idx[j]
                            c_id = cor_ids[c_idx]
                            is_match = (c_id in trues) if gt_dict else None
                            pair_args.append((q_id, c_id, q_name, cor_names[c_idx], q_addr, cor_addrs[c_idx], q_country, cor_countries[c_idx], rank+1, valid_scores[j], is_match))
                
        # 3. Parallel Feature Engineering
        if not pair_args: continue
        
        feats = Parallel(n_jobs=-1, prefer='threads')(
            delayed(generate_features)(*args) for args in pair_args
        )
        
        # 4. Save Chunk to Disk (Freeing RAM)
        chunk_path = out_dir / f"{mode}_chunk_{chunk_idx:03d}.parquet"
        pd.DataFrame(feats).to_parquet(chunk_path, index=False)
        chunk_files.append(chunk_path)
        
        logger.info(f"  -> Chunk {chunk_idx+1}/{n_chunks} saved to disk ({len(feats):,} pairs) in {time.time()-t_chunk:.1f}s")
        
        # Explicitly clean up heavy objects to prevent loop-transition memory spikes
        del sim_mat
        del pair_args
        del feats
        import gc
        gc.collect()
        
    return chunk_files

# ─────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    # Auto-detect if running on Kaggle or Locally
    is_kaggle = os.path.exists("/kaggle/input")
    if is_kaggle:
        logger.info("🌍 Detected Kaggle Environment!")
        base_dir = Path("/kaggle/working")
        train_norm_dir = Path("/kaggle/input/datasets/pavan9938/train-norm")
        gt_dir = Path("/kaggle/input/datasets/pavan9938/train-norm") 
        processed_dir = base_dir 
    else:
        logger.info("💻 Detected Local Environment!")
        base_dir = Path(__file__).resolve().parents[2]
        train_norm_dir = base_dir / "data" / "processed" / "train_norm"
        gt_dir = base_dir / "data" / "processed" / "train"
        processed_dir = base_dir / "data" / "processed" / "kaggle_pipeline"
        
    os.makedirs(processed_dir, exist_ok=True)
    train_chunk_dir = processed_dir / "train_chunks"
    
    # ── LOAD GROUND TRUTH (Always required for chunks & sweep) ──
    gt_tsv = gt_dir / "train_ground_truth.tsv"
    gt_parquet = gt_dir / "train_ground_truth.parquet"
    if gt_tsv.exists():
        gt_df = pd.read_csv(gt_tsv, sep='\t', dtype=str)
    else:
        gt_df = pd.read_parquet(gt_parquet)
        
    gt_df['source1_entity_id'] = gt_df['source1_entity_id'].astype(str)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].astype(str)
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        m_str = str(row['matched_entity_ids'])
        gt_dict[row['source1_entity_id']] = m_str.split(",") if m_str and m_str != "nan" else []
        
    # ── CHECK FOR CACHED CHUNKS FIRST ───────────────────────
    existing_chunks = sorted(list(train_chunk_dir.glob("*.parquet")))
    chunks_cached = len(existing_chunks) > 0
    
    if chunks_cached:
        logger.info(f"✅ Found {len(existing_chunks)} cached chunks! Skipping raw data loading and TF-IDF.")
        train_files = [str(f) for f in existing_chunks]
    else:
        # ── A. LOAD TRAIN DATA ──────────────────────────────────
        t_load = time.time()
        if is_kaggle:
            norm_cache_dir = Path("/kaggle/input/datasets/pavan9938/normalized-cache")
            save_cache_dir = processed_dir / "normalized_cache"
        else:
            norm_cache_dir = processed_dir / "normalized_cache"
            save_cache_dir = norm_cache_dir
            
        s1_cache_path = norm_cache_dir / "s1_train_norm.parquet"
        corpus_cache_path = norm_cache_dir / "corpus_train_norm.parquet"
        
        if s1_cache_path.exists() and corpus_cache_path.exists():
            logger.info("✅ Found pre-normalized data in Kaggle working cache! Loading directly...")
            s1_train = pd.read_parquet(s1_cache_path)
            corpus_train = pd.read_parquet(corpus_cache_path)
            logger.info(f"    -> Data loading completed in {time.time() - t_load:.2f}s")
        else:
            logger.info(f"Loading Train Data from {train_norm_dir}...")
            
            s1_full = pd.read_parquet(train_norm_dir / "train_source1.parquet")
            s2_full = pd.read_parquet(train_norm_dir / "train_source2.parquet")
            s3_full = pd.read_parquet(train_norm_dir / "train_source3.parquet")
            
            if not is_kaggle:
                logger.info("  -> Downsampling corpus for local run...")
                s1_train = s1_full.sample(n=10000, random_state=42).copy()
                s2_train = s2_full.sample(n=400000, random_state=42).copy()
                s3_train = s3_full.sample(n=400000, random_state=42).copy()
            else:
                s1_train = s1_full
                s2_train = s2_full
                s3_train = s3_full
                
            corpus_train = pd.concat([s2_train, s3_train]).drop_duplicates(subset=['entity_id'])
            
            # EXPLICIT RAM CLEANUP: Free the individual dataframes to prevent RAM doubling
            del s2_train, s3_train
            if not is_kaggle:
                del s1_full, s2_full, s3_full
            import gc; gc.collect()
            
            logger.info(f"    -> Data loading completed in {time.time() - t_load:.2f}s")
                
            t_check = time.time()
            logger.info("\n" + "="*50)
            logger.info(" [PHASE 2/4] TEXT NORMALIZATION")
            logger.info("="*50)
            
            from tqdm import tqdm
            tqdm.pandas(mininterval=10) # Only print progress every 10s to avoid spamming Kaggle logs
            
            for df_name, df in [("Queries (S1)", s1_train), ("Corpus (S2+S3)", corpus_train)]:
                if 'name_lex' not in df.columns:
                    logger.info(f"  -> Normalizing Business Names for {df_name} ({len(df):,} rows)...")
                    df['name_lex'] = df['business_name'].progress_apply(lambda x: advanced_clean_name(str(x), apply_token_sort=True))
                if 'addr_lex' not in df.columns:
                    logger.info(f"  -> Normalizing Addresses for {df_name} ({len(df):,} rows)...")
                    df['addr_lex'] = df['business_address'].progress_apply(lambda x: advanced_clean_address(str(x), apply_token_sort=True))
            
            os.makedirs(save_cache_dir, exist_ok=True)
            s1_train.to_parquet(save_cache_dir / "s1_train_norm.parquet", index=False)
            corpus_train.to_parquet(save_cache_dir / "corpus_train_norm.parquet", index=False)
            logger.info(f"    -> Saved normalized data to {save_cache_dir} for faster re-runs!")
            
        logger.info(f"    -> Extracting C-Arrays and Freeing Pandas DataFrames to save 10GB RAM...")
        s1_arrays = {
            'ids': s1_train['entity_id'].values.copy(),
            'names': s1_train['name_lex'].astype(object).fillna('').values,
            'addrs': s1_train['addr_lex'].astype(object).fillna('').values,
            'countries': s1_train['country'].astype(object).fillna('').values if 'country' in s1_train.columns else np.array([""] * len(s1_train))
        }
        
        # Sample Training Queries to drastically cut processing time (100k is enough for LightGBM)
        n_train_q = len(s1_arrays['ids'])
        if n_train_q > 100000:
            logger.info(f"    -> Sampling 100,000 queries out of {n_train_q} to speed up Training (42 Hours -> 30 Mins)...")
            np.random.seed(42)
            sample_idx = np.random.choice(n_train_q, 100000, replace=False)
            for k in s1_arrays:
                s1_arrays[k] = s1_arrays[k][sample_idx]
                
        corpus_arrays = {
            'ids': corpus_train['entity_id'].values.copy(),
            'names': corpus_train['name_lex'].astype(object).fillna('').values,
            'addrs': corpus_train['addr_lex'].astype(object).fillna('').values,
            'countries': corpus_train['country'].astype(object).fillna('').values if 'country' in corpus_train.columns else np.array([""] * len(corpus_train))
        }
        del s1_train, corpus_train
        import gc; gc.collect()
        
        import joblib
        if is_kaggle:
            vec_path = Path("/kaggle/input/datasets/pavan9938/vectorizer/vec_hvt.joblib") # PLEASE UPDATE THIS KAGGLE PATH IF NEEDED
        else:
            vec_path = processed_dir / "vec_hvt.joblib"
            
        if vec_path.exists():
            logger.info(f"    -> ✅ Found pre-trained Vectorizer at {vec_path}! Loading it...")
            vec_hvt = joblib.load(vec_path)
            vec_hvt.is_pretrained = True
        else:
            logger.info(f"    -> No pre-trained Vectorizer found. Initializing new TF-IDF setup...")
            vec_hvt = TfidfVectorizer(analyzer='word', ngram_range=(1,1), sublinear_tf=True, min_df=2, max_df=0.15)
            vec_hvt.is_pretrained = False
        
        # ── B. TRAIN CHUNK PROCESSING ───────────────────────────
        train_files = process_data_in_chunks(s1_arrays, corpus_arrays, gt_dict, vec_hvt, train_chunk_dir, "train")

    # ── C. LIGHTGBM TRAINING ON TRAIN CHUNKS ────────────────
    t_lgb = time.time()
    logger.info("Loading Train and Validation Chunks into LightGBM...")
    
    # Keep last 20% of chunks as held-out validation set
    if len(train_files) == 1:
        train_chunk_files = train_files
        val_chunk_files = train_files
    else:
        n_val_chunks = max(1, len(train_files) // 5)
        train_chunk_files = train_files[:-n_val_chunks]
        val_chunk_files = train_files[-n_val_chunks:]
    
    train_df = pd.concat([pd.read_parquet(f) for f in train_chunk_files], ignore_index=True)
    val_df = pd.concat([pd.read_parquet(f) for f in val_chunk_files], ignore_index=True)
    
    drop_cols = ['query_id', 'candidate_id', 'label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['label']
    
    val_meta = val_df[['query_id', 'candidate_id']].copy()
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['label']
    
    # 💥 CRITICAL RAM CLEANUP: Delete 10GB DataFrames before LightGBM allocation
    del train_df, val_df
    import gc; gc.collect()
    
    params = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 63, 'verbose': -1, 'n_jobs': -1}
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    
    logger.info(f"Training LightGBM on {len(X_train):,} pairs (held out {len(X_val):,} pairs for eval)...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
    
    import joblib
    model_path = processed_dir / "lgbm_kaggle.joblib"
    joblib.dump(model, model_path)
    logger.info(f"    -> LightGBM model saved to {model_path}")
    logger.info(f"    -> Total training phase took {time.time() - t_lgb:.2f}s")
    
    t_sweep = time.time()
    logger.info("Predicting on Held-Out Validation Chunks...")
    val_meta['pred_prob'] = model.predict_proba(X_val)[:, 1]
    
    logger.info("Sweeping LightGBM Classification Thresholds...")
    sweep_thresholds = np.arange(0.1, 0.99, 0.1)
    
    results = []
    val_qids = val_meta['query_id'].unique()
    
    for T in sweep_thresholds:
        # Get candidates above threshold T
        matches = val_meta[val_meta['pred_prob'] >= T]
        
        pred_dict = defaultdict(set)
        for _, row in matches.iterrows():
            pred_dict[row['query_id']].add(row['candidate_id'])
            
        f05_scores = []
        rec_scores = []
        prec_scores = []
        
        for q_id in val_qids:
            trues = set(gt_dict.get(q_id, []))
            preds = pred_dict.get(q_id, set())
            
            # Singleton handling matching Kaggle's exact criteria
            if not trues and not preds:
                f05_scores.append(1.0)
                rec_scores.append(1.0)
                prec_scores.append(1.0)
            elif not trues and preds:
                f05_scores.append(0.0)
                rec_scores.append(1.0)
                prec_scores.append(0.0)
            elif trues and not preds:
                f05_scores.append(0.0)
                rec_scores.append(0.0)
                prec_scores.append(1.0) # Predicting nothing means precision isn't ruined
            else:
                intersect = len(preds & trues)
                rec = intersect / len(trues)
                prec = intersect / len(preds)
                
                if (0.25 * prec + rec) > 0:
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                else:
                    f05 = 0.0
                    
                f05_scores.append(f05)
                rec_scores.append(rec)
                prec_scores.append(prec)
                
        macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
        macro_rec = float(np.mean(rec_scores)) if rec_scores else 0.0
        macro_prec = float(np.mean(prec_scores)) if prec_scores else 0.0
        
        results.append({'T': T, 'Recall': macro_rec, 'Precision': macro_prec, 'F05': macro_f05})
    logger.info(f"    -> Threshold sweep completed in {time.time() - t_sweep:.2f}s")
    
    print("\n" + "="*50)
    print(f"{'LGBM Thresh':>12} | {'Recall':>8} | {'Precision':>9} | {'F0.5':>8}")
    print("-" * 50)
    for r in results:
        flag = "  <-- BEST" if r['T'] == max(results, key=lambda x: x['F05'])['T'] else ""
        print(f"{r['T']:>12.1f} | {r['Recall']:>8.4f} | {r['Precision']:>9.4f} | {r['F05']:>8.4f}{flag}")
    print("="*50)
    
    best_t = max(results, key=lambda x: x['F05'])['T']
    logger.info(f"✅ Optimal LightGBM Threshold mathematically found: {best_t:.1f}")
    
    # Save optimal threshold for test script
    import json
    thresh_path = processed_dir / "lgbm_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump({"optimal_threshold": float(best_t)}, f)
    logger.info(f"    -> Saved optimal threshold to {thresh_path}")
    
if __name__ == "__main__":
    main()
