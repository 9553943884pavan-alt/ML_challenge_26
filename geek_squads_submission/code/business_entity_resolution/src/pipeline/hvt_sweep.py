import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
import sys
from pathlib import Path
import unicodedata
import re

# ─────────────────────────────────────────────────────────────
#  TEXT NORMALIZATION ENGINE (Copied for isolated sweep)
# ─────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = {
    r'\binc\b\.?': '', r'\bllc\b\.?': '', r'\bl\.l\.c\.\b': '', r'\bltd\b\.?': '', r'\blimited\b': '',
    r'\bcorp\b\.?': '', r'\bcorporation\b': '', r'\bco\b\.?': '', r'\bcompany\b': '', r'\bplc\b\.?': '',
    r'\bgmbh\b': '', r'\bsa\b': '', r'\bnv\b': '', r'\bbv\b': '', r'\bsrl\b': '', r'\bspa\b': ''
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

def advanced_clean_name(text):
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
    return sort_tokens(text)

def advanced_clean_address(text):
    if pd.isna(text) or text is None: return ""
    text = str(text)
    text = unicode_normalize(text)
    text = case_fold_and_whitespace(text)
    text = remove_business_noise(text)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = normalize_numbers(text)
    text = deduplicate_tokens(text)
    return sort_tokens(text)

# ─────────────────────────────────────────────────────────────
#  SWEEP LOGIC
# ─────────────────────────────────────────────────────────────
def main():
    print("Loading datasets...")
    base_dir = Path(__file__).resolve().parents[2]
    train_norm_dir = base_dir / "data" / "processed" / "train_norm"
    
    # Load 5000 queries and a smaller corpus to test speed
    s1 = pd.read_parquet(train_norm_dir / "train_source1.parquet").sample(n=5000, random_state=42)
    s2 = pd.read_parquet(train_norm_dir / "train_source2.parquet").sample(n=20000, random_state=42)
    s3 = pd.read_parquet(train_norm_dir / "train_source3.parquet").sample(n=20000, random_state=42)
    corpus = pd.concat([s2, s3]).drop_duplicates(subset=['entity_id'])

    print("Normalizing strings on the fly...")
    t_norm = time.time()
    for df in [s1, corpus]:
        df['name_lex'] = df['business_name'].apply(advanced_clean_name)
        df['addr_lex'] = df['business_address'].apply(advanced_clean_address)
    print(f"Normalization took {time.time()-t_norm:.2f}s")

    # Get True Matches
    q_ids = set(s1['entity_id'])
    
    # Check for Ground truth
    gt_path = base_dir / "data" / "processed" / "train" / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = base_dir / "data" / "raw" / "train" / "train_ground_truth.tsv"
        gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
    else:
        gt_df = pd.read_parquet(gt_path)
        
    gt_df = gt_df[gt_df['source1_entity_id'].isin(q_ids)]

    true_pairs = set()
    for _, row in gt_df.iterrows():
        q_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids'])
        if pd.isna(m_str) or m_str == 'nan' or not m_str: continue
        for c_id in m_str.split(','):
            if c_id in corpus['entity_id'].values:
                true_pairs.add((q_id, c_id))

    print(f"\nTotal valid ground truth pairs in this subset: {len(true_pairs)}")
    if len(true_pairs) == 0:
        print("No ground truth pairs found in sample. Exiting.")
        sys.exit(0)

    corpus_text = corpus['name_lex'].fillna('') + " " + corpus['addr_lex'].fillna('')
    s1_text = s1['name_lex'].fillna('') + " " + s1['addr_lex'].fillna('')

    s1_ids = s1['entity_id'].values
    cor_ids = corpus['entity_id'].values

    configs = [
        {"max_df": 0.02, "thresh": 0.28, "top_k": None},
        {"max_df": 0.30, "thresh": 0.15, "top_k": None},
        {"max_df": 0.30, "thresh": 0.10, "top_k": None},
        {"max_df": 0.30, "thresh": 0.05, "top_k": 20}, # Top-K limited!
        {"max_df": 1.00, "thresh": 0.10, "top_k": 20}, # No word dropping, Top-K limited!
    ]

    print("\n" + "="*80)
    print(f"{'max_df':>6} | {'thresh':>6} | {'top_k':>5} | {'Recall':>8} | {'Candidates/Query':>16} | {'Memory Scale':>12} | {'Time (s)':>8}")
    print("-" * 80)

    for cfg in configs:
        t0 = time.time()
        max_df = cfg["max_df"]
        thresh = cfg["thresh"]
        top_k = cfg["top_k"]

        vec = TfidfVectorizer(analyzer='word', ngram_range=(1,1), sublinear_tf=True, min_df=2, max_df=max_df)
        c_mat = vec.fit_transform(corpus_text)
        q_mat = vec.transform(s1_text)
        
        sim = q_mat.dot(c_mat.T)
        
        found = 0
        total_candidates = 0
        
        for i in range(sim.shape[0]):
            q_id = s1_ids[i]
            row = sim.getrow(i)
            mask = row.data >= thresh
            
            valid_c_idxs = row.indices[mask]
            valid_scores = row.data[mask]
            
            if top_k is not None and len(valid_scores) > top_k:
                # Keep only top_k indices
                top_indices = np.argsort(valid_scores)[::-1][:top_k]
                valid_c_idxs = valid_c_idxs[top_indices]
                
            total_candidates += len(valid_c_idxs)
            
            for c_idx in valid_c_idxs:
                c_id = cor_ids[c_idx]
                if (q_id, c_id) in true_pairs:
                    found += 1
                    
        elapsed = time.time() - t0
        recall = found / len(true_pairs)
        c_per_q = total_candidates / len(s1_ids)
        mem_scale = f"{total_candidates * 20:,}" # Extrapolated to 100k queries (100k = 20 * 5k)
        top_k_str = str(top_k) if top_k else "None"
        
        print(f"{max_df:>6.2f} | {thresh:>6.2f} | {top_k_str:>5} | {recall:>8.4f} | {c_per_q:>16.1f} | {mem_scale:>12} | {elapsed:>8.2f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
