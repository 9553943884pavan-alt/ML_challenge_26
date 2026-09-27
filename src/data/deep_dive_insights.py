import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("Loading Original Train Data...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet")
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    gt = pd.read_parquet(train_dir / "train_ground_truth.parquet")
    
    print(f"Loaded: S1 ({len(s1)}), S2 ({len(s2)}), S3 ({len(s3)}), GT ({len(gt)})")
    
    insights = []
    
    # 1. 1-to-Many Mappings (Franchises / Subsidiaries)
    # How many S1 queries map to multiple entities?
    match_counts = gt['matched_entity_ids'].apply(lambda x: len(str(x).split(",")) if str(x) != "None" else 0)
    multiple_matches = (match_counts > 1).sum()
    max_matches = match_counts.max()
    insights.append(f"INSIGHT 1 (1-to-Many Mapping): {multiple_matches} queries in S1 map to MULTIPLE true matches (Max matches for a single query: {max_matches}). Entity Resolution here is NOT 1-to-1. It's a clustering problem. If your model assumes 1-to-1, it is mathematically capped.")

    # 2. Source Bias Analysis
    # Are true matches uniformly distributed between Source 2 and Source 3?
    s2_ids = set(s2['entity_id'].values)
    s3_ids = set(s3['entity_id'].values)
    
    s2_hits, s3_hits, cross_hits = 0, 0, 0
    
    for _, row in gt.iterrows():
        matches = str(row['matched_entity_ids'])
        if matches == "None":
            continue
        m_list = matches.split(",")
        has_s2 = any(m in s2_ids for m in m_list)
        has_s3 = any(m in s3_ids for m in m_list)
        if has_s2 and not has_s3: s2_hits += 1
        if has_s3 and not has_s2: s3_hits += 1
        if has_s2 and has_s3: cross_hits += 1
        
    insights.append(f"INSIGHT 2 (Source Isolation): {s2_hits} queries match ONLY S2. {s3_hits} queries match ONLY S3. {cross_hits} match BOTH. This means the sources are highly distinct sub-corpora. If we train a cross-encoder indiscriminately, it might learn source-specific artifacts instead of true semantic matching.")

    # 3. The Country Rule Paradox
    # Does a true match ever cross country borders? (Which would invalidate strict country-blocking)
    # Let's map entity_id to country for S2 and S3
    corpus_country = pd.concat([
        s2[['entity_id', 'country']], 
        s3[['entity_id', 'country']]
    ]).drop_duplicates('entity_id')
    country_map = dict(zip(corpus_country['entity_id'], corpus_country['country']))
    s1_country_map = dict(zip(s1['entity_id'], s1['country']))
    
    cross_border_matches = 0
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        s1_c = str(s1_country_map.get(s1_id, "")).lower()
        
        matches = str(row['matched_entity_ids'])
        if matches == "None": continue
        
        for m in matches.split(","):
            target_c = str(country_map.get(m, "")).lower()
            if s1_c and target_c and s1_c != target_c and target_c != "none" and s1_c != "none":
                cross_border_matches += 1
                break # Count query once
                
    insights.append(f"INSIGHT 3 (Cross-Border Matching): There are {cross_border_matches} instances where Ground Truth matches two entities across DIFFERENT countries! If we hard-filter by Exact Country Match before retrieval, we artificially cap our maximum recall.")

    # 4. Token Asymmetry (SEO spam vs Registry Data)
    # Does S1 have longer names than S2/S3?
    s1_name_len = s1['business_name'].astype(str).apply(lambda x: len(x.split())).mean()
    s2_name_len = s2['business_name'].astype(str).apply(lambda x: len(x.split())).mean()
    s3_name_len = s3['business_name'].astype(str).apply(lambda x: len(x.split())).mean()
    
    insights.append(f"INSIGHT 4 (Token Asymmetry): Avg Name Length -> S1: {s1_name_len:.2f} words, S2: {s2_name_len:.2f} words, S3: {s3_name_len:.2f} words. If one source consistently has more words, it contains 'SEO fluff' or metadata. TF-IDF penalizes length asymmetry. This is why Reverse Retrieval (Target querying Query) is absolutely mandatory.")

    # 5. Missing Data Patterns
    s1_missing_address = s1['business_address'].isna().sum() / len(s1) * 100
    s2_missing_address = s2['business_address'].isna().sum() / len(s2) * 100
    s3_missing_address = s3['business_address'].isna().sum() / len(s3) * 100
    
    insights.append(f"INSIGHT 5 (Address Sparsity): Missing Addresses -> S1: {s1_missing_address:.1f}%, S2: {s2_missing_address:.1f}%, S3: {s3_missing_address:.1f}%. With such high sparsity in some sources, relying heavily on Address TF-IDF is dangerous. We must rely heavily on High-Value Token blocking on the Business Name.")

    print("\n" + "="*50)
    print("DEEP DIVE HIDDEN INSIGHTS")
    print("="*50)
    for ins in insights:
        print("\n" + ins)
    print("\n" + "="*50)
    
if __name__ == "__main__":
    main()
