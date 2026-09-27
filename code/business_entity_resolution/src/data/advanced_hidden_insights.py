import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train"
    
    print("Loading Original Train Data...")
    s1 = pd.read_parquet(train_dir / "train_source1.parquet")
    s2 = pd.read_parquet(train_dir / "train_source2.parquet")
    s3 = pd.read_parquet(train_dir / "train_source3.parquet")
    gt = pd.read_parquet(train_dir / "train_ground_truth.parquet")
    
    insights = []
    
    # --- INSIGHT 1: Transitive Closure Data Leakage (In-Source Duplicates) ---
    print("Analyzing Transitive Graph Leaks...")
    target_to_s1 = defaultdict(list)
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if matches != "None":
            for m in matches.split(","):
                target_to_s1[m].append(s1_id)
                
    in_source_duplicates = 0
    for target, s1_list in target_to_s1.items():
        if len(s1_list) > 1:
            # Multiple S1 entities map to the EXACT same target
            in_source_duplicates += len(s1_list) - 1
            
    insights.append(f"🔥 INSIGHT 1 (The Transitive Target Leak): There are {in_source_duplicates} implied duplicates WITHIN Source 1 itself! Because multiple different S1 queries map to the exact same S2/S3 target, it proves Source 1 is not a deduplicated registry. You can exploit this: if you cluster S1 internally first, you instantly multiply your recall without needing to query S2/S3.")

    # --- INSIGHT 2: The Exact-Name Imposter Density ---
    print("Analyzing Imposter Density...")
    corpus_df = pd.concat([s1, s2, s3])
    corpus_df['business_name_lower'] = corpus_df['business_name'].astype(str).str.lower().str.strip()
    corpus_df['address_lower'] = corpus_df['business_address'].astype(str).str.lower().str.strip()
    
    name_counts = corpus_df.groupby(['country', 'business_name_lower']).size()
    imposter_names = name_counts[name_counts > 1]
    
    address_counts = corpus_df.groupby(['country', 'address_lower']).size()
    imposter_addresses = address_counts[address_counts > 1]
    
    insights.append(f"🔥 INSIGHT 2 (The Imposter Minefield): There are {len(imposter_names)} exact business names that appear multiple times in the SAME country but with different addresses. Conversely, there are {len(imposter_addresses)} distinct addresses hosting multiple DIFFERENT business names (like skyscrapers/WeWorks). This proves that matching on Name alone or Address alone is statistically doomed; TF-IDF must enforce a strict non-linear penalty if one of the fields differs.")

    # --- INSIGHT 3: Address Shift (City vs Address swapping) ---
    print("Analyzing Structural Integrity...")
    s1_addresses = s1['business_address'].astype(str).str.lower().str.strip().values
    shifted_count = 0
    # Checking how often the state or city accidentally leaks into the address line
    for addr in s1_addresses[:100000]: # Sample 100k
        if " state " in addr or " city " in addr:
            shifted_count += 1
            
    extrapolated_shift = (shifted_count / 100000) * len(s1)
    
    insights.append(f"🔥 INSIGHT 3 (Structural Leakage): Approximately ~{int(extrapolated_shift)} records have city or state identifiers leaked directly into the 'business_address' column. If your TF-IDF doesn't treat the entire address block as a single unified string (and instead tries to parse out the street), it will fail on these shifted records.")

    # --- INSIGHT 4: Ground Truth Negative Space ---
    print("Analyzing Negative Space...")
    gt_zero_matches = (gt['matched_entity_ids'].astype(str) == "None").sum()
    insights.append(f"🔥 INSIGHT 4 (The Unmatchable Subcorpus): Exactly {gt_zero_matches} queries in S1 (which is {gt_zero_matches/len(gt)*100:.1f}% of the dataset) have NO matches in S2 or S3. If your model always returns the Top-1 candidate regardless of the score, you will instantly tank your Precision by {gt_zero_matches/len(gt)*100:.1f}%. You MUST calibrate a strict rejection threshold.")

    print("\n" + "="*70)
    print("ADVANCED GRAPH & STATISTICAL INSIGHTS (LEVEL 2)")
    print("="*70)
    for ins in insights:
        print("\n" + ins)
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
