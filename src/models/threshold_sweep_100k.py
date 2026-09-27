"""
Threshold Sweep Validation Script — 100k Queries
=================================================
Runs ONLY the HVT Retrieval + Threshold Sweep.
Computes Efficiency-Weighted F0.5 = F0.5(T) / log(K(T)) for each threshold.
Stops before feature generation — for decision-making only.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.text_normalization import advanced_clean_name, advanced_clean_address

def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = base_dir / "data" / "processed" / "train_norm"
    gt_dir    = base_dir / "data" / "processed" / "train"

    # ── 1. Load 100k Sample ─────────────────────────────────────
    print("Loading 1 Lakh sample...")
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

    s2_true = s2[s2['entity_id'].isin(true_match_ids)]
    s3_true = s3[s3['entity_id'].isin(true_match_ids)]
    s2_noise_len = min(400000, len(s2) - len(s2_true))
    s3_noise_len = min(400000, len(s3) - len(s3_true))
    s2_noise = s2[~s2['entity_id'].isin(true_match_ids)].sample(n=s2_noise_len, random_state=42)
    s3_noise = s3[~s3['entity_id'].isin(true_match_ids)].sample(n=s3_noise_len, random_state=42)

    corpus = pd.concat([s2_true, s3_true, s2_noise, s3_noise], ignore_index=True)\
               .drop_duplicates(subset=['entity_id'])

    print(f"Queries: {len(s1):,} | Corpus: {len(corpus):,} | True targets: {len(true_match_ids):,}")
    
    avg_true_per_query = len(true_match_ids) / len(s1)
    print(f"Avg true matches per query: {avg_true_per_query:.2f}")

    # ── 2. Text Normalization ────────────────────────────────────
    print("\nApplying text normalization...")
    for df in [s1, corpus]:
        df['name_lex'] = df['business_name'].apply(lambda x: advanced_clean_name(x, apply_token_sort=True))
        df['addr_lex'] = df['business_address'].apply(lambda x: advanced_clean_address(x, apply_token_sort=True))

    s1_ids  = s1['entity_id'].values
    cor_ids = corpus['entity_id'].values

    # ── 3. TF-IDF Vectorizer with max_df=0.25 ───────────────────
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
    s1_hvt     = vec_hvt.transform(s1['name_lex'] + " " + s1['addr_lex'])
    print(f"Vocabulary size: {len(vec_hvt.vocabulary_):,}")
    print(f"Matrix shapes: Queries {s1_hvt.shape}, Corpus {corpus_hvt.shape}")
    print(f"TF-IDF fit+transform done in {time.time()-t0:.1f}s")

    # ── 4. Low-Threshold Full Retrieval for Sweep ────────────────
    # Pull everything >= 0.10 in ONE chunked pass. Then filter in memory.
    SWEEP_MIN = 0.10
    CHUNK = 1000
    n_queries = s1_hvt.shape[0]
    n_chunks  = (n_queries + CHUNK - 1) // CHUNK

    # Store: {q_id: [(score, c_id), ...]} only for scores >= 0.10
    raw = {}
    print(f"\nRunning chunked retrieval (threshold=0.10, {n_chunks} chunks)...")
    t0 = time.time()

    for chunk_idx, start in enumerate(range(0, n_queries, CHUNK)):
        end       = min(start + CHUNK, n_queries)
        chunk_mat = s1_hvt[start:end]
        sim_mat   = chunk_mat.dot(corpus_hvt.T)

        for i in range(sim_mat.shape[0]):
            q_id = s1_ids[start + i]
            row  = sim_mat.getrow(i)
            mask = row.data >= SWEEP_MIN
            if mask.any():
                raw[q_id] = list(zip(row.data[mask], cor_ids[row.indices[mask]]))

        print(f"  Chunk {chunk_idx+1}/{n_chunks} done ({end:,}/{n_queries:,} queries)", flush=True)

    print(f"Retrieval done in {time.time()-t0:.1f}s\n")

    # ── 5. Efficiency-Weighted Threshold Sweep ───────────────────
    thresholds = np.arange(0.10, 0.51, 0.02)

    print("=" * 72)
    print(f"{'Thresh':>7} | {'Recall':>7} | {'Avg K':>8} | {'P(T)':>7} | {'F0.5(T)':>8} | {'EW-F0.5':>9} | {'Best?'}")
    print("-" * 72)

    results = []
    for T in thresholds:
        recalls, k_vals = [], []
        for q_id, trues in gt_dict.items():
            if not trues: continue
            pairs = raw.get(q_id, [])
            cands = {c_id for score, c_id in pairs if score >= T}
            k_vals.append(len(cands))
            recalls.append(len(set(trues) & cands) / len(trues))

        R  = float(np.mean(recalls)) if recalls else 0.0
        K  = float(np.mean(k_vals))  if k_vals  else 1.0
        P  = (R * avg_true_per_query / K) if K > 0 else 0.0
        
        # F0.5 = (1 + 0.25) * P * R / (0.25*P + R)
        F05 = (1.25 * P * R / (0.25 * P + R)) if (0.25 * P + R) > 0 else 0.0
        
        # Efficiency-Weighted F0.5 = F0.5 / log(K) — penalises compute cost
        EW  = (F05 / np.log(max(K, 2))) if K > 1 else F05

        results.append({'T': T, 'Recall': R, 'K': K, 'P': P, 'F05': F05, 'EW': EW})

    best_ew  = max(results, key=lambda x: x['EW'])
    best_f05 = max(results, key=lambda x: x['F05'])

    for r in results:
        flag = ""
        if r['T'] == best_ew['T']:  flag += "← EW-F0.5 Best"
        if r['T'] == best_f05['T']: flag += "  ← F0.5 Best"
        print(f"{r['T']:>7.2f} | {r['Recall']:>7.4f} | {r['K']:>8.1f} | {r['P']:>7.4f} | {r['F05']:>8.4f} | {r['EW']:>9.5f} | {flag}")

    print("=" * 72)
    print("\n>>> Efficiency-Weighted F0.5 selects  T = {:.2f}".format(best_ew['T']))
    print("    Recall={:.4f} | Avg K={:.1f} | F0.5={:.4f} | EW-F0.5={:.5f}".format(best_ew['Recall'], best_ew['K'], best_ew['F05'], best_ew['EW']))
    print("\n>>> Raw F0.5 selects                  T = {:.2f}".format(best_f05['T']))
    print("    Recall={:.4f} | Avg K={:.1f} | F0.5={:.4f}".format(best_f05['Recall'], best_f05['K'], best_f05['F05']))
    
    # ── 6. Save Raw Candidates for Re-use ─────────────────────────
    # Store everything >= 0.10 so we don't have to rerun the dot products!
    print("\nSaving raw candidates (>= 0.10) to disk for reuse...")
    raw_list = []
    for q_id, pairs in raw.items():
        # pairs is a list of (score, c_id) tuples
        if not pairs: continue
        # Sort by score descending before saving
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        raw_list.append({
            'query_id': q_id,
            'candidate_ids': [c_id for score, c_id in pairs],
            'scores': [score for score, c_id in pairs]
        })
        
    out_df = pd.DataFrame(raw_list)
    out_path = base_dir / "data" / "processed" / "raw_candidates_100k.parquet"
    out_df.to_parquet(out_path, index=False)
    
    print(f"Saved {len(out_df):,} queries' candidates to: {out_path}")
    print("\n⚠  STOPPING HERE — no features generated, no model trained.")
    print("   Review the table above and decide the final threshold.\n")

if __name__ == "__main__":
    main()
