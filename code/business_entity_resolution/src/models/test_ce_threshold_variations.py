"""
test_ce_threshold_variations.py
================================
Lightning fast evaluation of Cross-Encoder score thresholds from 0.10 to 0.99
using cached predictions on the 10k validation set.
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[2]
DATA_PROC = BASE_DIR / "data" / "processed"
TRAIN_DIR = DATA_PROC / "train"
CACHE_10K = DATA_PROC / "cross_encoder_preds_cache_10k_val.pkl"

def f05(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def compute_metrics(preds, truth):
    p_list, r_list, f_list = [], [], []
    for qid, true_list in truth.items():
        true_set = set(true_list)
        if not true_set:
            continue
        pred_set = set(preds.get(qid, []))
        tp = len(true_set & pred_set)
        prec = tp / len(pred_set) if pred_set else 0.0
        rec  = tp / len(true_set)
        p_list.append(prec)
        r_list.append(rec)
        f_list.append(f05(prec, rec))
    return float(np.mean(p_list)), float(np.mean(r_list)), float(np.mean(f_list))

def main():
    t0 = time.time()
    print("1. Loading Cached CE predictions...", flush=True)
    with open(CACHE_10K, "rb") as f:
        ce_cache = pickle.load(f)
    print(f"   Loaded {len(ce_cache):,} cached queries in {time.time()-t0:.2f}s", flush=True)

    t1 = time.time()
    print("2. Loading Ground Truth (filtered)...", flush=True)
    gt = pd.read_parquet(TRAIN_DIR / "train_ground_truth.parquet", columns=['source1_entity_id', 'matched_entity_ids'])
    
    # Filter ground truth directly to cached IDs (vectorized, no iterrows!)
    cache_keys = set(ce_cache.keys())
    gt = gt[gt['source1_entity_id'].isin(cache_keys)]
    
    gt_dict = {}
    for sid, matched in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        if pd.notna(matched) and str(matched).strip():
            gt_dict[sid] = [x.strip() for x in str(matched).split(',') if x.strip()]

    val_qids = [qid for qid in ce_cache if qid in gt_dict]
    truth = {qid: gt_dict[qid] for qid in val_qids}
    print(f"   Filtered to {len(val_qids):,} queries with GT matches in {time.time()-t1:.2f}s", flush=True)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    
    print("\n" + "=" * 78, flush=True)
    print(f"{'Threshold':<11} {'Precision':>11} {'Recall':>11} {'F0.5 Score':>13} {'Avg Matches/Q':>15}  {'Notes':<12}", flush=True)
    print("=" * 78, flush=True)

    results = []
    best_f, best_thr = -1, None

    for thr in thresholds:
        preds = {}
        total_preds = 0
        for qid in val_qids:
            cands = [cid for cid, score in ce_cache[qid] if score >= thr]
            preds[qid] = cands
            total_preds += len(cands)

        prec, rec, f_score = compute_metrics(preds, truth)
        avg_preds = total_preds / len(val_qids)
        results.append((thr, prec, rec, f_score, avg_preds))

        if f_score > best_f:
            best_f = f_score
            best_thr = thr

    for thr, prec, rec, f_score, avg_preds in results:
        notes = ""
        if thr == best_thr:
            notes = "[BEST F0.5]"
        elif thr == 0.20:
            notes = "<-- REQUESTED (0.20)"
        elif thr == 0.99:
            notes = "<-- REQUESTED (0.99)"

        print(f"{thr:<11.2f} {prec:>11.4f} {rec:>11.4f} {f_score:>13.4f} {avg_preds:>15.2f}  {notes}", flush=True)

    print("=" * 78, flush=True)
    print(f"Total time elapsed: {time.time()-t0:.2f}s", flush=True)

if __name__ == "__main__":
    main()
