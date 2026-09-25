# Cross-Encoder Optimization Report

**Date:** 2026-09-25

## 1. Baseline Performance (TF-IDF Only)
- **Model:** TF-IDF (Char N-Grams 2-4)
- **Threshold:** 0.60
- **Macro F0.5 Score:** `0.9234`

## 2. Initial Cross-Encoder (MiniLM)
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Output:** Raw logits passed through `sigmoid`
- **Optimal Threshold:** `0.995`
- **Macro F0.5 Score:** `0.9301`
- **Issue:** As an English-only model, it struggled with "Trap 5" (Hidden Cross-Language Matches like `televisor` vs `tv`), causing false negatives on valid non-English pairs.

## 3. Improved Cross-Encoder (BAAI BGE-M3)
- **Model:** `BAAI/bge-reranker-v2-m3` (Multilingual)
- **Architecture:** 560M parameters, natively supports 100+ languages. Outputs calibrated similarity scores directly.

## 4. Full Validation Scale (10,000 Queries Evaluation)

### Stage 1: Loosening the TF-IDF Bottleneck (Threshold > 0.40)
- **TF-IDF Threshold:** `0.40`
- **Stage 1 Recall:** **`96.69%`** (Up from 86.87% at threshold 0.60)
- **Candidate Pairs Evaluated by Cross-Encoder:** **`434,927 pairs`** (Average 43.5 candidates / query)

### Stage 2: Cross-Encoder Threshold Sweep on 10,000 Validation Set
Because the neural predictions are fully cached on disk (`cross_encoder_preds_cache_10k_val_tfidf040.pkl`), we swept the entire decision threshold spectrum:

| Cross-Encoder Threshold | Macro F0.5 Score |
| :--- | :--- |
| 0.05 | 0.7739 |
| 0.10 | 0.8219 |
| 0.15 | 0.8507 |
| 0.20 | 0.8705 |
| 0.25 | 0.8846 |
| 0.30 | 0.8968 |
| 0.40 | 0.9146 |
| 0.50 | 0.9270 |
| 0.60 | 0.9356 |
| 0.70 | 0.9412 |
| **0.80 (Optimal Peak)** | **`0.9431`** |
| 0.90 | 0.9380 |
| 0.95 | 0.9279 |

### Key Insights & Findings:
1. **Recall Improvement:** Loosening TF-IDF from 0.60 to 0.40 boosted candidate recall to **96.69%**, successfully feeding true matches to the neural model.
2. **Precision Weighting in F0.5:** Because F0.5 weights Precision 2× over Recall, lower thresholds (like 0.20) allow too many false positives on ambiguous pairs (Macro F0.5 = 0.8705).
3. **Global Optimum:** The peak Macro F0.5 score is achieved at **Cross-Encoder Threshold = `0.80`**, reaching a record **`0.9431`** Macro F0.5 on the 10,000 validation split.
