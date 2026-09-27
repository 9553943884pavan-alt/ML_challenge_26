# HVT-Only Retrieval Evaluation

- **Corpus size:** 117,184
- **S1 Queries:** 5,000
- **True targets:** 17,362
- **Vocabulary size (HVT):** 44,778
- **Retrieval time:** 5.0s
- **Avg candidates per query:** 22.00
- **Candidate reduction:** 99.98% of corpus eliminated

## HVT Retriever: Recall, Precision, F0.5 @ Fine-Grained K

| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 1 | **0.33232** | 0.99384 | 0.71084 |
| 2 | **0.60916** | 0.96211 | 0.86220 |
| 3 | **0.79805** | 0.89012 | 0.87004 |
| 5 | **0.95387** | 0.68690 | 0.72763 |
| 7 | **0.98269** | 0.51569 | 0.56985 |
| 10 | **0.98706** | 0.36353 | 0.41610 |
| 12 | **0.98914** | 0.30362 | 0.35247 |
| 15 | **0.99046** | 0.24326 | 0.28649 |
| 20 | **0.99198** | 0.18277 | 0.21841 |
| 22 | **0.99242** | 0.16624 | 0.19944 |

## Analysis
- Best Recall is at K=22: **0.99242**
- Best F0.5 is at K=1:  **0.71084**
- **Recommended K for LightGBM input:** K=20 (balances 99%+ Recall with minimal candidates)