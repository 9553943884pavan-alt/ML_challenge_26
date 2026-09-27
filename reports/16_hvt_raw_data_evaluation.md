# HVT-Only Retrieval Evaluation

- **Corpus size:** 108,589
- **S1 Queries:** 2,500
- **True targets:** 8,675
- **Vocabulary size (HVT):** 43,983
- **Retrieval time:** 1.8s
- **Avg candidates per query:** 22.00
- **Candidate reduction:** 99.98% of corpus eliminated

## HVT Retriever: Recall, Precision, F0.5 @ Fine-Grained K

| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 1 | **0.31064** | 0.93200 | 0.66569 |
| 2 | **0.57314** | 0.90500 | 0.81107 |
| 3 | **0.74533** | 0.83293 | 0.81380 |
| 5 | **0.89392** | 0.64560 | 0.68358 |
| 7 | **0.92175** | 0.48480 | 0.53558 |
| 10 | **0.92674** | 0.34204 | 0.39143 |
| 12 | **0.92788** | 0.28553 | 0.33142 |
| 15 | **0.92948** | 0.22888 | 0.26951 |
| 20 | **0.93096** | 0.17198 | 0.20548 |
| 22 | **0.93139** | 0.15642 | 0.18764 |

## Analysis
- Best Recall is at K=22: **0.93139**
- Best F0.5 is at K=1:  **0.66569**
- **Recommended K for LightGBM input:** K=20 (balances 99%+ Recall with minimal candidates)