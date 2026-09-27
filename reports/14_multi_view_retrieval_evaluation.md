# Multi-View Sparse Retrieval — Full Evaluation Report

Corpus: 108,589 targets | Queries: 2,500 | True matches: 8,674

---

## Root Cause — Why 0.68 vs 0.92?
The previous 0.92 F0.5 was a **final prediction** metric (top-1 binary match).
These metrics are **retrieval-stage** metrics evaluating Top-K candidates.
At K=50, the retriever returns 49 false positives per query by design —
the downstream re-ranker's job is to restore F0.5 back above 0.90.

---


### View: 1. Name Char TF-IDF
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.74003 | 0.53836 | 0.56940 |
| 10 | 0.79640 | 0.29454 | 0.33702 |
| 20 | 0.82655 | 0.15322 | 0.18304 |
| 40 | 0.85249 | 0.07911 | 0.09665 |
| 50 | 0.85840 | 0.06376 | 0.07825 |
| 100 | 0.87661 | 0.03255 | 0.04031 |

### View: 2. Name No-Space TF-IDF
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.71703 | 0.52131 | 0.55141 |
| 10 | 0.77758 | 0.28789 | 0.32938 |
| 20 | 0.81045 | 0.15045 | 0.17972 |
| 40 | 0.83765 | 0.07770 | 0.09492 |
| 50 | 0.84559 | 0.06275 | 0.07700 |
| 100 | 0.86659 | 0.03216 | 0.03983 |

### View: 3. Address Word TF-IDF
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.84671 | 0.61066 | 0.64672 |
| 10 | 0.89478 | 0.32962 | 0.37728 |
| 20 | 0.91523 | 0.16876 | 0.20165 |
| 40 | 0.92987 | 0.08583 | 0.10486 |
| 50 | 0.93381 | 0.06895 | 0.08463 |
| 100 | 0.94320 | 0.03480 | 0.04311 |

### View: 4. Name+Address Combo TF-IDF
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.87468 | 0.63589 | 0.67262 |
| 10 | 0.91212 | 0.33824 | 0.38692 |
| 20 | 0.92533 | 0.17155 | 0.20494 |
| 40 | 0.93702 | 0.08692 | 0.10619 |
| 50 | 0.94178 | 0.06986 | 0.08574 |
| 100 | 0.95238 | 0.03527 | 0.04369 |

### View: 5. Reverse Retrieval (S2+S3->S1)
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.19822 | 0.14220 | 0.15072 |
| 10 | 0.38093 | 0.13858 | 0.15878 |
| 20 | 0.64116 | 0.11790 | 0.14090 |
| 40 | 0.84970 | 0.07873 | 0.09618 |
| 50 | 0.87534 | 0.06495 | 0.07971 |
| 100 | 0.89278 | 0.03315 | 0.04106 |

### View: 6. High-Value Token Blocking
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.95479 | 0.68960 | 0.73016 |
| 10 | 0.98702 | 0.36441 | 0.41702 |
| 20 | 0.99259 | 0.18335 | 0.21908 |
| 40 | 0.99478 | 0.09189 | 0.11227 |
| 50 | 0.99527 | 0.07355 | 0.09027 |
| 100 | 0.99650 | 0.03682 | 0.04561 |

---

## Final Union (RRF) Evaluation

- **Avg candidates per S1 query:** 581.4
- **Corpus size:** 108,589
- **Candidate reduction ratio:** 0.0054 (99.46% eliminated)


### View: 7. Union RRF
| K | Recall@K | Precision@K | F0.5@K |
| :--- | :--- | :--- | :--- |
| 5 | 0.25039 | 0.17928 | 0.19008 |
| 10 | 0.42491 | 0.15499 | 0.17754 |
| 20 | 0.65170 | 0.12108 | 0.14463 |
| 40 | 0.83087 | 0.07707 | 0.09415 |
| 50 | 0.86123 | 0.06392 | 0.07845 |
| 100 | 0.90813 | 0.03371 | 0.04175 |

### Optimal K Search

Did not hit 99% recall within K=100. Best Recall@100 = 0.90813

ML Feature file (8 features per candidate pair): `C:\Users\Pavan\Downloads\ML_Challenge_26\data\processed\retrieval_ml_features.parquet`