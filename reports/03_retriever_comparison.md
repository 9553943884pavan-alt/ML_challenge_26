# Blocking Retrievers Benchmark: BM25 vs TF-IDF

**Date:** 2026-09-25
**Scope:** Comparing pure word-level BM25 (`rank_bm25` library) against Character N-Gram TF-IDF.
**Metric:** Recall@10 (Did the true match appear in the top 10 candidates generated?)

## Benchmark Results

| Algorithm | Retrieval Time (1000 Queries) | Recall@10 Score |
| :--- | :--- | :--- |
| **Exact BM25** (rank_bm25) | 168.29 seconds | 99.89% |
| **TF-IDF Char N-Grams** | 126.57 seconds | 99.89% |

## Conclusion & Architecture Decision
Both algorithms performed exceptionally well, successfully retrieving the true matches into the Top 10 candidate pool **99.89%** of the time. 

However, the architecture decision leans heavily toward **TF-IDF Character N-Grams** for the full scale run:
1. **Speed & Scalability:** The `rank_bm25` library is written in pure Python and requires double `for`-loops to calculate scores. It took almost 3 minutes just to search a 50k corpus. When we scale this up to the 12 Million record dataset, `rank_bm25` will take days to run. TF-IDF uses highly optimized C++ SciPy sparse matrices and is roughly 25% faster even in an unoptimized loop (and can be batched to run in seconds).
2. **Typo Resilience:** While both hit 99% on this split, standard BM25 operates strictly on whole words. As we found in the EDA, Source 2 is full of typos. Character N-Grams will mathematically preserve overlap on misspelled tokens where standard BM25 yields `0.0`.

**Verdict:** We will use the optimized TF-IDF Character N-Gram logic to generate our final `candidate_pairs.tsv` during the blocking stage.
