# Prototype 1: Lexical Retrieval (TF-IDF Character N-Grams)

**Date:** 2026-09-25
**Dataset:** Mini-Split (1,000 Source 1 Queries vs 103,489 noisy Corpus)
**Metric:** Macro F_0.5

## Execution Results
*   **Threshold Used:** `0.75`
*   **F_0.5 Score:** `0.8403`

## Performance Analysis
The model achieved an extraordinarily high **0.8403** F_0.5 score using purely lexical retrieval without a deep learning Cross-Encoder. 

**Why did it score so high?**
By adhering to Rule 9 and using vectorized Character N-Grams (`char_wb`), the model mathematically broke down the strings into 3-letter combinations. This made it entirely immune to the noise we discovered during EDA (e.g., `Ltd` vs `Limited`).

## Error Analysis (Weaknesses to fix in Stage 2)
While the score is excellent for a prototype, the error analysis logs revealed two major failure points that demand a Cross-Encoder for the final pipeline:

1.  **False Merges on Chains (Precision Loss):** The model falsely matched franchises. For example, it thought `Starbucks, 1st Ave, NY` and `Starbucks, 2nd Ave, NY` were the same entity because 95% of the text overlaps. A Lexical Retriever cannot understand that "1st" and "2nd" are deal-breakers.
2.  **Singleton Failures:** Because we used a static threshold (`0.75`), if a query had no true matches in the corpus, the model sometimes still found a random business that scored `0.76` and falsely merged it, instantly scoring 0.0 for that entity due to the harsh singleton penalty.

**Next Step:** The comparison script against the official `rank_bm25` library is still running, which will prove if we should swap this logic for exact word-matching!
