# Train Data Exploratory Data Analysis (EDA) Report

**Date:** 2026-09-25
**Scope:** Initial analysis of `train_source1.parquet`, `train_source2.parquet`, and `train_source3.parquet`.

## 1. Dataset Dimensions & Memory Footprint

After successfully downcasting categorical columns (Rule 4) and converting to Parquet format, the training dataset dimensions are as follows:

| Source | Rows | Columns | Memory Usage |
| :--- | :--- | :--- | :--- |
| **Source 1** (Reference) | 2,206,821 | 4 | ~496 MB |
| **Source 2** | 5,034,616 | 4 | ~1,171 MB |
| **Source 3** | 5,285,603 | 4 | ~1,250 MB |
| **Total** | **12,527,040** | - | **~2.9 GB** |

*Insight:* With over 12.5 million rows in training alone, O(N^2) comparison (comparing every Source 1 entity against every Source 2/3 entity) is mathematically impossible. This strictly mandates a highly optimized **Candidate Generation (Blocking)** phase using fast indexing (e.g., FAISS or Elasticsearch).

## 2. Null Value Distribution

| Column | Source 1 | Source 2 | Source 3 |
| :--- | :--- | :--- | :--- |
| `entity_id` | 0 | 0 | 0 |
| `business_name` | 0 | 2 | 13 |
| `business_address` | 0 | 168,967 | 175,916 |
| `country` | 0 | 0 | 0 |

## 3. Key Findings & Data Irregularities

1. **The Golden Reference (Source 1):** 
   Source 1 contains zero null values across all features. It serves as a perfectly clean anchor dataset for our queries.
2. **Massive Address Degradation in Targets:** 
   Combined, Source 2 and Source 3 are missing physical addresses for over **344,883 businesses**. 
3. **Trace Name Degradation:**
   15 records completely lack a business name, making them almost impossible to resolve purely on textual semantic meaning without an address.

## 4. Pipeline Action Items & Handling Strategy

To adhere to the challenge rules (specifically *Rule 8: Curate high-quality data*):
*   **Null Handling Strategy:** We will strictly fill all `NaN` values with an empty string `""`.
*   **Why?** Imputing with placeholder text (e.g., "Unknown Address") will cause our retrieval algorithms (BM25/TF-IDF) to falsely cluster unrelated businesses based on the artificial overlap of the word "Unknown". Empty strings enforce zero mathematical overlap, forcing the model to rely correctly on the `business_name` when the address is degraded.
*   **Vectorization Prep:** All text columns will be lowercased and stripped of excessive whitespace using Pandas vectorized `.str` methods prior to embedding generation.
