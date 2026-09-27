# Dual-Pipeline Threshold Optimization (2.5k Queries)

**Architecture:**
- **TF-IDF Retrieval:** Uses Heavily Normalized Lexical Text (Sorted, Stopwords Removed, Suffixes Stripped)
- **Cross-Encoder Re-Ranking:** Uses Lightly Normalized Semantic Text (Grammar and Word Order Intact)

## Top 10 Configurations

| TF-IDF Threshold | Cross-Encoder Threshold | F0.5 Score |
| :--- | :--- | :--- |
| 0.60 | 0.950 | **0.94066** |
| 0.55 | 0.990 | **0.93940** |
| 0.60 | 0.900 | **0.93930** |
| 0.60 | 0.800 | **0.93802** |
| 0.60 | 0.990 | **0.93757** |
| 0.55 | 0.950 | **0.93670** |
| 0.60 | 0.500 | **0.93481** |
| 0.55 | 0.995 | **0.93447** |
| 0.50 | 0.995 | **0.93404** |
| 0.50 | 0.990 | **0.93386** |
