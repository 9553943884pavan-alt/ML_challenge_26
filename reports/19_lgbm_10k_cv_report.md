# LightGBM 5-Fold Cross Validation Results

- **Total Candidate Pairs Evaluated:** 458,795
- **True Matches:** 34,527
- **False Positives:** 424,268
- **Model:** LightGBM Binary Classifier
- **Validation Strategy:** Stratified 5-Fold CV

## Performance Metrics (Out-Of-Fold)
The model predicts a probability for each pair. We tuned the probability threshold to explicitly maximize the **F0.5 score**.

- **Optimal Probability Threshold:** `0.77`
- **Precision:** `0.9888`
- **Recall:** `0.9500`
- **F0.5 Score:** `0.9808`

## Confusion Matrix
| | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :--- | :--- |
| **Actual Negative (0)** | 423,895 (True Neg) | 373 (False Pos) |
| **Actual Positive (1)** | 1,725 (False Neg) | 32,802 (True Pos) |

## Top 10 Feature Importances
Based on the average number of times a feature was used to split the data across the 5 folds.

| Feature | Split Importance |
| :--- | :--- |
| `hvt_cosine_score` | 1355.0 |
| `addr_token_sort` | 1311.8 |
| `addr_jaccard` | 1286.4 |
| `addr_jaro_winkler` | 1263.0 |
| `name_partial_ratio` | 1114.2 |
| `name_token_sort` | 1111.4 |
| `hvt_rank` | 995.0 |
| `name_jaro_winkler` | 963.0 |
| `addr_word_count_diff` | 853.0 |
| `name_3gram_jaccard` | 817.4 |

## Analysis
- **Precision Dominance:** Because the threshold is tuned for F0.5, the model strongly prioritizes minimizing False Positives over catching every single True Match.
- **Recall Context:** The 0.9500 recall here represents the LightGBM's recall *on the retrieved candidates*. The global pipeline recall is (HVT Recall * LGBM Recall).