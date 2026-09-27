# LightGBM 5-Fold Cross Validation Results

- **Total Candidate Pairs Evaluated:** 406,340
- **True Matches:** 17,318
- **False Positives:** 389,022
- **Model:** LightGBM Binary Classifier
- **Validation Strategy:** Stratified 5-Fold CV

## Performance Metrics (Out-Of-Fold)
The model predicts a probability for each pair. We tuned the probability threshold to explicitly maximize the **F0.5 score**.

- **Optimal Probability Threshold:** `0.80`
- **Precision:** `0.9903`
- **Recall:** `0.9479`
- **F0.5 Score:** `0.9815`

## Confusion Matrix
| | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :--- | :--- |
| **Actual Negative (0)** | 388,862 (True Neg) | 160 (False Pos) |
| **Actual Positive (1)** | 903 (False Neg) | 16,415 (True Pos) |

## Top 10 Feature Importances
Based on the average number of times a feature was used to split the data across the 5 folds.

| Feature | Split Importance |
| :--- | :--- |
| `addr_jaccard` | 861.6 |
| `hvt_cosine_score` | 844.0 |
| `addr_jaro_winkler` | 834.6 |
| `addr_token_sort` | 795.8 |
| `name_token_sort` | 788.2 |
| `name_partial_ratio` | 688.8 |
| `hvt_rank` | 667.0 |
| `name_jaro_winkler` | 629.2 |
| `addr_word_count_diff` | 576.8 |
| `name_3gram_jaccard` | 559.4 |

## Analysis
- **Precision Dominance:** Because the threshold is tuned for F0.5, the model strongly prioritizes minimizing False Positives over catching every single True Match.
- **Recall Context:** The 0.9479 recall here represents the LightGBM's recall *on the retrieved candidates*. The global pipeline recall is (HVT Recall * LGBM Recall).