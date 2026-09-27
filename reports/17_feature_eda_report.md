# Exploratory Data Analysis: Pairwise Features

**Dataset Size:** 100,000 pairs
**True Matches (1):** 17,227
**False Positives (0):** 82,773
**Imbalance Ratio:** 1 : 4.80

---

## 1. Mutual Information (Feature Importance)
Mutual Information (MI) measures the dependency between a feature and the target label. Higher means more predictive power.

| Feature | Mutual Information Score |
| :--- | :--- |
| `hvt_cosine_score` | 0.3949 |
| `addr_jaccard` | 0.2924 |
| `hvt_rank` | 0.2922 |
| `addr_jaro_winkler` | 0.2894 |
| `addr_token_sort` | 0.2870 |
| `name_levenshtein_ratio` | 0.2650 |
| `name_token_sort` | 0.2641 |
| `name_jaro_winkler` | 0.2641 |
| `name_3gram_jaccard` | 0.2524 |
| `name_token_set` | 0.2521 |
| `name_partial_ratio` | 0.2450 |
| `name_jaccard` | 0.2240 |
| `name_prefix_match` | 0.1344 |
| `exact_name_match` | 0.1318 |
| `exact_addr_match` | 0.0583 |
| `name_word_count_diff` | 0.0458 |
| `addr_word_count_diff` | 0.0292 |
| `country_match` | 0.0208 |

---

## 2. Univariate Analysis (Class Distribution)
Averages for True Matches (1) vs False Positives (0) to show how well each feature separates the classes.

| Feature | Mean (True Matches) | Mean (False Positives) | Separation (Absolute Diff) |
| :--- | :--- | :--- | :--- |
| `hvt_rank` | 2.7558 | 12.1117 | **9.3559** |
| `addr_word_count_diff` | 1.4094 | 2.7914 | **1.3819** |
| `name_3gram_jaccard` | 0.7059 | 0.0995 | **0.6065** |
| `name_prefix_match` | 0.6903 | 0.0898 | **0.6005** |
| `name_jaccard` | 0.6613 | 0.1022 | **0.5591** |
| `name_word_count_diff` | 0.8355 | 1.3943 | **0.5587** |
| `hvt_cosine_score` | 0.8118 | 0.2922 | **0.5196** |
| `addr_jaccard` | 0.6602 | 0.1845 | **0.4757** |
| `name_token_set` | 0.8708 | 0.4100 | **0.4608** |
| `name_token_sort` | 0.8264 | 0.3739 | **0.4525** |
| `name_levenshtein_ratio` | 0.8264 | 0.3739 | **0.4525** |
| `exact_name_match` | 0.4494 | 0.0081 | **0.4413** |
| `name_partial_ratio` | 0.8703 | 0.4796 | **0.3907** |
| `addr_token_sort` | 0.8313 | 0.4571 | **0.3741** |
| `name_jaro_winkler` | 0.8800 | 0.5627 | **0.3173** |
| `addr_jaro_winkler` | 0.8586 | 0.5651 | **0.2935** |
| `exact_addr_match` | 0.1824 | 0.0001 | **0.1822** |
| `country_match` | 1.0000 | 0.9654 | **0.0346** |

---

## 3. Multivariate Analysis (Feature Redundancy)
Highly correlated features (>0.85) provide redundant information. LightGBM handles this well, but it's good to know which features duplicate each other.

| Feature A | Feature B | Pearson Correlation |
| :--- | :--- | :--- |
| `name_token_sort` | `name_levenshtein_ratio` | 1.0000 |
| `name_jaccard` | `name_3gram_jaccard` | 0.9645 |
| `name_token_set` | `name_partial_ratio` | 0.9631 |
| `name_token_sort` | `name_token_set` | 0.9587 |
| `name_token_set` | `name_levenshtein_ratio` | 0.9587 |
| `addr_jaro_winkler` | `addr_token_sort` | 0.9459 |
| `name_token_sort` | `name_partial_ratio` | 0.9425 |
| `name_partial_ratio` | `name_levenshtein_ratio` | 0.9425 |
| `name_jaro_winkler` | `name_token_sort` | 0.9362 |
| `name_jaro_winkler` | `name_levenshtein_ratio` | 0.9362 |
| `name_token_sort` | `name_3gram_jaccard` | 0.9253 |
| `name_levenshtein_ratio` | `name_3gram_jaccard` | 0.9253 |
| `name_token_sort` | `name_jaccard` | 0.8937 |
| `name_levenshtein_ratio` | `name_jaccard` | 0.8937 |
| `name_jaro_winkler` | `name_token_set` | 0.8901 |
| `name_token_set` | `name_3gram_jaccard` | 0.8886 |
| `name_token_set` | `name_jaccard` | 0.8808 |
| `name_jaro_winkler` | `name_partial_ratio` | 0.8777 |
| `name_partial_ratio` | `name_3gram_jaccard` | 0.8688 |
| `name_jaro_winkler` | `name_3gram_jaccard` | 0.8634 |
| `addr_token_sort` | `addr_jaccard` | 0.8506 |