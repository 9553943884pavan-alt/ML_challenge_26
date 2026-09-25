# Comprehensive Insights & Hidden Traps 
**Amazon ML Challenge 2026 Dataset**

This document summarizes all the hidden traps, anomalies, and insights we discovered through our deep Exploratory Data Analysis (EDA) on both the Training and Test sets.

## 1. The "Hidden Character" Trap (Non-ASCII Anomaly)
*   **The Discovery:** While Source 1 is perfectly clean (0.00% non-ASCII), exactly **15% to 19%** of the names in Source 2 and Source 3 contain non-ASCII characters (e.g., emojis, zero-width spaces, cyrillic, French accents).
*   **The Danger:** If a standard word tokenizer (like built-in `rank_bm25` or standard BERT) is used, these hidden characters can cause the tokenizer to fail or misalign, resulting in a 0% match for otherwise identical businesses.
*   **Our Mitigation:** By using **Character N-Grams** (sliding 3-letter windows) instead of Word N-Grams, our pipeline completely ignores tokenizer failures and mathmatically captures the string overlap.

## 2. The "ID Recycling" Trap (Exact Duplicates)
*   **The Discovery:** We found that Source 2 and Source 3 contain over **38,000 exact duplicates** (businesses with the exact same name and address but *different* `entity_id`s). 
*   **The Danger:** Many teams will use a `Top-1` retrieval strategy (e.g., `return candidate with max score`). Because there are exact duplicates, a Top-1 strategy will randomly pick only one of the true matches and completely miss the others, destroying Recall.
*   **Our Mitigation:** We abandoned "Top K" logic and used **Probability Thresholding** (e.g., `Score > 0.995`). This ensures that if there are 5 exact duplicates, we retrieve all 5 of them!

## 3. The "Placeholder" Trap (Missing Addresses)
*   **The Discovery:** Source 2 and Source 3 are completely missing addresses for hundreds of thousands of records. 
*   **The Danger:** If missing addresses are filled with a default string like `"Unknown"`, `"NaN"`, or `"N/A"`, the clustering model will artificially learn that all missing businesses are semantically related (a massive False Positive trap).
*   **Our Mitigation:** We proactively fill all missing values with an empty string `""` so they contribute zero weight to the similarity score.

## 4. The "Open Set" Country Trap
*   **The Discovery:** The Training set only contains businesses in `US` and `India`. The documentation warned that the Test Set contains a new country. Our EDA confirmed this country is exactly `"France"`.
*   **The Danger:** Hardcoding rules (e.g., `if country == 'US'`) will instantly break the pipeline on the test set. 
*   **Our Mitigation:** We concatenate the country directly into the raw string `name | address | country`. The TF-IDF matrix treats "france" as just another categorical character pattern, requiring zero hardcoded logic.

## 5. The "Cross-Language" Trap (The Impossible Cases)
*   **The Discovery:** Our head-to-head Error Analysis revealed that the Ground Truth contains matches across different languages. For example, the English string `"dream construction limited | hyderabad"` is marked as a true match to the Telugu string `"డ్రీమ్ కన్స్ట్రక్షన్ లిమిటెడ్"`.
*   **The Danger:** Pure lexical math (TF-IDF) and English-trained Neural Networks (MS-MARCO) will always score this as a 0% match because they cannot natively translate Telugu to English. 
*   **Our Mitigation:** None required for a 72-hour hackathon. Fixing this requires passing 12 Million records through a Multilingual Translation Model, which is too computationally expensive. We accept this as the "ceiling" of our F0.5 score.
