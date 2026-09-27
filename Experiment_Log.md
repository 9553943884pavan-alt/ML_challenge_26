# Experiment Log & Iterative Architecture Development
**Team:** geek_squads  
**Project:** Business Entity Resolution (ML Challenge 2026)  

This document chronicles the step-by-step experimentation, failures, and breakthroughs that led to our final 3-Way Acceleration Engine architecture. Over the course of the challenge, we generated 19 distinct analysis reports evaluating everything from baseline algorithms to advanced neural cross-encoders.

---

## Phase 1: Exploratory Data Analysis & Baseline Prototyping
*(Reports: 01_train_data_eda, 04_test_data_eda, 17_feature_eda)*

**Objective:** Understand the 5 Million corpus scale and noise patterns.  
**Key Discoveries:**
1. **The Non-ASCII Anomaly:** Found that 15%-19% of names in Source 2 and 3 contained non-ASCII noise (emojis, zero-width spaces, regional scripts). Standard word tokenizers completely failed on these.
2. **Open-Set Target Leakage:** The test set contained a completely new country (`France`) not present in the training set (`US`, `India`), invalidating any hardcoded categorical splits.
3. **ID Recycling:** Discovered over 38,000 exact physical duplicates in S2 and S3 operating under different `entity_id`s. Standard "Top-1" retrieval approaches instantly crippled recall because they randomly selected only one of the twins.

**Action Taken:** Moved to Sub-word / Character N-Gram TF-IDF Vectorization to gracefully bypass tokenizer crashes, and abandoned hardcoded rules for global text concatenation (`name | address | country`).

---

## Phase 2: Retrieval Engine Comparisons
*(Reports: 02_retriever_prototype, 03_retriever_comparison, 08_final_insights_and_traps)*

**Objective:** Select the most memory-efficient retrieval algorithm to reduce 8.5 Trillion comparisons.  
**Experiments Conducted:**
- **BM25 & ElasticSearch:** Exceeded memory overhead constraints during index generation.
- **Neural Bi-Encoders (SentenceTransformers):** Embedding 5 Million strings caused out-of-memory crashes and failed to capture minute differences in street numbers.
- **Highly Variable TF-IDF (HVT):** Achieved the best balance of speed and memory. 
  - *Breakthrough:* We noticed that setting `max_df = 0.02` lost critical identifiers like "Pharmacy" and "Hospital". We explicitly raised `max_df = 0.15` which retained common words. 
  - *The Trap:* High `max_df` creates highly dense matrices, causing SciPy `dot()` to OOM. 
  - *The Solution:* We engineered the **3-Way Acceleration Engine**, offloading dense overlap math to `CuPy` (GPU) or `sparse_dot_topn` (Cython C++).

---

## Phase 3: The Neural Cross-Encoder Attempt
*(Reports: 06_cross_encoder_optimization, 07_error_comparison, 14_multi_view_retrieval)*

**Objective:** Use MS-MARCO Cross-Encoders to score candidate pairs.  
**Key Discoveries:**
- **The Branch Office Trap:** The Cross-Encoder learned excellent semantic flexibility but became *too* fuzzy. It would score `Bank | 100 Main St` and `Bank | 102 Main St` with a 0.95 probability, entirely ignoring the differing street numbers and causing catastrophic False Positives.
- **Cross-Script Trap:** English-trained models failed instantly when matching English query names to regional Indic scripts present in the dataset (e.g., Telugu, Hindi).

**Action Taken:** Neural Cross-Encoders were scrapped for the final inference pipeline due to computational cost (1.7M pairs would exceed the 12-hour timeout) and precision leakage. 

---

## Phase 4: LightGBM Meta-Classifier & Feature Engineering
*(Reports: 12_dual_pipeline_sweep, 13_fine_grained_sweep, 18_lgbm_5fold_cv, 19_lgbm_10k_cv)*

**Objective:** Build an ultra-fast, precision-heavy classifier to evaluate the Top-30 TF-IDF candidates.  
**Experiments Conducted:**
- Generated highly targeted string features using `RapidFuzz` (Token Sort Ratio) and `Jellyfish` (Jaro-Winkler) for Names and Addresses independently.
- Extracted deterministic numeric equality flags (e.g., matching street numbers extracted via Regex).
- **Model Training:** Trained a LightGBM Classifier over 10,000 Cross-Validation splits. We recognized that the evaluation metric (`F_0.5`) heavily penalizes False Positives (Precision is weighted 4x over Recall).
- **Dynamic Thresholding:** Instead of a static 0.50 cutoff, our pipeline programmatically sweeps LightGBM probabilities (`0.1` to `0.9`) on a 20% validation chunk to identify the mathematically perfect cutoff for F_0.5 optimization.

---

## Conclusion: The Ultimate Architecture (Report 09)
By merging our findings, we finalized our pipeline:
1. **Blocking:** `max_df=0.15` TF-IDF with GPU/Cython dot-product chunking.
2. **Feature Engineering:** String similarity metrics calculated only on the Top-30 retrieved candidates.
3. **Classification:** Memory-safe (sub-batching) LightGBM inference, stripping heavy feature columns prior to predicting to maintain a sub-1GB RAM footprint on Kaggle.

*All findings directly shaped the final output format in `matching_results.tsv` and `candidate_pairs.tsv`.*
