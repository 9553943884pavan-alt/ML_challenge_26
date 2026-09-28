# geek_squads - Business Entity Resolution Pipeline

This repository contains the end-to-end code to reproduce our submission for the ML Challenge 2026.

## 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 2. Directory Structure
Before running the scripts, ensure your dataset is placed at the root level relative to the execution context:
```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## 3. End-to-End Reproduction

### Step A: Training the Model & Finding the Optimal Threshold
Run the main training pipeline. This script will load the train dataset, generate TF-IDF blocks, compute features, train the LightGBM classifier, sweep for the optimal threshold, and save the artifacts (vectorizers, LightGBM model, optimal threshold JSON) to the `kaggle_working/` directory.

```bash
python src/pipeline/kaggle_main.py
```

### Step B: Inference & Generating the Final Submission
Once training completes and artifacts are generated, run the test inference script. This script loads the test dataset, extracts candidates using the pre-trained vectorizer, applies the trained LightGBM model, and writes the results to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

```bash
python src/pipeline/kaggle_test_main.py
```

## 4. Hardware Constraints
The pipeline is fully memory-optimized to run within a Kaggle Notebook environment (30GB RAM, 2 CPUs). It will dynamically leverage `cupy` (GPU) or `sparse_dot_topn` (Cython) depending on the available hardware.
