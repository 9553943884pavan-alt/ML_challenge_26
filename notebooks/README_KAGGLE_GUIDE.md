# Kaggle Execution Guide — Submission Generator (`matching_results.tsv`)

This guide walks you through running the Business Entity Resolution pipeline on **Kaggle's free GPU compute** (Tesla T4 × 2 or Tesla P100) to generate the official `matching_results.tsv` submission file.

---

## 1. Pipeline Overview

The notebook implements our competition-tested **2-Stage Hybrid Architecture**:
* **Stage 1 (Blocking / Retrieval):** Streaming Character N-Gram TF-IDF (2–4 n-grams, 50,000 features, threshold > 0.40, Top-15 candidate capping) achieving **~96.7% recall**.
* **Stage 2 (Neural Re-ranking):** Multilingual Cross-Encoder (**`BAAI/bge-reranker-v2-m3`**, 560M parameters, FP16 GPU inference, threshold = 0.80) achieving peak **0.9431 Macro F0.5**.
* **Stage 3 (Validation & Export):** Strictly formatted `matching_results.tsv` + `submission.zip` with built-in rule validation.

---

## 2. Step-by-Step Kaggle Setup

### Step 1: Upload Dataset to Kaggle
1. Log in to [Kaggle](https://www.kaggle.com/) and go to **Datasets &rarr; New Dataset**.
2. Create a new dataset (e.g., `amazon-ml-challenge-test-data`).
3. Upload either:
   - **Parquet files (Recommended, faster loading):**
     * `test_source1.parquet`
     * `test_source2.parquet`
     * `test_source3.parquet`
   - **OR raw TSV files:**
     * `test_source1.tsv`
     * `test_source2.tsv`
     * `test_source3.tsv`
   - *(Optional bonus)* If you already generated `test_corpus_tfidf_matrix.npz`, upload it too! The notebook will auto-detect and load it in 3 seconds.

### Step 2: Upload / Import the Notebook
1. Go to **Code &rarr; New Notebook**.
2. Click **File &rarr; Import Notebook** and upload `kaggle_generate_submission.ipynb` from this repo (`notebooks/kaggle_generate_submission.ipynb`).

### Step 3: Configure Kaggle Settings
In the right-hand panel of the Kaggle notebook interface:
1. **Accelerator:** Select **GPU T4 x 2** (preferred) or **GPU P100**.
2. **Internet:** Turn toggle **ON** (needed for `!pip install sentence-transformers` and downloading the `BAAI/bge-reranker-v2-m3` weights from HuggingFace).
3. **Session Options &rarr; Persistence:** Select **Files only** or **Variables and Files**.
4. Click **+ Add Input** &rarr; Search for your uploaded test dataset &rarr; Click **Add**.

---

## 3. Running the Pipeline

### Quick Verification (Debug Mode)
In **Cell 2**, `DEBUG_MODE` is provided:
```python
DEBUG_MODE = True       # Runs on 1,000 queries in ~30 seconds
DEBUG_SAMPLE_SIZE = 1000
```
Run all cells once with `DEBUG_MODE = True`. This quickly verifies:
- Data auto-discovery succeeds.
- Text cleaning and TF-IDF blocking run smoothly.
- GPU Cross-Encoder loads into VRAM and runs batch scoring.
- Submission formatting and validator pass with 0 errors.

### Full Competition Run
Once verified, change Cell 2 to:
```python
DEBUG_MODE = False      # Full 1.73M queries execution
```
Click **Run All** (or **Save Version &rarr; Save & Run All (Commit)** to let it run in the background).

---

## 4. Key Engineering Features (RULES.md Compliance)

| Rule | How It Is Implemented in the Notebook |
| :--- | :--- |
| **Rule 1 (Prototype First)** | `DEBUG_MODE` flag runs a 1,000-query sanity check in 30 seconds before scaling up. |
| **Rule 2 (Disk Caching & Resumable Checkpoints)** | Caches TF-IDF matrices to `.npz`. Cross-Encoder inference saves checkpoints every 20,000 queries. If interrupted, simply re-running resumes from the last checkpoint without re-scoring! |
| **Rule 4 (Efficient dtypes & Parquet)** | Automatic loading of Parquet or low-RAM TSV with categorical encoding. |
| **Rule 5 & 6 (Batched GPU & FP16)** | Inference uses `CE_BATCH_SIZE = 256` with FP16 half-precision (`model.half()`), utilizing full 16GB VRAM. |
| **Rule 7 (Chunked & Streaming RAM Protection)** | TF-IDF sparse matmul and candidate extraction run in chunks of 5,000 queries. DataFrames are cleared with `gc.collect()` to stay safely within Kaggle's 16GB RAM. |
| **Rule 8 & 9 (Curated & Vectorized Text)** | NaNs are filled with `""` so missing addresses don't inject noise. Vectorized string operations (no `iterrows()`). |
| **Rule 10 (Strict Competition Constraints)** | Strict TSV format (`source1_entity_id\tmatched_entity_ids`), exactly 1 row per S1 query, only S2/S3 IDs allowed, no duplicates, open-set country concatenation. |

---

## 5. Downloading the Output File

Once the notebook finishes Cell 8, 9, and 10:
1. **Interactive Notebook:** Click the blue direct download link generated in the final cell (`TSV File` or `ZIP Archive`).
2. **Kaggle Sidebar:** Look at the right sidebar under **Data &rarr; Output &rarr; /kaggle/working**.
   - Right-click `matching_results.tsv` (or `submission.zip`) &rarr; **Download**.
3. **Commit & Run Background Job:** Go to the notebook version page, open the **Output** tab, and click **Download** next to `matching_results.tsv`.
