# Project Rules: Amazon ML Challenge 2025

These core engineering and machine learning principles must be strictly followed throughout the entire lifecycle of this project.

---

## 1. Never Develop a Prototype on the Full Dataset
- Always develop, debug, and validate pipelines on a small stratified or representative subset (e.g., 500 – 2,000 samples or `sample_test.csv`).
- Only run training, evaluation, and full inference jobs on the entire dataset once the end-to-end pipeline executes without errors.

---

## 2. Precompute Expensive Operations, Cache Them to Disk, and Reuse Them
- Precompute heavy operations such as text parsing, regex extraction, tokenization, image downloading, and feature extraction.
- Persist intermediate features and embeddings to disk (e.g., in a dedicated `cache/` or `data/processed/` directory).
- Implement cache-check guards (e.g., `if os.path.exists(cache_file): load() else: compute_and_save()`) to avoid redundant recomputations.

---

## 3. Reduce Embedding Dimensionality Before Feeding into ML Models
- High-dimensional embeddings (e.g., 768-d text or 1024-d vision embeddings) should not be fed directly to downstream models (such as LightGBM, CatBoost, or Ridge/MLP) without compression.
- Apply dimensionality reduction techniques such as:
  - Principal Component Analysis (PCA) or TruncatedSVD
  - Learned projection layers / linear bottleneck layers
  - Feature selection or variance thresholding
- Measure reconstruction/variance ratio or downstream CV score to retain informative signals while cutting memory and compute footprint.

---

## 4. Use Efficient File Formats and Appropriate dtypes
- Avoid reading and writing uncompressed, repetitive `.csv` files during pipeline execution.
- Use columnar, compressed formats like **Apache Parquet** (`.parquet`) or **Feather** (`.feather`).
- Downcast numeric types:
  - Floating point: `float64` $\to$ `float32` or `float16`
  - Integers: `int64` $\to$ `int32`, `int16`, or `int8`
  - Categoricals: Convert low-cardinality strings to `category` dtype or integer label encodings.

---

## 5. Parallelize I/O-Bound and Independent Processing Wherever Possible
- For network I/O (e.g., downloading product images from `image_link` URLs):
  - Use thread pools (`concurrent.futures.ThreadPoolExecutor` or `asyncio` / `aiohttp`).
- For CPU-heavy independent tasks (e.g., text sanitization, entity parsing across millions of records):
  - Use process pools (`multiprocessing` / `joblib.Parallel` / `pandarallel`).

---

## 6. Batch GPU Operations Instead of Processing One Sample at a Time
- Never call deep learning models (PyTorch, Hugging Face transformers, Timm, CLIP, ResNet, etc.) inside a single-sample loop.
- Use `torch.utils.data.DataLoader` with appropriate `batch_size`, `pin_memory=True`, and `num_workers`.
- Enable mixed precision (`torch.cuda.amp.autocast()` / FP16/BF16) during GPU feature extraction and inference.

---

## 7. Use Chunked / Streaming Processing Instead of Loading Everything into RAM
- When processing large files (`train.csv`, `test.csv` ~70MB+ or multi-gigabyte extracted datasets), do not load unnecessary columns or unneeded rows all at once into memory if RAM is constrained.
- Use `pd.read_csv(..., chunksize=CHUNK_SIZE)` or Polars / PyArrow streaming engines.
- Write output chunks iteratively to disk when building intermediate datasets.

---

## 8. Curate High-Quality Data Instead of Blindly Using the Entire Dataset
- Inspect the data for noise, broken URLs, missing catalog text, corrupted records, extreme price outliers, and duplicates.
- Apply cleaning filters and sensible imputation strategies:
  - Discard or robustly handle anomalous targets (`price <= 0` or extreme log-price spikes).
  - Clean malformed text, strip HTML/special entities, normalize units (e.g., Ounce, Count, Pound, Fl Oz).
- High signal-to-noise ratio always beats raw uncurated volume.

---

## 9. Vectorize Operations
- Strictly avoid Python `for` loops across DataFrame rows (`iterrows()`, `itertuples()`).
- Leverage vectorized pandas / Polars methods, NumPy array operations, or vectorized string methods (`.str.extract()`, `.str.lower()`).
- Perform calculations across whole arrays or tensor slices simultaneously.

---

## 10. 2026 Challenge Specific Constraints
- **External Data Lookup is STRICTLY PROHIBITED:** You are not allowed to use external databases, commercial Entity Resolution APIs, government databases, geocoding APIs, or any internet sources to augment the provided data.
- **Model Size and License:** Final model must be an MIT or Apache 2.0 License model with a maximum of 8 Billion parameters.
- **Output Formatting:** Output must be strictly tab-separated (`.tsv`). Every Source 1 entity must appear exactly once in the output. `matched_entity_ids` must only contain Source 2 or Source 3 IDs. Duplicate IDs are rejected.
- **Open Set Features:** The `country` column must be treated as an open set. Do not hard-code pipelines to only expect `{US, India}` because the test set contains unseen labels (e.g., `France`).

---

## Suggested Directory Organization
```
ML_Challenge_2025/
│
├── RULES.md                    # Core project engineering rules
├── data/
│   ├── raw/                    # train.csv, test.csv, sample_test.csv (read-only)
│   ├── interim/                # Cleaned, parsed tabular features
│   ├── processed/              # Parquet files, reduced embeddings, train/val splits
│   └── images/                 # Downloaded / cached image files
├── cache/                      # Intermediate arrays, tokenized caches, PCA models
├── src/                        # Modular Python source code
│   ├── data/                   # Downloaders, text parsers, cleaners
│   ├── features/               # Feature engineering, vectorizers, embeddings, PCA
│   ├── models/                 # Training, cross-validation, hyperparameter tuning
│   └── utils/                  # I/O helpers, metrics, timers
├── models/                     # Saved model checkpoints and weights
├── notebooks/                  # Experimental notebooks (prototypes only)
└── submissions/                # Final formatted submission files
```
