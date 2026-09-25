"""
generate_submission.py
======================
Generates matching_results.tsv for the ML Challenge 2026
Business Entity Resolution task (leaderboard submission file).

Pipeline
--------
Stage 1 : TF-IDF character n-gram retrieval  (threshold = 0.40)
Stage 2 : BAAI/bge-reranker-v2-m3 Cross-Encoder re-ranking  (threshold = 0.80)

Key design decisions (aligned with RULES.md)
---------------------------------------------
- Rule 2  : All heavy results are cached to disk; subsequent runs skip re-computation.
- Rule 5  : Cross-Encoder inference uses GPU batches (batch_size=256), NOT single-sample loops.
- Rule 6  : Mixed-precision (FP16) enabled where CUDA is available.
- Rule 7  : TF-IDF similarity computed in sparse chunks (5 000 queries at a time) to avoid
            loading the full S1 x corpus dense matrix into RAM.
- Rule 8  : Missing values filled with "" (not "Unknown") so they contribute 0 similarity weight.
- Rule 9  : All pandas operations are vectorised; no iterrows() in the hot path.
- Rule 10 : Country treated as an open string -- concatenated verbatim into the text field.
            Output is strictly TSV; every S1 entity appears exactly once.

Thresholds (from report 06_cross_encoder_optimization_report.md)
-----------------------------------------------------------------
- TF-IDF threshold : 0.40  ->  Stage-1 recall ~96.69 %
- Cross-Encoder threshold : 0.80  ->  peak macro F0.5 = 0.9431 on 10k-query validation split
"""

import os
import sys
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parents[2]
TEST_DIR        = BASE_DIR / "data" / "processed" / "test"
RAW_TEST_DIR    = BASE_DIR / "data" / "raw" / "test"
OUTPUT_DIR      = BASE_DIR / "output"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
CACHE_DIR       = BASE_DIR / "data" / "processed"
LOG_PATH        = SUBMISSIONS_DIR / "run.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup  -- writes to both console and submissions/run.log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

def log_tqdm(msg: str):
    """Thread-safe print that doesn't clash with active tqdm bars."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")
    # Also write to file handler
    with open(LOG_PATH, "a", encoding="utf-8") as _f:
        _f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# ---------------------------------------------------------------------------
# Hyper-parameters  (validated against 10k-split in report 06)
# ---------------------------------------------------------------------------
TFIDF_THRESHOLD  = 0.40    # loosened from 0.60 -> recall 96.69 %
CE_THRESHOLD     = 0.80    # optimal on 10k val-split -> F0.5 = 0.9431
CE_MODEL_NAME    = "BAAI/bge-reranker-v2-m3"
CE_MAX_LEN       = 128
CE_BATCH_SIZE    = 128     # RTX 3050 (6GB VRAM) -- 560M param model in FP16 (~1.1GB weights, ~3GB activations)
TFIDF_CHUNK_SIZE = 5_000   # queries processed per sparse-matmul chunk (Rule 7)
TFIDF_NGRAM      = (2, 4)
TFIDF_MAX_FEAT   = 50_000

CACHE_PATH = CACHE_DIR / "cross_encoder_preds_cache_test_tfidf040.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(df):
    """
    Vectorised text assembly (Rule 9).
    Fills NaN with "" so missing addresses contribute zero TF-IDF weight (Rule 8).
    Country is concatenated verbatim as an open-set label (Rule 10).
    """
    name    = df["business_name"].astype(str).fillna("").str.lower().str.strip()
    address = df["business_address"].astype(str).fillna("").str.lower().str.strip()
    if "country" in df.columns:
        country = df["country"].astype(str).fillna("").str.lower().str.strip()
    else:
        country = pd.Series([""] * len(df), index=df.index)
    return name + " | " + address + " | " + country


def run_tfidf_stage(s1_matrix, corpus_matrix, s1_ids, corpus_ids, threshold, chunk_size):
    """
    Chunked sparse TF-IDF candidate generation (Rule 7).
    Avoids materialising the full S1 x corpus dense similarity matrix.
    Returns {s1_id: [candidate_corpus_id, ...]}
    """
    candidate_lists = {}
    n_queries = s1_matrix.shape[0]
    n_chunks   = (n_queries + chunk_size - 1) // chunk_size
    total_cands = 0

    with tqdm(total=n_queries, desc="[STEP 3] TF-IDF candidates",
              unit="query", dynamic_ncols=True, colour="cyan") as pbar:
        for start in range(0, n_queries, chunk_size):
            end = min(start + chunk_size, n_queries)
            chunk_sims = s1_matrix[start:end].dot(corpus_matrix.T)   # sparse x sparse

            chunk_cands = 0
            for local_i, global_i in enumerate(range(start, end)):
                row = chunk_sims.getrow(local_i)
                mask = row.data > threshold
                valid_idx = row.indices[mask]
                candidate_lists[s1_ids[global_i]] = corpus_ids[valid_idx].tolist()
                chunk_cands += len(valid_idx)

            total_cands += chunk_cands
            pbar.update(end - start)
            pbar.set_postfix({
                "cands_so_far": f"{total_cands:,}",
                "avg/q": f"{total_cands / (end):.1f}",
            })

    return candidate_lists


def run_cross_encoder_stage(candidate_lists, s1_ids, s1_text_lookup,
                             corpus_text_lookup, cache_path):
    """
    Batched GPU cross-encoder inference (Rules 5, 6).
    Saves predictions to disk so subsequent runs skip this expensive step (Rule 2).
    Returns {s1_id: [(candidate_id, score), ...]}
    """
    # Cache hit
    if cache_path.exists():
        log_tqdm(f"Loading cached CE predictions from {cache_path} ...")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # Load model
    log_tqdm(f"Loading Cross-Encoder: {CE_MODEL_NAME} ...")
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_tqdm(f"Device: {device.upper()}")
    if device == "cuda":
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log_tqdm(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {vram_total:.1f} GB")

    model = CrossEncoder(CE_MODEL_NAME, max_length=CE_MAX_LEN, device=device)

    # Enable FP16 on CUDA (Rule 6)
    if device == "cuda":
        try:
            model.model = model.model.half()
            log_tqdm("FP16 mixed-precision enabled.")
        except Exception as e:
            log_tqdm(f"FP16 skipped: {e}")

    # ── Build flat pair list with progress bar ────────────────────────────────
    log_tqdm("Building pair list for inference ...")
    all_pairs = []
    pair_map  = []    # (q_id, c_id)

    with tqdm(s1_ids, desc="[STEP 4a] Building pairs",
              unit="query", dynamic_ncols=True, colour="yellow") as pbar:
        for q_id in pbar:
            q_text = s1_text_lookup[q_id]
            cands  = candidate_lists.get(q_id, [])
            for c_id in cands:
                all_pairs.append([q_text, corpus_text_lookup[c_id]])
                pair_map.append((q_id, c_id))
            pbar.set_postfix({"pairs_built": f"{len(all_pairs):,}"})

    total_pairs = len(all_pairs)
    log_tqdm(f"Total pairs to score: {total_pairs:,}")

    query_candidate_probs = {q_id: [] for q_id in s1_ids}

    if total_pairs == 0:
        log_tqdm("No candidate pairs -- all entities will be singletons.")
        return query_candidate_probs

    # ── Manual batched inference with tqdm ───────────────────────────────────
    t0          = time.time()
    all_scores  = []
    n_batches   = (total_pairs + CE_BATCH_SIZE - 1) // CE_BATCH_SIZE

    with tqdm(total=total_pairs, desc="[STEP 4b] Cross-Encoder inference",
              unit="pair", dynamic_ncols=True, colour="green") as pbar:
        for batch_start in range(0, total_pairs, CE_BATCH_SIZE):
            batch_end    = min(batch_start + CE_BATCH_SIZE, total_pairs)
            batch        = all_pairs[batch_start:batch_end]
            batch_scores = model.predict(batch, batch_size=CE_BATCH_SIZE,
                                         show_progress_bar=False,
                                         convert_to_numpy=True)
            if isinstance(batch_scores, np.ndarray) and batch_scores.ndim > 1:
                batch_scores = batch_scores.flatten()
            all_scores.extend(batch_scores.tolist())

            elapsed  = time.time() - t0 + 1e-9
            done     = batch_end
            rate     = done / elapsed
            remaining = (total_pairs - done) / rate if rate > 0 else 0

            vram_str = ""
            if device == "cuda":
                used = torch.cuda.memory_reserved(0) / 1024**3
                vram_str = f"{used:.1f}GB"

            pbar.update(batch_end - batch_start)
            pbar.set_postfix({
                "pairs/s": f"{rate:.0f}",
                "ETA":     f"{remaining/60:.1f}min",
                "VRAM":    vram_str or "N/A",
            })

    scores  = np.array(all_scores, dtype=np.float32)
    elapsed = time.time() - t0
    log_tqdm(f"Inference done in {elapsed/60:.1f} min  ({total_pairs / elapsed:.0f} pairs/s)")

    # ── Collate results ───────────────────────────────────────────────────────
    with tqdm(zip(pair_map, scores), total=total_pairs,
              desc="[STEP 4c] Collating scores",
              unit="pair", dynamic_ncols=True, colour="blue") as pbar:
        for (q_id, c_id), score in pbar:
            query_candidate_probs[q_id].append((c_id, float(score)))

    # ── Persist cache (Rule 2) ────────────────────────────────────────────────
    log_tqdm(f"Saving CE cache -> {cache_path}")
    with open(cache_path, "wb") as fh:
        pickle.dump(query_candidate_probs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log_tqdm(f"Cache saved ({cache_path.stat().st_size / 1024**2:.1f} MB)")

    return query_candidate_probs


def build_output_rows(s1_ids, query_candidate_probs, ce_threshold):
    """
    Apply CE threshold, deduplicate, and build the matching_results DataFrame.
    Returns matching_df.
    """
    matching_rows = []

    for q_id in s1_ids:
        c_probs = query_candidate_probs.get(q_id, [])
        seen    = set()
        matched = []
        for c_id, score in c_probs:
            if score >= ce_threshold and c_id not in seen:
                seen.add(c_id)
                matched.append(c_id)

        matching_rows.append({
            "source1_entity_id":  q_id,
            "matched_entity_ids": ",".join(matched),
        })

    return pd.DataFrame(matching_rows)


def save_tsv(df, *paths):
    for p in paths:
        df.to_csv(p, sep="\t", index=False)
        size_kb = Path(p).stat().st_size / 1024
        log_tqdm(f"Saved -> {p}  ({size_kb:.0f} KB)")


def run_validator(base_dir, submissions_dir):
    import subprocess
    validator = base_dir / "src" / "utils" / "validate_submission.py"
    if not validator.exists():
        print("   [SKIP] validate_submission.py not found.")
        return

    cmd = [
        sys.executable,
        str(validator),
        "--matching", str(submissions_dir / "matching_results.tsv"),
        "--test-dir", str(base_dir / "data" / "raw" / "test"),
    ]
    print(f"   Running validator ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0:
        print("[SUCCESS] matching_results.tsv PASSED all validator checks!")
    else:
        print("[ERROR] Validator found issues. See output above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    sep = "=" * 70

    # STEP 1: Load test data
    log.info(sep)
    log.info("STEP 1 - Loading test parquet files ...")
    log.info(sep)
    t0 = time.time()
    for name, path in [("S1", TEST_DIR / "test_source1.parquet"),
                       ("S2", TEST_DIR / "test_source2.parquet"),
                       ("S3", TEST_DIR / "test_source3.parquet")]:
        log.info(f"  Loading {name} from {path.name} ...")
    s1 = pd.read_parquet(TEST_DIR / "test_source1.parquet")
    s2 = pd.read_parquet(TEST_DIR / "test_source2.parquet")
    s3 = pd.read_parquet(TEST_DIR / "test_source3.parquet")
    log.info(f"  S1 : {s1.shape[0]:,} rows  |  S2 : {s2.shape[0]:,} rows  |  S3 : {s3.shape[0]:,} rows  [{time.time()-t0:.1f}s]")

    # STEP 2: Build text representations
    log.info(sep)
    log.info("STEP 2 - Building text representations (name | address | country) ...")
    log.info(sep)
    t0 = time.time()
    s1["text"] = clean_text(s1)
    s2["text"] = clean_text(s2)
    s3["text"] = clean_text(s3)

    corpus_df    = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=["entity_id"])
    corpus_ids   = corpus_df["entity_id"].values
    corpus_texts = corpus_df["text"].tolist()
    corpus_text_lookup = dict(zip(corpus_ids, corpus_texts))

    s1_ids         = s1["entity_id"].tolist()
    s1_text_lookup = dict(zip(s1["entity_id"], s1["text"]))

    log.info(f"  S1 queries  : {len(s1_ids):,}")
    log.info(f"  Corpus size : {len(corpus_ids):,}  [{time.time()-t0:.1f}s]")

    # STEP 3: TF-IDF Candidate Generation
    log.info(sep)
    log.info(f"STEP 3 - TF-IDF Candidate Generation (threshold > {TFIDF_THRESHOLD}) ...")
    log.info(sep)
    log.info("  Fitting TF-IDF vectoriser on corpus ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM,
        analyzer="char_wb",
        max_features=TFIDF_MAX_FEAT,
        dtype=np.float32,
        sublinear_tf=True,
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    log.info(f"  Corpus matrix : {corpus_matrix.shape}  [{time.time()-t0:.1f}s]")

    log.info("  Transforming S1 queries ...")
    t0 = time.time()
    s1_matrix = vectorizer.transform(s1["text"].tolist())
    log.info(f"  S1 matrix     : {s1_matrix.shape}  [{time.time()-t0:.1f}s]")

    candidate_lists = run_tfidf_stage(
        s1_matrix, corpus_matrix, s1_ids, corpus_ids,
        threshold=TFIDF_THRESHOLD,
        chunk_size=TFIDF_CHUNK_SIZE,
    )

    total_cands = sum(len(v) for v in candidate_lists.values())
    log.info(f"  Total candidate pairs : {total_cands:,}")
    log.info(f"  Avg candidates/query  : {total_cands / len(s1_ids):.1f}")

    # STEP 4: Cross-Encoder Re-ranking
    log.info(sep)
    log.info(f"STEP 4 - Cross-Encoder Re-ranking (model = {CE_MODEL_NAME}) ...")
    log.info(sep)
    query_candidate_probs = run_cross_encoder_stage(
        candidate_lists, s1_ids, s1_text_lookup,
        corpus_text_lookup, cache_path=CACHE_PATH,
    )

    # STEP 5: Apply threshold & build matching DataFrame
    log.info(sep)
    log.info(f"STEP 5 - Applying CE threshold {CE_THRESHOLD} & building output rows ...")
    log.info(sep)
    matching_df = build_output_rows(
        s1_ids, query_candidate_probs, CE_THRESHOLD
    )

    matched_mask  = matching_df["matched_entity_ids"].str.len() > 0
    matched_count = matched_mask.sum()
    total_matches = int(matching_df["matched_entity_ids"].str.count(",").sum()) + int(matched_count)
    log.info(f"  S1 entities with >=1 match : {matched_count:,} / {len(s1_ids):,}  "
             f"({matched_count / len(s1_ids) * 100:.2f}%)")
    log.info(f"  Total match edges           : {total_matches:,}")

    # STEP 6: Save matching_results.tsv
    log.info(sep)
    log.info("STEP 6 - Saving matching_results.tsv ...")
    log.info(sep)
    save_tsv(matching_df,
             OUTPUT_DIR      / "matching_results.tsv",
             SUBMISSIONS_DIR / "matching_results.tsv")

    # STEP 7: Validate
    log.info(sep)
    log.info("STEP 7 - Validating submission format ...")
    log.info(sep)
    run_validator(BASE_DIR, SUBMISSIONS_DIR)

    # Summary
    elapsed_total = time.time() - total_start
    log.info(sep)
    log.info("DONE")
    log.info(sep)
    log.info(f"  TF-IDF threshold : {TFIDF_THRESHOLD}")
    log.info(f"  CE threshold     : {CE_THRESHOLD}  (peak val F0.5 = 0.9431)")
    log.info(f"  CE model         : {CE_MODEL_NAME}")
    log.info(f"  CE batch size    : {CE_BATCH_SIZE}  (RTX 3050 tuned)")
    log.info(f"  Wall-clock time  : {elapsed_total / 60:.1f} min")
    log.info(f"  Output           : {SUBMISSIONS_DIR / 'matching_results.tsv'}")
    log.info(f"  Full log         : {LOG_PATH}")


if __name__ == "__main__":
    main()
