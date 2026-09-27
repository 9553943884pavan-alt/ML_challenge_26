#!/usr/bin/env python3
"""
generate_submission_aws.py
==========================
AWS-patched version of generate_submission.py.
All paths are rewritten to /mnt/nvme/mlchallenge26 (the NVMe SSD).
Intermediate caches are uploaded to S3 incrementally.

This is identical in logic to generate_submission.py but:
 - Uses AWS_BASE_DIR for all paths
 - Uploads cache files to S3 after each major stage completes
 - Environment variable AWS_S3_BUCKET and AWS_REGION must be set
   (injected by user-data script)
"""

import os, sys, gc, time, pickle, json, subprocess, logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ── Environment ───────────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "mlchallenge26")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# ── Paths (NVMe SSD for fast I/O) ─────────────────────────────────────────────
BASE_DIR  = Path("/mnt/nvme/mlchallenge26")
TEST_DIR  = BASE_DIR / "data" / "processed" / "test"
RAW_TEST  = BASE_DIR / "data" / "raw" / "test"
CACHE_DIR = BASE_DIR / "cache"
SUBS_DIR  = BASE_DIR / "submissions"
for d in [CACHE_DIR, SUBS_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VECTORIZER_CACHE = CACHE_DIR / "tfidf_vectorizer.joblib"
CORPUS_NPZ       = CACHE_DIR / "test_corpus_tfidf_matrix.npz"
S1_NPZ           = CACHE_DIR / "test_s1_tfidf_matrix.npz"
CAND_CACHE       = CACHE_DIR / "tfidf_candidates_rolling.pkl"
PROGRESS_JSON    = CACHE_DIR / "progress.json"
PARTIAL_TSV      = SUBS_DIR  / "matching_results_partial.tsv"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
TFIDF_THRESHOLD  = 0.40
TOP_K_CANDIDATES = 15
CE_THRESHOLD     = 0.80
CE_MODEL_NAME    = "BAAI/bge-reranker-v2-m3"
CE_MAX_LEN       = 128
CE_BATCH_INIT    = 256
TFIDF_CHUNK      = 5_000
TFIDF_CKPT_EVERY = 50
QUERY_BLOCK      = 25_000
TFIDF_XFMBATCH   = 250_000
TFIDF_NGRAM      = (2, 4)
TFIDF_MAX_FEAT   = 50_000
TFIDF_SAMPLE     = 300_000

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── S3 upload helper ──────────────────────────────────────────────────────────
import boto3
s3_client = boto3.client("s3", region_name=AWS_REGION)

def s3_push(local_path: Path, s3_key: str):
    try:
        mb = local_path.stat().st_size / 1e6
        log.info(f"S3 upload: {local_path.name} ({mb:.0f} MB) → s3://{S3_BUCKET}/{s3_key}")
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        log.info(f"  ✓ Uploaded")
    except Exception as e:
        log.warning(f"  S3 upload failed: {e}")

def atomic_pickle(obj, path: Path, s3_key: str = None):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    if s3_key:
        s3_push(path, s3_key)

def atomic_json(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

# ── Text cleaning ─────────────────────────────────────────────────────────────
def make_text(df: pd.DataFrame) -> pd.Series:
    name    = df["business_name"].astype(object).fillna("").astype(str).str.lower().str.strip()
    address = df["business_address"].astype(object).fillna("").astype(str).str.lower().str.strip()
    country = df["country"].astype(object).fillna("").astype(str).str.lower().str.strip() if "country" in df.columns else pd.Series([""] * len(df), index=df.index)
    return name + " | " + address + " | " + country

def read_src(path: Path):
    import pyarrow.parquet as pq
    if path.suffix == ".parquet":
        tbl = pq.read_table(path, columns=["entity_id","business_name","business_address","country"])
        df  = tbl.to_pandas(); del tbl
    else:
        df = pd.read_csv(path, sep="\t", dtype=str)
    return df

# ── STAGE 1: Load data ────────────────────────────────────────────────────────
def stage_load():
    log.info("STAGE 1: Loading data ...")
    # Prefer parquet; fall back to TSV
    def best_path(name):
        pq = TEST_DIR / f"{name}.parquet"
        ts = RAW_TEST  / f"{name}.tsv"
        return pq if pq.exists() else ts

    s1_df = read_src(best_path("test_source1"))
    s1_df["text"] = make_text(s1_df)
    s1_ids = list(s1_df["entity_id"])
    s1_text_lookup = dict(zip(s1_df["entity_id"], s1_df["text"]))
    log.info(f"  S1: {len(s1_ids):,} queries")
    del s1_df; gc.collect()

    corpus_text_lookup = {}
    for name in ("test_source2", "test_source3"):
        df = read_src(best_path(name))
        df["text"] = make_text(df)
        for eid, txt in zip(df["entity_id"], df["text"]):
            corpus_text_lookup.setdefault(eid, txt)
        log.info(f"  {name}: {len(df):,} rows ingested")
        del df; gc.collect()

    corpus_ids   = np.array(list(corpus_text_lookup.keys()))
    corpus_texts = list(corpus_text_lookup.values())
    log.info(f"  Corpus: {len(corpus_ids):,} unique entities")
    return s1_ids, s1_text_lookup, corpus_ids, corpus_texts, corpus_text_lookup

# ── STAGE 2: TF-IDF ──────────────────────────────────────────────────────────
def stage_tfidf(s1_ids, corpus_ids, corpus_texts, s1_text_lookup):
    s1_texts = [s1_text_lookup[q] for q in s1_ids]

    import json as _json
    CHUNK_META = CACHE_DIR / "corpus_chunk_meta.json"
    CHUNK_DIR  = CACHE_DIR / "corpus_chunks"

    def invalid_cache():
        for p in [VECTORIZER_CACHE, S1_NPZ, CHUNK_META]:
            p.unlink(missing_ok=True)
        if CHUNK_DIR.exists():
            import shutil; shutil.rmtree(CHUNK_DIR)

    # Try cache
    if VECTORIZER_CACHE.exists() and CHUNK_META.exists() and S1_NPZ.exists():
        try:
            log.info("Loading TF-IDF caches ...")
            vec = joblib.load(VECTORIZER_CACHE)
            sm  = sp.load_npz(S1_NPZ)
            with open(CHUNK_META) as f: meta = _json.load(f)
            
            if sm.shape[1] != len(vec.vocabulary_) or sm.shape[0] != len(s1_texts):
                raise ValueError("Shape mismatch")
                
            chunk_files = [Path(p) for p in meta["chunk_files"]]
            log.info(f"  Loaded OK — corpus chunks {len(chunk_files)}, s1 {sm.shape}")
            return sm, chunk_files, vec
        except Exception as e:
            log.warning(f"  Cache invalid ({e}) — rebuilding")
            invalid_cache()

    # Fit
    log.info("Fitting TF-IDF vectorizer ...")
    half = TFIDF_SAMPLE // 2
    sample = s1_texts[:half] + corpus_texts[:half]
    vec = TfidfVectorizer(ngram_range=TFIDF_NGRAM, analyzer="char_wb",
                          max_features=TFIDF_MAX_FEAT, dtype=np.float32, sublinear_tf=True)
    vec.fit(sample); del sample; gc.collect()

    # Dynamic Corpus Chunking (save to disk immediately)
    log.info(f"Transforming corpus ({len(corpus_texts):,} docs) dynamically ...")
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    
    for i, start in enumerate(tqdm(range(0, len(corpus_texts), TFIDF_XFMBATCH), desc="Corpus")):
        batch = corpus_texts[start : start+TFIDF_XFMBATCH]
        chunk = vec.transform(batch)
        cpath = CHUNK_DIR / f"chunk_{i:05d}.npz"
        sp.save_npz(cpath, chunk)
        s3_push(cpath, f"cache/corpus_chunks/{cpath.name}")
        chunk_files.append(cpath)
        del chunk, batch; gc.collect()

    # Transform S1 (small enough to vstack)
    log.info(f"Transforming S1 ({len(s1_texts):,} queries) ...")
    chunks = []
    for i in tqdm(range(0, len(s1_texts), TFIDF_XFMBATCH), desc="S1"):
        chunks.append(vec.transform(s1_texts[i:i+TFIDF_XFMBATCH]))
    sm = sp.vstack(chunks, format="csr"); del chunks; gc.collect()
    log.info(f"  S1 matrix {sm.shape}")

    # Save vectorizer + S1 + Meta
    joblib.dump(vec, VECTORIZER_CACHE, compress=3)
    sp.save_npz(S1_NPZ, sm)
    
    tmp = CHUNK_META.with_suffix(".tmp")
    with open(tmp, "w") as f:
        _json.dump({"chunk_files": [str(p) for p in chunk_files]}, f)
    os.replace(tmp, CHUNK_META)

    s3_push(VECTORIZER_CACHE, "cache/tfidf_vectorizer.joblib")
    s3_push(S1_NPZ, "cache/test_s1_tfidf_matrix.npz")
    s3_push(CHUNK_META, "cache/corpus_chunk_meta.json")

    return sm, chunk_files, vec

# ── STAGE 3: Retrieval ────────────────────────────────────────────────────────
def stage_retrieval(s1_matrix, corpus_chunk_files, s1_ids, corpus_ids):
    candidate_lists = {}
    resume_qchunk = 0

    if CAND_CACHE.exists():
        try:
            with open(CAND_CACHE, "rb") as f:
                ckpt = pickle.load(f)
            candidate_lists = ckpt["candidate_lists"]
            resume_qchunk   = ckpt["next_qchunk"]
            log.info(f"Resumed retrieval: {len(candidate_lists):,} queries done, next qchunk={resume_qchunk}")
        except Exception as e:
            log.warning(f"Retrieval checkpoint unreadable ({e}) — starting fresh")

    n = s1_matrix.shape[0]
    total_cands = sum(len(v) for v in candidate_lists.values())
    qchunk_idx = 0

    log.info("Pre-computing corpus offsets ...")
    corpus_offsets = []
    offset = 0
    for cf in corpus_chunk_files:
        c = sp.load_npz(cf)
        nrows = c.shape[0]
        corpus_offsets.append((offset, offset + nrows))
        offset += nrows
        del c; gc.collect()

    with tqdm(total=n, initial=len(candidate_lists), desc="Retrieval") as pbar:
        for q_start in range(0, n, TFIDF_CHUNK):
            if qchunk_idx < resume_qchunk:
                qchunk_idx += 1; continue

            q_end = min(q_start + TFIDF_CHUNK, n)
            s1_block = s1_matrix[q_start:q_end]
            best = [dict() for _ in range(q_end - q_start)]

            # Stream corpus chunks one by one
            for ci, cf in enumerate(corpus_chunk_files):
                c_start, c_end = corpus_offsets[ci]
                corpus_chunk = sp.load_npz(cf)
                
                sims = s1_block.dot(corpus_chunk.T)
                iptr, idxs, dta = sims.indptr, sims.indices, sims.data
                
                for li in range(q_end - q_start):
                    rs, re = iptr[li], iptr[li+1]
                    sc, co = dta[rs:re], idxs[rs:re]
                    mask = sc > TFIDF_THRESHOLD
                    for lc, val in zip(co[mask], sc[mask]):
                        best[li][c_start + lc] = val
                        
                del corpus_chunk, sims; gc.collect()

            # Process top K
            for li, gi in enumerate(range(q_start, q_end)):
                d = best[li]
                if not d:
                    candidate_lists[s1_ids[gi]] = []; continue
                    
                cols = np.fromiter(d.keys(), dtype=np.int64)
                scores = np.fromiter(d.values(), dtype=np.float32)
                if len(scores) > TOP_K_CANDIDATES:
                    tk = np.argpartition(scores, -TOP_K_CANDIDATES)[-TOP_K_CANDIDATES:]
                    tk = tk[np.argsort(-scores[tk])]
                    cols = cols[tk]
                candidate_lists[s1_ids[gi]] = corpus_ids[cols].tolist()
                total_cands += len(cols)

            del best; qchunk_idx += 1
            pbar.update(q_end - q_start)

            # Rolling checkpoint
            if qchunk_idx % TFIDF_CKPT_EVERY == 0:
                atomic_pickle({"candidate_lists": candidate_lists, "next_qchunk": qchunk_idx},
                               CAND_CACHE, s3_key="cache/tfidf_candidates_rolling.pkl")

    atomic_pickle({"candidate_lists": candidate_lists, "next_qchunk": qchunk_idx},
                   CAND_CACHE, s3_key="cache/tfidf_candidates_rolling.pkl")
    log.info(f"Retrieval complete: {total_cands:,} candidates ({total_cands/len(s1_ids):.1f}/query)")
    return candidate_lists


