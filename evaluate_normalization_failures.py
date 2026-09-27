import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.text_normalization import apply_advanced_preprocessing

def split_to_df(texts):
    rows = []
    for t in texts:
        parts = t.split(" | ")
        if len(parts) == 3:
            rows.append({"business_name": parts[0], "business_address": parts[1], "country": parts[2]})
        elif len(parts) > 3:
            # Assume last part is country, and address is second to last
            country = parts[-1]
            address = parts[-2]
            name = " | ".join(parts[:-2])
            rows.append({"business_name": name, "business_address": address, "country": country})
        else:
            rows.append({"business_name": t, "business_address": "", "country": ""})
    return pd.DataFrame(rows)

def main():
    base_dir = Path(__file__).resolve().parents[0]
    shared_errors_file = base_dir / "reports" / "shared_errors.json"
    
    with open(shared_errors_file, "r", encoding="utf-8") as f:
        errors = json.load(f)
    
    print(f"Loaded {len(errors)} shared failures from {shared_errors_file}")
    
    # We will compute TF-IDF similarity between query and its true matches
    # under the old normalization (which is just the exact strings in the JSON)
    # and under the new advanced normalization.
    
    old_recovered_count = 0
    new_recovered_count = 0
    improvements = []
    
    for err in errors:
        query_old = err["query"]
        trues_old = err["true_matches"]
        
        if not trues_old:
            continue
            
        # Old Similarity (using char n-grams as in compare_errors.py)
        vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=50000, dtype=np.float32)
        # Fit on just this tiny local corpus (query + trues + some background noise if needed, but we can just use them)
        # Actually in the real system it's fit on the whole corpus, but for direct pairwise comparison,
        # cosine sim of TF-IDF fit on just these elements is an approximation. Let's fit on them.
        all_old_texts = [query_old] + trues_old
        try:
            old_matrix = vectorizer.fit_transform(all_old_texts)
            old_sims = old_matrix[0].dot(old_matrix[1:].T).toarray()[0]
        except ValueError:
            old_sims = np.zeros(len(trues_old))
            
        # Any > 0.60? (Wait, in the actual script, threshold is 0.60 on full corpus TFIDF, so this local one might be higher. 
        # But we just want to see the relative delta).
        old_max_sim = np.max(old_sims) if len(old_sims) > 0 else 0
        
        # New Similarity
        df_old = split_to_df(all_old_texts)
        new_texts_series = apply_advanced_preprocessing(df_old, apply_token_sort=True)
        new_texts = new_texts_series.tolist()
        
        query_new = new_texts[0]
        trues_new = new_texts[1:]
        
        try:
            new_matrix = vectorizer.fit_transform(new_texts)
            new_sims = new_matrix[0].dot(new_matrix[1:].T).toarray()[0]
        except ValueError:
            new_sims = np.zeros(len(trues_new))
            
        new_max_sim = np.max(new_sims) if len(new_sims) > 0 else 0
        
        # Track improvements
        if new_max_sim > old_max_sim + 0.05:
            improvements.append({
                "old_query": query_old,
                "old_best_match": trues_old[np.argmax(old_sims)] if len(old_sims) > 0 else "",
                "old_sim": float(old_max_sim),
                "new_query": query_new,
                "new_best_match": trues_new[np.argmax(new_sims)] if len(new_sims) > 0 else "",
                "new_sim": float(new_max_sim)
            })
            
        if old_max_sim >= 0.60:
            old_recovered_count += 1
        if new_max_sim >= 0.60:
            new_recovered_count += 1

    print(f"Total Errors Evaluated: {len(errors)}")
    print(f"Old Preprocessing (Local TF-IDF > 0.60): {old_recovered_count}")
    print(f"New Preprocessing (Local TF-IDF > 0.60): {new_recovered_count}")
    print(f"Number of queries with > 0.05 similarity improvement: {len(improvements)}")
    
    # Sort improvements by largest delta
    improvements = sorted(improvements, key=lambda x: x["new_sim"] - x["old_sim"], reverse=True)
    
    report_file = base_dir / "reports" / "10_normalization_improvement_report.md"
    content = f"# Text Normalization Improvement Analysis\n\n"
    content += f"Evaluated the {len(errors)} 'Shared Errors' (where both TF-IDF and Cross-Encoder failed previously) using the new Advanced NLP Normalization Pipeline.\n\n"
    content += f"## Summary Statistics\n"
    content += f"- **Errors Evaluated:** {len(errors)}\n"
    content += f"- **Matches recovered locally (Sim > 0.60) Old Pipeline:** {old_recovered_count}\n"
    content += f"- **Matches recovered locally (Sim > 0.60) New Pipeline:** {new_recovered_count}\n"
    content += f"- **Queries with > 5% similarity boost:** {len(improvements)}\n\n"
    
    content += f"## Top 5 Improvements (Where Normalization Saved the Match)\n\n"
    for imp in improvements[:5]:
        content += f"### Query: `{imp['old_query']}`\n"
        content += f"**Old Pipeline:**\n"
        content += f"- Best True Match: `{imp['old_best_match']}`\n"
        content += f"- TF-IDF Similarity: {imp['old_sim']:.3f}\n\n"
        content += f"**New Pipeline:**\n"
        content += f"- Transformed Query: `{imp['new_query']}`\n"
        content += f"- Transformed True Match: `{imp['new_best_match']}`\n"
        content += f"- New TF-IDF Similarity: {imp['new_sim']:.3f}\n"
        content += f"- **Improvement Delta:** +{imp['new_sim'] - imp['old_sim']:.3f}\n\n"
        content += "---\n\n"
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Report written to {report_file}")

if __name__ == "__main__":
    main()
