import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_macro_f05

def main():
    base_dir = Path(__file__).resolve().parents[2]
    cache_file = base_dir / "data" / "interim" / "optimization_cache_norm.pkl"
    
    print(f"Loading cached scores from {cache_file}...")
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
        
    gt_dict = cache["gt_dict"]
    ce_candidate_lists = cache["ce_candidate_lists"]
    tfidf_scores_dict = cache["tfidf_scores"]
    ce_scores_dict = cache["ce_scores"]

    print("Executing fine-grained threshold sweep from 0.1 to 1.0...")
    
    # Sweep from 0.10 to 1.00 in steps of 0.02
    tfidf_thresholds = np.arange(0.10, 1.01, 0.02)
    ce_thresholds = np.arange(0.10, 1.01, 0.02)
    
    best_f05 = -1
    best_config = (0, 0)
    
    results = []
    
    total_combinations = len(tfidf_thresholds) * len(ce_thresholds)
    count = 0
    
    for t_thresh in tfidf_thresholds:
        for ce_thresh in ce_thresholds:
            preds = {}
            for q_id, cands in ce_candidate_lists.items():
                final_cands = []
                for c_id in cands:
                    t_score = tfidf_scores_dict[q_id].get(c_id, 0)
                    c_score = ce_scores_dict[q_id].get(c_id, 0)
                    
                    if t_score >= t_thresh and c_score >= ce_thresh:
                        final_cands.append(c_id)
                preds[q_id] = final_cands
                
            f05 = calculate_macro_f05(gt_dict, preds)
            results.append((t_thresh, ce_thresh, f05))
            
            if f05 > best_f05:
                best_f05 = f05
                best_config = (t_thresh, ce_thresh)
                
            count += 1
            if count % 500 == 0:
                print(f"Swept {count}/{total_combinations} combinations...")
                
    results.sort(key=lambda x: x[2], reverse=True)
    
    report_file = base_dir / "reports" / "13_fine_grained_sweep_report.md"
    content = f"# Fine-Grained Threshold Optimization (0.1 to 1.0)\n\n"
    content += f"Swept {total_combinations} combinations across TF-IDF and Cross-Encoder thresholds using the dual-pipeline cached vectors.\n\n"
    
    content += f"## Top 20 Configurations\n\n"
    content += f"| TF-IDF Threshold | Cross-Encoder Threshold | F0.5 Score |\n"
    content += f"| :--- | :--- | :--- |\n"
    
    for t_thresh, ce_thresh, f05 in results[:20]:
        content += f"| {t_thresh:.2f} | {ce_thresh:.2f} | **{f05:.5f}** |\n"
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n==============================================")
    print(f"Absolute Best Configuration Found:")
    print(f"TF-IDF Threshold: {best_config[0]:.2f}")
    print(f"CE Threshold: {best_config[1]:.2f}")
    print(f"Max F0.5 Score: {best_f05:.5f}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
