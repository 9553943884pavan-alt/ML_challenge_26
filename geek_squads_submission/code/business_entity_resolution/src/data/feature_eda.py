"""
Exploratory Data Analysis (EDA) on Pairwise Features
Computes Mutual Information, Univariate distributions by class, and Multivariate correlations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings('ignore')

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "processed" / "pair_features_5k_sample.parquet"
    report_path = base_dir / "reports" / "17_feature_eda_report.md"
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Exclude IDs from features
    feature_cols = [c for c in df.columns if c not in ['query_id', 'candidate_id', 'label']]
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    lines = [
        "# Exploratory Data Analysis: Pairwise Features",
        f"\n**Dataset Size:** {len(df):,} pairs",
        f"**True Matches (1):** {y.sum():,}",
        f"**False Positives (0):** {len(df) - y.sum():,}",
        f"**Imbalance Ratio:** 1 : {(len(df) - y.sum()) / y.sum():.2f}",
        "\n---\n"
    ]
    
    # ─────────────────────────────────────────────────────────────
    # 1. MUTUAL INFORMATION (Predictive Power)
    # ─────────────────────────────────────────────────────────────
    print("Computing Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': feature_cols, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    
    lines.append("## 1. Mutual Information (Feature Importance)")
    lines.append("Mutual Information (MI) measures the dependency between a feature and the target label. Higher means more predictive power.\n")
    lines.append("| Feature | Mutual Information Score |")
    lines.append("| :--- | :--- |")
    for _, row in mi_df.iterrows():
        lines.append(f"| `{row['Feature']}` | {row['MI_Score']:.4f} |")
        
    lines.append("\n---\n")
    
    # ─────────────────────────────────────────────────────────────
    # 2. UNIVARIATE ANALYSIS (Class Separation)
    # ─────────────────────────────────────────────────────────────
    print("Computing Univariate Statistics...")
    lines.append("## 2. Univariate Analysis (Class Distribution)")
    lines.append("Averages for True Matches (1) vs False Positives (0) to show how well each feature separates the classes.\n")
    
    lines.append("| Feature | Mean (True Matches) | Mean (False Positives) | Separation (Absolute Diff) |")
    lines.append("| :--- | :--- | :--- | :--- |")
    
    sep_list = []
    for col in feature_cols:
        mean_1 = df[df['label'] == 1][col].mean()
        mean_0 = df[df['label'] == 0][col].mean()
        sep = abs(mean_1 - mean_0)
        sep_list.append({'col': col, 'm1': mean_1, 'm0': mean_0, 'sep': sep})
        
    # Sort by separation magnitude
    sep_list = sorted(sep_list, key=lambda x: x['sep'], reverse=True)
    
    for item in sep_list:
        lines.append(f"| `{item['col']}` | {item['m1']:.4f} | {item['m0']:.4f} | **{item['sep']:.4f}** |")

    lines.append("\n---\n")

    # ─────────────────────────────────────────────────────────────
    # 3. MULTIVARIATE ANALYSIS (Correlation & Redundancy)
    # ─────────────────────────────────────────────────────────────
    print("Computing Correlation Matrix...")
    lines.append("## 3. Multivariate Analysis (Feature Redundancy)")
    lines.append("Highly correlated features (>0.85) provide redundant information. LightGBM handles this well, but it's good to know which features duplicate each other.\n")
    
    corr_matrix = X.corr(method='pearson').abs()
    
    lines.append("| Feature A | Feature B | Pearson Correlation |")
    lines.append("| :--- | :--- | :--- |")
    
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.85:
                highly_correlated.append((col1, col2, corr_val))
                
    highly_correlated = sorted(highly_correlated, key=lambda x: x[2], reverse=True)
    
    if not highly_correlated:
        lines.append("| None | None | No features > 0.85 correlation |")
    else:
        for col1, col2, val in highly_correlated:
            lines.append(f"| `{col1}` | `{col2}` | {val:.4f} |")

    # ─────────────────────────────────────────────────────────────
    # Write to File
    # ─────────────────────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"EDA Report saved to {report_path}")

if __name__ == "__main__":
    main()
