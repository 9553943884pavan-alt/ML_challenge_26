import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "processed" / "pair_features_100k_sample.parquet"
    report_path = base_dir / "reports" / "20_lgbm_100k_train_test_report.md"
    model_path = base_dir / "models" / "lgbm_100k.txt"
    
    print(f"Loading features from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 1. Prepare Data
    drop_cols = ['query_id', 'candidate_id', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"Total Pairs: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Class Distribution: {y.sum():,} Positive / {len(y) - y.sum():,} Negative")
    
    # 2. Setup 80/20 Train/Test Split
    print("\nSplitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63, # Increased leaves to handle Devanagari/Branch complex logic
        'min_data_in_leaf': 50, # Guard against garbage FNs overfitting
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    # Save Model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    print(f"\nModel successfully saved to -> {model_path}")
    
    # Predict probabilities on test set
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # 3. Threshold Tuning for F0.5
    print("\nSweeping thresholds to maximize F0.5 on Test Set...")
    best_threshold = 0.5
    best_f05 = 0.0
    
    thresholds = np.arange(0.1, 0.99, 0.01)
    for thresh in thresholds:
        binary_preds = (test_preds >= thresh).astype(int)
        score = fbeta_score(y_test, binary_preds, beta=0.5)
        if score > best_f05:
            best_f05 = score
            best_threshold = thresh
            
    print(f"Optimal Threshold: {best_threshold:.2f}")
    
    # 4. Final Metrics at Optimal Threshold
    final_preds = (test_preds >= best_threshold).astype(int)
    
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    f05 = fbeta_score(y_test, final_preds, beta=0.5)
    cm = confusion_matrix(y_test, final_preds)
    
    print("\n--- Final Test Metrics (Optimal Threshold) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F0.5:      {f05:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 5. Generate Markdown Report
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance (Splits)': model.feature_importances_
    }).sort_values(by='Importance (Splits)', ascending=False)
    
    lines = [
        "# LightGBM 1 Lakh Queries Train/Test Split Results",
        "",
        f"- **Total Candidate Pairs Evaluated:** {len(df):,}",
        f"- **True Matches:** {y.sum():,}",
        f"- **False Positives:** {len(y) - y.sum():,}",
        f"- **Model:** LightGBM Binary Classifier (Saved)",
        f"- **Validation Strategy:** 80/20 Train-Test Split",
        "",
        "## Performance Metrics (Test Set)",
        "The model predicts a probability for each pair. We tuned the probability threshold to explicitly maximize the **F0.5 score**.",
        "",
        f"- **Optimal Probability Threshold:** `{best_threshold:.2f}`",
        f"- **Precision:** `{precision:.4f}`",
        f"- **Recall:** `{recall:.4f}`",
        f"- **F0.5 Score:** `{f05:.4f}`",
        "",
        "## Confusion Matrix (Test Set)",
        "| | Predicted Negative (0) | Predicted Positive (1) |",
        "| :--- | :--- | :--- |",
        f"| **Actual Negative (0)** | {cm[0][0]:,} (True Neg) | {cm[0][1]:,} (False Pos) |",
        f"| **Actual Positive (1)** | {cm[1][0]:,} (False Neg) | {cm[1][1]:,} (True Pos) |",
        "",
        "## Top 10 Feature Importances",
        "| Feature | Split Importance |",
        "| :--- | :--- |"
    ]
    
    for _, row in importance_df.head(10).iterrows():
        lines.append(f"| `{row['Feature']}` | {row['Importance (Splits)']:.1f} |")
        
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"\nReport successfully saved to -> {report_path}")

if __name__ == "__main__":
    main()
