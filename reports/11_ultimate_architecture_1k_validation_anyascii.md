# Ultimate Hybrid Architecture: 1K Validation Results

**Date:** 2026-09-25
**Sample:** 1,000 queries (700 train / 300 val) from 10k cached CE split

---

## Results by Architecture Tier

| Architecture Phase | Precision | Recall | F0.5 |
|---|:---:|:---:|:---:|
| Baseline: CE threshold=0.40 | 0.9542 | 0.8191 | **0.9077** |
| Tier 4: + Hard Veto Rules | 0.9359 | 0.7503 | **0.8678** |
| Tier 3: LightGBM | 0.9654 | 0.8367 | **0.9222** **<-- BEST** |
| Tier 3+4: LightGBM + Hard Veto | 0.9492 | 0.7659 | **0.8823** |
| Full: LightGBM + Veto + Union-Find | 0.9492 | 0.7659 | **0.8823** |

---

## Top 10 LightGBM Feature Importances

| Feature | Importance |
|---|---|
| name_jw | 1196.0 |
| name_tsr | 1095.0 |
| name_x_addr | 993.0 |
| ce_score | 980.0 |
| addr_tsr | 952.0 |
| ce_x_street | 831.0 |
| addr_jw | 582.0 |
| ce_x_name_jw | 568.0 |
| ce_x_addr_jw | 506.0 |
| street_num_match | 374.0 |

---

## Verdict

> [!CAUTION]
> Architecture insufficient - Best F0.5: **0.9222**. Redesign required.

