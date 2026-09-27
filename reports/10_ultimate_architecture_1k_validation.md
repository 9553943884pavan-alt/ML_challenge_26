# Ultimate Hybrid Architecture: 1K Validation Results

**Date:** 2026-09-25
**Sample:** 1,000 queries (700 train / 300 val) from 10k cached CE split

---

## Results by Architecture Tier

| Architecture Phase | Precision | Recall | F0.5 |
|---|:---:|:---:|:---:|
| Baseline: CE threshold=0.40 | 0.9542 | 0.8191 | **0.9077** |
| Tier 4: + Hard Veto Rules | 0.9359 | 0.7503 | **0.8678** |
| Tier 3: LightGBM | 0.9654 | 0.8375 | **0.9224** **<-- BEST** |
| Tier 3+4: LightGBM + Hard Veto | 0.9492 | 0.7659 | **0.8823** |
| Full: LightGBM + Veto + Union-Find | 0.9492 | 0.7659 | **0.8823** |

---

## Top 10 LightGBM Feature Importances

| Feature | Importance |
|---|---|
| name_jw | 1281.0 |
| ce_score | 1186.0 |
| name_tsr | 969.0 |
| name_x_addr | 904.0 |
| addr_tsr | 860.0 |
| ce_x_street | 707.0 |
| ce_x_name_jw | 610.0 |
| addr_jw | 545.0 |
| ce_x_addr_jw | 504.0 |
| street_num_match | 407.0 |

---

## Verdict

> [!CAUTION]
> Architecture insufficient - Best F0.5: **0.9224**. Redesign required.

