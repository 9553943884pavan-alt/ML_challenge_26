# Test Data Exploratory Data Analysis (EDA) Report

**Date:** 2026-09-25
**Scope:** Analysis of `test_source1.parquet`, `test_source2.parquet`, and `test_source3.parquet`.

## 1. Dataset Dimensions

| Source | Rows | Null Names | Null Addresses |
| :--- | :--- | :--- | :--- |
| **Source 1** (Reference) | 1,732,544 | 0 | 0 |
| **Source 2** | 4,887,273 | 46 | 129,408 |
| **Source 3** | 5,082,316 | 59 | 136,098 |
| **Total Test Space** | **11,702,133** | - | - |

## 2. The "Open Set" Country Trap Investigation
The challenge documentation warned that the Test Set contains a country not seen in the Training Set. The distribution analysis confirms this explicitly:

| Country | test_source1 | test_source2 | test_source3 |
| :--- | :--- | :--- | :--- |
| **India** | 809,986 | 2,312,565 | 2,405,000 |
| **US** | 663,106 | 1,871,330 | 1,945,701 |
| **France** | 259,452 | 703,378 | 731,615 |

**Conclusion:** There are no hidden traps. The string is exactly `"France"`. There are no surprise anomalies (e.g., `"FR"`, `"Japan"`, `"Unknown"`). 

## 3. Pipeline Compatibility 
Because the noise distribution in the Test Set perfectly mirrors the Training Set (massive missing addresses in S2/S3, 100% clean S1), our current pipeline architecture is fully compatible:
1.  **NaN filling:** We will continue to fill `NaN` addresses with `""`.
2.  **Country Blocking:** Our logic of concatenating the country string directly into the text for TF-IDF/BM25 remains mathematically robust because it doesn't care that `"France"` is a new word; it just treats it as a new categorical n-gram.
