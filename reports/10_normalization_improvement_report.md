# Text Normalization Improvement Analysis

Evaluated the 271 'Shared Errors' (where both TF-IDF and Cross-Encoder failed previously) using the new Advanced NLP Normalization Pipeline.

## Summary Statistics
- **Errors Evaluated:** 271
- **Matches recovered locally (Sim > 0.60) Old Pipeline:** 254
- **Matches recovered locally (Sim > 0.60) New Pipeline:** 256
- **Queries with > 5% similarity boost:** 124

## Top 5 Improvements (Where Normalization Saved the Match)

### Query: `harbor alliance | 45 goff street, corning, ny | us`
**Old Pipeline:**
- Best True Match: `harbor alliance | 45 goff st, po box 4918, corning, ny | us`
- TF-IDF Similarity: 0.700

**New Pipeline:**
- Transformed Query: `alliance harbor | 45 corning goff ny street | us`
- Transformed True Match: `alliance harbor | 45 corning goff ny street | us`
- New TF-IDF Similarity: 1.000
- **Improvement Delta:** +0.300

---

### Query: `marthena mowrer, p.a. | ar, 2703 55th street, rogers, unit apt 203 | us`
**Old Pipeline:**
- Best True Match: `marthena mowrer, p.a. [ltd] | 2703 55th st, rogers, ar | us`
- TF-IDF Similarity: 0.638

**New Pipeline:**
- Transformed Query: `marthena mowrer p | 2703 55th ar rogers street unit | us`
- Transformed True Match: `marthena mowrer p | 2703 55th ar rogers street | us`
- New TF-IDF Similarity: 0.906
- **Improvement Delta:** +0.268

---

### Query: `evangeline chase invesco llc | 13809 43rd avenue, spokane, wa | us`
**Old Pipeline:**
- Best True Match: `evangeline chase chase [invesco] | 13809 43rd ave, spokane, wa | us`
- TF-IDF Similarity: 0.738

**New Pipeline:**
- Transformed Query: `chase evangeline invesco | 13809 43rd avenue spokane wa | us`
- Transformed True Match: `chase evangeline invesco | 13809 43rd avenue spokane wa | us`
- New TF-IDF Similarity: 1.000
- **Improvement Delta:** +0.262

---

### Query: `ramos and benham fusion | burrell, pa, 378 devinney hollow road | us`
**Old Pipeline:**
- Best True Match: `ramos and benham | 378-380 devinney hollow road, burrell, pa | us`
- TF-IDF Similarity: 0.701

**New Pipeline:**
- Transformed Query: `benham fusion ramos | 378 burrell devinney hollow pa road | us`
- Transformed True Match: `benham fusion ramos | burrell devinney hollow pa road | us`
- New TF-IDF Similarity: 0.961
- **Improvement Delta:** +0.260

---

### Query: `ro atlantic galaxy inc | 2119 tipton street, seymour, in | us`
**Old Pipeline:**
- Best True Match: `ro atlantic | #2119 tipton st, seymour cty, in | us`
- TF-IDF Similarity: 0.622

**New Pipeline:**
- Transformed Query: `atlantic galaxy ro | 2119 in seymour street tipton | us`
- Transformed True Match: `atlantic ro | 2119 cty in seymour street tipton | us`
- New TF-IDF Similarity: 0.874
- **Improvement Delta:** +0.252

---

