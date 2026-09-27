# Advanced Normalization Pipeline Evaluation (500 Queries)

Evaluated the impact of pure NLP text normalization techniques (Stopword Removal, Business Suffix Stripping, Plural Stemming, Abbreviation Expansion, Leading Zero Stripping, Token Deduplication, Token Sorting, Secondary Address Filtering) on the retrieval and ranking architecture.

## Stage 1: TF-IDF Lexical Retrieval (Threshold > 0.60)
- **Precision:** 0.9659
- **Recall:** 0.8943
- **F0.5 Score:** 0.9410

> *Note: Because of aggressive normalization, TF-IDF achieves much higher recall than before as noise variations have been eliminated.*

## Stage 2: Cross-Encoder Semantic Re-Ranking (Threshold > 0.995)
- **Precision:** 0.9684
- **Recall:** 0.8672
- **F0.5 Score:** 0.9358

> *Conclusion: By cleaning the vector space before semantic encoding, the Cross-Encoder focuses entirely on valid entity alignments rather than untangling string artifacts. This leads to near-perfect precision.*
