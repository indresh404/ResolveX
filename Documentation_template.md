# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ResolveX  
**Team Members:** Indresh & Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

**ResolveX** is an ultra-scalable, country-agnostic business entity resolution system engineered for Amazon-scale record linkage across heterogeneous data sources. By combining linear $O(N)$ multi-strategy inverted index blocking with high-speed C++ string similarity feature extraction and a precision-calibrated LightGBM classifier, ResolveX achieves $>99.999\%$ candidate reduction, compact candidate generation (average 7.29 candidates per entity, strictly capped $\le 8$), and optimal Macro $F_{0.5}$ precision on the test benchmark of 1,732,544 reference entities against 9,969,589 candidate stream records.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and test sets (USA, India, France), several core linkage challenges were identified:
- **Severe Scale & Asymmetry:** Matching 1.73M Source 1 entities against ~10M Source 2/3 records yields $\approx 1.73 \times 10^{13}$ possible comparisons.
- **Heterogeneous Noise Patterns:** High frequency of accented characters (especially in French entities like *Société*, *Château*, *Île-de-France*), non-standard legal entity abbreviations (*GmbH*, *LLC*, *Pvt Ltd*, *SAS*, *SARL*), and diverse address formatting (*St*, *Str*, *Rue*, *Avenue*, *Boulevard*).
- **Franchise Discrepancies & False Positives:** Entities sharing identical brand names (e.g., *Starbucks Coffee*) at distinct physical addresses, and conversely, identical addresses hosting distinct corporate entities.
- **Extreme Class Imbalance:** Matches account for $< 0.05\%$ of all candidate pairs.

### 2.2 Solution Strategy
ResolveX utilizes a decoupled **Two-Stage Architecture**:

```
[Raw Sources: S1, S2, S3]
         │
         ▼
[Country-Agnostic Normalization (NFKD Unicode + Regex Legal/Street Maps)]
         │
         ▼
[Stage 1: Scalable Linear O(N) Inverted Index Blocker] ──► output/candidate_pairs.tsv (Avg: 7.29 cands)
         │
         ▼
[Stage 2: Vectorized RapidFuzz C++ Pairwise Feature Matrix]
         │
         ▼
[Cost-Sensitive LightGBM Classifier (Optimal θ* = 0.70)] ──► output/matching_results.tsv (Macro F0.5 Scored)
```

**Approach Type:** Multi-Strategy Inverted Index Blocking + Vectorized Gradient Boosted Decision Trees (LightGBM).  
**Core Innovation:** Single-pass compiled normalization with NFKD Unicode de-accenting, hit-accumulating composite index with frequency capping to eliminate popular token explosions, and direct numpy-vectorized RapidFuzz feature extraction scoring $\approx 3,000+$ entities/sec with $< 4\text{ GB}$ peak memory.

---

## 3. Candidate Generation (Blocking)

To guarantee high candidate recall while minimizing the candidate set size:
- **Blocking keys used:**
  1. *4-character and 3-character normalized name prefixes* (names $\ge 3$ characters).
  2. *Significant name tokens* ($\ge 3$ characters, top 4 informative tokens).
  3. *Numeric address tokens* (house numbers, street numbers, PIN / postal codes).
- **Candidate pairs generated:** Total **12,623,403** candidate pairs generated for 1,732,544 test S1 entities (**Average: 7.29 candidates / entity**, strictly capped at $K \le 8$).
- **How true matches were preserved:**
  - Multi-pass union across prefix, token, and numeric keys ensures that entities with minor spelling variations or missing street prefixes are captured.
  - Candidate hit accumulation with a conservative RapidFuzz floor ($\text{sim} \ge 0.30$) prunes noise while retaining true positives.

---

## 4. Matching Model

### 4.1 Features Used (21 Vectorized Features)
- **Name Features:** Levenshtein ratio, Token Set ratio, Token Sort ratio, Partial ratio, Exact match boolean, 3-character Prefix match, Length difference.
- **Address Features:** Address Levenshtein ratio, Token Set ratio, Token Sort ratio, Partial ratio, Exact match boolean, Length difference.
- **Numeric & Geographic Features:** Numeric token Jaccard similarity, Common number presence flag, ISO Country match indicator.
- **Interaction & Discrepancy Features:** `name_addr_interaction` ($\text{NameSim} \times \text{AddrSim}$), `name_high_addr_low` (detects distinct franchise branches), `name_low_addr_high` (detects co-located distinct businesses), and source provenance flags (`is_s2`, `is_s3`).

### 4.2 Model Type & Training
- **Classifier:** LightGBM (`LGBMClassifier`) configured with `class_weight='balanced'`, 200 estimators, `learning_rate=0.05`, and `num_leaves=31`.
- **Threshold Selection Method:** Direct validation sweep maximizing the competition objective function:
  $$\text{Macro } F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- The optimal threshold is calibrated at **$\theta^* = 0.70$**, placing high weight on precision to eliminate false positive merges.

---

## 5. Results & Error Analysis

- **Test S1 Entities Processed:** 1,732,544
- **Singletons (No Match):** 861,568 (49.7%)
- **Total Validated Matches Found:** 1,693,442
- **Average Candidates per Entity:** 7.29 (Max 8)
- **Validation Macro $F_{0.5}$ Score:** $0.884 \pm 0.012$
- **Common false positives (mitigated):** Co-located businesses sharing identical street addresses or parent holding companies with subsidiary naming overlaps.
- **Common false negatives (mitigated):** Heavily truncated trade names lacking street address numbers in Source 3.

---

## 6. Conclusion
ResolveX successfully solves enterprise-scale business entity resolution through linear-time composite inverted index blocking and precision-tuned gradient boosting. The system processed 1.73M test entities in streaming mode within ~10 minutes, satisfying all memory, parameter ($\le 8\text{B}$), and zero-API constraints while delivering verified challenge compliance.

---

## Appendix

### A. Code Artefacts
- **Directory:** `code/business_entity_resolution/src/`
- **Source Files:**
  - `src/preprocessing.py`: Country-agnostic regex cleaner & Unicode de-accenting.
  - `src/blocking.py`: `ScalableMultiStrategyBlocker` with low-memory indexing.
  - `src/features.py`: Vectorized RapidFuzz C++ feature extractor.
  - `src/train_matcher.py`: Model training & $F_{0.5}$ threshold sweep.
  - `src/infer.py` & `src/generate_outputs.py`: Streaming submission generator.
- **Execution:**
  ```powershell
  python -m src.generate_outputs --test-s1 student_resource/dataset/test/test_source1.tsv --test-s2 student_resource/dataset/test/test_source2.tsv --test-s3 student_resource/dataset/test/test_source3.tsv --output-dir output
  ```

### B. Visualizations & Result Figures
All charts are rendered and preserved under `docs/`:
1. `docs/system_architecture.png`: End-to-End System Architecture.
2. `docs/candidate_distribution.png`: Candidate Pool Size Distribution & Test Match Breakdown.
3. `docs/feature_importance.png`: LightGBM Feature Importance (Gini Gain).
4. `docs/threshold_tuning_curve.png`: Macro $F_{0.5}$ Threshold Optimization Curve.
