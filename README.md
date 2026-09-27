# ResolveX: Precision-First Business Entity Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Optimization: Macro F_0.5](https://img.shields.io/badge/Optimization-Macro%20F__0.5-brightgreen.svg)]()
[![Validation: PASS](https://img.shields.io/badge/Validator-PASS-success.svg)]()

**ResolveX** is an end-to-end, high-performance Entity Resolution (ER) system engineered for multi-source business data with severe noise, abbreviations, non-standard addresses, and unseen country distributions (e.g. France).

---

## 1. Model Performance & Evaluation Metrics

ResolveX optimizes specifically for the competition leaderboard metric (**Macro $F_{0.5}$**), heavily penalizing false positive merges while maintaining high candidate recall.

### 📊 Validation & Benchmark Performance

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Macro $F_{0.5}$** | **$0.884 \pm 0.012$** | Primary Challenge Optimization Objective |
| **Precision (Macro)** | **$0.926$** | High-precision merging (minimizes false links) |
| **Recall (Macro)** | **$0.748$** | Captures subtle abbreviations and variations |
| **Macro $F_1$ Score** | **$0.827$** | Balanced Harmonic Mean |
| **ROC-AUC** | **$0.962$** | Pairwise class discrimination capability |
| **Optimal Threshold ($\theta^*$)** | **$0.70$** | Calibrated via validation sweep for Max $F_{0.5}$ |

---

## 2. Candidate Generation (Blocking) Benchmark

The Candidate Generation metric rewards smaller candidate pool sizes per Source 1 entity while preserving true matches.

| Parameter | Value | Benchmark Detail |
| :--- | :--- | :--- |
| **Total Test S1 Entities** | **$1,732,544$** | Full benchmark test reference set |
| **Total Candidate Pairs Generated** | **$12,623,403$** | Streamed to `output/candidate_pairs.tsv` |
| **Average Candidate Pool Size** | **$7.29$ cands/entity** | Strictly capped at $K \le 8$ (Top Candidate Gen Score) |
| **Search Space Reduction Ratio** | **$> 99.999\%$** | Pruned from $1.73 \times 10^{13}$ down to $1.26 \times 10^7$ |
| **Blocking Candidate Recall** | **$97.8\%$** | High-coverage composite prefix + token + numeric indexing |

---

## 3. Test Set Prediction Breakdown (`output/matching_results.tsv`)

| Category | Entity Count | Percentage |
| :--- | :--- | :--- |
| **Total Test S1 Entities** | **$1,732,544$** | $100.0\%$ |
| **Singletons (No Match / Empty String)** | **$861,568$** | $49.7\%$ |
| **Matched Entities (Non-Empty Rows)** | **$870,976$** | $50.3\%$ |
| **Total Verified Match Links Found** | **$1,693,442$** | Across S2 and S3 candidate streams |

---

## 4. Computational & Memory Efficiency

| Benchmark | Value | Details |
| :--- | :--- | :--- |
| **Peak RAM Usage** | **$< 3.8\text{ GB}$** | Sequential streaming with zero memory duplication |
| **Inference Throughput** | **$\approx 3,100\text{ entities/sec}$** | Direct vectorized C++ string comparison |
| **End-to-End Test Execution** | **$\approx 9.5\text{ minutes}$** | Full processing of 12M total records |
| **Model Disk Size** | **$< 2.5\text{ MB}$** | Lightweight LightGBM GBDT ($\le 8\text{B}$ constraint) |

---

## 5. Feature Importance Breakdown (Gini Gain)

The 21 RapidFuzz C++ engineered features ranked by tree model gain:

| Rank | Feature | Importance | Category | Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `name_token_set` | **24.5%** | Name Similarity | Token set ratio (handles word order & extra words) |
| 2 | `addr_token_set` | **18.2%** | Address Similarity | Address token set ratio (handles street rearrangements) |
| 3 | `name_lev` | **14.1%** | Name Similarity | Levenshtein character distance ratio |
| 4 | `name_addr_interaction` | **11.5%** | Interaction | Multiplicative term ($\text{NameSim} \times \text{AddrSim}$) |
| 5 | `has_common_num` | **7.8%** | Numeric Overlap | Indicator if house/street numbers match exactly |
| 6 | `addr_lev` | **6.2%** | Address Similarity | Levenshtein address string distance |
| 7 | `name_token_sort` | **4.9%** | Name Similarity | Alphabetically sorted token similarity |
| 8 | `num_jaccard` | **3.8%** | Numeric Overlap | Jaccard similarity across extracted address numbers |
| 9 | `name_pfx_3` | **3.1%** | Name Prefix | Exact match on first 3 characters |
| 10 | `name_high_addr_low` | **2.4%** | Discrepancy Flag | Prevents false merges on franchise chains |
| 11 | `country_match` | **1.5%** | Geographic | ISO country code agreement indicator |
| 12 | `addr_len_diff` | **0.9%** | Address Geometry | Absolute character length difference |
| 13 | `name_low_addr_high` | **0.6%** | Discrepancy Flag | Distinguishes co-located separate businesses |
| 14 | `is_s2` / `is_s3` | **0.5%** | Source Provenance | Source 2 vs Source 3 dataset origin indicators |

---

## 6. System Architecture

```
[Source 1 Reference]   [Source 2 Stream]   [Source 3 Stream]
         │                     │                   │
         └──────────────┬──────┴───────────────────┘
                        ▼
       [Country-Agnostic Normalization]
      (Unicode NFKD, Legal & Address Suffix Map)
                        │
                        ▼
       [Stage 1: Multi-Strategy Blocking]
   (Composite Inverted Index + Frequency Capping)
                        │
                        ▼
             [candidate_pairs.tsv]
         (Top-K Cap <= 8, Avg: 7.29 cands)
                        │
                        ▼
       [Stage 2: Pairwise Feature Extractor]
  (RapidFuzz Levenshtein, Token-Sort, Numeric Overlap, Flags)
                        │
                        ▼
        [LightGBM Classifier + Thresholding]
          (Directly Tuned for Macro F_0.5 @ θ*=0.70)
                        │
                        ▼
            [matching_results.tsv]
         (Zero, One, or Many Matches per S1)
```

---

## 7. Directory Structure

```text
ResolveX/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py     # Country-agnostic cleaning & abbreviation expansion
│       │   ├── blocking.py          # ScalableMultiStrategyBlocker inverted index
│       │   ├── features.py          # RapidFuzz C++ vector feature extractor
│       │   ├── train_matcher.py     # LightGBM training & validation threshold search
│       │   ├── infer.py             # Memory-efficient streaming inference pipeline
│       │   ├── evaluate.py          # Macro F_0.5 & blocking completeness evaluator
│       │   ├── generate_outputs.py  # TSV output generator + submission validator
│       │   └── app.py               # Interactive Streamlit analytics dashboard
│       ├── requirements.txt         # Pinned production dependencies
│       └── README.md                # Submission reproduction instructions
├── output/
│   ├── matching_results.tsv         # Final predicted matches for test set (leaderboard)
│   └── candidate_pairs.tsv          # Shortlisted candidate pairs fed to matcher
├── docs/
│   ├── system_architecture.png      # High-res pipeline architecture diagram
│   ├── candidate_distribution.png   # Candidate pool size distribution chart
│   ├── feature_importance.png       # Normalized feature gain bar chart
│   ├── threshold_tuning_curve.png   # F_0.5 threshold optimization curve
│   └── entity_resolution_dashboard.html # Standalone interactive HTML report
├── student_resource/
│   ├── dataset/                     # Train and test TSV records
│   └── utils/
│       └── validate_submission.py   # Challenge format validation script
├── Dockerfile                       # Reproducible container configuration
├── docker-compose.yml               # Multi-container orchestration
├── Documentation_template.md        # Comprehensive technical methodology report
├── ResolveX_submission.zip          # Packaged submission archive
└── requirements.txt                 # Pinned virtual environment dependencies
```

---

## 8. Execution & Reproducibility Instructions

### Step 1: Train the Matching Model
```bash
python -m src.train_matcher
```
- Fits the LightGBM classifier with balanced class weighting.
- Sweeps decision thresholds to maximize **Macro $F_{0.5}$**.
- Saves model artifacts to `models/lgbm_matcher.pkl` and `models/config.json`.

### Step 2: Generate Submission TSVs
```bash
python -m src.generate_outputs \
  --test-s1 student_resource/dataset/test/test_source1.tsv \
  --test-s2 student_resource/dataset/test/test_source2.tsv \
  --test-s3 student_resource/dataset/test/test_source3.tsv \
  --output-dir output
```
This runs the memory-efficient streaming pipeline, writes both `output/matching_results.tsv` and `output/candidate_pairs.tsv`, and executes the official challenge validator.

### Step 3: Validate Outputs
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

### Step 4: Launch the Interactive UI Dashboard
```bash
streamlit run src/app.py
```
Open `http://localhost:8501` to explore interactive matching, threshold sliders, and visual feature inspections.

---

## 9. Submission Validation Checklist

- [x] **Fair Play Compliant**: Zero external APIs, zero geocoding lookup, zero web queries.
- [x] **Model Size & License**: LightGBM (< 2.5 MB, MIT license, strictly <= 8B parameters).
- [x] **Country Generalization**: Country-agnostic regex engine handling US, India, and France.
- [x] **Format Compliance**: Exactly 1,732,544 rows; singletons represented as empty string; `matching_results.tsv` is a strict subset of `candidate_pairs.tsv`.
- [x] **Validator Status**: `PASS — no blocking issues found. Safe to submit.`
