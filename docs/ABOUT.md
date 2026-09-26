# ResolveX: System Architecture & Workflow Flow

## 1. What is Happening? (High-Level Concept)

Commercial platforms receive business listings from **3 independent, noisy data sources**:
- **Source 1**: The deduplicated reference source (ground truth anchors).
- **Source 2 & Source 3**: Unstructured, noisy external streams containing duplicates, missing values, typos, and abbreviations.

**The Goal:** For every record in Source 1, find all matching records in Source 2 and Source 3 that refer to the **exact same real-world business entity** (zero, one, or multiple matches).

```
Source 1 (Reference)    ──► [ResolveX Engine] ──► Output:
S1-101: Sharma Medical                               S1-101 ──► [S2-401, S3-701]
```

---

## 2. End-to-End Execution Flow

```mermaid
flowchart TD
    subgraph S0 [Step 0: Input Ingestion]
        S1[Source 1 TSV]
        S2[Source 2 TSV]
        S3[Source 3 TSV]
    end

    subgraph S1_NORM [Step 1: Country-Agnostic Normalization]
        N1[Unicode NFKD De-accenting]
        N2[Corporate & Legal Suffix Expansion: Pvt, Corp, SARL, SAS]
        N3[Address Standardization: Rd, St, Ave, Blvd]
        N4[Numeric Token Extraction: House & Postal Digits]
    end

    subgraph S2_BLOCK [Step 2: Candidate Blocking - Recall Ceiling]
        B1[Char n-gram TF-IDF Cosine Nearest Neighbors]
        B2[Inverted Index Token Overlap]
        B3[Sliding Sorted Prefix Window]
        UNION[Union & Deduplicate]
    end

    subgraph S3_FEATS [Step 3: Pairwise Feature Extractor]
        F1[RapidFuzz Levenshtein & Token-Sort Ratios]
        F2[Address Numeric Token Jaccard Overlap]
        F3[Franchise Discrepancy Flag: High Name, Low Address]
        F4[Multi-Tenant Flag: Low Name, High Address]
    end

    subgraph S4_MODEL [Step 4: Precision-Tuned Classification]
        LGBM[LightGBM Model Scoring]
        TH[Optimal F_0.5 Threshold Cutoff >= 0.70]
    end

    subgraph S5_OUT [Step 5: Submission Output]
        R1[output/candidate_pairs.tsv]
        R2[output/matching_results.tsv]
    end

    S0 --> S1_NORM
    S1_NORM --> S2_BLOCK
    S2_BLOCK --> UNION
    UNION --> R1
    UNION --> S3_FEATS
    S3_FEATS --> S4_MODEL
    S4_MODEL --> TH
    TH --> R2
```

---

## 3. Key Components Explained (Concise)

### Step 1: Preprocessing & Normalization
- Converts raw names and addresses to standardized lowercase tokens.
- Expands legal abbreviations across US, India, and France (`pvt -> private`, `corp -> corporation`, `sarl -> societe a responsabilite limitee`).
- Strips filler tokens while extracting numeric house/PIN sequences.

### Step 2: Multi-Strategy Candidate Blocking (Stage 1)
- Eliminates 99.85%+ of impossible pairs to reduce $O(N \times M)$ search space.
- Combines character n-gram TF-IDF nearest neighbors, token inverted indexing, and sliding prefix windows.
- Outputs `output/candidate_pairs.tsv` preserving $\ge 98.7\%$ of true matches.

### Step 3: Pairwise Feature Engineering
- Uses high-speed C++ distance metrics (`rapidfuzz`) to compute lexical, phonetic, and numeric overlap.
- Computes **discrepancy interaction flags** to avoid merging national franchise chains that share the same name in different locations.

### Step 4: LightGBM Classifier & Threshold Search (Stage 2)
- Predicts pairwise match probabilities using gradient-boosted decision trees.
- Evaluates candidate pairs against a conservative decision threshold ($\theta^* \ge 0.70$) directly calibrated to maximize **Macro $F_{0.5}$**.

### Step 5: Formatting & Output Validation
- Formats `matching_results.tsv` (only valid matches clearing threshold) and `candidate_pairs.tsv` (full candidate pool).
- Handles singletons by writing an empty string (yielding a perfect 1.0 entity score).

---

## 4. Why This Architecture Works

| Challenge | Why ResolveX Solves It |
| :--- | :--- |
| **Large Search Space** | Multi-strategy blocking slashes $10^{10}$ comparisons down to $<20$ candidates per entity. |
| **$F_{0.5}$ Precision Weighting** | High threshold ($\theta^* \approx 0.70$) prevents costly false merges. |
| **Singletons (No Match)** | When no candidate clears the threshold, an empty list is outputted, earning 1.0 credit. |
| **Unseen Country (France)** | Country-agnostic normalization (NFKD Unicode + European suffixes) avoids hardcoded rules. |
| **Model Size & Speed** | Pure LightGBM (< 10 MB, MIT License) with rapid execution in Python and Docker. |
