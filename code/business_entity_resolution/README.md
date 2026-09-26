# ResolveX: Code Submission Package

This folder contains the complete, self-contained, reproducible source code for the **ResolveX** Business Entity Resolution pipeline.

## Structure
```
code/business_entity_resolution/
├── src/
│   ├── preprocessing.py       # normalization, abbreviation expansion, tokenization
│   ├── blocking.py            # multi-strategy candidate generation
│   ├── features.py            # pairwise feature engineering
│   ├── train_matcher.py       # LightGBM training + threshold tuning
│   ├── infer.py                # end-to-end inference: blocking → features → scoring → output
│   ├── evaluate.py             # local F_0.5 macro-average evaluator
│   ├── generate_outputs.py     # writes matching_results.tsv + candidate_pairs.tsv
│   └── app.py                  # interactive Streamlit dashboard
├── README.md                   # reproduction steps
└── requirements.txt             # pinned dependencies
```

## Quick Reproduction
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model and tune threshold
python -m src.train_matcher

# 3. Generate test outputs
python -m src.generate_outputs
```
All outputs are saved to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
