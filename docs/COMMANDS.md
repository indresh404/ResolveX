# ResolveX: Step-by-Step Execution Commands

Follow these exact steps in order from your terminal in the project directory (`D:\college\PROJECTS\Amazon2026`).

---

## Step 1: Activate Virtual Environment

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS / Git Bash:**
```bash
source .venv/bin/activate
```

---

## Step 2: (Optional) Train Model & Optimize Decision Threshold

Train the precision-tuned LightGBM classifier with hard negative sampling and sweep thresholds to maximize Macro $F_{0.5}$:

```powershell
python -m src.train_matcher
```

*Artifacts saved:*
- `models/lgbm_matcher.pkl`
- `models/config.json`

---

## Step 3: Generate Submission Files

Run end-to-end inference over the test dataset to generate your leaderboard submission file and blocking candidate file:

```powershell
python -m src.generate_outputs `
  --test-s1 student_resource/dataset/test/test_source1.tsv `
  --test-s2 student_resource/dataset/test/test_source2.tsv `
  --test-s3 student_resource/dataset/test/test_source3.tsv `
  --output-dir output
```

*Files generated in `output/`:*
- **`output/matching_results.tsv`** (Your leaderboard submission file)
- **`output/candidate_pairs.tsv`** (Your blocking candidate audit file)

---

## Step 4: Validate Before Uploading

Run the challenge validator to ensure your files pass 100% of the format and constraint checks:

```powershell
python student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```

*Expected output: `PASS (exit 0)`*

---

## Step 5: Upload to Leaderboard

1. Open the competition portal.
2. Go to the **Submission / Upload** page.
3. Select and upload: **`output/matching_results.tsv`**.

---

## Step 6: (Optional) Create Final Submission Zip

Package all code, outputs, and documentation for the final challenge review:

```powershell
Compress-Archive -Path output, code, Documentation_template.md -DestinationPath ResolveX_submission.zip -Force
```

---

## Step 7: (Optional) Launch Interactive UI Dashboard

To visually inspect matches, test custom records, and analyze threshold curves:

```powershell
streamlit run src/app.py
```

Or double-click and open in any browser:
- `docs/entity_resolution_dashboard.html`

---

## Important Commands Summary

| Action | Command |
| :--- | :--- |
| **Activate Environment** | `.\.venv\Scripts\Activate.ps1` |
| **Train Model** | `python -m src.train_matcher` |
| **Generate Submission TSVs** | `python -m src.generate_outputs --output-dir output` |
| **Validate Submission** | `python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --test-dir student_resource/dataset/test` |
| **Launch Visual Dashboard** | `streamlit run src/app.py` |
| **Package Submission Zip** | `Compress-Archive -Path output, code, Documentation_template.md -DestinationPath ResolveX_submission.zip -Force` |
