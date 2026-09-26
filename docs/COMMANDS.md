# ResolveX: Comprehensive Command Reference Manual

This guide contains the complete command reference for setting up, training, inferring, evaluating, visualizing, and containerizing the **ResolveX** Business Entity Resolution pipeline.

---

## 1. Environment Setup & Activation

### Step 1: Create Virtual Environment
Creates an isolated Python 3.11 virtual environment named `.venv` in the project root:
```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Windows Command Prompt (cmd.exe):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS / Git Bash:**
```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies
Upgrades `pip` and installs all pinned production requirements:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Interactive UI & Analytics Dashboards

### A. Launch Streamlit Web Application
Starts the full interactive analysis platform with live entity matching, feature inspection, and threshold sensitivity curves:
```powershell
streamlit run src/app.py
```
- **Access URL**: `http://localhost:8501`
- **Features**:
  - Interactive pipeline architecture diagram.
  - Pairwise Candidate Inspector (select reference S1 entity or type custom strings).
  - Decision Threshold ($F_{0.5}$) Trade-off Curve Simulator.
  - Zero-shot validation test bench for French records (*SARL*, *SAS*, *Société*).

### B. Open Standalone HTML Dashboard (Zero Setup)
Open [docs/entity_resolution_dashboard.html](file:///d:/college/PROJECTS/Amazon2026/docs/entity_resolution_dashboard.html) directly in any web browser (Chrome, Edge, Firefox). Requires no server.

---

## 3. Model Training & Threshold Tuning

### Train the Full Matching Model
Trains the LightGBM classifier on the complete dataset with real-time `tqdm` progress bars, sweeps decision thresholds on validation, and outputs evaluation metrics:
```powershell
python -m src.train_matcher
```

### Fast Training Test (Subsampled Smoke Run)
To verify the training loop and progress bars quickly on a sample subset:
```powershell
python -c "from src.train_matcher import train_matching_model; train_matching_model(sample_limit=500)"
```

**What the Training Command Does:**
1. Loads training files (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`) with chunked progress tracking.
2. Applies country-agnostic normalization (NFKD Unicode stripping, legal & address suffix expansion).
3. Splits S1 reference records into 80% Train / 20% Validation (entity-level split).
4. Generates candidate blocks via Multi-Strategy Blocking (TF-IDF + Inverted Index + Prefix Windowing).
5. Samples positive pairs and hard/medium negatives to construct the training set.
6. Trains a precision-weighted LightGBM binary classifier (`LGBMClassifier`).
7. Sweeps decision thresholds $[0.40, 0.95]$ on validation to maximize **Macro $F_{0.5}$**.
8. Saves the trained model to `models/lgbm_matcher.pkl` and configuration to `models/config.json`.

---

## 4. Output Generation & Submission Validation

### Generate Test Submission TSVs
Runs end-to-end inference on the test dataset, generates both required submission TSV files in `output/`, and automatically validates them:
```powershell
python -m src.generate_outputs `
  --test-s1 student_resource/dataset/test/test_source1.tsv `
  --test-s2 student_resource/dataset/test/test_source2.tsv `
  --test-s3 student_resource/dataset/test/test_source3.tsv `
  --output-dir output
```

**Generated Files:**
- `output/matching_results.tsv`: Final high-confidence matches (scored on the competition leaderboard).
- `output/candidate_pairs.tsv`: Candidate set shortlisted by Stage 1 blocking before model scoring.

**Command Options:**
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--test-s1` | `student_resource/dataset/test/test_source1.tsv` | Path to Source 1 test records |
| `--test-s2` | `student_resource/dataset/test/test_source2.tsv` | Path to Source 2 test records |
| `--test-s3` | `student_resource/dataset/test/test_source3.tsv` | Path to Source 3 test records |
| `--output-dir` | `output` | Directory where output TSVs are written |
| `--model-path` | `models/lgbm_matcher.pkl` | Path to trained model binary |
| `--config-path` | `models/config.json` | Path to threshold config JSON |
| `--threshold` | `None` (uses config) | Optional manual threshold override (e.g. `0.72`) |

---

### Run Challenge Validator Script Standalone
Audits `matching_results.tsv` and `candidate_pairs.tsv` against all formatting and constraint rules:
```powershell
python student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```
- **Exit Code 0 (`PASS`)**: Files are completely compliant and safe to upload.
- **Exit Code 1**: Prints a detailed list of formatting issues to resolve.

---

## 5. Dockerized Execution (Container Reproducibility)

### Build the Docker Image
```bash
docker build -t resolvex .
```

### Run Full Pipeline in Docker
Mounts local dataset and output folders, runs end-to-end inference, and outputs validated TSVs:
```bash
docker run --rm \
  -v "${PWD}/student_resource:/app/student_resource:ro" \
  -v "${PWD}/output:/app/output:rw" \
  resolvex
```

### Launch Web Dashboard via Docker Compose
```bash
docker compose up resolvex-dashboard
```
Access the dashboard at `http://localhost:8501`.

---

## 6. Quick Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Running script outside root or missing `sys.path` | Already patched in all `src/*.py` scripts. Always run commands from the project root directory. |
| PowerShell script execution error | Execution policy prevents `.ps1` execution | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your terminal. |
| High memory usage during blocking | Very large dataset indexing | Progress bars display chunked ingestion; candidate blocking processes data partitioned by country. |
| Output TSV format warnings | Incorrect column separators | Our pipeline explicitly uses `sep='\t'` and formats comma-separated entity IDs. |
