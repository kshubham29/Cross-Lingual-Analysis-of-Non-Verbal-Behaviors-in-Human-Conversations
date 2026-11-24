# How to run this project in VS Code

Cross-Lingual Prediction of Engagement from Non-Verbal Behaviour (NoXi), 24-conversation version. Shubham Kumar (23CSU388).

## 1. Set up (once)
1. Install Python 3.12 and VS Code (extensions: Python, Jupyter).
2. Unzip the project and open the folder in VS Code (File > Open Folder).
3. Open a terminal in VS Code (Terminal > New Terminal) and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Press Ctrl+Shift+P > "Python: Select Interpreter" and choose `.venv`.

## 2. What runs WITHOUT the NoXi data (everything is already in `reports/`)
- Notebooks **18** and **28 to 35** (master results, review experiments, final models, smoothing, language probe, speed and size, SHAP, expert vs novice, error analysis): they only load the saved result tables in `reports/*.csv` and draw the tables and charts. Open one in VS Code and press "Run All" (takes seconds), or from the terminal:
  ```
  python run.py 18 28 29 30 31 32 33 34 35
  ```
- `python tools/make_notebooks24.py` rebuilds notebooks 18, 28 and 29 (`tools/make_notebooks_30_35.py` rebuilds 30-35).
- `python tools/paired_stats.py` and `python tools/make_response_simple.py` read only `reports/*.csv`.

## 3. What needs the NoXi data (not included: End-User Licence Agreement)
Request access from the University of Augsburg (noxi@hcai.eu), then:
1. `python tools/data24/organise.py`  (set the folder with the NoXi zips at the top of the file)
2. `python tools/data24/build_features24.py`, then `python tools/data24/build_sequences.py`
3. Experiments, one at a time (they print progress lines and write to `reports/`):
   `python src/28_review_experiments.py`, `python src/29_final_models.py classic|ann|tr10|dense`, `python src/30_causal_smoothing.py dense`,
   `python src/31_language_probe.py`, `python src/32_latency_size.py`, `python src/33_shap_features.py`, `python src/34_role_analysis.py`, `python src/35_error_analysis.py`
4. Notebooks 01-13 (data and development steps, random split) need the data; run them with `python run.py 03` etc.
5. `python tools/make_results24.py` rebuilds `Cross Lingual 24 Conversations Results.docx`; `tools/make_response24.py` and `tools/make_response_simple.py` rebuild the two replies to the review.

Tip for slow laptops: set `OMP_NUM_THREADS=8` before long runs; the heavy ones (Transformer, dense head with context) take 1-3 hours.

## 4. Folders
| Folder | Content |
|---|---|
| `notebooks/` | 19 notebooks (01-04, 06, 07, 09, 11, 12, 13, 18, 28-35); each starts with an 'In short' summary |
| `src/`, `tools/` | all scripts |
| `reports/` | result tables (csv), logs and figures |
| `charts/` | saved notebook charts and the error-analysis figures |
| `documents/` | results document, the two replies to the review, paper v5, viva files |
| `HANDOVER_PHASE7.md` | full history, decisions and results |
