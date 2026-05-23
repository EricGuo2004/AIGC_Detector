# AIGC Frequency Fingerprint Detection

This repository contains the final-project implementation for frequency-domain
AIGC image detection on GenImage. The project focuses on interpretable spectral
features rather than CNN/ViT detectors.

## Current Final Result

Primary run:

```text
outputs_v2_full_best
feature_profile = fusion_freq
model_architecture = flat
model = LightGBM wide profile
train_augmentation = none
```

Clean validation performance:

| Task | Accuracy | Macro-F1 | AUC |
| --- | ---: | ---: | ---: |
| AI vs Nature | 0.9913 | 0.9913 | 0.9996 |
| Source attribution | 0.9991 | 0.9991 | - |

Robustness is evaluated separately on a 20% validation subset under JPEG,
resize, and Gaussian-noise perturbations. The final report treats robustness as
a diagnostic result: the `fusion_freq` model is very strong on clean data but
degrades under image post-processing.

## Data Layout

Do not commit GenImage data to GitHub. Put it outside the repository, for
example:

```text
C:\Users\99303\git\GenImage_data\
  ADM\
    train\ai
    train\nature
    val\ai
    val\nature
  BigGAN\
  VQDM\
  glide\
```

The project uses four GenImage generators: ADM, BigGAN, VQDM, and GLIDE.
`train` is used for training; `val` is used as the test split.

## Installation

```powershell
git clone https://github.com/EricGuo2004/AIGC_Detector.git
cd AIGC_Detector
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Core Commands

Quick smoke test:

```powershell
.\.venv\Scripts\python.exe test.py `
  --dataset-root C:\Users\99303\git\GenImage_data `
  --out-dir outputs_smoke_min `
  --sample-fraction 0.001 `
  --skip-robustness
```

Re-run the current best clean route:

```powershell
.\.venv\Scripts\python.exe test.py `
  --dataset-root C:\Users\99303\git\GenImage_data `
  --out-dir outputs_v2_full_best `
  --sample-fraction 1.0 `
  --sample-seed 42 `
  --skip-robustness `
  --lightgbm-device gpu `
  --num-workers 16 `
  --feature-chunksize 64 `
  --feature-cache-dir feature_cache_fusion `
  --sample-cache-dir sample_cache `
  --feature-set all `
  --feature-profile fusion_freq `
  --lgbm-profile wide `
  --model-set lightgbm `
  --model-architecture flat `
  --train-augmentation none `
  --calibrate-threshold `
  --resume-completed-tasks
```

Evaluate robustness for the saved best model without retraining:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_best_robustness.py `
  --dataset-root C:\Users\99303\git\GenImage_data `
  --model-output outputs_v2_full_best `
  --out-dir outputs_v2_full_best_robust_20pct `
  --sample-fraction 0.20 `
  --sample-seed 42 `
  --tasks both `
  --num-workers 16 `
  --feature-chunksize 64 `
  --robust-cache-dir robustness_cache_fusion
```

Build report assets:

```powershell
.\.venv\Scripts\python.exe scripts\build_report_assets.py `
  --dataset-root C:\Users\99303\git\GenImage_data `
  --report-dir report `
  --primary-output outputs_v2_full_best `
  --robustness-output outputs_v2_full_best_robust_20pct `
  --robustness-compare-outputs outputs_v2_full_best_robust_20pct outputs_4gen_full_best_robust_20pct
```

Validate the final submission loop:

```powershell
.\.venv\Scripts\python.exe scripts\validate_submission.py `
  --dataset-root C:\Users\99303\git\GenImage_data `
  --write-report
```

This checks data structure, final metrics, robustness CSVs, report assets,
Notebook JSON, PDF page count, and Git hygiene. The generated checklist is
written to `report/submission_checklist.md`.

## Outputs

Each experiment output contains:

```text
<out-dir>/
  binary_ai_vs_nature/
    best_model.joblib
    metrics_summary.json
    model_comparison.csv
    confusion_matrix_lightgbm.csv
    feature_importance.csv
  ai_subsource_attribution/
    best_model.joblib
    metrics_summary.json
    model_comparison.csv
    confusion_matrix_lightgbm.csv
    feature_importance.csv
```

Robustness outputs contain one `robustness_results.csv` per task, plus
attack-level confusion matrices.

## Final Submission Assets

The current final-submission materials are:

```text
report/
  main.tex
  main.pdf
  references.bib
  figures/
  tables/
notebooks/
  final_project_aigc_detector.ipynb
```

Large local directories such as `GenImage_data`, `feature_cache*`,
`robustness_cache*`, `sample_cache`, `.venv`, and full `outputs_*` runs should
not be uploaded unless explicitly required by the course submission format.
