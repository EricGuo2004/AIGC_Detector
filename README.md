# AIGC Frequency Fingerprint Detection (Final Project)

This project implements a frequency-domain AIGC detector based on your requested setup:

- `train` is the training split
- `val` is the test split
- `ai` means AI-generated images
- `nature` means natural/real images

It uses FFT/DCT spectrum features + tree-based models (LightGBM as main model) to do:

1. Binary detection: `ai` vs `nature`
2. Optional attribution: classify AI source (`ADM_SELECTED`, `BIGGan_selected`, `VDQM_selected`, `glide_selected`) when these are organized as subfolders under `ai`

---

## 1) Folder Structure

Prepare data in either of these formats.

### Format A (single merged root)

```text
data/
  train/
    ai/
      *.jpg|png...
    nature/
      *.jpg|png...
  val/
    ai/
      *.jpg|png...
    nature/
      *.jpg|png...
```

If you also want source attribution, organize AI images with one more level:

```text
data/
  train/
    ai/
      ADM_SELECTED/
      BIGGan_selected/
      VDQM_selected/
      glide_selected/
  val/
    ai/
      ADM_SELECTED/
      BIGGan_selected/
      VDQM_selected/
      glide_selected/
```

`nature` can still remain in `train/nature` and `val/nature` for binary classification.

### Format B (your current multi-root style)

```text
Final_Project/
  ADM_selected/imagenet_ai_0508_adm/
    train/ai, train/nature, val/ai, val/nature
  BigGAN_selected/imagenet_ai_0419_biggan/
    train/ai, train/nature, val/ai, val/nature
  VQDM_selected/imagenet_ai_0419_vqdm/
    train/ai, train/nature, val/ai, val/nature
  glide_selected/imagenet_glide/
    train/ai, train/nature, val/ai, val/nature
```

The script auto-detects all valid roots and merges them for binary training.
For attribution, each root is treated as one AI source label.

---

## 2) Installation

```bash
pip install -r requirements.txt
```

---

## 3) Train and Evaluate

Run from `Final_Project`:

Format A example:

```bash
python train.py --dataset-root data --out-dir outputs
```

Format B example (recommended for your folder now):

```bash
python train.py --dataset-root . --out-dir outputs
```

Optional arguments:

- `--image-size` (default `256`)
- `--radial-bins` (default `64`)
- `--angular-bins` (default `18`)
- `--patch-grid` (default `4`)
- `--skip-robustness` (skip JPEG/resize/noise robustness testing)

Quick bug-check run (same pipeline with 1/10 data per class, using `test.py`):

```bash
python test.py --dataset-root . --out-dir outputs_smoke
```

---

## 4) What the Pipeline Does

### Feature extraction (frequency-domain)
- 2D FFT power spectrum (`log(1 + |F|^2)`)
- Radial PSD bins
- Angular spectrum bins
- Low/Mid/High band energy ratios
- Spectral slope (`log-power` vs `log-frequency`)
- High-frequency residual statistics
- Patch-level high-frequency variation
- 2D DCT radial profile (complementary cue)

### Models
- LightGBM (main)
- RandomForest (baseline)
- LogisticRegression (baseline)

Best model is selected by `macro_f1` then `accuracy` on `val`.

### Which part matches your plan tasks

- **Plan Task 1: `是不是AI生成`**
  - Code entry: `train.py` -> call with `task_name="binary_ai_vs_nature"`
  - Data used: `train/ai + train/nature` for training, `val/ai + val/nature` for testing
  - Output folder: `outputs/binary_ai_vs_nature/`

- **Plan Task 2: `来自哪个AI模型`**
  - Code entry: `train.py` -> call with `task_name="ai_subsource_attribution"`
  - Data used:
    - Format A: `train/ai/<subsource>/...` and `val/ai/<subsource>/...`
    - Format B (your current folders): each model root is one class label
  - Output folder: `outputs/ai_subsource_attribution/`

- **Plan Task 3: `鲁棒性测试`**
  - Code entry: `src/robustness.py` -> `evaluate_robustness(...)`
  - Trigger: enabled by default during `binary_ai_vs_nature` task
  - Output file: `outputs/binary_ai_vs_nature/robustness_results.csv`

- **Progress bars you will see**
  - Feature extraction: in `train.py`
  - Candidate model training: in `src/training.py`
  - Robustness attack evaluation: in `src/robustness.py`

### Robustness tests (binary task)
- JPEG compression: `Q=95/75/50`
- Resize attack: `0.5x / 0.75x / 1.5x` and back-resize
- Gaussian noise: `sigma=2/5/10`

---

## 5) Outputs

After training, results are saved under:

```text
outputs/
  binary_ai_vs_nature/
    best_model.joblib
    model_comparison.csv
    metrics_summary.json
    feature_importance.csv
    classification_report_<model>.txt
    confusion_matrix_<model>.csv
    robustness_results.csv

  ai_subsource_attribution/   (only if ai subfolders are found)
    best_model.joblib
    model_comparison.csv
    metrics_summary.json
    feature_importance.csv
    classification_report_<model>.txt
    confusion_matrix_<model>.csv
```

---

## 6) Notes

- If your dataset is large, first run with fewer images to sanity-check the pipeline.
- For final report, use:
  - `metrics_summary.json`
  - `robustness_results.csv`
  - top rows of `feature_importance.csv` as key frequency sentinels
- Attribution task is automatically skipped if `ai` subfolders are not present.
