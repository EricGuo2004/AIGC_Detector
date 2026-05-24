# Worker Machine Setup

This file is for classmates who run extra AIGC detector experiments on a separate
machine. Do not commit or upload the GenImage data, feature caches, or experiment
outputs to GitHub.

## 1. Clone the code

```powershell
git clone https://github.com/EricGuo2004/AIGC_Detector.git
cd AIGC_Detector
```

## 2. Prepare Python

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test.py --help
```

The help output should include:

```text
--feature-profile
--model-architecture
--train-augmentation
--calibrate-threshold
```

## 3. Prepare GenImage data

Put the dataset outside the Git repository. One recommended layout is:

```text
D:\AIGC_Worker\GenImage_data\
  ADM\
  BigGAN\
  VQDM\
  glide\
```

Each generator directory should contain:

```text
train\ai
train\nature
val\ai
val\nature
```

## 4. Run an assigned experiment

Use a unique `--out-dir` for each machine. For the current 4-generator
extension stage, do not repeat `outputs_4gen_20pct_fusion_robust_aug`; the main
machine is already running it. The most useful worker jobs are below.

Set the data path first:

```powershell
$DATA = "D:\AIGC_Worker\GenImage_data"
```

### Worker A: stable_freq clean + robustness

```powershell
.\.venv\Scripts\python.exe test.py `
  --dataset-root $DATA `
  --out-dir outputs_worker_stable_freq_20pct `
  --sample-fraction 0.20 `
  --sample-seed 42 `
  --skip-robustness `
  --lightgbm-device cpu `
  --num-workers 12 `
  --feature-chunksize 64 `
  --feature-cache-dir feature_cache_stable `
  --sample-cache-dir sample_cache `
  --feature-set all `
  --feature-profile stable_freq `
  --lgbm-profile wide `
  --model-set lightgbm `
  --model-architecture flat `
  --train-augmentation none `
  --calibrate-threshold `
  --resume-completed-tasks

.\.venv\Scripts\python.exe scripts\evaluate_best_robustness.py `
  --dataset-root $DATA `
  --model-output outputs_worker_stable_freq_20pct `
  --out-dir outputs_worker_stable_freq_20pct_robust_20pct `
  --sample-fraction 0.20 `
  --sample-seed 42 `
  --tasks both `
  --num-workers 12 `
  --feature-chunksize 64 `
  --robust-cache-dir robustness_cache_stable
```

### Worker B: mild_aug clean + robustness

```powershell
.\.venv\Scripts\python.exe test.py `
  --dataset-root $DATA `
  --out-dir outputs_worker_fusion_mild_aug_20pct `
  --sample-fraction 0.20 `
  --sample-seed 42 `
  --skip-robustness `
  --lightgbm-device cpu `
  --num-workers 12 `
  --feature-chunksize 64 `
  --feature-cache-dir feature_cache_fusion `
  --sample-cache-dir sample_cache `
  --feature-set all `
  --feature-profile fusion_freq `
  --lgbm-profile wide `
  --model-set lightgbm `
  --model-architecture flat `
  --train-augmentation mild_freq `
  --calibrate-threshold `
  --resume-completed-tasks

.\.venv\Scripts\python.exe scripts\evaluate_best_robustness.py `
  --dataset-root $DATA `
  --model-output outputs_worker_fusion_mild_aug_20pct `
  --out-dir outputs_worker_fusion_mild_aug_20pct_robust_20pct `
  --sample-fraction 0.20 `
  --sample-seed 42 `
  --tasks both `
  --num-workers 12 `
  --feature-chunksize 64 `
  --robust-cache-dir robustness_cache_fusion
```

If the machine has a working LightGBM GPU setup, `--lightgbm-device cpu` can be
changed to `--lightgbm-device gpu`. CPU mode is the safest default.

Older example:

```powershell
.\.venv\Scripts\python.exe test.py `
  --dataset-root D:\AIGC_Worker\GenImage_data `
  --out-dir outputs_v2_5pct_color_freq_hierarchical_attribution `
  --sample-fraction 0.05 `
  --sample-seed 42 `
  --skip-robustness `
  --lightgbm-device cpu `
  --num-workers 12 `
  --feature-chunksize 32 `
  --feature-cache-dir feature_cache_color `
  --sample-cache-dir sample_cache `
  --feature-set all `
  --feature-profile color_freq `
  --lgbm-profile wide `
  --model-set lightgbm `
  --model-architecture hierarchical_attribution `
  --train-augmentation none `
  --calibrate-threshold
```

## 5. Return results

After the command finishes, send back only the assigned output directories, for
example `outputs_worker_stable_freq_20pct` and
`outputs_worker_stable_freq_20pct_robust_20pct`.
Do not send back:

```text
GenImage_data
.venv
feature_cache*
sample_cache
experiment_logs
```

The main machine will merge result folders and rebuild the report tables.
