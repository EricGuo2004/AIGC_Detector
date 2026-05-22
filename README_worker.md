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

Use a unique `--out-dir` for each machine. Example:

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

If the machine has a working LightGBM GPU setup, `--lightgbm-device cpu` can be
changed to `--lightgbm-device gpu`. CPU mode is the safest default.

## 5. Return results

After the command finishes, send back only the assigned `outputs_v2_*` directory.
Do not send back:

```text
GenImage_data
.venv
feature_cache*
sample_cache
experiment_logs
```

The main machine will merge result folders and rebuild the report tables.
