param(
  [string]$DatasetRoot = $(if ($env:GENIMAGE_DATA_ROOT) { $env:GENIMAGE_DATA_ROOT } else { "data\GenImage_data" }),
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$FeatureCacheDir = "feature_cache",
  [string]$SampleCacheDir = "sample_cache",
  [switch]$Force,
  [switch]$SkipSeeds,
  [switch]$SkipAblation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Get-CacheTag {
  param(
    [double]$Fraction,
    [int]$Seed
  )
  $fractionToken = ("$Fraction").Replace(".", "p")
  return "frac${fractionToken}_seed${Seed}"
}

function Test-FeatureCache {
  param(
    [double]$Fraction,
    [int]$Seed
  )
  $tag = Get-CacheTag -Fraction $Fraction -Seed $Seed
  $required = @(
    "binary_ai_vs_nature_train_$tag.npz",
    "binary_ai_vs_nature_val_$tag.npz",
    "ai_subsource_attribution_train_$tag.npz",
    "ai_subsource_attribution_val_$tag.npz"
  )
  foreach ($name in $required) {
    if (-not (Test-Path (Join-Path $FeatureCacheDir $name))) {
      return $false
    }
  }
  return $true
}

function Run-TestExperiment {
  param(
    [string]$Name,
    [double]$Fraction,
    [int]$Seed,
    [string]$FeatureSet = "all"
  )

  $outDir = Join-Path $ProjectRoot $Name
  $binaryMetrics = Join-Path $outDir "binary_ai_vs_nature\metrics_summary.json"
  $attrMetrics = Join-Path $outDir "ai_subsource_attribution\metrics_summary.json"
  $hasMetrics = (Test-Path $binaryMetrics) -and (Test-Path $attrMetrics)
  $hasCache = Test-FeatureCache -Fraction $Fraction -Seed $Seed
  if ((-not $Force) -and $hasMetrics -and (($FeatureSet -ne "all") -or $hasCache)) {
    Write-Host "[skip] $Name already has task metrics."
    return
  }

  $args = @(
    "test.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $Name,
    "--sample-fraction", "$Fraction",
    "--sample-seed", "$Seed",
    "--skip-robustness",
    "--lightgbm-device", "gpu",
    "--num-workers", "16",
    "--feature-chunksize", "64",
    "--feature-cache-dir", $FeatureCacheDir,
    "--sample-cache-dir", $SampleCacheDir,
    "--feature-set", $FeatureSet,
    "--feature-profile", "baseline"
  )

  Write-Host "[run] $Name fraction=$Fraction seed=$Seed feature_set=$FeatureSet"
  & $Python @args
}

# Re-run the primary 5% seed-42 experiment once with caching enabled.
Run-TestExperiment -Name "outputs_4gen_5pct" -Fraction 0.05 -Seed 42 -FeatureSet "all"

if (-not $SkipSeeds) {
  Run-TestExperiment -Name "outputs_4gen_5pct_seed123" -Fraction 0.05 -Seed 123 -FeatureSet "all"
  Run-TestExperiment -Name "outputs_4gen_5pct_seed2026" -Fraction 0.05 -Seed 2026 -FeatureSet "all"
}

$featureSets = @(
  "fft_radial",
  "fft_angular",
  "band_slope",
  "high_freq_stats",
  "patch_high_freq",
  "high_freq_all",
  "dct_radial",
  "no_dct"
)

if (-not $SkipAblation) {
  foreach ($featureSet in $featureSets) {
    Run-TestExperiment -Name "outputs_4gen_5pct_ablation_$featureSet" -Fraction 0.05 -Seed 42 -FeatureSet $featureSet
  }
}

$formalOutputs = @(
  "outputs_smoke_4gen_min",
  "outputs_4gen_1pct",
  "outputs_4gen_1pct_robust",
  "outputs_4gen_5pct",
  "outputs_4gen_5pct_seed123",
  "outputs_4gen_5pct_seed2026"
)
foreach ($featureSet in $featureSets) {
  $formalOutputs += "outputs_4gen_5pct_ablation_$featureSet"
}
$existingOutputs = @($formalOutputs | Where-Object { Test-Path $_ })

$assetArgs = @(
  "scripts\build_report_assets.py",
  "--dataset-root", $DatasetRoot,
  "--report-dir", "report",
  "--primary-output", "outputs_4gen_5pct",
  "--outputs"
) + $existingOutputs
& $Python @assetArgs
