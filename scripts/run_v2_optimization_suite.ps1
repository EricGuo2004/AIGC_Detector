param(
  [string]$DatasetRoot = "C:\Users\99303\git\GenImage_data",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$LightgbmDevice = "gpu",
  [int]$NumWorkers = 16,
  [int]$FeatureChunksize = 32,
  [int]$SampleSeed = 42,
  [switch]$Force,
  [switch]$SkipFull
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$CurrentFullCombined = 0.9041572078758394

function Get-FeatureCacheDir {
  param([string]$FeatureProfile)
  switch ($FeatureProfile) {
    "color_freq" { return "feature_cache_color" }
    "multiscale_freq" { return "feature_cache_multiscale" }
    "block_dct" { return "feature_cache_block_dct" }
    "residual_freq" { return "feature_cache_residual" }
    "fusion_freq" { return "feature_cache_fusion" }
    "enhanced" { return "feature_cache_enhanced" }
    default { return "feature_cache" }
  }
}

function Test-ExperimentComplete {
  param([string]$OutDir)
  return (
    (Test-Path -LiteralPath (Join-Path $OutDir "binary_ai_vs_nature\metrics_summary.json")) -and
    (Test-Path -LiteralPath (Join-Path $OutDir "ai_subsource_attribution\metrics_summary.json"))
  )
}

function Run-Experiment {
  param(
    [string]$Name,
    [double]$Fraction,
    [string]$FeatureProfile,
    [string]$ModelArchitecture = "flat",
    [string]$TrainAugmentation = "none",
    [switch]$CalibrateThreshold
  )

  if ((Test-ExperimentComplete $Name) -and (-not $Force)) {
    Write-Host "[skip] $Name already complete"
    return
  }

  $cacheDir = Get-FeatureCacheDir $FeatureProfile
  $args = @(
    "test.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $Name,
    "--sample-fraction", "$Fraction",
    "--sample-seed", "$SampleSeed",
    "--skip-robustness",
    "--lightgbm-device", $LightgbmDevice,
    "--num-workers", "$NumWorkers",
    "--feature-chunksize", "$FeatureChunksize",
    "--feature-cache-dir", $cacheDir,
    "--sample-cache-dir", "sample_cache",
    "--feature-set", "all",
    "--feature-profile", $FeatureProfile,
    "--lgbm-profile", "wide",
    "--model-set", "lightgbm",
    "--model-architecture", $ModelArchitecture,
    "--train-augmentation", $TrainAugmentation
  )
  if ($CalibrateThreshold) {
    $args += "--calibrate-threshold"
  }

  Write-Host "[run] $Name fraction=$Fraction feature=$FeatureProfile model=$ModelArchitecture aug=$TrainAugmentation calibrate=$CalibrateThreshold"
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: $Name"
  }
}

function Build-V2Assets {
  & $Python "scripts\build_optimization_v2_assets.py" --report-dir "report"
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to build v2 optimization assets"
  }
}

function Get-BestV2Row {
  $path = "report\tables\optimization_v2_summary.csv"
  if (-not (Test-Path -LiteralPath $path)) {
    return $null
  }
  $rows = Import-Csv -LiteralPath $path | Where-Object { $_.combined_macro_f1 -ne "" }
  if (-not $rows) {
    return $null
  }
  return $rows | Sort-Object {[double]$_.combined_macro_f1} -Descending | Select-Object -First 1
}

function Get-BestV2RowByPrefix {
  param([string]$Prefix)
  $path = "report\tables\optimization_v2_summary.csv"
  if (-not (Test-Path -LiteralPath $path)) {
    return $null
  }
  $rows = Import-Csv -LiteralPath $path | Where-Object { $_.run -like "$Prefix*" -and $_.combined_macro_f1 -ne "" }
  if (-not $rows) {
    return $null
  }
  return $rows | Sort-Object {[double]$_.combined_macro_f1} -Descending | Select-Object -First 1
}

Write-Host "[phase1] baseline error bottleneck summary"
& $Python "scripts\analyze_v2_error_bottlenecks.py" --output-dir "outputs_4gen_full_best" --report-dir "report"
if ($LASTEXITCODE -ne 0) {
  throw "Failed to summarize baseline errors"
}

$profiles = @("color_freq", "multiscale_freq", "block_dct", "residual_freq", "fusion_freq")

Write-Host "[phase1] 0.1% feature smoke tests"
foreach ($profile in $profiles) {
  Run-Experiment -Name "outputs_v2_smoke_${profile}" -Fraction 0.001 -FeatureProfile $profile -ModelArchitecture "flat" -CalibrateThreshold
}
Build-V2Assets

Write-Host "[phase2] 5% feature profile screening"
foreach ($profile in $profiles) {
  Run-Experiment -Name "outputs_v2_5pct_${profile}_flat" -Fraction 0.05 -FeatureProfile $profile -ModelArchitecture "flat" -CalibrateThreshold
  Build-V2Assets
}

$bestFeatureRow = Get-BestV2RowByPrefix "outputs_v2_5pct_"
if ($null -eq $bestFeatureRow) {
  throw "No 5% v2 feature result was found."
}
$bestFeature = $bestFeatureRow.feature_profile
Write-Host "[best feature] $bestFeature combined=$($bestFeatureRow.combined_macro_f1)"

Write-Host "[phase3] model architecture screening on best feature"
$architectures = @("hierarchical_attribution", "pairwise_ovo_attribution", "binary_expert_ensemble")
foreach ($arch in $architectures) {
  Run-Experiment -Name "outputs_v2_5pct_${bestFeature}_${arch}" -Fraction 0.05 -FeatureProfile $bestFeature -ModelArchitecture $arch -CalibrateThreshold
  Build-V2Assets
}

$bestRow = Get-BestV2RowByPrefix "outputs_v2_5pct_"
if ($null -eq $bestRow) {
  throw "No best v2 row found after model architecture screening."
}
$bestFeature = $bestRow.feature_profile
$bestArch = $bestRow.model_architecture
Write-Host "[best 5pct config] feature=$bestFeature arch=$bestArch combined=$($bestRow.combined_macro_f1)"

Write-Host "[phase4] scale-up best config"
Run-Experiment -Name "outputs_v2_20pct_best" -Fraction 0.20 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -CalibrateThreshold
Build-V2Assets

Run-Experiment -Name "outputs_v2_50pct_best" -Fraction 0.50 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -CalibrateThreshold
Build-V2Assets

$bestScaleRow = Get-BestV2RowByPrefix "outputs_v2_50pct_best"
if (($null -ne $bestScaleRow) -and ([double]$bestScaleRow.combined_macro_f1 -le $CurrentFullCombined)) {
  Write-Host "[skip] full scale skipped because 50pct combined=$($bestScaleRow.combined_macro_f1) <= current full baseline=$CurrentFullCombined"
} elseif (-not $SkipFull) {
  Run-Experiment -Name "outputs_v2_full_best" -Fraction 1.0 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -CalibrateThreshold
  Build-V2Assets
} else {
  Write-Host "[skip] full scale skipped by -SkipFull"
}

Write-Host "[phase5] mild training augmentation probe"
Run-Experiment -Name "outputs_v2_20pct_best_mild_aug" -Fraction 0.20 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -TrainAugmentation "mild_freq" -CalibrateThreshold
Build-V2Assets

$augRow = Get-BestV2RowByPrefix "outputs_v2_20pct_best_mild_aug"
$plain20 = Get-BestV2RowByPrefix "outputs_v2_20pct_best"
if (($null -ne $augRow) -and ($null -ne $plain20) -and ([double]$augRow.combined_macro_f1 -ge ([double]$plain20.combined_macro_f1 + 0.003))) {
  Run-Experiment -Name "outputs_v2_50pct_best_mild_aug" -Fraction 0.50 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -TrainAugmentation "mild_freq" -CalibrateThreshold
  Build-V2Assets
  if (-not $SkipFull) {
    Run-Experiment -Name "outputs_v2_full_best_mild_aug" -Fraction 1.0 -FeatureProfile $bestFeature -ModelArchitecture $bestArch -TrainAugmentation "mild_freq" -CalibrateThreshold
    Build-V2Assets
  }
} else {
  Write-Host "[skip] mild augmentation scale-up skipped; gain threshold not met"
}

Write-Host "[done] v2 optimization suite complete"
