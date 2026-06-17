param(
  [string]$DatasetRoot = $(if ($env:GENIMAGE_DATA_ROOT) { $env:GENIMAGE_DATA_ROOT } else { "data\GenImage_data" }),
  [string]$Python = ".\.venv\Scripts\python.exe",
  [ValidateSet("cpu", "gpu")]
  [string]$LightgbmDevice = "gpu",
  [int]$NumWorkers = 16,
  [int]$FeatureChunksize = 64,
  [double]$LogoFraction = 0.20,
  [double]$CandidateFraction = 0.20,
  [switch]$SkipLogo,
  [switch]$SkipStable,
  [switch]$SkipAugmentation,
  [switch]$SkipRobustness,
  [switch]$SkipConfidence
)

$ErrorActionPreference = "Stop"

function Get-FractionToken {
  param([double]$Fraction)
  if ($Fraction -ge 0.999) {
    return "full"
  }
  $pct = [int][Math]::Round($Fraction * 100)
  return "${pct}pct"
}

function Invoke-Python {
  param([string[]]$PythonArgs)
  Write-Host "[run] $Python $($PythonArgs -join ' ')"
  & $Python @PythonArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE"
  }
}

$logoToken = Get-FractionToken -Fraction $LogoFraction
$candidateToken = Get-FractionToken -Fraction $CandidateFraction
$logoOut = "outputs_4gen_logo_${logoToken}_fusion"
$stableOut = "outputs_4gen_${candidateToken}_stable_freq"
$mildOut = "outputs_4gen_${candidateToken}_fusion_mild_aug"
$robustOut = "outputs_4gen_${candidateToken}_fusion_robust_aug"

if (-not $SkipLogo) {
  Invoke-Python @(
    "scripts\run_logo_generalization.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $logoOut,
    "--sample-fraction", "$LogoFraction",
    "--sample-seed", "42",
    "--feature-profile", "fusion_freq",
    "--feature-set", "all",
    "--feature-cache-dir", "feature_cache_logo_fusion",
    "--model-set", "lightgbm",
    "--lightgbm-device", $LightgbmDevice,
    "--lgbm-profile", "wide",
    "--calibrate-threshold",
    "--num-workers", "$NumWorkers",
    "--feature-chunksize", "$FeatureChunksize",
    "--resume-completed"
  )
}

if (-not $SkipStable) {
  Invoke-Python @(
    "test.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $stableOut,
    "--sample-fraction", "$CandidateFraction",
    "--sample-seed", "42",
    "--feature-profile", "stable_freq",
    "--feature-set", "all",
    "--feature-cache-dir", "feature_cache_stable",
    "--sample-cache-dir", "sample_cache",
    "--model-set", "lightgbm",
    "--lightgbm-device", $LightgbmDevice,
    "--lgbm-profile", "wide",
    "--calibrate-threshold",
    "--num-workers", "$NumWorkers",
    "--feature-chunksize", "$FeatureChunksize",
    "--skip-robustness",
    "--resume-completed-tasks"
  )
}

if (-not $SkipAugmentation) {
  foreach ($item in @(
      @{ Name = $mildOut; Aug = "mild_freq" },
      @{ Name = $robustOut; Aug = "robust_freq" }
    )) {
    Invoke-Python @(
      "test.py",
      "--dataset-root", $DatasetRoot,
      "--out-dir", $item.Name,
      "--sample-fraction", "$CandidateFraction",
      "--sample-seed", "42",
      "--feature-profile", "fusion_freq",
      "--feature-set", "all",
      "--feature-cache-dir", "feature_cache_fusion",
      "--sample-cache-dir", "sample_cache",
      "--model-set", "lightgbm",
      "--model-architecture", "flat",
      "--lightgbm-device", $LightgbmDevice,
      "--lgbm-profile", "wide",
      "--train-augmentation", $item.Aug,
      "--calibrate-threshold",
      "--num-workers", "$NumWorkers",
      "--feature-chunksize", "$FeatureChunksize",
      "--skip-robustness",
      "--resume-completed-tasks"
    )
  }
}

if (-not $SkipRobustness) {
  foreach ($item in @(
      @{ Model = $stableOut; Out = "${stableOut}_robust_20pct"; Cache = "robustness_cache_stable" },
      @{ Model = $mildOut; Out = "${mildOut}_robust_20pct"; Cache = "robustness_cache_fusion" },
      @{ Model = $robustOut; Out = "${robustOut}_robust_20pct"; Cache = "robustness_cache_fusion" }
    )) {
    if (Test-Path (Join-Path $item.Model "binary_ai_vs_nature\best_model.joblib")) {
      Invoke-Python @(
        "scripts\evaluate_best_robustness.py",
        "--dataset-root", $DatasetRoot,
        "--model-output", $item.Model,
        "--out-dir", $item.Out,
        "--sample-fraction", "0.20",
        "--sample-seed", "42",
        "--tasks", "both",
        "--num-workers", "$NumWorkers",
        "--feature-chunksize", "$FeatureChunksize",
        "--robust-cache-dir", $item.Cache
      )
    } else {
      Write-Host "[skip] Missing model bundle under $($item.Model)"
    }
  }
}

if (-not $SkipConfidence) {
  foreach ($item in @(
      @{ Model = "outputs_v2_full_best"; Cache = "feature_cache_fusion"; Fraction = "1.0" },
      @{ Model = $stableOut; Cache = "feature_cache_stable"; Fraction = "$CandidateFraction" },
      @{ Model = $mildOut; Cache = "feature_cache_fusion"; Fraction = "$CandidateFraction" },
      @{ Model = $robustOut; Cache = "feature_cache_fusion"; Fraction = "$CandidateFraction" }
    )) {
    if (Test-Path (Join-Path $item.Model "binary_ai_vs_nature\best_model.joblib")) {
      Invoke-Python @(
        "scripts\analyze_confidence_rejection.py",
        "--dataset-root", $DatasetRoot,
        "--output-dir", $item.Model,
        "--tasks", "both",
        "--sample-fraction", $item.Fraction,
        "--sample-seed", "42",
        "--feature-cache-dir", $item.Cache,
        "--num-workers", "$NumWorkers",
        "--feature-chunksize", "$FeatureChunksize"
      )
    } else {
      Write-Host "[skip] Missing model bundle under $($item.Model)"
    }
  }
}

$compareOutputs = @("outputs_v2_full_best_robust_20pct", "outputs_4gen_full_best_robust_20pct")
foreach ($candidate in @("${stableOut}_robust_20pct", "${mildOut}_robust_20pct", "${robustOut}_robust_20pct")) {
  if (Test-Path $candidate) {
    $compareOutputs += $candidate
  }
}

$assetArgs = @(
  "scripts\build_report_assets.py",
  "--dataset-root", $DatasetRoot,
  "--report-dir", "report",
  "--primary-output", "outputs_v2_full_best",
  "--robustness-output", "outputs_v2_full_best_robust_20pct",
  "--robustness-compare-outputs"
) + $compareOutputs
Invoke-Python $assetArgs

Write-Host "[complete] 4-generator extension suite finished."
