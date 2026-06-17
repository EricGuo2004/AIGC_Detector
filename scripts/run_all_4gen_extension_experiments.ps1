param(
  [string]$DatasetRoot = $(if ($env:GENIMAGE_DATA_ROOT) { $env:GENIMAGE_DATA_ROOT } else { "data\GenImage_data" }),
  [string]$Python = ".\.venv\Scripts\python.exe",
  [ValidateSet("cpu", "gpu")]
  [string]$LightgbmDevice = "gpu",
  [int]$NumWorkers = 16,
  [int]$FeatureChunksize = 64,
  [double]$LogoFraction = 0.20,
  [double]$CandidateFraction = 0.20,
  [double]$RobustnessFraction = 0.20,
  [int]$SampleSeed = 42,
  [switch]$Force,
  [switch]$SkipLogo,
  [switch]$SkipStable,
  [switch]$SkipAugmentation,
  [switch]$SkipRobustness,
  [switch]$SkipConfidence,
  [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Get-FractionToken {
  param([double]$Fraction)
  if ($Fraction -ge 0.999) {
    return "full"
  }
  $pct = [int][Math]::Round($Fraction * 100)
  return "${pct}pct"
}

function Test-File {
  param([string]$Path)
  return (Test-Path -LiteralPath $Path -PathType Leaf)
}

function Test-TaskComplete {
  param([string]$OutDir, [string]$Task)
  $taskDir = Join-Path $OutDir $Task
  return (
    (Test-File (Join-Path $taskDir "metrics_summary.json")) -and
    (Test-File (Join-Path $taskDir "model_comparison.csv")) -and
    (Test-File (Join-Path $taskDir "best_model.joblib"))
  )
}

function Test-ExperimentComplete {
  param([string]$OutDir)
  return (
    (Test-TaskComplete $OutDir "binary_ai_vs_nature") -and
    (Test-TaskComplete $OutDir "ai_subsource_attribution")
  )
}

function Test-RobustnessComplete {
  param([string]$OutDir)
  return (
    (Test-File (Join-Path $OutDir "binary_ai_vs_nature\robustness_results.csv")) -and
    (Test-File (Join-Path $OutDir "ai_subsource_attribution\robustness_results.csv"))
  )
}

function Test-ConfidenceComplete {
  param([string]$OutDir)
  return (
    (Test-File (Join-Path $OutDir "binary_ai_vs_nature\confidence_coverage_accuracy.csv")) -and
    (Test-File (Join-Path $OutDir "ai_subsource_attribution\confidence_coverage_accuracy.csv"))
  )
}

function Invoke-PythonStep {
  param(
    [string]$Name,
    [string[]]$PythonArgs,
    [scriptblock]$IsComplete = $null
  )

  if (($null -ne $IsComplete) -and (& $IsComplete) -and (-not $Force)) {
    Write-Host "[skip] $Name already complete"
    return
  }

  $started = Get-Date
  Write-Host ""
  Write-Host "===== START $Name ====="
  Write-Host "[time] $($started.ToString('yyyy-MM-dd HH:mm:ss'))"
  Write-Host "[run] $Python $($PythonArgs -join ' ')"
  & $Python @PythonArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name (exit code $LASTEXITCODE)"
  }
  $finished = Get-Date
  $minutes = [Math]::Round(($finished - $started).TotalMinutes, 2)
  Write-Host "[done] $Name in ${minutes}min"
  Write-Host "===== END $Name ====="
}

function Invoke-RobustnessIfModelExists {
  param(
    [string]$ModelOutput,
    [string]$OutDir,
    [string]$CacheDir
  )
  if (-not (Test-TaskComplete $ModelOutput "binary_ai_vs_nature")) {
    Write-Host "[skip] robustness for $ModelOutput because model output is incomplete"
    return
  }
  Invoke-PythonStep `
    -Name "robustness $ModelOutput" `
    -IsComplete { Test-RobustnessComplete $OutDir } `
    -PythonArgs @(
      "scripts\evaluate_best_robustness.py",
      "--dataset-root", $DatasetRoot,
      "--model-output", $ModelOutput,
      "--out-dir", $OutDir,
      "--sample-fraction", "$RobustnessFraction",
      "--sample-seed", "$SampleSeed",
      "--tasks", "both",
      "--num-workers", "$NumWorkers",
      "--feature-chunksize", "$FeatureChunksize",
      "--robust-cache-dir", $CacheDir
    )
}

function Invoke-ConfidenceIfModelExists {
  param(
    [string]$ModelOutput,
    [string]$CacheDir,
    [double]$Fraction
  )
  if (-not (Test-TaskComplete $ModelOutput "binary_ai_vs_nature")) {
    Write-Host "[skip] confidence for $ModelOutput because model output is incomplete"
    return
  }
  Invoke-PythonStep `
    -Name "confidence $ModelOutput" `
    -IsComplete { Test-ConfidenceComplete $ModelOutput } `
    -PythonArgs @(
      "scripts\analyze_confidence_rejection.py",
      "--dataset-root", $DatasetRoot,
      "--output-dir", $ModelOutput,
      "--tasks", "both",
      "--sample-fraction", "$Fraction",
      "--sample-seed", "$SampleSeed",
      "--feature-cache-dir", $CacheDir,
      "--num-workers", "$NumWorkers",
      "--feature-chunksize", "$FeatureChunksize"
    )
}

$logoToken = Get-FractionToken -Fraction $LogoFraction
$candidateToken = Get-FractionToken -Fraction $CandidateFraction
$robustToken = Get-FractionToken -Fraction $RobustnessFraction

$logoOut = "outputs_4gen_logo_${logoToken}_fusion"
$stableOut = "outputs_4gen_${candidateToken}_stable_freq"
$mildOut = "outputs_4gen_${candidateToken}_fusion_mild_aug"
$robustOut = "outputs_4gen_${candidateToken}_fusion_robust_aug"
$stableRobustOut = "${stableOut}_robust_${robustToken}"
$mildRobustOut = "${mildOut}_robust_${robustToken}"
$robustRobustOut = "${robustOut}_robust_${robustToken}"

Write-Host "[config] DatasetRoot=$DatasetRoot"
Write-Host "[config] Device=$LightgbmDevice workers=$NumWorkers chunksize=$FeatureChunksize"
Write-Host "[config] LogoFraction=$LogoFraction CandidateFraction=$CandidateFraction RobustnessFraction=$RobustnessFraction Seed=$SampleSeed"

Invoke-PythonStep `
  -Name "preflight py_compile" `
  -PythonArgs @(
    "-m", "py_compile",
    "train.py",
    "test.py",
    "src\features.py",
    "src\training.py",
    "src\robustness.py",
    "scripts\run_logo_generalization.py",
    "scripts\analyze_confidence_rejection.py",
    "scripts\evaluate_best_robustness.py",
    "scripts\build_report_assets.py"
  )

if (-not $SkipLogo) {
  Invoke-PythonStep `
    -Name "LOGO generalization $logoOut" `
    -IsComplete {
      (Test-File (Join-Path $logoOut "logo_generalization_summary.csv")) -and
      (Test-TaskComplete (Join-Path $logoOut "leave_ADM_out") "binary_ai_vs_nature") -and
      (Test-TaskComplete (Join-Path $logoOut "leave_BigGAN_out") "binary_ai_vs_nature") -and
      (Test-TaskComplete (Join-Path $logoOut "leave_VQDM_out") "binary_ai_vs_nature") -and
      (Test-TaskComplete (Join-Path $logoOut "leave_glide_out") "binary_ai_vs_nature")
    } `
    -PythonArgs @(
      "scripts\run_logo_generalization.py",
      "--dataset-root", $DatasetRoot,
      "--out-dir", $logoOut,
      "--sample-fraction", "$LogoFraction",
      "--sample-seed", "$SampleSeed",
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
  Invoke-PythonStep `
    -Name "stable_freq clean $stableOut" `
    -IsComplete { Test-ExperimentComplete $stableOut } `
    -PythonArgs @(
      "test.py",
      "--dataset-root", $DatasetRoot,
      "--out-dir", $stableOut,
      "--sample-fraction", "$CandidateFraction",
      "--sample-seed", "$SampleSeed",
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
    Invoke-PythonStep `
      -Name "$($item.Aug) clean $($item.Name)" `
      -IsComplete { Test-ExperimentComplete $item.Name } `
      -PythonArgs @(
        "test.py",
        "--dataset-root", $DatasetRoot,
        "--out-dir", $item.Name,
        "--sample-fraction", "$CandidateFraction",
        "--sample-seed", "$SampleSeed",
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
  Invoke-RobustnessIfModelExists -ModelOutput $stableOut -OutDir $stableRobustOut -CacheDir "robustness_cache_stable"
  Invoke-RobustnessIfModelExists -ModelOutput $mildOut -OutDir $mildRobustOut -CacheDir "robustness_cache_fusion"
  Invoke-RobustnessIfModelExists -ModelOutput $robustOut -OutDir $robustRobustOut -CacheDir "robustness_cache_fusion"
}

if (-not $SkipConfidence) {
  Invoke-ConfidenceIfModelExists -ModelOutput "outputs_v2_full_best" -CacheDir "feature_cache_fusion" -Fraction 1.0
  Invoke-ConfidenceIfModelExists -ModelOutput $stableOut -CacheDir "feature_cache_stable" -Fraction $CandidateFraction
  Invoke-ConfidenceIfModelExists -ModelOutput $mildOut -CacheDir "feature_cache_fusion" -Fraction $CandidateFraction
  Invoke-ConfidenceIfModelExists -ModelOutput $robustOut -CacheDir "feature_cache_fusion" -Fraction $CandidateFraction
}

$compareOutputs = @("outputs_v2_full_best_robust_20pct", "outputs_4gen_full_best_robust_20pct")
foreach ($candidate in @($stableRobustOut, $mildRobustOut, $robustRobustOut)) {
  if (Test-Path -LiteralPath $candidate) {
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

Invoke-PythonStep -Name "build report assets" -PythonArgs $assetArgs

if (-not $SkipValidation) {
  Invoke-PythonStep `
    -Name "validate submission assets" `
    -PythonArgs @(
      "scripts\validate_submission.py",
      "--dataset-root", $DatasetRoot,
      "--write-report"
    )
}

Write-Host ""
Write-Host "[complete] all 4-generator extension experiments finished."
