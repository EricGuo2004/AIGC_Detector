param(
  [string]$DatasetRoot = "C:\Users\99303\git\GenImage_data",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$FeatureCacheDir = "feature_cache",
  [string]$SampleCacheDir = "sample_cache",
  [double]$ImprovementThreshold = 0.005,
  [switch]$Force,
  [switch]$SkipTenPctSeeds
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$Profiles = @("baseline", "regularized", "large", "wide")
$FeatureSets = @("all", "no_dct")
$Tasks = @("binary_ai_vs_nature", "ai_subsource_attribution")

function Test-TaskMetrics {
  param([string]$Name)
  foreach ($task in $Tasks) {
    if (-not (Test-Path (Join-Path $ProjectRoot "$Name\$task\metrics_summary.json"))) {
      return $false
    }
    if (-not (Test-Path (Join-Path $ProjectRoot "$Name\$task\model_comparison.csv"))) {
      return $false
    }
  }
  return $true
}

function Run-TuningExperiment {
  param(
    [string]$Name,
    [double]$Fraction,
    [int]$Seed,
    [string]$Profile,
    [string]$FeatureSet
  )

  if ((-not $Force) -and (Test-TaskMetrics -Name $Name)) {
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
    "--lgbm-profile", $Profile,
    "--model-set", "lightgbm",
    "--num-workers", "16",
    "--feature-chunksize", "64",
    "--feature-cache-dir", $FeatureCacheDir,
    "--sample-cache-dir", $SampleCacheDir,
    "--feature-set", $FeatureSet,
    "--feature-profile", "baseline"
  )

  Write-Host "[run] $Name fraction=$Fraction seed=$Seed profile=$Profile feature_set=$FeatureSet"
  & $Python @args
}

function Get-TaskMacroF1 {
  param(
    [string]$Name,
    [string]$Task
  )
  $summaryPath = Join-Path $ProjectRoot "$Name\$Task\metrics_summary.json"
  $comparisonPath = Join-Path $ProjectRoot "$Name\$Task\model_comparison.csv"
  if ((-not (Test-Path $summaryPath)) -or (-not (Test-Path $comparisonPath))) {
    return $null
  }
  $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
  $rows = Import-Csv $comparisonPath
  $best = $rows | Where-Object { $_.model -eq $summary.best_model } | Select-Object -First 1
  if ($null -eq $best) {
    $best = $rows | Select-Object -First 1
  }
  return [double]$best.macro_f1
}

function Get-CombinedMacroF1 {
  param([string]$Name)
  $vals = @()
  foreach ($task in $Tasks) {
    $value = Get-TaskMacroF1 -Name $Name -Task $task
    if ($null -eq $value) {
      return $null
    }
    $vals += $value
  }
  return ($vals | Measure-Object -Average).Average
}

function Parse-TuneName {
  param([string]$Name)
  if ($Name -match "^outputs_4gen_5pct_tune_(baseline|regularized|large|wide)_(.+)$") {
    return @{
      Profile = $Matches[1]
      FeatureSet = $Matches[2]
    }
  }
  throw "Cannot parse tuning output name: $Name"
}

function Write-SelectionTable {
  param(
    [array]$Rows,
    [string]$Path
  )
  $Rows | Sort-Object combined_macro_f1 -Descending | Export-Csv $Path -NoTypeInformation -Encoding UTF8
}

$tuningOutputs = @()
foreach ($profile in $Profiles) {
  foreach ($featureSet in $FeatureSets) {
    $name = "outputs_4gen_5pct_tune_${profile}_${featureSet}"
    Run-TuningExperiment -Name $name -Fraction 0.05 -Seed 42 -Profile $profile -FeatureSet $featureSet
    $tuningOutputs += $name
  }
}

$selectionRows = @()
foreach ($name in $tuningOutputs) {
  $combined = Get-CombinedMacroF1 -Name $name
  if ($null -eq $combined) {
    continue
  }
  $config = Parse-TuneName -Name $name
  $row = [ordered]@{
    run = $name
    profile = $config.Profile
    feature_set = $config.FeatureSet
    binary_macro_f1 = Get-TaskMacroF1 -Name $name -Task "binary_ai_vs_nature"
    attribution_macro_f1 = Get-TaskMacroF1 -Name $name -Task "ai_subsource_attribution"
    combined_macro_f1 = $combined
  }
  $selectionRows += New-Object psobject -Property $row
}

if ($selectionRows.Count -eq 0) {
  throw "No complete tuning outputs were found."
}

New-Item -ItemType Directory -Force -Path "report\tables" | Out-Null
Write-SelectionTable -Rows $selectionRows -Path "report\tables\lgbm_tuning_selection.csv"

$best = $selectionRows | Sort-Object combined_macro_f1 -Descending | Select-Object -First 1
Write-Host "[best] profile=$($best.profile) feature_set=$($best.feature_set) combined_macro_f1=$($best.combined_macro_f1)"

Run-TuningExperiment `
  -Name "outputs_4gen_10pct_best" `
  -Fraction 0.10 `
  -Seed 42 `
  -Profile $best.profile `
  -FeatureSet $best.feature_set

$baselineName = "outputs_4gen_5pct"
$tenPctName = "outputs_4gen_10pct_best"
$allTasksImproved = $true
foreach ($task in $Tasks) {
  $baseline = Get-TaskMacroF1 -Name $baselineName -Task $task
  $tenPct = Get-TaskMacroF1 -Name $tenPctName -Task $task
  $delta = $tenPct - $baseline
  Write-Host "[compare] $task 5pct=$baseline 10pct=$tenPct delta=$delta"
  if ($delta -lt $ImprovementThreshold) {
    $allTasksImproved = $false
  }
}

if ((-not $SkipTenPctSeeds) -and $allTasksImproved) {
  Run-TuningExperiment `
    -Name "outputs_4gen_10pct_best_seed123" `
    -Fraction 0.10 `
    -Seed 123 `
    -Profile $best.profile `
    -FeatureSet $best.feature_set
  Run-TuningExperiment `
    -Name "outputs_4gen_10pct_best_seed2026" `
    -Fraction 0.10 `
    -Seed 2026 `
    -Profile $best.profile `
    -FeatureSet $best.feature_set
} else {
  Write-Host "[skip] 10pct seed expansion skipped. allTasksImproved=$allTasksImproved threshold=$ImprovementThreshold"
}

$formalOutputs = @(
  "outputs_smoke_4gen_min",
  "outputs_4gen_1pct",
  "outputs_4gen_1pct_robust",
  "outputs_4gen_5pct",
  "outputs_4gen_5pct_seed123",
  "outputs_4gen_5pct_seed2026",
  "outputs_4gen_10pct_best",
  "outputs_4gen_10pct_best_seed123",
  "outputs_4gen_10pct_best_seed2026"
) + $tuningOutputs

$featureAblations = @(
  "fft_radial",
  "fft_angular",
  "band_slope",
  "high_freq_stats",
  "patch_high_freq",
  "high_freq_all",
  "dct_radial",
  "no_dct"
)
foreach ($featureSet in $featureAblations) {
  $formalOutputs += "outputs_4gen_5pct_ablation_$featureSet"
}

$existingOutputs = @($formalOutputs | Where-Object { Test-Path $_ })
$primaryOutput = if (Test-Path "outputs_4gen_10pct_best") { "outputs_4gen_10pct_best" } else { "outputs_4gen_5pct" }

$assetArgs = @(
  "scripts\build_report_assets.py",
  "--dataset-root", $DatasetRoot,
  "--report-dir", "report",
  "--primary-output", $primaryOutput,
  "--outputs"
) + $existingOutputs
& $Python @assetArgs
