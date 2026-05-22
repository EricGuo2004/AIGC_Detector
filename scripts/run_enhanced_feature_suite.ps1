param(
  [string]$DatasetRoot = "C:\Users\99303\git\GenImage_data",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$FeatureCacheDir = "feature_cache_enhanced",
  [string]$SampleCacheDir = "sample_cache",
  [double]$MinFreeGB = 100,
  [double]$TenPctImprovementThreshold = 0.005,
  [double]$FullEnhancedImprovementThreshold = 0.003,
  [switch]$Force,
  [switch]$SkipFull
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$Tasks = @("binary_ai_vs_nature", "ai_subsource_attribution")
$Profiles = @("wide", "regularized", "large")

function Get-FreeGB {
  return [math]::Round((Get-PSDrive C).Free / 1GB, 2)
}

function Assert-FreeSpace {
  $free = Get-FreeGB
  if ($free -lt $MinFreeGB) {
    throw "C: free space is ${free}GB, below MinFreeGB=${MinFreeGB}GB. Stop before starting the next enhanced stage."
  }
  Write-Host "[space] C: free=${free}GB"
}

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

function Invoke-EnhancedExperiment {
  param(
    [string]$Name,
    [double]$Fraction,
    [int]$Seed,
    [string]$Profile
  )

  if ((-not $Force) -and (Test-TaskMetrics -Name $Name)) {
    Write-Host "[skip] $Name already has task metrics."
    return
  }

  Assert-FreeSpace
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
    "--feature-set", "all",
    "--feature-profile", "enhanced"
  )

  Write-Host "[run] $Name fraction=$Fraction seed=$Seed profile=$Profile feature_profile=enhanced"
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
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

function Get-BestExistingBaselineCombined {
  $names = @(
    "outputs_4gen_10pct_best",
    "outputs_4gen_20pct_best",
    "outputs_4gen_50pct_best",
    "outputs_4gen_full_best"
  )
  $best = $null
  foreach ($name in $names) {
    $value = Get-CombinedMacroF1 -Name $name
    if ($null -eq $value) {
      continue
    }
    if ($null -eq $best -or $value -gt $best) {
      $best = $value
    }
  }
  return $best
}

function Update-ReportAssets {
  $primary = if (Test-Path "outputs_4gen_10pct_best") { "outputs_4gen_10pct_best" } else { "outputs_4gen_5pct" }
  & $Python "scripts\build_report_assets.py" `
    --dataset-root $DatasetRoot `
    --report-dir "report" `
    --primary-output $primary
  if ($LASTEXITCODE -ne 0) {
    throw "build_report_assets.py failed with exit code $LASTEXITCODE"
  }

  $selectionPath = "report\tables\final_result_selection.csv"
  if (Test-Path $selectionPath) {
    $selected = Import-Csv $selectionPath | Where-Object { $_.selected -eq "True" } | Select-Object -First 1
    if ($null -ne $selected -and $selected.run -ne $primary -and (Test-Path $selected.run)) {
      Write-Host "[assets] rerun with selected primary=$($selected.run)"
      & $Python "scripts\build_report_assets.py" `
        --dataset-root $DatasetRoot `
        --report-dir "report" `
        --primary-output $selected.run
      if ($LASTEXITCODE -ne 0) {
        throw "build_report_assets.py failed with selected primary exit code $LASTEXITCODE"
      }
    }
  }
}

Invoke-EnhancedExperiment -Name "outputs_smoke_enhanced_min" -Fraction 0.001 -Seed 42 -Profile "wide"

$baselineTenPct = Get-CombinedMacroF1 -Name "outputs_4gen_10pct_best"
if ($null -eq $baselineTenPct) {
  throw "Missing baseline run outputs_4gen_10pct_best. Run the tuning suite first."
}

$rows = @()
foreach ($profile in $Profiles) {
  $name = "outputs_4gen_10pct_enhanced_$profile"
  Invoke-EnhancedExperiment -Name $name -Fraction 0.10 -Seed 42 -Profile $profile
  $combined = Get-CombinedMacroF1 -Name $name
  if ($null -ne $combined) {
    $rows += New-Object psobject -Property ([ordered]@{
      run = $name
      profile = $profile
      binary_macro_f1 = Get-TaskMacroF1 -Name $name -Task "binary_ai_vs_nature"
      attribution_macro_f1 = Get-TaskMacroF1 -Name $name -Task "ai_subsource_attribution"
      combined_macro_f1 = $combined
      delta_vs_10pct_baseline = $combined - $baselineTenPct
    })
  }
}

if ($rows.Count -eq 0) {
  throw "No complete enhanced 10pct outputs were found."
}

New-Item -ItemType Directory -Force -Path "report\tables" | Out-Null
$rows | Sort-Object combined_macro_f1 -Descending | Export-Csv "report\tables\enhanced_feature_selection.csv" -NoTypeInformation -Encoding UTF8
$best = $rows | Sort-Object combined_macro_f1 -Descending | Select-Object -First 1
Write-Host "[best enhanced 10pct] profile=$($best.profile) combined=$($best.combined_macro_f1) delta=$($best.delta_vs_10pct_baseline)"

if ($best.delta_vs_10pct_baseline -ge $TenPctImprovementThreshold) {
  Invoke-EnhancedExperiment -Name "outputs_4gen_20pct_enhanced_best" -Fraction 0.20 -Seed 42 -Profile $best.profile
  Invoke-EnhancedExperiment -Name "outputs_4gen_50pct_enhanced_best" -Fraction 0.50 -Seed 42 -Profile $best.profile

  $bestBaselineScale = Get-BestExistingBaselineCombined
  $fiftyEnhanced = Get-CombinedMacroF1 -Name "outputs_4gen_50pct_enhanced_best"
  if ($null -ne $bestBaselineScale -and $null -ne $fiftyEnhanced) {
    $delta = $fiftyEnhanced - $bestBaselineScale
    Write-Host "[compare] 50pct enhanced=$fiftyEnhanced best_baseline_scale=$bestBaselineScale delta=$delta"
    if ((-not $SkipFull) -and $delta -ge $FullEnhancedImprovementThreshold) {
      Invoke-EnhancedExperiment -Name "outputs_4gen_full_enhanced_best" -Fraction 1.00 -Seed 42 -Profile $best.profile
    } else {
      Write-Host "[skip] full enhanced skipped. skipFull=$SkipFull delta=$delta threshold=$FullEnhancedImprovementThreshold"
    }
  }
} else {
  Write-Host "[skip] enhanced scale-up skipped because 10pct gain is below $TenPctImprovementThreshold."
}

Update-ReportAssets
Write-Host "[complete] enhanced feature suite finished."
