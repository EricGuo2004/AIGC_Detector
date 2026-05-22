param(
  [string]$DatasetRoot = "C:\Users\99303\git\GenImage_data",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$FeatureCacheDir = "feature_cache",
  [string]$SampleCacheDir = "sample_cache",
  [double]$MinFreeGB = 100,
  [double]$MaxStageHours = 24,
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

function Get-FreeGB {
  return [math]::Round((Get-PSDrive C).Free / 1GB, 2)
}

function Assert-FreeSpace {
  $free = Get-FreeGB
  if ($free -lt $MinFreeGB) {
    throw "C: free space is ${free}GB, below MinFreeGB=${MinFreeGB}GB. Stop before starting the next scale-up stage."
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

function Invoke-ScaleExperiment {
  param(
    [string]$Name,
    [double]$Fraction
  )

  if ((-not $Force) -and (Test-TaskMetrics -Name $Name)) {
    Write-Host "[skip] $Name already has task metrics."
    return $true
  }

  Assert-FreeSpace
  $args = @(
    "test.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $Name,
    "--sample-fraction", "$Fraction",
    "--sample-seed", "42",
    "--skip-robustness",
    "--lightgbm-device", "gpu",
    "--lgbm-profile", "wide",
    "--model-set", "lightgbm",
    "--num-workers", "16",
    "--feature-chunksize", "64",
    "--feature-cache-dir", $FeatureCacheDir,
    "--sample-cache-dir", $SampleCacheDir,
    "--feature-set", "all",
    "--feature-profile", "baseline"
  )

  Write-Host "[run] $Name fraction=$Fraction profile=baseline/wide"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
  $sw.Stop()
  $hours = $sw.Elapsed.TotalHours
  Write-Host "[done] $Name elapsed_hours=$([math]::Round($hours, 3))"
  if ($hours -gt $MaxStageHours) {
    Write-Host "[stop] $Name exceeded MaxStageHours=$MaxStageHours. Later stages will be skipped."
    return $false
  }
  return $true
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

$stages = @(
  @{ Name = "outputs_4gen_20pct_best"; Fraction = 0.20 },
  @{ Name = "outputs_4gen_50pct_best"; Fraction = 0.50 }
)
if (-not $SkipFull) {
  $stages += @{ Name = "outputs_4gen_full_best"; Fraction = 1.00 }
}

foreach ($stage in $stages) {
  $continue = Invoke-ScaleExperiment -Name $stage.Name -Fraction $stage.Fraction
  Update-ReportAssets
  if (-not $continue) {
    break
  }
}

Write-Host "[complete] baseline scale-up suite finished."
