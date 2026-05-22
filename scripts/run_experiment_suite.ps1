param(
  [string]$DatasetRoot = "C:\Users\99303\git\GenImage_data",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [switch]$Force,
  [switch]$IncludeRobust5Pct
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Run-Experiment {
  param(
    [string]$Name,
    [double]$Fraction,
    [int]$Seed = 42,
    [bool]$Robust = $false
  )

  $outDir = Join-Path $ProjectRoot $Name
  $binaryMetrics = Join-Path $outDir "binary_ai_vs_nature\metrics_summary.json"
  $attrMetrics = Join-Path $outDir "ai_subsource_attribution\metrics_summary.json"
  if ((-not $Force) -and (Test-Path $binaryMetrics) -and (Test-Path $attrMetrics)) {
    Write-Host "[skip] $Name already has task metrics."
    return
  }

  $args = @(
    "test.py",
    "--dataset-root", $DatasetRoot,
    "--out-dir", $Name,
    "--sample-fraction", "$Fraction",
    "--sample-seed", "$Seed",
    "--num-workers", "12",
    "--feature-chunksize", "32",
    "--lightgbm-device", "gpu"
  )
  if (-not $Robust) {
    $args += "--skip-robustness"
  }

  Write-Host "[run] $Name fraction=$Fraction seed=$Seed robust=$Robust"
  & $Python @args
}

Run-Experiment -Name "outputs_4gen_1pct" -Fraction 0.01 -Seed 42 -Robust $false
Run-Experiment -Name "outputs_4gen_5pct" -Fraction 0.05 -Seed 42 -Robust $false
Run-Experiment -Name "outputs_4gen_10pct" -Fraction 0.10 -Seed 42 -Robust $false

Run-Experiment -Name "outputs_4gen_5pct_seed42" -Fraction 0.05 -Seed 42 -Robust $false
Run-Experiment -Name "outputs_4gen_5pct_seed123" -Fraction 0.05 -Seed 123 -Robust $false
Run-Experiment -Name "outputs_4gen_5pct_seed2026" -Fraction 0.05 -Seed 2026 -Robust $false

Run-Experiment -Name "outputs_4gen_1pct_robust" -Fraction 0.01 -Seed 42 -Robust $true
if ($IncludeRobust5Pct) {
  Run-Experiment -Name "outputs_4gen_5pct_robust" -Fraction 0.05 -Seed 42 -Robust $true
}

& $Python "scripts\build_report_assets.py" `
  --dataset-root $DatasetRoot `
  --report-dir "report" `
  --primary-output "outputs_4gen_10pct"
