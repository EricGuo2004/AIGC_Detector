param(
  [string]$DatasetRoot = $(if ($env:GENIMAGE_DATA_ROOT) { $env:GENIMAGE_DATA_ROOT } else { "data\GenImage_data" }),
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$PrimaryOutput = "",
  [switch]$SkipErrorExport,
  [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Get-RunFraction {
  param([string]$Name)
  if ($Name -match "10pct") { return 0.10 }
  if ($Name -match "20pct") { return 0.20 }
  if ($Name -match "50pct") { return 0.50 }
  if ($Name -match "full") { return 1.00 }
  if ($Name -match "5pct") { return 0.05 }
  if ($Name -match "1pct") { return 0.01 }
  throw "Cannot infer sample fraction from run name: $Name"
}

function Get-SelectedPrimary {
  if ($PrimaryOutput) {
    return $PrimaryOutput
  }
  $selectionPath = "report\tables\final_result_selection.csv"
  if (Test-Path $selectionPath) {
    $selected = Import-Csv $selectionPath | Where-Object { $_.selected -eq "True" } | Select-Object -First 1
    if ($null -ne $selected) {
      return $selected.run
    }
  }
  return "outputs_4gen_10pct_best"
}

& $Python "scripts\build_report_assets.py" `
  --dataset-root $DatasetRoot `
  --report-dir "report" `
  --primary-output "outputs_4gen_10pct_best"
if ($LASTEXITCODE -ne 0) {
  throw "Initial report asset build failed with exit code $LASTEXITCODE"
}

$selectedPrimary = Get-SelectedPrimary
if (-not (Test-Path $selectedPrimary)) {
  throw "Selected primary output does not exist: $selectedPrimary"
}

& $Python "scripts\build_report_assets.py" `
  --dataset-root $DatasetRoot `
  --report-dir "report" `
  --primary-output $selectedPrimary
if ($LASTEXITCODE -ne 0) {
  throw "Final report asset build failed with exit code $LASTEXITCODE"
}

if (-not $SkipErrorExport) {
  $fraction = Get-RunFraction -Name $selectedPrimary
  $featureCacheDir = if ($selectedPrimary -match "enhanced") { "feature_cache_enhanced" } else { "feature_cache" }
  foreach ($task in @("binary_ai_vs_nature", "ai_subsource_attribution")) {
    & $Python "scripts\export_error_analysis.py" `
      --dataset-root $DatasetRoot `
      --output-dir $selectedPrimary `
      --task $task `
      --sample-fraction $fraction `
      --sample-seed 42 `
      --feature-cache-dir $featureCacheDir
    if ($LASTEXITCODE -ne 0) {
      throw "Error export failed for $selectedPrimary/$task with exit code $LASTEXITCODE"
    }
  }
}

if (-not $SkipPdf) {
  $xelatex = Get-Command xelatex -ErrorAction SilentlyContinue
  $bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
  if ($null -eq $xelatex -or $null -eq $bibtex) {
    Write-Host "[pdf] xelatex/bibtex not found. Install MiKTeX or TeX Live, then rerun this script."
  } else {
    Push-Location "report"
    try {
      & xelatex -interaction=nonstopmode main.tex
      & bibtex main
      & xelatex -interaction=nonstopmode main.tex
      & xelatex -interaction=nonstopmode main.tex
      if (-not (Test-Path "main.pdf")) {
        throw "LaTeX completed without producing report\main.pdf"
      }
    } finally {
      Pop-Location
    }
  }
}

Write-Host "[complete] finalized assets with primary=$selectedPrimary"
