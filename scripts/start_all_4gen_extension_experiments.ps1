param(
  [string]$DatasetRoot = $(if ($env:GENIMAGE_DATA_ROOT) { $env:GENIMAGE_DATA_ROOT } else { "data\GenImage_data" }),
  [ValidateSet("cpu", "gpu")]
  [string]$LightgbmDevice = "gpu",
  [int]$NumWorkers = 16,
  [int]$FeatureChunksize = 64,
  [double]$LogoFraction = 0.20,
  [double]$CandidateFraction = 0.20,
  [double]$RobustnessFraction = 0.20,
  [int]$SampleSeed = 42,
  [switch]$Force,
  [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "experiment_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $LogDir "all_4gen_extension_${timestamp}.out.log"
$stderr = Join-Path $LogDir "all_4gen_extension_${timestamp}.err.log"
$commandFile = Join-Path $LogDir "run_all_4gen_extension_${timestamp}.ps1"

$runner = Join-Path $ProjectRoot "scripts\run_all_4gen_extension_experiments.ps1"
$argsList = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "`"$runner`"",
  "-DatasetRoot", "`"$DatasetRoot`"",
  "-LightgbmDevice", $LightgbmDevice,
  "-NumWorkers", "$NumWorkers",
  "-FeatureChunksize", "$FeatureChunksize",
  "-LogoFraction", "$LogoFraction",
  "-CandidateFraction", "$CandidateFraction",
  "-RobustnessFraction", "$RobustnessFraction",
  "-SampleSeed", "$SampleSeed"
)
if ($Force) {
  $argsList += "-Force"
}
if ($SkipValidation) {
  $argsList += "-SkipValidation"
}

$commandText = @"
Set-Location "$ProjectRoot"
powershell.exe $($argsList -join " ")
"@
$commandText | Set-Content -LiteralPath $commandFile -Encoding UTF8

$process = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList $argsList `
  -WorkingDirectory $ProjectRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Write-Host "[started] all 4-generator extension experiments"
Write-Host "PID: $($process.Id)"
Write-Host "stdout: $stdout"
Write-Host "stderr: $stderr"
Write-Host "command: $commandFile"
Write-Host ""
Write-Host "Progress check:"
Write-Host "  Get-Content -Wait `"$stdout`""
Write-Host "  Get-Content -Wait `"$stderr`""
