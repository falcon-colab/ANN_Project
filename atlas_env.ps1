# atlas_env.ps1 -- point the pipeline at the Atlas results of record.
#
# Dot-source it, do not run it, or the variables die with the child process:
#
#     . .\atlas_env.ps1
#
# config.py only honours ATLAS_* when PROJECT_ENV is "atlas", which is why
# setting ATLAS_ARTIFACT_DIR alone silently left everything reading data\.
# Every script that reads or writes results needs these four set, and every
# new PowerShell window forgets them.

$repo = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path

$env:PROJECT_ENV        = "atlas"
$env:ATLAS_ARTIFACT_DIR = Join-Path $repo "results\atlas"
$env:ATLAS_WORK_DIR     = Join-Path $repo "data\work"
$env:ATLAS_DATA_DIR     = Join-Path $repo "data\zenodo"

# One thread per BLAS pool: joblib already forks workers, and oversubscription
# makes runs slower and latency figures noisy.
$env:OMP_NUM_THREADS    = "1"
$env:MKL_NUM_THREADS    = "1"
$env:OPENBLAS_NUM_THREADS = "1"

Write-Host "PROJECT_ENV=atlas" -ForegroundColor Green
Write-Host "  ARTIFACT_DIR $env:ATLAS_ARTIFACT_DIR"
python (Join-Path $repo "src\common\config.py")

$oof = Get-ChildItem (Join-Path $env:ATLAS_ARTIFACT_DIR "oof_predictions_*.csv") -ErrorAction SilentlyContinue
Write-Host "  out-of-fold files present: $($oof.Count) of 4"
if (-not (Test-Path (Join-Path $env:ATLAS_ARTIFACT_DIR "s1_features.csv"))) {
    Write-Host "  WARNING: s1_features.csv is missing from the artifact folder;" -ForegroundColor Yellow
    Write-Host "           mlp_data_efficiency.py and analyze_results.py need it." -ForegroundColor Yellow
}
