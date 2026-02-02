# Pre-Push CI Check Script
# Run this before pushing to GitHub to ensure CI will pass

$ErrorActionPreference = "Stop"
$PYTHON = "C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\LegoSorterProject\Environments\venv_server\Scripts\python.exe"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Pre-Push CI Check" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Change to project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
Write-Host "Working directory: $ProjectRoot`n"

# 1. Black Formatting
Write-Host "[1/4] Running Black formatter..." -ForegroundColor Yellow
& $PYTHON -m black . 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Black formatting applied. Re-checking..." -ForegroundColor Yellow
}
& $PYTHON -m black . --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Black check failed" -ForegroundColor Red
    exit 1
}
Write-Host "PASS: Black" -ForegroundColor Green

# 2. Flake8
Write-Host "`n[2/4] Running Flake8..." -ForegroundColor Yellow
& $PYTHON -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Flake8 found errors" -ForegroundColor Red
    exit 1
}
Write-Host "PASS: Flake8" -ForegroundColor Green

# 3. Mypy
Write-Host "`n[3/4] Running Mypy..." -ForegroundColor Yellow
& $PYTHON -m mypy .
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Mypy found type errors" -ForegroundColor Red
    exit 1
}
Write-Host "PASS: Mypy" -ForegroundColor Green

# 4. Pytest
Write-Host "`n[4/4] Running Pytest..." -ForegroundColor Yellow
& $PYTHON -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Tests failed" -ForegroundColor Red
    exit 1
}
Write-Host "PASS: Pytest" -ForegroundColor Green

# Summary
Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "All checks passed! Safe to push." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
