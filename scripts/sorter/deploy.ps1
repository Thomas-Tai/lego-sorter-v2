<#
.SYNOPSIS
    Deploys the Lego Sorter codebase to the Sorter Raspberry Pi.
.DESCRIPTION
    Uses rsync for efficient delta transfers. Only changed files are synced.
    Deploys sorter_app, modules, and configuration needed for Pi-side inference.
.PARAMETER Clean
    If specified, removes code folders on Pi before deploying (preserves data/).
.EXAMPLE
    .\scripts\sorter\deploy.ps1
.EXAMPLE
    .\scripts\sorter\deploy.ps1 -Clean
#>

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Configuration
$PiHost = "pi@lego-sorter.local"  # Adjust hostname if different
$RemotePath = "~/lego-sorter-v2"

# Get project root (two levels up from this script)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..").Path

Write-Host "=== Deploying to Sorter Pi ===" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray

# Change to project root
Push-Location $ProjectRoot
Write-Host ""

# Function to find rsync
function Find-Rsync {
    $rsync = Get-Command rsync -ErrorAction SilentlyContinue
    if ($rsync) { return $rsync.Source }

    $GitPaths = @(
        "C:\Program Files\Git\usr\bin\rsync.exe",
        "C:\Program Files (x86)\Git\usr\bin\rsync.exe",
        "$env:LOCALAPPDATA\Programs\Git\usr\bin\rsync.exe",
        "$env:USERPROFILE\scoop\apps\git\current\usr\bin\rsync.exe"
    )

    foreach ($path in $GitPaths) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

$RsyncPath = Find-Rsync
if ($RsyncPath) {
    Write-Host "Using rsync: $RsyncPath" -ForegroundColor Gray
    $UseRsync = $true
} else {
    Write-Warning "rsync not found. Falling back to scp (slower)."
    $UseRsync = $false
}

$SshOpts = @("-o", "ConnectTimeout=10", "-o", "BatchMode=yes")

# Clean option
if ($Clean) {
    Write-Host ""
    Write-Host "[0/5] Cleaning code folders on Pi (preserving data/)..." -ForegroundColor Yellow
    ssh $SshOpts $PiHost "rm -rf $RemotePath/modules $RemotePath/sorter_app $RemotePath/scripts/sorter $RemotePath/config"
    Write-Host "  Cleaned: modules/, sorter_app/, scripts/sorter/, config/" -ForegroundColor Gray
}

# Create remote directories
Write-Host ""
Write-Host "[1/5] Creating remote directories..."
ssh $SshOpts $PiHost "mkdir -p $RemotePath/modules $RemotePath/sorter_app $RemotePath/scripts/sorter $RemotePath/config $RemotePath/data/captures"

if ($UseRsync) {
    Write-Host ""
    Write-Host "[2/5] Syncing sorter_app/..."
    & $RsyncPath -avz --progress -e ssh ./sorter_app/ "${PiHost}:${RemotePath}/sorter_app/"

    Write-Host ""
    Write-Host "[3/5] Syncing modules/..."
    & $RsyncPath -avz --progress -e ssh ./modules/ "${PiHost}:${RemotePath}/modules/"

    Write-Host ""
    Write-Host "[4/5] Syncing scripts/sorter/ and config/..."
    & $RsyncPath -avz --progress -e ssh ./scripts/sorter/ "${PiHost}:${RemotePath}/scripts/sorter/"
    & $RsyncPath -avz --progress -e ssh ./config/ "${PiHost}:${RemotePath}/config/"

    Write-Host ""
    Write-Host "[5/5] Syncing requirements..."
    & $RsyncPath -avz --progress -e ssh ./requirements.txt "${PiHost}:${RemotePath}/"
    & $RsyncPath -avz --progress -e ssh ./requirements-pi.txt "${PiHost}:${RemotePath}/"
} else {
    Write-Host ""
    Write-Host "[2/5] Syncing sorter_app/ (scp)..."
    scp -r ./sorter_app "${PiHost}:${RemotePath}/"

    Write-Host ""
    Write-Host "[3/5] Syncing modules/..."
    scp -r ./modules "${PiHost}:${RemotePath}/"

    Write-Host ""
    Write-Host "[4/5] Syncing scripts/sorter/ and config/..."
    scp -r ./scripts/sorter "${PiHost}:${RemotePath}/scripts/"
    scp -r ./config "${PiHost}:${RemotePath}/"

    Write-Host ""
    Write-Host "[5/5] Syncing requirements..."
    scp ./requirements.txt "${PiHost}:${RemotePath}/"
    scp ./requirements-pi.txt "${PiHost}:${RemotePath}/"
}

# Make run script executable
Write-Host ""
Write-Host "Making scripts executable..."
ssh $SshOpts $PiHost "chmod +x $RemotePath/scripts/sorter/*.sh 2>/dev/null || true"

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. ssh $PiHost"
Write-Host "  2. cd ~/lego-sorter-v2"
Write-Host "  3. pip install -r requirements-pi.txt"
Write-Host "  4. Edit scripts/sorter/sorter.env with your PC's IP address"
Write-Host "  5. ./scripts/sorter/run_sorter.sh"
Write-Host ""
Write-Host "PC-Side (run in separate terminal):" -ForegroundColor Yellow
Write-Host "  python run_api.py --host 0.0.0.0 --port 8000"

# Restore original location
Pop-Location
