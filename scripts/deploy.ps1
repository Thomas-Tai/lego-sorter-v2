<#
.SYNOPSIS
    Deploys the Lego Sorter codebase to the Raspberry Pi.
.DESCRIPTION
    Uses SCP to sync source code to the Pi. Run from the lego-sorter-v2 directory.
.EXAMPLE
    .\scripts\deploy.ps1
#>

$ErrorActionPreference = "Stop"

# Configuration
$PiHost = "legoSorter"  # SSH alias from ~/.ssh/config
$RemotePath = "~/lego-sorter-v2"

# Directories to sync
$SyncDirs = @("sorter_app", "tools", "config", "data", "scripts", "Project_Manage", "tests")
$SyncFiles = @("requirements.txt", "run_importer.py", "run_acquirer.py", "pytest.ini", "lego_sorter.py")

Write-Host "=== Deploying to Raspberry Pi ===" -ForegroundColor Cyan
Write-Host ""

# Ensure remote directory exists
Write-Host "[1/3] Creating remote directory..."
ssh $PiHost "mkdir -p $RemotePath"

# Sync directories
Write-Host ""
Write-Host "[2/3] Syncing directories..."
foreach ($dir in $SyncDirs) {
    if (Test-Path $dir) {
        Write-Host "  - $dir"
        scp -r $dir "${PiHost}:${RemotePath}/"
    }
}

# Sync files
Write-Host ""
Write-Host "[3/3] Syncing files..."
foreach ($file in $SyncFiles) {
    if (Test-Path $file) {
        Write-Host "  - $file"
        scp $file "${PiHost}:${RemotePath}/"
    }
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. ssh $PiHost"
Write-Host "  2. bash ~/lego-sorter-v2/scripts/setup_pi.sh  # First time only"
Write-Host "  3. source ~/lego-sorter-env/bin/activate"
Write-Host "  4. python run_acquirer.py"
