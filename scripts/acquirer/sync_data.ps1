<#
.SYNOPSIS
    Syncs collected data (DB, images, manifest) from the Acquirer Pi to local PC.
.DESCRIPTION
    Uses direct scp transfer with progress display. Simple and reliable.
.EXAMPLE
    .\scripts\acquirer\sync_data.ps1
#>

$ErrorActionPreference = "Stop"

# Configuration
$PiHost = "pi@lego-sorter.local"
$RemoteBase = "~/lego-sorter-v2"

# Local paths (using absolute paths to avoid issues)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$LocalDbDir = Join-Path $LocalRoot "data\db"
$LocalDbPath = Join-Path $LocalDbDir "lego_parts.sqlite"
$LocalImagesPath = Join-Path $LocalRoot "data\images"

Write-Host "=== Syncing Data from Acquirer Pi ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Local root: $LocalRoot" -ForegroundColor Gray

# Ensure local directories exist
New-Item -ItemType Directory -Path $LocalDbDir -Force | Out-Null
New-Item -ItemType Directory -Path $LocalImagesPath -Force | Out-Null

# ============================================================
# [1/3] Backup existing database
# ============================================================
if (Test-Path $LocalDbPath) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BackupPath = "$LocalDbPath.$Timestamp.bak"
    Write-Host "[1/3] Backing up local database..."
    Copy-Item $LocalDbPath $BackupPath
    Write-Host "  Backup: $BackupPath" -ForegroundColor Gray
    
    # Keep only last 3 backups
    $OldBackups = Get-ChildItem "$LocalDbDir\*.bak" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 3
    if ($OldBackups) {
        $OldBackups | Remove-Item -Force
        Write-Host "  Cleaned up $($OldBackups.Count) old backup(s)" -ForegroundColor Gray
    }
}
else {
    Write-Host "[1/3] No local database to backup (first sync)"
}

# ============================================================
# [2/3] Sync Database (direct scp - simple and reliable)
# ============================================================
Write-Host ""
Write-Host "[2/3] Syncing database from Pi..."

try {
    # Check if remote file exists and get size
    $RemoteDbInfo = ssh $PiHost "stat -c'%s' $RemoteBase/data/db/lego_parts.sqlite 2>/dev/null"
    
    if ($RemoteDbInfo) {
        $RemoteSize = [int64]$RemoteDbInfo
        Write-Host ("  Remote size: {0:N1} MB" -f ($RemoteSize / 1MB)) -ForegroundColor Gray
        
        # Compare with local size (quick check)
        $NeedSync = $true
        if (Test-Path $LocalDbPath) {
            $LocalSize = (Get-Item $LocalDbPath).Length
            if ($LocalSize -eq $RemoteSize) {
                # Same size - do hash check
                Write-Host "  Same size - checking hash..." -ForegroundColor Gray
                $RemoteHash = ssh $PiHost "md5sum $RemoteBase/data/db/lego_parts.sqlite | cut -d' ' -f1"
                $LocalHash = (Get-FileHash $LocalDbPath -Algorithm MD5).Hash.ToLower()
                if ($RemoteHash -eq $LocalHash) {
                    Write-Host "  Database unchanged - skipping" -ForegroundColor Green
                    $NeedSync = $false
                }
            }
        }
        
        if ($NeedSync) {
            Write-Host "  Transferring database..." -ForegroundColor Gray
            scp "${PiHost}:${RemoteBase}/data/db/lego_parts.sqlite" "$LocalDbPath"
            $NewSize = (Get-Item $LocalDbPath).Length
            Write-Host ("  Database synced: {0:N1} MB" -f ($NewSize / 1MB)) -ForegroundColor Green
        }
    }
    else {
        Write-Host "  No database found on Pi" -ForegroundColor Yellow
    }
}
catch {
    Write-Warning "Failed to sync database: $_"
}

# ============================================================
# [3/3] Sync Images (scp -r)
# ============================================================
Write-Host ""
Write-Host "[3/3] Syncing images from Pi..."

try {
    # Check if remote images exist
    $RemoteImageCount = ssh $PiHost "find $RemoteBase/data/images/raw -name '*.jpg' 2>/dev/null | wc -l"
    Write-Host "  Remote images: $RemoteImageCount files" -ForegroundColor Gray
    
    if ([int]$RemoteImageCount -gt 0) {
        Write-Host "  Transferring images..." -ForegroundColor Gray
        scp -r "${PiHost}:${RemoteBase}/data/images/." "$LocalImagesPath"
        
        # Count local images
        $LocalImageCount = (Get-ChildItem "$LocalImagesPath\raw" -Recurse -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host "  Images synced: $LocalImageCount files" -ForegroundColor Green
    }
    else {
        Write-Host "  No images found on Pi" -ForegroundColor Yellow
    }
}
catch {
    Write-Warning "Failed to sync images: $_"
}

# ============================================================
# Summary
# ============================================================
Write-Host ""
Write-Host "=== Sync Complete ===" -ForegroundColor Cyan
Write-Host ""

# Show final stats
$DbExists = Test-Path $LocalDbPath
$ImgCount = (Get-ChildItem "$LocalImagesPath\raw" -Recurse -Filter *.jpg -ErrorAction SilentlyContinue).Count

Write-Host "Results:" -ForegroundColor Gray
if ($DbExists) {
    $DbSize = (Get-Item $LocalDbPath).Length / 1MB
    Write-Host ("  Database: {0:N1} MB" -f $DbSize)
}
Write-Host "  Images: $ImgCount files"
Write-Host ""
Write-Host "Paths:" -ForegroundColor Gray
Write-Host "  Database: $LocalDbPath"
Write-Host "  Images:   $LocalImagesPath"
