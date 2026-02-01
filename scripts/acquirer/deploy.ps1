<#
.SYNOPSIS
    Deploys the Lego Sorter codebase to the Acquirer Raspberry Pi.
.DESCRIPTION
    Uses rsync for efficient delta transfers. Only changed files are synced.
    Auto-detects rsync from Git for Windows installation.
.PARAMETER Clean
    If specified, removes code folders on Pi before deploying (preserves data/).
.EXAMPLE
    .\scripts\acquirer\deploy.ps1
.EXAMPLE
    .\scripts\acquirer\deploy.ps1 -Clean
#>

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Configuration
$PiHost = "pi@lego-sorter.local"  # Direct hostname (SSH alias had issues)
$RemotePath = "~/lego-sorter-v2"

Write-Host "=== Deploying to Acquirer Pi ===" -ForegroundColor Cyan
Write-Host ""

# Function to find rsync
function Find-Rsync {
    # Check if rsync is in PATH
    $rsync = Get-Command rsync -ErrorAction SilentlyContinue
    if ($rsync) { return $rsync.Source }

    # Common Git for Windows and cwRsync installation paths
    $GitPaths = @(
        "C:\Program Files\Git\usr\bin\rsync.exe",
        "C:\Program Files (x86)\Git\usr\bin\rsync.exe",
        "$env:LOCALAPPDATA\Programs\Git\usr\bin\rsync.exe",
        "$env:USERPROFILE\scoop\apps\git\current\usr\bin\rsync.exe",
        "C:\Program Files\cwRsync\bin\rsync.exe",
        "C:\Program Files (x86)\cwRsync\bin\rsync.exe",
        "C:\cwRsync\bin\rsync.exe"
    )

    foreach ($path in $GitPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

$RsyncPath = Find-Rsync
if ($RsyncPath) {
    Write-Host "Using rsync: $RsyncPath" -ForegroundColor Gray
    $UseRsync = $true
}
else {
    Write-Warning "rsync not found. Falling back to scp (slower)."
    Write-Host "  Tip: Install Git for Windows to get rsync" -ForegroundColor Yellow
    $UseRsync = $false
}
# SSH options to prevent hanging
$SshOpts = @("-o", "ConnectTimeout=10", "-o", "BatchMode=yes")

# Clean option: remove code folders only (preserves data/)
if ($Clean) {
    Write-Host ""
    Write-Host "[0/4] Cleaning code folders on Pi (preserving data/)..." -ForegroundColor Yellow
    ssh $SshOpts $PiHost "sudo rm -rf $RemotePath/modules $RemotePath/scripts $RemotePath/config"
    Write-Host "  Cleaned: modules/, scripts/, config/" -ForegroundColor Gray
    Write-Host "  Preserved: data/ (database, images)" -ForegroundColor Green
}

# Ensure remote directory exists
Write-Host ""
Write-Host "[1/4] Creating remote directories..."
ssh $SshOpts $PiHost "mkdir -p $RemotePath/modules $RemotePath/scripts/acquirer $RemotePath/sorter_app $RemotePath/data/db $RemotePath/config"

if ($UseRsync) {
    # rsync options:
    # -a: archive mode (preserves permissions, timestamps)
    # -v: verbose
    # -z: compress during transfer
    # --progress: show progress
    # -e ssh: use SSH for transport

    Write-Host ""
    Write-Host "[2/4] Syncing modules/ & sorter_app/ (rsync -avz)..."
    & $RsyncPath -avz --progress -e ssh ./modules/ "${PiHost}:${RemotePath}/modules/"
    & $RsyncPath -avz --progress -e ssh ./sorter_app/ "${PiHost}:${RemotePath}/sorter_app/"

    Write-Host ""
    Write-Host "[3/4] Syncing scripts/acquirer/ and config/..."
    & $RsyncPath -avz --progress -e ssh ./scripts/acquirer/ "${PiHost}:${RemotePath}/scripts/acquirer/"
    & $RsyncPath -avz --progress -e ssh ./config/ "${PiHost}:${RemotePath}/config/"

    Write-Host ""
    Write-Host "[4/4] Syncing database and requirements..."
    & $RsyncPath -avz --progress -e ssh ./data/db/ "${PiHost}:${RemotePath}/data/db/"
    & $RsyncPath -avz --progress -e ssh ./requirements.txt "${PiHost}:${RemotePath}/"
    & $RsyncPath -avz --progress -e ssh ./requirements-pi.txt "${PiHost}:${RemotePath}/"
}
else {
    # Fallback to scp with hash check for large files
    Write-Host ""
    Write-Host "[2/4] Syncing modules/ & sorter_app/ (scp)..."
    scp -r ./modules "${PiHost}:${RemotePath}/"
    scp -r ./sorter_app "${PiHost}:${RemotePath}/"

    Write-Host ""
    Write-Host "[3/4] Syncing scripts/acquirer/ and config/..."
    scp -r ./scripts/acquirer "${PiHost}:${RemotePath}/scripts/"
    scp -r ./config "${PiHost}:${RemotePath}/"

    Write-Host ""
    Write-Host "[4/4] Syncing database and requirements..."
        
    $LocalDb = "./data/db/lego_parts.sqlite"
    if (Test-Path $LocalDb) {
        Write-Host "  Checking if database needs sync (hash comparison)..." -ForegroundColor Gray
            
        # Get local MD5 hash
        $LocalHash = (Get-FileHash $LocalDb -Algorithm MD5).Hash
        Write-Host "    Local hash:  $LocalHash" -ForegroundColor Gray
            
        # Get remote MD5 hash (if file exists)
        $RemoteHash = ssh $PiHost "md5sum $RemotePath/data/db/lego_parts.sqlite 2>/dev/null | cut -d' ' -f1" 2>$null
        if ($RemoteHash) {
            $RemoteHash = $RemoteHash.ToUpper()
            Write-Host "    Remote hash: $RemoteHash" -ForegroundColor Gray
        }
        else {
            $RemoteHash = ""
            Write-Host "    Remote hash: (file not found)" -ForegroundColor Gray
        }
            
        if ($LocalHash -eq $RemoteHash) {
            Write-Host "  Database unchanged - skipping transfer" -ForegroundColor Green
        }
        else {
            Write-Host "  Database changed - compressing and transferring..."
                
            # Compress on Windows using PowerShell
            $LocalDbFull = (Resolve-Path $LocalDb).Path
            $CompressedDb = $LocalDbFull + ".gz"
            Write-Host "    Compressing..." -ForegroundColor Gray
                
            # Use .NET GZip compression
            $sourceStream = [System.IO.File]::OpenRead($LocalDbFull)
            $destStream = [System.IO.File]::Create($CompressedDb)
            $gzipStream = New-Object System.IO.Compression.GZipStream($destStream, [System.IO.Compression.CompressionMode]::Compress)
            $sourceStream.CopyTo($gzipStream)
            $gzipStream.Close()
            $destStream.Close()
            $sourceStream.Close()
                
            $origSize = (Get-Item $LocalDb).Length / 1MB
            $compSize = (Get-Item $CompressedDb).Length / 1MB
            Write-Host ("    Compressed: {0:N1}MB -> {1:N1}MB ({2:P0} reduction)" -f $origSize, $compSize, (1 - $compSize / $origSize)) -ForegroundColor Gray
                
            # Transfer compressed file
            Write-Host "    Transferring..." -ForegroundColor Gray
            scp $CompressedDb "${PiHost}:${RemotePath}/data/db/"
                
            # Decompress on Pi
            Write-Host "    Decompressing on Pi..." -ForegroundColor Gray
            ssh $PiHost "cd $RemotePath/data/db && gunzip -f lego_parts.sqlite.gz"
                
            # Clean up local compressed file
            Remove-Item $CompressedDb -Force
            Write-Host "  Database synced successfully." -ForegroundColor Green
        }
    }
        
    scp ./requirements.txt "${PiHost}:${RemotePath}/"
    scp ./requirements-pi.txt "${PiHost}:${RemotePath}/"
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. ssh $PiHost"
Write-Host "  2. pip install -r ~/lego-sorter-v2/requirements-pi.txt"
Write-Host "  3. export LEGO_API_URL='http://<PC_IP>:8000'"
Write-Host "  4. python ~/lego-sorter-v2/sorter_app/main.py"
