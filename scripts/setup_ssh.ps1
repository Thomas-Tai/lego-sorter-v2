$ErrorActionPreference = "Stop"

$SshDir = "$env:USERPROFILE\.ssh"
$ConfigFile = "$SshDir\config"
$KeyFile = "$SshDir\id_rsa"
$PubKeyFile = "$SshDir\id_rsa.pub"
$RemoteHost = "lego-sorter.local"
$RemoteUser = "pi"
$Alias = "legoSorter"

# 1. Ensure .ssh directory exists
if (-not (Test-Path $SshDir)) {
    Write-Host "Creating .ssh directory..."
    New-Item -ItemType Directory -Force -Path $SshDir | Out-Null
}

# 2. Configure SSH Config file
$ConfigContent = @"

Host $Alias
    HostName $RemoteHost
    User $RemoteUser
    ServerAliveInterval 60
    ServerAliveCountMax 120
"@

if (Test-Path $ConfigFile) {
    if ((Get-Content $ConfigFile) -match "Host $Alias") {
        Write-Host "Configuration for '$Alias' already exists in $ConfigFile."
    }
    else {
        Write-Host "Appending configuration to $ConfigFile..."
        Add-Content -Path $ConfigFile -Value $ConfigContent
    }
}
else {
    Write-Host "Creating new config file at $ConfigFile..."
    Set-Content -Path $ConfigFile -Value $ConfigContent
}

# 3. Generate SSH Key if not exists
# Check for both private and public keys
if (-not (Test-Path $KeyFile) -or -not (Test-Path $PubKeyFile)) {
    $SshKeyGen = "C:\Windows\System32\OpenSSH\ssh-keygen.exe"
    
    if (Test-Path $KeyFile) {
        Write-Host "Private key exists but public key is missing. Regenerating public key..."
        
        try {
            # Capture output directly in PowerShell
            $PubContent = & $SshKeyGen -y -f $KeyFile
            if (-not [string]::IsNullOrWhiteSpace($PubContent)) {
                $PubContent | Set-Content -Path $PubKeyFile -NoNewline -Encoding ASCII
                Write-Host "Successfully regenerated public key."
            }
            else {
                Write-Error "ssh-keygen returned empty output."
            }
        }
        catch {
            Write-Error "Failed to run ssh-keygen: $_"
        }
    }
    else {
        Write-Host "Generating SSH key pair..."
        # Using full path to ensure it runs
        & $SshKeyGen -t rsa -b 4096 -f $KeyFile -N ""
    }
}
else {
    Write-Host "SSH key pair already exists."
}

# Double check that the public key file actually exists now
if (-not (Test-Path $PubKeyFile)) {
    Write-Error "CRITICAL ERROR: Public key file ($PubKeyFile) was not found after generation attempt."
    exit 1
}

# 4. Copy ID to Remote Host (Requires Password)
Write-Host "Copying public key to $RemoteHost..."
Write-Host "You will be asked for the password for $RemoteUser@$RemoteHost one last time."

$PublicKey = Get-Content $PubKeyFile

try {
    # Using straight ssh command to clear out conflicting entries if any and append key
    $Command = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PublicKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    ssh -o StartSSH=true $RemoteUser@$RemoteHost $Command
    
    Write-Host "---------------------------------------------------"
    Write-Host "Success! You can now connect using: ssh $Alias"
    Write-Host "---------------------------------------------------"
}
catch {
    Write-Error "Failed to copy SSH key. Please ensure the Pi is on and the password is correct."
    Write-Error $_
}
