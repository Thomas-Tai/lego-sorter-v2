# Raspberry Pi SSH Setup Guide

This guide describes how to configure a persistent, passwordless SSH connection from Windows to the Raspberry Pi.

## Prerequisites
*   Windows 10/11
*   PowerShell
*   Raspberry Pi connected to the network (hostname: `lego-sorter.local`)

## Automated Setup
A helper script `setup_ssh.ps1` is included in this repository. Run it in PowerShell to automatically configure your SSH config and keys:
```powershell
.\scripts\setup_ssh.ps1
```

## Manual Setup Steps

If you prefer to set it up manually, follow these steps:

### 1. Generate SSH Key
Open PowerShell and check if you have an SSH key. If not, generated one:
```powershell
ssh-keygen -t rsa -b 4096
```
*   Press **Enter** to accept default defaults.
*   If asked to overwrite, be careful (only `y` if you don't use the existing key).
*   Press **Enter** twice for an empty passphrase (for automatic login).

### 2. Configure SSH Alias
Create or edit your SSH config file at `~/.ssh/config` (User Profile -> .ssh -> config).
Add the following entry:

```text
Host legoSorter
    HostName lego-sorter.local
    User pi
    ServerAliveInterval 60
    ServerAliveCountMax 120
```

*   **Host**: The short name you will use (e.g., `ssh legoSorter`).
*   **ServerAliveInterval**: Keeps the connection alive to prevent timeouts.

### 3. Copy Key to Raspberry Pi
Run the following command to copy your public key to the Pi (this enables passwordless login).
*Replace `lego-sorter.local` with the IP address if the hostname doesn't resolve.*

```powershell
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh pi@lego-sorter.local "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
*   You will be asked to enter the Raspberry Pi's password one last time.

### 4. Connect
You can now connect using the alias:
```powershell
ssh legoSorter
```

## Troubleshooting
*   **"Host does not exist"**: Ensure the Pi is powered on and connected to the same network. Try `ping lego-sorter.local`.
*   **"Permission denied"**: Check if your public key was correctly copied to `~/.ssh/authorized_keys` on the Pi.
