$ErrorActionPreference = "Stop"

$projectPath = "c:\D\WorkSpace\[Cloud]_Company_Sync\MSC\OwnInfo\MyResearchProject\Lego_Sorter_V2\CodeBase\lego-sorter-v2"
$repoUrl = "https://github.com/Thomas-Tai/lego-sorter-v2.git"

Write-Host "Navigating to project directory..."
Set-Location -LiteralPath $projectPath

Write-Host "Checking git remotes..."
$remotes = git remote -v
if ($remotes) {
    Write-Host "Remote 'origin' found."
}
else {
    Write-Host "No remote found. Adding 'origin'..."
    git remote add origin $repoUrl
}

Write-Host "Pushing to origin/main..."
try {
    git branch -M main
    git push -u origin main
    Write-Host "Push Successful!"
}
catch {
    Write-Error "Push Failed. Details: $_"
}

Write-Host "Git Status:"
git status
exit 0
