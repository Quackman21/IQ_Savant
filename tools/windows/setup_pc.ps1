# One-time setup on a Windows PC, run from the project folder after cloning:
#     powershell -ExecutionPolicy Bypass -File tools\windows\setup_pc.ps1
# Makes the Python environment (.venv), installs everything, runs the tests, and puts "IQ Savant" on the Desktop.
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Repo
Write-Host "Project: $Repo"

$py = $null
foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
  & py "-$v" --version *> $null
  if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
}
if (-not $py) { throw "Python not found. Install Python 3.12 from python.org (tick 'Add python.exe to PATH'), then run this again." }
Write-Host "Using Python $($py[1])"

if (-not (Test-Path ".venv")) { & $py[0] $py[1] -m venv .venv }
$Pip = ".venv\Scripts\python.exe"
& $Pip -m pip install --upgrade pip
& $Pip -m pip install -r requirements.txt
& $Pip -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "Some tests failed -- send the output to Claude." -ForegroundColor Yellow }

$desk = [Environment]::GetFolderPath("Desktop")
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desk "IQ Savant.lnk"))
$lnk.TargetPath = "powershell.exe"
$lnk.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Repo\tools\windows\IQ_Savant.ps1`""
$lnk.WorkingDirectory = $Repo
$lnk.IconLocation = "$Repo\tools\IQ_icon.ico"
$lnk.Description = "IQ Savant"
$lnk.Save()
Write-Host "`nDone. 'IQ Savant' is on your Desktop -- double-click it to start the site." -ForegroundColor Green
Write-Host "Right-click it -> Pin to taskbar to keep it there."
