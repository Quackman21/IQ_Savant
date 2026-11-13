# IQ Savant for Windows -- what the Desktop shortcut runs (made by setup_pc.ps1).
#   (no argument)  sync (install newer version zips, test, commit, download latest, push), start the site, open it,
#                  add new games every 3 hours while it's open. Already running -> Open / Restart / Stop.
#   -Mode stop | restart | sync | loop
# Log: data\app_log.txt
param([string]$Mode = "start")
$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Repo
$Py   = Join-Path $Repo ".venv\Scripts\python.exe"
$Port = 8501
$Url  = "http://localhost:$Port"
$Log  = Join-Path $Repo "data\app_log.txt"
$Every = 3 * 3600
New-Item -ItemType Directory -Force -Path (Join-Path $Repo "data") | Out-Null
$env:GIT_TERMINAL_PROMPT = "0"

function Log($m) { Add-Content -Path $Log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m) }
function SiteUp { [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) }
function Run($exe, $arguments) { & $exe @arguments *>> $Log; return $LASTEXITCODE }
function Ask($text) {
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.MessageBox]::Show($text, "IQ Savant", "YesNoCancel", "Information")
}
function Tell($text) { Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show($text, "IQ Savant") }

function Ver($s) { try { [version](($s -replace '[^0-9.]', '') + ".0" * (3 - ($s.Split('.').Count - 1))) } catch { [version]"0.0" } }

function Pull {
  if ((Run git @("pull", "--rebase", "--autostash")) -eq 0) { Log "git pull ok" }
  else { Run git @("rebase", "--abort") | Out-Null; Log "git pull FAILED (conflict) -- using what's here" }
}

function InstallZips {     # ONLY versions newer than the installed one (oldest first); older ones -> old_zips
  $dev = Split-Path $Repo -Parent
  $zips = Get-ChildItem -Path $dev -Filter "IQ_Savant_v*.zip" -ErrorAction SilentlyContinue |
          Sort-Object { Ver ($_.BaseName -replace '^IQ_Savant_v', '') }
  foreach ($z in $zips) {
    $cur = (Get-Content (Join-Path $Repo "VERSION") -ErrorAction SilentlyContinue | Select-Object -First 1)
    $v = $z.BaseName -replace '^IQ_Savant_v', ''
    if ((Ver $v) -gt (Ver $cur)) {
      Expand-Archive -Path $z.FullName -DestinationPath $Repo -Force
      Remove-Item $z.FullName; Log "installed v$v (was v$cur)"
    } else {
      $old = Join-Path $dev "old_zips"; New-Item -ItemType Directory -Force -Path $old | Out-Null
      Move-Item $z.FullName $old -Force; Log "skipped $($z.Name) -- not newer than v$cur (moved to old_zips)"
    }
  }
}

function Sync {
  InstallZips
  if ((Run $Py @("-m", "pytest", "-q", "-x")) -ne 0) {
    Log "TESTS FAILED -- not pushing"; Pull; return $false
  }
  Run git @("add", "-A") | Out-Null
  & git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) {
    $v = Get-Content (Join-Path $Repo "VERSION") | Select-Object -First 1
    Run git @("commit", "-q", "-m", "Auto-sync from $env:COMPUTERNAME`: v$v") | Out-Null; Log "committed local changes"
  }
  Pull
  if ((Run git @("push", "-q")) -eq 0) { Log "git push ok" } else { Log "git push FAILED" }
  return $true
}

function StopAll {
  Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -match "IQ_Savant.ps1" -and $_.CommandLine -match "loop" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
  Log "stopped"
}

function Start-Site {
  if (-not (Test-Path $Py)) { Tell "No .venv found in $Repo -- run tools\windows\setup_pc.ps1 first."; return }
  Log "===== starting IQ Savant ====="
  $ok = Sync
  Start-Process -FilePath $Py -ArgumentList "-m", "streamlit", "run", "app.py", "--server.headless", "true",
    "--server.port", "$Port" -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Repo "data\app_out.txt") -RedirectStandardError (Join-Path $Repo "data\app_err.txt")
  for ($i = 0; $i -lt 90 -and -not (SiteUp); $i++) { Start-Sleep 1 }
  if (SiteUp) {
    Start-Process $Url; Log "site up at $Url"
    if (-not $ok) { Tell "Tests failed, so nothing was pushed to GitHub. The site is running -- see data\app_log.txt" }
  } else { Log "site did NOT start"; Tell "The site didn't start -- see data\app_err.txt"; return }
  Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    "`"$PSCommandPath`"", "-Mode", "loop"
}

function Loop {
  Start-Sleep 120
  while (SiteUp) {
    if (-not (Test-Path (Join-Path $Repo "data\overnight.lock"))) {
      Pull
      $out = & $Py -m pipeline.update --light 2>&1 | Out-String
      Add-Content -Path $Log -Value $out
    }
    for ($i = 0; $i -lt ($Every / 60) -and (SiteUp); $i++) { Start-Sleep 60 }
  }
  Log "site stopped -- updater stopped"
}

switch ($Mode) {
  "start" {
    if (SiteUp) {
      $r = Ask "IQ Savant is running.`n`nYes = open the site`nNo = restart (installs any new version)`nCancel = stop it"
      if ($r -eq "Yes") { Start-Process $Url }
      elseif ($r -eq "No") { StopAll; Start-Sleep 2; Start-Site }
      else { StopAll }
    } else { Start-Site }
  }
  "stop"    { StopAll }
  "restart" { StopAll; Start-Sleep 2; Start-Site }
  "sync"    { Sync | Out-Null }
  "loop"    { Loop }
}
