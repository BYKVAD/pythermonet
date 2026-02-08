# setup.ps1
# Automated environment setup for pythermonetII (Windows PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=== pythermonetII setup ==="

# Go to script directory (project root)
Set-Location -Path $PSScriptRoot

# 1) Ensure Python exists
try {
  $py = (Get-Command python).Source
  Write-Host "Python found: $py"
} catch {
  Write-Error "Python not found in PATH. Install Python 3.10+ and ensure 'python' is available."
  exit 1
}

# 2) Create venv if missing
if (!(Test-Path ".\.venv")) {
  Write-Host "Creating virtual environment: .venv"
  python -m venv .venv
} else {
  Write-Host "Virtual environment exists: .venv"
}

# 3) Ensure activation script exists
if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
  Write-Error "Activate.ps1 not found. The venv may be corrupted. Delete .venv and rerun."
  exit 1
}

# 4) Activate venv
Write-Host "Activating venv..."
& .\.venv\Scripts\Activate.ps1

# 5) Upgrade pip tooling
Write-Host "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# 6) Install requirements
if (!(Test-Path ".\requirements.txt")) {
  Write-Error "requirements.txt not found in project root."
  exit 1
}

Write-Host "Installing requirements..."
pip install -r .\requirements.txt

# 7) Sanity check
Write-Host "Sanity check:"
python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy, scipy, mpmath; print('OK: numpy/scipy/mpmath imported')"

Write-Host "=== Setup completed successfully ==="
