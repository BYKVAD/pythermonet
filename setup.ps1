# setup.ps1
# Automated environment setup for pythermonetII (Windows PowerShell)
# Usage: .\setup.ps1                        — production install (core deps only)
#        .\setup.ps1 -Dev                   — dev install (pinned versions from requirements.lock, includes matplotlib)
#        .\setup.ps1 -VenvPath my_venv      — use a custom venv name/path
#        .\setup.ps1 -Dev -VenvPath my_venv — both

param(
    [switch]$Dev,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "=== pythermonetII setup ==="

# Go to script directory (project root)
Set-Location -Path $PSScriptRoot

# 1) Find Python 3.11+
$pythonCmd = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $version = & $candidate -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>$null
        if ($version) {
            $major, $minor = $version -split " "
            if ([int]$major -eq 3 -and [int]$minor -ge 11) {
                $pythonCmd = $candidate
                Write-Host "Python found: $candidate ($major.$minor)"
                break
            } else {
                Write-Host "Skipping $candidate — version $major.$minor is below 3.11"
            }
        }
    } catch {
        # candidate not found, try next
    }
}

if ($null -eq $pythonCmd) {
    Write-Error "No suitable Python 3.11+ found. Tried: python, python3, py."
    exit 1
}

# 2) Create venv if missing
if (!(Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment: $VenvPath"
    & $pythonCmd -m venv $VenvPath
} else {
    Write-Host "Virtual environment exists: $VenvPath"
}

# 3) Ensure activation script exists
if (!(Test-Path "$VenvPath\Scripts\Activate.ps1")) {
  Write-Error "Activate.ps1 not found in $VenvPath. The venv may be corrupted. Delete it and rerun."
  exit 1
}

# 4) Activate venv
Write-Host "Activating venv..."
& "$VenvPath\Scripts\Activate.ps1"

# 5) Upgrade pip tooling
Write-Host "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# 6) Install dependencies and local package
if ($Dev) {
    if (!(Test-Path ".\requirements.lock")) {
        Write-Error "requirements.lock not found. Generate it first with: pip-compile --extra dev -o requirements.lock pyproject.toml"
        exit 1
    }
    Write-Host "Installing pinned dev dependencies from requirements.lock..."
    pip install -r .\requirements.lock
    Write-Host "Installing pythermonet (editable, no dep resolution)..."
    pip install -e . --no-deps
} else {
    Write-Host "Installing pythermonet (editable)..."
    pip install -e .
}

# 8) Sanity check
Write-Host "Sanity check:"
python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy, scipy, mpmath; print('OK: numpy/scipy/mpmath imported')"

Write-Host "=== Setup completed successfully ==="
