# setup.ps1
# Automated environment setup for pythermonetII (Windows PowerShell)

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
if (!(Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment: .venv"
    & $pythonCmd -m venv .venv
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

# 7) Install local package
Write-Host "Installing pythermonet (editable)..."
pip install -e .

# 8) Sanity check
Write-Host "Sanity check:"
python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy, scipy, mpmath; print('OK: numpy/scipy/mpmath imported')"

Write-Host "=== Setup completed successfully ==="
