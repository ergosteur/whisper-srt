# Usage examples:
#   .\setup_venv.ps1                 # auto-detect (CUDA if present, else CPU)
#   .\setup_venv.ps1 -Flavour cpu
#   .\setup_venv.ps1 -Flavour cu121
#   .\setup_venv.ps1 -Flavour cu124

param(
  [ValidateSet("auto","cpu","cu121","cu124")]
  [string]$Flavour = "auto",
  [string]$Python = "python",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

& $Python -m venv $VenvDir
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip wheel

function Get-Flavour {
  param([string]$Requested)

  if ($Requested -eq "cpu") { return "cpu" }
  # Try nvidia-smi
  $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($nvidia) {
    if ($Requested -eq "cu124") { return "cu124" }
    if ($Requested -eq "cu121") { return "cu121" }
    return "cu124" # sensible default
  }
  return "cpu"
}

$Chosen = Get-Flavour -Requested $Flavour
Write-Host "[i] Installing PyTorch flavour: $Chosen"

$Py = "$VenvDir\Scripts\python.exe"
switch ($Chosen) {
  "cu124" { & $Py -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio }
  "cu121" { & $Py -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio }
  "cpu"   { & $Py -m pip install --index-url https://download.pytorch.org/whl/cpu  torch torchaudio }
  default { throw "Unknown flavour $Chosen" }
}

& $Py -m pip install -r requirements-core.txt

Write-Host "`n[ok] Virtualenv ready in $VenvDir"
Write-Host "To use it: `"$VenvDir\Scripts\Activate.ps1`""
