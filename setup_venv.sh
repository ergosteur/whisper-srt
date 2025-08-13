#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./setup_venv.sh                 # auto-detect (CUDA if present, else CPU)
#   ./setup_venv.sh cpu             # force CPU
#   ./setup_venv.sh cu121           # force CUDA 12.1 wheels
#   ./setup_venv.sh cu124           # force CUDA 12.4 wheels

TORCH_FLAVOUR="${1:-auto}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

$PYTHON_BIN -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

choose_flavour() {
  if [[ "$TORCH_FLAVOUR" == "cpu" ]]; then
    echo "cpu"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Heuristic: prefer cu124, fall back to cu121
    if [[ -n "$(python - <<'PY'
import subprocess, re
try:
    out=subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
    m=re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', out)
    print(m.group(1) if m else '')
except Exception:
    print('')
PY
)" ]]; then
      # If user forced a specific CUDA, honour it
      if [[ "$TORCH_FLAVOUR" == "cu124" ]]; then echo "cu124"; return; fi
      if [[ "$TORCH_FLAVOUR" == "cu121" ]]; then echo "cu121"; return; fi
      # Default to newest common wheel:
      echo "cu124"; return
    fi
  fi
  echo "cpu"
}

FLAVOUR="$(choose_flavour)"
echo "[i] Installing PyTorch flavour: ${FLAVOUR}"

case "$FLAVOUR" in
  cu124)
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
    ;;
  cu121)
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
    ;;
  cpu)
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    ;;
  *)
    echo "[!] Unknown flavour '$FLAVOUR'"; exit 1;;
esac

# Now install the core deps
python -m pip install -r requirements-core.txt

echo
echo "[ok] Virtualenv ready in ${VENV_DIR}"
echo "To use it: source ${VENV_DIR}/bin/activate"
