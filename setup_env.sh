#!/usr/bin/env bash
# Create the project environment at review-ai/.venv (the one every result was produced with).
#
#   bash setup_env.sh            # direct dependencies (requirements.txt)
#   LOCK=1 bash setup_env.sh     # exact full environment (requirements-lock.txt)
#
# Requires Python 3.10 (python3.10 on PATH, or set PYTHON=/path/to/python3.10).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.10}"
VENV=review-ai/.venv
REQ=requirements.txt
[[ "${LOCK:-0}" == 1 ]] && REQ=requirements-lock.txt

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.10 or set PYTHON=..." >&2
  exit 1
fi
echo "Python: $("$PYTHON" --version)"

if [[ -d "$VENV" ]]; then
  echo "Environment already exists: $VENV (delete it first to rebuild)"
else
  echo "Creating $VENV"
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REQ"

echo
echo "Check:"
"$VENV/bin/python" - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__} | transformers {transformers.__version__} | CUDA available: {torch.cuda.is_available()}")
PY
echo "Done. Use: review-ai/.venv/bin/python <script>  (or: source review-ai/.venv/bin/activate)"
