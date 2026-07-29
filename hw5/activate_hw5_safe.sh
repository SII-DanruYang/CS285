conda activate cs285_hw3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

unset PYTHONHOME
unset PYTHONPATH
unset LD_PRELOAD

export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD

export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" \
  | tr ':' '\n' \
  | grep -v '/usr/local/lib/python3.12' \
  | grep -v 'dist-packages/torch' \
  | grep -v 'site-packages/torch' \
  | paste -sd ':' -)"

python - <<'PY'
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
PY
