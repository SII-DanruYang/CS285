SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs285_hw3

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/usr/local/lib/python3.12' | grep -v 'dist-packages/torch/lib' | paste -sd ':' -)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:$PWD/cs285/scripts"

python - <<'PY'
import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("cuda:", torch.cuda.is_available())
PY
