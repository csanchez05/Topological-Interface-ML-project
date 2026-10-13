# Source from Bash to activate the project-local Uni-HamGNN environment.
export UNIHAMGNN_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UNIHAMGNN_PROJECT_DIR="$(cd -- "$UNIHAMGNN_SETUP_DIR/../.." && pwd)"
export UNIHAMGNN_CALCULATIONS_DIR="$UNIHAMGNN_PROJECT_DIR/calculations/uni-hamgnn"
export UNIHAMGNN_DIR="$UNIHAMGNN_SETUP_DIR/source"
export UNIHAMGNN_MODEL="$UNIHAMGNN_SETUP_DIR/models/uni-hamgnn_2_1.pkl"

if [[ -f "$UNIHAMGNN_SETUP_DIR/.venv/bin/activate" ]]; then
    source "$UNIHAMGNN_SETUP_DIR/.venv/bin/activate"
fi
export PYTHON="$UNIHAMGNN_SETUP_DIR/.venv/bin/python"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$UNIHAMGNN_SETUP_DIR/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TMPDIR="$UNIHAMGNN_SETUP_DIR/.tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" \
    "$PIP_CACHE_DIR" "$TMPDIR" "$UNIHAMGNN_CALCULATIONS_DIR"
