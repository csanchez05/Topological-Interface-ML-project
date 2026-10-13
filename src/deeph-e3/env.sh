# Source from Bash to activate the project-local DeepH-E3 environment.
export DEEPHE3_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DEEPHE3_PROJECT_DIR="$(cd -- "$DEEPHE3_SETUP_DIR/../.." && pwd)"
export DEEPHE3_CALCULATIONS_DIR="$DEEPHE3_PROJECT_DIR/calculations/deeph-e3"
export DEEPHE3_DIR="$DEEPHE3_SETUP_DIR/source"

# Activate first: virtualenv activation can restore PATH from a previous environment.
if [[ -f "$DEEPHE3_SETUP_DIR/.venv/bin/activate" ]]; then
    source "$DEEPHE3_SETUP_DIR/.venv/bin/activate"
fi
export PATH="$DEEPHE3_SETUP_DIR/.julia-runtime/bin:$PATH"
export XDG_CACHE_HOME="$DEEPHE3_SETUP_DIR/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TMPDIR="$DEEPHE3_SETUP_DIR/.tmp"
export PYTHONNOUSERSITE=1
export PYTHON="$DEEPHE3_SETUP_DIR/.venv/bin/python"
export JULIA_PROJECT="$DEEPHE3_SETUP_DIR/julia-project"
export JULIA_DEPOT_PATH="$DEEPHE3_SETUP_DIR/.julia-depot"
export JULIA_PKG_PRECOMPILE_AUTO=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export JULIA_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$DEEPHE3_CALCULATIONS_DIR"
