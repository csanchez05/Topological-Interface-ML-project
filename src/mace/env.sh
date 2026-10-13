# Source this file from Bash to activate the project-local MACE-H environment.
export MACE_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MACE_PROJECT_DIR="$(cd -- "$MACE_SETUP_DIR/../.." && pwd)"
export MACE_CALCULATIONS_DIR="$MACE_PROJECT_DIR/calculations/mace"
export MACEH_DIR="$MACE_SETUP_DIR/source"
export XDG_CACHE_HOME="$MACE_SETUP_DIR/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TMPDIR="$MACE_SETUP_DIR/.tmp"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export JULIA_PROJECT="$MACE_SETUP_DIR/julia-project"
export JULIA_DEPOT_PATH="$MACE_SETUP_DIR/.julia-depot"
export JULIA_NUM_THREADS="$OMP_NUM_THREADS"
export JULIA_PKG_PRECOMPILE_AUTO=0
export PATH="$MACE_SETUP_DIR/.julia-runtime/bin:$PATH"
export PYTHON="$MACE_SETUP_DIR/.venv/bin/python"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$MACE_CALCULATIONS_DIR"
if [[ -f "$MACE_SETUP_DIR/.venv/bin/activate" ]]; then
    source "$MACE_SETUP_DIR/.venv/bin/activate"
fi
