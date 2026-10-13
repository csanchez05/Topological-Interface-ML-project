# Source to activate the isolated SOAP/ML environment (DScribe, scikit-learn, UMAP).
export MLENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$MLENV_DIR/.cache/pip"
export NUMBA_CACHE_DIR="$MLENV_DIR/.cache/numba"
export MPLCONFIGDIR="$MLENV_DIR/.cache/matplotlib"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Use the Slurm allocation's CPUs when present; one thread otherwise (shared login node).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
source "$MLENV_DIR/.venv/bin/activate"
