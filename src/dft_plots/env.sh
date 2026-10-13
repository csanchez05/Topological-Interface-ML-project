# Source to activate the isolated plotting environment.
export COGE_PLOT_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${COGE_PLOT_SCRATCH:-${TMPDIR:-/tmp}}/coge-matplotlib-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${COGE_PLOT_SCRATCH:-${TMPDIR:-/tmp}}/coge-plot-cache-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
source "$COGE_PLOT_ENV_DIR/.venv/bin/activate"
