#!/usr/bin/env bash
#SBATCH --job-name=coge100-plots
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/results/CoGe_100_plots/logs/%A_%a.out
set -euo pipefail
export COGE_PLOT_SCRATCH
COGE_PLOT_SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/coge-plots-${SLURM_JOB_ID}-XXXXXX")"
trap 'rm -rf -- "$COGE_PLOT_SCRATCH"' EXIT
source src/dft_plots/env.sh
PLOT_PYTHON="$(/usr/bin/python3 -B src/dft_plots/coge_bulk/stage_environment.py)"
export VIRTUAL_ENV="${PLOT_PYTHON%/bin/python}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "Rendering with node-local environment: $PLOT_PYTHON"
"$PLOT_PYTHON" -B -u -m src.dft_plots.coge_bulk.plot_all --index "$SLURM_ARRAY_TASK_ID" --dpi 160
