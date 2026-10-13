#!/usr/bin/env bash
#SBATCH --job-name=coge-ace
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/descriptors/ACE/logs/ace-%j.out
#
# Stage 1, steps 1.1-1.3 with ACE: bulk ACE, bulk invariance and handedness check, GB ACE, GB checks.
# Cost (smoke test 2026-10-13 on qcg3-0): about 10 CPU-ms per atom at the default settings, so the
# 4.56 M atoms of the catalogue take about 13 CPU-h: about 15 min on 64 cores. Workers stay below
# 0.5 GB each; the main process holds about 4.4 GB of vectors per radius (float32).
# Settings are read from the environment, so a second run is one line:
#   N3=6 L3=4 sbatch src/ACE/CoGe/run_ace.slurm
#   DENSITY=dirac sbatch src/ACE/CoGe/run_ace.slurm
#   NU_MAX=2 RADII="4 6 8 10" sbatch src/ACE/CoGe/run_ace.slurm
# OVERWRITE=1 replaces an existing run with the same ACE settings.
# The steps run in order; a failed check does not stop the later steps, but the job then exits 1.
set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p calculations/descriptors/ACE/logs
source src/mlenv/env.sh

ACE_ARGS=(--r-cut "${R_CUT:-6.0}" --n-max "${N_MAX:-8}" --l-max "${L_MAX:-6}" --sigma "${SIGMA:-0.5}"
          --smoothing "${SMOOTHING:-0.5}" --density "${DENSITY:-gaussian}" --nu-max "${NU_MAX:-3}"
          --n3 "${N3:-4}" --l3 "${L3:-4}")
read -r -a RADII_LIST <<< "${RADII:-8}"
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == 1 ]]; then
    OVERWRITE_FLAG=(--overwrite)
fi
CODE=src/ACE/CoGe
WORKERS="${SLURM_CPUS_PER_TASK:-1}"
status=0

python -B -u "$CODE/bulk/bulk_descriptor.py" "${ACE_ARGS[@]}" --workers "$WORKERS" "${OVERWRITE_FLAG[@]}"
python -B -u "$CODE/bulk/check_bulk_invariance.py" "${ACE_ARGS[@]}" || status=1
python -B -u "$CODE/gb/gb_descriptor.py" "${ACE_ARGS[@]}" --radii "${RADII_LIST[@]}" \
    --workers "$WORKERS" "${OVERWRITE_FLAG[@]}" || status=1
python -B -u "$CODE/gb/check_gb.py" "${ACE_ARGS[@]}" || status=1
exit "$status"
