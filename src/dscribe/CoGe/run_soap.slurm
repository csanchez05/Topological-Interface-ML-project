#!/usr/bin/env bash
#SBATCH --job-name=coge-soap
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/descriptors/dscribe/logs/soap-%j.out
#
# Stage 1, steps 1.1-1.3: bulk SOAP, bulk invariance check, GB SOAP, GB checks.
# Settings are read from the environment, so a second run is one line:
#   R_CUT=8 N_MAX=10 L_MAX=8 RADII="4 6 8 10 12" sbatch src/dscribe/CoGe/run_soap.slurm
# OVERWRITE=1 replaces an existing run with the same SOAP settings.
set -euo pipefail
export OMP_NUM_THREADS=1
source src/mlenv/env.sh

SOAP_ARGS=(--r-cut "${R_CUT:-6.0}" --n-max "${N_MAX:-8}" --l-max "${L_MAX:-6}" --sigma "${SIGMA:-0.5}")
read -r -a RADII_LIST <<< "${RADII:-4 6 8 10}"
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == 1 ]]; then
    OVERWRITE_FLAG=(--overwrite)
fi
CODE=src/dscribe/CoGe

python -B -u "$CODE/bulk/bulk_descriptor.py" "${SOAP_ARGS[@]}" "${OVERWRITE_FLAG[@]}"
python -B -u "$CODE/bulk/check_bulk_invariance.py" "${SOAP_ARGS[@]}"
python -B -u "$CODE/gb/gb_descriptor.py" "${SOAP_ARGS[@]}" --radii "${RADII_LIST[@]}" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" "${OVERWRITE_FLAG[@]}"
python -B -u "$CODE/gb/check_gb.py" "${SOAP_ARGS[@]}"
