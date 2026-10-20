#!/usr/bin/env bash
#SBATCH --job-name=coge-cov-vs-labels
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --nodelist=qcg0
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/ML_dscribe/CoGe/comparison/covariant_vs_labels/logs/covariant-vs-labels-%j.out
#
# Covariant SOAP vs residual-symmetry labels (SOAP guide Stage 1 follow-up, preparing Stages 2-3):
#   step 1  gb_symmetry.py: spglib on the 1,541 GB cells at 8x (plus a 4x/12x sample), the label table
#           calculations/ML_dscribe/CoGe/labels/gb_symmetry.csv; kept if it already exists and passed
#   step 2  covariant_vs_labels.py: the six inputs on the 3,078 interface rows (edge, mid of 1,539
#           families), the 4 geometry labels, 5 outer x 3 inner folds, into
#           calculations/ML_dscribe/CoGe/comparison/covariant_vs_labels/interface_geometry_labels/
# A failed check in step 1 stops the job before step 2 (set -e).
# On qcg0 (the user's choice for submissions; qcg3-0 runs GB VASP), 32 of its 96 CPUs: step 1 uses one
# worker per CPU, step 2 uses them as BLAS threads. More (sbatch -c 88 ...) only shortens step 1.
# Smoke test (2026-10-13, 8 CPUs): peak 3.7 GB (whole npz members are read); full run estimated at
# 5-6 GB, step 1 about 15 min on 8 workers, step 2 under 30 min (src/ML_dscribe_scripts/README.md).
#
#   sbatch src/ML_dscribe_scripts/covariant_vs_labels.slurm
#   other targets:  sbatch src/ML_dscribe_scripts/covariant_vs_labels.slurm --rows cell --targets-table <csv> --targets <col>
# Check first (login node, nothing runs):
#   src/dft_plots/.venv/bin/python -B src/ML_dscribe_scripts/gb_symmetry.py --dry-run
#   source src/mlenv/env.sh; python -B src/ML_dscribe_scripts/covariant_vs_labels.py --dry-run
set -euo pipefail
cpus="${SLURM_CPUS_PER_TASK:-32}"
root="$PWD"
work="$(mktemp -d "${TMPDIR:-/tmp}/cov-vs-labels-XXXX")"   # nothing may land in the project root

table=calculations/ML_dscribe/CoGe/labels/gb_symmetry.json
if [[ -f "$table" ]]; then
    status="$(src/dft_plots/.venv/bin/python -c "import json, sys; print(json.load(open(sys.argv[1]))['status'])" "$table")"
    echo "step 1: $table exists (status $status), kept"
    [[ "$status" == "pass" ]] || { echo "step 1 failed before; look at $table"; exit 1; }
else
    (cd "$work" && "$root/src/dft_plots/.venv/bin/python" -B -u "$root/src/ML_dscribe_scripts/gb_symmetry.py" --workers "$cpus")
fi

export OMP_NUM_THREADS="$cpus"   # before env.sh, which copies it to the BLAS variables
source src/mlenv/env.sh
(cd "$work" && python -B -u "$root/src/ML_dscribe_scripts/covariant_vs_labels.py" "$@")
rmdir "$work" 2>/dev/null || true
