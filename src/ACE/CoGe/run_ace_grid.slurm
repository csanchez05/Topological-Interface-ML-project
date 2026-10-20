#!/usr/bin/env bash
#SBATCH --job-name=coge-ace-grid
#SBATCH --partition=general
#SBATCH --nodelist=qcg0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/descriptors/ACE/logs/ace-grid-%j.out
#
# ACE descriptor grid (src/ACE/CoGe/ace_grid.py): covariant ACE for the representation study, the
# reference at 8x with radii 6/8/10, and 27 one-knob settings at 8x. PARALLEL tasks (default 3) run at once,
# each with SLURM_CPUS_PER_TASK / PARALLEL workers, so one task's Python start-up (minutes on a busy node:
# about 840 modules over NFS) overlaps the others' compute.
# Cost estimate (2026-10-13 smoke timings on qcg0): invariant ACE 3-10 CPU-ms per atom, 1.4-3.5 CPU-h per
# 8x run (507k atoms); covariant ACE 18 CPU-ms per atom on 1.3 M atoms, about 6.5 CPU-h. In all about
# 70 CPU-h plus start-up, about 1.5-2 h on 96 cores or 3-4 h on 40. The covariant run holds about 27 GB in
# memory and on disk.
# Finished runs are skipped, so the job can simply be resubmitted; ONLY="covariant" runs a subset.
# Fewer cores when qcg0 is partly busy: sbatch --cpus-per-task=40 --mem=160G src/ACE/CoGe/run_ace_grid.slurm
set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p calculations/descriptors/ACE/logs
source src/mlenv/env.sh

ONLY_ARGS=()
if [[ -n "${ONLY:-}" ]]; then
    eval "ONLY_ARGS=(--only ${ONLY})"
fi
python -B -u src/ACE/CoGe/ace_grid.py --workers "${SLURM_CPUS_PER_TASK:-1}" --parallel "${PARALLEL:-3}" "${ONLY_ARGS[@]}"
