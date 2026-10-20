#!/usr/bin/env bash
#SBATCH --job-name=coge-ace-screens
#SBATCH --partition=general
#SBATCH --nodelist=qcg0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/ML_ACE/CoGe/logs/ace-screens-%j.out
#
# The descriptor-study screens for ACE (SOAP guide Stage 1, as the SOAP screens of descriptor_grid.py):
# descriptor_sensitivity.py, then selection_benchmark.py, then plot_descriptor_study.py, all with
# --descriptor-type ace. Needs every run of ace_grid.py's settings grid (run_ace_grid.slurm) and the ACE KRR
# run below. Outputs in calculations/ML_ACE/CoGe/{descriptor_sensitivity,selection_benchmark}/, figures in
# plots/CoGe/dim-reduction/ACE/. A step whose summary.json exists is skipped, so the job can simply be
# resubmitted; the plots are always redrawn.
# Cost estimate (smoke of 2026-10-13 on qcg2-2, 2 workers, archive/ace_20261013/smoke/screens/): about 55 CPU-s
# per setting and 65 CPU-s per t-SNE/UMAP map, so the 41 settings and about 100 maps take under 3 CPU-h; the
# benchmark (41 settings x 10 test sets x 20 random draws) a few CPU-h more. About 1 h on 40 cores.
# Memory: the reference vectors (all thicknesses) are 1.5 GB; the rest are 8x only (0.2 GB each).
# Whole node when it is free: sbatch --cpus-per-task=96 --mem=400G src/ACE/CoGe/run_ace_screens.slurm
set -euo pipefail
export OMP_NUM_THREADS=1   # before env.sh, which copies it to the BLAS and numba variables
mkdir -p calculations/ML_ACE/CoGe/logs
source src/mlenv/env.sh

JOBS=${SLURM_CPUS_PER_TASK:-1}
ML=src/ML_dscribe_scripts
OUT=calculations/ML_ACE/CoGe
KRR=$OUT/rc6.0_n8_l6_s0.5_nu3-n4-l4/interface_r8_both_t4_no_boundary

step() {  # step <name> <summary.json> <command...>
    local name=$1 done=$2
    shift 2
    if [[ -n "$done" && -f "$done" ]]; then
        echo "== $(date '+%F %T') $name: skipped ($done exists)"
        return
    fi
    echo "== $(date '+%F %T') $name"
    "$@"
}

step sensitivity "$OUT/descriptor_sensitivity/summary.json" \
    python -B -u $ML/descriptor_sensitivity.py --descriptor-type ace --jobs "$JOBS"
step benchmark "$OUT/selection_benchmark/summary.json" \
    python -B -u $ML/selection_benchmark.py --descriptor-type ace --jobs "$JOBS" --krr-settings "$KRR"
step plots "" \
    python -B -u $ML/plot_descriptor_study.py --descriptor-type ace
echo "== $(date '+%F %T') done"
