#!/usr/bin/env bash
#SBATCH --job-name=coge-representation
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=qcg3-0
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/descriptors/dscribe/logs/representation-%j.out
#
# Fifth and sixth studies (SOAP guide Stage 1, notes in results/dscribe/descriptor_study.md), one node,
# in two jobs with a stop between them:
#   part A: geometry labels, Stage 1.4 KRR rerun without D0985, selection benchmark v2, covariant
#           SOAP with lambda = 2 (all 13,868 structures) + check against the lambda <= 1 run, the 10
#           new standard and 12 covariant settings of the sixth study, fifth study phase 1 (bulk test,
#           PCA, UMAP) and its figures.
#   part B (only after the phase 1 table has been looked at): fifth study phase 2 (every other
#           method), the thickness check, the sixth study, and their figures.
# descriptor_grid.py --study representation keeps every CPU of the allocation busy within its memory;
# finished tasks are skipped, so a job can simply be submitted again. Task logs and status:
# calculations/descriptors/dscribe/logs/representation/part<A|B>/.
#
#   sbatch src/ML_dscribe_scripts/representation_study.slurm --part A
#   sbatch src/ML_dscribe_scripts/representation_study.slurm --part B
#   qcg0 instead:  sbatch --nodelist=qcg0 --cpus-per-task=96 --mem=700G src/ML_dscribe_scripts/representation_study.slurm --part A
# Check first (login node, nothing runs):
#   source src/mlenv/env.sh; python -B src/ML_dscribe_scripts/descriptor_grid.py --study representation --part A --dry-run --cpus 56 --mem-gb 360
#
# The defaults above fit the 56 CPUs that qcg3-0 had free on 2026-10-05; more CPUs only make it faster.
# Cost estimates are in src/ML_dscribe_scripts/README.md (representation_study.py).
set -euo pipefail
export OMP_NUM_THREADS=1   # before env.sh, which copies it to the BLAS and numba variables
source src/mlenv/env.sh
# Node-local numba cache, seeded from the shared one: with the cache on NFS (env.sh), dozens of UMAP,
# pynndescent and PaCMAP workers rewrite the same cache files at once and wait on NFS for minutes
# (seen in the smoke test, 2026-10-05).
shared_cache="$NUMBA_CACHE_DIR"
export NUMBA_CACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/numba-${SLURM_JOB_ID:-local}-XXXX")"
cp -r "$shared_cache"/. "$NUMBA_CACHE_DIR"/ 2>/dev/null || true
python -B -u src/ML_dscribe_scripts/descriptor_grid.py --study representation --cpus "${SLURM_CPUS_PER_TASK}" "$@"
