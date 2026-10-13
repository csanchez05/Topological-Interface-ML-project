#!/usr/bin/env bash
#SBATCH --job-name=coge-descriptor-grid
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=qcg3-0
#SBATCH --cpus-per-task=256
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=qcg-docker
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/descriptors/dscribe/logs/grid-%j.out
#
# Descriptor study (SOAP guide Stage 1, before the Stage 2 picks), one whole node:
# 17 SOAP runs on all 13,868 GB structures, the covariant run, then descriptor_sensitivity.py
# and selection_benchmark.py, and last the figures in plots/CoGe/dim-reduction/
# (plot_descriptor_study.py). descriptor_grid.py keeps every CPU of the allocation busy and
# stays inside the memory of the allocation. Resubmit at any time: finished tasks are skipped.
#
# Cost (smoke tests of 2026-10-03, archive/descriptor_study_20261003/smoke/): per SOAP run a serial
# bulk step of 3-8 min, then about 15 min for the 13,868 GB structures on 16 workers; covariant
# about 38 CPU-ms per atom, 48 CPU-h for the 4.56 million atoms; both analyses together under
# 30 min. In all about 200 CPU-h: about 1.5 h on qcg3-0, 2-2.5 h on qcg0. Disk about 22 GB.
# Memory seen: 1.1 GB per sensitivity worker, 0.3 GB per covariant worker; the defaults
# (16 GB per SOAP run, 64 GB covariant, 64 GB per analysis) leave headroom.
#
# Other nodes (options on the command line override the lines above):
#   qcg0, whole node:     sbatch --nodelist=qcg0 --cpus-per-task=96 src/ML_dscribe_scripts/descriptor_grid.slurm
# A partial qcg3-0 (e.g. --cpus-per-task=120 --mem=48G) is limited by memory: at 16 GB per SOAP
# run only two run at a time.
# Options for the driver go after the script name, e.g. --only reference "n_max 12", --no-covariant, --force.
# Check first: python -B src/ML_dscribe_scripts/descriptor_grid.py --dry-run --cpus 256 --mem-gb 630
set -euo pipefail
export OMP_NUM_THREADS=1   # before env.sh, which copies it to the BLAS and numba variables
source src/mlenv/env.sh
python -B -u src/ML_dscribe_scripts/descriptor_grid.py --cpus "${SLURM_CPUS_PER_TASK}" "$@"
