#!/usr/bin/env bash
#SBATCH --job-name=coge-rcut-all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=qcg0
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
#SBATCH --output=/home/calvinsanchez/ML_Interface_Project/calculations/ML_dscribe/CoGe/standard/rcut_scan_all/logs/driver-%j.out
#
# r_cut scan on every structure (4x-12x), one whole node. rcut_scan_all.py keeps all
# CPUs of the allocation busy with the SOAP runs and the separate analyses.
# The job can be resubmitted at any time: finished analyses are skipped.
#
# Memory: a pooled analysis runs --pooled-cpus processes (default 6) of about 8 GB each,
# and all 11 pooled analyses run together: about 530 GB.
#
# On qcg3-0 (256 CPUs):
#   sbatch --nodelist=qcg3-0 --cpus-per-task=256 src/ML_dscribe_scripts/rcut_scan_all.slurm --pooled-cpus 8
# Options for the driver go after the script name, e.g. "--r-cuts 3 7" or "--force".
# Afterwards: python -B src/ML_dscribe_scripts/rcut_scan_summary.py
set -euo pipefail
export OMP_NUM_THREADS=1
source src/mlenv/env.sh
python -B -u src/ML_dscribe_scripts/rcut_scan_all.py --cpus "${SLURM_CPUS_PER_TASK}" "$@"
