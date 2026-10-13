#!/usr/bin/env bash
#SBATCH --job-name=coge-gb-trial
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --exclude=qcg-docker
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
# One array task = one SCF trial (scf_trials.py TRIALS[index]): one or two VASP runs in one scratch.
# Submit with scf_trials.py submit, which sets --array, --mem, --output and passes: RECORDS.
set -euo pipefail
records="${1:?records folder}"
py=/usr/bin/python3
tool=src/vasp_inputs/scf_trials.py
echo "trial ${SLURM_ARRAY_TASK_ID} on $(hostname), job ${SLURM_JOB_ID}, ${SLURM_NTASKS} ranks, $(date -Is)"
scratch="$($py -B $tool prepare "$SLURM_ARRAY_TASK_ID" --records "$records")"
echo "scratch $scratch"

# Job environment, verbatim from the campaign (CLAUDE.md section 2)
set +u
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge; module load vasp/vasp.6.4.3-intel2024
set -u
[[ "$(readlink -f -- "$(command -v vasp_std)")" == "/mnt/apps/QCGCodes/codes/VASP/vasp.6.4.3/bin/vasp_std" ]]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_STACKSIZE=512M OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=core I_MPI_PIN_RESPECT_CPUSET=1 I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=1
ulimit -s unlimited

N="$SLURM_NTASKS"
for run in $($py -B $tool runs "$scratch"); do
    $py -B $tool stage "$scratch" "$run" || { echo "not starting $run"; break; }
    $py -B src/vasp_inputs/vasp_task.py sample "$scratch/$run" &
    sampler=$!
    status=0
    (cd "$scratch/$run" && srun --mpi=pmi2 --kill-on-bad-exit=1 --wait=120 --ntasks=$N --ntasks-per-node=$N --cpus-per-task=1 --cpu-bind=cores vasp_std > vasp.out 2>&1) || status=$?
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    echo "$run: VASP exit $status, $(date -Is)"
    [[ $status -eq 0 ]] || break
done
$py -B $tool summarize "$scratch" --records "$records"
echo "scratch kept at $scratch on $(hostname)"
