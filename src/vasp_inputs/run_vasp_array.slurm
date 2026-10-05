#!/usr/bin/env bash
#SBATCH --job-name=coge-gb-vasp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --exclude=qcg-docker
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/calvinsanchez/ML_Interface_Project
# One array task = one line of TASKS (a generated stage folder). Submit with submit_vasp.py,
# which sets --array, --mem, --ntasks, --output and passes: TASKS RECORDS.
set -euo pipefail
tasks="${1:?task list}"
records="${2:?records folder}"
py=/usr/bin/python3
folder="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$tasks")"
[[ -n "$folder" ]] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $tasks"; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID}: $folder on $(hostname), job ${SLURM_JOB_ID}, ${SLURM_NTASKS} ranks, $(date -Is)"

scratch="$($py -B src/vasp_inputs/vasp_task.py prepare "$folder" --records "$records")"
binary="$($py -B src/vasp_inputs/vasp_task.py binary "$folder")"
echo "scratch $scratch, binary $binary"

# Job environment, verbatim from the campaign (CLAUDE.md section 2)
set +u
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge; module load vasp/vasp.6.4.3-intel2024
set -u
[[ "$(readlink -f -- "$(command -v "$binary")")" == "/mnt/apps/QCGCodes/codes/VASP/vasp.6.4.3/bin/$binary" ]]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_STACKSIZE=512M OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=core I_MPI_PIN_RESPECT_CPUSET=1 I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=1
ulimit -s unlimited

$py -B src/vasp_inputs/vasp_task.py sample "$scratch" &
sampler=$!
N="$SLURM_NTASKS"
status=0
(cd "$scratch" && srun --mpi=pmi2 --kill-on-bad-exit=1 --wait=120 --ntasks=$N --ntasks-per-node=$N --cpus-per-task=1 --cpu-bind=cores "$binary" > vasp.out 2>&1) || status=$?
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
echo "VASP exit $status, $(date -Is)"
$py -B src/vasp_inputs/vasp_task.py finish "$folder" "$scratch" --exit-code "$status" --records "$records"
