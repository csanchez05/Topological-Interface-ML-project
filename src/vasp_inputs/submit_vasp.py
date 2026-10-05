#!/usr/bin/env python3
"""Submit generated stage folders as one Slurm array of run_vasp_array.slurm (standard library only).

Reads a task list written by make_vasp_inputs.py --write-tasks and keeps only the folders that are
pending: no VASP output yet, and no claim in RECORDS/claims (a claim means the folder is running, or
failed earlier and its claim was not deleted). It writes the pending list to
RECORDS/submissions/<time>/tasks.txt, submits the array, and records the command and job id in
submit.json there. Slurm logs go to RECORDS/logs/<array job>_<task>.out.

Example (convergence sweep, 2 test points first):
  python3 -B src/vasp_inputs/submit_vasp.py \\
      --tasks results/CoGe_interface/100/convergence_8x/tasks.txt \\
      --only sym00431/8x/convergence/e500_k1x9x9 sym00177/8x/convergence/e500_k1x9x9 \\
      --mem 96G --dry-run
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402
import vasp_task  # noqa: E402

PROJECT_ROOT = mvi.PROJECT_ROOT
SLURM_SCRIPT = Path(__file__).resolve().parent / "run_vasp_array.slurm"


def pending(folders, records):
    """Split folders into (pending, skipped[(folder, reason)])."""
    root = vasp_task.PROJECT_ROOT
    todo, skipped = [], []
    for rel in folders:
        folder = root / rel
        if not (folder / "INCAR").is_file():
            skipped.append((rel, "no INCAR (not generated)"))
            continue
        names = {p.name for p in folder.iterdir()}
        if names & mvi.VASP_OUTPUT:
            skipped.append((rel, "holds VASP output" + (" (done)" if "result.json" in names else "")))
            continue
        info = vasp_task.identify(folder)
        claim = vasp_task.claim_path(records, info)
        if claim.exists():
            skipped.append((rel, f"claimed ({claim.relative_to(root)})"))
            continue
        todo.append(rel)
    return todo, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True, help="task list from make_vasp_inputs.py")
    parser.add_argument("--records", type=Path, help="records folder (default: the task list's folder)")
    parser.add_argument("--only", nargs="+", help="keep folders whose path contains one of these strings")
    parser.add_argument("--mem", default="96G", help="memory per array task (default 96G)")
    parser.add_argument("--nodelist", help="e.g. qcg3-0 (default: any node of the partition)")
    parser.add_argument("--throttle", type=int, default=8, help="array tasks running at once (default 8)")
    parser.add_argument("--time", default="2-00:00:00", help="time limit per task (default 2 days)")
    parser.add_argument("--dry-run", action="store_true", help="show the pending list and command only")
    args = parser.parse_args(argv)

    tasks = args.tasks if args.tasks.is_absolute() else PROJECT_ROOT / args.tasks
    records = args.records or tasks.parent
    records = records if records.is_absolute() else PROJECT_ROOT / records
    folders = [line.strip() for line in tasks.read_text().splitlines() if line.strip()]
    if args.only:
        folders = [f for f in folders if any(key in f for key in args.only)]
    todo, skipped = pending(folders, records)
    for rel, why in skipped:
        print(f"  skip {rel}: {why}")
    if not todo:
        sys.exit("nothing pending")
    ranks = set()
    for rel in todo:
        info = vasp_task.identify(PROJECT_ROOT / rel)
        _, mesh = info["recipe"]["points"][info["point"]]
        ranks.add(mvi.check_profile(info["recipe"], mesh)[0])
    if len(ranks) != 1:
        sys.exit(f"STOP: the folders need different rank counts {sorted(ranks)}; submit them separately")
    n_ranks = ranks.pop()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_dir = records / "submissions" / stamp
    command = ["sbatch", "--parsable", f"--array=0-{len(todo) - 1}%{args.throttle}", f"--mem={args.mem}",
               f"--ntasks={n_ranks}", f"--ntasks-per-node={n_ranks}", f"--time={args.time}",
               f"--output={records / 'logs'}/%A_%a.out"]
    if args.nodelist:
        command.append(f"--nodelist={args.nodelist}")
    command += [str(SLURM_SCRIPT), str(sub_dir / "tasks.txt"), str(records)]
    print(f"{len(todo)} pending folder(s), {n_ranks} ranks each, {args.mem}, throttle {args.throttle}:")
    for rel in todo:
        print("  " + rel)
    print(" ".join(command))
    if args.dry_run:
        return
    sub_dir.mkdir(parents=True)
    (records / "logs").mkdir(parents=True, exist_ok=True)
    (sub_dir / "tasks.txt").write_text("".join(rel + "\n" for rel in todo))
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    job = result.stdout.strip().split(";")[0]
    record = {"command": command, "returncode": result.returncode, "stdout": result.stdout,
              "stderr": result.stderr, "job_id": job if re.fullmatch(r"\d+", job) else None,
              "tasks": todo, "submitted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "user": os.environ.get("USER")}
    (sub_dir / "submit.json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode != 0:
        sys.exit(f"sbatch failed: {result.stderr.strip()}")
    print(f"submitted array job {job}; records in {sub_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
