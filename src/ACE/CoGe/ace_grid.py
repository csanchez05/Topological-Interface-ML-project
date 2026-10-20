#!/usr/bin/env python3
"""ACE descriptor grid: every ACE run the descriptor screens need, as one whole-node job.

The ACE counterpart of the descriptor runs of src/ML_dscribe_scripts/descriptor_grid.py. Each setting
changes one knob away from the reference ACE run (rc6.0_n8_l6_s0.5_nu3-n4-l4, which holds all 13,868
structures, job 12969):

  r_cut      4 5 8 10 12 A            (both expansions)
  n_max      4 6 10 12                (nu1 and nu2 expansion)
  l_max      2 4 8 10                 (nu1 and nu2 expansion)
  sigma      0.3 0.75 1.0 A           (both expansions)
  n3         2 3 5 6                  (nu3 expansion)
  l3         2 3 5 6                  (nu3 expansion)
  weighting  pow at r_cut 6, 8, 12    (w = 1 / (1 + (r / 3 A)^4), as the SOAP study; featomic has no poly)

Tasks (--parallel K runs K at a time, each with --workers / K workers; K = 1 runs them one after the other):
  covariant      covariant ACE (lambda = 1, 2) at the reference setting for the representation study:
                 every GB at 8x plus the 300 GBs of its thickness check at all 9 thicknesses
                 (structure list written first), then check_covariant.py
  region radii   the reference setting at 8x with interface radii 6, 8 and 10 A (<tag>_t8), for the
                 region r6 / r10 vectors; bulk ACE + bulk invariance check
  <knob> <value> one per setting above at 8x (<tag>_t8): bulk ACE + bulk invariance check, GB ACE
Every step is a new Python process that imports about 840 modules over NFS; on a busy node that takes
minutes (qcg0, 2026-10-13: about 0.2 s per module), so with K > 1 one task's start-up overlaps the others'
compute.
Finished runs are recognised by their output files and skipped, so the job can simply be submitted
again; the checks always rerun. A failed task does not stop the others; the job then exits 1.
Task logs: calculations/descriptors/ACE/logs/grid/<task>.log; one line per task in status.tsv there.

Smoke test (a few structures into a scratch folder):
  python -B src/ACE/CoGe/ace_grid.py --limit 1 --standard-root <scratch>/standard \\
      --covariant-root <scratch>/covariant --log-dir <scratch>/logs --only "n3 2" covariant
"""
import argparse
import csv
import hashlib
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "covariant"))

import numpy as np

import ace_common as ac
import covariant_ace as cov

sc = ac.sc
ML_SCRIPTS = ac.PROJECT_ROOT / "src" / "ML_dscribe_scripts"
# The ACE flags of the reference run; every other run changes one or two of them.
REFERENCE = {"r_cut": 6.0, "n_max": 8, "l_max": 6, "sigma": 0.5, "smoothing": 0.5, "density": "gaussian",
             "nu_max": 3, "n3": 4, "l3": 4, "weighting": "none"}
FLAGS = {"r_cut": "--r-cut", "n_max": "--n-max", "l_max": "--l-max", "sigma": "--sigma", "smoothing": "--smoothing",
         "density": "--density", "nu_max": "--nu-max", "n3": "--n3", "l3": "--l3", "weighting": "--weighting"}
REGION_RADII = (6.0, 8.0, 10.0)
THICKNESS_GBS, THICKNESS_SEED = 300, 11
STRUCTURE_LIST = f"structure_lists/t8-thick{THICKNESS_GBS}.csv"   # under the covariant root
LOG_DIR = ac.LOG_DIR / "grid"


def ace_runs():
    """(knob, value, ACE flags) of every invariant ACE run of the screens; the reference first."""
    runs = [("reference", "reference", {})]
    runs += [("r_cut", f"{v:g}", {"r_cut": float(v)}) for v in (4, 5, 8, 10, 12)]
    runs += [("n_max", str(v), {"n_max": v}) for v in (4, 6, 10, 12)]
    runs += [("l_max", str(v), {"l_max": v}) for v in (2, 4, 8, 10)]
    runs += [("sigma", f"{v:g}", {"sigma": v}) for v in (0.3, 0.75, 1.0)]
    runs += [("n3", str(v), {"n3": v}) for v in (2, 3, 5, 6)]
    runs += [("l3", str(v), {"l3": v}) for v in (2, 3, 5, 6)]
    runs += [("weighting", f"pow rc{r}", {"r_cut": float(r), "weighting": "pow"}) for r in (6, 8, 12)]
    return [(knob, value, {**REFERENCE, **changes}) for knob, value, changes in runs]


def settings_of(flags):
    """ace_common settings dict of a set of ACE flags (the weighting at its defaults r0 3, m 4)."""
    return ac.ace_settings(argparse.Namespace(weighting_r0=3.0, weighting_m=4, **flags))


def run_name(flags):
    """Folder name of a run: the ACE tag for the reference, <tag>_t8 for the 8x runs."""
    tag = ac.ace_tag(settings_of(flags))
    return tag if flags == REFERENCE else f"{tag}_t8"


def cli(flags):
    out = []
    for key, flag in FLAGS.items():
        value = flags[key]
        out += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
    return out


# --------------------------------------------------------------------------- structure list

def thickness_subset(n_gbs=THICKNESS_GBS, seed=THICKNESS_SEED):
    """The GBs of representation_study.py's thickness check, (family, plane) rows.

    Same rule as representation_study.thickness_check(): the census GBs without D0985
    (representation_study.gb_table), then per plane round(n_gbs x share) of them with
    numpy default_rng(seed), planes in sorted order.
    """
    sys.path.insert(0, str(ML_SCRIPTS))
    import representation_study as rs

    gbs = rs.gb_table()
    rng = np.random.default_rng(seed)
    chosen = []
    for _, group in gbs.groupby("plane"):
        count = max(1, int(round(n_gbs * len(group) / len(gbs))))
        chosen.extend(rng.choice(group.index.to_numpy(), min(count, len(group)), replace=False).tolist())
    return gbs.loc[sorted(chosen), ["family", "plane"]].reset_index(drop=True)


def structure_list_rows(subset, families_at_8):
    """Every family at 8x, plus the subset at every thickness: rows of the covariant structure list."""
    in_subset = set(subset["family"])
    rows = [{"family": f, "thickness": 8, "thickness_check": int(f in in_subset)} for f in sorted(families_at_8)]
    rows += [{"family": f, "thickness": t, "thickness_check": 1}
             for f in sorted(in_subset) for t in range(4, 13) if t != 8]
    return sorted(rows, key=lambda r: (r["family"], r["thickness"]))


def write_structure_list(path, rows, dry_run=False):
    """Write the list; an identical file is kept, a different one stops the task."""
    fields = ["family", "thickness", "thickness_check"]
    text = "".join([",".join(fields) + "\n"] + [f"{r['family']},{r['thickness']},{r['thickness_check']}\n" for r in rows])
    if path.exists():
        if path.read_text() == text:
            return "identical"
        raise RuntimeError(f"{sc.relative_path(path)} exists with other contents; not replaced")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return "written"


def catalogue_families():
    """Every GB family with an 8x structure (sym folders of the catalogue)."""
    jobs, _ = ac.collect_structures(sc.GB_ROOT, list(sc.PLANES), [8], 0)
    return {f"{plane}/{sym}" for plane, sym, _, _ in jobs}


# --------------------------------------------------------------------------- tasks

class Task:
    def __init__(self, name, steps, done=None, info=""):
        self.name, self.steps, self.done, self.info = name, steps, done, info

    def finished(self):
        return self.done is not None and self.done.exists()


def descriptor_task(name, flags, standard_root, workers, limit, radii=(8.0,)):
    """Bulk ACE + bulk invariance check + GB ACE at 8x for one setting."""
    run = run_name(flags) if radii == (8.0,) else f"{ac.ace_tag(settings_of(flags))}_t8"
    global_dir, interface_dir = standard_root / "global" / run, standard_root / "both_interfaces" / run
    common = cli(flags)
    extra = ["--limit", str(limit)] if limit else []
    py = [sys.executable, "-B", "-u"]
    steps = [
        ("bulk", py + [str(HERE / "bulk" / "bulk_descriptor.py"), *common, "--out-dir", str(global_dir),
                       "--workers", str(workers), *extra], global_dir / ac.BULK_FILE),
        ("gb", py + [str(HERE / "gb" / "gb_descriptor.py"), *common, "--thickness", "8", "--radii",
                     *[f"{r:g}" for r in radii], "--out-root", str(standard_root), "--name", run,
                     "--workers", str(workers), *extra], interface_dir / ac.INTERFACE_FILE),
        ("bulk check", py + [str(HERE / "bulk" / "check_bulk_invariance.py"), *common, "--run-dir", str(global_dir)], None),
    ]
    return Task(name, steps, interface_dir / ac.INTERFACE_FILE, run)


def covariant_task(covariant_root, workers, limit):
    run = cov.run_name(settings_of(REFERENCE)) + (f"_lim{limit}" if limit else "")
    interface_dir = covariant_root / "both_interfaces" / run
    structure_list = covariant_root / STRUCTURE_LIST
    py = [sys.executable, "-B", "-u"]
    extra = ["--limit", str(limit)] if limit else []
    steps = [
        ("list", "structure list", structure_list),
        ("covariant", py + [str(HERE / "covariant" / "covariant_descriptor.py"), *cli(REFERENCE),
                            "--structures", str(structure_list), "--out-root", str(covariant_root), "--name", run,
                            "--workers", str(workers), "--chunk", "128", *extra], interface_dir / cov.FILE),
        ("check", py + [str(HERE / "covariant" / "check_covariant.py"), "--run-dir", str(interface_dir)], None),
    ]
    return Task("covariant", steps, interface_dir / cov.FILE, run)


def all_tasks(args):
    workers = max(1, args.workers // args.parallel)
    tasks = [covariant_task(args.covariant_root, workers, args.limit),
             descriptor_task("region radii", REFERENCE, args.standard_root, workers, args.limit, REGION_RADII)]
    for knob, value, flags in ace_runs()[1:]:
        tasks.append(descriptor_task(f"{knob} {value}", flags, args.standard_root, workers, args.limit))
    return tasks


def run_step(label, command, log):
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    started = time.time()
    status = subprocess.call(command, cwd=ac.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return status, time.time() - started


def run_task(task, log_dir, status_file, lock):
    """Run one task's steps in order; return its failures. Status lines are written under the lock."""
    failures = []
    log_path = log_dir / f"{task.name.replace(' ', '_')}.log"
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {task.name} ({task.info})\n")
        for label, command, output in task.steps:
            if output is not None and output.exists() and label != "list":
                status, seconds = "skipped (output present)", 0.0
            elif label == "list":
                try:
                    rows = structure_list_rows(thickness_subset(), catalogue_families())
                    status, seconds = write_structure_list(output, rows), 0.0
                    log.write(f"structure list {sc.relative_path(output)}: {status}, {len(rows)} rows, sha256 "
                              f"{hashlib.sha256(output.read_bytes()).hexdigest()}\n")
                except Exception as exc:  # recorded; the task's later steps are skipped
                    status, seconds = f"failed: {exc}", 0.0
            else:
                code, seconds = run_step(label, command, log)
                status = "ok" if code == 0 else f"exit {code}"
            with lock:
                status_file.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{task.name}\t{task.info}\t{label}\t"
                                  f"{status}\t{seconds:.0f}\t{sc.relative_path(log_path)}\n")
                status_file.flush()
                print(f"{task.name:14s} {label:10s} {status} ({seconds:.0f} s)", flush=True)
            if status.startswith(("exit", "failed")):
                failures.append(f"{task.name} / {label}: {status}")
                if label != "bulk check":   # a failed check leaves the run in place; others stop the task
                    break
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None,
                        help='task names to run, e.g. covariant "region radii" "n3 6" (default all)')
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="worker processes in all (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="tasks run at the same time; each gets --workers / --parallel workers (default 1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: only the first N sym folders per plane (needs scratch roots)")
    parser.add_argument("--standard-root", type=Path, default=ac.STANDARD_ROOT)
    parser.add_argument("--covariant-root", type=Path, default=cov.COVARIANT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--dry-run", action="store_true", help="list the tasks, their state and commands, and stop")
    args = parser.parse_args()

    if args.limit and (args.standard_root == ac.STANDARD_ROOT or args.covariant_root == cov.COVARIANT_ROOT):
        print("--limit writes partial runs: give scratch --standard-root and --covariant-root", file=sys.stderr)
        return 2
    if args.parallel < 1 or args.parallel > args.workers:
        print("--parallel must be between 1 and --workers", file=sys.stderr)
        return 2
    tasks = all_tasks(args)
    if args.only:
        unknown = set(args.only) - {t.name for t in tasks}
        if unknown:
            print(f"unknown tasks {sorted(unknown)}; known: {[t.name for t in tasks]}", file=sys.stderr)
            return 2
        tasks = [t for t in tasks if t.name in args.only]
    reference_run = ac.STANDARD_ROOT / "both_interfaces" / run_name(REFERENCE) / ac.INTERFACE_FILE
    print(f"reference run {sc.relative_path(reference_run)}: {'present' if reference_run.exists() else 'MISSING'}")
    print(f"{len(tasks)} tasks, {args.parallel} at a time, {max(1, args.workers // args.parallel)} workers each")
    for task in tasks:
        state = "done (checks only)" if task.finished() else "to run"
        print(f"{task.name:14s} {task.info:44s} {state}")
        if args.dry_run:
            for label, command, _ in task.steps:
                print(f"    {label}: {command if isinstance(command, str) else ' '.join(command[2:])}")
    if args.dry_run:
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "status.tsv"
    new_status = not status_path.exists()
    failures = []
    lock = threading.Lock()
    with open(status_path, "a") as status_file:
        if new_status:
            status_file.write("time\ttask\trun\tstep\tstatus\tseconds\tlog\n")
        # Tasks start in list order (covariant, the longest, first); each step is a subprocess.
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for task_failures in pool.map(lambda task: run_task(task, args.log_dir, status_file, lock), tasks):
                failures.extend(task_failures)
    print(f"{len(tasks)} tasks, {len(failures)} failures" + (":\n  " + "\n  ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
