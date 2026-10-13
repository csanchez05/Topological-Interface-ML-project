#!/usr/bin/env python3
"""r_cut scan on every structure (all thicknesses), run in parallel on one node.

For each cutoff this needs one SOAP run over the whole catalogue and then many
separate analyses: steps 1.5 and 1.4 for every thickness on its own, and once
more with all thicknesses pooled, each for two GB vectors (interface radius
r_cut + 2 A, and a fixed radius). This driver holds all of them as a task list
and keeps the node's CPUs busy: a task starts as soon as the tasks it depends
on are done and enough CPUs are free.

  descriptor   one per cutoff      bulk SOAP + check, GB SOAP on all thicknesses, checks
  pooled       cutoff x vector     pooled_duplicates.py, krr_classify.py --thickness all
  thickness    cutoff x vector x T find_duplicates.py, krr_classify.py --thickness T

A pooled analysis is itself split up: krr_classify.py --jobs runs its 40 folds
(4 data sets x 2 groupings x 5 folds) in separate one-thread processes, as many
at a time as the task has CPUs. A fold of the 13,868-structure data set needs
about 8 GB, so --pooled-cpus also sets the memory used: 11 pooled analyses x
6 processes x 8 GB = 530 GB at the default.

Finished tasks are recognised by their output files and skipped, so a run that
was cancelled or partly failed can simply be started again. A failed task does
not stop the others; tasks that depend on it are skipped and listed at the end.

Results: calculations/ML_dscribe/CoGe/rcut_scan_all/{by_thickness,pooled}/rc<r_cut>/<vector tag>/
Then:    python -B src/ML_dscribe_scripts/rcut_scan_summary.py
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import ml_common as mc
from ml_common import sc

SOAP_CODE = mc.SOAP_SCRIPTS
ML_CODE = Path(__file__).resolve().parent
ORDER = {"descriptor": 0, "pooled": 1, "thickness": 2}


@dataclass
class Task:
    name: str
    kind: str
    cpus: int
    commands: list            # [(argv, may_fail)]
    outputs: list             # files that exist once the task is done
    needs: list = field(default_factory=list)
    threads: int = 1          # BLAS threads given to each command
    mem_gb: float = 0.0       # memory the task needs; only counted when run() has a memory budget
    weight: float = 0.0       # larger starts first among tasks of one kind
    state: str = "waiting"    # waiting, done (already), running, ok, failed, skipped
    seconds: float = 0.0

    def finished(self):
        return all(Path(path).is_file() for path in self.outputs)


PYTHON = [sys.executable, "-B", "-u"]


def stored_radii(folder):
    path = Path(folder) / sc.INTERFACE_FILE
    if not path.is_file():
        return []
    with np.load(path) as data:
        return [float(r) for r in data["radii"]]


def soap_arguments(settings):
    """Command-line SOAP settings for the descriptor scripts."""
    argv = ["--r-cut", f"{settings['r_cut']:g}", "--n-max", str(settings["n_max"]),
            "--l-max", str(settings["l_max"]), "--sigma", f"{settings['sigma']:g}"]
    if settings.get("weighting"):
        argv += ["--weighting", json.dumps(settings["weighting"])]
    return argv


def descriptor_task(name, settings, radii, root, cpus, thicknesses, limit=0, weight=0.0):
    """Bulk SOAP + check, GB SOAP on every structure and the GB checks, for one setting.

    Returns (task, both_interfaces folder). A finished run that holds every radius
    is marked done; a run that lacks a radius is redone with its own radii kept.
    """
    tag = sc.soap_tag({"rbf": "gto", "weighting": None, **settings})
    global_dir, gb_dir = Path(root) / "global" / tag, Path(root) / "both_interfaces" / tag
    soap = soap_arguments(settings)
    have = stored_radii(gb_dir)
    complete = all(any(np.isclose(r, h) for h in have) for r in radii)
    compute = sorted(set(radii) | set(have))
    gb_command = [*PYTHON, str(SOAP_CODE / "gb" / "gb_descriptor.py"), *soap, "--radii", *[f"{r:g}" for r in compute],
                  "--workers", str(cpus), "--out-root", str(root), "--name", tag, "--overwrite",
                  "--thickness", *[str(t) for t in thicknesses]]
    if limit:
        gb_command += ["--limit", str(limit)]
    task = Task(
        name=name, kind="descriptor", cpus=cpus, weight=weight,
        commands=[([*PYTHON, str(SOAP_CODE / "bulk" / "bulk_descriptor.py"), *soap, "--out-dir", str(global_dir)], False),
                  ([*PYTHON, str(SOAP_CODE / "bulk" / "check_bulk_invariance.py"), *soap,
                    "--run-dir", str(global_dir)], False),
                  (gb_command, False),
                  ([*PYTHON, str(SOAP_CODE / "gb" / "check_gb.py"), "--run-dir", str(gb_dir)], True)],
        outputs=[gb_dir / sc.INTERFACE_FILE, gb_dir / "checks.json", global_dir / sc.GLOBAL_FILE])
    if complete and task.finished():
        task.state = "done"
    return task, gb_dir


def build_tasks(args):
    python = PYTHON
    scan = args.scan_dir
    tasks = []
    # krr_classify.py refuses to replace its outputs unless told to; --force tells it to.
    redo = ["--overwrite"] if args.force else []
    for r_cut in args.r_cuts:
        settings = {"r_cut": r_cut, "n_max": args.n_max, "l_max": args.l_max, "sigma": args.sigma}
        radii = sorted({r_cut + 2.0, args.fixed_radius})
        name = f"descriptor rc{r_cut:g}"
        descriptor, gb_dir = descriptor_task(name, settings, radii, args.descriptor_root, args.descriptor_cpus,
                                             args.thicknesses, args.limit, weight=r_cut)
        tasks.append(descriptor)

        for radius in radii:
            vector = ["--descriptor", str(gb_dir), "--radius", f"{radius:g}"]
            out = scan / "pooled" / f"rc{r_cut:g}" / mc.vector_tag("interface", radius, "both", "all")
            tasks.append(Task(
                name=f"pooled rc{r_cut:g} r{radius:g}", kind="pooled", cpus=args.pooled_cpus, threads=args.pooled_cpus,
                weight=r_cut, needs=[name],
                commands=[([*python, str(ML_CODE / "pooled_duplicates.py"), *vector, "--out-dir", str(out)], False),
                          ([*python, str(ML_CODE / "krr_classify.py"), *vector, "--thickness", "all",
                            "--jobs", str(args.pooled_cpus), "--out-dir", str(out), *redo], False)],
                outputs=[out / "pooled_summary.json", out / "krr_summary.csv"]))
            for thickness in args.thicknesses:
                out = scan / "by_thickness" / f"rc{r_cut:g}" / mc.vector_tag("interface", radius, "both", thickness)
                one = [*vector, "--thickness", str(thickness), "--out-dir", str(out)]
                tasks.append(Task(
                    name=f"thickness rc{r_cut:g} r{radius:g} t{thickness}", kind="thickness",
                    cpus=args.thickness_cpus, threads=args.thickness_cpus, weight=thickness, needs=[name],
                    commands=[([*python, str(ML_CODE / "find_duplicates.py"), *one], False),
                              ([*python, str(ML_CODE / "krr_classify.py"), *one, *redo], False)],
                    outputs=[out / "duplicates_summary.json", out / "krr_summary.csv"]))
    for task in tasks:
        if task.kind != "descriptor" and not args.force and task.finished():
            task.state = "done"
    return tasks


def execute(task, log_dir, events):
    """Run the commands of one task in order; report (task, succeeded) on the queue."""
    started = time.time()
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = str(task.threads)
    succeeded = True
    log_path = log_dir / (task.name.replace(" ", "_") + ".log")
    with open(log_path, "w") as log:
        for argv, may_fail in task.commands:
            log.write("$ " + " ".join(argv) + "\n")
            log.flush()
            status = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                    cwd=mc.PROJECT_ROOT).returncode
            log.write(f"[exit status {status}]\n")
            if status != 0 and not may_fail:
                succeeded = False
                break
    task.seconds = time.time() - started
    events.put((task, succeeded and task.finished()))


def run(tasks, total_cpus, log_dir, status_path, total_mem_gb=None, order=None):
    """Run tasks as CPUs (and, with total_mem_gb, memory) allow. order ranks task kinds."""
    by_name = {task.name: task for task in tasks}
    order = order or ORDER
    events, free, running = queue.Queue(), total_cpus, 0
    free_mem = float("inf") if total_mem_gb is None else total_mem_gb
    started = time.time()

    def report(task, text):
        print(f"[{time.time() - started:7.0f} s] {text:8s} {task.name}"
              + (f" ({task.seconds:.0f} s)" if task.seconds else ""), flush=True)

    def write_status():
        mc.write_json(status_path, {
            "total_cpus": total_cpus, "elapsed_seconds": round(time.time() - started),
            "counts": {state: sum(t.state == state for t in tasks)
                       for state in ("done", "ok", "running", "waiting", "failed", "skipped")},
            "tasks": [{"name": t.name, "state": t.state, "cpus": t.cpus, "seconds": round(t.seconds)} for t in tasks]})

    while True:
        for task in sorted((t for t in tasks if t.state == "waiting"), key=lambda t: (order[t.kind], -t.weight)):
            states = [by_name[name].state for name in task.needs]
            if any(state in ("failed", "skipped") for state in states):
                task.state = "skipped"
                report(task, "skipped")
            elif all(state in ("done", "ok") for state in states) and task.cpus <= free and task.mem_gb <= free_mem:
                task.state = "running"
                free -= task.cpus
                free_mem -= task.mem_gb
                running += 1
                report(task, "started")
                threading.Thread(target=execute, args=(task, log_dir, events), daemon=True).start()
        write_status()
        if running == 0:
            break
        task, succeeded = events.get()
        task.state = "ok" if succeeded else "failed"
        free += task.cpus
        free_mem += task.mem_gb
        running -= 1
        report(task, "finished" if succeeded else "FAILED")
    write_status()
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r-cuts", type=float, nargs="+", default=[4, 5, 6, 8, 10, 12])
    parser.add_argument("--thicknesses", type=int, nargs="+", default=list(range(4, 13)))
    parser.add_argument("--fixed-radius", type=float, default=8.0,
                        help="second GB vector: this interface radius at every cutoff (default 8)")
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                        help="CPUs to keep busy (default: $SLURM_CPUS_PER_TASK, else all CPUs of this machine)")
    parser.add_argument("--descriptor-cpus", type=int, default=16, help="worker processes per SOAP run (default 16)")
    parser.add_argument("--pooled-cpus", type=int, default=6,
                        help="processes per pooled analysis, about 8 GB each (default 6)")
    parser.add_argument("--thickness-cpus", type=int, default=1, help="threads per one-thickness analysis (default 1)")
    parser.add_argument("--scan-dir", type=Path, default=mc.OUTPUT_ROOT / "rcut_scan_all")
    parser.add_argument("--descriptor-root", type=Path, default=sc.STANDARD_ROOT,
                        help="folder holding global/ and both_interfaces/ "
                             "(default calculations/descriptors/dscribe/standard)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (testing)")
    parser.add_argument("--force", action="store_true", help="redo analyses whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="print the task list and stop")
    args = parser.parse_args()
    for name in ("descriptor_cpus", "pooled_cpus", "thickness_cpus"):
        setattr(args, name, max(1, min(getattr(args, name), args.cpus)))

    tasks = build_tasks(args)
    todo = [task for task in tasks if task.state == "waiting"]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {args.cpus} CPUs")
    for kind in ORDER:
        chosen = [task for task in tasks if task.kind == kind]
        print(f"  {kind:10s} {len(chosen):4d} tasks, {sum(t.state == 'waiting' for t in chosen):4d} to run, "
              f"{chosen[0].cpus if chosen else 0} CPUs each")
    if args.dry_run:
        for task in tasks:
            print(f"  {task.state:8s} {task.cpus:3d} CPUs  {task.name}")
        return 0

    log_dir = args.scan_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run(tasks, args.cpus, log_dir, args.scan_dir / "scan_status.json")
    bad = [task for task in tasks if task.state in ("failed", "skipped")]
    print(f"\n{sum(t.state == 'ok' for t in tasks)} tasks ran, {sum(t.state == 'done' for t in tasks)} were already "
          f"done, {len(bad)} failed or skipped")
    for task in bad:
        print(f"  {task.state:8s} {task.name}  (log: {log_dir / (task.name.replace(' ', '_') + '.log')})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
