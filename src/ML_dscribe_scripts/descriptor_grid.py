#!/usr/bin/env python3
"""Descriptor study: every SOAP run and both analyses as one whole-node job.

Tasks, started as CPUs and memory allow (a task waits for the tasks it needs):
  covariant          covariant SOAP at the reference setting + check_covariant.py (part B)
  descriptor <name>  one per setting of descriptor_sensitivity.soap_runs(): bulk SOAP +
                     invariance check, GB SOAP on all 13,868 structures (interface radii
                     8 A and r_cut + 2), check_gb.py. Settings already stored are skipped.
  sensitivity        descriptor_sensitivity.py (part C), once every descriptor task is done
  benchmark          selection_benchmark.py (part D), once every descriptor task is done
  plots              plot_descriptor_study.py: the figures in plots/CoGe/dim-reduction/SOAP/standard/, once
                     both analyses and the covariant run are done

Finished tasks are recognised by their output files and skipped, so the job can simply be
submitted again. A failed task does not stop the others; tasks that need it are skipped
and listed at the end. Task logs and status: calculations/descriptors/dscribe/logs/grid/.

Smoke test (a few structures into a scratch folder, no analyses):
  python -B src/ML_dscribe_scripts/descriptor_grid.py --limit 2 --no-analyses \\
      --descriptor-root <scratch>/standard --covariant-root <scratch>/covariant --only "n_max 12"

--study representation: the fifth and sixth studies (representation_study.py) in two jobs.
  --part A  labels         geometry_labels.py (free labels for Stage 1.4 and the studies)
            krr            krr_classify.py --regression, Stage 1.4 rerun without D0985, into
                           <tag>/interface_r8_both_t4_no_boundary/
            benchmark      selection_benchmark.py --no-descriptors into selection_benchmark_v2/
            covariant      covariant SOAP with lambda = 2 at the reference setting, all 13,868
                           structures, named <tag>_lam2; check_covariant.py --compare-with <tag> must
                           pass (it reproduces the existing lambda <= 1 vectors)
            grid <tag>     the 12 standard SOAP settings of the sixth study (existing ones are skipped)
            covariant grid <tag>  lambda >= 1 covariant SOAP at 8x for each setting, named <tag>_t8
            phase1         representation_study.py --phase 1 (bulk test, PCA and UMAP)
            plots          plot_descriptor_study.py: selection/, free_labels/ (SOAP/standard/),
                           representation_response/ (SOAP/comparison/)
  --part B  phase2, thickness, settings_grid (representation_study.py) and their figures.
  The reference covariant run is <tag>_lam2 while that folder exists, else <tag> (after the rename).
  --limit N passes to every descriptor run and to geometry_labels.py, and makes the studies use only
  those N sym folders per plane; krr and benchmark need the full catalogue and are left out.

--study covariant: the standard-SOAP analyses repeated on covariant SOAP ([C_edge | C_mid] of the
lambda >= 1 blocks), outputs in calculations/ML_dscribe/CoGe/covariant/ and figures in
plots/CoGe/dim-reduction/SOAP/covariant/ (one job).
  covariant reference      the reference run (all 13,868 structures, every block); only checked if present
  covariant <knob> <value> one per setting of descriptor_sensitivity.covariant_runs(): lambda >= 1 at 8x,
                           named <tag>_t8; check_covariant.py must pass. Runs already stored are only checked.
  duplicates               find_duplicates.py --structure-census --single-interfaces (Stage 1.5) into
                           covariant/<tag>/interface_r8_both_t4/
  krr                      krr_classify.py --regression --census-grouping (Stage 1.4, D0985 dropped) into
                           covariant/<tag>/interface_r8_both_t4_no_boundary/ (as for standard SOAP);
                           "census" folds keep turned copies of one GB together
  select, select plane100  select_gbs.py (Stage 2 lists), and with --planes 100 --min-pair 1.8
  scan rc<r>               the 8x r_cut scan (r_cut 4 5 6 8 10 12, interface radius 8 A, D0985 kept as in
                           the standard 8x scan): find_duplicates.py + krr_classify.py --thickness 8
                           --census-grouping into
                           covariant/rcut_scan_8x/rc<r>/interface_r8_both_t8/
  scan summary             rcut_scan_summary.py --scan-dir covariant/rcut_scan_8x
  sensitivity, benchmark   descriptor_sensitivity.py / selection_benchmark.py --descriptor-type covariant
                           (the benchmark takes its KRR settings from the krr task)
  plots                    plot_descriptor_study.py --descriptor-type covariant, then the standard stage1/
                           figures and the grain-class overview in SOAP/comparison/
  --limit N (smoke test) needs --no-analyses and a scratch --covariant-root.
"""
import argparse
import os
import sys
from pathlib import Path

import descriptor_sensitivity as ds
import geometry_labels as gl
import ml_common as mc
import rcut_scan_all as rs
from ml_common import sc

ORDER = {"labels": 0, "covariant": 1, "descriptor": 2, "analysis": 3, "study": 4, "plots": 5}
LOG_DIR = sc.LOG_DIR / "grid"
REPRESENTATION_LOG_DIR = sc.LOG_DIR / "representation"
COVARIANT_LOG_DIR = sc.LOG_DIR / "covariant_study"
SCAN_R_CUTS = (4.0, 5.0, 6.0, 8.0, 10.0, 12.0)  # the r_cut values of the standard 8x scan (rcut_scan_8x/)
GRID = {"r_cut": (5.0, 6.0, 8.0), "l_max": (4, 8), "sigma": (0.3, 0.5)}  # = representation_study.GRID
VECTORIAL = ("polar", "axial", "tensor", "pseudotensor")
# plot_descriptor_study.py sections drawn for covariant SOAP (its COVARIANT_TYPE_SECTIONS)
COVARIANT_PLOT_SECTIONS = ("descriptor_settings", "maps_by_setting", "seed_stability", "thickness", "selection",
                           "vector_sizes", "free_labels", "stage1")
# Figures that the representation study redraws (selection/ from benchmark v2) are kept here first.
ARCHIVE_OLD = sc.PROJECT_ROOT / "archive" / "representation_study_20261005"


def soap_weight(settings):
    """Rough relative cost of one SOAP run, so the longest runs start first."""
    return settings["r_cut"] ** 3 * settings["n_max"] ** 2 * (settings["l_max"] + 1)


def build_tasks(args):
    tasks = []
    runs = [] if args.no_descriptors else ds.soap_runs()
    if args.only:
        known = {v.name for v in ds.variants()}
        if set(args.only) - known:
            raise SystemExit(f"unknown settings {sorted(set(args.only) - known)}; known: {sorted(known)}")
        runs = [run for run in runs if ("reference" if run[0] == "reference" else f"{run[0]} {run[1]}") in args.only]
    for knob, value, settings in runs:
        name = "descriptor reference" if knob == "reference" else f"descriptor {knob} {value}"
        radii = sorted({8.0, settings["r_cut"] + 2.0})
        task, _ = rs.descriptor_task(name, settings, radii, args.descriptor_root, args.descriptor_cpus,
                                     args.thicknesses, args.limit, weight=soap_weight(settings))
        task.mem_gb = args.descriptor_mem
        tasks.append(task)

    if not args.no_covariant:
        tag = ds.soap_tag(ds.REFERENCE)
        folder = Path(args.covariant_root) / "both_interfaces" / tag
        command = [*rs.PYTHON, str(rs.SOAP_CODE / "covariant" / "covariant_descriptor.py"),
                   "--workers", str(args.covariant_cpus), "--out-root", str(args.covariant_root), "--name", tag,
                   "--radii", "8", "--thickness", *[str(t) for t in args.thicknesses], "--overwrite"]
        if args.limit:
            command += ["--limit", str(args.limit)]
        task = rs.Task(name="covariant", kind="covariant", cpus=args.covariant_cpus, mem_gb=args.covariant_mem,
                       commands=[(command, False),
                                 ([*rs.PYTHON, str(rs.SOAP_CODE / "covariant" / "check_covariant.py"),
                                   "--run-dir", str(folder)], True)],
                       outputs=[folder / "covariant_soap.npz", folder / "checks.json"])
        if task.finished():
            task.state = "done"
        tasks.append(task)

    if not args.no_analyses:
        needs = [task.name for task in tasks if task.kind == "descriptor"]
        only = ["--only", *args.only] if args.only else []
        for name, script, out in (("sensitivity", "descriptor_sensitivity.py", args.out_root / "descriptor_sensitivity"),
                                  ("benchmark", "selection_benchmark.py", args.out_root / "selection_benchmark")):
            task = rs.Task(name=name, kind="analysis", cpus=args.analysis_cpus, mem_gb=args.analysis_mem, needs=needs,
                           commands=[([*rs.PYTHON, str(rs.ML_CODE / script), "--jobs", str(args.analysis_cpus),
                                       "--out-dir", str(out), *only], False)],
                           outputs=[out / "summary.json"])
            if not args.force and task.finished():
                task.state = "done"
            tasks.append(task)

    if not args.no_analyses and not args.no_plots:
        tag = ds.soap_tag(ds.REFERENCE)
        command = [*rs.PYTHON, str(rs.ML_CODE / "plot_descriptor_study.py"),
                   "--sensitivity-dir", str(args.out_root / "descriptor_sensitivity"),
                   "--benchmark-dir", str(args.out_root / "selection_benchmark"),
                   "--covariant-dir", str(Path(args.covariant_root) / "both_interfaces" / tag),
                   "--out-dir", str(args.plot_dir)]
        task = rs.Task(name="plots", kind="plots", cpus=1, mem_gb=4.0,
                       needs=[t.name for t in tasks if t.kind in ("analysis", "covariant")],
                       commands=[(command, False)],
                       outputs=[args.plot_dir / "descriptor_settings" / "overview.png",
                                args.plot_dir / "selection" / "learning_curves_grain_class.png"])
        if not args.force and task.finished():
            task.state = "done"
        tasks.append(task)
    return tasks


def grid_settings(args):
    return [{**ds.REFERENCE, "r_cut": r, "l_max": l, "sigma": s}
            for r in args.grid_r_cut for l in args.grid_l_max for s in args.grid_sigma]


def grid_arguments(args):
    """The grid flags of representation_study.py, so the study reads the runs this job makes."""
    return ["--grid-r-cut", *[f"{r:g}" for r in args.grid_r_cut], "--grid-l-max", *[str(l) for l in args.grid_l_max],
            "--grid-sigma", *[f"{s:g}" for s in args.grid_sigma]]


def covariant_reference(args):
    """Name of the lambda = 2 reference run: <tag>_lam2 until it replaces <tag>."""
    tag = ds.soap_tag(ds.REFERENCE)
    if args.part == "A" or (Path(args.covariant_root) / "both_interfaces" / f"{tag}_lam2").is_dir():
        return f"{tag}_lam2"
    return tag


def covariant_task(name, settings, run_name, args, cpus, mem_gb, thicknesses, blocks=None, compare_with=None, weight=0.0):
    """covariant_descriptor.py (never --overwrite) + check_covariant.py, which must pass."""
    folder = Path(args.covariant_root) / "both_interfaces" / run_name
    command = [*rs.PYTHON, str(rs.SOAP_CODE / "covariant" / "covariant_descriptor.py"),
               "--r-cut", f"{settings['r_cut']:g}", "--n-max", str(settings["n_max"]), "--l-max", str(settings["l_max"]),
               "--sigma", f"{settings['sigma']:g}", "--workers", str(cpus), "--out-root", str(args.covariant_root),
               "--name", run_name, "--radii", "8", "--thickness", *[str(t) for t in thicknesses]]
    if blocks:
        command += ["--blocks", *blocks]
    if args.limit:
        command += ["--limit", str(args.limit)]
    check = [*rs.PYTHON, str(rs.SOAP_CODE / "covariant" / "check_covariant.py"), "--run-dir", str(folder)]
    if compare_with:
        check += ["--compare-with", compare_with]
    commands = [(check, False)] if (folder / "covariant_soap.npz").is_file() else [(command, False), (check, False)]
    task = rs.Task(name=name, kind="covariant", cpus=cpus, mem_gb=mem_gb, weight=weight, commands=commands,
                   outputs=[folder / "covariant_soap.npz", folder / "checks.json"])
    if task.finished() and checks_pass(folder):
        task.state = "done"
    return task


def checks_pass(folder):
    import json

    path = Path(folder) / "checks.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "pass"


def study_task(name, argv, out_dir, args, needs, cpus=None, mem_gb=None):
    cpus = cpus or args.analysis_cpus
    command = [*rs.PYTHON, str(rs.ML_CODE / "representation_study.py"), *argv, "--jobs", str(cpus),
               "--descriptor-root", str(args.descriptor_root), "--covariant-root", str(args.covariant_root),
               "--labels", str(args.labels_dir / gl.CSV_NAME),
               "--out-dir", str(out_dir)]
    if args.limit:
        command += ["--families-per-plane", str(args.limit)]
    task = rs.Task(name=name, kind="study", cpus=cpus, mem_gb=mem_gb or args.analysis_mem, needs=needs,
                   commands=[(command, False)], outputs=[out_dir / "results.csv"])
    if task.finished():
        task.state = "done"
    return task


def representation_tasks(args):
    """Tasks of --study representation (see the module docstring)."""
    tag = ds.soap_tag(ds.REFERENCE)
    reference = covariant_reference(args)
    study_root = args.comparison_root / "representation_study"
    tasks = []
    if args.part == "A":
        command = [*rs.PYTHON, str(rs.ML_CODE / "geometry_labels.py"), "--jobs", str(args.labels_cpus),
                   "--out-dir", str(args.labels_dir)]
        if args.limit:
            command += ["--limit", str(args.limit)]
        labels = rs.Task(name="labels", kind="labels", cpus=args.labels_cpus, mem_gb=args.labels_mem, commands=[(command, False)],
                         outputs=[args.labels_dir / gl.CSV_NAME, args.labels_dir / gl.JSON_NAME])
        if labels.finished():
            labels.state = "done"
        tasks.append(labels)
        if not args.limit:
            krr_dir = args.out_root / tag / "interface_r8_both_t4_no_boundary"
            krr = rs.Task(name="krr", kind="analysis", cpus=args.analysis_cpus, mem_gb=args.analysis_mem, needs=["labels"],
                          commands=[([*rs.PYTHON, str(rs.ML_CODE / "krr_classify.py"), "--regression",
                                      "--labels", str(args.labels_dir / gl.CSV_NAME),
                                      "--jobs", str(args.analysis_cpus), "--out-dir", str(krr_dir)], False)],
                          outputs=[krr_dir / "krr_summary.csv", krr_dir / "krr_regression.csv"])
            benchmark_dir = args.out_root / "selection_benchmark_v2"
            benchmark = rs.Task(name="benchmark", kind="analysis", cpus=args.analysis_cpus, mem_gb=args.analysis_mem,
                                needs=["labels"],
                                commands=[([*rs.PYTHON, str(rs.ML_CODE / "selection_benchmark.py"), "--no-descriptors",
                                            "--labels", str(args.labels_dir / gl.CSV_NAME),
                                            "--jobs", str(args.analysis_cpus), "--out-dir", str(benchmark_dir)], False)],
                                outputs=[benchmark_dir / "summary.json"])
            for task in (krr, benchmark):
                if task.finished():
                    task.state = "done"
                tasks.append(task)
        if not args.no_covariant:
            tasks.append(covariant_task("covariant", ds.REFERENCE, reference, args, args.covariant_cpus, args.covariant_mem,
                                        args.thicknesses, compare_with=tag))
        for settings in grid_settings(args):
            name = ds.soap_tag(settings)
            if not args.no_descriptors:
                task, _ = rs.descriptor_task(f"grid {name}", settings, [8.0], args.descriptor_root, args.descriptor_cpus,
                                             args.thicknesses, args.limit, weight=soap_weight(settings))
                task.mem_gb = args.descriptor_mem
                tasks.append(task)
            if not args.no_covariant:
                tasks.append(covariant_task(f"covariant grid {name}", settings, f"{name}_t8", args, args.grid_covariant_cpus,
                                            args.grid_covariant_mem, [8], blocks=VECTORIAL, weight=soap_weight(settings)))
        if not args.no_analyses:
            tasks.append(study_task("phase1", ["--phase", "1", "--covariant", reference], study_root / "phase1", args,
                                    needs=["labels"] + (["covariant"] if not args.no_covariant else [])))
    else:
        for phase in ("2", "thickness"):
            folder = study_root / ("thickness" if phase == "thickness" else f"phase{phase}")
            tasks.append(study_task(f"phase{phase}" if phase == "2" else "thickness",
                                    ["--phase", phase, "--covariant", reference], folder, args, needs=[],
                                    mem_gb=args.analysis_mem * (2 if phase == "thickness" else 1)))
        tasks.append(study_task("settings_grid", ["--study", "settings_grid", "--covariant", reference,
                                                  *grid_arguments(args)],
                                args.comparison_root / "settings_grid", args, needs=[]))
    if not args.no_analyses and not args.no_plots:
        sections = ["selection", "free_labels", "representation_response"] if args.part == "A" else \
            ["representation_response", "settings_grid"]
        command = [*rs.PYTHON, str(rs.ML_CODE / "plot_descriptor_study.py"), "--only", *sections,
                   "--benchmark-dir", str(args.out_root / "selection_benchmark_v2"),
                   "--labels-file", str(args.labels_dir / gl.CSV_NAME),
                   "--krr-dir", str(args.out_root / tag / "interface_r8_both_t4_no_boundary"),
                   "--representation-dir", str(study_root), "--grid-dir", str(args.comparison_root / "settings_grid"),
                   "--out-dir", str(args.plot_dir), "--comparison-out-dir", str(args.comparison_plot_dir)]
        if args.limit:
            command += ["--title-prefix", "SMOKE TEST"]
        else:
            command += ["--archive-old", str(ARCHIVE_OLD / f"replaced_by_part{args.part}")]
        marker = args.comparison_plot_dir / "representation_response" / ("phase1_done.txt" if args.part == "A" else "phase2_done.txt")
        command += ["--done-marker", str(marker)]
        task = rs.Task(name=f"plots {args.part}", kind="plots", cpus=1, mem_gb=8.0,
                       needs=[t.name for t in tasks if t.kind in ("analysis", "study", "labels")],
                       commands=[(command, False)], outputs=[marker])
        if not args.force and task.finished():
            task.state = "done"
        tasks.append(task)
    return tasks


def analysis_task(name, script, argv, outputs, cpus, mem_gb, needs, args):
    """One analysis script; done when its outputs exist (unless --force)."""
    task = rs.Task(name=name, kind="analysis", cpus=cpus, mem_gb=mem_gb, needs=needs,
                   commands=[([*rs.PYTHON, str(rs.ML_CODE / script), *argv], False)], outputs=outputs)
    if not args.force and task.finished():
        task.state = "done"
    return task


def covariant_study_tasks(args):
    """Tasks of --study covariant (see the module docstring)."""
    tag = ds.soap_tag(ds.REFERENCE)
    covariant = ["--descriptor-type", "covariant"]
    runs = ds.covariant_runs()
    if args.only:
        known = {"reference" if knob == "reference" else f"{knob} {value}" for knob, value, _ in runs}
        if set(args.only) - known:
            raise SystemExit(f"unknown settings {sorted(set(args.only) - known)}; known: {sorted(known)}")
    tasks, run_of = [], {}
    for knob, value, settings in runs:
        setting = "reference" if knob == "reference" else f"{knob} {value}"
        if args.only and setting != "reference" and setting not in args.only:
            continue
        run_name = ds.run_name(settings, "covariant")
        if knob == "reference":
            task = covariant_task("covariant reference", settings, run_name, args, args.covariant_cpus,
                                  args.covariant_mem, args.thicknesses)
        else:
            task = covariant_task(f"covariant {setting}", settings, run_name, args, args.grid_covariant_cpus,
                                  args.grid_covariant_mem, [8], blocks=VECTORIAL, weight=soap_weight(settings))
        tasks.append(task)
        if knob in ("reference", "r_cut"):
            run_of[settings["r_cut"]] = (run_name, task.name)
    if args.no_analyses:
        return tasks

    root = args.covariant_out_root
    runs_done = [task.name for task in tasks]
    duplicates_dir, krr_dir = root / tag / "interface_r8_both_t4", root / tag / "interface_r8_both_t4_no_boundary"
    labels = ["--labels", str(args.labels_dir / gl.CSV_NAME)]
    only = ["--only", *args.only] if args.only else []
    tasks.append(analysis_task("duplicates", "find_duplicates.py",
                               [*covariant, "--structure-census", "--single-interfaces", "--out-dir", str(duplicates_dir)],
                               [duplicates_dir / "duplicates_summary.json"], 2, args.duplicates_mem,
                               ["covariant reference"], args))
    tasks.append(analysis_task("krr", "krr_classify.py",
                               [*covariant, "--regression", *labels, "--census-grouping", "--jobs", str(args.krr_cpus),
                                "--out-dir",
                                str(krr_dir)],
                               [krr_dir / "krr_summary.csv", krr_dir / "krr_regression.csv"], args.krr_cpus,
                               args.krr_mem, ["covariant reference", "duplicates"], args))
    for name, subset, folder in (("select", [], ""), ("select plane100", ["--planes", "100", "--min-pair", "1.8"],
                                                       "plane100_dmin1.8")):
        out = root / "dim-reduction" / folder if folder else root / "dim-reduction"
        plots = args.covariant_plot_dir / folder if folder else args.covariant_plot_dir
        tasks.append(analysis_task(name, "select_gbs.py", [*covariant, *subset, "--out-dir", str(out),
                                                           "--plot-dir", str(plots)], [out / "run.json"], 1,
                                   args.duplicates_mem, ["covariant reference"], args))

    scan_dir = root / "rcut_scan_8x"
    scan_tasks = []
    for r_cut in SCAN_R_CUTS:
        if r_cut not in run_of:
            continue
        run_name, needs = run_of[r_cut]
        out = scan_dir / f"rc{r_cut:g}" / mc.vector_tag("interface", 8.0, "both", 8)
        vector = [*covariant, "--descriptor", run_name, "--thickness", "8", "--out-dir", str(out)]
        task = rs.Task(name=f"scan rc{r_cut:g}", kind="analysis", cpus=args.scan_cpus, mem_gb=args.scan_mem,
                       needs=[needs], weight=r_cut,
                       commands=[([*rs.PYTHON, str(rs.ML_CODE / "find_duplicates.py"), *vector], False),
                                 ([*rs.PYTHON, str(rs.ML_CODE / "krr_classify.py"), *vector, "--keep-no-boundary", "--census-grouping",
                                   "--jobs", str(args.scan_cpus)], False)],
                       outputs=[out / "duplicates_summary.json", out / "krr_summary.csv"])
        if not args.force and task.finished():
            task.state = "done"
        scan_tasks.append(task)
    tasks += scan_tasks
    if scan_tasks:
        summary = rs.Task(name="scan summary", kind="analysis", cpus=1, mem_gb=4.0,
                          needs=[task.name for task in scan_tasks],
                          commands=[([*rs.PYTHON, str(rs.ML_CODE / "rcut_scan_summary.py"), "--scan-dir",
                                      str(scan_dir)], False)],
                          outputs=[scan_dir / "rcut_scan_summary.csv"])
        if not args.force and summary.finished():
            summary.state = "done"
        tasks.append(summary)

    sensitivity_dir, benchmark_dir = root / "descriptor_sensitivity", root / "selection_benchmark"
    tasks.append(analysis_task("sensitivity", "descriptor_sensitivity.py",
                               [*covariant, "--jobs", str(args.analysis_cpus), "--out-dir", str(sensitivity_dir), *only],
                               [sensitivity_dir / "summary.json"], args.analysis_cpus, args.sensitivity_mem,
                               runs_done, args))
    tasks.append(analysis_task("benchmark", "selection_benchmark.py",
                               [*covariant, "--krr-settings", str(krr_dir), *labels, "--jobs", str(args.analysis_cpus),
                                "--existing-dir", str(args.covariant_plot_dir), "--out-dir", str(benchmark_dir), *only],
                               [benchmark_dir / "summary.json"], args.analysis_cpus, args.analysis_mem,
                               runs_done + ["krr", "select"], args))

    if not args.no_plots:
        plot = [*rs.PYTHON, str(rs.ML_CODE / "plot_descriptor_study.py")]
        marker = args.covariant_plot_dir / "plots_done.txt"
        commands = [([*plot, *covariant, "--sensitivity-dir", str(sensitivity_dir), "--benchmark-dir", str(benchmark_dir),
                      "--krr-dir", str(krr_dir), "--scan-dir", str(scan_dir),
                      "--covariant-dir", str(Path(args.covariant_root) / "both_interfaces" / tag),
                      "--labels-file", str(args.labels_dir / gl.CSV_NAME), "--out-dir", str(args.covariant_plot_dir),
                      "--comparison-out-dir", str(args.comparison_plot_dir),
                      "--only", *COVARIANT_PLOT_SECTIONS, "grain_class_overview", "--done-marker", str(marker)], False),
                    # the same stage1/ figures for standard SOAP, so both trees show the same things
                    ([*plot, "--only", "stage1", "--out-dir", str(args.plot_dir)], True)]
        task = rs.Task(name="plots", kind="plots", cpus=1, mem_gb=8.0,
                       needs=[t.name for t in tasks if t.kind == "analysis"], commands=commands, outputs=[marker])
        if not args.force and task.finished():
            task.state = "done"
        tasks.append(task)
    return tasks


def default_memory():
    """Memory of the Slurm allocation in GB, less 10% headroom; None outside Slurm."""
    value = os.environ.get("SLURM_MEM_PER_NODE")
    return 0.9 * int(value) / 1024 if value else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=("descriptor", "representation", "covariant"), default="descriptor",
                        help="descriptor: the descriptor study of 2026-10-03 (default); representation: the "
                             "fifth and sixth studies; covariant: the standard-SOAP analyses on covariant SOAP")
    parser.add_argument("--part", choices=("A", "B"), default="A", help="representation study only: job A or job B")
    parser.add_argument("--only", nargs="+", help="setting names of descriptor_sensitivity.py to include "
                                                  "(e.g. reference \"n_max 12\"); default all")
    parser.add_argument("--thicknesses", type=int, nargs="+", default=list(range(4, 13)))
    parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                        help="CPUs to keep busy (default: $SLURM_CPUS_PER_TASK, else all CPUs of this machine)")
    parser.add_argument("--mem-gb", type=float, default=default_memory(),
                        help="memory budget in GB (default: 90%% of the Slurm allocation; none outside Slurm)")
    parser.add_argument("--descriptor-cpus", type=int, default=16, help="worker processes per SOAP run (default 16)")
    parser.add_argument("--descriptor-mem", type=float, default=16.0, help="GB per SOAP run (default 16)")
    parser.add_argument("--covariant-cpus", type=int, default=64, help="worker processes of the covariant run (default 64)")
    parser.add_argument("--covariant-mem", type=float, default=None,
                        help="GB for the covariant run (default 64; 96 for --study representation, whose lambda = 2 "
                             "blocks more than double the arrays)")
    parser.add_argument("--analysis-cpus", type=int, default=32, help="worker processes per analysis (default 32)")
    parser.add_argument("--analysis-mem", type=float, default=64.0, help="GB per analysis (default 64)")
    parser.add_argument("--labels-cpus", type=int, default=16, help="worker processes of geometry_labels.py (default 16)")
    parser.add_argument("--labels-mem", type=float, default=16.0, help="GB for geometry_labels.py (default 16)")
    parser.add_argument("--grid-covariant-cpus", type=int, default=16,
                        help="worker processes per covariant grid run at 8x (default 16)")
    parser.add_argument("--grid-covariant-mem", type=float, default=24.0, help="GB per covariant grid run (default 24)")
    parser.add_argument("--krr-cpus", type=int, default=16,
                        help="--study covariant: folds run at once by the Stage 1.4 krr task (default 16)")
    parser.add_argument("--krr-mem", type=float, default=112.0,
                        help="GB for that task (default 112: about 5 GB per fold process plus the vectors)")
    parser.add_argument("--scan-cpus", type=int, default=8, help="--study covariant: folds at once per scan task (default 8)")
    parser.add_argument("--scan-mem", type=float, default=56.0, help="GB per scan task (default 56)")
    parser.add_argument("--duplicates-mem", type=float, default=40.0,
                        help="--study covariant: GB for find_duplicates.py and select_gbs.py on all thicknesses (default 40)")
    parser.add_argument("--sensitivity-mem", type=float, default=160.0,
                        help="--study covariant: GB for descriptor_sensitivity.py (default 160)")
    parser.add_argument("--labels-dir", type=Path, default=gl.OUT_DIR, help="where geometry_labels.py writes")
    parser.add_argument("--grid-r-cut", type=float, nargs="+", default=list(GRID["r_cut"]),
                        help="sixth study: r_cut values (default 5 6 8)")
    parser.add_argument("--grid-l-max", type=int, nargs="+", default=list(GRID["l_max"]), help="l_max values (default 4 8)")
    parser.add_argument("--grid-sigma", type=float, nargs="+", default=list(GRID["sigma"]),
                        help="sigma values (default 0.3 0.5)")
    parser.add_argument("--descriptor-root", type=Path, default=sc.STANDARD_ROOT)
    parser.add_argument("--covariant-root", type=Path, default=sc.COVARIANT_ROOT)
    parser.add_argument("--out-root", type=Path, default=mc.OUTPUT_ROOT,
                        help="where the standard-SOAP analyses write (default calculations/ML_dscribe/CoGe/standard)")
    parser.add_argument("--covariant-out-root", type=Path, default=mc.OUTPUT_ROOTS["covariant"],
                        help="--study covariant: where the analyses write (default calculations/ML_dscribe/CoGe/covariant)")
    parser.add_argument("--comparison-root", type=Path, default=mc.COMPARISON_ROOT,
                        help="--study representation: where representation_study and settings_grid write "
                             "(default calculations/ML_dscribe/CoGe/comparison)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help=f"default {sc.relative_path(LOG_DIR)}, or {sc.relative_path(REPRESENTATION_LOG_DIR)} "
                             "for --study representation")
    parser.add_argument("--no-descriptors", action="store_true", help="no standard SOAP runs")
    parser.add_argument("--no-covariant", action="store_true")
    parser.add_argument("--no-analyses", action="store_true", help="no analyses (and so no figures)")
    parser.add_argument("--no-plots", action="store_true", help="no figures")
    parser.add_argument("--plot-dir", type=Path, default=mc.plot_root("soap"),
                        help="where plot_descriptor_study.py writes its standard-SOAP folders")
    parser.add_argument("--covariant-plot-dir", type=Path, default=mc.plot_root("covariant"),
                        help="--study covariant: where the covariant figures go")
    parser.add_argument("--comparison-plot-dir", type=Path, default=mc.COMPARISON_PLOTS,
                        help="where it writes representation_response/ and settings_grid/")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (smoke test)")
    parser.add_argument("--force", action="store_true", help="redo analyses and figures whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="print the task list and stop")
    args = parser.parse_args()
    for name in ("descriptor_cpus", "covariant_cpus", "analysis_cpus", "labels_cpus", "grid_covariant_cpus", "krr_cpus",
                 "scan_cpus"):
        setattr(args, name, max(1, min(getattr(args, name), args.cpus)))
    if args.covariant_mem is None:
        args.covariant_mem = 96.0 if args.study == "representation" else 64.0
    if args.log_dir is None:
        args.log_dir = {"representation": REPRESENTATION_LOG_DIR / f"part{args.part}",
                        "covariant": COVARIANT_LOG_DIR}.get(args.study, LOG_DIR)
    if args.limit and not args.no_analyses and args.study in ("descriptor", "covariant"):
        parser.error("--limit gives an incomplete catalogue; add --no-analyses")
    if args.limit and args.study == "covariant" and args.covariant_root == sc.COVARIANT_ROOT:
        parser.error("--limit writes partial runs; give a scratch --covariant-root")
    if args.limit and args.study != "covariant" and (args.descriptor_root == sc.STANDARD_ROOT
                                                     or args.covariant_root == sc.COVARIANT_ROOT):
        parser.error("--limit writes partial runs; give a scratch --descriptor-root and --covariant-root")
    if args.limit and args.study == "representation":
        defaults = {"--labels-dir": (args.labels_dir, gl.OUT_DIR), "--out-root": (args.out_root, mc.OUTPUT_ROOT),
                    "--comparison-root": (args.comparison_root, mc.COMPARISON_ROOT),
                    "--plot-dir": (args.plot_dir, mc.plot_root("soap")),
                    "--comparison-plot-dir": (args.comparison_plot_dir, mc.COMPARISON_PLOTS),
                    "--log-dir": (args.log_dir, REPRESENTATION_LOG_DIR / f"part{args.part}")}
        unchanged = [flag for flag, (value, default) in defaults.items() if Path(value) == Path(default)]
        if unchanged:
            parser.error(f"--limit is a smoke test; give scratch folders for {' '.join(unchanged)}")

    tasks = {"representation": representation_tasks, "covariant": covariant_study_tasks}.get(args.study, build_tasks)(args)
    if args.mem_gb:
        # A task asking for more than the whole budget would never start.
        for task in tasks:
            task.mem_gb = min(task.mem_gb, args.mem_gb)
    todo = [task for task in tasks if task.state == "waiting"]
    budget = f"{args.mem_gb:.0f} GB" if args.mem_gb else "no memory budget"
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {args.cpus} CPUs, {budget}")
    for task in tasks:
        print(f"  {task.state:8s} {task.cpus:3d} CPUs {task.mem_gb:5.0f} GB  {task.name}")
    if args.dry_run:
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    rs.run(tasks, args.cpus, args.log_dir, args.log_dir / "status.json", total_mem_gb=args.mem_gb, order=ORDER)
    bad = [task for task in tasks if task.state in ("failed", "skipped")]
    print(f"\n{sum(t.state == 'ok' for t in tasks)} tasks ran, {sum(t.state == 'done' for t in tasks)} were already "
          f"done, {len(bad)} failed or skipped")
    for task in bad:
        print(f"  {task.state:8s} {task.name}  (log: {args.log_dir / (task.name.replace(' ', '_') + '.log')})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
