#!/usr/bin/env python3
"""Covariant ACE (lambda = 1 and 2) for the bulk cells and the grain boundaries.

The lambda >= 1 companions of the invariant ACE blocks (see covariant_ace.py): nu1_{polar,tensor},
nu2_{polar,axial,tensor,pseudotensor} and nu3_{polar,axial,tensor,pseudotensor}, with the ACE settings
of the invariant run (same flags as src/ACE/CoGe/gb/gb_descriptor.py). They are what the representation
study needs for its covariant rows C and D. Averaging, plane assignment and the structure list are the
ones of the invariant scripts; the bulk cells go through the same workers first (numbered -1, -2, ...).

A covariant vector is large (569,600 entries at the default n3 = l3 = 4, 2.3 MB in float32), so a run
over the whole catalogue would take about 95 GB. --structures limits it to a list of (family,
thickness) pairs; src/ACE/CoGe/ace_grid.py writes the list of the reference run (every GB at 8x plus
the 300 GBs of the representation study's thickness check at every thickness).

Worker memory: featomic's DensityCorrelations.compute (0.6.7) leaves arrays registered in metatensor
(metatensor._data._array._KNOWN_ARRAY_WRAPPERS) after every call, and gc does not free them: about 3,100
arrays and 60-160 MB per 8x structure here, against about 1,400 small ones for invariant ACE. In job 13012
(2026-10-13) the workers reached 24 GB each after about 150 structures, so each worker is now replaced
after --max-tasks-per-worker structures (default 30, about 5 GB at most).

Outputs, under calculations/descriptors/ACE/covariant/:
  global/<name>/           covariant_ace.npz (X_<block>: N, C, 2P), bulk_covariant_ace.npz, index.csv,
                           bulk_index.csv, settings.json, skipped.csv
  both_interfaces/<name>/  covariant_ace.npz (X_<block>_r<R>: N, plane, C, 2P; n_r<R>), index.csv,
                           settings.json, skipped.csv
Then: python -B src/ACE/CoGe/covariant/check_covariant.py --run-dir <name>
"""
import os

# One thread per worker process; the parallelism is over structures.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import covariant_ace as cov

ac, sc = cov.ac, cov.sc
ALL_THICKNESSES = tuple(range(4, 13))
INDEX_FIELDS = ["index", "plane", "sym", "thickness", "family", "n_atoms", "n_Co", "n_Ge", "Lx", "Ly", "Lz"]
_STATE = {}


def index_fields(blocks):
    return INDEX_FIELDS + [f"atom_norm_{name}" for name in blocks] + ["source"]


def bulk_fields(blocks):
    return ["index", "kind", "plane", "sym", "n_atoms"] + [f"atom_norm_{name}" for name in blocks] + ["source"]


def block_arrays(blocks, radii=None):
    """Description of covariant_ace.npz, written into settings.json."""
    planes = "plane, " if radii else ""
    suffix = "_r<R>" if radii else ""
    text = {}
    for name in blocks:
        nu, lam, sig = cov.BLOCKS[name]
        axis = "x, y, z" if lam == 1 else "m = -2 ... 2"
        text[f"X_{name}{suffix}"] = (f"(N, {planes}{2 * lam + 1}, 2P) body order {nu + 1}, lambda {lam}, "
                                     f"sigma {sig:+d}; component axis {axis}")
    text.update({
        "properties_<block>": "(P, k) featomic property labels of one species half; names in property_names_<block>",
        "vector layout": "last axis [0:P] Co-centred mean, [P:2P] Ge-centred mean (P differs between blocks)",
        "rows": "same order as index.csv",
    })
    if radii:
        text["n_r<R>"] = "(N, plane, species) atoms behind each vector (the same for every block)"
        text["plane axis"] = "0 = edge (x = 0), 1 = mid (x = Lx/2); each atom belongs to its nearer plane"
        text["radii"] = radii
    return text


def read_structure_list(path):
    """{(family, thickness)} from a CSV with columns family (e.g. 100/sym00010) and thickness."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"family", "thickness"} <= set(rows[0]):
        raise ValueError(f"{path}: needs the columns family and thickness")
    return {(row["family"], int(row["thickness"])) for row in rows}


def select_jobs(jobs, wanted):
    """The jobs whose (family, thickness) is in `wanted`, and the wanted pairs with no structure."""
    kept = [job for job in jobs if (f"{job[0]}/{job[1]}", job[2]) in wanted]
    found = {(f"{job[0]}/{job[1]}", job[2]) for job in kept}
    return kept, sorted(wanted - found)


def default_name(args, settings):
    """cov.run_name() with a suffix when the catalogue was filtered (as gb_descriptor.default_name)."""
    name = cov.run_name(settings)
    if set(args.planes) != set(sc.PLANES):
        name += "_p" + "-".join(sorted(args.planes))
    if set(args.thickness) != set(ALL_THICKNESSES):
        name += "_t" + "-".join(str(t) for t in sorted(args.thickness))
    if args.structures:
        name += "_" + args.structures.stem
    if args.limit:
        name += f"_lim{args.limit}"
    return name


def _init_worker(settings, chunk, radii, labels_path):
    calculator = cov.CovariantCalculator(settings, chunk)
    # Property labels come through a file, as in gb_descriptor.py (passed directly they slow the start-up).
    with np.load(labels_path) as data:
        for block in calculator.blocks:
            calculator.properties[block] = data[f"properties_{block}"]
            calculator.property_names[block] = [str(name) for name in data[f"property_names_{block}"]]
    _STATE.update(calculator=calculator, radii=radii)


def _work(task):
    i, path = task
    try:
        if i < 0:  # bulk cell number -1 - i
            return i, cov.describe_bulk(path, _STATE["calculator"]), None
        return i, cov.describe_structure(path, _STATE["calculator"], _STATE["radii"]), None
    except Exception as exc:  # recorded in skipped.csv and reported at the end
        return i, None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ac.add_ace_arguments(parser)
    parser.add_argument("--radii", type=float, nargs="+", default=[8.0], help="interface radii in A (default 8)")
    parser.add_argument("--gb-root", type=Path, default=sc.GB_ROOT)
    parser.add_argument("--bulk-root", type=Path, default=sc.BULK_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--thickness", type=int, nargs="+", default=list(ALL_THICKNESSES),
                        choices=ALL_THICKNESSES, metavar="N", help="thicknesses to include, 4..12 (default all)")
    parser.add_argument("--structures", type=Path, default=None,
                        help="CSV with columns family, thickness: only these structures (default every one)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (testing)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="worker processes (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--max-tasks-per-worker", type=int, default=30,
                        help="replace each worker after this many structures, which bounds the memory featomic "
                             "leaks (about 160 MB per 8x structure); 0 = never (default 30)")
    parser.add_argument("--chunk", type=int, default=64,
                        help="atoms per featomic call; bounds memory, does not change the numbers (default 64)")
    parser.add_argument("--out-root", type=Path, default=cov.COVARIANT_ROOT,
                        help="folder holding global/ and both_interfaces/ (default calculations/descriptors/ACE/covariant)")
    parser.add_argument("--name", default=None,
                        help="run folder name (default: the ACE tag plus _lam1-2, with a suffix when filtered)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--dry-run", action="store_true",
                        help="count the structures, print the vector sizes, output size and time per atom of the "
                             "pristine cell, and stop")
    args = parser.parse_args()

    settings = ac.ace_settings(args)
    blocks = cov.blocks_for(settings)
    radii = sorted(set(args.radii))
    name = args.name or default_name(args, settings)
    global_dir, out_dir = args.out_root / "global" / name, args.out_root / "both_interfaces" / name
    if (out_dir / cov.FILE).exists() and not args.overwrite:
        print(f"{out_dir / cov.FILE} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    cells = ac.collect_cells(args.bulk_root, args.planes, args.limit)
    jobs, skipped = ac.collect_structures(args.gb_root, args.planes, sorted(args.thickness), args.limit)
    structure_list = None
    if args.structures:
        wanted = read_structure_list(args.structures)
        jobs, missing = select_jobs(jobs, wanted)
        listed_skips = [s for s in skipped if (f"{s[0]}/{s[1]}", s[2]) in wanted]
        skipped = listed_skips
        missing = [m for m in missing if m not in {(f"{s[0]}/{s[1]}", s[2]) for s in listed_skips}]
        if missing and not args.limit:
            print(f"{len(missing)} listed structures not found, e.g. {missing[:3]}", file=sys.stderr)
            return 2
        structure_list = {"path": str(sc.relative_path(args.structures.resolve())), "n_pairs": len(wanted),
                          "sha256": hashlib.sha256(args.structures.read_bytes()).hexdigest()}
    if not jobs:
        print("no structures found", file=sys.stderr)
        return 2

    start = time.time()
    # The pristine cell fixes the block widths and property labels every structure must reproduce.
    calculator = cov.CovariantCalculator(settings, args.chunk)
    pristine, _ = sc.read_structure(sc.PRISTINE_POSCAR)
    calculator.per_atom(pristine)
    widths = calculator.widths()
    shapes = {block: (cov.components(block), 2 * widths[block]) for block in blocks}
    n_vector = sum(c * p for c, p in shapes.values())
    n, n_radii = len(jobs), len(radii)
    gigabytes = 4 * n * n_vector * (1 + 2 * n_radii) / 1e9
    summary = (f"{len(cells)} bulk cells, {n} structures, {len(skipped)} skipped up front; blocks "
               + ", ".join(f"{block} {c}x{p}" for block, (c, p) in shapes.items())
               + f" ({n_vector} entries per vector); radii {radii} A")
    if args.dry_run:
        tick = time.time()
        calculator.per_atom(pristine)
        ms_per_atom = 1000 * (time.time() - tick) / len(pristine)
        print(summary)
        print(f"output about {gigabytes:.1f} GB (float32, global + {n_radii} radii x 2 planes); pristine cell "
              f"{ms_per_atom:.1f} ms per atom on this CPU (overhead-dominated for 8 atoms); folders "
              f"{sc.relative_path(global_dir)} and {sc.relative_path(out_dir)}")
        return 0
    print(summary + f", {args.workers} worker(s), about {gigabytes:.1f} GB of output", flush=True)

    X_global = {block: np.zeros((n,) + shape, dtype=np.float32) for block, shape in shapes.items()}
    X_interface = {block: np.zeros((n, n_radii, 2) + shape, dtype=np.float32) for block, shape in shapes.items()}
    n_interface = np.zeros((n, n_radii, 2, 2), dtype=np.int32)
    meta = [None] * n
    bulk = {block: np.zeros((len(cells),) + shape) for block, shape in shapes.items()}
    bulk_meta, bulk_errors = [None] * len(cells), []
    # Bulk cells first, then the largest GB cells first so the workers stay evenly loaded.
    order = sorted(range(n), key=lambda i: -jobs[i][3].stat().st_size)
    bulk_tasks = [(-1 - k, cell[3]) for k, cell in enumerate(cells)]
    tasks = bulk_tasks + [(i, jobs[i][3]) for i in order]
    pool_start = time.time()

    def store(i, result, error):
        if i < 0:
            k = -1 - i
            if error:
                bulk_errors.append(f"{cells[k][3]}: {error}")
                return
            for block in blocks:
                bulk[block][k] = result["vectors"][block]
            bulk_meta[k] = result["meta"]
            return
        if error:
            skipped.append((*jobs[i], error))
            return
        for block in blocks:
            X_global[block][i] = result["global"][block]
            X_interface[block][i] = result["interface"][block]
        n_interface[i] = result["n_interface"]
        meta[i] = result["meta"]

    def report(done):
        if done == len(cells):
            print(f"  {len(cells)} bulk cells, {time.time() - pool_start:.0f} s", flush=True)
        elif done > len(cells) and ((done - len(cells)) % 250 == 0 or done == len(tasks)):
            print(f"  {done - len(cells)}/{n} structures, {time.time() - pool_start:.0f} s", flush=True)

    with tempfile.TemporaryDirectory(prefix="ace_covariant_labels_") as scratch:
        labels_path = Path(scratch) / "labels.npz"
        np.savez(labels_path, **cov.label_arrays(calculator))
        init = (settings, args.chunk, radii, str(labels_path))
        if args.workers > 1:
            # featomic has already run in this process, and forking after that can hang the workers.
            context = multiprocessing.get_context("spawn")
            done = 0
            # The bulk cells are small and leak little, so their pool keeps its workers; the GB pool replaces
            # each worker after --max-tasks-per-worker structures.
            for part, max_tasks in ((bulk_tasks, None), (tasks[len(bulk_tasks):], args.max_tasks_per_worker or None)):
                with context.Pool(args.workers, initializer=_init_worker, initargs=init,
                                  maxtasksperchild=max_tasks) as pool:
                    for outcome in pool.imap_unordered(_work, part, chunksize=1):
                        done += 1
                        store(*outcome)
                        report(done)
        else:
            _init_worker(*init)
            for done, task in enumerate(tasks, 1):
                store(*_work(task))
                report(done)
    if bulk_errors:
        print("bulk cells failed; nothing written:\n  " + "\n  ".join(bulk_errors), file=sys.stderr)
        return 1
    bulk_rows = [{"index": k, "kind": kind, "plane": plane, "sym": sym, **bulk_meta[k],
                  "source": sc.relative_path(path)} for k, (kind, plane, sym, path) in enumerate(cells)]

    keep = np.array([m is not None for m in meta])
    rows = []
    for i in np.flatnonzero(keep):
        plane, sym, thickness, path = jobs[i]
        rows.append({"index": len(rows), "plane": plane, "sym": sym, "thickness": thickness,
                     "family": f"{plane}/{sym}", **meta[i], "source": sc.relative_path(path)})
    if not rows:
        print("every structure failed; nothing written", file=sys.stderr)
        return 1

    labels = cov.label_arrays(calculator)
    global_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.write_npz(global_dir / cov.FILE, **{f"X_{b}": X_global[b][keep] for b in blocks}, **labels,
                 species=np.array(sc.SPECIES))
    sc.write_npz(global_dir / cov.BULK_FILE, **{f"X_{b}": bulk[b] for b in blocks}, **labels,
                 species=np.array(sc.SPECIES))
    sc.write_csv(global_dir / "bulk_index.csv", bulk_rows, bulk_fields(blocks))
    per_radius = {}
    for r, radius in enumerate(radii):
        key = sc.radius_key(radius)
        per_radius.update({f"X_{b}_{key}": X_interface[b][keep][:, r] for b in blocks})
        per_radius[f"n_{key}"] = n_interface[keep][:, r]
    sc.write_npz(out_dir / cov.FILE, **per_radius, **labels, radii=np.array(radii, dtype=float),
                 plane_names=np.array(sc.PLANE_NAMES), species=np.array(sc.SPECIES))
    skipped_rows = [{"plane": plane, "sym": sym, "thickness": thickness, "reason": reason,
                     "source": sc.relative_path(path)} for plane, sym, thickness, path, reason in skipped]
    common = {
        "ace": settings, "featomic_hypers": ac.featomic_hypers(settings), "invariant_tag": ac.ace_tag(settings),
        "blocks": {b: {"nu": cov.BLOCKS[b][0], "lambda": cov.BLOCKS[b][1], "sigma": cov.BLOCKS[b][2]} for b in blocks},
        "radii": radii, "planes": sorted(args.planes), "thicknesses": sorted(args.thickness),
        "structure_list": structure_list, "limit": args.limit,
        "n_features_per_component": widths, "vector_shape": {b: list(s) for b, s in shapes.items()},
        "n_structures": len(rows), "n_atoms": int(sum(int(row["n_atoms"]) for row in rows)),
        "n_skipped": len(skipped_rows), "n_bulk_cells": len(bulk_rows), "workers": args.workers,
        "chunk": args.chunk, "max_tasks_per_worker": args.max_tasks_per_worker, "seconds": round(time.time() - start, 2), "array_dtype": "float32",
        "scaling": "loaders divide each block by atom_norm_<block> of the pristine row of bulk_index.csv",
    }
    for folder, arrays in ((global_dir, block_arrays(blocks)), (out_dir, block_arrays(blocks, radii))):
        sc.write_csv(folder / "index.csv", rows, index_fields(blocks))
        sc.write_csv(folder / "skipped.csv", skipped_rows, ["plane", "sym", "thickness", "reason", "source"])
        sc.write_json(folder / "settings.json", {**common, "arrays": arrays, **ac.run_metadata()})

    print(f"wrote {len(rows)} structures ({common['n_atoms']} atoms) and {len(bulk_rows)} bulk cells to "
          f"{sc.relative_path(global_dir)} and {sc.relative_path(out_dir)} in {time.time() - start:.0f} s")
    unexpected = [row for row in skipped_rows if row["reason"] != "empty file"]
    for row in skipped_rows:
        print(f"  skipped {row['source']}: {row['reason']}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
