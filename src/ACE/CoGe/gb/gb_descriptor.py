#!/usr/bin/env python3
"""Step 1.3 (ACE): one ACE vector per grain boundary, global and interface-centred.

Same structures, plane assignment and averaging as the SOAP script src/dscribe/CoGe/gb/gb_descriptor.py
(its structure list and helpers are imported). For every structure the per-atom ACE blocks are
computed once and stored as:

  X_<block>         mean over every atom in the cell
  X_<block>_r<R>    mean over atoms within R of an interface plane, for every R in --radii,
                    kept separately for the edge (x = 0) and the middle (x = Lx/2) interface

Blocks: nu1 (2-body), nu2 (3-body, the SOAP power spectrum), nu3_even and nu3_odd (4-body; odd
flips sign under inversion), see ace_common.py. There is no X_nonbulk: its test against pristine
bulk would flag every atom of a mirrored grain through the odd block.

Outputs go into two folders under calculations/descriptors/ACE/standard/:
  global/<tag>/           global_ace.npz
  both_interfaces/<tag>/  interface_ace.npz (one set of X_<block>_r<R> and one n_r<R> per radius)
each with index.csv (one row per structure, same order), settings.json and skipped.csv.
"""
import os

# One thread per worker process; the parallelism is over structures.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ace_common as ac

sc = ac.sc
ALL_THICKNESSES = tuple(range(4, 13))
INDEX_FIELDS = ["index", "plane", "sym", "thickness", "family", "n_atoms", "n_Co", "n_Ge",
                "Lx", "Ly", "Lz", "area", "min_pair_distance"]
_STATE = {}


def index_fields(blocks):
    return INDEX_FIELDS + [f"atom_norm_{name}" for name in blocks] + ["source"]


def global_arrays(widths):
    return {
        **{f"X_{name}": f"(N, {2 * width}) mean over all atoms" for name, width in widths.items()},
        "properties_<block>": "(P, k) featomic property labels of one species half; names in property_names_<block>",
        "vector layout": "[0:P] Co-centred mean, [P:2P] Ge-centred mean (P differs between blocks)",
        "rows": "same order as index.csv",
    }


def interface_arrays(widths, radii):
    return {
        **{f"X_{name}_r<R>": f"(N, plane, {2 * width}) mean over atoms within R A of that plane"
           for name, width in widths.items()},
        "n_r<R>": "(N, plane, species) atoms behind each X_<block>_r<R> vector (the same for every block)",
        "radii": radii,
        "plane axis": "0 = edge (x = 0), 1 = mid (x = Lx/2); each atom belongs to its nearer plane",
        "properties_<block>": "(P, k) featomic property labels of one species half; names in property_names_<block>",
        "vector layout": "[0:P] Co-centred mean, [P:2P] Ge-centred mean (P differs between blocks)",
        "rows": "same order as index.csv",
    }


def describe_structure(path, calculator, radii):
    """All vectors and diagnostics for one grain-boundary cell."""
    atoms, _ = sc.read_structure(path)
    cell = atoms.cell.array
    lengths = np.diag(cell).copy()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6:
        raise ValueError("cell is not orthogonal")
    if lengths[0] < lengths[1:].max():
        raise ValueError("x is not the longest axis, so the interface normal is unclear")
    lx = lengths[0]
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    per_atom = calculator.per_atom(atoms)
    distance, plane = sc.plane_distances(atoms.get_scaled_positions(wrap=True)[:, 0] * lx, lx)

    global_vectors, counts = ac.species_means(per_atom, is_co)
    interface = {name: np.empty((len(radii), 2, 2 * values.shape[1]), dtype=np.float32)
                 for name, values in per_atom.items()}
    n_interface = np.empty((len(radii), 2, 2), dtype=np.int32)
    for r, radius in enumerate(radii):
        for p in (0, 1):
            vectors, n_interface[r, p] = ac.species_means(per_atom, is_co, (distance <= radius) & (plane == p))
            for name, vector in vectors.items():
                interface[name][r, p] = vector
    meta = {
        "n_atoms": len(atoms), "n_Co": int(counts[0]), "n_Ge": int(counts[1]),
        "Lx": f"{lengths[0]:.8f}", "Ly": f"{lengths[1]:.8f}", "Lz": f"{lengths[2]:.8f}",
        "area": f"{lengths[1] * lengths[2]:.8f}",
        "min_pair_distance": f"{sc.min_pair_distance(atoms):.6f}",
        **ac.atom_norms(per_atom),
    }
    return {"global": {name: vector.astype(np.float32) for name, vector in global_vectors.items()},
            "interface": interface, "n_interface": n_interface, "meta": meta}


def _init_worker(settings, chunk, radii, properties, property_names):
    calculator = ac.Calculator(settings, chunk)
    # Every worker checks its property labels against the ones the main process found.
    calculator.properties, calculator.property_names = properties, property_names
    _STATE.update(calculator=calculator, radii=radii)


def _work(task):
    i, path = task
    try:
        return i, describe_structure(path, _STATE["calculator"], _STATE["radii"]), None
    except Exception as exc:  # recorded in skipped.csv and reported at the end
        return i, None, f"{type(exc).__name__}: {exc}"


def default_name(args, settings):
    """The run's folder name: the ACE tag, with a suffix when the catalogue was filtered."""
    name = ac.ace_tag(settings)
    if set(args.planes) != set(sc.PLANES):
        name += "_p" + "-".join(sorted(args.planes))
    if set(args.thickness) != set(ALL_THICKNESSES):
        name += "_t" + "-".join(str(t) for t in sorted(args.thickness))
    if args.limit:
        name += f"_lim{args.limit}"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ac.add_ace_arguments(parser)
    parser.add_argument("--radii", type=float, nargs="+", default=[8.0],
                        help="interface radii in A: atoms within this distance of an interface plane, "
                             "on both sides (default 8)")
    parser.add_argument("--gb-root", type=Path, default=sc.GB_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--thickness", type=int, nargs="+", default=list(ALL_THICKNESSES),
                        choices=ALL_THICKNESSES, metavar="N", help="thicknesses to include, 4..12 (default all)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (testing)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="worker processes (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--chunk", type=int, default=128,
                        help="atoms per featomic call; bounds memory, does not change the numbers (default 128)")
    parser.add_argument("--out-root", type=Path, default=ac.STANDARD_ROOT,
                        help="folder holding global/ and both_interfaces/ (default calculations/descriptors/ACE/standard)")
    parser.add_argument("--name", default=None,
                        help="run folder name inside each region (default: the ACE tag, with a suffix when filtered)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--dry-run", action="store_true",
                        help="count the structures, print the vector sizes and the time per atom of the pristine "
                             "cell, and stop")
    args = parser.parse_args()

    settings = ac.ace_settings(args)
    blocks = ac.blocks_for(settings)
    radii = sorted(set(args.radii))
    name = args.name or default_name(args, settings)
    global_dir, out_dir = args.out_root / "global" / name, args.out_root / "both_interfaces" / name
    if (out_dir / ac.INTERFACE_FILE).exists() and not args.overwrite:
        print(f"{out_dir / ac.INTERFACE_FILE} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    jobs, skipped = ac.collect_structures(args.gb_root, args.planes, sorted(args.thickness), args.limit)
    if not jobs:
        print("no structures found", file=sys.stderr)
        return 2
    start = time.time()
    # The pristine cell fixes the block widths and property labels every structure must reproduce.
    calculator = ac.Calculator(settings, args.chunk)
    pristine, _ = sc.read_structure(sc.PRISTINE_POSCAR)
    calculator.per_atom(pristine)
    widths = calculator.widths()
    n_vector = sum(2 * width for width in widths.values())
    n, n_radii = len(jobs), len(radii)
    if args.dry_run:
        tick = time.time()
        calculator.per_atom(pristine)
        ms_per_atom = 1000 * (time.time() - tick) / len(pristine)
        gigabytes = 4 * n * n_vector * (1 + 2 * n_radii) / 1e9
        print(f"{n} structures, {len(skipped)} skipped up front; blocks "
              + ", ".join(f"{block} {2 * width}" for block, width in widths.items())
              + f" entries per vector ({n_vector} in all); radii {radii} A")
        print(f"output about {gigabytes:.1f} GB (float32, global + {n_radii} radii x 2 planes); "
              f"pristine cell {ms_per_atom:.1f} ms per atom on this CPU; folders "
              f"{sc.relative_path(global_dir)} and {sc.relative_path(out_dir)}")
        return 0
    print(f"{n} structures, {len(skipped)} skipped up front, entries per vector "
          + ", ".join(f"{block} {2 * width}" for block, width in widths.items())
          + f", radii {radii} A, {args.workers} worker(s)", flush=True)

    X_global = {block: np.zeros((n, 2 * width), dtype=np.float32) for block, width in widths.items()}
    X_interface = {block: np.zeros((n, n_radii, 2, 2 * width), dtype=np.float32) for block, width in widths.items()}
    n_interface = np.zeros((n, n_radii, 2, 2), dtype=np.int32)
    meta = [None] * n

    # Largest cells first so the workers stay evenly loaded.
    order = sorted(range(n), key=lambda i: -jobs[i][3].stat().st_size)
    tasks = [(i, jobs[i][3]) for i in order]
    gb_start = time.time()

    def store(i, result, error):
        if error:
            skipped.append((*jobs[i], error))
            return
        for block in blocks:
            X_global[block][i] = result["global"][block]
            X_interface[block][i] = result["interface"][block]
        n_interface[i] = result["n_interface"]
        meta[i] = result["meta"]

    def report(done):
        if done % 500 == 0 or done == n:
            print(f"  {done}/{n} structures, {time.time() - gb_start:.0f} s", flush=True)

    init = (settings, args.chunk, radii, calculator.properties, calculator.property_names)
    if args.workers > 1:
        # featomic has already run in this process, and forking after that can hang the workers.
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
            for done, outcome in enumerate(pool.imap_unordered(_work, tasks, chunksize=2), 1):
                store(*outcome)
                report(done)
    else:
        _init_worker(*init)
        for done, task in enumerate(tasks, 1):
            store(*_work(task))
            report(done)

    keep = np.array([m is not None for m in meta])
    rows = []
    for i in np.flatnonzero(keep):
        plane, sym, thickness, path = jobs[i]
        rows.append({"index": len(rows), "plane": plane, "sym": sym, "thickness": thickness,
                     "family": f"{plane}/{sym}", **meta[i], "source": sc.relative_path(path)})
    if not rows:
        print("every structure failed; nothing written", file=sys.stderr)
        return 1

    labels = ac.label_arrays(calculator.properties, calculator.property_names)
    global_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.write_npz(global_dir / ac.GLOBAL_FILE, **{f"X_{b}": X_global[b][keep] for b in blocks}, **labels,
                 species=np.array(sc.SPECIES))
    per_radius = {}
    for r, radius in enumerate(radii):
        key = sc.radius_key(radius)
        per_radius.update({f"X_{b}_{key}": X_interface[b][keep][:, r] for b in blocks})
        per_radius[f"n_{key}"] = n_interface[keep][:, r]
    sc.write_npz(out_dir / ac.INTERFACE_FILE, **per_radius, **labels, radii=np.array(radii, dtype=float),
                 plane_names=np.array(sc.PLANE_NAMES), species=np.array(sc.SPECIES))
    skipped_rows = [{"plane": plane, "sym": sym, "thickness": thickness, "reason": reason,
                     "source": sc.relative_path(path)} for plane, sym, thickness, path, reason in skipped]
    common = {
        "ace": settings, "featomic_hypers": ac.featomic_hypers(settings), "blocks": blocks, "radii": radii,
        "planes": sorted(args.planes), "thicknesses": sorted(args.thickness), "limit": args.limit,
        "n_features_per_atom": widths, "vector_length": {b: 2 * w for b, w in widths.items()},
        "n_structures": len(rows), "n_atoms": int(sum(int(row["n_atoms"]) for row in rows)),
        "n_skipped": len(skipped_rows), "workers": args.workers, "chunk": args.chunk,
        "seconds": round(time.time() - start, 2), "array_dtype": "float32",
    }
    for folder, arrays in ((global_dir, global_arrays(widths)), (out_dir, interface_arrays(widths, radii))):
        sc.write_csv(folder / "index.csv", rows, index_fields(blocks))
        sc.write_csv(folder / "skipped.csv", skipped_rows, ["plane", "sym", "thickness", "reason", "source"])
        sc.write_json(folder / "settings.json", {**common, "arrays": arrays, **ac.run_metadata()})

    print(f"wrote {len(rows)} structures ({common['n_atoms']} atoms) to {sc.relative_path(global_dir)} and "
          f"{sc.relative_path(out_dir)} in {time.time() - start:.0f} s")
    unexpected = [row for row in skipped_rows if row["reason"] != "empty file"]
    for row in skipped_rows:
        print(f"  skipped {row['source']}: {row['reason']}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
