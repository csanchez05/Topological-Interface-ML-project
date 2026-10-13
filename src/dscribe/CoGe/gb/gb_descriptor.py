#!/usr/bin/env python3
"""Step 1.3: one SOAP vector per grain boundary, global and interface-centred.

Each GB cell holds two interfaces: one in the middle (x = Lx/2) and one at the
cell edge (x = 0, shared with the periodic image). For every structure this
computes per-atom SOAP once and stores:

  X_global     mean over every atom in the cell
  X_interface  mean over atoms within R of an interface plane, for every R in
               --radii, kept separately for the edge and the middle interface
  X_nonbulk    mean over atoms whose SOAP differs from pristine bulk

Outputs go into two folders under calculations/descriptors/dscribe/standard/:
  global/<tag>/           global_soap.npz
  both_interfaces/<tag>/  interface_soap.npz (one X_r<R> array per radius, and X_nonbulk)
each with index.csv (one row per structure, same order), settings.json and skipped.csv.
"""
import os

# One thread per worker process; the parallelism is over structures.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import soap_common as sc

ALL_THICKNESSES = tuple(range(4, 13))
INDEX_FIELDS = ["index", "plane", "sym", "thickness", "family", "n_atoms", "n_Co", "n_Ge",
                "Lx", "Ly", "Lz", "area", "min_pair_distance",
                "n_nonbulk_edge", "n_nonbulk_mid", "max_nonbulk_dist_edge", "max_nonbulk_dist_mid",
                "max_deviation", "max_bulklike_deviation", "source"]
GLOBAL_ARRAYS = {
    "X": "(N, 2F) mean over all atoms",
    "vector layout": "[0:F] Co-centred mean, [F:2F] Ge-centred mean",
    "rows": "same order as index.csv",
}
_STATE = {}


def interface_arrays(radii):
    """Description of interface_soap.npz, written into settings.json."""
    return {
        "X_r<R>": f"(N, plane, 2F) mean over atoms within R A of that plane, one array per R in {radii}",
        "n_r<R>": "(N, plane, species) atoms behind each X_r<R> vector",
        "X_nonbulk": "(N, plane, 2F) mean over atoms whose SOAP differs from bulk",
        "n_nonbulk": "(N, plane, species) atoms behind each X_nonbulk vector",
        "n_nonbulk_within": "(N, radius) disturbed atoms that fall inside each radius, in the order of radii",
        "plane axis": "0 = edge (x = 0), 1 = mid (x = Lx/2); each atom belongs to its nearer plane",
        "vector layout": "[0:F] Co-centred mean, [F:2F] Ge-centred mean",
        "rows": "same order as index.csv",
    }


def collect_structures(gb_root, planes, thicknesses, limit):
    """Returns (jobs, skipped). A job is (plane, sym, thickness, path)."""
    jobs, skipped = [], []
    for plane in planes:
        root = gb_root / f"CoGe_{plane}_GBs" / "Interface"
        sym_dirs = sorted(root.glob("sym[0-9]*"))
        if limit:
            sym_dirs = sym_dirs[:limit]
        for sym_dir in sym_dirs:
            for thickness in thicknesses:
                path = sym_dir / f"{thickness}x" / "POSCAR"
                entry = (plane, sym_dir.name, thickness, path)
                if not path.is_file():
                    skipped.append((*entry, "missing file"))
                elif path.stat().st_size == 0:
                    skipped.append((*entry, "empty file"))
                else:
                    jobs.append(entry)
    return jobs, skipped


def describe_structure(path, soap, radii, bulk_tol, bulk_ref):
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

    X = soap.create(atoms)
    distance, plane = sc.plane_distances(atoms.get_scaled_positions(wrap=True)[:, 0] * lx, lx)
    reference_norm = np.linalg.norm(bulk_ref, axis=1)
    deviation = (np.linalg.norm(X - bulk_ref[np.where(is_co, 0, 1)], axis=1)
                 / reference_norm[np.where(is_co, 0, 1)])
    nonbulk = deviation > bulk_tol

    n_vector = 2 * X.shape[1]
    interface = np.empty((len(radii), 2, n_vector), dtype=np.float32)
    n_interface = np.empty((len(radii), 2, 2), dtype=np.int32)
    n_nonbulk_within = np.empty(len(radii), dtype=np.int32)
    for r, radius in enumerate(radii):
        inside = distance <= radius
        n_nonbulk_within[r] = int((nonbulk & inside).sum())
        for p in (0, 1):
            interface[r, p], n_interface[r, p] = sc.species_mean(X, is_co, inside & (plane == p))
    nonbulk_vector = np.empty((2, n_vector), dtype=np.float32)
    n_nonbulk = np.empty((2, 2), dtype=np.int32)
    for p in (0, 1):
        nonbulk_vector[p], n_nonbulk[p] = sc.species_mean(X, is_co, nonbulk & (plane == p))
    global_vector, counts = sc.species_mean(X, is_co)

    def farthest(p):
        chosen = nonbulk & (plane == p)
        return float(distance[chosen].max()) if chosen.any() else 0.0

    meta = {
        "n_atoms": len(atoms), "n_Co": int(counts[0]), "n_Ge": int(counts[1]),
        "Lx": f"{lengths[0]:.8f}", "Ly": f"{lengths[1]:.8f}", "Lz": f"{lengths[2]:.8f}",
        "area": f"{lengths[1] * lengths[2]:.8f}",
        "min_pair_distance": f"{sc.min_pair_distance(atoms):.6f}",
        "n_nonbulk_edge": int(n_nonbulk[0].sum()), "n_nonbulk_mid": int(n_nonbulk[1].sum()),
        "max_nonbulk_dist_edge": f"{farthest(0):.4f}", "max_nonbulk_dist_mid": f"{farthest(1):.4f}",
        "max_deviation": f"{deviation.max():.6e}",
        "max_bulklike_deviation": f"{deviation[~nonbulk].max() if (~nonbulk).any() else 0.0:.6e}",
    }
    return {"global": global_vector.astype(np.float32), "interface": interface, "n_interface": n_interface,
            "nonbulk": nonbulk_vector, "n_nonbulk": n_nonbulk, "n_nonbulk_within": n_nonbulk_within,
            "meta": meta}


def _init_worker(settings, radii, bulk_tol, bulk_ref):
    _STATE.update(soap=sc.make_soap(settings), radii=radii, bulk_tol=bulk_tol, bulk_ref=bulk_ref)


def _work(task):
    i, path = task
    try:
        return i, describe_structure(path, _STATE["soap"], _STATE["radii"], _STATE["bulk_tol"],
                                     _STATE["bulk_ref"]), None
    except Exception as exc:  # recorded in skipped.csv and reported at the end
        return i, None, f"{type(exc).__name__}: {exc}"


def default_name(args, settings):
    """The run's folder name: the SOAP tag, with a suffix when the catalogue was filtered."""
    name = sc.soap_tag(settings)
    if set(args.planes) != set(sc.PLANES):
        name += "_p" + "-".join(sorted(args.planes))
    if set(args.thickness) != set(ALL_THICKNESSES):
        name += "_t" + "-".join(str(t) for t in sorted(args.thickness))
    if args.limit:
        name += f"_lim{args.limit}"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_soap_arguments(parser)
    parser.add_argument("--radii", type=float, nargs="+", default=[4.0, 6.0, 8.0, 10.0],
                        help="interface radii in A: atoms within this distance of an interface plane, "
                             "on both sides (default 4 6 8 10)")
    parser.add_argument("--bulk-tol", type=float, default=1e-3,
                        help="an atom counts as disturbed when its SOAP differs from bulk by more than "
                             "this relative amount (default 1e-3)")
    parser.add_argument("--gb-root", type=Path, default=sc.GB_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--thickness", type=int, nargs="+", default=list(ALL_THICKNESSES),
                        choices=ALL_THICKNESSES, metavar="N", help="thicknesses to include, 4..12 (default all)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (testing)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="worker processes (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--out-root", type=Path, default=sc.STANDARD_ROOT,
                        help="folder holding global/ and both_interfaces/ "
                             "(default calculations/descriptors/dscribe/standard)")
    parser.add_argument("--name", default=None,
                        help="run folder name inside each region (default: the SOAP tag, with a suffix when filtered)")
    parser.add_argument("--csv", type=float, default=None, metavar="RADIUS",
                        help="also write the interface vectors for this radius as a plain CSV "
                             "(both_interfaces/<tag>/interface_r<RADIUS>.csv)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args()

    settings = sc.soap_settings(args)
    radii = sorted(set(args.radii))
    if args.csv is not None and not any(np.isclose(args.csv, radii)):
        parser.error(f"--csv {args.csv:g} is not one of --radii {radii}")
    name = args.name or default_name(args, settings)
    global_dir, out_dir = args.out_root / "global" / name, args.out_root / "both_interfaces" / name
    if (out_dir / sc.INTERFACE_FILE).exists() and not args.overwrite:
        print(f"{out_dir / sc.INTERFACE_FILE} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    jobs, skipped = collect_structures(args.gb_root, args.planes, sorted(args.thickness), args.limit)
    if not jobs:
        print("no structures found", file=sys.stderr)
        return 2
    soap = sc.make_soap(settings)
    n_vector = 2 * soap.get_number_of_features()
    bulk_ref = sc.bulk_reference(soap)
    n, n_radii = len(jobs), len(radii)
    print(f"{n} structures, {len(skipped)} skipped up front, {n_vector} entries per vector, "
          f"radii {radii} A, {args.workers} worker(s)", flush=True)

    X_global = np.zeros((n, n_vector), dtype=np.float32)
    X_interface = np.zeros((n, n_radii, 2, n_vector), dtype=np.float32)
    n_interface = np.zeros((n, n_radii, 2, 2), dtype=np.int32)
    X_nonbulk = np.zeros((n, 2, n_vector), dtype=np.float32)
    n_nonbulk = np.zeros((n, 2, 2), dtype=np.int32)
    n_nonbulk_within = np.zeros((n, n_radii), dtype=np.int32)
    meta = [None] * n

    # Largest cells first so the workers stay evenly loaded.
    order = sorted(range(n), key=lambda i: -jobs[i][3].stat().st_size)
    tasks = [(i, jobs[i][3]) for i in order]
    start = time.time()

    def store(i, result, error):
        if error:
            skipped.append((*jobs[i], error))
            return
        X_global[i], X_interface[i], n_interface[i] = result["global"], result["interface"], result["n_interface"]
        X_nonbulk[i], n_nonbulk[i] = result["nonbulk"], result["n_nonbulk"]
        n_nonbulk_within[i] = result["n_nonbulk_within"]
        meta[i] = result["meta"]

    def report(done):
        if done % 500 == 0 or done == n:
            print(f"  {done}/{n} structures, {time.time() - start:.0f} s", flush=True)

    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=(settings, radii, args.bulk_tol, bulk_ref)) as pool:
            for done, outcome in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
                store(*outcome)
                report(done)
    else:
        _init_worker(settings, radii, args.bulk_tol, bulk_ref)
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

    sc.write_gb_arrays(global_dir, out_dir, X_global[keep], X_interface[keep], n_interface[keep],
                       X_nonbulk[keep], n_nonbulk[keep], n_nonbulk_within[keep], radii)
    skipped_rows = [{"plane": plane, "sym": sym, "thickness": thickness, "reason": reason,
                     "source": sc.relative_path(path)} for plane, sym, thickness, path, reason in skipped]
    common = {
        "soap": settings, "radii": radii, "bulk_tol": args.bulk_tol,
        "planes": sorted(args.planes), "thicknesses": sorted(args.thickness), "limit": args.limit,
        "n_features_per_atom": n_vector // 2, "vector_length": n_vector,
        "n_structures": len(rows), "n_skipped": len(skipped_rows), "workers": args.workers,
        "seconds": round(time.time() - start, 2), "array_dtype": "float32",
    }
    for folder, arrays in ((global_dir, GLOBAL_ARRAYS), (out_dir, interface_arrays(radii))):
        sc.write_csv(folder / "index.csv", rows, INDEX_FIELDS)
        sc.write_csv(folder / "skipped.csv", skipped_rows, ["plane", "sym", "thickness", "reason", "source"])
        sc.write_json(folder / "settings.json", {**common, "arrays": arrays, **sc.run_metadata()})
    if args.csv is not None:
        r = int(np.flatnonzero(np.isclose(radii, args.csv))[0])
        combined = sc.combine_planes(X_interface[keep][:, r], n_interface[keep][:, r])
        csv_path = out_dir / f"interface_r{args.csv:g}.csv"
        tmp = csv_path.with_name(f".{csv_path.name}.tmp")
        with open(tmp, "w") as handle:
            handle.write("family,plane,sym,thickness," + ",".join(f"f{j}" for j in range(n_vector)) + "\n")
            for row, vector in zip(rows, combined):
                handle.write(f"{row['family']},{row['plane']},{row['sym']},{row['thickness']},"
                             + ",".join(f"{v:.8e}" for v in vector) + "\n")
        os.replace(tmp, csv_path)

    print(f"wrote {len(rows)} structures to {global_dir} and {out_dir} in {time.time() - start:.0f} s")
    unexpected = [row for row in skipped_rows if row["reason"] != "empty file"]
    for row in skipped_rows:
        print(f"  skipped {row['source']}: {row['reason']}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
