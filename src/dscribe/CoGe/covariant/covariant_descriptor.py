#!/usr/bin/env python3
"""Covariant (lambda-) SOAP for the bulk cells and every grain boundary.

Standard SOAP is a rotation invariant: it gives the same numbers however the
structure is turned. lambda-SOAP keeps some of the direction information:

  invariant     lambda = 0, sigma = +1   rotation-invariant, like standard SOAP
  polar         lambda = 1, sigma = +1   turns like an ordinary vector, flips under inversion
  axial         lambda = 1, sigma = -1   turns like a vector, unchanged by inversion (as a spin does)
  tensor        lambda = 2, sigma = +1   turns like a symmetric traceless rank-2 tensor (5
                                         components), unchanged by inversion
  pseudotensor  lambda = 2, sigma = -1   turns like tensor, flips under inversion

Under inversion a block is multiplied by sigma * (-1)^lambda.
(lambda = 0, sigma = -1 is zero for a power spectrum, so it is not stored.)
Computed with featomic's EquivariantPowerSpectrum on a GTO spherical expansion.
--blocks chooses which blocks to compute (default all five; the settings-grid runs use the
four with lambda >= 1).

The averaging is the same as for standard SOAP: [Co-centred mean | Ge-centred mean]
over all atoms (global) and over atoms within each radius of an interface plane,
edge and mid kept apart (both_interfaces). The three lambda = 1 components are stored
in Cartesian order x, y, z; the five lambda = 2 components in featomic's order m = -2 ... 2
(real spherical harmonics).

Outputs, under calculations/descriptors/dscribe/covariant/:
  global/<tag>/           covariant_soap.npz, bulk_covariant.npz, index.csv, bulk_index.csv, settings.json
  both_interfaces/<tag>/  covariant_soap.npz (one set of arrays per radius), index.csv, settings.json
Then: python -B src/dscribe/CoGe/covariant/check_covariant.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gb"))
sys.path.insert(0, str(HERE.parent / "bulk"))

import numpy as np

import soap_common as sc
from bulk_descriptor import collect_cells
from gb_descriptor import ALL_THICKNESSES, collect_structures

# name: (lambda, sigma)
BLOCKS = {"invariant": (0, 1), "polar": (1, 1), "axial": (1, -1), "tensor": (2, 1), "pseudotensor": (2, -1)}
ATOMIC_NUMBERS = {"Co": 27, "Ge": 32}
# featomic's real spherical harmonics for lambda = 1 are ordered m = -1, 0, 1, i.e. y, z, x.
TO_XYZ = [2, 0, 1]
_STATE = {}


def inversion_sign(name):
    """Factor a block picks up under r -> -r: sigma * (-1)^lambda."""
    o3_lambda, o3_sigma = BLOCKS[name]
    return float(o3_sigma * (-1) ** o3_lambda)


def index_fields(blocks):
    return (["index", "plane", "sym", "thickness", "family", "n_atoms", "n_Co", "n_Ge", "Lx", "Ly", "Lz"]
            + [f"atom_norm_{name}" for name in blocks] + ["source"])


def bulk_fields(blocks):
    return ["index", "kind", "plane", "sym", "n_atoms"] + [f"atom_norm_{name}" for name in blocks] + ["source"]


# Columns of index.csv and bulk_index.csv for a run with every block.
INDEX_FIELDS = index_fields(BLOCKS)
BULK_FIELDS = bulk_fields(BLOCKS)


def hypers(settings):
    return {"cutoff": {"radius": settings["r_cut"],
                       "smoothing": {"type": "ShiftedCosine", "width": settings["smoothing"]}},
            "density": {"type": "Gaussian", "width": settings["sigma"]},
            "basis": {"type": "TensorProduct", "max_angular": settings["l_max"],
                      "radial": {"type": "Gto", "max_radial": settings["n_max"] - 1}}}


def make_calculator(settings, blocks=None):
    """(calculator, keys) for the chosen blocks (default all); keys selects them from featomic's output."""
    import featomic
    from featomic.clebsch_gordan import EquivariantPowerSpectrum
    from metatensor import Labels

    blocks = list(blocks or BLOCKS)
    calculator = EquivariantPowerSpectrum(featomic.SphericalExpansion(**hypers(settings)),
                                          neighbor_types=[ATOMIC_NUMBERS[s] for s in sc.SPECIES])
    keys = Labels(["o3_lambda", "o3_sigma"], np.array([BLOCKS[name] for name in blocks], dtype=np.int32))
    return calculator, keys


def chosen_blocks(keys):
    """Block names in the order of the selected keys."""
    by_key = {value: name for name, value in BLOCKS.items()}
    return [by_key[(int(lam), int(sig))] for lam, sig in np.asarray(keys.values)]


def tag(settings):
    """Same naming as standard SOAP: n is the number of radial functions."""
    return sc.soap_tag({**settings, "rbf": "gto", "weighting": None})


def per_atom_blocks(atoms, calculator, keys):
    """{block name: (n_atoms, components, P)} for the blocks selected by keys, lambda = 1
    components in x, y, z order, and the property labels of each block."""
    tensor = calculator.compute(atoms, selected_keys=keys, neighbors_to_properties=True)
    blocks, properties = {}, {}
    for name in chosen_blocks(keys):
        o3_lambda, o3_sigma = BLOCKS[name]
        values = None
        for species in sc.SPECIES:
            block = tensor.block(o3_lambda=o3_lambda, o3_sigma=o3_sigma, center_type=ATOMIC_NUMBERS[species])
            if values is None:
                values = np.zeros((len(atoms), 2 * o3_lambda + 1, block.values.shape[-1]))
                properties[name] = np.asarray(block.properties.values)
            values[np.asarray(block.samples.column("atom"))] = block.values
        blocks[name] = values[:, TO_XYZ] if o3_lambda == 1 else values
    return blocks, properties


def split_mean(values, is_co, mask=None):
    """values (n_atoms, C, P) -> (C, 2P) as [Co mean | Ge mean] for each component, and the counts."""
    n_atoms, n_comp, n_prop = values.shape
    vector, counts = sc.species_mean(values.reshape(n_atoms, -1), is_co, mask)
    return vector.reshape(2, n_comp, n_prop).transpose(1, 0, 2).reshape(n_comp, 2 * n_prop), counts


def atom_norms(blocks):
    """Mean length of one atom's block vector, to judge how small an average is."""
    return {f"atom_norm_{name}": f"{np.linalg.norm(values.reshape(len(values), -1), axis=1).mean():.6e}"
            for name, values in blocks.items()}


def describe_structure(path, calculator, keys, radii):
    atoms, _ = sc.read_structure(path)
    cell = atoms.cell.array
    lengths = np.diag(cell).copy()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6:
        raise ValueError("cell is not orthogonal")
    if lengths[0] < lengths[1:].max():
        raise ValueError("x is not the longest axis, so the interface normal is unclear")
    lx = lengths[0]
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    blocks, _ = per_atom_blocks(atoms, calculator, keys)
    distance, plane = sc.plane_distances(atoms.get_scaled_positions(wrap=True)[:, 0] * lx, lx)
    result = {"global": {}, "interface": {}, "n_interface": np.zeros((len(radii), 2, 2), dtype=np.int32)}
    for name, values in blocks.items():
        result["global"][name] = split_mean(values, is_co)[0].astype(np.float32)
        out = np.empty((len(radii), 2) + result["global"][name].shape, dtype=np.float32)
        for r, radius in enumerate(radii):
            for p in (0, 1):
                out[r, p], result["n_interface"][r, p] = split_mean(values, is_co, (distance <= radius) & (plane == p))
        result["interface"][name] = out
    result["meta"] = {"n_atoms": len(atoms), "n_Co": int(is_co.sum()), "n_Ge": int((~is_co).sum()),
                      "Lx": f"{lengths[0]:.8f}", "Ly": f"{lengths[1]:.8f}", "Lz": f"{lengths[2]:.8f}",
                      **atom_norms(blocks)}
    return result


def describe_bulk(path, calculator, keys):
    """Whole-cell average of every block of one bulk cell (float64) and its atom norms."""
    atoms, _ = sc.read_structure(path)
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    blocks, _ = per_atom_blocks(atoms, calculator, keys)
    return {"vectors": {name: split_mean(values, is_co)[0] for name, values in blocks.items()},
            "meta": {"n_atoms": len(atoms), **atom_norms(blocks)}}


def _init_worker(settings, radii, blocks=None):
    calculator, keys = make_calculator(settings, blocks)
    _STATE.update(calculator=calculator, keys=keys, radii=radii)


def _work(task):
    i, path = task
    try:
        if i < 0:  # bulk cell number -1 - i
            return i, describe_bulk(path, _STATE["calculator"], _STATE["keys"]), None
        return i, describe_structure(path, _STATE["calculator"], _STATE["keys"], _STATE["radii"]), None
    except Exception as exc:  # recorded in skipped.csv and reported at the end
        return i, None, f"{type(exc).__name__}: {exc}"


def bulk_cells(bulk_root, planes, limit=0):
    """collect_cells(), with only the first `limit` transformed cells per plane when limit is set."""
    cells = collect_cells(bulk_root, planes)
    if not limit:
        return cells
    seen, kept = {}, []
    for cell in cells:
        if cell[0] == "transformed":
            seen[cell[1]] = seen.get(cell[1], 0) + 1
            if seen[cell[1]] > limit:
                continue
        kept.append(cell)
    return kept


def block_arrays(radii=None, blocks=None):
    """Description of covariant_soap.npz, written into settings.json."""
    planes = "plane, " if radii else ""
    suffix = "_r<R>" if radii else ""
    shapes = {"invariant": f"(N, {planes}1, 2P) lambda 0, sigma +1",
              "polar": f"(N, {planes}3, 2P) lambda 1, sigma +1; component axis x, y, z",
              "axial": f"(N, {planes}3, 2P) lambda 1, sigma -1; component axis x, y, z",
              "tensor": f"(N, {planes}5, 2P) lambda 2, sigma +1; component axis m = -2 ... 2",
              "pseudotensor": f"(N, {planes}5, 2P) lambda 2, sigma -1; component axis m = -2 ... 2"}
    text = {f"X_{name}{suffix}": shapes[name] for name in (blocks or BLOCKS)}
    text.update({
        "properties_<block>": "(P, 6) labels of the last axis within one species half: "
                              "l_1, l_2, neighbor_1_type, n_1, neighbor_2_type, n_2",
        "vector layout": "last axis [0:P] Co-centred mean, [P:2P] Ge-centred mean (P differs between blocks)",
        "rows": "same order as index.csv",
    })
    if radii:
        text["n_r<R>"] = "(N, plane, species) atoms behind each vector"
        text["plane axis"] = "0 = edge (x = 0), 1 = mid (x = Lx/2); each atom belongs to its nearer plane"
        text["radii"] = radii
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r-cut", type=float, default=6.0, help="cutoff radius in A (default 6)")
    parser.add_argument("--n-max", type=int, default=8, help="radial functions (default 8, as standard SOAP)")
    parser.add_argument("--l-max", type=int, default=6, help="largest angular order of the density (default 6)")
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian width of each atom in A (default 0.5)")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="width of the shifted-cosine cutoff smoothing in A (default 0.5)")
    parser.add_argument("--radii", type=float, nargs="+", default=[8.0], help="interface radii in A (default 8)")
    parser.add_argument("--gb-root", type=Path, default=sc.GB_ROOT)
    parser.add_argument("--bulk-root", type=Path, default=sc.BULK_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--thickness", type=int, nargs="+", default=list(ALL_THICKNESSES),
                        choices=ALL_THICKNESSES, metavar="N", help="thicknesses to include, 4..12 (default all)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sym folders per plane (testing)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--out-root", type=Path, default=sc.COVARIANT_ROOT,
                        help="folder holding global/ and both_interfaces/ "
                             "(default calculations/descriptors/dscribe/covariant)")
    parser.add_argument("--name", default=None, help="run folder name (default: the tag, e.g. rc6.0_n8_l6_s0.5)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--blocks", nargs="+", default=list(BLOCKS), choices=list(BLOCKS),
                        help=f"blocks to compute (default all: {' '.join(BLOCKS)})")
    parser.add_argument("--dry-run", action="store_true", help="count the structures and stop")
    args = parser.parse_args()

    settings = {"r_cut": args.r_cut, "n_max": args.n_max, "l_max": args.l_max, "sigma": args.sigma,
                "smoothing": args.smoothing}
    blocks = [name for name in BLOCKS if name in args.blocks]
    radii = sorted(set(args.radii))
    name = args.name or tag(settings) + (f"_lim{args.limit}" if args.limit else "")
    global_dir, out_dir = args.out_root / "global" / name, args.out_root / "both_interfaces" / name
    if (out_dir / "covariant_soap.npz").exists() and not args.overwrite:
        print(f"{out_dir / 'covariant_soap.npz'} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    start = time.time()
    cells = bulk_cells(args.bulk_root, args.planes, args.limit)
    jobs, skipped = collect_structures(args.gb_root, args.planes, sorted(args.thickness), args.limit)
    if args.dry_run:
        print(f"{len(cells)} bulk cells, {len(jobs)} structures, {len(skipped)} skipped up front, "
              f"blocks {blocks}, folders {global_dir} and {out_dir}")
        return 0
    calculator, keys = make_calculator(settings, blocks)
    probe = describe_structure(jobs[0][3], calculator, keys, radii)
    _, properties = per_atom_blocks(sc.read_structure(jobs[0][3])[0], calculator, keys)
    n, n_radii = len(jobs), len(radii)
    shapes = {block: probe["global"][block].shape for block in blocks}
    print(f"{len(cells)} bulk cells, {n} structures, {len(skipped)} skipped up front, blocks "
          + ", ".join(f"{block} {shape}" for block, shape in shapes.items())
          + f", radii {radii} A, {args.workers} worker(s)", flush=True)
    X_global = {block: np.zeros((n,) + shape, dtype=np.float32) for block, shape in shapes.items()}
    X_interface = {block: np.zeros((n, n_radii, 2) + shape, dtype=np.float32) for block, shape in shapes.items()}
    n_interface = np.zeros((n, n_radii, 2, 2), dtype=np.int32)
    meta = [None] * n
    bulk = {block: np.zeros((len(cells),) + shapes[block]) for block in blocks}
    bulk_meta, bulk_errors = [None] * len(cells), []
    order = sorted(range(n), key=lambda i: -jobs[i][3].stat().st_size)
    # Bulk cells (numbered -1, -2, ...) go first through the same workers.
    tasks = [(-1 - k, cell[3]) for k, cell in enumerate(cells)] + [(i, jobs[i][3]) for i in order]
    gb_start = time.time()

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
            print(f"  {len(cells)} bulk cells, {time.time() - gb_start:.0f} s", flush=True)
        elif done > len(cells) and ((done - len(cells)) % 1000 == 0 or done == len(tasks)):
            print(f"  {done - len(cells)}/{n} structures, {time.time() - gb_start:.0f} s", flush=True)

    if args.workers > 1:
        # featomic has already run in this process, and forking after that can hang the workers.
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker, initargs=(settings, radii, blocks)) as pool:
            for done, outcome in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
                store(*outcome)
                report(done)
    else:
        _init_worker(settings, radii, blocks)
        for done, task in enumerate(tasks, 1):
            store(*_work(task))
            report(done)
    if bulk_errors:
        print("bulk cells failed; nothing written:\n  " + "\n  ".join(bulk_errors), file=sys.stderr)
        return 1
    bulk_rows = [{"index": k, "kind": kind, "plane": plane, "sym": sym, **bulk_meta[k], "source": sc.relative_path(path)}
                 for k, (kind, plane, sym, path) in enumerate(cells)]

    keep = np.array([m is not None for m in meta])
    rows = []
    for i in np.flatnonzero(keep):
        plane, sym, thickness, path = jobs[i]
        rows.append({"index": len(rows), "plane": plane, "sym": sym, "thickness": thickness,
                     "family": f"{plane}/{sym}", **meta[i], "source": sc.relative_path(path)})
    if not rows:
        print("every structure failed; nothing written", file=sys.stderr)
        return 1
    labels = {f"properties_{block}": properties[block] for block in blocks}
    common = {"blocks": {block: {"lambda": BLOCKS[block][0], "sigma": BLOCKS[block][1]} for block in blocks},
              "species": list(sc.SPECIES), "atomic_numbers": ATOMIC_NUMBERS}
    global_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.write_npz(global_dir / "covariant_soap.npz", **{f"X_{b}": X_global[b][keep] for b in blocks}, **labels,
                 species=np.array(sc.SPECIES))
    sc.write_npz(global_dir / "bulk_covariant.npz", **{f"X_{b}": bulk[b] for b in blocks}, **labels,
                 species=np.array(sc.SPECIES))
    sc.write_csv(global_dir / "bulk_index.csv", bulk_rows, bulk_fields(blocks))
    per_radius = {}
    for r, radius in enumerate(radii):
        key = sc.radius_key(radius)
        per_radius.update({f"X_{b}_{key}": X_interface[b][keep][:, r] for b in blocks})
        per_radius[f"n_{key}"] = n_interface[keep][:, r]
    sc.write_npz(out_dir / "covariant_soap.npz", **per_radius, **labels, radii=np.array(radii, dtype=float),
                 plane_names=np.array(sc.PLANE_NAMES), species=np.array(sc.SPECIES))
    skipped_rows = [{"plane": plane, "sym": sym, "thickness": thickness, "reason": reason,
                     "source": sc.relative_path(path)} for plane, sym, thickness, path, reason in skipped]
    run = {"settings": settings, "featomic_hypers": hypers(settings), **common,
           "planes": sorted(args.planes), "thicknesses": sorted(args.thickness), "limit": args.limit,
           "n_structures": len(rows), "n_skipped": len(skipped_rows), "n_bulk_cells": len(bulk_rows),
           "workers": args.workers, "seconds": round(time.time() - start, 2), "array_dtype": "float32"}
    for folder, arrays in ((global_dir, block_arrays(None, blocks)), (out_dir, block_arrays(radii, blocks))):
        sc.write_csv(folder / "index.csv", rows, index_fields(blocks))
        sc.write_csv(folder / "skipped.csv", skipped_rows, ["plane", "sym", "thickness", "reason", "source"])
        sc.write_json(folder / "settings.json", {**run, "arrays": arrays, **sc.run_metadata(),
                                                 "featomic": __import__("featomic").__version__})
    print(f"wrote {len(rows)} structures to {global_dir} and {out_dir} in {time.time() - start:.0f} s")
    unexpected = [row for row in skipped_rows if row["reason"] != "empty file"]
    for row in skipped_rows:
        print(f"  skipped {row['source']}: {row['reason']}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
