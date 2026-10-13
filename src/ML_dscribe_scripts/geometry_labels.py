#!/usr/bin/env python3
"""Free labels that depend on the shift: the geometry of each interface, read from the atoms.

The grain class only says how grain B is turned. GBs of one class differ by how grain B is
shifted, and so do their energies (Stage 2) and spin textures (Stage 3). These labels measure
the shift's effect directly from the unrelaxed GB cells, with no DFT. They serve as targets for
the Stage 1.4 KRR test (krr_classify.py --regression), the selection benchmark and the fifth study.

Each cell holds two interfaces: edge (x = 0) and mid (x = Lx/2). Every atom belongs to its nearer
plane, as in the descriptors, and only atoms within --radius of that plane count:

  d_min          shortest distance from one of these atoms to any atom (A); 2.3975 A in bulk
  n_close_pairs  pairs closer than --close (1.5 A); a pair counts half at each of its two atoms
  n_close        n_close_pairs per nm^2 of interface (area = Ly * Lz)
  log1p_n_close  log(1 + n_close): the regression target, since most values are small and a few large
  cn_excess      sum over these atoms of (neighbours within the cutoff - the bulk number), per nm^2;
                 negative when atoms lost neighbours
  cn_defect      the same sum of |neighbours - bulk number|, per nm^2

Region "cell" is both interfaces together: d_min the smaller of the two, n_close_pairs the sum,
the per-nm^2 labels over both areas (the mean of the two).

Neighbour cutoff: the middle of the widest gap between the pristine crystal's neighbour shells
below 4 A (3.33 A: 13 neighbours, 7 Co-Ge at 2.40-2.58 A and 6 of the same species at 2.84-2.87 A).
--cn-cutoff overrides it.

Checks, written to the json; a failed check stops the run before anything is written:
  - every transformed bulk cell (Bulk/symNNNNN/POSCAR) gives no close pair, the bulk
    neighbour numbers on every atom and the pristine d_min
  - the NO_BOUNDARY families (ml_common.py; plain bulk) give the same
  - each family's labels agree at every thickness (tolerance --thickness-tol)
Distances are compared to POSITION_TOL = 1e-3 A: the catalogue POSCARs give y and z to 3-4
decimals (0.7867 for 0.78667), so atoms sit up to about 4e-4 A off (the 111 bulk cells give
d_min 2.3970 A against the pristine 2.3975 A).

The json also reports how much of each label's spread plane and grain class explain, over the
distinct GBs at 8x without the NO_BOUNDARY pair: a share near 1 would mean the label adds
nothing the grain class does not already carry.

Outputs, in calculations/ML_dscribe/CoGe/labels/ (refused if present, unless --overwrite):
  geometry_labels.csv   one row per family x thickness x region (edge, mid, cell)
  geometry_labels.json  settings, pristine shells, checks, variance shares
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import ml_common as mc
from ml_common import sc

OUT_DIR = mc.LABELS_DIR
CSV_NAME = "geometry_labels.csv"
JSON_NAME = "geometry_labels.json"
THICKNESSES = tuple(range(4, 13))
REGIONS = ("edge", "mid", "cell")
LABELS = ("d_min", "n_close_pairs", "n_close", "log1p_n_close", "cn_excess", "cn_defect")
# The labels used as continuous targets: the overlap count enters as log(1 + n_close).
TARGETS = ("d_min", "log1p_n_close", "cn_excess", "cn_defect")
REGION_OF_PLANES = {"both": "cell", "edge": "edge", "mid": "mid"}
SHELL_CEILING = 4.0  # A; the pristine shells are read up to here
POSITION_TOL = 1e-3  # A; catalogue coordinates are rounded to 3-4 decimals (up to ~4e-4 A, see the docstring)


# --------------------------------------------------------------------------- pieces

def pristine_shells(atoms, cutoff=None, ceiling=SHELL_CEILING, decimals=4):
    """Neighbour shells of a perfect crystal: (shells, cutoff, neighbours per species).

    shells: list of {distance, species pairs} up to `ceiling`.
    cutoff: as given, or else the middle of the widest gap between consecutive shell distances.
    neighbours per species: {species: neighbours within the cutoff}; the same for every atom.
    """
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, ceiling)
    species = np.array(atoms.get_chemical_symbols())
    distances = np.unique(np.round(d, decimals))
    gaps = np.diff(distances)
    widest = int(np.argmax(gaps))
    if cutoff is None:
        cutoff = float(0.5 * (distances[widest] + distances[widest + 1]))
    shells = []
    for value in distances:
        at = np.round(d, decimals) == value
        shells.append({"distance": float(value), "pairs": sorted({f"{species[a]}-{species[b]}" for a, b in
                                                                   zip(i[at], j[at])})})
    neighbours = {}
    for name in sorted(set(species)):
        counts = np.bincount(i[d < cutoff], minlength=len(atoms))[species == name]
        if len(set(counts.tolist())) != 1:
            raise ValueError(f"{name} atoms of the pristine cell have different neighbour numbers {sorted(set(counts))}")
        neighbours[name] = int(counts[0])
    return shells, cutoff, neighbours


def structure_labels(atoms, radius, close, cutoff, neighbours):
    """{region: {label: value, n_atoms: ...}} for one GB cell (x the interface normal)."""
    from ase.neighborlist import neighbor_list

    cell = atoms.cell.array
    lengths = np.diag(cell).copy()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6:
        raise ValueError("cell is not orthogonal")
    lx, area_nm2 = lengths[0], lengths[1] * lengths[2] / 100.0
    distance, plane = sc.plane_distances(atoms.get_scaled_positions(wrap=True)[:, 0] * lx, lx)
    species = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, max(cutoff, close))
    n = len(atoms)
    nearest = np.full(n, np.inf)
    np.minimum.at(nearest, i, d)
    change = np.bincount(i[d < cutoff], minlength=n) - np.array([neighbours[s] for s in species])
    close_at = np.bincount(i[d < close], minlength=n)  # each close pair counted at both atoms
    out, total = {}, {"n_atoms": 0, "n_close_pairs": 0.0, "excess": 0.0, "defect": 0.0, "d_min": np.inf}
    for p, region in enumerate(sc.PLANE_NAMES):
        inside = (plane == p) & (distance <= radius)
        pairs = 0.5 * float(close_at[inside].sum())
        excess, defect = float(change[inside].sum()), float(np.abs(change[inside]).sum())
        out[region] = region_labels(int(inside.sum()), float(nearest[inside].min()), pairs, excess, defect, area_nm2)
        total["n_atoms"] += int(inside.sum())
        total["n_close_pairs"] += pairs
        total["excess"] += excess
        total["defect"] += defect
        total["d_min"] = min(total["d_min"], out[region]["d_min"])
    out["cell"] = region_labels(total["n_atoms"], total["d_min"], total["n_close_pairs"], total["excess"],
                                total["defect"], 2.0 * area_nm2)
    for region in out:
        out[region]["area_nm2"] = area_nm2
    return out


def region_labels(n_atoms, d_min, pairs, excess, defect, area_nm2):
    n_close = pairs / area_nm2
    return {"n_atoms": n_atoms, "d_min": d_min, "n_close_pairs": pairs, "n_close": n_close,
            "log1p_n_close": float(np.log1p(n_close)), "cn_excess": excess / area_nm2,
            "cn_defect": defect / area_nm2}


def bulk_cell_check(atoms, close, cutoff, neighbours, d_bulk, tol=POSITION_TOL):
    """None if a bulk cell has the pristine geometry, else what differs."""
    from ase.neighborlist import neighbor_list

    i, d = neighbor_list("id", atoms, max(cutoff, close))
    species = np.array(atoms.get_chemical_symbols())
    counts = np.bincount(i[d < cutoff], minlength=len(atoms))
    wrong = counts != np.array([neighbours[s] for s in species])
    problems = []
    if wrong.any():
        problems.append(f"{int(wrong.sum())} atoms with a non-bulk neighbour number")
    if (d < close).any():
        problems.append(f"{int((d < close).sum() // 2)} close pairs")
    if abs(float(d.min()) - d_bulk) > tol:
        problems.append(f"d_min {d.min():.4f} instead of {d_bulk:.4f}")
    return "; ".join(problems) or None


def variance_shares(values, groups):
    """Share of the spread of values explained by the group means (eta squared)."""
    values = np.asarray(values, dtype=float)
    total = float(np.square(values - values.mean()).sum())
    if total == 0:
        return None
    between = 0.0
    for g in np.unique(groups):
        part = values[groups == g]
        between += len(part) * (part.mean() - values.mean()) ** 2
    return float(between / total)


def thickness_spread(table):
    """Largest change of each label between thicknesses of one family and region."""
    spread = table.groupby(["family", "region"])[list(LABELS)].agg(lambda v: float(v.max() - v.min()))
    return {label: float(spread[label].max()) for label in LABELS}, spread


def load_geometry(index, planes="both", path=None):
    """Geometry labels for the rows of a vector index (columns family, thickness), in row order.

    planes: both -> region "cell", edge or mid -> that interface. index may also carry a
    region column (edge/mid per row), which then takes precedence.
    """
    import pandas as pd

    table = pd.read_csv(path or OUT_DIR / CSV_NAME, dtype={"plane": str})
    keys = ["family", "thickness"]
    wanted = index[keys + (["region"] if "region" in index else [])].copy()
    if "region" not in wanted:
        wanted["region"] = REGION_OF_PLANES[planes]
    rows = wanted.merge(table.drop(columns=["plane", "sym"]), on=keys + ["region"], how="left", validate="many_to_one")
    missing = rows[list(TARGETS)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"no geometry labels for {int(missing.sum())} rows, e.g. "
                         f"{rows.loc[missing, keys + ['region']].head(3).to_dict('records')}; run geometry_labels.py")
    return rows


# --------------------------------------------------------------------------- jobs

def collect(planes, thicknesses, limit):
    """GB cells and bulk cells to read: (gb jobs, bulk jobs, skipped)."""
    gb, bulk, skipped = [], [], []
    for plane in planes:
        root = sc.GB_ROOT / f"CoGe_{plane}_GBs"
        sym_dirs = sorted((root / "Interface").glob("sym[0-9]*"))[:limit or None]
        for sym_dir in sym_dirs:
            bulk.append((plane, sym_dir.name, root / "Bulk" / sym_dir.name / "POSCAR"))
            for thickness in thicknesses:
                path = sym_dir / f"{thickness}x" / "POSCAR"
                if not path.is_file():
                    skipped.append({"family": f"{plane}/{sym_dir.name}", "thickness": thickness, "reason": "missing file"})
                elif path.stat().st_size == 0:
                    skipped.append({"family": f"{plane}/{sym_dir.name}", "thickness": thickness, "reason": "empty file"})
                else:
                    gb.append((plane, sym_dir.name, thickness, path))
    return gb, bulk, skipped


def gb_task(jobs, radius, close, cutoff, neighbours):
    rows = []
    for plane, sym, thickness, path in jobs:
        atoms, _ = sc.read_structure(path)
        for region, values in structure_labels(atoms, radius, close, cutoff, neighbours).items():
            rows.append({"family": f"{plane}/{sym}", "plane": plane, "sym": sym, "thickness": thickness,
                         "region": region} | values)
    return rows


def bulk_task(jobs, close, cutoff, neighbours, d_bulk):
    problems = {}
    for plane, sym, path in jobs:
        atoms, _ = sc.read_structure(path)
        problem = bulk_cell_check(atoms, close, cutoff, neighbours, d_bulk)
        if problem:
            problems[f"{plane}/{sym}"] = problem
    return problems


def chunks(items, n):
    n = max(1, min(n, len(items)))
    return [items[k::n] for k in range(n)]


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius", type=float, default=8.0, help="atoms within this distance of a plane count (A, default 8)")
    parser.add_argument("--close", type=float, default=1.5, help="pairs closer than this are overlaps (A, default 1.5)")
    parser.add_argument("--cn-cutoff", type=float, default=None,
                        help="neighbour cutoff in A (default: the middle of the widest pristine shell gap)")
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--thicknesses", type=int, nargs="+", default=list(THICKNESSES))
    parser.add_argument("--limit", type=int, default=None, help="first N sym folders per plane (tests)")
    parser.add_argument("--thickness-tol", type=float, default=POSITION_TOL,
                        help=f"largest allowed change of a label between thicknesses (default {POSITION_TOL:g}, the "
                             "coordinate rounding; a neighbour gained or lost changes the per-nm^2 labels by far more)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="count the structures and stop")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="worker processes (default: $SLURM_CPUS_PER_TASK, else 1)")
    args = parser.parse_args()

    started = time.time()
    out = args.out_dir
    if (out / CSV_NAME).exists() and not args.overwrite and not args.dry_run:
        parser.error(f"{sc.relative_path(out / CSV_NAME)} exists; pass --overwrite to replace it")
    gb_jobs, bulk_jobs, skipped = collect(args.planes, sorted(args.thicknesses), args.limit)
    print(f"{len(gb_jobs)} GB cells, {len(bulk_jobs)} bulk cells, {len(skipped)} skipped: {skipped}", flush=True)
    if args.dry_run:
        return 0

    import pandas as pd

    pristine, _ = sc.read_structure(sc.PRISTINE_POSCAR)
    shells, cutoff, neighbours = pristine_shells(pristine, args.cn_cutoff)
    d_bulk = shells[0]["distance"]
    print(f"neighbour cutoff {cutoff:.4f} A, bulk neighbours {neighbours}, bulk d_min {d_bulk:.4f} A", flush=True)

    from concurrent.futures import ProcessPoolExecutor

    order = sorted(gb_jobs, key=lambda job: -job[3].stat().st_size)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        gb_futures = [pool.submit(gb_task, part, args.radius, args.close, cutoff, neighbours)
                      for part in chunks(order, 4 * args.jobs)]
        bulk_futures = [pool.submit(bulk_task, part, args.close, cutoff, neighbours, d_bulk)
                        for part in chunks(bulk_jobs, args.jobs)]
        rows = [row for future in gb_futures for row in future.result()]
        bulk_problems = {}
        for future in bulk_futures:
            bulk_problems.update(future.result())
    table = pd.DataFrame(rows)
    table["region"] = pd.Categorical(table["region"], REGIONS, ordered=True)
    table = table.sort_values(["family", "thickness", "region"]).reset_index(drop=True)
    table["region"] = table["region"].astype(str)
    print(f"[{time.time() - started:5.0f} s] {len(table)} rows", flush=True)

    # ---- checks
    failures = []
    if bulk_problems:
        failures.append(f"bulk cells differ from pristine: {dict(list(bulk_problems.items())[:5])}")
    no_boundary = table[mc.no_boundary_rows(table)]
    if len(no_boundary):
        flat = ((no_boundary["n_close_pairs"] == 0) & (no_boundary["cn_defect"] == 0)
                & ((no_boundary["d_min"] - d_bulk).abs() < POSITION_TOL))
        if not flat.all():
            failures.append(f"NO_BOUNDARY families are not plain bulk: {no_boundary.loc[~flat, 'family'].unique().tolist()}")
    largest, spread = thickness_spread(table)
    if len(set(args.thicknesses)) > 1 and max(largest.values()) > args.thickness_tol:
        worst = spread.max(axis=1).sort_values(ascending=False).head(5)
        failures.append(f"labels change with thickness: largest {largest}; worst {worst.to_dict()}")
    if failures:
        for failure in failures:
            print(f"CHECK FAILED: {failure}", file=sys.stderr)
        return 1

    # ---- how much plane and grain class explain (distinct GBs at 8x, no NO_BOUNDARY)
    shares = {}
    labels = pd.read_csv(mc.LABELS_DIR / "gb_labels.csv", dtype={"plane": str})[["family", "grain_class"]]
    families = set(mc.distinct_families()) - set(mc.NO_BOUNDARY)
    reference_thickness = 8 if 8 in args.thicknesses else max(args.thicknesses)
    part = table[(table["thickness"] == reference_thickness) & table["family"].isin(families)].merge(labels, on="family")
    for region in REGIONS:
        sub = part[part["region"] == region]
        shares[region] = {"n_gbs": len(sub)}
        for label in TARGETS:
            shares[region][label] = {
                "plane": variance_shares(sub[label], sub["plane"].to_numpy()),
                "plane_and_grain_class": variance_shares(sub[label], (sub["plane"] + " " + sub["grain_class"]).to_numpy()),
                "min": float(sub[label].min()), "median": float(sub[label].median()), "max": float(sub[label].max()),
                "share_zero": float((sub[label] == 0).mean())}
    examples = part[(part["region"] == "cell") & part["family"].isin(["100/sym00001", "100/sym00002"])]

    out.mkdir(parents=True, exist_ok=True)
    mc.write_table(out / CSV_NAME, table)
    mc.write_json(out / JSON_NAME, {
        "what": "interface geometry of the unrelaxed GB cells; no DFT",
        "settings": {"radius": args.radius, "close": args.close, "neighbour_cutoff": cutoff,
                     "neighbour_cutoff_rule": "given" if args.cn_cutoff is not None else
                     "middle of the widest gap between pristine shells below 4 A",
                     "bulk_neighbours": neighbours, "bulk_d_min": d_bulk, "planes": args.planes,
                     "thicknesses": sorted(args.thicknesses), "limit": args.limit},
        "pristine_shells": shells,
        "n_rows": len(table), "n_structures": int(table[["family", "thickness"]].drop_duplicates().shape[0]),
        "skipped": skipped,
        "checks": {"bulk_cells_checked": len(bulk_jobs), "bulk_cells_differing": bulk_problems,
                   "no_boundary_families_flat": True, "largest_change_between_thicknesses": largest,
                   "thickness_tol": args.thickness_tol},
        "variance_explained_by_plane_and_grain_class": {"thickness": reference_thickness,
                                                        "gbs": "distinct GBs without NO_BOUNDARY", **shares},
        "examples_class_E_cell": examples[["family", "d_min", "n_close", "cn_excess"]].to_dict("records"),
        "seconds": round(time.time() - started, 1),
        "run": mc.run_metadata()})
    print(f"done in {time.time() - started:.0f} s; outputs in {sc.relative_path(out)}")
    for region in ("cell",):
        for label in TARGETS:
            s = shares[region][label]
            print(f"  {region} {label:14s} median {s['median']:.3f}  explained by plane {s['plane']:.2f}, "
                  f"plane + grain class {s['plane_and_grain_class']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
