#!/usr/bin/env python3
"""Residual-symmetry labels of the GB cells: which symmetry operations survive in each joined cell.

SOAP guide Stage 1 follow-up, preparing Stages 2-3: covariant_vs_labels.py compares these labels
with covariant SOAP as inputs for the same property.

For every family (GB sym folder) the 8x cell
  structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/Interface/symNNNNN/8x/POSCAR
goes through spglib. Every operation becomes a Cartesian matrix R and is sorted against the
interface normal n (= x, the long axis of the orthogonal cell):

  inversion         R = -1
  mirror_parallel   R = 1 - 2 n n^T   a mirror or glide plane parallel to the interface
  twofold_normal    R = 2 n n^T - 1   a 2-fold rotation or 2_1 screw along x
  twofold_inplane   det R = +1, trace R = -1, R n = -n   a 2-fold rotation or screw with an in-plane axis

The operations are the full list of the space group spglib identifies, taken from spglib's table for
the standard setting and moved into the input frame (x_std = P x + p). spglib's own list for the input
cell can miss operations that do not keep the rectangular input cell (the 111 cells have a centred
in-plane lattice); the table cannot. Every operation must map the cell onto itself within 0.1 A.

The labels use symprec 1e-3 A. The run repeats at 5e-3 and 2e-2 A and lists the families whose labels
change there. Two columns follow from the grain class:
  lattice  twinned for R60[111], I.R60[111], R70.53[1-10], I.R70.53[1-10]; else shared
  motif    turned for R90[100], I.R90[100], R70.53[1-10], I.R70.53[1-10]; else aligned
With handedness they recode the 8 grain classes one to one (asserted).

Checks (a failure gives exit 1; the table is still written and the JSON says "status": "fail"):
  a  the plain-bulk pair 111/sym00119 and 111/sym00175 is space group 198 (P2_13)
  b  a cell with any improper operation (inversion, mirror, ...) has opposite-handed grains
  c  the members of each census distinct_id group share their labels (space group up to its
     enantiomorphic partner, and the four flags)
  d  a sample of families (20 per plane over the classes, plus the plain-bulk pair) has the same
     labels at 4x and 12x as at 8x
  e  inversion centres and mirror planes parallel to the interface sit at an interface (x = 0 or
     Lx/2), never inside a grain: a chiral grain cannot be its own mirror image, so a hit is a bug
  f  in every cell the operations map the cell onto itself (misfit < 0.1 A, a guard on the frame
     change), their rotations equal the order of the point group, and only the plain-bulk cells have
     operations that tilt the normal

Output (refused if it exists, unless --overwrite):
  calculations/ML_dscribe/CoGe/labels/gb_symmetry.csv   one row per family
  calculations/ML_dscribe/CoGe/labels/gb_symmetry.json  settings, checks, changed families, timing
gb_labels.csv is not touched.

Run with the spglib environment, on a compute node (about 1,600 POSCARs):
  srun -p general --exclude=qcg-docker,qcg3-0 -c 4 --mem=4G -t 00:30:00 bash -lc \\
      'cd <project root> && src/dft_plots/.venv/bin/python -B src/ML_dscribe_scripts/gb_symmetry.py --workers 4'
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "dft_plots" / "coge_interface"))

import ml_common as mc  # noqa: E402
from overlap_survey import read_poscar  # noqa: E402

GB_ROOT = PROJECT_ROOT / "structures" / "CoGeGrainBoundaries"
NAME = "gb_symmetry"
PLANES = ("100", "110", "111")
FLAGS = ("inversion", "mirror_parallel", "twofold_normal", "twofold_inplane")
LABELS = ("space_group",) + FLAGS
PLAIN_BULK_GROUP = 198
MATRIX_TOL = 0.02  # on the elements of a Cartesian operation; they are 0, +-1/2, +-sqrt(3)/2 or +-1
FRAME_MISFIT_A = 0.1  # an operation moved into the wrong frame misplaces atoms by much more
CLASSES = {  # grain class: (handedness, lattice, motif)
    "E": ("same", "shared", "aligned"),
    "I": ("opposite", "shared", "aligned"),
    "R90[100]": ("same", "shared", "turned"),
    "I.R90[100]": ("opposite", "shared", "turned"),
    "R60[111]": ("same", "twinned", "aligned"),
    "I.R60[111]": ("opposite", "twinned", "aligned"),
    "R70.53[1-10]": ("same", "twinned", "turned"),
    "I.R70.53[1-10]": ("opposite", "twinned", "turned"),
}
ENANTIOMORPHS = ((76, 78), (91, 95), (92, 96), (144, 145), (151, 153), (152, 154), (169, 170), (171, 172),
                 (178, 179), (180, 181), (212, 213))
PARTNER = {a: b for pair in ENANTIOMORPHS for a, b in (pair, pair[::-1])}


class SymmetryError(Exception):
    pass


# --------------------------------------------------------------------------- operations

def tol_name(value):
    """1e-3 -> '1e-3', 0.005 -> '5e-3'."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def cartesian(W, lattice):
    """Cartesian matrix of an operation W that acts on fractional column vectors (lattice rows = vectors)."""
    A = np.asarray(lattice, dtype=float).T
    return A @ np.asarray(W, dtype=float) @ np.linalg.inv(A)


def classify(R, n, tol=MATRIX_TOL):
    """Which of the four flags the Cartesian operation R is, plus whether it is improper."""
    R, n = np.asarray(R, dtype=float), np.asarray(n, dtype=float)
    eye, nn = np.eye(3), np.outer(n, n)

    def close(a, b):
        return bool(np.abs(a - b).max() < tol)

    det = float(np.linalg.det(R))
    return {"inversion": close(R, -eye),
            "mirror_parallel": close(R, eye - 2 * nn),
            "twofold_normal": close(R, 2 * nn - eye),
            "twofold_inplane": det > 0 and abs(np.trace(R) + 1) < tol and close(R @ n, -n),
            "improper": det < 0}


def centre_offset(tau_x, length):
    """Where an operation x -> -x + tau_x leaves planes fixed, as the distance to the nearest
    interface (x = 0 or Lx/2) in units of Lx/2: 0 at an interface, 0.5 in the middle of a grain."""
    half = length / 2
    x0 = (tau_x / 2) % half
    return float(min(x0, half - x0) / half)


def edge_image(tau_x, length):
    """Where the edge interface (x = 0) goes, in units of Lx/2 from the edge: 0 kept, 1 onto mid.

    Every operation maps x to +-x + tau_x, so x = 0 goes to tau_x."""
    position = (tau_x % length) / (length / 2)
    return float(min(position, 2 - position))


def point_group_order(hall_number):
    import spglib

    return len({tuple(np.asarray(r).ravel()) for r in spglib.get_symmetry_from_database(int(hall_number))["rotations"]})


def count_distinct(matrices, tol=1e-3):
    kept = []
    for M in matrices:
        if not any(np.abs(M - K).max() < tol for K in kept):
            kept.append(M)
    return len(kept)


def misfit(lattice, frac, kinds, R, tau):
    """Largest distance (A) from an atom's image under x -> R x + tau to the nearest atom of its species.

    Computed in the input cell, which must be orthogonal (the per-axis minimum image is then exact
    for any distance below half the shortest edge)."""
    lengths = np.diag(lattice)
    cart = frac * lengths
    image = (cart @ np.asarray(R).T + tau) / lengths
    worst = 0.0
    for s in np.unique(kinds):
        d = image[kinds == s][:, None, :] - frac[kinds == s][None, :, :]
        d -= np.rint(d)
        worst = max(worst, float(np.sqrt(np.square(d * lengths).sum(axis=-1)).min(axis=1).max()))
    return worst


def space_group_operations(dataset):
    """Every operation of the space group spglib found, acting on the input cell's fractional coordinates.

    They come from spglib's table for the standard setting, so the list is complete, and are moved
    to the input setting with x_std = P x + p (spglib: L_input = L_std P). W may be non-integer when
    an operation does not keep the input cell."""
    import spglib

    P = np.asarray(dataset.transformation_matrix, dtype=float)
    p = np.asarray(dataset.origin_shift, dtype=float)
    P_inv = np.linalg.inv(P)
    table = spglib.get_symmetry_from_database(int(dataset.hall_number))
    return [(P_inv @ Ws @ P, P_inv @ (Ws @ p + ws - p)) for Ws, ws in zip(table["rotations"], table["translations"])]


def cell_labels(lattice, frac, kinds, symprec):
    """Space group and flags of one orthogonal cell whose normal is x."""
    import warnings

    import spglib

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

    lattice = np.asarray(lattice, dtype=float)
    if np.abs(lattice - np.diag(np.diag(lattice))).max() > 1e-8:
        raise SymmetryError("the cell is not orthogonal")
    dataset = spglib.get_symmetry_dataset((lattice, frac, kinds), symprec=symprec)
    if dataset is None:
        raise SymmetryError(f"spglib found no symmetry: {spglib.get_error_message()}")
    n = np.array([1.0, 0.0, 0.0])
    length = float(lattice[0, 0])
    flags = dict.fromkeys(FLAGS + ("improper",), False)
    rotations, worst, worst_orthogonality = [], 0.0, 0.0
    improper_offset, swaps, tilts = float("nan"), False, False
    for W, t in space_group_operations(dataset):
        R, tau = cartesian(W, lattice), lattice.T @ t
        rotations.append(R)
        worst_orthogonality = max(worst_orthogonality, float(np.abs(R.T @ R - np.eye(3)).max()))
        for name, value in classify(R, n).items():
            flags[name] |= value
        worst = max(worst, misfit(lattice, frac, kinds, R, tau))
        if min(np.abs(R @ n - n).max(), np.abs(R @ n + n).max()) > MATRIX_TOL:
            tilts = True  # only possible without an interface (the plain-bulk cells)
            continue
        if np.linalg.det(R) < 0 and np.abs(R @ n + n).max() < MATRIX_TOL:
            offset = centre_offset(tau[0], length)
            improper_offset = offset if np.isnan(improper_offset) else max(improper_offset, offset)
        swaps |= edge_image(tau[0], length) > 0.5
    if worst_orthogonality > 0.05:
        raise SymmetryError(f"an operation is not a rotation (|R^T R - 1| = {worst_orthogonality:.3g})")
    return {"space_group": int(dataset.number), "sg_symbol": str(dataset.international),
            "point_group": str(dataset.pointgroup), "n_rotations": count_distinct(rotations),
            "point_group_order": point_group_order(dataset.hall_number),
            "n_rotations_input_cell": len({tuple(np.asarray(W).ravel()) for W in dataset.rotations}),
            **{name: bool(value) for name, value in flags.items()},
            "max_misfit_A": round(worst, 6), "improper_centre_offset": improper_offset, "swaps_edge_mid": bool(swaps),
            "tilts_normal": bool(tilts)}


# --------------------------------------------------------------------------- families and tasks

def poscar_path(family, thickness):
    plane, sym = family.split("/")
    return GB_ROOT / f"CoGe_{plane}_GBs" / "Interface" / sym / f"{thickness}x" / "POSCAR"


def run_task(task):
    """(family, thickness, symprecs) -> (family, thickness, {symprec: labels}, error)."""
    family, thickness, symprecs = task
    try:
        lattice, frac, kinds = read_poscar(poscar_path(family, thickness).read_text())
        frac = frac - np.floor(frac)
        return family, thickness, {s: cell_labels(lattice, frac, kinds, s) for s in symprecs}, None
    except Exception as error:  # one bad cell must not stop the run; it is reported and fails the run
        return family, thickness, {}, f"{type(error).__name__}: {error}"


def recode(table):
    """lattice and motif columns from the grain class; asserts the one-to-one recoding of the 8 classes."""
    unknown = sorted(set(table["grain_class"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown grain classes {unknown}")
    expected = table["grain_class"].map(lambda c: CLASSES[c][0])
    wrong = table.loc[expected != table["handedness"], "family"].tolist()
    if wrong:
        raise ValueError(f"handedness does not match the grain class for {wrong[:5]}")
    out = table.assign(lattice=table["grain_class"].map(lambda c: CLASSES[c][1]),
                       motif=table["grain_class"].map(lambda c: CLASSES[c][2]))
    triples = out.groupby("grain_class")[["handedness", "lattice", "motif"]].nunique()
    if (triples > 1).any().any() or out[["handedness", "lattice", "motif"]].drop_duplicates().shape[0] != \
            out["grain_class"].nunique():
        raise AssertionError("(handedness, lattice, motif) does not recode the grain classes one to one")
    return out


def select_families(labels, limit):
    """All families, or the first `limit` sym folders per plane plus the plain-bulk pair."""
    if not limit:
        return labels.sort_values("family").reset_index(drop=True)
    keep = labels.sort_values("family").groupby("plane", sort=True).head(limit)["family"]
    keep = set(keep) | set(mc.NO_BOUNDARY)
    return labels[labels["family"].isin(keep)].sort_values("family").reset_index(drop=True)


def check_sample(table, per_plane, seed):
    """Families for the 4x/12x check: per plane, round robin over the grain classes, plus the plain-bulk pair."""
    rng = np.random.default_rng(seed)
    chosen = []
    for _, plane_rows in table.groupby("plane", sort=True):
        pools = [list(rng.permutation(rows["family"].to_numpy()))
                 for _, rows in plane_rows.groupby("grain_class", sort=True)]
        picked = []
        while len(picked) < per_plane and any(pools):
            for pool in pools:
                if pool and len(picked) < per_plane:
                    picked.append(pool.pop(0))
        chosen.extend(sorted(picked))
    chosen.extend(f for f in mc.NO_BOUNDARY if f in set(table["family"]) and f not in chosen)
    return chosen


def canonical_group(number):
    return min(int(number), PARTNER.get(int(number), int(number)))


def label_key(row, enantiomorph_free=False):
    group = canonical_group(row["space_group"]) if enantiomorph_free else int(row["space_group"])
    return (group,) + tuple(int(row[f]) for f in FLAGS)


# --------------------------------------------------------------------------- checks

def check_plain_bulk(table):
    rows = table[table["family"].isin(mc.NO_BOUNDARY)]
    found = dict(zip(rows["family"], rows["space_group"].astype(int)))
    return {"pass": len(found) == len(mc.NO_BOUNDARY) and all(v == PLAIN_BULK_GROUP for v in found.values()),
            "found": found, "expected": PLAIN_BULK_GROUP}


def check_handedness(table):
    bad = table[(table["improper"] == 1) & (table["handedness"] != "opposite")]
    return {"pass": bad.empty, "n_improper": int((table["improper"] == 1).sum()),
            "n_improper_same_handed": len(bad), "examples": bad["family"].head(10).tolist()}


def check_groups(table, census):
    merged = table.merge(census[["family", "distinct_id"]], on="family", how="left", validate="one_to_one")
    bad, n_groups = [], 0
    for distinct_id, rows in merged.groupby("distinct_id"):
        if len(rows) < 2:
            continue
        n_groups += 1
        keys = {label_key(r, enantiomorph_free=True) for r in rows.to_dict("records")}
        if len(keys) > 1:
            bad.append({"distinct_id": distinct_id, "families": rows["family"].tolist(),
                        "labels": sorted(map(list, keys))})
    exact = sum(len({label_key(r) for r in rows.to_dict("records")}) > 1
                for _, rows in merged.groupby("distinct_id") if len(rows) > 1)
    return {"pass": not bad, "n_groups_tested": n_groups, "n_groups_differing": len(bad),
            "n_groups_differing_only_by_enantiomorph": int(exact - len(bad)), "examples": bad[:10]}


def check_thickness(table, extra):
    """extra: {(family, thickness): labels at the primary symprec}."""
    at8 = {r["family"]: label_key(r) for r in table.to_dict("records")}
    extra = {key: v for key, v in extra.items() if key[0] in at8}
    bad = [{"family": f, "thickness": t, "at_8x": list(at8[f]), "here": list(label_key(v))}
           for (f, t), v in sorted(extra.items()) if label_key(v) != at8[f]]
    return {"pass": not bad, "n_compared": len(extra), "n_differing": len(bad), "examples": bad[:10]}


def check_centres(table):
    offsets = table["improper_centre_offset"].dropna()
    bad = table.loc[table["improper_centre_offset"] > 0.25, "family"].tolist()
    return {"pass": not bad, "n_cells_with_such_operations": int(offsets.size),
            "largest_offset_in_half_cells": float(offsets.max()) if offsets.size else None, "examples": bad[:10]}


def check_operations(table):
    """Operations fit the cell, rotations = point-group order, only the plain-bulk cells tilt the normal."""
    bad = table.loc[(table["n_rotations"] != table["point_group_order"]) | (table["max_misfit_A"] >= FRAME_MISFIT_A),
                    "family"].tolist()
    short = int((table["n_rotations_input_cell"] < table["n_rotations"]).sum())
    tilting = table.loc[table["tilts_normal"] == 1, "family"].tolist()
    return {"pass": not bad and set(tilting) <= set(mc.NO_BOUNDARY), "examples": bad[:10],
            "largest_misfit_A": float(table["max_misfit_A"].max()) if len(table) else None,
            "n_cells_where_the_input_cell_misses_rotations": short, "cells_with_operations_tilting_the_normal": tilting}


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symprec", type=float, default=1e-3, help="tolerance (A) for the labels")
    parser.add_argument("--repeat-symprec", type=float, nargs="*", default=[5e-3, 2e-2],
                        help="tolerances (A) to repeat at; families whose labels change are listed")
    parser.add_argument("--thickness", type=int, default=8)
    parser.add_argument("--check-thicknesses", type=int, nargs="*", default=[4, 12])
    parser.add_argument("--check-sample", type=int, default=20, help="families per plane in check d")
    parser.add_argument("--seed", type=int, default=0, help="seed of the check-d sample")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: the first N sym folders per plane, plus the plain-bulk pair")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=mc.LABELS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_table(families, results, args):
    import pandas as pd

    records, changed = [], {tol_name(s): [] for s in args.repeat_symprec}
    for row in families.to_dict("records"):
        found = results[row["family"]]
        base = found[args.symprec]
        record = {**{k: row[k] for k in ("family", "plane", "sym", "grain_class", "handedness", "lattice", "motif")},
                  **{k: base[k] for k in ("space_group", "sg_symbol", "point_group", "n_rotations")},
                  **{k: int(base[k]) for k in ("improper",) + FLAGS},
                  "max_misfit_A": base["max_misfit_A"]}
        for s in args.repeat_symprec:
            other = found[s]
            same = label_key(other) == label_key(base)
            record[f"space_group_{tol_name(s)}"] = other["space_group"]
            record[f"same_labels_{tol_name(s)}"] = int(same)
            if not same:
                changed[tol_name(s)].append({"family": row["family"], "at_" + tol_name(args.symprec): list(label_key(base)),
                                             "here": list(label_key(other)), "sg_symbol_here": other["sg_symbol"],
                                             "max_misfit_A_here": other["max_misfit_A"]})
        record.update({k: base[k] for k in ("point_group_order", "n_rotations_input_cell", "improper_centre_offset")})
        record.update({k: int(base[k]) for k in ("swaps_edge_mid", "tilts_normal")})
        records.append(record)
    return pd.DataFrame.from_records(records), changed


def summary_counts(table):
    out = {}
    for plane, rows in table.groupby("plane"):
        out[plane] = {"n": len(rows), "space_groups": {str(k): int(v) for k, v in rows["space_group"].value_counts().sort_index().items()},
                      **{f: int(rows[f].sum()) for f in ("improper",) + FLAGS},
                      "by_grain_class": {c: {f: int(r[f].sum()) for f in FLAGS} | {"n": len(r)}
                                         for c, r in rows.groupby("grain_class")}}
    return out


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    import pandas as pd

    out_csv, out_json = Path(args.out_dir) / f"{NAME}.csv", Path(args.out_dir) / f"{NAME}.json"
    if (out_csv.exists() or out_json.exists()) and not args.overwrite and not args.dry_run:
        print(f"refused: {mc.relative_path(out_csv)} exists (use --overwrite)", file=sys.stderr)
        return 2
    labels = pd.read_csv(mc.LABELS_DIR / "gb_labels.csv", dtype={"plane": str})
    census = pd.read_csv(mc.CENSUS, dtype={"plane": str})
    families = recode(select_families(labels, args.limit)[["family", "plane", "sym", "grain_class", "handedness"]])
    sample = check_sample(families, args.check_sample, args.seed)
    symprecs = (args.symprec,) + tuple(args.repeat_symprec)
    tasks = [(f, args.thickness, symprecs) for f in families["family"]]
    tasks += [(f, t, (args.symprec,)) for t in args.check_thicknesses for f in sample]
    print(f"{len(families)} families at {args.thickness}x (symprec {', '.join(map(tol_name, symprecs))} A), "
          f"{len(sample)} families at {args.check_thicknesses}x; {len(tasks)} POSCARs, {args.workers} workers; "
          f"output {mc.relative_path(out_csv)}", flush=True)
    if args.dry_run:
        return 0

    results, extra, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (family, thickness, found, error) in enumerate(pool.map(run_task, tasks, chunksize=4), 1):
            if error:
                errors.append({"family": family, "thickness": thickness, "error": error})
            elif thickness == args.thickness and len(found) == len(symprecs):
                results[family] = found
            else:
                extra[(family, thickness)] = found[args.symprec]
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} cells, {time.time() - started:.0f} s", flush=True)
    if errors:
        print(f"{len(errors)} cells failed, e.g. {errors[:3]}", file=sys.stderr)
        families = families[families["family"].isin(results)]
    table, changed = build_table(families, results, args)
    checks = {"a_plain_bulk_is_198": check_plain_bulk(table),
              "b_improper_only_opposite_handed": check_handedness(table),
              "c_same_cell_same_labels": check_groups(table, census),
              "d_same_labels_at_4x_12x": check_thickness(table, extra),
              "e_improper_centres_at_interfaces": check_centres(table),
              "f_operations_complete_and_fitting": check_operations(table)}
    checks["a_plain_bulk_is_198"]["at_other_symprecs"] = {
        tol_name(s): {f: results[f][s]["space_group"] for f in mc.NO_BOUNDARY if f in results} for s in args.repeat_symprec}
    status = "pass" if all(c["pass"] for c in checks.values()) and not errors else "fail"
    payload = {"what": "residual symmetry of the joined GB cells (spglib); one row per family in gb_symmetry.csv",
               "status": status, "n_families": len(table), "thickness": args.thickness,
               "symprec_A": args.symprec, "repeat_symprec_A": args.repeat_symprec, "matrix_tolerance": MATRIX_TOL,
               "columns": {
                   "space_group": f"International number at symprec {tol_name(args.symprec)} A",
                   "inversion": "1 if the cell has an inversion centre",
                   "mirror_parallel": "1 if it has a mirror or glide plane parallel to the interface",
                   "twofold_normal": "1 if it has a 2-fold rotation or 2_1 screw along the normal x",
                   "twofold_inplane": "1 if it has a 2-fold rotation or screw with an axis in the interface plane",
                   "improper": "1 if it has any operation with det -1",
                   "lattice": "twinned (R60[111], R70.53[1-10] and their I. partners) or shared",
                   "motif": "turned (R90[100], R70.53[1-10] and their I. partners) or aligned",
                   "max_misfit_A": "largest distance from an atom's image to the nearest atom of its species, over all operations",
                   "space_group_<tol>, same_labels_<tol>": "space group and agreement of all five labels at the repeat tolerance",
                   "point_group_order, n_rotations_input_cell": "diagnostics of check f",
                   "improper_centre_offset": "largest distance of an inversion centre or parallel mirror from an interface, in units of Lx/2 (check e)",
                   "swaps_edge_mid": "1 if some operation maps the edge interface onto the mid one",
                   "tilts_normal": "1 if some operation does not keep the normal (plain bulk only; check f)"},
               "checks": checks, "changed_families": changed,
               "n_changed": {k: len(v) for k, v in changed.items()},
               "n_cells_with_an_operation_swapping_edge_and_mid": int(table["swaps_edge_mid"].sum()),
               "counts": summary_counts(table), "thickness_sample": sample, "errors": errors,
               "seconds": round(time.time() - started, 1),
               "run": {**mc.run_metadata(), "spglib": __import__("spglib").__version__}}
    mc.write_table(out_csv, table)
    mc.write_json(out_json, payload)
    for name, check in checks.items():
        print(f"  check {name}: {'pass' if check['pass'] else 'FAIL'}")
    print(f"labels changed: {payload['n_changed']}; status {status}; wrote {mc.relative_path(out_csv)} "
          f"({time.time() - started:.0f} s)")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
