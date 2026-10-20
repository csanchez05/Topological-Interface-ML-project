#!/usr/bin/env python3
"""Checks on the output of covariant_descriptor.py (covariant ACE), as src/dscribe/CoGe/covariant/check_covariant.py.

Writes checks.json into the run's both_interfaces folder and coinciding_copies.csv next to it. Checks:

  rotation        turn the smallest GB cell of each plane: lambda = 1 blocks turn with it as x, y, z
                  vectors; lambda = 2 blocks: each atom's Gram matrix over the 5 components is unchanged
                  and one orthogonal 5 x 5 matrix maps every atom's components onto the turned ones
  inversion       r -> -r: every block is multiplied by sigma (-1)^lambda
  bulk            every bulk cell's lambda >= 1 averages are ~0 against one atom's vector (point group 23
                  allows no vector and no rank-2 tensor, in any orientation of the crystal)
  thickness       interface vectors of one GB agree across thicknesses, for each interface on its own
                  and for the mean of both (informational; only GBs with several thicknesses in the run)
  coinciding GBs  for the families that the standard SOAP census puts in one structure class, whether
                  the covariant vectors are equal or turned copies (informational)

The rotation and inversion tests fail the run above --tol; the bulk test above --bulk-tol. Vectors
below --zero-tol of one atom's vector count as zero by symmetry, as in the SOAP check.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import covariant_ace as cov

ac, sc = cov.ac, cov.sc
CENSUS = (sc.PROJECT_ROOT / "calculations" / "ML_dscribe" / "CoGe" / "standard" / "rc6.0_n8_l6_s0.5"
          / "interface_r8_both_t4" / "distinct_gbs.csv")


def relative(a, b):
    """|a - b| / |b|, with |a - b| when b is zero."""
    scale = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / scale) if scale > 0 else float(np.linalg.norm(a - b))


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def fitted_transformation(before, after):
    """D with after[a] ~ D before[a] for every atom (least squares); (D, residual, |D D^T - I| max)."""
    n_comp = before.shape[1]
    A = before.transpose(1, 0, 2).reshape(n_comp, -1)
    B = after.transpose(1, 0, 2).reshape(n_comp, -1)
    D = B @ np.linalg.pinv(A)
    return D, relative(D @ A, B), float(np.abs(D @ D.T - np.eye(n_comp)).max())


def gram_change(before, after):
    """Relative change of the per-atom Gram matrices V^T V (P x P), via a QR of [V_after^T | V_before^T]."""
    n_comp = before.shape[1]
    R = np.linalg.qr(np.concatenate([after, before], axis=1).transpose(0, 2, 1), mode="r")
    R1, R0 = R[..., :n_comp], R[..., n_comp:]
    return relative(R1 @ R1.transpose(0, 2, 1), R0 @ R0.transpose(0, 2, 1))


def symmetry_tests(path, calculator, seed=7):
    """Rotation and inversion of one GB cell, atom by atom (a cell average can vanish by symmetry)."""
    atoms, _ = sc.read_structure(path)
    reference = calculator.per_atom(atoms)
    rotation = random_rotation(seed)
    turned = atoms.copy()
    turned.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    turned.positions = atoms.positions @ rotation.T
    rotated = calculator.per_atom(turned)
    inverted_atoms = atoms.copy()
    inverted_atoms.positions = -atoms.positions
    inverted_atoms.wrap()
    inverted = calculator.per_atom(inverted_atoms)
    out = {"structure": sc.relative_path(path), "n_atoms": len(atoms), "rotation_matrix": rotation.tolist(),
           "rotation": {}, "inversion": {}}
    for name in reference:
        if cov.BLOCKS[name][1] == 2:
            _, residual, orthogonality = fitted_transformation(reference[name], rotated[name])
            gram = gram_change(reference[name], rotated[name])
            out["rotation"][name] = {"rel_error": max(gram, residual, orthogonality), "gram_rel_change": gram,
                                     "one_matrix_residual": residual, "matrix_orthogonality": orthogonality,
                                     "rel_change_if_not_turned": relative(rotated[name], reference[name])}
        else:
            expected = np.einsum("ij,ajp->aip", rotation, reference[name])
            out["rotation"][name] = {"rel_error": relative(rotated[name], expected),
                                     "rel_change_if_not_turned": relative(rotated[name], reference[name])}
        sign = cov.inversion_sign(name)
        out["inversion"][name] = {"rel_error": relative(inverted[name], sign * reference[name]), "expected_sign": sign}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="run name or folder under calculations/descriptors/ACE/covariant")
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--tol", type=float, default=1e-6, help="tolerance of the symmetry tests (default 1e-6)")
    parser.add_argument("--bulk-tol", type=float, default=1e-5,
                        help="largest bulk cell average against one atom's vector (default 1e-5)")
    parser.add_argument("--consistency-tol", type=float, default=1e-3)
    parser.add_argument("--zero-tol", type=float, default=1e-3,
                        help="vectors shorter than this fraction of one atom's vector count as zero (default 1e-3)")
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--skip-symmetry", action="store_true", help="leave out the rotation and inversion tests")
    args = parser.parse_args()

    import pandas as pd

    folder = cov.run_dir("both_interfaces", args.run_dir)
    global_dir = cov.run_dir("global", folder)
    run = json.loads((folder / "settings.json").read_text())
    settings = run["ace"]
    blocks = [name for name in cov.BLOCKS if name in run["blocks"]]
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    summary = {"run_dir": sc.relative_path(folder), "ace": settings, "blocks": blocks}
    failures = []

    summary["symmetry"] = []
    if not args.skip_symmetry:
        calculator = cov.CovariantCalculator(settings, run.get("chunk", 64))
        for plane in sc.PLANES:
            rows = index[index["plane"] == plane]
            if rows.empty:
                continue
            row = rows.sort_values(["n_atoms", "sym"]).iloc[0]
            test = symmetry_tests(sc.PROJECT_ROOT / row["source"], calculator)
            summary["symmetry"].append(test)
            for kind in ("rotation", "inversion"):
                for name, entry in test[kind].items():
                    if entry["rel_error"] > args.tol:
                        failures.append(f"{kind} {name} {test['structure']}: {entry['rel_error']:.2e}")

    bulk, bulk_index = cov.bulk_averages(folder, blocks)
    kinds = bulk_index["kind"].to_numpy()
    entry = {"n_cells": int(len(bulk_index)), "tolerance": args.bulk_tol}
    for name in blocks:
        size = np.linalg.norm(bulk[name].reshape(len(bulk_index), -1), axis=1) / bulk_index[f"atom_norm_{name}"].to_numpy()
        entry[name] = {"pristine": float(size[kinds == "pristine"][0]), "max": float(size.max()),
                       "max_by_kind": {kind: float(size[kinds == kind].max()) for kind in np.unique(kinds)}}
        if size.max() > args.bulk_tol:
            failures.append(f"bulk cell with a non-zero {name} average ({size.max():.1e} of one atom's vector)")
    summary["bulk"] = entry

    # Interface vectors, one block at a time (a block of the whole run fits in a few GB).
    key = sc.radius_key(args.radius)
    lx, thickness = index["Lx"].to_numpy(), index["thickness"].to_numpy()
    groups = {family: rows[lx[rows] / 4.0 >= args.radius] for family, rows in index.groupby("family").indices.items()}
    groups = {family: rows for family, rows in groups.items() if len(rows) >= 2}
    census = pd.read_csv(args.census, dtype={"plane": str})
    at8 = index[index["thickness"] == 8].reset_index()
    row_of = dict(zip(at8["family"], at8["index"]))
    classes = [(c, sorted(m)) for c, m in census.groupby("structure_class")["family"]
               if len(m) >= 2 and all(f in row_of for f in m)]
    consistency, zero = {}, {}
    copies = {c: {"structure_class": c, "n_families": len(m), "families": " ".join(m)} for c, m in classes}
    differ = {c: [] for c, _ in classes}
    spread = {c: [] for c, _ in classes}
    with np.load(folder / cov.FILE) as data:
        n = data[f"n_{key}"]
        for name in blocks:
            X = data[f"X_{name}_{key}"].astype(np.float64)              # (N, plane, C, 2P)
            per_plane = X.reshape(X.shape[0], 2, -1)
            both = np.stack([sc.combine_planes(X[:, :, c], n) for c in range(X.shape[2])], axis=1)
            both = both.reshape(X.shape[0], -1)
            del X
            atom_norm = index[f"atom_norm_{name}"].to_numpy()
            for where in ("edge", "mid", "both"):
                label = f"{name} {where}"
                consistency[label], zero[label] = [], 0
                values_all = both if where == "both" else per_plane[:, sc.PLANE_NAMES.index(where)]
                for rows in groups.values():
                    values = values_all[rows]
                    size = np.linalg.norm(values, axis=1) / atom_norm[rows]
                    if size.max() < args.zero_tol:
                        zero[label] += 1
                        continue
                    k = int(np.argmax(thickness[rows]))
                    scale = max(np.linalg.norm(values[k]), args.zero_tol * atom_norm[rows[k]])
                    consistency[label].append(float((np.linalg.norm(values - values[k], axis=1) / scale).max()))
            for c, members in classes:
                rows = [row_of[m] for m in members]
                # Both interfaces, each on its own; a copy may have edge and mid the other way round.
                values = np.hstack([per_plane[rows, 0], per_plane[rows, 1]])
                swapped = np.hstack([per_plane[rows, 1], per_plane[rows, 0]])
                norms = np.linalg.norm(values, axis=1)
                record = copies[c]
                record[f"{name}_size"] = float((norms / atom_norm[rows]).max())
                record[f"{name}_max_rel_diff"] = float(max(min(relative(values[k], values[0]), relative(swapped[k], values[0]))
                                                           for k in range(len(rows))))
                record[f"{name}_norm_spread"] = float((norms.max() - norms.min()) / max(norms.max(), 1e-30))
                if record[f"{name}_size"] >= args.zero_tol:
                    differ[c].append(record[f"{name}_max_rel_diff"])
                    spread[c].append(record[f"{name}_norm_spread"])
            del per_plane, both
    summary["thickness_consistency"] = {
        label: {"n_families": len(v), "n_zero_by_symmetry": zero[label], "max": float(np.max(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "n_above_tol": int(np.sum(np.array(v) > args.consistency_tol))} for label, v in consistency.items()}
    summary["thickness_consistency"].update(tolerance=args.consistency_tol, zero_tol=args.zero_tol,
                                            n_gbs_with_several_thicknesses=len(groups))
    verdicts = {}
    for c, record in copies.items():
        if not differ[c]:
            verdict = "no vector (zero by symmetry)"
        elif max(differ[c]) <= 1e-4:
            verdict = "same"
        else:
            verdict = "turned copy" if max(spread[c]) < 1e-4 else "vectors differ"
        record["verdict"] = verdict
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    if copies:
        rows_out = list(copies.values())
        sc.write_csv(folder / "coinciding_copies.csv", rows_out, list(rows_out[0]))
    summary["coinciding_groups"] = {"census": sc.relative_path(args.census), "thickness": 8, "n_groups": len(copies),
                                    "verdicts": verdicts}
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    sc.write_json(folder / "checks.json", summary)

    for test in summary["symmetry"]:
        print(f"{test['structure']}: rotation error max {max(v['rel_error'] for v in test['rotation'].values()):.1e}, "
              f"inversion error max {max(v['rel_error'] for v in test['inversion'].values()):.1e}")
    print("bulk: largest cell average / one atom's vector: "
          + ", ".join(f"{name} {entry[name]['max']:.1e}" for name in blocks))
    for label, c in summary["thickness_consistency"].items():
        if not isinstance(c, dict):
            continue
        zeros = f" ({c['n_zero_by_symmetry']} zero by symmetry)" if c["n_zero_by_symmetry"] else ""
        if c["n_families"]:
            print(f"thickness consistency {label}: max {c['max']:.1e}, median {c['median']:.1e}, "
                  f"{c['n_above_tol']} of {c['n_families']} GBs above {args.consistency_tol:g}{zeros}")
    print(f"coinciding groups at 8x: {len(copies)}, verdicts {verdicts}")
    print(f"status: {summary['status']}; report in {folder / 'checks.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
