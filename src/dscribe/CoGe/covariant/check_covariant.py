#!/usr/bin/env python3
"""Checks on the output of covariant_descriptor.py.

Writes checks.json into the run's both_interfaces folder and
coinciding_copies.csv (one row per coinciding group) next to it. Checks:

  rotation         turn a GB cell: invariant block unchanged, lambda = 1 blocks turn with it as
                   vectors; lambda = 2 blocks: each atom's Gram matrix over the 5 components is
                   unchanged, and one orthogonal 5 x 5 matrix maps every atom's components
                   onto the turned ones (featomic has no Wigner D matrices to compare with)
  inversion        r -> -r: every block is multiplied by sigma * (-1)^lambda
  bulk             the pristine cell's lambda >= 1 averages are ~0 (the crystal's 12 rotations
                   allow no vector and no rank-2 tensor); every transformed bulk cell has the
                   pristine invariant block (only when the run has the invariant block)
  thickness        interface vectors of one GB agree across thicknesses; polar and axial for each
                   interface on its own (edge, mid) and for the mean of both (informational)
  coinciding GBs   for the families that the standard SOAP census puts in one structure class,
                   whether the covariant vectors are equal, or turned copies of each other
  same as an older run (--compare-with RUN): every block both runs hold, for the structures and
                   bulk cells both hold, must agree to --compare-tol (relative to the block's
                   largest entry); whether they are bit-identical is recorded (vectors from different
                   CPU types can differ in the last digits)

The blocks of the run are read from its settings.json, so a run without the invariant
block skips the checks that need it.

Polar and axial vectors that symmetry forces to zero come out as rounding noise of the POSCAR
coordinates (1e-15 to 1e-5 of one atom's vector). Vectors below --zero-tol of one atom's vector
count as zero by symmetry and are left out of the relative comparisons. The mean of both
interfaces can be zero although each interface has a vector: the two point in opposite
directions. Use the interfaces on their own.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

import soap_common as sc
from covariant_descriptor import BLOCKS, inversion_sign, make_calculator, per_atom_blocks, split_mean

CENSUS = (sc.PROJECT_ROOT / "calculations" / "ML_dscribe" / "CoGe" / "standard" / "rc6.0_n8_l6_s0.5"
          / "interface_r8_both_t4" / "distinct_gbs.csv")


def relative(a, b):
    """|a - b| / |b|, with 0 when both are zero."""
    scale = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / scale) if scale > 0 else float(np.linalg.norm(a - b))


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def fitted_transformation(before, after):
    """The one matrix D with after[a] ~ D @ before[a] for every atom a (least squares over all atoms).

    before, after: (n_atoms, C, P). Returns (D, relative residual, |D D^T - I| largest entry).
    """
    n_comp = before.shape[1]
    A = before.transpose(1, 0, 2).reshape(n_comp, -1)
    B = after.transpose(1, 0, 2).reshape(n_comp, -1)
    D = B @ np.linalg.pinv(A)
    residual = relative(D @ A, B)
    return D, residual, float(np.abs(D @ D.T - np.eye(n_comp)).max())


def gram_change(before, after):
    """Relative change of the per-atom Gram matrices V^T V (P x P, summed over the components), all atoms together.

    The P x P matrices are never formed (P is thousands, so they would need tens of GB): a QR of
    [V_after^T | V_before^T] (P x 2C) per atom gives V^T = Q R_part with orthonormal Q, so
    |V1^T V1 - V0^T V0| = |R1 R1^T - R0 R0^T| (2C x 2C), as accurate as the direct product.
    """
    n_comp = before.shape[1]
    stacked = np.concatenate([after, before], axis=1).transpose(0, 2, 1)  # (atoms, P, 2C)
    R = np.linalg.qr(stacked, mode="r")
    R1, R0 = R[..., :n_comp], R[..., n_comp:]
    return relative(R1 @ R1.transpose(0, 2, 1), R0 @ R0.transpose(0, 2, 1))


def symmetry_tests(path, calculator, keys, seed=7):
    """Rotation and inversion of one GB cell, compared atom by atom. Returns relative errors per block.

    Each atom's own vector is used rather than the cell average, because the
    average can vanish by symmetry and then says nothing.
    """
    atoms, _ = sc.read_structure(path)
    reference, _ = per_atom_blocks(atoms, calculator, keys)
    rotation = random_rotation(seed)
    turned = atoms.copy()
    turned.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    turned.positions = atoms.positions @ rotation.T
    rotated, _ = per_atom_blocks(turned, calculator, keys)
    inverted_atoms = atoms.copy()
    inverted_atoms.positions = -atoms.positions
    inverted_atoms.wrap()
    inverted, _ = per_atom_blocks(inverted_atoms, calculator, keys)
    out = {"structure": sc.relative_path(path), "n_atoms": len(atoms), "rotation_matrix": rotation.tolist(),
           "rotation": {}, "inversion": {}}
    for name in reference:
        o3_lambda = BLOCKS[name][0]
        if o3_lambda == 2:
            _, residual, orthogonality = fitted_transformation(reference[name], rotated[name])
            gram = gram_change(reference[name], rotated[name])
            # The error is the larger of: Gram matrices changed, no single D fits, D not orthogonal.
            out["rotation"][name] = {"rel_error": max(gram, residual, orthogonality), "gram_rel_change": gram,
                                     "one_matrix_residual": residual, "matrix_orthogonality": orthogonality,
                                     "rel_change_if_not_turned": relative(rotated[name], reference[name])}
        else:
            expected = reference[name] if o3_lambda == 0 else np.einsum("ij,ajp->aip", rotation, reference[name])
            out["rotation"][name] = {"rel_error": relative(rotated[name], expected),
                                     "rel_change_if_not_turned": relative(rotated[name], reference[name])}
        sign = inversion_sign(name)
        out["inversion"][name] = {"rel_error": relative(inverted[name], sign * reference[name]),
                                  "expected_sign": sign}
    return out


def compare_runs(folder, other, blocks, radius):
    """Largest differences between two runs, block by block, on the structures and bulk cells both hold."""
    import pandas as pd

    out = {}
    key = sc.radius_key(radius)
    for region, name, suffix in (("both_interfaces", "covariant_soap.npz", f"_{key}"), ("global", "covariant_soap.npz", ""),
                                 ("global", "bulk_covariant.npz", "")):
        here, there = sc.run_dir(region, folder), sc.run_dir(region, other)
        index_name = "bulk_index.csv" if name.startswith("bulk") else "index.csv"
        columns = ["kind", "plane", "sym"] if index_name == "bulk_index.csv" else ["family", "thickness"]
        a = pd.read_csv(here / index_name, dtype={"plane": str}).fillna({"plane": ""})[columns]
        b = pd.read_csv(there / index_name, dtype={"plane": str}).fillna({"plane": ""})[columns]
        merged = a.assign(index_a=np.arange(len(a))).merge(b.assign(index_b=np.arange(len(b))), on=columns,
                                                           validate="one_to_one")
        with np.load(here / name) as data_a, np.load(there / name) as data_b:
            shared = [block for block in blocks if f"X_{block}{suffix}" in data_b.files]
            for block in shared:
                x = data_a[f"X_{block}{suffix}"][merged["index_a"].to_numpy()]
                y = data_b[f"X_{block}{suffix}"][merged["index_b"].to_numpy()]
                difference = float(np.abs(x.astype(np.float64) - y).max())
                out[f"{region}/{name}/{block}"] = {"n": int(len(merged)), "max_abs_difference": difference,
                                                    "relative": difference / float(np.abs(y).max()),
                                                    "bit_identical": bool(np.array_equal(x, y))}
            if suffix:
                out[f"{region}/{name}/n{suffix}"] = {"n": int(len(merged)), "identical_counts": bool(np.array_equal(
                    data_a[f"n{suffix}"][merged["index_a"].to_numpy()], data_b[f"n{suffix}"][merged["index_b"].to_numpy()]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="rc6.0_n8_l6_s0.5",
                        help="tag or run folder under calculations/descriptors/dscribe/covariant (default rc6.0_n8_l6_s0.5)")
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--tol", type=float, default=1e-6, help="tolerance of the symmetry tests (default 1e-6)")
    parser.add_argument("--consistency-tol", type=float, default=1e-3)
    parser.add_argument("--zero-tol", type=float, default=1e-3,
                        help="polar/axial vectors shorter than this fraction of one atom's vector count as zero "
                             "by symmetry (default 1e-3)")
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--compare-with", default=None,
                        help="an older run (tag or folder) whose shared blocks this run must reproduce")
    parser.add_argument("--compare-tol", type=float, default=1e-6,
                        help="largest allowed difference relative to the block's largest entry (default 1e-6)")
    args = parser.parse_args()

    import pandas as pd

    folder = sc.run_dir("both_interfaces", args.run_dir, root=sc.COVARIANT_ROOT)
    global_dir = sc.run_dir("global", folder)
    run = json.loads((folder / "settings.json").read_text())
    settings = run["settings"]
    blocks = [name for name in BLOCKS if name in run.get("blocks", ("invariant", "polar", "axial"))]
    vectorial = [name for name in blocks if BLOCKS[name][0] > 0]
    calculator, keys = make_calculator(settings, blocks)
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    summary = {"run_dir": sc.relative_path(folder), "settings": settings, "blocks": blocks}
    failures = []

    # Rotation and inversion on a small 4x cell of each plane.
    summary["symmetry"] = []
    for plane in sc.PLANES:
        rows = index[(index["plane"] == plane) & (index["thickness"] == index["thickness"].min())]
        if rows.empty:
            continue
        test = symmetry_tests(sc.PROJECT_ROOT / rows.iloc[0]["source"], calculator, keys)
        summary["symmetry"].append(test)
        for kind in ("rotation", "inversion"):
            for name, entry in test[kind].items():
                if entry["rel_error"] > args.tol:
                    failures.append(f"{kind} {name} {test['structure']}: {entry['rel_error']:.2e}")

    # Bulk cells.
    bulk_index = pd.read_csv(global_dir / "bulk_index.csv", dtype={"plane": str})
    with np.load(global_dir / "bulk_covariant.npz") as data:
        bulk = {name: data[f"X_{name}"] for name in blocks}
    pristine = int(np.flatnonzero(bulk_index["kind"] == "pristine")[0])
    transformed = (bulk_index["kind"] == "transformed").to_numpy()
    entry = {}
    for name in vectorial:
        # Size of the cell average against the size of one atom's vector.
        size = np.linalg.norm(bulk[name].reshape(len(bulk_index), -1), axis=1) / bulk_index[f"atom_norm_{name}"]
        entry[f"pristine_{name}_average_over_atom_norm"] = float(size[pristine])
        entry[f"max_{name}_average_over_atom_norm_all_cells"] = float(size.max())
        entry[f"{name}_average_over_atom_norm_by_kind"] = {
            kind: float(size[(bulk_index["kind"] == kind).to_numpy()].max()) for kind in bulk_index["kind"].unique()}
        if entry[f"max_{name}_average_over_atom_norm_all_cells"] > 1e-5:
            failures.append(f"bulk cell with a non-zero {name} average")
    if "invariant" in blocks:
        invariant = bulk["invariant"].reshape(len(bulk_index), -1)
        difference = np.linalg.norm(invariant - invariant[pristine], axis=1) / np.linalg.norm(invariant[pristine])
        entry["invariant_rel_diff_from_pristine"] = {
            plane: float(difference[transformed & (bulk_index["plane"] == plane).to_numpy()].max())
            for plane in sc.PLANES if (transformed & (bulk_index["plane"] == plane).to_numpy()).any()}
        if max(entry["invariant_rel_diff_from_pristine"].values()) > 1e-6:
            failures.append("bulk invariant block differs from pristine")
    entry["n_cells"] = int(len(bulk_index))
    summary["bulk"] = entry

    # Interface vectors of every GB.
    key = sc.radius_key(args.radius)
    with np.load(folder / "covariant_soap.npz") as data:
        n = data[f"n_{key}"]
        vectors, per_plane = {}, {}
        for name in blocks:
            X = data[f"X_{name}_{key}"].astype(np.float64)  # (N, plane, components, 2P)
            per_plane[name] = X.reshape(X.shape[0], 2, -1)
            # combine_planes splits the last axis into species halves, so each component goes on its own.
            vectors[name] = np.stack([sc.combine_planes(X[:, :, c], n) for c in range(X.shape[2])], axis=1)
    lx = index["Lx"].to_numpy()
    thickness = index["thickness"].to_numpy()

    atom_norm = {name: index[f"atom_norm_{name}"].to_numpy() for name in blocks}
    views = {name: ("both",) if name == "invariant" else ("edge", "mid", "both") for name in blocks}

    def flat(name, rows, where="both"):
        if where == "both":
            return vectors[name][rows].reshape(len(rows), -1)
        return per_plane[name][rows, sc.PLANE_NAMES.index(where)]

    consistency = {f"{name} {where}": [] for name in blocks for where in views[name]}
    zero = {label: 0 for label in consistency}
    for family, rows in index.groupby("family").indices.items():
        rows = rows[lx[rows] / 4.0 >= args.radius]
        if len(rows) < 2:
            continue
        k = int(np.argmax(thickness[rows]))
        for name in blocks:
            for where in views[name]:
                label = f"{name} {where}"
                values = flat(name, rows, where)
                size = np.linalg.norm(values, axis=1) / atom_norm[name][rows]
                if name != "invariant" and size.max() < args.zero_tol:
                    zero[label] += 1
                    continue
                scale = max(np.linalg.norm(values[k]), args.zero_tol * atom_norm[name][rows[k]])
                consistency[label].append(float((np.linalg.norm(values - values[k], axis=1) / scale).max()))
    summary["thickness_consistency"] = {
        label: {"n_families": len(v), "n_zero_by_symmetry": zero[label], "max": float(np.max(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "n_above_tol": int(np.sum(np.array(v) > args.consistency_tol))} for label, v in consistency.items()}
    summary["thickness_consistency"]["tolerance"] = args.consistency_tol
    summary["thickness_consistency"]["zero_tol"] = args.zero_tol

    # Families that standard SOAP cannot tell apart: equal covariant vectors, or turned copies?
    census = pd.read_csv(args.census, dtype={"plane": str})
    at8 = index[index["thickness"] == 8].reset_index()
    row_of = dict(zip(at8["family"], at8["index"]))
    rows_out = []
    for structure_class, members in census.groupby("structure_class")["family"]:
        members = sorted(members)
        if len(members) < 2 or not all(m in row_of for m in members):
            continue
        rows = [row_of[m] for m in members]
        record = {"structure_class": structure_class, "n_families": len(members), "families": " ".join(members)}
        differ, spread = [], []
        for name in blocks:
            if name == "invariant":
                values = flat(name, rows)
                swapped = values
            else:
                # Both interfaces, each on its own; a copy may have edge and mid the other way round.
                values = np.hstack([flat(name, rows, "edge"), flat(name, rows, "mid")])
                swapped = np.hstack([flat(name, rows, "mid"), flat(name, rows, "edge")])
            norms = np.linalg.norm(values, axis=1)
            record[f"{name}_size"] = float((norms / atom_norm[name][rows]).max())
            record[f"{name}_max_rel_diff"] = float(max(min(relative(values[k], values[0]), relative(swapped[k], values[0]))
                                                       for k in range(len(rows))))
            record[f"{name}_norm_spread"] = float((norms.max() - norms.min()) / max(norms.max(), 1e-30))
            if name != "invariant" and record[f"{name}_size"] >= args.zero_tol:
                differ.append(record[f"{name}_max_rel_diff"])
                spread.append(record[f"{name}_norm_spread"])
        verdict = "same"
        if record.get("invariant_max_rel_diff", 0.0) > 1e-4:
            verdict = "invariant differs"
        elif not differ:
            verdict = "no vector (zero by symmetry)"
        elif max(differ) > 1e-4:
            verdict = "turned copy" if max(spread) < 1e-4 else "vectors differ"
        record["verdict"] = verdict
        rows_out.append(record)
    if rows_out:
        sc.write_csv(folder / "coinciding_copies.csv", rows_out, list(rows_out[0]))
    verdicts = {}
    for record in rows_out:
        verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
    summary["coinciding_groups"] = {"census": sc.relative_path(args.census), "thickness": 8,
                                    "n_groups": len(rows_out), "verdicts": verdicts,
                                    "meaning": {"same": "covariant vectors equal too: the copies have the same orientation",
                                                "no vector (zero by symmetry)": "polar and axial vectors are zero, so "
                                                                                "orientation cannot be compared",
                                                "turned copy": "invariant equal, vectors of equal length pointing "
                                                               "differently: the same GB turned in space"}}
    if args.compare_with:
        other = sc.run_dir("both_interfaces", args.compare_with, root=sc.COVARIANT_ROOT)
        comparison = compare_runs(folder, other, blocks, args.radius)
        summary["compared_with"] = {"run_dir": sc.relative_path(other), "tolerance": args.compare_tol, "arrays": comparison}
        for name, difference in comparison.items():
            if difference.get("relative", 0.0) > args.compare_tol or difference.get("identical_counts") is False:
                failures.append(f"differs from {sc.relative_path(other)}: {name} {difference}")
        if not comparison:
            failures.append(f"no shared block with {sc.relative_path(other)}")
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    sc.write_json(folder / "checks.json", summary)

    for test in summary["symmetry"]:
        print(f"{test['structure']}: rotation error "
              + ", ".join(f"{k} {v['rel_error']:.1e}" for k, v in test["rotation"].items())
              + "; inversion error " + ", ".join(f"{k} {v['rel_error']:.1e}" for k, v in test["inversion"].items()))
    print("bulk: largest cell average / one atom's vector: "
          + ", ".join(f"{name} {entry[f'max_{name}_average_over_atom_norm_all_cells']:.1e}" for name in vectorial)
          + ("; invariant vs pristine " + ", ".join(f"{p} {v:.1e}" for p, v in
                                                   entry["invariant_rel_diff_from_pristine"].items())
             if "invariant_rel_diff_from_pristine" in entry else ""))
    for label in consistency:
        c = summary["thickness_consistency"][label]
        zeros = f" ({c['n_zero_by_symmetry']} zero by symmetry)" if c["n_zero_by_symmetry"] else ""
        if c["n_families"]:
            print(f"thickness consistency {label}: max {c['max']:.1e}, median {c['median']:.1e}, "
                  f"{c['n_above_tol']} of {c['n_families']} families above {args.consistency_tol:g}{zeros}")
        else:
            print(f"thickness consistency {label}: no family with a vector{zeros}")
    print(f"coinciding groups at 8x: {len(rows_out)}, verdicts {verdicts}")
    if args.compare_with:
        arrays = summary["compared_with"]["arrays"]
        worst = max((entry.get("relative", 0.0) for entry in arrays.values()), default=float("nan"))
        identical = all(entry.get("bit_identical", entry.get("identical_counts", True)) for entry in arrays.values())
        print(f"against {summary['compared_with']['run_dir']}: {len(arrays)} arrays, largest relative difference "
              f"{worst:.1e}, bit-identical: {identical}")
    print(f"status: {summary['status']}; report in {folder / 'checks.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
