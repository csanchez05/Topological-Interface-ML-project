#!/usr/bin/env python3
"""Step 1.2: confirm every transformed bulk cell gives the pristine SOAP vector.

Reads the output of bulk_descriptor.py and writes bulk_invariance.csv and
bulk_invariance.json into the same folder.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import soap_common as sc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_soap_arguments(parser)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default: calculations/descriptors/dscribe/standard/global/<tag> "
                             "for the SOAP settings given")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="largest accepted relative difference from the pristine vector (default 1e-6)")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if any cell exceeds --tol")
    args = parser.parse_args()

    folder = args.run_dir or sc.run_dir("global", sc.soap_tag(sc.soap_settings(args)))
    with np.load(folder / "bulk_soap.npz") as data:
        X = data["X"]
    index = sc.read_csv(folder / "bulk_index.csv")
    pristine = [i for i, row in enumerate(index) if row["kind"] == "pristine"]
    if len(pristine) != 1:
        raise SystemExit("expected exactly one pristine row in bulk_index.csv")
    reference = X[pristine[0]]
    difference = np.linalg.norm(X - reference, axis=1) / np.linalg.norm(reference)

    rows = []
    for row, diff in zip(index, difference):
        rows.append({
            "kind": row["kind"], "plane": row["plane"], "sym": row["sym"],
            "rel_diff_from_pristine": f"{diff:.6e}",
            "within_species_spread": row["within_species_spread"],
            "passes": int(diff <= args.tol),
        })
    sc.write_csv(folder / "bulk_invariance.csv", rows, list(rows[0]))

    summary = {"tolerance": args.tol, "n_cells": len(index), "planes": {}}
    print(f"{'plane':>6} {'cells':>6} {'max rel diff':>14} {'median':>12} {'max within-species':>20} {'fail':>5}")
    for plane in sc.PLANES:
        chosen = np.array([row["kind"] == "transformed" and row["plane"] == plane for row in index])
        if not chosen.any():
            continue
        spread = np.array([float(row["within_species_spread"]) for row in index])[chosen]
        entry = {
            "n_cells": int(chosen.sum()),
            "max_rel_diff": float(difference[chosen].max()),
            "median_rel_diff": float(np.median(difference[chosen])),
            "max_within_species_spread": float(spread.max()),
            "n_fail": int((difference[chosen] > args.tol).sum()),
        }
        summary["planes"][plane] = entry
        print(f"{plane:>6} {entry['n_cells']:>6} {entry['max_rel_diff']:>14.3e} {entry['median_rel_diff']:>12.3e} "
              f"{entry['max_within_species_spread']:>20.3e} {entry['n_fail']:>5}")
    references = {row["sym"]: float(diff) for row, diff in zip(index, difference) if row["kind"] == "reference"}
    summary["orientation_references_rel_diff"] = references
    n_fail = int((difference > args.tol).sum())
    summary["n_fail"] = n_fail
    summary["status"] = "pass" if n_fail == 0 else "fail"
    sc.write_json(folder / "bulk_invariance.json", summary)
    print(f"status: {summary['status']} ({n_fail} of {len(index)} cells above {args.tol:g}); report in {folder}")
    return 1 if (n_fail and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
