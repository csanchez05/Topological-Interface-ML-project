#!/usr/bin/env python3
"""Step 1.2 (ACE): every transformed bulk cell must give the pristine ACE vector, up to handedness.

The inversion-even blocks (nu1, nu2, nu3_even) of every cell must equal those of the pristine cell.
The odd block (nu3_odd) changes sign under inversion, so a cell made by an improper operation (the
other B20 enantiomer) must give minus the pristine odd vector. Its sign is compared with the
`handedness` column of the descriptor-free labels (gb_labels.csv: "same" -> +1, "opposite" -> -1;
the pristine cell and the orientation references are expected at +1). SOAP passes this step on all
cells only because it cannot see handedness; for ACE the sign test is the first real difference.

Reads the output of bulk_descriptor.py and writes bulk_invariance.csv and bulk_invariance.json into
the same folder. Exits with status 1 if any cell exceeds --tol or has the wrong odd sign.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ace_common as ac

sc = ac.sc
EXPECTED_SIGN = {"same": 1, "opposite": -1}


def expected_signs(index, labels_path):
    """+1 / -1 per bulk row: +1 for pristine and references, the label's handedness for the others."""
    labels = {row["family"]: row["handedness"] for row in sc.read_csv(labels_path)}
    signs, handedness = [], []
    for row in index:
        if row["kind"] != "transformed":
            signs.append(1)
            handedness.append("")
            continue
        family = f"{row['plane']}/{row['sym']}"
        if family not in labels:
            raise SystemExit(f"{family} has no row in {labels_path}")
        if labels[family] not in EXPECTED_SIGN:
            raise SystemExit(f"{family}: unknown handedness {labels[family]!r}")
        signs.append(EXPECTED_SIGN[labels[family]])
        handedness.append(labels[family])
    return np.array(signs), handedness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ac.add_ace_arguments(parser)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default: calculations/descriptors/ACE/standard/global/<tag> for the ACE settings given")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="largest accepted relative difference from the pristine vector, per block (default 1e-6)")
    parser.add_argument("--labels", type=Path, default=ac.GB_LABELS,
                        help="labels with a handedness column (default calculations/ML_dscribe/CoGe/labels/gb_labels.csv)")
    args = parser.parse_args()

    folder = ac.run_dir("global", args.run_dir or ac.ace_tag(ac.ace_settings(args)))
    with np.load(folder / ac.BULK_FILE) as data:
        blocks = ac.stored_blocks(data)
        X = {name: data[f"X_{name}"] for name in blocks}
    index = sc.read_csv(folder / "bulk_index.csv")
    pristine = [i for i, row in enumerate(index) if row["kind"] == "pristine"]
    if len(pristine) != 1:
        raise SystemExit("expected exactly one pristine row in bulk_index.csv")
    expected, handedness = expected_signs(index, args.labels)

    difference, sign = {}, None
    for name in blocks:
        reference = X[name][pristine[0]]
        if ac.BLOCKS[name][1] == 1:
            difference[name] = np.linalg.norm(X[name] - reference, axis=1) / np.linalg.norm(reference)
        else:
            # Distance to whichever of +pristine and -pristine is nearer, and which one that is.
            sign = np.sign(X[name] @ reference).astype(int)
            difference[name] = (np.linalg.norm(X[name] - sign[:, None] * reference, axis=1)
                                / np.linalg.norm(reference))
    has_odd = sign is not None
    worst = np.max(np.stack(list(difference.values())), axis=0)
    sign_ok = (sign == expected) if has_odd else np.ones(len(index), dtype=bool)
    passes = (worst <= args.tol) & sign_ok

    rows = []
    for i, row in enumerate(index):
        out = {"kind": row["kind"], "plane": row["plane"], "sym": row["sym"], "handedness": handedness[i]}
        out.update({f"rel_diff_{name}": f"{difference[name][i]:.6e}" for name in blocks})
        if has_odd:
            out.update({"odd_sign": int(sign[i]), "expected_odd_sign": int(expected[i])})
        out.update({f"within_species_spread_{name}": row[f"within_species_spread_{name}"] for name in blocks})
        out["passes"] = int(passes[i])
        rows.append(out)
    sc.write_csv(folder / "bulk_invariance.csv", rows, list(rows[0]))

    summary = {"tolerance": args.tol, "blocks": blocks, "n_cells": len(index), "labels": sc.relative_path(args.labels),
               "odd_block_compared_with": "+pristine for handedness 'same', -pristine for 'opposite'"
               if has_odd else "no odd block in this run", "planes": {}}
    header = "".join(f"{'max ' + name:>16}" for name in blocks)
    print(f"{'plane':>6} {'cells':>6}{header} {'sign wrong':>11} {'fail':>5}")
    for plane in sc.PLANES:
        chosen = np.array([row["kind"] == "transformed" and row["plane"] == plane for row in index])
        if not chosen.any():
            continue
        entry = {"n_cells": int(chosen.sum()),
                 "max_rel_diff": {name: float(difference[name][chosen].max()) for name in blocks},
                 "median_rel_diff": {name: float(np.median(difference[name][chosen])) for name in blocks},
                 "n_wrong_odd_sign": int((~sign_ok[chosen]).sum()),
                 "n_fail": int((~passes[chosen]).sum())}
        if has_odd:
            entry["n_opposite_handedness"] = int((expected[chosen] == -1).sum())
            entry["n_negative_odd_sign"] = int((sign[chosen] == -1).sum())
        summary["planes"][plane] = entry
        print(f"{plane:>6} {entry['n_cells']:>6}"
              + "".join(f"{entry['max_rel_diff'][name]:>16.3e}" for name in blocks)
              + f" {entry['n_wrong_odd_sign']:>11} {entry['n_fail']:>5}")
    summary["orientation_references"] = {
        row["sym"]: {**{name: float(difference[name][i]) for name in blocks},
                     **({"odd_sign": int(sign[i])} if has_odd else {})}
        for i, row in enumerate(index) if row["kind"] == "reference"}
    n_fail = int((~passes).sum())
    summary["n_wrong_odd_sign"] = int((~sign_ok).sum())
    summary["n_fail"] = n_fail
    summary["failures"] = [f"{row['kind']} {row['plane']} {row['sym']}" for row, ok in zip(index, passes) if not ok][:100]
    summary["status"] = "pass" if n_fail == 0 else "fail"
    sc.write_json(folder / "bulk_invariance.json", summary)
    if has_odd:
        transformed = np.array([row["kind"] == "transformed" for row in index])
        print(f"odd-block sign: {int((sign == expected)[transformed].sum())} of {int(transformed.sum())} transformed "
              f"cells match their handedness label ({int((expected[transformed] == -1).sum())} 'opposite')")
    print(f"status: {summary['status']} ({n_fail} of {len(index)} cells fail; tolerance {args.tol:g}); "
          f"report in {sc.relative_path(folder)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
