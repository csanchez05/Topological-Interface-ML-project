#!/usr/bin/env python3
"""Checks on the output of the ACE gb_descriptor.py (step 1.3).

Writes checks.csv (one row per GB family) and checks.json (summary) into the run's both_interfaces
folder. Checks, per block:

  thickness consistency  the interface vector of one GB should not depend on how much bulk sits on
                         each side (thicknesses whose slabs meet, Lx/4 < R, are left out)

Differences are divided by the length of the pristine cell's vector of the same block (from
bulk_ace.npz of the same setting; without it, by the length of the vector compared against).
  edge vs mid            whether the two interfaces in a cell are the same
  close contacts         GBs whose closest atom pair is unphysically short
  missing species        an interface slab without Co or without Ge (gives NaN; a failure)
  handedness             global odd vector of the thickest cell over the pristine one, split by the
                         handedness label: about +1 for 'same' GBs, about 0 for 'opposite' ones
                         (half of the cell is each enantiomer); a report, not a test

The SOAP checks that need the nonbulk atoms (plane sanity, radius coverage) have no ACE counterpart.
Exits with status 1 if a slab misses a species.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ace_common as ac

sc = ac.sc


def relative_difference(vectors, reference, scale=None):
    """Length of the difference over `scale` (the pristine block's length), else over the reference's length.

    The odd block of a GB whose two grains are mirror images can vanish by symmetry, so a difference
    relative to the vector itself would divide by rounding noise.
    """
    denominator = scale if scale is not None else np.linalg.norm(reference)
    return np.linalg.norm(vectors - reference, axis=-1) / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ac.add_ace_arguments(parser)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the run's both_interfaces folder, or a tag (default: the tag of the ACE settings "
                             "given, under calculations/descriptors/ACE/standard)")
    parser.add_argument("--consistency-tol", type=float, default=1e-3,
                        help="largest accepted relative change of a vector across thicknesses (default 1e-3)")
    parser.add_argument("--contact", type=float, default=1.5,
                        help="pairs closer than this many A are listed as close contacts (default 1.5)")
    parser.add_argument("--labels", type=Path, default=ac.GB_LABELS,
                        help="labels with a handedness column (default calculations/ML_dscribe/CoGe/labels/gb_labels.csv)")
    args = parser.parse_args()

    folder = ac.run_dir("both_interfaces", args.run_dir or ac.ace_tag(ac.ace_settings(args)))
    index = sc.read_csv(folder / "index.csv")
    with np.load(ac.run_dir("global", folder) / ac.GLOBAL_FILE) as data:
        blocks = ac.stored_blocks(data)
        X_global = {name: data[f"X_{name}"].astype(np.float64) for name in blocks}
    with np.load(folder / ac.INTERFACE_FILE) as data:
        radii = data["radii"]
        per_plane = {(name, r): data[f"X_{name}_{sc.radius_key(radius)}"].astype(np.float64)
                     for name in blocks for r, radius in enumerate(radii)}
        counts = [data[f"n_{sc.radius_key(radius)}"] for radius in radii]
    both = {(name, r): sc.combine_planes(per_plane[name, r], counts[r]) for name in blocks for r in range(len(radii))}

    global_folder = ac.run_dir("global", folder)
    has_bulk = (global_folder / ac.BULK_FILE).exists()
    scale = ac.pristine_norms(global_folder, blocks) if has_bulk else dict.fromkeys(blocks)
    thickness = np.array([int(row["thickness"]) for row in index])
    lx = np.array([float(row["Lx"]) for row in index])
    min_pair = np.array([float(row["min_pair_distance"]) for row in index])
    families = defaultdict(list)
    for i, row in enumerate(index):
        families[row["family"]].append(i)

    missing = []
    for r, radius in enumerate(radii):
        empty = np.argwhere(counts[r] == 0)   # (structure, plane, species)
        missing += [f"{index[i]['source']} r{radius:g} {sc.PLANE_NAMES[p]} no {sc.SPECIES[s]}" for i, p, s in empty]

    rows = []
    for family, members in families.items():
        members = np.array(sorted(members, key=lambda i: thickness[i]))
        thickest = members[-1]
        row = {"family": family, "plane": index[thickest]["plane"], "n_thicknesses": len(members),
               "min_pair_distance": f"{min_pair[members].min():.6f}"}
        # The largest radius whose two slabs still fit in the thickest cell.
        fitting = [r for r, radius in enumerate(radii) if lx[thickest] / 4.0 >= radius]
        for name in blocks:
            if fitting:
                edge, mid = per_plane[name, fitting[-1]][thickest]
                row[f"{name}_edge_mid_rel_diff"] = (f"{relative_difference(edge, mid, scale[name]):.3e}"
                                                    if np.isfinite(edge).all() and np.isfinite(mid).all() else "")
            else:
                row[f"{name}_edge_mid_rel_diff"] = ""
            row[f"{name}_global_thickness_rel_diff"] = (
                f"{relative_difference(X_global[name][members], X_global[name][thickest], scale[name]).max():.3e}")
            for r, radius in enumerate(radii):
                fits = members[lx[members] / 4.0 >= radius]
                vectors = both[name, r]
                row[f"{name}_r{radius:g}_thickness_rel_diff"] = (
                    f"{relative_difference(vectors[fits], vectors[thickest], scale[name]).max():.3e}"
                    if len(fits) > 1 else "")
        for r, radius in enumerate(radii):
            row[f"r{radius:g}_thicknesses_that_fit"] = int((lx[members] / 4.0 >= radius).sum())
        rows.append(row)
    sc.write_csv(folder / "checks.csv", rows, list(rows[0]))

    def column(name):
        return np.array([float(row[name]) for row in rows if row[name] != ""])

    def stats(values):
        undefined = int((~np.isfinite(values)).sum())
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"n_families": 0, "n_undefined": undefined}
        return {"n_families": int(values.size), "n_undefined": undefined, "max": float(values.max()),
                "median": float(np.median(values)), "n_above_tol": int((values > args.consistency_tol).sum())}

    handedness = {}
    odd = next((name for name in blocks if ac.BLOCKS[name][1] == -1), None)
    if odd and has_bulk and args.labels.exists():
        bulk, bulk_index = ac.load_bulk(global_folder, [odd])
        pristine_vector = bulk[odd][np.flatnonzero(bulk_index["kind"].to_numpy() == "pristine")[0]]
        norm = float(np.linalg.norm(pristine_vector))
        labels = {row["family"]: row["handedness"] for row in sc.read_csv(args.labels)}
        for label in ("same", "opposite"):
            thickest = [max(members, key=lambda i: thickness[i]) for family, members in families.items()
                        if labels.get(family) == label]
            if thickest:
                # Projection on the pristine odd vector: +1 is pristine, -1 its mirror image, 0 a racemic mix.
                projection = X_global[odd][thickest] @ pristine_vector / norm ** 2
                handedness[label] = {"n_families": len(thickest), "median_projection": float(np.median(projection)),
                                     "min": float(projection.min()), "max": float(projection.max())}

    skipped = sc.read_csv(folder / "skipped.csv") if (folder / "skipped.csv").exists() else []
    contacts = np.flatnonzero(min_pair < args.contact)
    contact_families = sorted({index[i]["family"] for i in contacts})
    summary = {
        "run_dir": sc.relative_path(folder),
        "blocks": blocks,
        "n_structures": len(index),
        "n_families": len(families),
        "n_families_per_plane": {plane: sum(1 for row in rows if row["plane"] == plane) for plane in sc.PLANES},
        "n_skipped": len(skipped),
        "skipped": skipped,
        "missing_species": {"n": len(missing), "examples": missing[:50]},
        "difference_denominator": "length of the pristine vector of the block" if has_bulk
                                  else "length of the vector compared against (no bulk_ace.npz)",
        "thickness_consistency": {
            "tolerance": args.consistency_tol,
            "interface_vector": {name: {f"{radius:g}": stats(column(f"{name}_r{radius:g}_thickness_rel_diff"))
                                        for radius in radii} for name in blocks},
            "global_vector_for_comparison": {name: stats(column(f"{name}_global_thickness_rel_diff"))
                                             for name in blocks},
        },
        "edge_vs_mid": {name: {"n_families_compared": int(column(f"{name}_edge_mid_rel_diff").size),
                               "n_families_with_two_different_interfaces":
                                   int((column(f"{name}_edge_mid_rel_diff") > args.consistency_tol).sum())}
                        for name in blocks},
        "handedness_global_odd_projection": handedness or "not computed (no odd block, bulk file or labels)",
        "close_contacts": {"threshold_A": args.contact, "n_structures": int(contacts.size),
                           "n_families": len(contact_families), "smallest_pair_A": float(min_pair.min()),
                           "families": contact_families[:200]},
        "status": "fail" if missing else "pass",
    }
    sc.write_json(folder / "checks.json", summary)

    print(f"{summary['n_structures']} structures in {summary['n_families']} families, {summary['n_skipped']} skipped")
    for name in blocks:
        for radius, entry in summary["thickness_consistency"]["interface_vector"][name].items():
            if entry["n_families"]:
                print(f"thickness consistency, {name}, radius {radius} A: max {entry['max']:.2e}, "
                      f"{entry['n_above_tol']} of {entry['n_families']} families above {args.consistency_tol:g}")
        edge_mid = summary["edge_vs_mid"][name]
        print(f"  {name}: {edge_mid['n_families_with_two_different_interfaces']} of {edge_mid['n_families_compared']} "
              "families hold two different interfaces in one cell")
    for label, entry in handedness.items():
        print(f"global odd vector over pristine, '{label}' GBs (thickest cell): median {entry['median_projection']:+.3f} "
              f"(range {entry['min']:+.3f} to {entry['max']:+.3f}, {entry['n_families']} families)")
    print(f"close contacts: {contacts.size} structures in {len(contact_families)} families below {args.contact} A "
          f"(smallest {min_pair.min():.3f} A)")
    print(f"missing species in a slab: {len(missing)}")
    print(f"status: {summary['status']}; report in {sc.relative_path(folder)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
