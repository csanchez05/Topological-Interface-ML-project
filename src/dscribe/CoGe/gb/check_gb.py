#!/usr/bin/env python3
"""Checks on the output of gb_descriptor.py.

Writes checks.csv (one row per GB family) and checks.json (summary) into
the run's both_interfaces folder. Checks:

  thickness consistency  the interface vector of one GB should not depend on
                         how much bulk sits on each side
  plane sanity           disturbed atoms must sit near x = 0 or x = Lx/2
  radius coverage        share of disturbed atoms inside each radius
  close contacts         GBs whose closest atom pair is unphysically short
  edge vs mid            whether the two interfaces in a cell are the same
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import soap_common as sc


def relative_difference(vectors, reference):
    return np.linalg.norm(vectors - reference, axis=-1) / np.linalg.norm(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_soap_arguments(parser)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the run's both_interfaces folder, or a tag (default: the tag of the "
                             "SOAP settings given, under calculations/descriptors/dscribe/standard)")
    parser.add_argument("--consistency-tol", type=float, default=1e-3,
                        help="largest accepted relative change of a vector across thicknesses (default 1e-3)")
    parser.add_argument("--plane-margin", type=float, default=3.0,
                        help="disturbed atoms may sit up to r_cut + this many A from a plane (default 3)")
    parser.add_argument("--contact", type=float, default=1.5,
                        help="pairs closer than this many A are listed as close contacts (default 1.5)")
    args = parser.parse_args()

    folder = sc.run_dir("both_interfaces", args.run_dir or sc.soap_tag(sc.soap_settings(args)))
    settings = json.loads((folder / "settings.json").read_text())
    r_cut = settings["soap"]["r_cut"]
    index = sc.read_csv(folder / "index.csv")
    with np.load(sc.run_dir("global", folder) / sc.GLOBAL_FILE) as data:
        X_global = data["X"]
    with np.load(folder / sc.INTERFACE_FILE) as data:
        radii = data["radii"]
        both = [sc.combine_planes(data[f"X_{sc.radius_key(radius)}"], data[f"n_{sc.radius_key(radius)}"])
                for radius in radii]
        nonbulk_both = sc.combine_planes(data["X_nonbulk"], data["n_nonbulk"])
        X_nonbulk, n_nonbulk = data["X_nonbulk"], data["n_nonbulk"]
        n_nonbulk_within = data["n_nonbulk_within"]

    thickness = np.array([int(row["thickness"]) for row in index])
    lx = np.array([float(row["Lx"]) for row in index])
    min_pair = np.array([float(row["min_pair_distance"]) for row in index])
    farthest = np.array([max(float(row["max_nonbulk_dist_edge"]), float(row["max_nonbulk_dist_mid"]))
                         for row in index])
    n_disturbed = n_nonbulk.sum(axis=(1, 2))
    families = defaultdict(list)
    for i, row in enumerate(index):
        families[row["family"]].append(i)

    rows = []
    for family, members in families.items():
        members = np.array(sorted(members, key=lambda i: thickness[i]))
        thickest = members[-1]
        row = {"family": family, "plane": index[thickest]["plane"], "n_thicknesses": len(members),
               "min_pair_distance": f"{min_pair[members].min():.6f}",
               "n_nonbulk_edge": index[thickest]["n_nonbulk_edge"],
               "n_nonbulk_mid": index[thickest]["n_nonbulk_mid"],
               "max_nonbulk_dist": f"{farthest[members].max():.4f}"}
        # The two interfaces of the thickest cell: 0 means they are the same interface.
        edge, mid = X_nonbulk[thickest, 0].astype(float), X_nonbulk[thickest, 1].astype(float)
        row["edge_mid_rel_diff"] = (f"{relative_difference(edge, mid):.3e}"
                                    if np.isfinite(edge).all() and np.isfinite(mid).all() else "")
        row["nonbulk_thickness_rel_diff"] = f"{relative_difference(nonbulk_both[members], nonbulk_both[thickest]).max():.3e}"
        row["global_thickness_rel_diff"] = f"{relative_difference(X_global[members], X_global[thickest]).max():.3e}"
        for r, radius in enumerate(radii):
            # A cell narrower than 4 radii cannot hold the full slab around both planes.
            fits = members[lx[members] / 4.0 >= radius]
            row[f"r{radius:g}_thickness_rel_diff"] = (
                f"{relative_difference(both[r][fits], both[r][thickest]).max():.3e}" if len(fits) > 1 else "")
            row[f"r{radius:g}_thicknesses_that_fit"] = len(fits)
        rows.append(row)
    sc.write_csv(folder / "checks.csv", rows, list(rows[0]))

    def column(name):
        return np.array([float(row[name]) for row in rows if row[name] != ""])

    def stats(values):
        # A family with no disturbed atom has no nonbulk vector, so its difference is undefined.
        undefined = int((~np.isfinite(values)).sum())
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"n_families": 0, "n_undefined": undefined}
        return {"n_families": int(values.size), "n_undefined": undefined, "max": float(values.max()),
                "median": float(np.median(values)), "n_above_tol": int((values > args.consistency_tol).sum())}

    skipped = sc.read_csv(folder / "skipped.csv") if (folder / "skipped.csv").exists() else []
    plane_failures = [index[i]["source"] for i in np.flatnonzero(farthest > r_cut + args.plane_margin)]
    undisturbed = [index[i]["source"] for i in np.flatnonzero(n_disturbed == 0)]
    edge_mid = column("edge_mid_rel_diff")
    contacts = np.flatnonzero(min_pair < args.contact)
    contact_families = sorted({index[i]["family"] for i in contacts})
    summary = {
        "run_dir": sc.relative_path(folder),
        "n_structures": len(index),
        "n_families": len(families),
        "n_families_per_plane": {plane: sum(1 for row in rows if row["plane"] == plane) for plane in sc.PLANES},
        "n_skipped": len(skipped),
        "skipped": skipped,
        "plane_sanity": {"limit_A": r_cut + args.plane_margin, "farthest_disturbed_atom_A": float(farthest.max()),
                         "n_failures": len(plane_failures), "failures": plane_failures[:50]},
        "structures_without_disturbed_atoms": {"n": len(undisturbed), "examples": undisturbed[:20]},
        "radius_coverage": {f"{radius:g}": float(n_nonbulk_within[:, r].sum() / max(n_disturbed.sum(), 1))
                            for r, radius in enumerate(radii)},
        "thickness_consistency": {
            "tolerance": args.consistency_tol,
            "nonbulk_vector": stats(column("nonbulk_thickness_rel_diff")),
            "interface_vector": {f"{radius:g}": stats(column(f"r{radius:g}_thickness_rel_diff")) for radius in radii},
            "global_vector_for_comparison": stats(column("global_thickness_rel_diff")),
        },
        "edge_vs_mid": {"n_families_compared": int(edge_mid.size),
                        "n_families_with_two_different_interfaces": int((edge_mid > args.consistency_tol).sum())},
        "close_contacts": {"threshold_A": args.contact, "n_structures": int(contacts.size),
                           "n_families": len(contact_families), "smallest_pair_A": float(min_pair.min()),
                           "families": contact_families[:200]},
    }
    sc.write_json(folder / "checks.json", summary)

    print(f"{summary['n_structures']} structures in {summary['n_families']} families, {summary['n_skipped']} skipped")
    print(f"plane sanity: farthest disturbed atom {farthest.max():.2f} A from a plane "
          f"(limit {r_cut + args.plane_margin:.1f}), {len(plane_failures)} failures")
    print("share of disturbed atoms inside each radius: "
          + ", ".join(f"{radius} A: {share:.1%}" for radius, share in summary["radius_coverage"].items()))
    consistency = summary["thickness_consistency"]
    for radius, entry in consistency["interface_vector"].items():
        if entry["n_families"]:
            print(f"thickness consistency, radius {radius} A: max {entry['max']:.2e}, "
                  f"{entry['n_above_tol']} of {entry['n_families']} families above {args.consistency_tol:g}")
    if consistency["global_vector_for_comparison"]["n_families"]:
        print(f"global vector changes with thickness by up to {consistency['global_vector_for_comparison']['max']:.2e} (expected)")
    print(f"{summary['edge_vs_mid']['n_families_with_two_different_interfaces']} of {edge_mid.size} families "
          "hold two different interfaces in one cell")
    print(f"{len(undisturbed)} structures have no disturbed atom at all (the cell is plain bulk)")
    print(f"close contacts: {contacts.size} structures in {len(contact_families)} families below {args.contact} A "
          f"(smallest {min_pair.min():.3f} A)")
    print(f"report in {folder}")
    return 1 if plane_failures else 0


if __name__ == "__main__":
    sys.exit(main())
