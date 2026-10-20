#!/usr/bin/env python3
"""Step 1.5 on every thickness at once: do all structures of one GB share a vector?

find_duplicates.py looks at one thickness. This script takes every structure of
a descriptor run (all thicknesses of all families) and answers:

  - how many groups of coinciding vectors there are, and how many distinct GBs
    remain once the thicknesses of each family are counted as one GB;
  - which families are NOT thickness-consistent (their thicknesses fall into
    more than one vector group), and how far apart their thicknesses are;
  - which pairs of different structures are merged (share a vector group), and
    at which thicknesses;
  - how well separated different structures are, overall and per thickness.

"Different structures" uses the structure classes already established from the
POSCARs (the structure_class column of a find_duplicates.py --structure-census
run), so this script does no atom-by-atom comparison of its own: cells of
different thickness cannot be compared that way.

--descriptor-type ace uses the ACE vectors and writes into calculations/ML_ACE/CoGe/; the
structure classes stay those of the SOAP census (they come from the POSCARs).

Outputs, in calculations/ML_dscribe/CoGe/standard/<descriptor tag>/<vector tag>/:
  pooled_families.csv      one row per family
  pooled_merges.csv        pairs of families of different structure that share a vector group
  pooled_by_thickness.csv  counts and separations for each thickness
  pooled_summary.json      totals, tolerance, settings
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import gb_labels
import ml_common as mc

DEFAULT_CLASSES = mc.OUTPUT_ROOT / mc.DEFAULT_DESCRIPTOR / "interface_r8_both_t4" / "distinct_gbs.csv"


def load_structure_classes(path, families):
    """Integer structure class per family from a census table, or None when unavailable."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    if "structure_class" not in table.columns:
        return None
    lookup = dict(zip(table["family"], table["structure_class"]))
    if any(family not in lookup for family in families):
        return None
    return pd.factorize(np.array([lookup[family] for family in families]))[0]


def block_extremes(D, same):
    """(largest distance among pairs marked same, smallest among the others), upper triangle only."""
    upper = np.triu(np.ones(D.shape, dtype=bool), 1)
    inside, outside = D[upper & same], D[upper & ~same]
    return (float(inside.max()) if inside.size else None, float(outside.min()) if outside.size else None)


def analyse(X, family_code, thickness, structure, tol):
    """All pooled results as plain arrays and dicts; see main() for how they are written."""
    D = mc.relative_distances(X)
    vector_group = mc.coincidence_groups(D, tol)
    merged_group = mc.family_groups(family_code, vector_group)
    n_family = int(family_code.max()) + 1

    groups_per_family = np.zeros(n_family, dtype=int)
    spread = np.zeros(n_family)
    members_of = [np.flatnonzero(family_code == f) for f in range(n_family)]
    for f, members in enumerate(members_of):
        groups_per_family[f] = len(set(vector_group[members].tolist()))
        if len(members) > 1:
            spread[f] = D[np.ix_(members, members)].max()

    result = {"D": D, "vector_group": vector_group, "merged_group": merged_group,
              "groups_per_family": groups_per_family, "spread": spread}

    # Structure of each row: the census class when available, otherwise the family itself.
    reference = structure[family_code] if structure is not None else family_code
    same = reference[:, None] == reference[None, :]
    result["largest_same_structure"], result["smallest_different_structure"] = block_extremes(D, same)

    per_thickness = []
    for t in sorted(set(thickness.tolist())):
        rows = np.flatnonzero(thickness == t)
        block = D[np.ix_(rows, rows)]
        inside, outside = block_extremes(block, same[np.ix_(rows, rows)])
        per_thickness.append({
            "thickness": int(t), "n_structures": len(rows),
            "n_distinct_vectors": int(len(set(mc.coincidence_groups(block, tol).tolist()))),
            "largest_same_structure_distance": inside, "smallest_different_structure_distance": outside,
        })
    result["per_thickness"] = per_thickness

    # Pairs of rows of different structure that coincide, gathered by family pair.
    merges = {}
    close_i, close_j = np.nonzero(np.triu((D <= tol) & ~same, 1))
    for i, j in zip(close_i, close_j):
        key = (min(family_code[i], family_code[j]), max(family_code[i], family_code[j]))
        entry = merges.setdefault(key, {"n_pairs": 0, "smallest": np.inf, "thicknesses": set()})
        entry["n_pairs"] += 1
        entry["smallest"] = min(entry["smallest"], float(D[i, j]))
        entry["thicknesses"].add((int(thickness[i]), int(thickness[j])))
    result["merges"] = merges
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_vector_arguments(parser)
    parser.set_defaults(thickness="all")
    parser.add_argument("--tol", type=float, default=mc.DEFAULT_TOL,
                        help=f"two vectors coincide when their relative distance is at most this (default {mc.DEFAULT_TOL:g})")
    parser.add_argument("--structure-classes", type=Path, default=DEFAULT_CLASSES,
                        help="distinct_gbs.csv of a find_duplicates.py --structure-census run; its structure_class "
                             "column says which families are the same structure")
    args = parser.parse_args()
    mc.resolve_vector_arguments(args, parser)
    if args.thickness != "all":
        parser.error("pooled_duplicates.py always uses every thickness; use find_duplicates.py for one")
    import pandas as pd

    started = time.time()
    X, index, notes = mc.load_family_vectors(args.descriptor, args.kind, args.radius, args.planes, "all",
                                             **mc.vector_source(args))
    if len(index) == 0:
        parser.error(f"no structure in {args.descriptor}")
    family_code, families = pd.factorize(index["family"])
    thickness = index["thickness"].to_numpy()
    structure = load_structure_classes(args.structure_classes, families)
    if structure is None:
        notes["structure_classes"] = (f"no usable structure_class table at {mc.relative_path(args.structure_classes)}; "
                                      "every family is treated as its own structure")
    labels = gb_labels.load_labels().set_index("family")

    result = analyse(X, family_code, thickness, structure, args.tol)
    vector_group, merged_group = result["vector_group"], result["merged_group"]
    first_row = np.array([np.flatnonzero(family_code == f)[0] for f in range(len(families))])

    family_rows = []
    for f, family in enumerate(families):
        row = {"family": family, "plane": index["plane"].iloc[first_row[f]],
               "n_thicknesses": int((family_code == f).sum()),
               "n_vector_groups": int(result["groups_per_family"][f]),
               "thickness_consistent": bool(result["groups_per_family"][f] == 1),
               "largest_distance_between_thicknesses": f"{result['spread'][f]:.3e}",
               "distinct_id": f"P{merged_group[first_row[f]]:04d}",
               "grain_class": labels.at[family, "grain_class"] if family in labels.index else ""}
        if structure is not None:
            row["structure_class"] = f"S{structure[f]:04d}"
        family_rows.append(row)

    merge_rows = []
    for (a, b), entry in sorted(result["merges"].items(), key=lambda item: item[1]["smallest"]):
        merge_rows.append({
            "family_a": families[a], "family_b": families[b],
            "grain_class_a": labels.at[families[a], "grain_class"] if families[a] in labels.index else "",
            "grain_class_b": labels.at[families[b], "grain_class"] if families[b] in labels.index else "",
            "n_coinciding_structure_pairs": entry["n_pairs"],
            "smallest_rel_distance": f"{entry['smallest']:.3e}",
            "thicknesses": " ".join(f"{t}x-{u}x" for t, u in sorted(entry["thicknesses"])),
        })

    merged_of_family = merged_group[first_row]
    n_distinct = int(len(set(merged_of_family.tolist())))
    per_plane = {}
    for plane in mc.sc.PLANES:
        chosen = (index["plane"].to_numpy()[first_row] == plane)
        if chosen.any():
            per_plane[plane] = {"n_families": int(chosen.sum()),
                                "n_distinct": int(len(set(merged_of_family[chosen].tolist()))),
                                "n_thickness_consistent": int((result["groups_per_family"][chosen] == 1).sum())}
    summary = {
        "vector": mc.vector_settings(args),
        "n_structures": len(index), "n_families": len(families),
        "n_vector_groups": int(len(set(vector_group.tolist()))),
        "n_distinct": n_distinct,
        "n_structure_classes": int(structure.max()) + 1 if structure is not None else None,
        "n_thickness_consistent_families": int((result["groups_per_family"] == 1).sum()),
        "largest_distance_between_thicknesses": float(result["spread"].max()),
        "n_merged_family_pairs_of_different_structure": len(merge_rows),
        "per_plane": per_plane,
        "tolerance": {
            "value": args.tol,
            "definition": "|a - b| / mean(|a|, |b|) <= tolerance; groups are chains of such pairs",
            "largest_coinciding_distance": result["largest_same_structure"],
            "smallest_other_distance": result["smallest_different_structure"],
            "meaning": "largest distance between two structures of the same GB (any thicknesses) and smallest "
                       "distance between two structures of different GBs",
        },
        "per_thickness": result["per_thickness"],
        "notes": notes,
        "seconds": round(time.time() - started, 1),
        "run": mc.run_metadata(),
    }

    out = mc.output_dir(args)
    mc.write_table(out / "pooled_families.csv", pd.DataFrame(family_rows))
    mc.write_table(out / "pooled_merges.csv", pd.DataFrame(
        merge_rows, columns=["family_a", "family_b", "grain_class_a", "grain_class_b",
                             "n_coinciding_structure_pairs", "smallest_rel_distance", "thicknesses"]))
    mc.write_table(out / "pooled_by_thickness.csv", pd.DataFrame(result["per_thickness"]))
    mc.write_json(out / "pooled_summary.json", summary)

    print(f"{len(index)} structures in {len(families)} families, vector "
          f"{mc.vector_tag(args.kind, args.radius, args.planes, 'all')}, tolerance {args.tol:g}")
    print(f"vector groups: {summary['n_vector_groups']}; distinct GBs after joining thicknesses: {n_distinct}"
          + (f" (structure classes: {summary['n_structure_classes']})" if structure is not None else ""))
    print(f"thickness-consistent families: {summary['n_thickness_consistent_families']} of {len(families)}; "
          f"largest distance between thicknesses of one family {summary['largest_distance_between_thicknesses']:.2e}")
    print(f"pairs of different structures sharing a vector group: {len(merge_rows)}")
    same, different = result["largest_same_structure"], result["smallest_different_structure"]
    if same is not None and different is not None:
        print(f"farthest two structures of one GB {same:.2e}; closest two different GBs {different:.2e}")
    print(f"tables in {out} ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
