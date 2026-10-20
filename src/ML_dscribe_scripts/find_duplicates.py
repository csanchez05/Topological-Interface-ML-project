#!/usr/bin/env python3
"""Step 1.5: which grain boundaries have the same SOAP vector, and are they the same interface?

Takes one vector per GB family (one thickness), groups the families whose
vectors coincide within a tolerance, counts the distinct GBs per plane and
overall, and lists every coinciding group with the operations that built its
members. For each group the POSCARs are then compared directly, allowing any
rotation, mirror or inversion, to see whether the members really are one
structure built several times.

Rerun on another descriptor with --descriptor <tag or folder>; compare two
runs with compare_duplicates.py. --descriptor-type ace uses the ACE vectors
(src/ACE/CoGe) instead of SOAP, and writes into calculations/ML_ACE/CoGe/.

Outputs, in calculations/ML_dscribe/CoGe/standard/<descriptor tag>/<vector tag>/:
  distinct_gbs.csv         every family with its distinct-GB number
  duplicate_groups.csv     one row per coinciding group
  duplicate_members.csv    one row per family in a coinciding group
  label_merges.csv         operation labels that share a group
  near_misses.csv          closest pairs that do not coincide, with the structure test
  duplicates_summary.json  counts, tolerance, sensitivity, structure-test totals

Options --structure-census and --single-interfaces add two further counts; see the README.
"""
import argparse
import sys
import time
from collections import Counter

import numpy as np

import gb_labels
import ml_common as mc
import structure_match as sm
from ml_common import sc

SENSITIVITY = (1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
LABELS = ("handedness", "grain_class", "first_op", "rot_set")


def interface_signature(result, i, kind, planes):
    """What the vector of member i is built from, in terms of interface classes."""
    if kind == "global":
        return f"cell{result['cell_class'][i]}"
    if planes == "both":
        return "+".join(sorted((result["edge_interface"][i], result["mid_interface"][i])))
    return result["edge_interface"][i] if planes == "edge" else result["mid_interface"][i]


def group_verdict(result, kind, planes):
    """same cell: one rigid motion maps every member's whole cell onto the first.
    same interfaces: the cells differ, but the interfaces behind the vector match.
    different: neither."""
    n = len(result["cell_class"])
    if len(set(result["cell_class"])) == 1:
        return "same cell"
    if len({interface_signature(result, i, kind, planes) for i in range(n)}) == 1:
        return "same interfaces"
    return "different"


def structure_census(index, position_tol, log=None):
    """Rigid-motion classes of the whole cells, found from the POSCARs alone (no SOAP).

    Every cell is compared with one representative of each class already found
    on its plane. Returns one class number per row of `index`, and the shortest
    distance between two atoms of one species in any cell.
    """
    classes = np.full(len(index), -1)
    offset, shortest = 0, np.inf
    for plane in sc.PLANES:
        rows = np.flatnonzero(index["plane"].to_numpy() == plane)
        if rows.size == 0:
            continue
        cells = [sm.Structure.from_file(mc.PROJECT_ROOT / index["source"].iloc[i]) for i in rows]
        local = sm.classify(cells, lambda i, j: sm.match_cells(cells[i], cells[j], position_tol) is not None)
        classes[rows] = np.array(local) + offset
        offset += max(local) + 1
        shortest = min([shortest] + [cell.shortest_same_species_distance() for cell in cells])
        if log:
            log(f"  structure census, {plane}: {len(rows)} cells in {max(local) + 1} classes")
    return classes, shortest


def partition_differences(first, second):
    """Pairs of rows together in `first` but apart in `second`, and the reverse."""
    together_first = first[:, None] == first[None, :]
    together_second = second[:, None] == second[None, :]
    upper = np.triu(np.ones_like(together_first), 1)
    return (int((together_first & ~together_second & upper).sum()),
            int((~together_first & together_second & upper).sum()))


def single_interface_count(args, families):
    """Distinct single interfaces: the edge and mid vectors of every cell, grouped together."""
    stacked, owner = [], []
    for plane in sc.PLANE_NAMES:
        X, index, _ = mc.load_family_vectors(args.descriptor, args.kind, args.radius, plane, args.thickness,
                                             **mc.vector_source(args))
        keep = index["family"].isin(families).to_numpy()
        stacked.append(X[keep])
        owner.extend((family, plane) for family in index.loc[keep, "family"])
    D = mc.relative_distances(np.vstack(stacked))
    groups = mc.coincidence_groups(D, args.tol)
    surface = np.array([family.split("/")[0] for family, _ in owner])
    below, above = D[np.triu(D <= args.tol, 1)], D[np.triu(D > args.tol, 1)]
    edge = {family: groups[i] for i, (family, plane) in enumerate(owner) if plane == "edge"}
    mid = {family: groups[i] for i, (family, plane) in enumerate(owner) if plane == "mid"}
    return {
        "n_interfaces": len(owner), "n_distinct": int(groups.max() + 1),
        "n_distinct_per_plane": {p: int(len(set(groups[surface == p]))) for p in sc.PLANES},
        "n_cells_whose_two_interfaces_coincide": int(sum(edge[f] == mid[f] for f in edge if f in mid)),
        "largest_coinciding_distance": float(below.max()) if below.size else None,
        "smallest_other_distance": float(above.min()) if above.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_vector_arguments(parser)
    parser.add_argument("--tol", type=float, default=mc.DEFAULT_TOL,
                        help="two vectors coincide when their relative distance is at most this "
                             f"(default {mc.DEFAULT_TOL:g}; see the README for why)")
    parser.add_argument("--near-miss", type=float, default=10.0,
                        help="also structure-test pairs up to this many times the tolerance (default 10)")
    parser.add_argument("--no-structure-check", action="store_true", help="skip the POSCAR comparison")
    parser.add_argument("--position-tol", type=float, default=sm.DEFAULT_TOL,
                        help=f"atoms match within this distance in A (default {sm.DEFAULT_TOL})")
    parser.add_argument("--slab", type=float, default=sm.DEFAULT_HALFWIDTH,
                        help="half-width in A of the slab compared around each interface plane "
                             f"(default {sm.DEFAULT_HALFWIDTH:g})")
    parser.add_argument("--structure-census", action="store_true",
                        help="also sort ALL cells into rigid-motion classes from the POSCARs alone, as a count "
                             "of distinct GBs that does not depend on SOAP (about a minute for the full catalogue)")
    parser.add_argument("--single-interfaces", action="store_true",
                        help="also count distinct single interfaces (edge and mid vectors pooled); "
                             "needs two more passes over the descriptor file")
    args = parser.parse_args()
    mc.resolve_vector_arguments(args, parser)
    if args.thickness == "all":
        parser.error("find_duplicates.py works on one thickness; use pooled_duplicates.py for --thickness all")
    import pandas as pd

    started = time.time()
    X_all, index_all = mc.load_all_vectors(args.descriptor, args.kind, args.radius, args.planes, **mc.vector_source(args))
    noise = mc.thickness_noise(X_all, index_all, args.thickness, args.kind, args.radius)
    X, index, notes = mc.select_thickness(X_all, index_all, args.thickness, args.kind, args.radius)
    del X_all
    if len(index) == 0:
        parser.error(f"no structure with thickness {args.thickness} in {args.descriptor}")
    labels = gb_labels.load_labels().set_index("family")
    missing = sorted(set(index["family"]) - set(labels.index))
    if missing:
        parser.error(f"{len(missing)} families have no label, e.g. {missing[:3]}")
    family = index["family"].to_numpy()
    plane = index["plane"].to_numpy()

    D = mc.relative_distances(X)
    groups = mc.coincidence_groups(D, args.tol)
    diameters = mc.group_diameters(D, groups)
    sizes = np.bincount(groups)
    nearest, nearest_distance = mc.nearest_other(D, exclude=groups)
    upper = np.triu(np.ones_like(D, dtype=bool), 1)
    pair_distances = D[upper]
    below, above = pair_distances[pair_distances <= args.tol], pair_distances[pair_distances > args.tol]
    multi = np.flatnonzero(sizes > 1)

    # ---- direct structure comparison
    structure_cache = {}

    def structure(i):
        if i not in structure_cache:
            structure_cache[i] = sm.Structure.from_file(mc.PROJECT_ROOT / index["source"].iloc[i])
        return structure_cache[i]

    checks, shortest_same = {}, np.inf
    if not args.no_structure_check:
        for done, g in enumerate(multi, 1):
            members = np.flatnonzero(groups == g)
            cells = [structure(i) for i in members]
            shortest_same = min([shortest_same] + [cell.shortest_same_species_distance() for cell in cells])
            checks[g] = sm.compare_group(cells, args.slab, args.position_tol)
            for i in members:
                structure_cache.pop(i, None)
            if done % 50 == 0 or done == len(multi):
                print(f"  structure test: {done}/{len(multi)} groups, {time.time() - started:.0f} s", flush=True)

    census = None
    if args.structure_census and not args.no_structure_check:
        census, shortest_all = structure_census(index, args.position_tol, log=lambda text: print(text, flush=True))
        shortest_same = min(shortest_same, shortest_all)

    # ---- tables
    representative = {g: family[np.flatnonzero(groups == g)[0]] for g in range(len(sizes))}
    distinct_rows, member_rows, group_rows = [], [], []
    for i in range(len(family)):
        g = groups[i]
        row = {"family": family[i], "plane": plane[i], "sym": index["sym"].iloc[i],
               "distinct_id": f"D{g:04d}", "representative": representative[g], "group_size": int(sizes[g]),
               "nearest_other_family": family[nearest[i]] if np.isfinite(nearest_distance[i]) else "",
               "nearest_other_rel_distance": f"{nearest_distance[i]:.3e}" if np.isfinite(nearest_distance[i]) else ""}
        row.update({name: labels.at[family[i], name] for name in LABELS})
        if census is not None:
            row["structure_class"] = f"S{census[i]:04d}"
        distinct_rows.append(row)
    verdicts = Counter()
    n_cell_classes_total = int((sizes == 1).sum())
    for g in multi:
        members = np.flatnonzero(groups == g)
        result = checks.get(g)
        verdict = group_verdict(result, args.kind, args.planes) if result else ""
        if result:
            verdicts[verdict] += 1
            n_cell_classes_total += len(set(result["cell_class"]))
        group_rows.append({
            "distinct_id": f"D{g:04d}", "plane": "|".join(sorted(set(plane[members]))), "size": len(members),
            "families": "|".join(family[members]),
            "max_rel_distance": f"{diameters[g]:.3e}",
            "grain_classes": "|".join(sorted(set(labels.loc[family[members], "grain_class"]))),
            "first_ops": "|".join(sorted(set(labels.loc[family[members], "first_op"]))),
            "n_rot_sets": labels.loc[family[members], "rot_set"].nunique(),
            "structure_verdict": verdict,
            "n_cell_classes": len(set(result["cell_class"])) if result else "",
            "n_interface_classes": len(set(result["edge_interface"]) | set(result["mid_interface"])) if result else "",
        })
        for position, i in enumerate(members):
            row = {"distinct_id": f"D{g:04d}", "group_size": len(members), "family": family[i], "plane": plane[i],
                   "rel_distance_to_first": f"{D[members[0], i]:.3e}"}
            row.update({name: labels.at[family[i], name] for name in LABELS})
            row.update({"n_ops": labels.at[family[i], "n_ops"], "rotations": labels.at[family[i], "rotations"],
                        "first_op_as_written": labels.at[family[i], "first_op_as_written"]})
            if result:
                row.update({"cell_class": result["cell_class"][position],
                            "edge_interface": result["edge_interface"][position],
                            "mid_interface": result["mid_interface"][position],
                            "motion_onto_class_representative": result["cell_operation"][position],
                            "group_verdict": verdict})
            member_rows.append(row)

    merge_rows = []
    for name in LABELS:
        merged = Counter()
        for g in multi:
            members = np.flatnonzero(groups == g)
            values = tuple(sorted(set(labels.loc[family[members], name])))
            if len(values) > 1:
                merged[(plane[members[0]], values)] += 1
        for (surface, values), count in sorted(merged.items(), key=lambda item: (-item[1], item[0])):
            merge_rows.append({"label": name, "plane": surface, "labels_sharing_a_vector": " = ".join(values),
                               "n_groups": count})

    # ---- closest pairs that do not coincide
    near_rows = []
    close_i, close_j = np.nonzero(np.triu((D > args.tol) & (D <= args.tol * args.near_miss), 1))
    order = np.argsort(D[close_i, close_j])
    for done, k in enumerate(order, 1):
        i, j = close_i[k], close_j[k]
        row = {"family_a": family[i], "family_b": family[j], "rel_distance": f"{D[i, j]:.3e}",
               "same_grain_class": labels.at[family[i], "grain_class"] == labels.at[family[j], "grain_class"]}
        if not args.no_structure_check:
            result = sm.compare_group([structure(i), structure(j)], args.slab, args.position_tol)
            pair_a = sorted((result["edge_interface"][0], result["mid_interface"][0]))
            pair_b = sorted((result["edge_interface"][1], result["mid_interface"][1]))
            row["same_cell"] = result["cell_class"][0] == result["cell_class"][1]
            row["interfaces_a"], row["interfaces_b"] = "+".join(pair_a), "+".join(pair_b)
            row["same_two_interfaces"] = pair_a == pair_b
            row["shares_an_interface"] = bool(set(pair_a) & set(pair_b))
        near_rows.append(row)
    if near_rows and not args.no_structure_check:
        print(f"  structure test: {len(near_rows)} near-miss pairs, {time.time() - started:.0f} s", flush=True)

    out = mc.output_dir(args)
    mc.write_table(out / "distinct_gbs.csv", pd.DataFrame(distinct_rows))
    mc.write_table(out / "duplicate_groups.csv", pd.DataFrame(group_rows))
    mc.write_table(out / "duplicate_members.csv", pd.DataFrame(member_rows))
    mc.write_table(out / "label_merges.csv", pd.DataFrame(merge_rows, columns=["label", "plane",
                                                                              "labels_sharing_a_vector", "n_groups"]))
    near_columns = ["family_a", "family_b", "rel_distance", "same_grain_class"]
    if not args.no_structure_check:
        near_columns += ["same_cell", "interfaces_a", "interfaces_b", "same_two_interfaces", "shares_an_interface"]
    near = pd.DataFrame(near_rows, columns=near_columns)
    mc.write_table(out / "near_misses.csv", near)

    per_plane = {p: {"n_families": int((plane == p).sum()), "n_distinct": int(len(set(groups[plane == p]))),
                     "n_groups": int(sum(1 for g in multi if plane[np.flatnonzero(groups == g)[0]] == p)),
                     "n_families_in_groups": int(sum(sizes[groups[i]] > 1 for i in np.flatnonzero(plane == p)))}
                 for p in sc.PLANES if (plane == p).any()}
    summary = {
        "vector": mc.vector_settings(args),
        "n_families": len(family), "n_distinct": int(len(sizes)),
        "n_coinciding_groups": int(len(multi)), "n_families_in_groups": int(sizes[sizes > 1].sum()),
        "group_size_histogram": {str(size): int(count) for size, count in sorted(Counter(sizes[sizes > 1].tolist()).items())},
        "per_plane": per_plane,
        "n_groups_spanning_planes": int(sum(len(set(plane[groups == g])) > 1 for g in multi)),
        "tolerance": {
            "value": args.tol,
            "definition": "|a - b| / mean(|a|, |b|) <= tolerance; groups are chains of such pairs",
            "thickness_noise": noise,
            "largest_coinciding_distance": float(below.max()) if below.size else None,
            "smallest_other_distance": float(above.min()) if above.size else None,
            "largest_group_diameter": float(diameters.max()),
            "n_distinct_at_other_tolerances": {f"{t:g}": int(mc.coincidence_groups(D, t).max() + 1) for t in SENSITIVITY},
        },
        "labels_merged_in_groups": {name: int(sum(1 for row in merge_rows if row["label"] == name)) for name in LABELS},
        "groups_with_more_than_one": {name: int(sum(labels.loc[family[groups == g], name].nunique() > 1 for g in multi))
                                      for name in LABELS},
        "notes": notes,
        "run": mc.run_metadata(),
    }
    if not args.no_structure_check:
        summary["structure_check"] = {
            "position_tol_A": args.position_tol, "slab_halfwidth_A": args.slab,
            "shortest_same_species_distance_A": float(shortest_same) if np.isfinite(shortest_same) else None,
            "tolerance_is_safe": bool(2.0 * args.position_tol < shortest_same),
            "n_groups_tested": int(len(multi)), "verdicts": dict(verdicts),
            "n_distinct_cells_by_structure": n_cell_classes_total,
            "near_misses": {"range": [args.tol, args.tol * args.near_miss], "n_pairs": len(near_rows),
                            "n_same_cell": int(near["same_cell"].sum()),
                            "n_same_two_interfaces": int(near["same_two_interfaces"].sum()),
                            "n_sharing_an_interface": int(near["shares_an_interface"].sum())},
        }
    if census is not None:
        same_vector_other_cell, same_cell_other_vector = partition_differences(groups, census)
        summary["structure_census"] = {
            "what": "rigid-motion classes of the whole cells, from the POSCARs alone",
            "n_classes": int(census.max() + 1),
            "n_classes_per_plane": {p: int(len(set(census[plane == p]))) for p in per_plane},
            "pairs_with_one_vector_but_different_cells": same_vector_other_cell,
            "pairs_with_one_cell_but_different_vectors": same_cell_other_vector,
            "agrees_with_vector_groups": same_vector_other_cell == 0 and same_cell_other_vector == 0,
        }
    if args.single_interfaces and args.kind != "global":
        summary["single_interfaces"] = single_interface_count(args, set(family))
    summary["seconds"] = round(time.time() - started, 1)
    mc.write_json(out / "duplicates_summary.json", summary)

    # ---- report
    print(f"{len(family)} families at {args.thickness}x, vector {mc.vector_tag(args.kind, args.radius, args.planes, args.thickness)}, "
          f"descriptor {args.descriptor_type} {mc.descriptor_tag(args.descriptor, args.descriptor_type)}")
    if noise:
        print(f"thickness noise of one family's vector: max {noise['max']:.1e}, median {noise['median']:.1e}; "
              f"tolerance {args.tol:g}")
    else:
        print(f"tolerance {args.tol:g} (no other thickness to measure the noise against)")
    if below.size and above.size:
        print(f"coinciding pairs differ by at most {below.max():.1e}; the closest other pair differs by {above.min():.1e}")
    print(f"{'plane':6s} {'families':>9s} {'distinct':>9s} {'groups':>7s} {'in groups':>10s}")
    for p, entry in per_plane.items():
        print(f"{p:6s} {entry['n_families']:9d} {entry['n_distinct']:9d} {entry['n_groups']:7d} {entry['n_families_in_groups']:10d}")
    print(f"{'all':6s} {len(family):9d} {len(sizes):9d} {len(multi):7d} {int(sizes[sizes > 1].sum()):10d}")
    print("group sizes: " + ", ".join(f"{count} of {size}" for size, count in summary["group_size_histogram"].items()))
    print("groups whose members carry different labels: "
          + ", ".join(f"{name} {count}" for name, count in summary["groups_with_more_than_one"].items()))
    if not args.no_structure_check:
        check = summary["structure_check"]
        print("structure test of the groups: " + (", ".join(f"{count} {verdict}" for verdict, count in verdicts.items()) or "no groups"))
        print(f"distinct cells confirmed by structure: {check['n_distinct_cells_by_structure']}")
        print(f"near misses ({args.tol:g} to {args.tol * args.near_miss:g}): {check['near_misses']['n_pairs']} pairs, "
              f"{check['near_misses']['n_same_cell']} are the same cell")
        if not check["tolerance_is_safe"]:
            print(f"WARNING: position tolerance {args.position_tol} A is not below half the shortest "
                  f"same-species distance ({shortest_same:.3f} A)")
    if census is not None:
        entry = summary["structure_census"]
        print(f"structure census (no SOAP): {entry['n_classes']} distinct cells ("
              + ", ".join(f"{p}: {n}" for p, n in entry["n_classes_per_plane"].items()) + "); "
              + ("identical to the vector groups" if entry["agrees_with_vector_groups"] else
                 f"{entry['pairs_with_one_vector_but_different_cells']} pairs share a vector but not a cell, "
                 f"{entry['pairs_with_one_cell_but_different_vectors']} pairs share a cell but not a vector"))
    if "single_interfaces" in summary:
        single = summary["single_interfaces"]
        print(f"single interfaces: {single['n_distinct']} distinct among {single['n_interfaces']} "
              f"({', '.join(f'{p}: {n}' for p, n in single['n_distinct_per_plane'].items())})")
    print(f"tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
