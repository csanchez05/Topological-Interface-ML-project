#!/usr/bin/env python3
"""Compare the coinciding groups found by find_duplicates.py for two descriptors.

Typical use: after raising n_max and l_max, see which GBs that shared a vector
before are now told apart, and which still coincide.

    python -B src/ML_dscribe_scripts/compare_duplicates.py rc6.0_n8_l6_s0.5 rc6.0_n10_l8_s0.5

Each argument is a descriptor tag (its find_duplicates output is looked up with
the vector options below) or a folder that holds distinct_gbs.csv.

Outputs, in calculations/ML_dscribe/CoGe/comparisons/<A>__<B>/<vector tag>/:
  comparison_groups.csv   every coinciding group of either run and what became of it
  comparison_summary.json counts
"""
import argparse
import sys
from pathlib import Path

import ml_common as mc


def resolve(argument, args):
    """(folder holding distinct_gbs.csv, short name) for a tag or a folder."""
    folder = Path(argument)
    if (folder / "distinct_gbs.csv").is_file():
        return folder, folder.resolve().parent.name + "_" + folder.resolve().name
    tag = mc.descriptor_tag(argument)
    return mc.OUTPUT_ROOT / tag / mc.vector_tag(args.kind, args.radius, args.planes, args.thickness), tag


def partition(table):
    """{family: frozenset of the families sharing its vector}."""
    members = table.groupby("distinct_id")["family"].apply(frozenset)
    return {family: members[group] for family, group in zip(table["family"], table["distinct_id"])}


def compare_partitions(first, second):
    """Compare two partitions of the same families.

    first, second: {family: frozenset of families in its group}. Returns one row
    per group of more than one family in either partition:
      unchanged  the same families coincide in both
      split      coincided in the first, told apart (at least partly) in the second
      merged     a group of the second that is not a group of the first
    """
    rows = []
    for group in sorted({g for g in first.values() if len(g) > 1}, key=sorted):
        pieces = {second[family] & group for family in group}
        extra = set().union(*(second[family] for family in group)) - group
        if len(pieces) == 1 and not extra:
            status = "unchanged"
        elif len(pieces) > 1:
            status = "split"
        else:
            status = "grew"
        rows.append({"found_in": "first" if status != "unchanged" else "both", "status": status,
                     "size": len(group), "families": "|".join(sorted(group)),
                     "n_pieces_in_second": len(pieces),
                     "pieces_in_second": " / ".join("|".join(sorted(piece)) for piece in sorted(pieces, key=sorted)),
                     "n_added_in_second": len(extra)})
    known = {g for g in first.values() if len(g) > 1}
    for group in sorted({g for g in second.values() if len(g) > 1} - known, key=sorted):
        pieces = {first[family] & group for family in group}
        if len(pieces) > 1:
            rows.append({"found_in": "second", "status": "merged", "size": len(group),
                         "families": "|".join(sorted(group)), "n_pieces_in_second": 1,
                         "pieces_in_second": "|".join(sorted(group)), "n_added_in_second": 0})
    return rows


def count_pairs(groups):
    return sum(len(g) * (len(g) - 1) // 2 for g in set(groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", help="descriptor tag or find_duplicates output folder")
    parser.add_argument("second", help="descriptor tag or find_duplicates output folder")
    parser.add_argument("--kind", default="interface", choices=mc.KINDS)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--planes", default="both", choices=mc.PLANE_CHOICES)
    parser.add_argument("--thickness", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    import pandas as pd

    (folder_a, name_a), (folder_b, name_b) = resolve(args.first, args), resolve(args.second, args)
    tables = []
    for folder in (folder_a, folder_b):
        if not (folder / "distinct_gbs.csv").is_file():
            parser.error(f"no distinct_gbs.csv in {folder}; run find_duplicates.py for that descriptor first")
        tables.append(pd.read_csv(folder / "distinct_gbs.csv", dtype={"plane": str}))
    common = sorted(set(tables[0]["family"]) & set(tables[1]["family"]))
    only = [sorted(set(table["family"]) - set(common)) for table in tables]
    tables = [table[table["family"].isin(common)] for table in tables]
    first, second = partition(tables[0]), partition(tables[1])
    rows = compare_partitions(first, second)

    # Structure verdicts of the second run, for the groups that still coincide.
    verdict = {}
    if (folder_b / "duplicate_groups.csv").is_file():
        groups_b = pd.read_csv(folder_b / "duplicate_groups.csv", dtype={"plane": str})
        if "structure_verdict" in groups_b:
            verdict = {"|".join(sorted(text.split("|"))): value
                       for text, value in zip(groups_b["families"], groups_b["structure_verdict"])}
    for row in rows:
        row["structure_verdict_in_second"] = verdict.get(row["families"], "") if row["status"] == "unchanged" else ""

    pairs_a, pairs_b = count_pairs(first), count_pairs(second)
    both = sum(len(g & second[f]) - 1 for f, g in first.items()) // 2
    status = pd.Series([row["status"] for row in rows]).value_counts().to_dict() if rows else {}
    planes = sorted({family.split("/")[0] for family in common})
    summary = {
        "first": {"name": name_a, "folder": mc.relative_path(folder_a)},
        "second": {"name": name_b, "folder": mc.relative_path(folder_b)},
        "n_families_compared": len(common),
        "families_only_in_first": len(only[0]), "families_only_in_second": len(only[1]),
        "n_distinct": {"first": len(set(first.values())), "second": len(set(second.values()))},
        "n_distinct_per_plane": {p: {"first": len({g for f, g in first.items() if f.startswith(p + "/")}),
                                     "second": len({g for f, g in second.items() if f.startswith(p + "/")})}
                                 for p in planes},
        "coinciding_pairs": {"first": pairs_a, "second": pairs_b, "in_both": both,
                             "separated_by_second": pairs_a - both, "new_in_second": pairs_b - both},
        "groups": status,
        "closest_pair_that_does_not_coincide": {
            name: {p: float(table.loc[table["plane"] == p, "nearest_other_rel_distance"].min()) for p in planes}
            for name, table in zip(("first", "second"), tables)},
        "run": mc.run_metadata(),
    }
    out = args.out_dir or (mc.OUTPUT_ROOT / "comparisons" / f"{name_a}__{name_b}"
                           / mc.vector_tag(args.kind, args.radius, args.planes, args.thickness))
    columns = ["found_in", "status", "size", "families", "n_pieces_in_second", "pieces_in_second",
               "n_added_in_second", "structure_verdict_in_second"]
    mc.write_table(out / "comparison_groups.csv", pd.DataFrame(rows, columns=columns))
    mc.write_json(out / "comparison_summary.json", summary)

    print(f"{len(common)} families compared: {name_a} -> {name_b}")
    print(f"distinct GBs: {summary['n_distinct']['first']} -> {summary['n_distinct']['second']} ("
          + ", ".join(f"{p}: {v['first']} -> {v['second']}" for p, v in summary["n_distinct_per_plane"].items()) + ")")
    print(f"coinciding pairs: {pairs_a} -> {pairs_b}; {both} coincide in both, {pairs_a - both} are told apart "
          f"by the second, {pairs_b - both} coincide only in the second")
    closest = summary["closest_pair_that_does_not_coincide"]
    print("closest pair that does not coincide: "
          + ", ".join(f"{p}: {closest['first'][p]:.1e} -> {closest['second'][p]:.1e}" for p in planes))
    print("groups: " + (", ".join(f"{count} {name}" for name, count in status.items()) or "none"))
    if verdict:
        kept = [row["structure_verdict_in_second"] for row in rows if row["status"] == "unchanged"]
        print("groups that still coincide, by structure test: "
              + (", ".join(f"{count} {name or 'not tested'}" for name, count in pd.Series(kept).value_counts().items())
                 or "none"))
    print(f"tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
