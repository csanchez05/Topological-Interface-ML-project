#!/usr/bin/env python3
"""Collect an r_cut scan into one table and figures.

Reads every finished analysis under the scan folder: one-thickness runs
(find_duplicates.py + krr_classify.py) and pooled runs (pooled_duplicates.py +
krr_classify.py --thickness all), for both GB vectors (interface radius
r_cut + 2 A, and the fixed radius).

Writes into the scan folder:
  rcut_scan_summary.csv / .json   one row per cutoff, vector and thickness (4..12 or "all")
  rcut_scan_by_thickness.png      cutoff x thickness grids (when several thicknesses were run)
  rcut_scan_pooled.png            accuracy, distinct count and separation against the cutoff
                                  (pooled runs; for a single-thickness scan, that thickness)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_common as mc

sc = mc.sc  # soap_common, as loaded by ml_common

LABELS = ("plane", "grain_class", "handedness", "first_op", "rot_set")
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 of the dataviz reference palette
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # sequential ramp
VECTORS = (("matched", "Interface radius = r_cut + 2 Å (every disturbed atom)"),
           ("fixed", "Interface radius fixed (the same atoms at every cutoff)"))
_CACHE = {}


def family_vectors(descriptor_dir, radius):
    """Every stored vector of one descriptor and radius; only the latest is kept in memory."""
    key = (str(descriptor_dir), float(radius))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = mc.load_all_vectors(descriptor_dir, "interface", radius, "both")
    return _CACHE[key]


def recorded_descriptor(text):
    """The descriptor folder an analysis recorded that it used.

    Analyses run before 2026-10-03 recorded calculations/dscribe/CoGe/gb/<tag>;
    those runs are now found by tag in calculations/descriptors/dscribe/standard/.
    An 8x-only copy (<tag>_t8) maps to the full run, which holds the same 8x rows.
    """
    folder = sc.PROJECT_ROOT / text
    if (folder / "settings.json").is_file():
        return folder
    return sc.run_dir("both_interfaces", re.sub(r"_t[0-9-]+$", "", Path(text).name))


def bulk_vector(soap):
    X_bulk, bulk_index = sc.load_bulk(sc.soap_tag(soap))
    return X_bulk[(bulk_index["kind"] == "pristine").to_numpy()][0]


def read_run(folder, fixed_radius):
    """One row of the summary from one analysis folder, or None if it is incomplete."""
    krr_path = folder / "krr_summary.csv"
    pooled = (folder / "pooled_summary.json").exists()
    duplicates_path = folder / ("pooled_summary.json" if pooled else "duplicates_summary.json")
    if not (duplicates_path.exists() and krr_path.exists()):
        return None
    duplicates = json.loads(duplicates_path.read_text())
    vector = duplicates["vector"]
    descriptor_dir = recorded_descriptor(vector["descriptor_dir"])
    settings = json.loads((descriptor_dir / "settings.json").read_text())
    r_cut, radius, thickness = settings["soap"]["r_cut"], float(vector["radius"]), vector["thickness"]
    row = {
        "r_cut": r_cut, "interface_radius": radius, "thickness": thickness,
        "vector": "matched" if np.isclose(radius, r_cut + 2) else "fixed",
        "also_fixed": bool(np.isclose(radius, fixed_radius)),
        "n_max": settings["soap"]["n_max"], "l_max": settings["soap"]["l_max"], "sigma": settings["soap"]["sigma"],
        "vector_length": settings["vector_length"],
        "n_families": duplicates["n_families"],
        "n_structures": duplicates.get("n_structures", duplicates["n_families"]),
        "n_distinct": duplicates["n_distinct"],
        "tolerance": duplicates["tolerance"]["value"],
        "largest_coinciding_distance": duplicates["tolerance"]["largest_coinciding_distance"],
        "smallest_other_distance": duplicates["tolerance"]["smallest_other_distance"],
    }
    for plane, entry in duplicates["per_plane"].items():
        row[f"n_distinct_{plane}"] = entry["n_distinct"]
    if pooled:
        row["n_distinct_by_structure"] = duplicates["n_structure_classes"] or ""
        row["n_vector_groups"] = duplicates["n_vector_groups"]
        row["n_thickness_consistent_families"] = duplicates["n_thickness_consistent_families"]
        row["largest_distance_between_thicknesses"] = duplicates["largest_distance_between_thicknesses"]
        row["merged_pairs_of_different_structures"] = duplicates["n_merged_family_pairs_of_different_structure"]
    else:
        check = duplicates.get("structure_check", {})
        verdicts = check.get("verdicts", {})
        row["n_coinciding_groups"] = duplicates["n_coinciding_groups"]
        row["groups_same_cell"] = verdicts.get("same cell", 0)
        row["groups_not_same_cell"] = sum(count for verdict, count in verdicts.items() if verdict != "same cell")
        row["n_distinct_by_structure"] = check.get("n_distinct_cells_by_structure", "")
        row["near_miss_pairs"] = check.get("near_misses", {}).get("n_pairs", "")
        noise = duplicates["tolerance"].get("thickness_noise") or {}
        row["thickness_noise_max"] = noise.get("max", "")

    for entry in sc.read_csv(krr_path):
        if entry["grouping"] != "distinct":
            continue
        if entry["dataset"] == "all" and entry["label"] in LABELS:
            row[f"acc_{entry['label']}"] = float(entry["krr_accuracy"])
            row[f"baseline_{entry['label']}"] = float(entry["baseline_majority"])
            row[f"knn_{entry['label']}"] = float(entry["knn_accuracy"])
        elif entry["dataset"] != "all" and entry["label"] == "grain_class":
            row[f"acc_grain_class_{entry['dataset']}"] = float(entry["krr_accuracy"])
    krr = json.loads((folder / "krr_summary.json").read_text())
    row["structures_where_radius_exceeds_quarter_cell"] = krr["notes"].get("families_where_radius_exceeds_quarter_cell", 0)
    row["krr_seconds"] = krr["seconds"]

    bulk_path = sc.run_dir("global", sc.soap_tag(settings["soap"])) / "bulk_invariance.json"
    if bulk_path.exists():
        bulk = json.loads(bulk_path.read_text())
        row["bulk_max_rel_diff"] = max(entry["max_rel_diff"] for entry in bulk["planes"].values())

    # How far the GB vectors sit from the bulk vector, relative to its length.
    X, index = family_vectors(descriptor_dir, radius)
    chosen = np.isfinite(X).all(axis=1)
    if thickness != "all":
        chosen &= (index["thickness"] == thickness).to_numpy()
    reference = bulk_vector(settings["soap"])
    distance = np.linalg.norm(X[chosen].astype(np.float64) - reference, axis=1) / np.linalg.norm(reference)
    row["median_distance_from_bulk"] = float(np.median(distance))
    row["folder"] = sc.relative_path(folder)
    return row


# --------------------------------------------------------------------------- figures

def style(plt):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED})


def rows_of(rows, vector):
    """Rows of one vector choice; the fixed-radius view also takes matched rows at that radius."""
    if vector == "matched":
        return [row for row in rows if row["vector"] == "matched"]
    return [row for row in rows if row["vector"] == "fixed" or row["also_fixed"]]


def plot_lines(rows, path, structure_count, what):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    vectors = [(key, title) for key, title in VECTORS if rows_of(rows, key)]
    figure, axes = plt.subplots(len(vectors), 3, figsize=(13.5, 3.9 * len(vectors)), squeeze=False, facecolor=SURFACE)
    cutoffs = sorted({row["r_cut"] for row in rows})
    counts = [row["n_distinct"] for row in rows] + ([structure_count] if structure_count else [])

    def line(axis, x, y, color, label):
        axis.plot(x, y, color=color, linewidth=2, marker="o", markersize=6.5, markeredgecolor=SURFACE,
                  markeredgewidth=1.5, solid_capstyle="round", label=label, zorder=3)

    for r, (key, title) in enumerate(vectors):
        chosen = sorted(rows_of(rows, key), key=lambda row: row["r_cut"])
        x = [row["r_cut"] for row in chosen]
        for axis in axes[r]:
            axis.set_facecolor(SURFACE)
            axis.grid(True, color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(AXIS)
            axis.tick_params(length=0)
            axis.set_xticks(cutoffs)
            axis.set_xlabel("SOAP cutoff r_cut (Å)")

        accuracy = axes[r][0]
        for color, label, name in zip(SERIES, ("plane", "grain_class", "handedness"),
                                      ("Plane", "Operation (grain class)", "Handedness")):
            line(accuracy, x, [row.get(f"acc_{label}", np.nan) for row in chosen], color, name)
        baseline = np.mean([row["baseline_grain_class"] for row in chosen if "baseline_grain_class" in row])
        accuracy.axhline(baseline, color=MUTED, linewidth=1)
        accuracy.annotate(f"majority-class baseline for operation ({baseline:.2f})", (x[0], baseline),
                          xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
        accuracy.set_ylim(0, 1.05)
        accuracy.set_ylabel("KRR accuracy on held-out GBs")
        accuracy.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.77), fontsize=8,
                        labelcolor=SECONDARY)
        accuracy.set_title("Step 1.4: what the vector predicts", loc="left", fontsize=10, color=INK)

        distinct = axes[r][1]
        line(distinct, x, [row["n_distinct"] for row in chosen], SERIES[0], "Distinct GBs")
        if structure_count:
            distinct.axhline(structure_count, color=MUTED, linewidth=1)
            distinct.annotate(f"distinct structures ({structure_count:,})", (x[0], structure_count),
                              xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
        step = 5 if max(counts) - min(counts) <= 40 else 10 ** int(np.floor(np.log10(max(counts) - min(counts))))
        distinct.set_ylim(step * np.floor(min(counts) / step) - step, step * np.ceil(max(counts) / step) + step)
        distinct.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        distinct.set_ylabel(f"Distinct GBs by vector (of {chosen[0]['n_families']:,} families)")
        distinct.set_title("Step 1.5: how many GBs it tells apart", loc="left", fontsize=10, color=INK)

        gap = axes[r][2]
        line(gap, x, [row["smallest_other_distance"] for row in chosen], SERIES[0], "Closest two different GBs")
        line(gap, x, [row["largest_coinciding_distance"] for row in chosen], SERIES[1],
             "Farthest two copies of one GB")
        tolerance = chosen[0]["tolerance"]
        gap.axhline(tolerance, color=MUTED, linewidth=1)
        gap.annotate(f"counted as the same vector below {tolerance:g}", (x[0], tolerance), xytext=(0, -11),
                     textcoords="offset points", color=MUTED, fontsize=8)
        gap.set_yscale("log")
        values = [row[name] for row in chosen for name in ("smallest_other_distance", "largest_coinciding_distance")
                  if row[name]]
        gap.set_ylim(min(1e-5, 0.5 * min(values)), max(2e-3, 3 * max(values)))
        gap.set_ylabel("Relative distance between vectors")
        gap.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=SECONDARY)
        gap.set_title("Separation between vectors", loc="left", fontsize=10, color=INK)

    figure.tight_layout(rect=(0, 0, 1, 0.93 - 0.03 / len(vectors)), h_pad=5.0, w_pad=2.5)
    for r, (key, title) in enumerate(vectors):
        box = axes[r][0].get_position()
        figure.text(box.x0, box.y1 + 0.055 * 2 / len(vectors), title, fontsize=11, color=INK, fontweight="bold")
    figure.text(0.012, 0.992, what, fontsize=12, color=INK, fontweight="bold", va="top")
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def plot_grids(rows, path, structure_count):
    """Cutoff x thickness grids: one blue ramp per panel, the value written in every cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgb

    style(plt)
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    cutoffs = sorted({row["r_cut"] for row in rows})
    thicknesses = sorted({row["thickness"] for row in rows})
    vectors = [(key, title) for key, title in VECTORS if rows_of(rows, key)]
    panels = [
        ("acc_grain_class", "Step 1.4: operation accuracy (grain class)", lambda v: f"{v:.2f}", Normalize),
        ("n_distinct", "Step 1.5: distinct GB vectors" + (f" ({structure_count:,} structures)" if structure_count else ""),
         lambda v: f"{int(v):,}", Normalize),
        ("smallest_other_distance", "Closest two different GBs (relative distance ×10⁻⁴)",
         lambda v: f"{v * 1e4:.1f}", LogNorm),
    ]
    figure, axes = plt.subplots(len(vectors), 3, figsize=(14.5, 4.9 * len(vectors)), squeeze=False, facecolor=SURFACE)
    flagged = False
    for r, (key, title) in enumerate(vectors):
        chosen = {(row["r_cut"], row["thickness"]): row for row in rows_of(rows, key)}
        for c, (name, heading, text, norm_type) in enumerate(panels):
            axis = axes[r][c]
            values = np.full((len(thicknesses), len(cutoffs)), np.nan)
            for (r_cut, thickness), row in chosen.items():
                if row.get(name) not in (None, ""):
                    values[thicknesses.index(thickness), cutoffs.index(r_cut)] = row[name]
            finite = values[np.isfinite(values)]
            low, high = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
            if high <= low:
                low, high = low - 0.5 * abs(low) - 1e-12, high + 0.5 * abs(high) + 1e-12
            norm = norm_type(vmin=low, vmax=high)
            axis.set_facecolor(SURFACE)
            mesh = axis.pcolormesh(np.arange(len(cutoffs) + 1), np.arange(len(thicknesses) + 1),
                                   np.ma.masked_invalid(values), cmap=ramp, norm=norm,
                                   edgecolors=SURFACE, linewidth=2)
            for i, thickness in enumerate(thicknesses):
                for j, r_cut in enumerate(cutoffs):
                    if not np.isfinite(values[i, j]):
                        continue
                    red, green, blue = to_rgb(mesh.cmap(norm(values[i, j])))
                    dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.55
                    thin = chosen[(r_cut, thickness)].get("structures_where_radius_exceeds_quarter_cell", 0) > 0
                    flagged |= thin
                    axis.text(j + 0.5, i + 0.5, text(values[i, j]) + ("*" if thin else ""), ha="center", va="center",
                              fontsize=8, color="#ffffff" if dark else INK)
            axis.set_xticks(np.arange(len(cutoffs)) + 0.5, [f"{r_cut:g}" for r_cut in cutoffs])
            axis.set_yticks(np.arange(len(thicknesses)) + 0.5, [f"{thickness}x" for thickness in thicknesses])
            axis.tick_params(length=0)
            for side in axis.spines.values():
                side.set_visible(False)
            axis.set_xlabel("SOAP cutoff r_cut (Å)")
            if c == 0:
                axis.set_ylabel("Thickness of each grain")
            axis.set_title(heading, loc="left", fontsize=10, color=INK)
    figure.tight_layout(rect=(0, 0.03, 1, 0.93 - 0.03 / len(vectors)), h_pad=5.0, w_pad=2.0)
    for r, (key, title) in enumerate(vectors):
        box = axes[r][0].get_position()
        figure.text(box.x0, box.y1 + 0.05 * 2 / len(vectors), title, fontsize=11, color=INK, fontweight="bold")
    figure.text(0.012, 0.992, "Each thickness analysed on its own (darker = larger)", fontsize=12, color=INK,
                fontweight="bold", va="top")
    if flagged:
        figure.text(0.012, 0.008, "* the interface radius exceeds a quarter of the cell for some GBs, so the two "
                    "interfaces of the cell are no longer described separately", fontsize=8.5, color=SECONDARY)
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-dir", type=Path, default=mc.OUTPUT_ROOT / "rcut_scan_all")
    parser.add_argument("--fixed-radius", type=float, default=8.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    folders = sorted(folder for pattern in ("rc*/interface_r*", "by_thickness/rc*/interface_r*",
                                            "pooled/rc*/interface_r*") for folder in args.scan_dir.glob(pattern))
    rows = [row for folder in folders if (row := read_run(folder, args.fixed_radius))]
    if not rows:
        raise SystemExit(f"no finished runs under {args.scan_dir}")
    rows.sort(key=lambda row: (row["vector"] != "matched", str(row["thickness"]) != "all",
                               row["thickness"] if row["thickness"] != "all" else 0, row["r_cut"]))

    fields = list(dict.fromkeys(key for row in rows for key in row))
    sc.write_csv(args.scan_dir / "rcut_scan_summary.csv", [{key: row.get(key, "") for key in fields} for row in rows],
                 fields)
    structure_counts = {row["n_distinct_by_structure"] for row in rows if row["n_distinct_by_structure"] != ""}
    structure_count = max(structure_counts) if structure_counts else None
    sc.write_json(args.scan_dir / "rcut_scan_summary.json", {
        "rows": rows, "distinct_structures": sorted(structure_counts), "n_analyses": len(rows),
        "note": "accuracies use the 'distinct' grouping: copies of one GB (and, when pooled, its other "
                "thicknesses) never straddle training and testing",
        **sc.run_metadata()})

    pooled = [row for row in rows if row["thickness"] == "all"]
    single = [row for row in rows if row["thickness"] != "all"]
    thicknesses = sorted({row["thickness"] for row in single})
    if not args.no_plot:
        if pooled:
            plot_lines(pooled, args.scan_dir / "rcut_scan_pooled.png", structure_count,
                       f"All thicknesses pooled ({pooled[0]['n_structures']:,} structures per cutoff)")
        elif len(thicknesses) == 1:
            plot_lines(single, args.scan_dir / "rcut_scan_pooled.png", structure_count,
                       f"{thicknesses[0]}x thickness only ({single[0]['n_structures']:,} structures per cutoff)")
        if len(thicknesses) > 1:
            plot_grids(single, args.scan_dir / "rcut_scan_by_thickness.png", structure_count)

    print(f"{'vector':>8} {'thick':>5} {'r_cut':>5} {'radius':>6} {'plane':>6} {'operation':>9} {'(base)':>6} {'handed':>6} "
          f"{'distinct':>8} {'closest different':>17} {'farthest copies':>15} {'from bulk':>9}")
    for row in rows:
        print(f"{row['vector']:>8} {str(row['thickness']):>5} {row['r_cut']:>5g} {row['interface_radius']:>6g} "
              f"{row.get('acc_plane', np.nan):>6.3f} {row.get('acc_grain_class', np.nan):>9.3f} "
              f"{row.get('baseline_grain_class', np.nan):>6.2f} {row.get('acc_handedness', np.nan):>6.3f} "
              f"{row['n_distinct']:>8} {row['smallest_other_distance']:>17.2e} "
              f"{row['largest_coinciding_distance']:>15.2e} {row['median_distance_from_bulk']:>9.3f}")
    print(f"{len(rows)} analyses; tables and figures in {args.scan_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
