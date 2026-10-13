#!/usr/bin/env python3
"""Figures that explain the descriptor study, one folder per question.

Reads the outputs of descriptor_sensitivity.py, selection_benchmark.py, the covariant run,
geometry_labels.py, krr_classify.py --regression and representation_study.py (no new computation
beyond small re-alignments of stored maps) and writes into plots/CoGe/dim-reduction/SOAP/:

  descriptor_settings/   which SOAP setting changes which GBs look alike (part C)
      knob_trends.png          each score against the value of r_cut, n_max, l_max, sigma
      weighting.png            pow and poly radial weighting against no weighting, per r_cut
      region_and_processing.png  which atoms are averaged, and how the vector is scaled
      knob_ranking.png         the largest change each knob makes
      pca_dimensions.png       how many principal components the GBs need, per setting
      cost.png                 vector length against how much a setting changes
      overview.png             every setting and every score in one chart
  maps_by_setting/       PCA, t-SNE and UMAP maps for every value of one knob, side by side
      <pca|tsne|umap>_<knob>.png
  seed_stability/        do t-SNE/UMAP maps and their picks survive a new random seed? (item 3)
      <tsne|umap>_seeds.png    the maps of every seed with their 30 FPS picks
      pick_frequency.png       how many seeds pick each GB
      seed_vs_setting.png      a setting's change against the seed-to-seed spread
  thickness/             does the vector follow the GB or the cell thickness?
      pc1_by_thickness.png     PC1 of every structure against thickness, global and interface vector
      variance_shares.png      share of the spread explained by thickness and by the GB
  selection/             which way of picking GBs teaches a model most from the fewest (part D)
      learning_curves_<label>.png  accuracy and balanced accuracy against the number of GBs labelled
      grain_class_per_plane.png    the same grain-class predictions scored on each plane (benchmark v2)
      learning_curves_free_labels.png  R^2 of the geometry labels against GBs labelled (benchmark v2)
      coverage.png             how well the picks cover the descriptor space
      existing_lists.png       the six existing lists of 30 against every strategy at n = 30
      picks_full_space.png, picks_map_rules.png  where each strategy picks, on the three maps
      descriptor_efficiency.png  FPS and random learning on every descriptor setting (benchmark v1 only)
  covariant/             the covariant (lambda-SOAP) vectors, kept for Stage 3
      vector_sizes.png         polar and axial vector size at each interface, and their mean
  free_labels/           the shift-dependent geometry labels (geometry_labels.py)
      by_grain_class.png       each label by plane and grain class
      edge_vs_mid.png          the two interfaces of one cell
      krr_r2.png               Stage 1.4 KRR on the labels (krr_classify.py --regression)
  representation_response/  fifth study (representation_study.py, phase 1 and 2 together)
      bulk_test.png            covariant bulk averages; whole cell against its two interfaces
      maps_<method>.png        the map of A, B, C, D, for all GBs and per plane
      pair_changes.png         map change per pair of representations, against the noise floor
      sensitivity.png, sensitivity_by_plane.png  map change / raw-descriptor change
      organisation.png         kNN accuracy (plane, grain class, edge/mid) and kNN R^2 (geometry labels)
      faithfulness.png         trustworthiness, continuity, triplet accuracy
      effects.png, map_effects.png  two-way PERMANOVA / ANOVA: region against covariance
      method_numbers.png       each method's own diagnostics
      thickness.png            the thickness check
  settings_grid/         sixth study (representation_study.py --study settings_grid)
      variance_shares.png      representation, settings, interaction, seed noise
      anchor_distance.png      each setting's change from the reference setting
      between_settings_and_representations.png

Sections whose inputs are missing are skipped with a note. Existing files are replaced, or with
--archive-old ROOT first moved to ROOT/<their path in the project> (never overwriting a file there);
the six lists in FPS/ and median/ (select_gbs.py) are not touched. --done-marker FILE is written
when every chosen section has been drawn (the driver uses it to recognise a finished plot task).

Figures not drawn from the full run on the real structures (smoke tests, synthetic data) get
--title-prefix, which starts every title with that text. Such figures, and any figure drawn from
inputs under archive/, are refused in plots/CoGe/ and results/.

  python -B src/ML_dscribe_scripts/plot_descriptor_study.py            # on a compute node
  python -B src/ML_dscribe_scripts/plot_descriptor_study.py --only thickness selection
  python -B src/ML_dscribe_scripts/plot_descriptor_study.py --title-prefix "SMOKE TEST" \
      --sensitivity-dir <smoke outputs> --out-dir <a folder outside plots/CoGe and results>
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import descriptor_sensitivity as ds
import geometry_labels as gl
import ml_common as mc
import selection_benchmark as sb
from ml_common import sc

OUT_DIR = mc.plot_root("soap")
SENSITIVITY_DIR = mc.OUTPUT_ROOT / "descriptor_sensitivity"
BENCHMARK_DIR = mc.OUTPUT_ROOT / "selection_benchmark"
COVARIANT_DIR = sc.COVARIANT_ROOT / "both_interfaces" / ds.soap_tag(ds.REFERENCE)
REPRESENTATION_DIR = mc.OUTPUT_ROOT / "representation_study"
GRID_DIR = mc.OUTPUT_ROOT / "settings_grid"
LABELS_FILE = gl.OUT_DIR / gl.CSV_NAME
KRR_DIR = mc.OUTPUT_ROOT / ds.soap_tag(ds.REFERENCE) / "interface_r8_both_t4_no_boundary"
SECTIONS = ("descriptor_settings", "maps_by_setting", "seed_stability", "thickness", "selection", "covariant",
            "free_labels", "representation_response", "settings_grid")
# Folders that hold only results from the real structures.
PROTECTED = (sc.PROJECT_ROOT / "plots" / "CoGe", sc.PROJECT_ROOT / "results")
TITLE_PREFIX = ""  # set by --title-prefix
ARCHIVE_ROOT = None  # set by --archive-old

INK, MUTED, GRID = ds.INK, ds.MUTED, ds.GRID
BLUE, ORANGE, AQUA, YELLOW = sb.SLOTS
GRAY = sb.GRAY
NUMERIC_KNOBS = ("r_cut", "n_max", "l_max", "sigma")
UNITS = {"r_cut": "r_cut (Å)", "n_max": "n_max (radial functions)", "l_max": "l_max (angular channels)",
         "sigma": "sigma (Å, Gaussian width)"}
# Scores against the baseline: column, label, value for "no change".
SCORES = (("spearman", "Distance rank correlation", 1.0),
          ("knn10_kept", "10 nearest neighbours kept", 1.0),
          ("fps_overlap", f"FPS picks shared (of {ds.N_PICKS})", float(ds.N_PICKS)),
          ("procrustes_pca", "PCA map change (Procrustes)", 0.0))
STRATEGY_COLORS = {"random": GRAY, "stratified": BLUE, "fps": ORANGE, "kmeans": AQUA, "gp_variance": YELLOW}
MAP_COLORS = {"pca": BLUE, "tsne": ORANGE, "umap": AQUA}
STRATEGY_NAMES = {"random": "random", "stratified": "stratified (plane x grain class)", "fps": "FPS, full space",
                  "kmeans": "k-means, full space", "gp_variance": "GP variance (label-free active learning)",
                  "map_fps_pca": "FPS on PCA map", "map_kmeans_pca": "k-means on PCA map",
                  "map_fps_tsne": "FPS on t-SNE map", "map_kmeans_tsne": "k-means on t-SNE map",
                  "map_fps_umap": "FPS on UMAP map", "map_kmeans_umap": "k-means on UMAP map"}
LABEL_NAMES = {"grain_class": "grain class", "plane": "plane", "handedness": "handedness"}
PLANE_COLORS = {"100": BLUE, "110": ORANGE, "111": AQUA}
GEOMETRY_NAMES = {"d_min": "shortest distance d_min (Å)", "log1p_n_close": "log(1 + close pairs per nm²)",
                  "cn_excess": "coordination change per nm²", "cn_defect": "coordination defect per nm²"}
# Fifth and sixth studies (representation_study.py).
REP_COLORS = {"A": BLUE, "B": ORANGE, "C": AQUA, "D": YELLOW, "C0": GRAY}
REP_NAMES = {"A": "A standard, interface", "B": "B standard, whole cell", "C": "C covariant, interface",
             "D": "D covariant, whole cell", "C0": "C0 = C with lambda 0"}
SUBSET_NAMES = {"pooled": "all GBs", "100": "100 GBs", "110": "110 GBs", "111": "111 GBs"}
PAIRS = ("A-B", "A-C", "A-D", "B-C", "B-D", "C-D")
METHOD_NAMES = {"pca": "PCA", "umap": "UMAP", "kpca": "kernel PCA", "ica": "ICA", "lda_plane": "LDA (plane)",
                "lda_grain": "LDA (grain class)", "pls_plane": "PLS (plane)", "pls_grain": "PLS (grain class)",
                "pls_geometry": "PLS (geometry)", "pcovr_plane": "PCovR (plane)", "pcovr_grain": "PCovR (grain class)",
                "pcovr_geometry": "PCovR (geometry)", "mds": "metric MDS", "isomap": "Isomap", "tsne": "t-SNE",
                "pacmap": "PaCMAP", "diffusion": "diffusion map", "sketchmap": "sketch-map", "raw": "raw descriptors"}
PAIR_VARIABLES = (("procrustes", "Procrustes disparity"), ("jaccard15", "1 - Jaccard, 15 neighbours"),
                  ("jaccard50", "1 - Jaccard, 50 neighbours"), ("spearman", "1 - distance rank correlation"),
                  ("ari", "1 - ARI, 8 Ward clusters"))
ORGANISATION = {"knn_plane": "plane (kNN accuracy)", "knn_grain_class": "grain class (kNN accuracy)",
                "knn_interface": "edge or mid (kNN accuracy)", "knn_grain_class@100": "grain class, 100 GBs",
                "knn_grain_class@110": "grain class, 110 GBs", "knn_grain_class@111": "grain class, 111 GBs",
                **{f"knn_r2_{t}": f"{t} (kNN R²)" for t in gl.TARGETS}}
FAITHFULNESS = (("trustworthiness", "trustworthiness (15 neighbours)"), ("continuity", "continuity (15 neighbours)"),
                ("triplet_accuracy", "random-triplet accuracy"))
# (method, variable, short name) for method_numbers.png.
METHOD_NUMBERS = (("pca", "participation_ratio", "participation ratio"), ("pca", "n_pc90", "PCs for 90%"),
                  ("pca", "subspace_overlap10", "10-PC subspace overlap"), ("pca", "cka_scores10", "CKA of 10 PCs"),
                  ("kpca", "eigenvalue_share_1", "1st eigenvalue share"), ("ica", "reproducible_components",
                                                                           "reproducible of 10"),
                  ("lda_plane", "cv_accuracy", "CV accuracy"), ("lda_grain", "cv_accuracy", "CV accuracy"),
                  ("lda_grain", "discriminant_ratio_1", "1st discriminant share"),
                  ("pls_grain", "q2_2", "Q² (2 components)"), ("pls_grain", "q2_10", "Q² (10 components)"),
                  ("pls_geometry", "q2_10", "Q² (10 components)"),
                  ("pcovr_geometry", "loss_regression@0.5", "regression loss"), ("mds", "stress1_d2", "stress-1 in 2D"),
                  ("mds", "stress1_d10", "stress-1 in 10D"), ("isomap", "residual_variance_d2", "residual variance 2D"),
                  ("isomap", "graph_components", "graph components"), ("tsne", "kl_perplexity30", "KL, perplexity 30"),
                  ("umap", "graph_components_n15", "graph components (15)"), ("pacmap", "triplet_accuracy", "triplets"),
                  ("diffusion", "spectral_gap_after", "eigenvalues before max gap"), ("diffusion", "eigenvalue_2", "2nd eigenvalue"),
                  ("sketchmap", "error", "final error"), ("sketchmap", "dimred_calls", "dimred calls"))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _figure(rows, columns, width, height, **kwargs):
    plt = _plt()
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), dpi=200, facecolor="white", squeeze=False,
                             **kwargs)
    return plt, fig, axes


def _finish(plt, fig, path, title=None, note=None, rect_bottom=0.0):
    """Title (what the figure asks), note (how to read it), save, close.

    Both are wrapped to the figure width; their height is set aside in inches, so a tall figure
    gets no extra gap and a narrow one no cut-off title.
    """
    if TITLE_PREFIX:
        title = f"{TITLE_PREFIX}: {title}" if title else TITLE_PREFIX
    width, height = fig.get_size_inches()
    top = 1.0 - 0.08 / height
    for text, size, color, chars_per_inch, line in ((title, 11, INK, 12, 0.2), (note, 8, MUTED, 16, 0.15)):
        if not text:
            continue
        wrapped = textwrap.fill(text, max(30, int(width * chars_per_inch)))
        fig.text(0.01, top, wrapped, fontsize=size, color=color, ha="left", va="top")
        top -= (line * (wrapped.count("\n") + 1) + 0.06) / height
    fig.tight_layout(rect=(0, rect_bottom, 1, top - 0.04 / height))
    path.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_ROOT is not None and path.is_file():
        keep_old(path, ARCHIVE_ROOT)
    ds._save(fig, path)
    plt.close(fig)
    print(f"  {sc.relative_path(path)}")


def keep_old(path, root):
    """Move an existing figure to root/<its path in the project>, adding .1, .2 ... rather than overwrite."""
    target = Path(root) / sc.relative_path(path)
    k = 0
    while target.exists():
        k += 1
        target = target.with_name(f"{Path(sc.relative_path(path)).stem}.{k}{path.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(path).rename(target)
    print(f"  kept the old {sc.relative_path(path)} as {sc.relative_path(target)}")


def _maps_style(ax, title):
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_color(GRID)


def _grain_legend(plt, fig):
    """Grain-class legend under the axes, in as many rows as the width needs; returns the height share it takes."""
    from select_gbs import GRAIN_COLORS
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=5, label=g)
               for g, c in GRAIN_COLORS.items()]
    width, height = fig.get_size_inches()
    rows, columns = legend_grid(len(handles), width)
    fig.legend(handles=handles, loc="lower center", ncol=columns, fontsize=8, frameon=False, title="Grain class",
               title_fontsize=8)
    return (0.3 + 0.2 * rows) / height


def legend_grid(n, width, inches_per_entry=1.4):
    """(rows, columns) for n legend entries: as few rows as the width allows, filled evenly (8 -> 2 x 4, not 7 + 1)."""
    most = max(2, min(n, int(width / inches_per_entry)))
    rows = int(np.ceil(n / most))
    return rows, int(np.ceil(n / rows))


def _scatter_grain(ax, coordinates, grain_classes, size=2.0):
    from select_gbs import GRAIN_COLORS
    for grain_class, color in GRAIN_COLORS.items():
        members = grain_classes == grain_class
        ax.scatter(coordinates[members, 0], coordinates[members, 1], s=size, color=color, linewidths=0)


def _turned(reference_map, other_map):
    """`other_map` shifted, turned and scaled onto `reference_map` (rows with NaN stay NaN)."""
    keep = np.isfinite(reference_map).all(axis=1) & np.isfinite(other_map).all(axis=1)
    _, first, second = ds.aligned(reference_map[keep], other_map[keep])
    out_a = np.full_like(reference_map, np.nan, dtype=float)
    out_b = np.full_like(other_map, np.nan, dtype=float)
    out_a[keep], out_b[keep] = first, second
    return out_a, out_b


def pick_frequency(pick_lists):
    """Number of GBs picked by exactly k of the given lists, k = 1 .. len(pick_lists)."""
    counts = {}
    for picks in pick_lists:
        for family in set(picks):
            counts[family] = counts.get(family, 0) + 1
    return np.bincount(list(counts.values()), minlength=len(pick_lists) + 1)[1:]


def numeric_value(knob, value):
    return float(value) if knob in NUMERIC_KNOBS else value


# --------------------------------------------------------------------------- descriptor settings

def present_knobs(table):
    return [k for k in NUMERIC_KNOBS if (table["knob"] == k).any()]


def knob_trends(table, path, n_gbs=None):
    knobs = present_knobs(table)
    plt, fig, axes = _figure(len(SCORES), len(knobs), 3.2 * len(knobs) + 0.6, 2.1 * len(SCORES) + 0.9,
                             sharey="row")
    for col, knob in enumerate(knobs):
        part = table[table["knob"] == knob]
        ref_value = ds.REFERENCE[knob]
        x = np.array([float(v) for v in part["value"]] + [ref_value])
        order = np.argsort(x)
        for row, (column, label, same) in enumerate(SCORES):
            ax = axes[row, col]
            y = np.append(part[column].to_numpy(dtype=float), same)
            ax.plot(x[order], y[order], color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
            ax.scatter([ref_value], [same], s=70, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4,
                       label="reference setting" if (row, col) == (0, 0) else None)
            ax.axhline(same, color=MUTED, linewidth=0.8, linestyle="--")
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8.5, color=INK)
            if row == len(SCORES) - 1:
                ax.set_xlabel(UNITS[knob], fontsize=8.5, color=INK)
            if row == 0:
                ax.set_title(knob, fontsize=9.5, color=INK)
    axes[0, 0].legend(loc="lower left", fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "How far each SOAP setting moves the GBs from the reference (r_cut 6, n_max 8, l_max 6, sigma 0.5)",
            f"One setting changed at a time, {_gbs(n_gbs)} at 8x. Dashed line and open circle: no change. "
            "Lower correlation / fewer shared neighbours or picks / larger Procrustes = a bigger change.")


def weighting(table, path):
    part = table[table["knob"] == "weighting"].copy()
    if part.empty:
        return
    part["r_cut"] = part["value"].str.extract(r"rc(\d+)")[0].astype(int)
    part["kind"] = part["value"].str.split().str[0]
    plt, fig, axes = _figure(1, len(SCORES), 3.4 * len(SCORES), 2.8, sharey=True)
    r_cuts = sorted(part["r_cut"].unique())
    y = np.arange(len(r_cuts))[::-1]
    for ax, (column, label, same) in zip(axes[0], SCORES):
        for kind, color, shift in (("pow", BLUE, 0.12), ("poly", ORANGE, -0.12)):
            values = part[part["kind"] == kind].set_index("r_cut").reindex(r_cuts)[column]
            ax.scatter(values, y + shift, s=36, color=color, linewidths=0, zorder=3,
                       label={"pow": "pow (Willatt 2018: r0 3, m 4)", "poly": "poly (Caro 2019: r0 = r_cut, m 2)"}[kind])
        ax.axvline(same, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=9, color=INK)
        ds._style(ax)
    axes[0, 0].set_yticks(y, [f"r_cut {r} Å" for r in r_cuts], fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    _finish(plt, fig, path, "Radial weighting: does down-weighting distant atoms change which GBs look alike?",
            "Each weighting is compared with the unweighted run at the same r_cut. Dashed line: no change.",
            rect_bottom=0.1)


def region_and_processing(table, path):
    part = table[table["knob"].isin(["region", "processing"])]
    if part.empty:
        return
    labels = [f"{k}: {v}" for k, v in zip(part["knob"], part["value"])]
    y = np.arange(len(part))[::-1]
    plt, fig, axes = _figure(1, len(SCORES), 3.3 * len(SCORES), 0.32 * len(part) + 1.6, sharey=True)
    for ax, (column, label, same) in zip(axes[0], SCORES):
        colors = [BLUE if k == "region" else ORANGE for k in part["knob"]]
        ax.scatter(part[column], y, s=36, color=colors, linewidths=0, zorder=3)
        ax.axvline(same, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=9, color=INK)
        ds._style(ax)
    axes[0, 0].set_yticks(y, labels, fontsize=8.5)
    _finish(plt, fig, path, "Which atoms are averaged (region, blue) and how the vector is scaled (processing, orange)",
            "Against the reference vector: atoms within 8 Å of both interfaces, [Co mean | Ge mean], no scaling. "
            "Dashed line: no change.")


def knob_ranking(ranking, path):
    if ranking.empty:
        return
    ranking = ranking.copy()
    ranking["distance_change"] = 1.0 - ranking["min_spearman"]
    ranking["picks_lost"] = ds.N_PICKS - ranking["min_fps_overlap"]
    ranking = ranking.sort_values("mean_rank")
    y = np.arange(len(ranking))[::-1]
    panels = (("distance_change", "1 - lowest distance\nrank correlation"),
              ("min_knn10_kept", "fewest of the 10 nearest\nneighbours kept (lower = more change)"),
              ("picks_lost", f"most FPS picks lost\n(of {ds.N_PICKS})"),
              ("max_procrustes_pca", "largest PCA map change\n(Procrustes)"))
    plt, fig, axes = _figure(1, len(panels), 3.0 * len(panels) + 2.5, 0.5 * len(ranking) + 1.8, sharey=True)
    for ax, (column, title) in zip(axes[0], panels):
        ax.barh(y, ranking[column], height=0.65, color=BLUE, linewidth=0)
        for yy, value in zip(y, ranking[column]):
            ax.text(value, yy, f" {value:.2f}" if column != "picks_lost" else f" {value:.0f}",
                    va="center", fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ds._style(ax)
    axes[0, 0].set_yticks(y, [textwrap.fill(f"{k} ({v})", 38) for k, v in zip(ranking["knob"], ranking["values"])],
                          fontsize=8)
    _finish(plt, fig, path, "Which knob matters most? The largest change each knob makes over its values",
            "Knobs sorted by mean rank over the scores (top = changes most). Values tried are in brackets.")


def pca_dimensions(table, path, n_gbs=None):
    reference = table[table["name"] == "reference"].iloc[0]
    rows = (("n_pc90", "PCs for 90% of the spread"), ("n_pc99", "PCs for 99% of the spread"),
            ("pc1", "share of spread in PC1"), ("pc2", "share of spread in PC2"))
    knobs = present_knobs(table)
    plt, fig, axes = _figure(len(rows), len(knobs), 3.2 * len(knobs) + 0.6, 1.9 * len(rows) + 0.9, sharey="row")
    for col, knob in enumerate(knobs):
        part = table[table["knob"] == knob]
        x = np.array([float(v) for v in part["value"]] + [ds.REFERENCE[knob]])
        order = np.argsort(x)
        for row, (column, label) in enumerate(rows):
            ax = axes[row, col]
            y = np.append(part[column].to_numpy(dtype=float), reference[column])
            ax.plot(x[order], y[order], color=AQUA, linewidth=2, marker="o", markersize=5, zorder=3)
            ax.scatter([ds.REFERENCE[knob]], [reference[column]], s=70, facecolor="white", edgecolor=INK,
                       linewidth=1.2, zorder=4)
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8.5, color=INK)
            if row == len(rows) - 1:
                ax.set_xlabel(UNITS[knob], fontsize=8.5, color=INK)
            if row == 0:
                ax.set_title(knob, fontsize=9.5, color=INK)
    _finish(plt, fig, path, "How many dimensions do the GB vectors really use?",
            f"PCA of {_gbs(n_gbs)} for each setting (open circle: reference). Fewer PCs = the GBs differ "
            "along fewer directions; a 2D map then shows more of the full picture.")


KNOB_COLORS = {"r_cut": BLUE, "n_max": ORANGE, "l_max": AQUA, "sigma": YELLOW, "weighting": "#e87ba4",
               "region": "#008300", "processing": "#4a3aa7"}


def cost(table, path):
    reference = table[table["name"] == "reference"].iloc[0]
    part = table[table["name"] != "reference"].copy()
    floor = 1e-4
    part["change"] = np.maximum(1.0 - part["spearman_vs_reference"].fillna(part["spearman"]), floor)
    # Only the vector length: the stored SOAP times are not comparable (some runs share one timing from an
    # earlier scan, others ran three at a time on one node and waited on the same file reads).
    panels = ((part[part["vector_length"] != reference["vector_length"]], "vector_length",
               "vector length (entries)", "settings that change the vector length"),)
    plt, fig, axes = _figure(1, len(panels), 7, 4.4, sharey=True)
    for ax, (subset, column, xlabel, title) in zip(axes[0], panels):
        for knob, group in subset.groupby("knob", sort=False):
            ax.scatter(group[column], group["change"], s=34, color=KNOB_COLORS.get(knob, GRAY), linewidths=0,
                       label=knob, zorder=3)
            for _, row in group.iterrows():
                ax.annotate(str(row["value"]), (row[column], row["change"]), fontsize=6.5, color=MUTED,
                            xytext=(3, 2), textcoords="offset points")
        ax.axvline(reference[column], color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate("reference", (reference[column], 1), xycoords=("data", "axes fraction"), fontsize=7,
                    color=MUTED, xytext=(3, -10), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        if not subset.empty:
            ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    axes[0, 0].set_ylabel("1 - distance rank correlation with the reference (log)", fontsize=8.5, color=INK)
    _finish(plt, fig, path, "Does a longer vector change the picture more?",
            "Each point is one setting against the reference (r_cut 6, n_max 8, l_max 6, sigma 0.5); dashed line: "
            f"the reference length. Higher = a bigger change; values below {floor:g} are drawn at {floor:g}. "
            "Only n_max, l_max and the single mean over all atoms change the length. SOAP time is not shown: the "
            "stored timings are not "
            "comparable between runs (shared timings, runs that waited on each other's file reads).")


def descriptor_settings(sensitivity_dir, out):
    import pandas as pd
    table = pd.read_csv(sensitivity_dir / "sensitivity.csv", dtype={"value": str})
    ranking = pd.read_csv(sensitivity_dir / "knob_ranking.csv") if (sensitivity_dir / "knob_ranking.csv").is_file() \
        else pd.DataFrame()
    folder = out / "descriptor_settings"
    if present_knobs(table):
        knob_trends(table, folder / "knob_trends.png", pool_count(sensitivity_dir))
        pca_dimensions(table, folder / "pca_dimensions.png", pool_count(sensitivity_dir))
    weighting(table, folder / "weighting.png")
    region_and_processing(table, folder / "region_and_processing.png")
    knob_ranking(ranking, folder / "knob_ranking.png")
    cost(table, folder / "cost.png")
    plt, fig = ds.draw_sensitivity(table)
    _finish(plt, fig, folder / "overview.png",
            "Every setting and every score in one chart: how far does each setting move the GBs from the reference?",
            "One bar per setting, grouped by knob; dashed line = no change (the reference against itself). For "
            "rank correlation, neighbours kept and FPS picks, shorter bars = bigger change; for Procrustes and the "
            "two reconstruction errors (GFRE), longer bars = bigger change. GFRE is not 0 for the reference itself, "
            "so its dashed line is the reference's own value. Weighting at r_cut 8 and 12 is compared with the "
            "unweighted run at the same r_cut.")


# --------------------------------------------------------------------------- maps by setting

def maps_by_setting(sensitivity_dir, out):
    import pandas as pd
    table = pd.read_csv(sensitivity_dir / "sensitivity.csv", dtype={"value": str})
    with np.load(sensitivity_dir / "maps.npz") as data:
        names = [str(n) for n in data["names"]]
        grain = data["grain_class"].astype(str)
        pca_maps = data["pca_maps"]
        embedded = {key: data[key] for key in data.files if "__seed" in key and not key.endswith(("__fps", "__median"))}
    folder = out / "maps_by_setting"
    knob_of = dict(zip(table["name"], table["knob"]))
    for knob in ds.KNOBS:
        members = ["reference"] + [n for n in names if knob_of.get(n) == knob]
        if len(members) < 2:
            continue
        for method in ("pca", "tsne", "umap"):
            maps = []
            for name in members:
                if method == "pca":
                    maps.append(pca_maps[names.index(name)])
                else:
                    key = f"{method}__{name}__seed{ds.SEEDS[0]}"
                    if key not in embedded:
                        break
                    maps.append(embedded[key][:-1])  # last row: bulk
            if len(maps) != len(members):
                continue
            if method != "pca":
                maps = [maps[0]] + [_turned(maps[0], m)[1] for m in maps[1:]]
                maps[0] = _turned(maps[0], maps[0])[0]
            columns = min(len(members), 6)
            rows = int(np.ceil(len(members) / columns))
            plt, fig, axes = _figure(rows, columns, 2.7 * columns, 2.7 * rows + 1.1)
            for ax in axes.ravel()[len(members):]:
                ax.set_visible(False)
            for ax, name, coordinates in zip(axes.ravel(), members, maps):
                _scatter_grain(ax, coordinates, grain)
                _maps_style(ax, name)
            bottom = _grain_legend(plt, fig)
            label = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}[method]
            _finish(plt, fig, folder / f"{method}_{knob}.png",
                    f"{label} map of {_gbs(len(grain))} for every value of {knob}",
                    "Each map turned and scaled onto the reference map (Procrustes), so shapes can be compared. "
                    f"Colour: grain class. {'t-SNE/UMAP with seed 42, as select_gbs.py.' if method != 'pca' else ''}",
                    rect_bottom=bottom)


# --------------------------------------------------------------------------- seed stability

def seed_stability(sensitivity_dir, out):
    import pandas as pd
    folder = out / "seed_stability"
    with np.load(sensitivity_dir / "maps.npz") as data:
        grain = data["grain_class"].astype(str)
        families = data["families"].astype(str)
        stored = {key: data[key] for key in data.files if key.startswith(("tsne__reference__", "umap__reference__"))}
    frequency = {}
    for method in ds.METHODS:
        seeds = [s for s in ds.SEEDS if f"{method}__reference__seed{s}" in stored]
        if len(seeds) < 2:
            continue
        base = stored[f"{method}__reference__seed{seeds[0]}"][:-1]
        columns = min(len(seeds), 5)
        rows = int(np.ceil(len(seeds) / columns))
        plt, fig, axes = _figure(rows, columns, 2.9 * columns, 2.9 * rows + 1.2)
        for ax in axes.ravel()[len(seeds):]:
            ax.set_visible(False)
        position = {f: i for i, f in enumerate(families)}
        for ax, seed in zip(axes.ravel(), seeds):
            coordinates = stored[f"{method}__reference__seed{seed}"][:-1]
            turned = _turned(base, coordinates)[1] if seed != seeds[0] else _turned(base, base)[0]
            _scatter_grain(ax, turned, grain, size=1.5)
            picks = [position[f] for f in stored.get(f"{method}__reference__seed{seed}__fps", []) if f in position]
            ax.scatter(turned[picks, 0], turned[picks, 1], s=22, facecolor="none", edgecolor=INK, linewidth=0.9,
                       zorder=4)
            _maps_style(ax, f"seed {seed}")
        bottom = _grain_legend(plt, fig)
        label = {"tsne": "t-SNE", "umap": "UMAP"}[method]
        _finish(plt, fig, folder / f"{method}_seeds.png",
                f"{label} of the reference vector with {len(seeds)} random seeds; black rings: the 30 FPS picks of that map",
                "Every map turned onto the seed-42 map. Same GBs, same settings: only the seed differs.",
                rect_bottom=bottom)
        for rule in ("fps", "median"):
            lists = [list(stored[f"{method}__reference__seed{s}__{rule}"]) for s in seeds
                     if f"{method}__reference__seed{s}__{rule}" in stored]
            if lists:
                frequency[(method, rule)] = pick_frequency(lists)

    if frequency:
        plt, fig, axes = _figure(1, len(frequency), 3.4 * len(frequency), 3.2, sharey=True)
        for ax, ((method, rule), counts) in zip(axes[0], frequency.items()):
            k = np.arange(1, len(counts) + 1)
            ax.bar(k, counts, width=0.7, color=BLUE if rule == "fps" else AQUA, linewidth=0)
            for kk, c in zip(k, counts):
                if c:
                    ax.text(kk, c, f"{c}", ha="center", va="bottom", fontsize=7, color=INK)
            ax.set_xticks(k)
            ax.set_xlabel(f"picked by k of {len(counts)} seeds", fontsize=8.5, color=INK)
            ax.set_title(f"{'t-SNE' if method == 'tsne' else 'UMAP'}, {'FPS' if rule == 'fps' else 'k-means'} picks",
                         fontsize=9, color=INK)
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.grid(axis="x", visible=False)
        axes[0, 0].set_ylabel("number of GBs", fontsize=8.5, color=INK)
        _finish(plt, fig, folder / "pick_frequency.png",
                "How often is the same GB picked when only the random seed changes?",
                "Bars at the right = GBs every seed agrees on; bars at the left = picks that depend on the seed. "
                "30 picks per map.")

    if (sensitivity_dir / "seed_robustness.csv").is_file():
        seeds_table = pd.read_csv(sensitivity_dir / "seed_robustness.csv")
        drawn = ds.draw_seeds(seeds_table)
        pairs = seeds_table[(seeds_table["method"] == "tsne") & (seeds_table["comparison"] == "seed pair")]
        thin = (f" The t-SNE Procrustes band is too thin to see: seeds move the t-SNE map by at most "
                f"{pairs['procrustes'].max():.3f}." if not pairs.empty and pairs["procrustes"].max() < 0.01 else "")
        if drawn is not None:
            plt, fig = drawn
            _finish(plt, fig, folder / "seed_vs_setting.png", "Is a setting change larger than a new random seed?",
                    "Gray band: the range over all pairs of the 10 seeds at the reference setting (line: median). "
                    "Blue: each setting's map (seed 42) against the reference map (seed 42). A blue point inside "
                    "the band = the setting changes the map or its 30 FPS picks no more than a new seed does; for "
                    "Procrustes, right of the band = a bigger change; for picks shared, left of the band = a bigger "
                    f"change.{thin}", rect_bottom=0.08)


# --------------------------------------------------------------------------- thickness

def thickness(sensitivity_dir, out):
    import pandas as pd
    folder = out / "thickness"
    path = sensitivity_dir / "thickness_pcs.npz"
    if path.is_file():
        with np.load(path) as data:
            pcs = {kind: data[f"{kind}_pcs"] for kind in ("global", "interface")}
            share = {kind: data[f"{kind}_share"] for kind in ("global", "interface")}
            thick, family = data["thickness"], data["family"].astype(str)
        rng = np.random.default_rng(0)
        unique = np.unique(family)
        sample = set(rng.choice(unique, size=min(150, len(unique)), replace=False))
        thicknesses = np.unique(thick)
        plt, fig, axes = _figure(1, 2, 12, 4.6)
        for ax, kind, title in ((axes[0, 0], "global", "Global vector (mean over the whole cell)"),
                                (axes[0, 1], "interface", "Interface vector (atoms within 8 Å of the interfaces)")):
            values = pcs[kind][:, 0]
            for f in sample:
                rows = np.where(family == f)[0]
                rows = rows[np.argsort(thick[rows])]
                ax.plot(thick[rows], values[rows], color=GRAY, linewidth=0.5, alpha=0.5)
            groups = [values[(thick == t) & np.isfinite(values)] for t in thicknesses]
            ax.boxplot(groups, positions=thicknesses, widths=0.45, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "none", "edgecolor": BLUE, "linewidth": 1.4},
                       medianprops={"color": ORANGE, "linewidth": 2}, whiskerprops={"color": BLUE},
                       capprops={"color": BLUE})
            ax.set_xticks(thicknesses, [f"{t}x" for t in thicknesses])
            ax.set_xlabel("cell thickness", fontsize=8.5, color=INK)
            ax.set_ylabel(f"PC1 ({share[kind][0] * 100:.0f}% of the spread)", fontsize=8.5, color=INK)
            ax.set_title(title, fontsize=9.5, color=INK)
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
        _finish(plt, fig, folder / "pc1_by_thickness.png",
                "Does PC1 follow the GB or the thickness of the cell?",
                "All 13,868 structures. Gray lines: one GB at its nine thicknesses (150 GBs shown). Boxes: all GBs at "
                "that thickness. A sloping line = the vector changes with thickness although the GB does not.")
    if (sensitivity_dir / "thickness_test.csv").is_file():
        table = pd.read_csv(sensitivity_dir / "thickness_test.csv")
        table = table[table["subset"] == "all structures"]
        y_labels, values_t, values_g = [], [], []
        for vector in ("global", "interface r8 both"):
            for axis in ("PC1", "PC2", "all features"):
                row = table[(table["vector"] == vector) & (table["axis"] == axis)]
                if row.empty:
                    continue
                y_labels.append(f"{'global' if vector == 'global' else 'interface'}: {axis}")
                values_t.append(float(row["thickness_share"].iloc[0]))
                values_g.append(float(row["gb_share"].iloc[0]))
        y = np.arange(len(y_labels))[::-1]
        plt, fig, axes = _figure(1, 1, 8.5, 0.45 * len(y_labels) + 1.8)
        ax = axes[0, 0]
        ax.barh(y + 0.18, values_t, height=0.34, color=ORANGE, linewidth=0, label="explained by thickness")
        ax.barh(y - 0.18, values_g, height=0.34, color=BLUE, linewidth=0, label="explained by the GB (family)")
        for yy, a, b in zip(y, values_t, values_g):
            ax.text(a, yy + 0.18, f" {a:.3g}", va="center", fontsize=7.5, color=INK)
            ax.text(b, yy - 0.18, f" {b:.3g}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, y_labels, fontsize=8.5)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("share of the variance", fontsize=8.5, color=INK)
        ds._style(ax)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False)
        _finish(plt, fig, folder / "variance_shares.png",
                "How much of the spread is thickness, and how much is the GB itself?",
                "Between-group share of the variance (all 13,868 structures). Ideal: GB share near 1, thickness share near 0.")


# --------------------------------------------------------------------------- selection

def learning_curves(summary, majority, n_classes, label, path):
    reference = summary[(summary["descriptor"] == "reference") & (summary["label"] == label)]
    if reference.empty:
        return
    columns = (("krr", "full"), ("1nn", "full"), ("krr", "maps"), ("1nn", "maps"))
    rows = (("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy"))
    plt, fig, axes = _figure(len(rows), len(columns), 4.2 * len(columns), 3.3 * len(rows) + 2.6, sharex=True,
                             sharey="row")
    legends = {}
    for col, (model, family) in enumerate(columns):
        strategies = sb.FULL_SPACE if family == "full" else ("random",) + sb.MAP_RULES
        for row, (metric, metric_label) in enumerate(rows):
            ax = axes[row, col]
            for strategy in strategies:
                part = reference[(reference["strategy"] == strategy) & (reference["model"] == model)].sort_values("n")
                if part.empty:
                    continue
                if strategy in STRATEGY_COLORS:
                    color, style = STRATEGY_COLORS[strategy], "-"
                else:
                    color = MAP_COLORS[strategy.split("_")[-1]]
                    style = "-" if "_fps_" in strategy else "--"
                mean, std = part[f"{metric}_mean"].to_numpy(), part[f"{metric}_std"].fillna(0).to_numpy()
                line, = ax.plot(part["n"], mean, color=color, linestyle=style, linewidth=1.8)
                ax.fill_between(part["n"], mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
                legends.setdefault(family, {})[STRATEGY_NAMES.get(strategy, strategy)] = line
            baseline = majority[label] if metric == "accuracy" else 1.0 / n_classes[label]
            ax.axhline(baseline, color=MUTED, linewidth=0.9, linestyle=":" if metric == "accuracy" else "-.")
            ax.set_xscale("log")
            ax.set_xticks(sb.SIZES, [str(s) for s in sb.SIZES])
            ax.minorticks_off()
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            if row == 0:
                ax.set_title(f"{'KRR (fixed Stage 1 settings)' if model == 'krr' else '1-nearest neighbour'}, "
                             f"{'full-space strategies' if family == 'full' else 'rules on 2D maps'}",
                             fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8.5, color=INK)
            if row == len(rows) - 1:
                ax.set_xlabel("GBs labelled (n)", fontsize=8.5, color=INK)
    baselines = {"accuracy: always guessing the largest class": plt.Line2D([], [], color=MUTED, linestyle=":"),
                 "balanced accuracy: chance (1 / number of classes)": plt.Line2D([], [], color=MUTED, linestyle="-.")}
    for family, x in (("full", 0.25), ("maps", 0.75)):
        entries = {**legends.get(family, {}), **baselines}
        fig.legend(list(entries.values()), list(entries.keys()), loc="lower center", bbox_to_anchor=(x, 0.0),
                   ncol=2, fontsize=7.5, frameon=False)
    n_splits = int(reference["n_splits"].max())
    _finish(plt, fig, path, f"Learning {LABEL_NAMES[label]} from n labelled GBs: which way of choosing them works best?",
            f"Mean over {n_splits} held-out test sets of {sb.N_TEST} GBs (band: one standard deviation). Left pair: "
            "strategies in the full descriptor space; right pair: the six map rules of select_gbs.py (solid FPS, "
            "dashed k-means). Balanced accuracy weights every class equally, so always guessing the largest class "
            "scores chance.", rect_bottom=0.8 / (3.3 * len(rows) + 2.6))


def existing_lists(existing, n_groups, path, n_gbs=None):
    existing = existing[existing["n"] == 30]
    if existing.empty:
        return
    panels = (("coverage_radius", "coverage radius / median distance\n(lower = no GB far from a pick)"),
              ("mean_nearest_pick", "mean distance to nearest pick\n(lower = picks closer to every GB)"),
              ("groups_covered", f"plane x grain-class groups covered\n(of {n_groups})"),
              ("grain_class_1nn_balanced_accuracy", "grain class, 1-NN\nbalanced accuracy"),
              ("plane_1nn_accuracy", "plane, 1-NN\naccuracy"))
    grouped = existing.groupby("list", sort=False)
    names = list(grouped.groups)
    y = np.arange(len(names))[::-1]
    plt, fig, axes = _figure(1, len(panels), 3.0 * len(panels), 0.32 * len(names) + 1.9, sharey=True)
    for ax, (column, title) in zip(axes[0], panels):
        if column not in existing:
            ax.set_visible(False)
            continue
        stats = grouped[column].agg(["mean", "min", "max"]).reindex(names)
        colors = [ORANGE if n.startswith("existing") else BLUE for n in names]
        ax.hlines(y, stats["min"], stats["max"], color=GRID, linewidth=3)
        ax.scatter(stats["mean"], y, s=30, color=colors, linewidths=0, zorder=3)
        n_existing = sum(n.startswith("existing") for n in names)
        if 0 < n_existing < len(names):
            ax.axhline(y[n_existing - 1] - 0.5, color=MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=8.5, color=INK)
        ds._style(ax)
    pretty = [n.replace("existing ", "existing: ").replace("strategy ", "") for n in names]
    pretty = [STRATEGY_NAMES.get(p, p) for p in pretty]
    axes[0, 0].set_yticks(y, pretty, fontsize=8)
    _finish(plt, fig, path, f"The six existing lists of 30 (orange) against every strategy picking 30 from {_gbs(n_gbs)} (blue)",
            "Accuracy is on all GBs not picked. Gray bar: range over the random draws (random, stratified). "
            "The union of the six lists (162 GBs) is left out: it is not comparable on size.")


def picks_on_maps(picks, maps_path, names, path, title):
    if picks.empty or not maps_path.is_file():
        return
    with np.load(maps_path) as data:
        families = data["families"].astype(str)
        maps = {m: data[m] for m in ("pca", "tsne", "umap") if m in data.files}
    position = {f: i for i, f in enumerate(families)}
    rows = [n for n in names if (picks["list"] == f"strategy {n}").any()]
    if not rows:
        return
    plt, fig, axes = _figure(len(rows), len(maps), 3.0 * len(maps), 2.6 * len(rows) + 0.6)
    for r, strategy in enumerate(rows):
        chosen = picks[(picks["list"] == f"strategy {strategy}") & (picks["draw"] == 0)]
        idx = [position[f] for f in chosen["family"] if f in position]
        for c, (method, coordinates) in enumerate(maps.items()):
            ax = axes[r, c]
            points, bulk = coordinates[:-1], coordinates[-1]
            ax.scatter(points[:, 0], points[:, 1], s=1.5, color="#d0d0d0", linewidths=0)
            ax.scatter(points[idx, 0], points[idx, 1], s=16, color=STRATEGY_COLORS.get(strategy, MAP_COLORS.get(
                strategy.split("_")[-1], BLUE)), edgecolor="white", linewidth=0.4, zorder=3)
            ax.scatter([bulk[0]], [bulk[1]], marker="*", s=70, color=INK, zorder=4)
            _maps_style(ax, f"{STRATEGY_NAMES.get(strategy, strategy)} | {method.upper() if method == 'pca' else {'tsne': 't-SNE', 'umap': 'UMAP'}[method]}")
    _finish(plt, fig, path, title, f"Gray: {_gbs(len(families))}; coloured: the 30 picks; star: pristine bulk. "
            "Maps as in select_gbs.py (seed 42).")


def selection(benchmark_dir, out, krr_dir=None):
    import pandas as pd
    folder = out / "selection"
    summary = pd.read_csv(benchmark_dir / "summary.csv")
    meta = json.loads((benchmark_dir / "summary.json").read_text())
    majority = meta["majority_class_share"]
    with np.load(benchmark_dir / "maps.npz") as data:
        labels = {label: data[label].astype(str) for label in sb.LABELS if label in data.files}
        n_gbs = len(data["families"])
    n_classes = {label: len(np.unique(values)) for label, values in labels.items()}
    n_groups = len(set(zip(labels["plane"], labels["grain_class"])))
    for label in n_classes:
        learning_curves(summary, majority, n_classes, label, folder / f"learning_curves_{label}.png")
    coverage = pd.read_csv(benchmark_dir / "coverage.csv")
    existing = pd.read_csv(benchmark_dir / "existing_lists.csv")
    plt, fig = sb.draw_coverage(coverage, existing)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_text(STRATEGY_NAMES.get(text.get_text(), text.get_text()))
    _finish(plt, fig, folder / "coverage.png",
            f"How evenly do the picks cover {_gbs(n_gbs)}, without using any labels?",
            "Top row: full-space strategies; bottom row: the six map rules, with the existing lists of 30 as markers "
            "(diamond FPS, square k-means). Left: the largest distance from any GB to its nearest pick; middle: the "
            "mean of that distance (both in units of the median GB-to-GB distance; lower = better covered). Right: "
            f"plane x grain-class groups that hold at least one pick (of {n_groups}; higher = better). 'Stratified' "
            "uses these labels to choose, so its right panel is full by design. Picks come from the "
            f"{n_gbs - sb.N_TEST:,} pool GBs of each test split; mean over splits.")
    existing_lists(existing, n_groups, folder / "existing_lists.png", n_gbs)
    if (benchmark_dir / "picks.csv").is_file():
        picks = pd.read_csv(benchmark_dir / "picks.csv")
        picks_on_maps(picks, benchmark_dir / "maps.npz", sb.FULL_SPACE, folder / "picks_full_space.png",
                      "Where the full-space strategies pick 30 GBs, shown on the three maps")
        picks_on_maps(picks, benchmark_dir / "maps.npz", sb.MAP_RULES, folder / "picks_map_rules.png",
                      "Where the six map rules of select_gbs.py pick 30 GBs, shown on all three maps")
    drawn = sb.draw_descriptors(summary)
    if drawn is not None:
        plt, fig = drawn
        for ax in fig.axes:
            ax.axvline(majority["grain_class"], color=INK, linewidth=0.8, linestyle=":")
            ax.annotate("always the largest class", (majority["grain_class"], 1), xycoords=("data", "axes fraction"),
                        fontsize=7, color=MUTED, xytext=(3, -10), textcoords="offset points")
        part = summary[(summary["label"] == "grain_class") & summary["strategy"].isin(["fps", "random"])]
        spread = part.groupby("strategy")["accuracy_std"].median()
        _finish(plt, fig, folder / "descriptor_efficiency.png", "Does any SOAP setting make grain class easier to learn?",
                "Each point: grain-class test accuracy averaged over n = 5-100 labelled GBs, for one descriptor setting "
                "(dashed: the reference; dotted: always answering the largest class, "
                f"{majority['grain_class']:.3f}). Between test splits, the accuracy at one n varies by about "
                f"{spread.get('fps', np.nan):.2f} for FPS and {spread.get('random', np.nan):.2f} for random "
                "(median standard deviation), so differences of a few hundredths are within that spread.")
    if "label" in summary and summary["label"].astype(str).str.startswith("grain_class@").any():
        grain_class_per_plane(summary, meta, krr_dir, folder / "grain_class_per_plane.png")
    geometry = [label for label in gl.TARGETS if (summary["label"] == label).any()]
    if geometry and "r2_mean" in summary:
        free_label_curves(summary, geometry, krr_dir, folder / "learning_curves_free_labels.png")


# --------------------------------------------------------------------------- covariant

def covariant(covariant_dir, out):
    import pandas as pd
    path = covariant_dir / "covariant_soap.npz"
    if not path.is_file():
        print(f"  skipped covariant: {sc.relative_path(path)} not found")
        return
    index = pd.read_csv(covariant_dir / "index.csv", dtype={"plane": str})
    rows = np.where(index["thickness"].to_numpy() == 8)[0]
    key = sc.radius_key(8.0)
    sizes = {}
    with np.load(path) as data:
        n = data[f"n_{key}"][rows]
        for name in ("polar", "axial"):
            X = data[f"X_{name}_{key}"][rows].astype(np.float64)  # (N, plane, 3, 2P)
            atom = index[f"atom_norm_{name}"].to_numpy()[rows]
            both = np.stack([sc.combine_planes(X[:, :, c], n) for c in range(3)], axis=1)
            sizes[name] = {"edge": np.linalg.norm(X[:, 0].reshape(len(rows), -1), axis=1) / atom,
                           "mid": np.linalg.norm(X[:, 1].reshape(len(rows), -1), axis=1) / atom,
                           "both": np.linalg.norm(both.reshape(len(rows), -1), axis=1) / atom}
    planes = index["plane"].to_numpy()[rows]
    floor = 1e-6
    plt, fig, axes = _figure(1, 4, 16, 4.2)
    plane_colors = {"100": BLUE, "110": ORANGE, "111": AQUA}
    for col, name in enumerate(("polar", "axial")):
        ax = axes[0, 2 * col]
        for plane, color in plane_colors.items():
            m = planes == plane
            ax.scatter(np.maximum(sizes[name]["edge"][m], floor), np.maximum(sizes[name]["mid"][m], floor), s=6,
                       color=color, linewidths=0, label=plane, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1e-3, color=MUTED, linewidth=0.8, linestyle="--")
        ax.axhline(1e-3, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("edge interface: |vector| / one atom's |vector|", fontsize=8, color=INK)
        ax.set_ylabel("mid interface", fontsize=8, color=INK)
        ax.set_title(f"{name}: the two interfaces of each cell", fontsize=9, color=INK)
        ds._style(ax)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, title="plane", title_fontsize=7.5)
        ax = axes[0, 2 * col + 1]
        each = np.maximum(sizes[name]["edge"], sizes[name]["mid"])
        for plane, color in plane_colors.items():
            m = planes == plane
            ax.scatter(np.maximum(each[m], floor), np.maximum(sizes[name]["both"][m], floor), s=6, color=color,
                       linewidths=0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lims = [floor, max(each.max(), 1e-2) * 1.5]
        ax.plot(lims, lims, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("larger of the two interfaces", fontsize=8, color=INK)
        ax.set_ylabel("mean of both interfaces", fontsize=8, color=INK)
        ax.set_title(f"{name}: averaging both interfaces", fontsize=9, color=INK)
        ds._style(ax)
    zero = {name: int((np.maximum(sizes[name]["edge"], sizes[name]["mid"]) < 1e-3).sum()) for name in sizes}
    _finish(plt, fig, out / "covariant" / "vector_sizes.png",
            f"Covariant SOAP: how large are the polar and axial (spin-like) vectors at each interface? "
            f"(8x, {len(rows):,} GB families)",
            f"Dashed lines at 1e-3: below that a vector is zero by symmetry (polar {zero['polar']}, axial {zero['axial']} "
            "GBs). Right panels: points below the diagonal lose their vector when both interfaces are averaged.")


# --------------------------------------------------------------------------- shared by the 2026-10-05 sections

def _dot_rows(ax, table, names, value, colors, label_of=None, baseline=None, error=None):
    """One row per name (top to bottom), one marker per colour key (table columns: name, key, value)."""
    y = np.arange(len(names))[::-1]
    position = dict(zip(names, y))
    keys = [k for k in colors if (table["key"] == k).any()]
    offsets = np.linspace(-0.25, 0.25, len(keys)) if len(keys) > 1 else [0.0]
    for offset, key in zip(offsets, keys):
        part = table[(table["key"] == key) & table["name"].isin(names)]
        rows = np.array([position[n] for n in part["name"]]) + offset
        ax.scatter(part[value], rows, s=22, color=colors[key], linewidths=0, zorder=3, label=key)
        if error is not None and error in part:
            ax.errorbar(part[value], rows, xerr=part[error].fillna(0), fmt="none", ecolor=colors[key], elinewidth=0.8,
                        zorder=2)
    if baseline is not None and np.isfinite(baseline):
        ax.axvline(baseline, color=MUTED, linewidth=0.9, linestyle=":")
    ax.set_yticks(y, [label_of(n) if label_of else n for n in names], fontsize=7.5)
    ds._style(ax)


def _heatmap(ax, matrix, rows, columns, significant=None, norm=None, cmap="viridis", fmt="{:.2f}"):
    """matrix (rows x columns) with its numbers; bold where significant, gray text where not."""
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(columns)), columns, fontsize=7.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=7.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    for i in range(len(rows)):
        for j in range(len(columns)):
            if not np.isfinite(matrix[i, j]):
                continue
            strong = significant is None or bool(significant[i, j])
            rgba = image.cmap(image.norm(matrix[i, j]))
            light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=6.5,
                    color=(INK if light else "white") if strong else ("#808080" if light else "#c8c8c8"),
                    fontweight="bold" if strong and significant is not None else "normal")
    return image


def _stacked(ax, shares, names, factors, colors):
    """Horizontal stacked bars: shares[name][factor] for every name (top to bottom)."""
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for factor in factors:
        width = np.array([shares.get(n, {}).get(factor, np.nan) for n in names], dtype=float)
        width = np.nan_to_num(np.clip(width, 0, None))
        ax.barh(y, width, left=left, color=colors[factor], height=0.7, label=factor.replace("_", " "))
        left += width
    for row, name in zip(y, names):
        if not np.isfinite([shares.get(name, {}).get(f, np.nan) for f in factors]).any():
            ax.text(0.02, row, "n/a", va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_yticks(y, names, fontsize=7.5)
    ds._style(ax)


def pool_count(sensitivity_dir):
    """Distinct GBs a descriptor_sensitivity.py run used, from its summary.json ("1,112 distinct GBs ...")."""
    path = Path(sensitivity_dir) / "summary.json"
    if not path.is_file():
        return None
    head = str(json.loads(path.read_text()).get("pool", "")).split(" ")[0].replace(",", "")
    return int(head) if head.isdigit() else None


def _gbs(count):
    return f"{count:,} distinct GBs" if count else "the distinct GBs"


# --------------------------------------------------------------------------- selection (v2 additions)

def grain_class_per_plane(summary, meta, krr_dir, path):
    """Grain class learning curves scored on each plane alone, with chance, majority and the full KRR."""
    import pandas as pd
    reference = summary[summary["descriptor"] == "reference"]
    planes = [p for p in sc.PLANES if (reference["label"] == f"grain_class@{p}").any()]
    if not planes:
        return
    krr = pd.DataFrame()
    if krr_dir is not None and (Path(krr_dir) / "krr_by_plane.csv").is_file():
        krr = pd.read_csv(Path(krr_dir) / "krr_by_plane.csv", dtype={"plane": str})
        krr = krr[(krr["label"] == "grain_class") & (krr["model"] == "krr")]
        if "grouping" in krr and (krr["grouping"] == "distinct").any():
            krr = krr[krr["grouping"] == "distinct"]
    chance = {"100": 0.25, "110": 0.25, "111": 0.125}
    plane_majority = meta.get("grain_class_majority_per_plane", {})
    plt, fig, axes = _figure(1, len(planes), 4.4 * len(planes), 5.4, sharey=True)
    handles = {}
    for ax, plane in zip(axes[0], planes):
        part = reference[reference["label"] == f"grain_class@{plane}"]
        for strategy in sb.FULL_SPACE:
            for model, style in (("krr", "-"), ("1nn", "--")):
                line = part[(part["strategy"] == strategy) & (part["model"] == model)].sort_values("n")
                if line.empty:
                    continue
                handle, = ax.plot(line["n"], line["accuracy_mean"], color=STRATEGY_COLORS.get(strategy, GRAY),
                                  linestyle=style, linewidth=1.6)
                if model == "krr":
                    handles[STRATEGY_NAMES.get(strategy, strategy)] = handle
        ax.axhline(chance[plane], color=MUTED, linewidth=0.9, linestyle="-.")
        if plane in plane_majority:
            ax.axhline(plane_majority[plane], color=MUTED, linewidth=0.9, linestyle=":")
        for trained, style in (("all planes", "-"), (f"{plane} only", "--")):
            row = krr[(krr["trained_on"] == trained) & (krr["plane"] == plane)] if not krr.empty else krr
            if not row.empty:
                ax.axhline(float(row["accuracy"].iloc[0]), color=INK, linewidth=1.2, linestyle=style)
        ax.set_xscale("log")
        ax.set_xticks(sb.SIZES, [str(s) for s in sb.SIZES])
        ax.minorticks_off()
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_title(f"{plane} GBs ({int(1 / chance[plane])} possible classes)", fontsize=9, color=INK)
        ax.set_xlabel("GBs labelled (n), all planes together", fontsize=8.5, color=INK)
    axes[0, 0].set_ylabel("grain-class accuracy on this plane's test GBs", fontsize=8.5, color=INK)
    extra = {"KRR (solid) / 1-NN (dashed)": plt.Line2D([], [], color=GRAY),
             "chance (1 / possible classes)": plt.Line2D([], [], color=MUTED, linestyle="-."),
             "always this plane's largest class": plt.Line2D([], [], color=MUTED, linestyle=":")}
    if not krr.empty:
        extra["Stage 1.4 KRR, all GBs, one model for all planes"] = plt.Line2D([], [], color=INK)
        extra["Stage 1.4 KRR, all GBs, one model per plane"] = plt.Line2D([], [], color=INK, linestyle="--")
    entries = {**handles, **extra}
    fig.legend(list(entries.values()), list(entries.keys()), loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "Is grain class learnable on each plane, against that plane's own chance level?",
            "The same test predictions as learning_curves_grain_class.png, scored on the 100, 110 and 111 GBs "
            "separately. A 100 or 110 GB can be one of 4 classes, a 111 GB one of 8. Black lines: the Stage 1.4 "
            "KRR with 5-fold cross-validation on all GBs (krr_by_plane.csv), the most labels one can have.",
            rect_bottom=1.0 / 5.4)


def free_label_curves(summary, labels, krr_dir, path):
    """R^2 of the geometry labels against the number of labelled GBs."""
    import pandas as pd
    reference = summary[summary["descriptor"] == "reference"]
    full = pd.DataFrame()
    if krr_dir is not None and (Path(krr_dir) / "krr_regression.csv").is_file():
        full = pd.read_csv(Path(krr_dir) / "krr_regression.csv")
        full = full[(full["dataset"] == "all") & (full["scored_on"] == "all")]
        if "grouping" in full and (full["grouping"] == "distinct").any():
            full = full[full["grouping"] == "distinct"]
    models = (("krr", "KRR (settings by leave-one-out on the picks)"), ("1nn", "1-nearest neighbour"))
    plt, fig, axes = _figure(len(models), len(labels), 3.6 * len(labels), 3.0 * len(models) + 1.6, sharex=True)
    handles = {}
    for col, label in enumerate(labels):
        for row, (model, model_name) in enumerate(models):
            ax = axes[row, col]
            for strategy in sb.FULL_SPACE:
                line = reference[(reference["label"] == label) & (reference["strategy"] == strategy)
                                 & (reference["model"] == model)].sort_values("n")
                if line.empty:
                    continue
                mean, std = line["r2_mean"].to_numpy(float), line["r2_std"].fillna(0).to_numpy(float)
                handle, = ax.plot(line["n"], mean, color=STRATEGY_COLORS.get(strategy, GRAY), linewidth=1.6)
                ax.fill_between(line["n"], mean - std, mean + std, color=STRATEGY_COLORS.get(strategy, GRAY),
                                alpha=0.12, linewidth=0)
                handles[STRATEGY_NAMES.get(strategy, strategy)] = handle
            ax.axhline(0.0, color=MUTED, linewidth=0.9, linestyle=":")
            if not full.empty and model == "krr" and (full["target"] == label).any():
                ax.axhline(float(full.loc[full["target"] == label, "krr_r2"].iloc[0]), color=INK, linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xticks(sb.SIZES, [str(s) for s in sb.SIZES])
            ax.minorticks_off()
            ax.set_ylim(max(-1.0, ax.get_ylim()[0]), 1.0)
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            if row == 0:
                ax.set_title(GEOMETRY_NAMES.get(label, label), fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"{model_name}\nR² on the test GBs", fontsize=8, color=INK)
            if row == len(models) - 1:
                ax.set_xlabel("GBs labelled (n)", fontsize=8.5, color=INK)
    entries = {**handles, "R² = 0: no better than the mean of the picks": plt.Line2D([], [], color=MUTED, linestyle=":")}
    if not full.empty:
        entries["Stage 1.4 KRR, all GBs, 5-fold"] = plt.Line2D([], [], color=INK)
    fig.legend(list(entries.values()), list(entries.keys()), loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "Can the shift-dependent interface geometry be learned from a few labelled GBs?",
            "Labels from geometry_labels.py (whole cell, both interfaces), computed from the unrelaxed atoms, no DFT. "
            "Unlike grain class, they differ between GBs of one class. R² below 0 means worse than always answering "
            "the picks' mean (axis cut at -1). Mean over the test splits; band: one standard deviation.",
            rect_bottom=0.9 / (3.0 * len(models) + 1.6))


# --------------------------------------------------------------------------- free labels

def free_labels(labels_file, krr_dir, out):
    """What the geometry labels look like, and how well the Stage 1.4 KRR predicts them."""
    import pandas as pd
    folder = out / "free_labels"
    table = pd.read_csv(labels_file, dtype={"plane": str})
    meta_path = Path(labels_file).with_name(gl.JSON_NAME)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    grain = pd.read_csv(mc.LABELS_DIR / "gb_labels.csv", dtype={"plane": str})[["family", "grain_class"]]
    distinct = set(mc.distinct_families()) - set(mc.NO_BOUNDARY)
    at8 = table[(table["thickness"] == 8) & table["family"].isin(distinct)].merge(grain, on="family", how="left")
    if at8.empty:
        print("  skipped free_labels: no 8x rows of distinct GBs")
        return
    n_gbs = at8["family"].nunique()
    cell = at8[at8["region"] == "cell"]
    shares = meta.get("variance_explained_by_plane_and_grain_class", {}).get("cell", {})

    # 1. by plane and grain class
    from select_gbs import GRAIN_COLORS
    groups = [(p, g) for p in sc.PLANES for g in GRAIN_COLORS if ((cell["plane"] == p) & (cell["grain_class"] == g)).any()]
    plt, fig, axes = _figure(len(gl.TARGETS), 1, max(8.0, 0.55 * len(groups) + 2), 2.2 * len(gl.TARGETS) + 1.2,
                             sharex=True)
    rng = np.random.default_rng(0)
    for ax, label in zip(axes[:, 0], gl.TARGETS):
        for x, (plane, grain_class) in enumerate(groups):
            values = cell.loc[(cell["plane"] == plane) & (cell["grain_class"] == grain_class), label].to_numpy(float)
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(values)), values, s=4, color=GRAIN_COLORS[grain_class],
                       linewidths=0, alpha=0.7)
            ax.hlines(np.median(values), x - 0.35, x + 0.35, color=INK, linewidth=1.2)
        share = shares.get(label, {}).get("plane_and_grain_class")
        ax.set_ylabel(GEOMETRY_NAMES.get(label, label) + (f"\nplane + class explain {share:.0%}" if share is not None
                                                         else ""), fontsize=8, color=INK)
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.grid(axis="x", visible=False)
    axes[-1, 0].set_xticks(range(len(groups)), [f"{p} {g}" for p, g in groups], fontsize=7, rotation=35,
                           ha="right", rotation_mode="anchor")
    _finish(plt, fig, folder / "by_grain_class.png",
            f"Do GBs of one grain class share their interface geometry? ({n_gbs:,} distinct GBs at 8x)",
            "Each dot is one GB (both interfaces, unrelaxed atoms); black bar: median. Spread within a column is "
            "what the shift between the grains adds beyond the grain class. The share of each label's variance "
            "that plane and grain class explain is given at the left; far below 100% means the label carries "
            "information the grain class does not.")

    # 2. edge against mid
    edge = at8[at8["region"] == "edge"].set_index("family")
    mid = at8[at8["region"] == "mid"].set_index("family")
    both = edge.join(mid, lsuffix="_edge", rsuffix="_mid", how="inner")
    plt, fig, axes = _figure(1, len(gl.TARGETS), 3.4 * len(gl.TARGETS), 3.9)
    for ax, label in zip(axes[0], gl.TARGETS):
        for plane, color in PLANE_COLORS.items():
            part = both[both["plane_edge"] == plane]
            ax.scatter(part[f"{label}_edge"], part[f"{label}_mid"], s=5, color=color, linewidths=0, alpha=0.7,
                       label=plane)
        low = np.nanmin([both[f"{label}_edge"].min(), both[f"{label}_mid"].min()])
        high = np.nanmax([both[f"{label}_edge"].max(), both[f"{label}_mid"].max()])
        ax.plot([low, high], [low, high], color=MUTED, linewidth=0.8, linestyle="--")
        same = float(np.mean(np.isclose(both[f"{label}_edge"], both[f"{label}_mid"], atol=1e-6)))
        ax.set_title(f"{GEOMETRY_NAMES.get(label, label)}\nequal at both: {same:.0%} of GBs", fontsize=8.5, color=INK)
        ax.set_xlabel("edge interface (x = 0)", fontsize=8, color=INK)
        ax.set_ylabel("mid interface (x = Lx/2)", fontsize=8, color=INK)
        ds._style(ax)
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False, title="plane", title_fontsize=7)
    _finish(plt, fig, folder / "edge_vs_mid.png",
            "Do the two interfaces of one cell have the same geometry?",
            f"{len(both):,} distinct GBs at 8x. On the dashed line both interfaces are alike; the fifth study keeps "
            "edge and mid as separate rows because they need not be.")

    # 3. Stage 1.4 KRR on the labels
    path = Path(krr_dir) / "krr_regression.csv" if krr_dir is not None else None
    if path is None or not path.is_file():
        print(f"  skipped free_labels/krr_r2.png: {path} not found")
        return
    krr = pd.read_csv(path)
    krr = krr[krr["dataset"] == "all"]
    if "grouping" in krr and (krr["grouping"] == "distinct").any():
        krr = krr[krr["grouping"] == "distinct"]
    scopes = [s for s in ("all", *sc.PLANES) if (krr["scored_on"].astype(str) == s).any()]
    plt, fig, axes = _figure(1, len(gl.TARGETS), 3.4 * len(gl.TARGETS), 3.6, sharey=True)
    width = 0.38
    for ax, label in zip(axes[0], gl.TARGETS):
        part = krr[krr["target"] == label]
        part = part.set_index(part["scored_on"].astype(str))
        x = np.arange(len(scopes))
        for offset, (model, color) in zip((-width / 2, width / 2), (("krr", BLUE), ("knn", ORANGE))):
            values = [float(part.loc[s, f"{model}_r2"]) if s in part.index else np.nan for s in scopes]
            ax.bar(x + offset, values, width=width, color=color, label="KRR" if model == "krr" else "kNN")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, ["all GBs" if s == "all" else f"{s} GBs" for s in scopes], fontsize=7.5)
        ax.set_title(GEOMETRY_NAMES.get(label, label), fontsize=8.5, color=INK)
        bottom = min(-0.2, ax.get_ylim()[0])
        ax.set_ylim(bottom, 1.22)  # R² <= 1; the band above holds the legend
        ax.set_yticks([tick for tick in ax.get_yticks() if bottom <= tick <= 1.0 + 1e-9])
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("R², 5-fold, coinciding vectors in one fold", fontsize=8, color=INK)
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=7.5, frameon=False)
    _finish(plt, fig, folder / "krr_r2.png",
            "Stage 1.4 with free labels: how well does the GB vector predict the interface geometry?",
            "krr_classify.py --regression (one model on all planes, scored on all GBs and on each plane). R² = 0: no "
            "better than the training mean; 1: perfect.")


# --------------------------------------------------------------------------- representation response

def representation_tables(representation_dir):
    """results.csv of phase 1 and 2 together, their maps and phase 1's summary (missing parts are left out)."""
    import pandas as pd
    tables, maps, summaries = [], {}, {}
    for phase in ("phase1", "phase2"):
        folder = Path(representation_dir) / phase
        if not (folder / "results.csv").is_file():
            continue
        tables.append(pd.read_csv(folder / "results.csv", dtype={"subset": str, "representation": str}))
        with np.load(folder / "maps.npz", allow_pickle=False) as data:
            maps.update({key: data[key] for key in data.files if key not in maps})
        summaries[phase] = json.loads((folder / "summary.json").read_text())
    if not tables:
        return None, {}, {}
    table = pd.concat(tables, ignore_index=True)
    # Both phases store the raw-descriptor rows and the baselines; keep them once.
    shared = table["kind"].isin(("raw", "raw_pair", "baseline"))
    keys = ["kind", "method", "subset", "representation", "variable"]
    repeated = shared & table[keys].astype(str).duplicated()
    return table[~repeated].reset_index(drop=True), maps, summaries


def method_order(table):
    present = set(table["method"].dropna())
    order = [m for m in METHOD_NAMES if m in present and m != "raw"]
    return order


def bulk_figure(table, path):
    bulk = table[table["kind"] == "bulk"]
    averages = bulk[bulk["variable"] == "cell_average_over_atom_norm"]
    if averages.empty:
        return
    plt, fig, axes = _figure(1, 3, 15, 4.4)
    ax = axes[0, 0]
    blocks = [b for b in ("polar", "axial", "tensor", "pseudotensor") if (averages["method"] == b).any()]
    kinds = list(dict.fromkeys(averages["representation"]))
    colors = dict(zip(kinds, [INK, BLUE, ORANGE, AQUA, YELLOW]))
    for k, kind in enumerate(kinds):
        part = averages[(averages["representation"] == kind) & (averages["subset"] == "all")]
        x = [blocks.index(b) + (k - (len(kinds) - 1) / 2) * 0.15 for b in part["method"]]
        ax.scatter(x, np.maximum(part["value"], 1e-16), marker="v", s=30, color=colors[kind], label=f"{kind} (largest)")
        ax.scatter(x, np.maximum(part["main"], 1e-16), marker="o", s=14, color=colors[kind], alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e2)  # room above the 1e-3 line for the legend
    ax.axhline(1e-3, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(blocks)), [f"{b}\n(lambda {1 if b in ('polar', 'axial') else 2})" for b in blocks], fontsize=7.5)
    ax.set_ylabel("|cell average| / mean |one atom's vector|", fontsize=8, color=INK)
    ax.set_title("bulk cells: do the lambda >= 1 vectors average to zero?", fontsize=8.5, color=INK)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ds._style(ax)
    for ax, test, variables, title in (
            (axes[0, 1], "D_vs_C", ("relative_residual", "cosine_with_interfaces"),
             "D against C: is the whole cell the atom-weighted sum of its two interfaces?"),
            (axes[0, 2], "B_vs_A", ("relative_residual", "residual_over_gb_signal", "share_of_atoms_near_planes"),
             "B against A: is the whole cell interfaces + bulk?")):
        part = bulk[bulk["method"] == test]
        subsets = [s for s in ("all", *sc.PLANES) if (part["subset"] == s).any()]
        width = 0.8 / max(1, len(variables))
        top = 0.0
        for v, (variable, color) in enumerate(zip(variables, (BLUE, ORANGE, AQUA))):
            rows = part[part["variable"] == variable].set_index("subset")
            x = np.arange(len(subsets)) + (v - (len(variables) - 1) / 2) * width
            median = [float(rows.loc[s, "value"]) if s in rows.index else np.nan for s in subsets]
            largest = [float(rows.loc[s, "main"]) if s in rows.index else np.nan for s in subsets]
            ax.bar(x, median, width=width, color=color, label=variable.replace("_", " ") + " (median)")
            ax.scatter(x, largest, marker="_", s=120, color=INK, zorder=3)
            top = np.nanmax([top, *median, *largest])
        ax.set_ylim(0, 1.45 * top if top > 0 else 1.0)  # legend above the bars
        ax.set_xticks(range(len(subsets)), ["all GBs" if s == "all" else f"{s} GBs" for s in subsets], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.grid(axis="x", visible=False)
    _finish(plt, fig, path, "Bulk test: do covariant vectors vanish in bulk, and what does the whole-cell vector add?",
            "Left: pristine, plane-reference and transformed bulk cells (triangle: the largest cell, dot: the median); "
            "dashed line 1e-3 = zero by symmetry, as in check_covariant.py. Middle and right: least-squares fit of "
            "the whole-cell vector from the two interface vectors weighted by their atom counts (plus the bulk vector "
            "for standard SOAP); bars are the median over GBs, black dash the largest. A residual near 0 means the "
            "whole-cell vector holds nothing beyond the interfaces.")


def maps_figure(maps, method, path):
    subsets = [s for s in SUBSET_NAMES if f"rows|{s}" in maps and any(f"{method}|{s}|{r}" in maps for r in "ABCD")]
    reps = [r for r in "ABCD" if any(f"{method}|{s}|{r}" in maps for s in subsets)]
    if not subsets or not reps:
        return
    from select_gbs import GRAIN_COLORS
    plane = maps["table|plane"].astype(str)
    grain = maps["table|grain_class"].astype(str)
    plt, fig, axes = _figure(len(subsets), len(reps), 3.0 * len(reps), 2.8 * len(subsets) + 1.4)
    for i, subset in enumerate(subsets):
        rows = maps[f"rows|{subset}"]
        reference = None
        for j, rep in enumerate(reps):
            ax = axes[i, j]
            key = f"{method}|{subset}|{rep}"
            if key not in maps:
                ax.set_visible(False)
                continue
            Y = np.asarray(maps[key], dtype=float)
            if reference is None:
                reference = Y
            elif len(Y) == len(reference):
                Y = _turned(reference, Y)[1]
            if subset == "pooled":
                for p, color in PLANE_COLORS.items():
                    m = plane[rows] == p
                    ax.scatter(Y[m, 0], Y[m, 1], s=2, color=color, linewidths=0)
            else:
                _scatter_grain(ax, Y, grain[rows])
            _maps_style(ax, f"{REP_NAMES[rep]} | {SUBSET_NAMES[subset]}")
    # Two legends: some plane and grain-class colours look alike, and each applies to different rows.
    dot = lambda c, label: plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=5, label=label)
    per_plane = len(subsets) > 1
    fig.legend(handles=[dot(c, f"plane {p}") for p, c in PLANE_COLORS.items()], loc="lower left",
               bbox_to_anchor=(0.06, 0.0), ncol=1, fontsize=7.5, frameon=False,
               title="top row: plane" if per_plane else "plane", title_fontsize=7.5)
    if per_plane:
        fig.legend(handles=[dot(c, g) for g, c in GRAIN_COLORS.items()], loc="lower right",
                   bbox_to_anchor=(0.97, 0.0), ncol=4, fontsize=7.5, frameon=False, title="other rows: grain class",
                   title_fontsize=7.5)
    rows_ = 3
    layout = ("Rows: all GBs (colour = plane), then each plane alone (colour = grain class)." if per_plane else
              "All GBs only (colour = plane): a map fitted to the plane is not drawn per plane.")
    _finish(plt, fig, path, f"{METHOD_NAMES.get(method, method)}: how does the map change with the representation?",
            "Columns: the four representations (A/B standard SOAP, C/D covariant lambda 1-2; A/C interface atoms, "
            f"B/D whole cell). {layout} Each GB gives two points, its edge and mid interface; B and D give both the "
            "same vector. Maps B-D are turned and scaled onto map A (Procrustes).",
            rect_bottom=(0.25 + 0.2 * rows_) / fig.get_size_inches()[1])


def pair_matrix(table, methods, variable, subset, column="value"):
    part = table[(table["kind"] == "pair") & (table["variable"] == variable) & (table["subset"] == subset)]
    part = part.set_index(["method", "representation"])
    values = np.full((len(methods), len(PAIRS)), np.nan)
    significant = np.zeros_like(values, dtype=bool)
    for i, method in enumerate(methods):
        for j, pair in enumerate(PAIRS):
            if (method, pair) in part.index:
                row = part.loc[(method, pair)]
                row = row.iloc[0] if getattr(row, "ndim", 1) > 1 else row
                values[i, j] = row[column]
                significant[i, j] = str(row["significant"]) == "True"
    return values, significant


def pair_changes_figure(table, methods, path):
    import matplotlib.colors as mcolors
    variables = [(v, name) for v, name in PAIR_VARIABLES if ((table["kind"] == "pair") & (table["variable"] == v)).any()]
    if not variables:
        return
    raw = table[(table["kind"] == "raw_pair") & (table["subset"] == "pooled")].set_index(["variable", "representation"])
    names = [METHOD_NAMES.get(m, m) for m in methods]
    height = 0.32 * (len(methods) + 1) + 2.2
    plt, fig, axes = _figure(1, len(variables), 3.0 * len(variables) + 1, height)
    for ax, (variable, name) in zip(axes[0], variables):
        values, significant = pair_matrix(table, methods, variable, "pooled")
        raw_row = np.array([[float(raw.loc[(variable, p), "value"]) if (variable, p) in raw.index else np.nan
                             for p in PAIRS]])
        # Every panel gets the raw row (blank where a score needs a 2D map), so the rows line up.
        matrix = np.vstack([values, raw_row])
        rows = names + ["raw descriptors"]
        mask = np.vstack([significant, np.ones((1, len(PAIRS)), dtype=bool)])
        top = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
        _heatmap(ax, matrix, rows, PAIRS, mask, norm=mcolors.Normalize(0, max(top, 1e-12)), cmap="magma_r")
        ax.set_title(name, fontsize=8.5, color=INK)
        if ax is not axes[0, 0]:
            ax.set_yticklabels([])
    _finish(plt, fig, path, "Which representation changes move each map more than seeds or subsamples do?",
            "All GBs (pooled), both interface rows. Each cell: mean change between the replicate maps of two "
            "representations (larger = more different; Procrustes 0-1, the others 1 - score). Bold: larger than the "
            "95th percentile of the replicate-to-replicate change of both representations (the noise floor); gray: "
            "within the noise. Last row: the same change between the raw descriptors (no map), the input change; "
            "blank for Procrustes and ARI, which need a 2D map.")


def sensitivity_figure(table, methods, path, by_plane_path):
    import matplotlib.colors as mcolors
    names = [METHOD_NAMES.get(m, m) for m in methods]
    variables = [(v, n) for v, n in PAIR_VARIABLES if v in ("jaccard15", "jaccard50", "spearman")
                 and ((table["kind"] == "pair") & (table["variable"] == v) & table["sensitivity"].notna()).any()]
    if not variables:
        return
    norm = mcolors.LogNorm(1 / 8, 8)
    height = 0.32 * len(methods) + 2.2
    note = ("Sensitivity = change of the map / change of the raw descriptors for the same two representations. "
            "Above 1 (red): the method magnifies the representation change; below 1 (blue): it hides it. Bold: the "
            "map change is above the noise floor; gray: within it, so the ratio means little.")
    plt, fig, axes = _figure(1, len(variables), 3.0 * len(variables) + 1, height)
    for ax, (variable, name) in zip(axes[0], variables):
        values, significant = pair_matrix(table, methods, variable, "pooled", "sensitivity")
        _heatmap(ax, values, names, PAIRS, significant, norm=norm, cmap="RdBu_r")
        ax.set_title(name, fontsize=8.5, color=INK)
        if ax is not axes[0, 0]:
            ax.set_yticklabels([])
    _finish(plt, fig, path, "Does each method magnify or hide the difference between two representations?",
            "All GBs (pooled). " + note)
    subsets = [s for s in SUBSET_NAMES if ((table["kind"] == "pair") & (table["subset"] == s)).any()]
    variable = variables[0][0]
    plt, fig, axes = _figure(1, len(subsets), 3.0 * len(subsets) + 1, height)
    for ax, subset in zip(axes[0], subsets):
        values, significant = pair_matrix(table, methods, variable, subset, "sensitivity")
        _heatmap(ax, values, names, PAIRS, significant, norm=norm, cmap="RdBu_r")
        ax.set_title(SUBSET_NAMES[subset], fontsize=8.5, color=INK)
        if ax is not axes[0, 0]:
            ax.set_yticklabels([])
    _finish(plt, fig, by_plane_path,
            f"Is the sensitivity the same within each plane? ({dict(PAIR_VARIABLES)[variable]})",
            "Each plane's GBs mapped on their own (LDA and PLS/PCovR on plane are left out there). " + note +
            " Blank: not defined (a subset or its 80% subsamples with fewer rows than neighbours + 3).")


def organisation_figure(table, methods, path):
    maps = table[(table["kind"] == "map") & (table["subset"] == "pooled")]
    raw = table[(table["kind"] == "raw") & (table["subset"] == "pooled")]
    baseline = table[(table["kind"] == "baseline") & (table["subset"] == "pooled")].set_index("variable")["value"]
    variables = [v for v in ORGANISATION if (maps["variable"] == v).any()]
    if not variables:
        return
    names = methods + ["raw"]
    columns = min(4, len(variables))
    rows = int(np.ceil(len(variables) / columns))
    plt, fig, axes = _figure(rows, columns, 3.3 * columns + 0.8, (0.22 * len(names) + 1.0) * rows + 1.2, sharey=True)
    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)
    for ax, variable in zip(axes.ravel(), variables):
        part = pd_concat([maps[maps["variable"] == variable], raw[raw["variable"] == variable]])
        part = part.rename(columns={"method": "name", "representation": "key"})
        _dot_rows(ax, part, names, "value", {r: REP_COLORS[r] for r in "ABCD"}, label_of=lambda m: METHOD_NAMES.get(m, m),
                  baseline=float(baseline.get(variable, np.nan)))
        ax.set_title(ORGANISATION[variable], fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=REP_COLORS[r], markersize=5, label=REP_NAMES[r])
               for r in "ABCD"] + [plt.Line2D([], [], color=MUTED, linestyle=":", label="guessing the largest class")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "What organises each map: plane, grain class, edge/mid, or the interface geometry?",
            "Each row predicted from its 10 nearest rows on the map (the other interface of the same cell left out): "
            "kNN accuracy for the classes, kNN R² for the geometry labels. All GBs (pooled), main map. Last row: the "
            "same on the raw descriptors. Dotted line: always answering the largest class.",
            rect_bottom=0.5 / ((0.22 * len(names) + 1.0) * rows + 1.2))


def pd_concat(parts):
    import pandas as pd
    return pd.concat(parts, ignore_index=True)


def faithfulness_figure(table, methods, path):
    maps = table[(table["kind"] == "map") & (table["subset"] == "pooled")]
    variables = [(v, n) for v, n in FAITHFULNESS if (maps["variable"] == v).any()]
    if not variables:
        return
    plt, fig, axes = _figure(1, len(variables), 3.4 * len(variables) + 0.8, 0.24 * len(methods) + 2.2, sharey=True)
    for ax, (variable, name) in zip(axes[0], variables):
        part = maps[maps["variable"] == variable].rename(columns={"method": "name", "representation": "key"})
        _dot_rows(ax, part, methods, "value", {r: REP_COLORS[r] for r in "ABCD"},
                  label_of=lambda m: METHOD_NAMES.get(m, m), error="sd")
        ax.set_title(name, fontsize=8.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=REP_COLORS[r], markersize=5, label=REP_NAMES[r])
               for r in "ABCD"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "How faithful is each map to its own input?",
            "Trustworthiness: are a row's 15 map neighbours also near in the descriptor space? Continuity: the "
            "reverse. Triplet accuracy: share of random triplets whose closer pair is the same in map and input. 1 = "
            "perfect. Main map; bar: spread over the replicates.", rect_bottom=0.45 / (0.24 * len(methods) + 2.2))


def effects_figure(table, methods, path, kind="effect", variables=None, title=None):
    effects = table[(table["kind"] == kind) & (table["subset"] == "pooled")]
    variables = variables or PAIR_VARIABLES
    variables = [(v, n) for v, n in variables if (effects["variable"] == v).any()]
    if not variables:
        return
    factors = ("region", "covariance", "interaction", "residual")
    colors = {"region": BLUE, "covariance": ORANGE, "interaction": AQUA, "residual": "#d0d0d0"}
    names = [METHOD_NAMES.get(m, m) for m in methods]
    plt, fig, axes = _figure(1, len(variables), 3.0 * len(variables) + 1, 0.26 * len(methods) + 2.0, sharey=True)
    for ax, (variable, name) in zip(axes[0], variables):
        part = effects[effects["variable"] == variable]
        shares = {METHOD_NAMES.get(m, m): dict(zip(g["factor"], g["value"])) for m, g in part.groupby("method")}
        _stacked(ax, shares, names, factors, colors)
        for i, method in enumerate(methods):
            row = part[(part["method"] == method) & (part["factor"].isin(("region", "covariance", "interaction")))]
            stars = "".join(f[0].upper() for f, p in zip(row["factor"], row["p_value"]) if p < 0.05)
            ax.text(1.01, len(methods) - 1 - i, stars, fontsize=6.5, color=INK, va="center",
                    transform=ax.get_yaxis_transform())
        ax.set_title(name, fontsize=8.5, color=INK)
        ax.set_xlabel("share of the variation", fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f], label=f) for f in factors]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    factors_text = ("share of the variation due to region (A, C vs B, D), covariance (A, B vs C, D), their "
                    "interaction, and the rest (seeds or subsamples).")
    if kind == "effect":
        note = ("Two-way PERMANOVA (Anderson 2001) on the map-to-map changes of all replicate maps of A-D: "
                + factors_text + " Letters right of a bar: R, C, I significant at p < 0.05 (permutation test).")
    else:
        note = ("Two-way ANOVA of each per-map score over all replicate maps of A-D: " + factors_text
                + " Letters right of a bar: R, C, I significant at p < 0.05 (F test).")
    _finish(plt, fig, path, title or "Is it the region (interface or whole cell) or the covariance that changes the map?",
            note, rect_bottom=0.45 / (0.26 * len(methods) + 2.0))


def method_numbers_figure(table, path):
    rows = table[(table["kind"] == "method") & (table["subset"] == "pooled")]
    panels = [(m, v, n) for m, v, n in METHOD_NUMBERS if ((rows["method"] == m) & (rows["variable"] == v)).any()]
    if not panels:
        return
    columns = min(5, len(panels))
    count = int(np.ceil(len(panels) / columns))
    plt, fig, axes = _figure(count, columns, 2.8 * columns, 2.1 * count + 1.0)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (method, variable, name) in zip(axes.ravel(), panels):
        part = rows[(rows["method"] == method) & (rows["variable"] == variable)].set_index("representation")
        keys = [k for k in (list("ABCD") + list(PAIRS)) if k in part.index]
        ax.bar(range(len(keys)), [float(part.loc[k, "value"]) for k in keys],
               color=[REP_COLORS.get(k, GRAY) for k in keys])
        ax.set_xticks(range(len(keys)), keys, fontsize=7)
        ax.set_title(f"{METHOD_NAMES.get(method, method)}: {name}", fontsize=7.5, color=INK)
        ds._style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.grid(axis="x", visible=False)
    _finish(plt, fig, path, "Do the methods' own diagnostics change with the representation?",
            "Main map of all GBs (pooled). Colours: A blue, B orange, C aqua, D yellow; pairs (A-B ...) in gray. "
            "Definitions in src/ML_dscribe_scripts/README.md (representation_study.py).")


def thickness_response(thickness_dir, path):
    import pandas as pd
    table = pd.read_csv(Path(thickness_dir) / "results.csv", dtype={"subset": str})
    part = table[table["kind"] == "thickness"]
    if part.empty:
        return
    methods = [m for m in METHOD_NAMES if (part["method"] == m).any()]
    variables = (("thickness_share", "share of the map's spread due to thickness\n(0 = ideal)"),
                 ("gb_share", "share due to the GB\n(1 = ideal)"),
                 ("same_gb_neighbours", "8 nearest rows that are the same GB\nat another thickness"))
    pcs = table[table["kind"] == "all_structures_pca"]
    columns = len(variables) + (1 if not pcs.empty else 0)
    plt, fig, axes = _figure(1, columns, 3.4 * columns + 0.8, 0.24 * len(methods) + 2.4)
    for ax, (variable, name) in zip(axes[0], variables):
        rows = part[part["variable"] == variable].rename(columns={"method": "name", "representation": "key"})
        _dot_rows(ax, rows, methods, "value", {r: REP_COLORS[r] for r in "ABCD"}, label_of=lambda m: METHOD_NAMES.get(m, m))
        ax.set_title(name, fontsize=8, color=INK)
        if ax is not axes[0, 0]:
            ax.set_yticklabels([])
    if not pcs.empty:
        ax = axes[0, -1]
        axes_names = list(dict.fromkeys(pcs["subset"]))
        width = 0.2
        for k, rep in enumerate("ABCD"):
            values = [pcs[(pcs["representation"] == rep) & (pcs["subset"] == a) & (pcs["variable"] == "thickness_share")]
                      ["value"].mean() for a in axes_names]
            ax.bar(np.arange(len(axes_names)) + (k - 1.5) * width, values, width=width, color=REP_COLORS[rep])
        ax.set_xticks(range(len(axes_names)), axes_names, fontsize=7.5)
        n_rows = int(pcs["n"].max())
        ax.set_title(f"PCA of all {n_rows:,} interface rows (every thickness):\nthickness share", fontsize=8, color=INK)
        ds._style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=REP_COLORS[r], markersize=5, label=REP_NAMES[r])
               for r in "ABCD"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    _finish(plt, fig, path, "Does any representation or method follow the cell thickness instead of the GB?",
            "Every method on all nine thicknesses (4x-12x) of a fixed set of GBs (thickness/summary.json), one map "
            "each. A GB's versions at different thicknesses should land together: low thickness share, high GB "
            "share, many same-GB neighbours.", rect_bottom=0.45 / (0.24 * len(methods) + 2.4))


def representation_response(representation_dir, out):
    table, maps, summaries = representation_tables(representation_dir)
    folder = out / "representation_response"
    if table is None:
        print(f"  skipped representation_response: no results.csv in {sc.relative_path(representation_dir)}")
        return
    methods = method_order(table)
    bulk_figure(table, folder / "bulk_test.png")
    for method in methods:
        maps_figure(maps, method, folder / f"maps_{method}.png")
    pair_changes_figure(table, methods, folder / "pair_changes.png")
    sensitivity_figure(table, methods, folder / "sensitivity.png", folder / "sensitivity_by_plane.png")
    organisation_figure(table, methods, folder / "organisation.png")
    faithfulness_figure(table, methods, folder / "faithfulness.png")
    effects_figure(table, methods, folder / "effects.png")
    effects_figure(table, methods, folder / "map_effects.png", kind="map_effect",
                   variables=[(v, ORGANISATION.get(v, dict(FAITHFULNESS).get(v, v))) for v in
                              ("knn_plane", "knn_grain_class", "knn_r2_d_min", "trustworthiness", "continuity")],
                   title="Which factor changes what organises the map: region or covariance?")
    method_numbers_figure(table, folder / "method_numbers.png")
    thickness_dir = Path(representation_dir) / "thickness"
    if (thickness_dir / "results.csv").is_file():
        thickness_response(thickness_dir, folder / "thickness.png")


# --------------------------------------------------------------------------- settings grid

def _grid_description(tags):
    """'r_cut 5/6/8 x l_max 4/8 x sigma 0.3/0.5' for the tags present (rc5.0_n8_l4_s0.5, ...)."""
    fields = [dict((p[:2], p[2:]) if p.startswith("rc") else (p[:1], p[1:]) for p in t.split("_")) for t in tags]
    values = lambda key: "/".join(f"{v:g}" for v in sorted({float(f[key]) for f in fields}))
    return f"r_cut {values('rc')} x l_max {values('l')} x sigma {values('s')}"


def settings_grid_figures(grid_dir, out):
    import pandas as pd
    folder = out / "settings_grid"
    table = pd.read_csv(Path(grid_dir) / "results.csv", dtype={"subset": str, "representation": str})
    methods = [m for m in METHOD_NAMES if (table["method"] == m).any()]
    names = [METHOD_NAMES.get(m, m) for m in methods]
    effects = table[table["kind"] == "effect"]
    variables = [(v, n) for v, n in PAIR_VARIABLES if (effects["variable"] == v).any()]
    factors = ("representation", "settings", "interaction", "seed_noise", "residual")
    colors = {"representation": BLUE, "settings": ORANGE, "interaction": AQUA, "seed_noise": "#d0d0d0",
              "residual": "#d0d0d0"}
    if variables:
        plt, fig, axes = _figure(1, len(variables), 3.0 * len(variables) + 1, 0.3 * len(methods) + 2.0, sharey=True)
        for ax, (variable, name) in zip(axes[0], variables):
            part = effects[effects["variable"] == variable]
            shares = {METHOD_NAMES.get(m, m): dict(zip(g["factor"], g["value"])) for m, g in part.groupby("method")}
            _stacked(ax, shares, names, factors, colors)
            ax.set_title(name, fontsize=8.5, color=INK)
            ax.set_xlabel("share of the variation", fontsize=8, color=INK)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f], label=f.replace("_", " ")) for f in factors[:4]]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
        summary = json.loads((Path(grid_dir) / "summary.json").read_text())
        _finish(plt, fig, folder / "variance_shares.png",
                "Is the map change due to the representation or to the SOAP settings?",
                f"Two-way PERMANOVA over {len(summary['tags'])} settings ({_grid_description(summary['tags'])}) x the 4 "
                f"representations, {summary['settings']['replicates']} replicates each: share of the variation between "
                "maps due to the representation, the settings, their interaction, and seed or subsample noise. Raw "
                "descriptors: no replicates, so the rest is the interaction. n/a: not defined (e.g. more neighbours "
                "than rows in a subsample).", rect_bottom=0.45 / (0.3 * len(methods) + 2.0))
    anchors = table[table["kind"] == "anchor"]
    floor = table[(table["kind"] == "grid_mean") & (table["representation"] == "replicates")]
    variable = next((v for v, _ in PAIR_VARIABLES if (anchors["variable"] == v).any() and v.startswith("jaccard")), None)
    if variable:
        tags = list(dict.fromkeys(anchors.loc[anchors["variable"] == variable, "factor"]))
        plt, fig, axes = _figure(1, len(methods), 3.4 * len(methods), 4.2, sharey=True)
        for ax, method in zip(axes[0], methods):
            part = anchors[(anchors["method"] == method) & (anchors["variable"] == variable)]
            for rep in "ABCD":
                rows = part[part["representation"] == rep].set_index("factor")
                ax.plot(range(len(tags)), [float(rows.loc[t, "value"]) if t in rows.index else np.nan for t in tags],
                        marker="o", markersize=3.5, color=REP_COLORS[rep], linewidth=1.2, label=REP_NAMES[rep])
            noise = floor[(floor["method"] == method) & (floor["variable"] == variable)]["noise_floor"]
            if not noise.empty:
                ax.axhline(float(noise.iloc[0]), color=MUTED, linewidth=0.9, linestyle=":")
            ax.set_xticks(range(len(tags)), [t.replace("_n8", "").replace("rc", "r") for t in tags], rotation=90,
                          fontsize=6.5)
            ax.set_title(METHOD_NAMES.get(method, method), fontsize=8.5, color=INK)
            ds._style(ax)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
        axes[0, 0].set_ylabel(dict(PAIR_VARIABLES)[variable] + "\nfrom the reference setting", fontsize=8, color=INK)
        axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
        _finish(plt, fig, folder / "anchor_distance.png",
                "How far does each SOAP setting move the map away from the reference setting (r_cut 6, l_max 6, sigma 0.5)?",
                "Mean change between the reference map and the replicate maps of each grid setting, per representation. "
                "Dotted line: 95th percentile of the replicate-to-replicate change (noise floor).")
    means = table[table["kind"] == "grid_mean"]
    if not means.empty and variables:
        plt, fig, axes = _figure(1, len(variables), 3.0 * len(variables) + 1, 0.3 * len(methods) + 2.0, sharey=True)
        groups = (("between_representations", BLUE, "same setting, other representation"),
                  ("between_settings", ORANGE, "same representation, other setting"),
                  ("replicates", "#b0b0b0", "same both (replicates)"))
        y = np.arange(len(methods))[::-1]
        for ax, (variable, name) in zip(axes[0], variables):
            for k, (label, color, legend) in enumerate(groups):
                part = means[(means["variable"] == variable) & (means["representation"] == label)].set_index("method")
                values = [float(part.loc[m, "value"]) if m in part.index else np.nan for m in methods]
                errors = np.array([float(part.loc[m, "sd"]) if m in part.index else 0.0 for m in methods])
                errors = np.nan_to_num(errors)
                lower = np.minimum(errors, np.nan_to_num(values))  # a change cannot go below 0
                ax.barh(y + (1 - k) * 0.26, values, height=0.26, xerr=[lower, errors], color=color, label=legend,
                        error_kw={"elinewidth": 0.6, "ecolor": MUTED})
            ax.set_yticks(y, names, fontsize=7.5)
            ax.set_title(name, fontsize=8.5, color=INK)
            ds._style(ax)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, c, l in groups]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
        _finish(plt, fig, folder / "between_settings_and_representations.png",
                "Which matters more for the map: the representation or the SOAP setting?",
                "Mean change between two maps (bar: standard deviation) that share the setting but not the "
                "representation, share the representation but not the setting, or share both (replicates).",
                rect_bottom=0.45 / (0.3 * len(methods) + 2.0))


def _inside(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def refusal(out_dir, prefix, inputs):
    """Why drawing into out_dir is refused, or None.

    Figures marked as not from the real full run (a title prefix), or drawn from inputs under
    archive/, never go into plots/CoGe/ or results/.
    """
    protected = [folder for folder in PROTECTED if _inside(out_dir, folder)]
    if not protected:
        return None
    where = sc.relative_path(protected[0])
    if prefix:
        return f"--title-prefix marks these figures as not from the real structures; they may not go into {where}/"
    for path in inputs:
        if _inside(path, sc.PROJECT_ROOT / "archive"):
            return f"input {sc.relative_path(path)} is under archive/; figures from it may not go into {where}/"
    return None


def main():
    global TITLE_PREFIX, ARCHIVE_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity-dir", type=Path, default=SENSITIVITY_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--covariant-dir", type=Path, default=COVARIANT_DIR)
    parser.add_argument("--labels-file", type=Path, default=LABELS_FILE, help="geometry_labels.csv")
    parser.add_argument("--krr-dir", type=Path, default=KRR_DIR,
                        help="krr_classify.py output with krr_by_plane.csv and krr_regression.csv")
    parser.add_argument("--representation-dir", type=Path, default=REPRESENTATION_DIR,
                        help="representation_study.py output (phase1/, phase2/, thickness/)")
    parser.add_argument("--grid-dir", type=Path, default=GRID_DIR, help="representation_study.py --study settings_grid output")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--archive-old", type=Path, default=None,
                        help="move each figure about to be redrawn to ARCHIVE_OLD/<its path in the project> first")
    parser.add_argument("--done-marker", type=Path, default=None, help="file written when every chosen section is drawn")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, help="sections to draw (default all)")
    parser.add_argument("--title-prefix", default="",
                        help='start every title with this, e.g. "SMOKE TEST (2 seeds)" or "SYNTHETIC DATA"; required for '
                             "figures not drawn from the full run on the real structures, and refused in plots/CoGe/ "
                             "and results/")
    parser.add_argument("--dry-run", action="store_true", help="list the sections and whether their inputs exist")
    args = parser.parse_args()

    needs = {"descriptor_settings": args.sensitivity_dir / "sensitivity.csv",
             "maps_by_setting": args.sensitivity_dir / "maps.npz",
             "seed_stability": args.sensitivity_dir / "maps.npz",
             "thickness": args.sensitivity_dir / "thickness_test.csv",
             "selection": args.benchmark_dir / "maps.npz",
             "covariant": args.covariant_dir / "covariant_soap.npz",
             "free_labels": args.labels_file,
             "representation_response": args.representation_dir / "phase1" / "results.csv",
             "settings_grid": args.grid_dir / "results.csv"}
    chosen = args.only or SECTIONS
    inputs = [needs[section].parent for section in chosen]
    if "selection" in chosen or "free_labels" in chosen:
        inputs.append(args.krr_dir)
    refused = refusal(args.out_dir, args.title_prefix, inputs)
    if refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    if args.archive_old is not None and not _inside(args.archive_old, sc.PROJECT_ROOT / "archive"):
        print("refused: --archive-old must be a folder under archive/", file=sys.stderr)
        return 2
    TITLE_PREFIX = args.title_prefix.strip()
    ARCHIVE_ROOT = args.archive_old
    if args.dry_run:
        for section in chosen:
            print(f"{section:20s} {'ok' if needs[section].is_file() else 'MISSING'}  {sc.relative_path(needs[section])}")
        return 0
    draw = {"descriptor_settings": lambda: descriptor_settings(args.sensitivity_dir, args.out_dir),
            "maps_by_setting": lambda: maps_by_setting(args.sensitivity_dir, args.out_dir),
            "seed_stability": lambda: seed_stability(args.sensitivity_dir, args.out_dir),
            "thickness": lambda: thickness(args.sensitivity_dir, args.out_dir),
            "selection": lambda: selection(args.benchmark_dir, args.out_dir, args.krr_dir),
            "covariant": lambda: covariant(args.covariant_dir, args.out_dir),
            "free_labels": lambda: free_labels(args.labels_file, args.krr_dir, args.out_dir),
            "representation_response": lambda: representation_response(args.representation_dir, args.out_dir),
            "settings_grid": lambda: settings_grid_figures(args.grid_dir, args.out_dir)}
    skipped = []
    for section in chosen:
        if not needs[section].is_file():
            print(f"{section}: skipped, {sc.relative_path(needs[section])} not found")
            skipped.append(section)
            continue
        print(f"{section}:")
        draw[section]()
    if args.done_marker is not None:
        if skipped:
            print(f"no done marker: {', '.join(skipped)} skipped", file=sys.stderr)
            return 1
        args.done_marker.parent.mkdir(parents=True, exist_ok=True)
        args.done_marker.write_text(f"sections drawn: {' '.join(chosen)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
