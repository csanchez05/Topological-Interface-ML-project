"""Select 30 distinct CoGe GBs in each of three two-dimensional SOAP (or ACE) maps.

Run from the project root in src/mlenv, or pass --verify-only to check an
existing run without fitting the embeddings again. The historical output
folder name 'median' means the actual member nearest a K-means centroid.

--descriptor-type ace maps the ACE vectors of src/ACE/CoGe (default run
rc6.0_n8_l6_s0.5_nu3-n4-l4, all blocks, each divided by its pristine-bulk length;
--ace-blocks / --ace-scaling change that) instead of SOAP. Its default folders are
plots/CoGe/dim-reduction/ACE/ and calculations/ML_ACE/CoGe/dim-reduction/.
--descriptor-type covariant maps covariant SOAP (default run rc6.0_n8_l6_s0.5, the lambda >= 1
blocks unscaled, the two interfaces joined as [edge | mid]; --covariant-blocks changes the
blocks); its folders are plots/CoGe/dim-reduction/SOAP/covariant/ and
calculations/ML_dscribe/CoGe/covariant/dim-reduction/. Standard SOAP (the default) uses
.../SOAP/standard/ and .../CoGe/standard/dim-reduction/. The distinct GBs are the same 1,112
in every case (structure census).

Subset selection (--planes, --min-pair, --n-select): only the GBs of the given
planes whose closest pair of atoms at 8x is >= --min-pair (from the overlap
survey) are embedded, together with pristine bulk, and sampled. The default
folders then get a tag, e.g. plots/CoGe/dim-reduction/SOAP/standard/plane100_dmin1.8/, so
the Stage 2 lists are never overwritten. --dry-run only counts the subset.

  python -B src/ML_dscribe_scripts/select_gbs.py --planes 100 --min-pair 1.8 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

import ml_common as mc

sc = mc.sc
ROOT = mc.PROJECT_ROOT
DESCRIPTOR = "rc6.0_n8_l6_s0.5"
THICKNESS = 8
CENSUS = mc.OUTPUT_ROOT / DESCRIPTOR / "interface_r8_both_t4" / "distinct_gbs.csv"
GB_LABELS = mc.LABELS_DIR / "gb_labels.csv"
PLOT_DIR = mc.plot_root("soap")
OUT_DIR = mc.OUTPUT_ROOT / "dim-reduction"
# Default folders per descriptor type (the SOAP ones are PLOT_DIR and OUT_DIR).
PLOT_DIRS = {kind: mc.plot_root(kind) for kind in mc.DESCRIPTOR_TYPES}
OUT_DIRS = {kind: mc.OUTPUT_ROOTS[kind] / "dim-reduction" for kind in mc.DESCRIPTOR_TYPES}
SURVEY = ROOT / "results" / "CoGe_interface" / "overlap_survey.tsv"
METHODS = ("pca", "tsne", "umap")
SAMPLERS = ("FPS", "median")
N_SELECT = 30
TITLE_WIDTH = 90  # characters of a map title on one line (the covariant label is 86)
GRAIN_COLORS = {
    "E": "#0072B2", "I": "#E69F00",
    "R90[100]": "#009E73", "I.R90[100]": "#CC79A7",
    "R60[111]": "#56B4E9", "I.R60[111]": "#D55E00",
    "R70.53[1-10]": "#8A5FB5", "I.R70.53[1-10]": "#6B4C3B",
}
GRAY = "#B0B0B0"
RED = "#FF0000"


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while reading: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def representative_rows(index, census):
    """Return index positions for the first family of every structure class."""
    if index["family"].duplicated().any() or census["family"].duplicated().any():
        raise ValueError("expected exactly one row per GB family")
    if set(index["family"]) != set(census["family"]):
        raise ValueError("descriptor families and structure census do not match")
    if census["structure_class"].isna().any():
        raise ValueError("missing structure census class")
    merged = index.assign(vector_row=np.arange(len(index))).merge(
        census[["family", "structure_class"]], on="family", validate="one_to_one")
    chosen = merged.sort_values("family").drop_duplicates("structure_class")
    return chosen["vector_row"].to_numpy(), chosen.drop(columns="vector_row").reset_index(drop=True)


def attach_grain_classes(index, labels):
    """Join the established grain labels by family, independently of row order."""
    if labels["family"].duplicated().any():
        raise ValueError("duplicate family in grain-class labels")
    merged = index.merge(labels[["family", "grain_class"]], on="family",
                         how="left", validate="one_to_one", sort=False)
    if merged["grain_class"].isna().any():
        raise ValueError("missing grain-class label for a descriptor family")
    unknown = set(merged["grain_class"]) - set(GRAIN_COLORS)
    if unknown:
        raise ValueError(f"unknown grain-class labels: {sorted(unknown)}")
    return merged


def overlap_filter(index, census, survey, planes, min_pair, thickness=THICKNESS):
    """Mask over the representative rows: the GBs of `planes` whose closest pair of atoms at
    `thickness`x is >= min_pair (A; None = no limit) in every family of their structure class.

    `survey` is overlap_survey.tsv (plane, sym, thickness, status, dmin_A rounded to 0.001 A).
    A class whose families fall on both sides of the limit, a class spanning planes, or a family
    of `planes` without a closest pair stops the run.
    """
    planes = {str(plane) for plane in planes}
    census = census.assign(plane=census["plane"].astype(str))
    mixed = census.groupby("structure_class")["plane"].nunique() > 1
    if mixed.any():
        raise ValueError(f"structure classes spanning planes: {sorted(mixed[mixed].index)[:5]}")
    members = census[census["plane"].isin(planes)]
    if min_pair is None:
        passing = set(members["structure_class"])
    else:
        rows = survey[survey["thickness"].astype(str) == f"{thickness}x"]
        lookup = {(str(plane), sym): (status, dmin) for plane, sym, status, dmin in
                  zip(rows["plane"], rows["sym"], rows["status"], rows["dmin_A"])}
        verdicts = {}
        for family, plane, sym, cls in zip(members["family"], members["plane"], members["sym"],
                                           members["structure_class"]):
            status, dmin = lookup.get((plane, sym), ("missing", np.nan))
            if status != "ok" or not np.isfinite(float(dmin)):
                raise ValueError(f"no {thickness}x closest pair for {family} ({status})")
            verdicts.setdefault(cls, set()).add(float(dmin) >= min_pair)
        split = sorted(cls for cls, verdict in verdicts.items() if len(verdict) > 1)
        if split:
            raise ValueError(f"structure classes straddling {min_pair} A: {split}")
        passing = {cls for cls, verdict in verdicts.items() if verdict == {True}}
    mask = index["structure_class"].isin(passing).to_numpy()
    return mask, {"planes": sorted(planes), "min_pair_A": min_pair, "thickness": thickness,
                  "families": int(members["structure_class"].isin(passing).sum()),
                  "distinct_gbs": int(mask.sum())}


def selection_tag(planes, min_pair, n_select):
    """Folder name of a selection other than the Stage 2 default (None for the default)."""
    parts = []
    if set(planes) != set(sc.PLANES):
        parts.append("plane" + "-".join(sorted(planes)))
    elif min_pair is not None or n_select != N_SELECT:
        parts.append("allplanes")
    if min_pair is not None:
        parts.append(f"dmin{min_pair:g}")
    if n_select != N_SELECT:
        parts.append(f"n{n_select}")
    return "_".join(parts) or None


def subset_title(subset):
    planes = "/".join(f"[{plane}]" for plane in subset["planes"])
    limit = "" if subset["min_pair_A"] is None else f", closest pair ≥ {subset['min_pair_A']:g} Å"
    return f"{planes} GBs{limit} ({subset['distinct_gbs']} distinct)"


def default_folders(descriptor_type, descriptor_tag, ace_blocks, ace_scaling, planes, min_pair, n_select,
                    covariant_blocks=None):
    """(plot folder, output folder) of a selection.

    The Stage 2 default (SOAP rc6.0_n8_l6_s0.5, all planes, 30 per list) and the default ACE and
    covariant vectors use the type's root folders; any other descriptor, block choice, scaling or
    subset adds a tag, so a selection never overwrites another one.
    """
    parts = []
    if descriptor_type == "ace":
        if descriptor_tag != mc.DEFAULT_DESCRIPTORS["ace"]:
            parts.append(descriptor_tag)
        parts.append(mc.ace_suffix(ace_blocks, ace_scaling).lstrip("_"))
    elif descriptor_type == "covariant":
        if descriptor_tag != mc.DEFAULT_DESCRIPTORS["covariant"]:
            parts.append(descriptor_tag)
        parts.append(mc.covariant_module().blocks_suffix(covariant_blocks).lstrip("_"))
    elif descriptor_tag != DESCRIPTOR:
        parts.append(descriptor_tag)
    parts.append(selection_tag(planes, min_pair, n_select) or "")
    tag = "_".join(part for part in parts if part)
    plot_root, out_root = PLOT_DIRS[descriptor_type], OUT_DIRS[descriptor_type]
    return (plot_root / tag, out_root / tag) if tag else (plot_root, out_root)


def existing_outputs(plot_dir, out_dir):
    """Files of an earlier selection in the target folders."""
    paths = [Path(out_dir) / name for name in ("run.json", "structures.csv", "embeddings.npz", "sample_list.csv")]
    paths += [Path(plot_dir) / sampler / f"{method}.txt" for sampler in SAMPLERS for method in METHODS]
    return [path for path in paths if path.exists()]


def sample_list_text(sources):
    """The union of the six lists in the layout of results/dscribe/total_sample_list.csv."""
    return "Structure\n" + "".join(source + "\n" for source in sorted(set(sources)))


def _sampling_input(points, families, count):
    points = np.asarray(points, dtype=np.float64)
    families = np.asarray(families, dtype=str)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("expected finite two-dimensional coordinates")
    if len(families) != len(points) or len(set(families)) != len(families):
        raise ValueError("expected one unique family name per point")
    if not 1 <= count <= len(points):
        raise ValueError("selection count must be between one and the candidate count")
    return points, np.argsort(families, kind="stable")


def farthest_points(points, families, count=N_SELECT, bulk=(0.0, 0.0)):
    """Seed farthest from bulk, then maximize distance to the selected set.

    Squared Euclidean distances preserve the ordering. Sorting candidate
    indices by family before argmax resolves exact ties independently of
    input row order. Already selected rows are explicitly excluded.
    """
    points, order = _sampling_input(points, families, count)
    bulk = np.asarray(bulk, dtype=np.float64)
    if bulk.shape != (2,) or not np.isfinite(bulk).all():
        raise ValueError("expected a finite two-dimensional bulk reference")
    initial = np.square(points - bulk).sum(axis=1)
    current = int(order[np.argmax(initial[order])])
    nearest = np.full(len(points), np.inf)
    chosen = []
    for _ in range(count):
        chosen.append(current)
        nearest = np.minimum(nearest, np.square(points - points[current]).sum(axis=1))
        nearest[chosen] = -np.inf
        current = int(order[np.argmax(nearest[order])])
    return np.array(chosen, dtype=int)


def nearest_centroids(points, families, labels, centers):
    """Choose an actual member of each cluster, with family-name tie breaks."""
    centers = np.asarray(centers, dtype=np.float64)
    points, order = _sampling_input(points, families, len(centers))
    labels = np.asarray(labels)
    if centers.shape != (len(centers), 2) or not np.isfinite(centers).all():
        raise ValueError("expected finite two-dimensional centroids")
    if labels.shape != (len(points),) or set(labels) != set(range(len(centers))):
        raise ValueError("every cluster must have at least one member")
    chosen = []
    for cluster, center in enumerate(centers):
        members = order[labels[order] == cluster]
        distances = np.square(points[members] - center).sum(axis=1)
        chosen.append(int(members[np.argmin(distances)]))
    return np.array(chosen, dtype=int)


def cluster_centers(points, families, seed, count=N_SELECT):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=count, n_init=20, random_state=seed)
    labels = model.fit_predict(points)
    selected = nearest_centroids(points, families, labels, model.cluster_centers_)
    return selected, labels, model.cluster_centers_, model.get_params(deep=False)


def load_dataset(planes=sc.PLANES, min_pair=None, survey_path=SURVEY, descriptor_type="soap", descriptor=None,
                 ace_blocks=None, ace_scaling="pristine", covariant_blocks=None):
    """Vectors of the distinct 8x GBs plus pristine bulk (last row). With a plane or closest-pair
    filter only the GBs that pass are kept; `subset` describes the filter (None without one).
    The settings returned are the descriptor settings of the run (its "soap" or "ace" entry, or a
    covariant run's "settings"). Covariant vectors join the two interfaces, [edge | mid]."""
    import pandas as pd

    descriptor = descriptor or mc.DEFAULT_DESCRIPTORS[descriptor_type]
    gb_dir = mc.run_dir("both_interfaces", descriptor, descriptor_type)
    bulk_dir = mc.run_dir("global", descriptor, descriptor_type)
    bulk_settings_name = "bulk_settings.json"
    if descriptor_type == "ace":
        ac = mc.ace_module()
        files = (ac.INTERFACE_FILE, ac.BULK_FILE)
        code = [Path(sc.__file__), Path(ac.__file__)]
    elif descriptor_type == "covariant":
        cc = mc.covariant_module()
        files = (cc.FILE, cc.BULK_FILE)
        code = [Path(sc.__file__), Path(cc.__file__)]
        bulk_settings_name = cc.SETTINGS_FILE   # the global run's settings.json covers its bulk cells
    else:
        files = (sc.INTERFACE_FILE, "bulk_soap.npz")
        code = [Path(sc.__file__)]
    inputs = [folder / name for folder, names in (
        (gb_dir, ("settings.json", "index.csv", files[0])),
        (bulk_dir, (bulk_settings_name, "bulk_index.csv", files[1]))) for name in names]
    inputs += [CENSUS, GB_LABELS, Path(__file__), Path(mc.__file__)] + code
    if min_pair is not None:
        inputs.append(Path(survey_path))
    hashes = {sc.relative_path(path): fingerprint(path) for path in inputs}
    gb_settings = json.loads((gb_dir / "settings.json").read_text())
    bulk_settings = json.loads((bulk_dir / bulk_settings_name).read_text())
    key = "settings" if descriptor_type == "covariant" else descriptor_type
    if gb_settings[key] != bulk_settings[key]:
        raise ValueError(f"bulk and GB {descriptor_type.upper()} settings do not match")
    source = {"descriptor_type": descriptor_type, "ace_blocks": ace_blocks, "ace_scaling": ace_scaling,
              "covariant_blocks": covariant_blocks}
    X, index, notes = mc.load_family_vectors(descriptor, "interface", 8.0, "both", THICKNESS, **source)
    if notes:
        raise ValueError(f"{THICKNESS}x catalogue is incomplete or too thin: {notes}")
    census = pd.read_csv(CENSUS, dtype={"plane": str})
    labels = pd.read_csv(GB_LABELS, dtype={"plane": str})
    index = attach_grain_classes(index, labels)
    actual_classes = index.set_index("family")["grain_class"]
    if not actual_classes.reindex(census["family"]).to_numpy().tolist() == census["grain_class"].tolist():
        raise ValueError("grain-class labels disagree with the Stage 1 census")
    if (census.groupby("structure_class")["grain_class"].nunique() > 1).any():
        raise ValueError("a structure census class contains conflicting grain-class labels")
    positions, index = representative_rows(index, census)
    X = X[positions]
    if descriptor_type == "soap" and descriptor == DESCRIPTOR:
        width = 1904
    elif descriptor_type == "soap":
        width = gb_settings["vector_length"]
    elif descriptor_type == "covariant":
        width = cc.vector_length(gb_dir, covariant_blocks, "interface", "both")
    else:
        width = sum(length for name, length in gb_settings["vector_length"].items()
                    if ace_blocks is None or name in ace_blocks)
    if len(census) != 1541 or len(index) != 1112 or X.shape != (1112, width):
        raise ValueError("the stored catalogue no longer matches the validated Stage 1 dataset")
    if descriptor_type == "ace":
        bulk_X, bulk_index = mc.ace_module().load_bulk_vectors(bulk_dir, ace_blocks, ace_scaling)
    elif descriptor_type == "covariant":
        bulk_X, bulk_index = cc.load_bulk_vectors(bulk_dir, covariant_blocks, "interface", "both")
    else:
        bulk_X, bulk_index = sc.load_bulk(descriptor)
    pristine = np.flatnonzero(bulk_index["kind"].to_numpy() == "pristine")
    if len(pristine) != 1 or bulk_X.shape[1] != X.shape[1]:
        raise ValueError("expected one pristine bulk vector matching the GB feature layout")
    if set(index["grain_class"]) != set(GRAIN_COLORS):
        raise ValueError("expected all eight established grain classes")
    if not (index["thickness"] == THICKNESS).all() or not index["source"].str.endswith(f"/{THICKNESS}x/POSCAR").all():
        raise ValueError("descriptor sources do not point to 8x structures")
    subset = None
    if set(planes) != set(sc.PLANES) or min_pair is not None:
        survey = None if min_pair is None else pd.read_csv(survey_path, sep="\t", dtype=str)
        mask, subset = overlap_filter(index, census, survey, planes, min_pair)
        if not mask.any():
            raise ValueError(f"no GB passes the subset filter {subset}")
        X, index = X[mask], index[mask].reset_index(drop=True)
        census = census[census["structure_class"].isin(set(index["structure_class"]))].reset_index(drop=True)
        subset["embedded"] = "only these GBs and pristine bulk"
        if min_pair is not None:
            subset["survey"] = sc.relative_path(survey_path)
            subset["survey_sha256"] = hashes[sc.relative_path(survey_path)]["sha256"]
    table = index[["family", "structure_class", "grain_class", "plane", "sym", "thickness", "source", "index"]].copy()
    table = table.rename(columns={"index": "descriptor_row"})
    table.insert(0, "kind", "gb")
    table.loc[len(table)] = {
        "kind": "pristine", "family": "bulk/pristine", "structure_class": "", "grain_class": "",
        "plane": "", "sym": "pristine", "thickness": "",
        "source": bulk_index.iloc[pristine[0]]["source"], "descriptor_row": int(pristine[0])}
    table.insert(0, "row", np.arange(len(table)))
    for source in table["source"]:
        path = ROOT / source
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty structure: {source}")
    vectors = np.vstack((X, bulk_X[pristine]))
    if not np.isfinite(vectors).all():
        raise ValueError(f"nonfinite {descriptor_type.upper()} features")
    return vectors, table, census, hashes, gb_settings[key], subset


def embed(vectors, method, seed):
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2, svd_solver="full", whiten=False)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                     max_iter=1000, metric="euclidean", random_state=seed, n_jobs=1)
    elif method == "umap":
        from umap import UMAP
        model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=500,
                     metric="euclidean", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"unknown embedding {method}")
    started = time.monotonic()
    coordinates = np.asarray(model.fit_transform(vectors), dtype=np.float64)
    details = {"parameters": model.get_params(deep=False),
               "seconds": round(time.monotonic() - started, 3),
               "original_bulk_coordinates": coordinates[-1].tolist()}
    coordinates -= coordinates[-1].copy()
    if coordinates.shape != (len(vectors), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{method} produced invalid coordinates")
    if method == "pca":
        details["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
    if method == "tsne":
        details["kl_divergence"] = float(model.kl_divergence_)
        details["n_iter"] = int(model.n_iter_)
    return coordinates, details


def map_axis_labels(method, details):
    if method == "pca":
        return tuple(f"PC{i + 1} ({ratio:.1%} variance)"
                     for i, ratio in enumerate(details["explained_variance_ratio"]))
    name = {"tsne": "t-SNE", "umap": "UMAP"}[method]
    return (f"{name} 1", f"{name} 2")


def draw_map(path, coordinates, grain_classes, selected=None, *, axis_labels, selected_background="gray",
             title=None):
    """Map of the GBs (bulk = last row). Only the grain classes present get a legend entry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A title longer than one line (descriptor label; subset) is shown on two; the PNG metadata keeps it whole.
    shown = "\n".join(textwrap.fill(part, TITLE_WIDTH) for part in title.split("; ")) if title else ""
    fig = plt.figure(figsize=(8, 6), dpi=200, facecolor="white")
    ax = fig.add_axes((0.12, 0.13, 0.61, 0.82 if len(title or "") <= TITLE_WIDTH else 0.78))
    grain_classes = np.asarray(grain_classes, dtype=str)
    class_colored = selected is None or selected_background == "grain_class"
    if title:
        fig.text(0.425, 0.99, title if len(title) <= TITLE_WIDTH else shown, ha="center", va="top", fontsize=10)
    if class_colored:
        for grain_class, color in GRAIN_COLORS.items():
            members = grain_classes == grain_class
            if not members.any():
                continue
            ax.scatter(coordinates[:-1][members, 0], coordinates[:-1][members, 1],
                       c=color, s=14, linewidths=0, alpha=1, zorder=1, label=grain_class)
    else:
        ax.scatter(coordinates[:-1, 0], coordinates[:-1, 1], c=GRAY, s=14,
                   linewidths=0, alpha=1, zorder=1, label="Other GBs")
    ax.scatter(*coordinates[-1], c=GRAY, marker="D", s=36, linewidths=0, zorder=2,
               label="Pristine bulk")
    if selected is not None:
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], c=RED,
                   s=48, linewidths=0, alpha=1, zorder=3, label=f"Selected ({len(selected)})")
    limits = np.stack((coordinates.min(axis=0), coordinates.max(axis=0)))
    padding = np.maximum(limits[1] - limits[0], 1e-12) * 0.05
    ax.set_xlim(limits[0, 0] - padding[0], limits[1, 0] + padding[0])
    ax.set_ylim(limits[0, 1] - padding[1], limits[1, 1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_labels[0], fontsize=11, labelpad=8)
    ax.set_ylabel(axis_labels[1], fontsize=11, labelpad=8)
    ax.tick_params(axis="both", labelsize=9, direction="out", colors="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#666666")
    ax.set_axisbelow(True)
    ax.grid(color="#E6E6E6", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if selected is not None:
        handles, labels = [handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]]
    legend_title = "Grain class" if selected is None else "Grain class / selection" if class_colored else "Selection"
    legend = ax.legend(handles, labels, title=legend_title, loc="upper left", bbox_to_anchor=(1.03, 1),
                       fontsize=9, title_fontsize=10, framealpha=1, borderaxespad=0)
    fig.canvas.draw()
    bounds = legend.get_window_extent()
    if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
        plt.close(fig)
        raise ValueError(f"legend would be clipped: {path}")
    metadata = {"Legend": json.dumps([text.get_text() for text in legend.get_texts()]),
                "Legend title": legend_title, "Axis labels": json.dumps(axis_labels),
                "Color by": "grain_class" if class_colored else "selection"}
    if title:
        metadata["Title"] = title
    tmp = sc._temporary(path)
    fig.savefig(tmp, dpi=200, facecolor="white", metadata=metadata)
    plt.close(fig)
    os.replace(tmp, path)


def write_text(path, text):
    tmp = sc._temporary(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def verify_outputs(plot_dir, out_dir, manifest):
    """Replay sampling, check structure identities, and decode all twelve plots."""
    from PIL import Image

    subset = manifest.get("subset")
    n_gbs, n_select = manifest["n_distinct_gbs"], manifest.get("n_select", N_SELECT)
    if subset is None and n_gbs != 1112:
        raise ValueError("a selection without a subset filter must hold the 1,112 distinct GBs")
    with (out_dir / "structures.csv").open() as handle:
        structures = list(csv.DictReader(handle))
    gbs, bulk = structures[:-1], structures[-1]
    if len(gbs) != n_gbs or any(row["kind"] != "gb" for row in gbs) or bulk["kind"] != "pristine":
        raise ValueError("incorrect structure mapping")
    families = np.array([row["family"] for row in gbs])
    grain_classes = np.array([row["grain_class"] for row in gbs])
    if len(set(families)) != n_gbs or len({row["structure_class"] for row in gbs}) != n_gbs:
        raise ValueError("structure mapping contains duplicate GBs")
    if subset is not None:
        if n_gbs != subset["distinct_gbs"] or any(row["plane"] not in subset["planes"] for row in gbs):
            raise ValueError("structure mapping does not match the subset filter")
        if subset["min_pair_A"] is not None:
            if fingerprint(ROOT / subset["survey"])["sha256"] != subset["survey_sha256"]:
                raise ValueError(f"overlap survey changed: {subset['survey']}")
            with (ROOT / subset["survey"]).open(newline="") as handle:
                dmin = {(row["plane"], row["sym"]): float(row["dmin_A"])
                        for row in csv.DictReader(handle, delimiter="\t")
                        if row["thickness"] == f"{subset['thickness']}x" and row["status"] == "ok"}
            short = [row["family"] for row in gbs if dmin[(row["plane"], row["sym"])] < subset["min_pair_A"]]
            if short:
                raise ValueError(f"GBs below the closest-pair limit in the mapping: {short[:5]}")
    label_lookup = {row["family"]: row["grain_class"] for row in sc.read_csv(GB_LABELS)}
    for row in gbs:
        if row["thickness"] != str(THICKNESS) or not row["source"].endswith(f"/{THICKNESS}x/POSCAR"):
            raise ValueError("structure mapping contains a non-8x GB")
        if row["grain_class"] != label_lookup[row["family"]]:
            raise ValueError("grain-class mapping does not match the established family labels")
    present = [name for name in GRAIN_COLORS if name in set(grain_classes)]
    if subset is None and present != list(GRAIN_COLORS):
        raise ValueError("structure mapping does not cover all eight grain classes")
    report = {"n_gbs": len(gbs), "n_select": n_select, "subset": subset, "bulk_selectable": False,
              "lists": {}, "images": {}}
    selected_sources = set()
    with np.load(out_dir / "embeddings.npz", allow_pickle=False) as arrays:
        for method in METHODS:
            coordinates = arrays[method]
            if (coordinates.shape != (n_gbs + 1, 2) or not np.isfinite(coordinates).all()
                    or not np.array_equal(coordinates[-1], np.zeros(2))):
                raise ValueError(f"invalid {method} embedding or bulk origin")
            replay_fps = farthest_points(coordinates[:-1], families, count=n_select)
            replay_center, labels, centers, _ = cluster_centers(coordinates[:-1], families, manifest["seed"],
                                                                count=n_select)
            np.testing.assert_array_equal(labels, arrays[f"{method}_cluster_labels"])
            np.testing.assert_array_equal(centers, arrays[f"{method}_cluster_centers"])
            for sampler, replay in (("FPS", replay_fps), ("median", replay_center)):
                selected = arrays[f"{method}_{sampler}_indices"]
                np.testing.assert_array_equal(selected, replay)
                if (selected.shape != (n_select,) or len(set(selected.tolist())) != n_select
                        or np.any(selected < 0) or np.any(selected >= len(gbs))):
                    raise ValueError(f"invalid selection: {method}/{sampler}")
                rows = [gbs[int(i)] for i in selected]
                if len({row["structure_class"] for row in rows}) != n_select:
                    raise ValueError(f"repeated structure class: {method}/{sampler}")
                sources = sorted(row["source"] for row in rows)
                text_path = plot_dir / sampler / f"{method}.txt"
                if text_path.read_text() != "".join(source + "\n" for source in sources):
                    raise ValueError(f"list does not match selected coordinates: {text_path}")
                for source in sources:
                    path = ROOT / source
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"missing or empty selection: {source}")
                selected_sources.update(sources)
                report["lists"][f"{sampler}/{method}.txt"] = {
                    "count": len(sources), "distinct_structures": n_select,
                    "plane_counts": {plane: sum(row["plane"] == plane for row in rows) for plane in sc.PLANES},
                    "grain_class_counts": {name: sum(row["grain_class"] == name for row in rows) for name in GRAIN_COLORS}}
                for view in ("colored", "selected"):
                    image_path = plot_dir / sampler / f"{method}_{view}.png"
                    with Image.open(image_path) as img:
                        img.load()
                        if img.size != (1600, 1200):
                            raise ValueError(f"incorrect PNG dimensions: {image_path}")
                        rgb = np.asarray(img.convert("RGB"))
                        class_colored = view == "colored" or manifest["plots"]["selected_background"] == "grain_class"
                        expected_legend = [*present, "Pristine bulk"] if class_colored else ["Other GBs", "Pristine bulk"]
                        if view == "selected":
                            expected_legend.insert(0, f"Selected ({n_select})")
                        if json.loads(img.info.get("Legend", "[]")) != expected_legend:
                            raise ValueError(f"missing or incorrect legend entries: {image_path}")
                        if img.info.get("Title") != manifest["plots"].get("title"):
                            raise ValueError(f"missing or incorrect title: {image_path}")
                        if json.loads(img.info.get("Axis labels", "[]")) != list(manifest["plots"]["axis_labels"][method]):
                            raise ValueError(f"incorrect axis labels: {image_path}")
                    red_pixels = int(np.all(rgb == (255, 0, 0), axis=2).sum())
                    if (view == "selected") != (red_pixels > 0):
                        raise ValueError(f"incorrect selection coloring: {image_path}")
                    if view == "selected" and not class_colored:
                        # Antialiased red on white/gray still has G == B and R >= G.
                        if np.any(rgb[:, :, 1] != rgb[:, :, 2]) or np.any(rgb[:, :, 0] < rgb[:, :, 1]):
                            raise ValueError(f"unexpected colors in red/gray map: {image_path}")
                    report["images"][f"{sampler}/{image_path.name}"] = {
                        "size": [1600, 1200], "bright_red_pixels": red_pixels,
                        "legend_entries": expected_legend, "color_by": "grain_class" if class_colored else "selection"}
            if ((plot_dir / "FPS" / f"{method}_colored.png").read_bytes() !=
                    (plot_dir / "median" / f"{method}_colored.png").read_bytes()):
                raise ValueError(f"colored views do not match for {method}")
    union = out_dir / "sample_list.csv"
    if sc.relative_path(union) in manifest.get("outputs", {}) and union.read_text() != sample_list_text(selected_sources):
        raise ValueError(f"union list does not match the six lists: {union}")
    for source, expected in manifest["selected_structure_hashes"].items():
        if fingerprint(ROOT / source)["sha256"] != expected["sha256"]:
            raise ValueError(f"selected structure changed: {source}")
    for source, expected in manifest.get("outputs", {}).items():
        if fingerprint(ROOT / source)["sha256"] != expected["sha256"]:
            raise ValueError(f"output changed: {source}")
    report.update(status="passed", thickness=THICKNESS, grain_classes=present,
                  legends_verified=True, text_files=len(report["lists"]), png_files=len(report["images"]),
                  unique_structures_across_lists=len(selected_sources), sampling_replay="identical",
                  checked_at=datetime.now(timezone.utc).isoformat())
    return report


def run(args):
    started = time.monotonic()
    print(f"Reading validated {args.descriptor_type.upper()} descriptors and structure census...", flush=True)
    vectors, table, census, hashes, settings, subset = load_dataset(
        args.planes, args.min_pair, args.overlap_survey, args.descriptor_type, args.descriptor,
        args.ace_blocks, args.ace_scaling, args.covariant_blocks)
    title = None if subset is None else subset_title(subset)
    if args.descriptor_type == "ace":
        # ACE maps say so on the figure; the SOAP maps keep their established look.
        label = (f"ACE {args.descriptor_tag}" + (f", blocks {'+'.join(args.ace_blocks)}" if args.ace_blocks else "")
                 + (", unscaled blocks" if args.ace_scaling == "none" else ""))
        title = label if title is None else f"{label}; {title}"
    elif args.descriptor_type == "covariant":
        blocks = args.covariant_blocks or list(mc.covariant_module().DEFAULT_BLOCKS)
        label = f"covariant SOAP {args.descriptor_tag}, [edge | mid], blocks {'+'.join(blocks)}"
        title = label if title is None else f"{label}; {title}"
    if subset is not None:
        print(f"Subset: {title}; {subset['families']} catalogue families.", flush=True)
    families = table.iloc[:-1]["family"].to_numpy()
    grain_classes = table.iloc[:-1]["grain_class"].to_numpy()
    arrays, embeddings = {}, {}
    for method in METHODS:
        print(f"Fitting {method.upper()} for {len(families)} distinct GBs and pristine bulk...", flush=True)
        coordinates, details = embed(vectors, method, args.seed)
        selected, labels, centers, parameters = cluster_centers(coordinates[:-1], families, args.seed,
                                                                count=args.n_select)
        arrays.update({method: coordinates,
                       f"{method}_FPS_indices": farthest_points(coordinates[:-1], families, count=args.n_select),
                       f"{method}_median_indices": selected, f"{method}_cluster_labels": labels,
                       f"{method}_cluster_centers": centers})
        details["clustering_parameters"] = parameters
        embeddings[method] = details
        print(f"  {method.upper()} finished in {details['seconds']:.1f} s; both selections contain {args.n_select} GBs.", flush=True)
    for path, expected in hashes.items():
        stat = (ROOT / path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
            raise RuntimeError(f"input changed during analysis: {path}")
    selected_rows = set(int(i) for method in METHODS for sampler in SAMPLERS
                        for i in arrays[f"{method}_{sampler}_indices"])
    selected_hashes = {table.iloc[i]["source"]: fingerprint(ROOT / table.iloc[i]["source"])
                       for i in sorted(selected_rows)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "seed": args.seed, "threads": 1, "n_select": args.n_select, "n_families": len(census),
        "n_distinct_gbs": len(families), "bulk_row": len(families), args.descriptor_type: settings,
        "vector": {"descriptor": args.descriptor_tag, "bulk_descriptor": args.descriptor_tag,
                   "kind": "interface", "radius": 8.0,
                   "planes": "both", "thickness": THICKNESS, "feature_scaling": "none"},
        "deduplication": "structure_class census; alphabetically first family per class",
        "sampling": {"space": "each 2D embedding", "metric": "Euclidean",
                     "FPS": "start farthest from bulk; maximize minimum distance to selected GBs",
                     "median": "actual cluster member nearest the K-means centroid",
                     "ties": "alphabetically first family", "overlap_between_lists": "allowed"},
        "plots": {"directory": sc.relative_path(args.plot_dir), "pixels": [1600, 1200],
                  "dpi": 200, "color_by": "grain_class", "grain_class_colors": GRAIN_COLORS,
                  "grain_class_source": sc.relative_path(GB_LABELS),
                  "selected_background": args.selected_background, "legends": True,
                  "selected_color": RED, "other_color": GRAY,
                  "bulk_marker": "gray diamond", "point_labels": False, "axes": True,
                  "axis_labels": {method: map_axis_labels(method, embeddings[method]) for method in METHODS},
                  "equal_aspect": True, "title": title},
        "embeddings": embeddings, "inputs": hashes, "selected_structure_hashes": selected_hashes,
        "versions": {package: metadata.version(package) for package in (
            "numpy", "scipy", "pandas", "scikit-learn", "umap-learn", "numba", "matplotlib", "pillow")}}
    manifest["versions"]["python"] = sys.version.split()[0]
    if args.descriptor_type == "ace":
        manifest["vector"].update(descriptor_type="ace", ace_blocks=args.ace_blocks or "all stored",
                                  ace_scaling=args.ace_scaling)
    elif args.descriptor_type == "covariant":
        manifest["vector"].update(descriptor_type="covariant", planes="both, joined as [edge | mid]",
                                  covariant_blocks=args.covariant_blocks or list(mc.covariant_module().DEFAULT_BLOCKS))
    if subset is not None:
        manifest["subset"] = subset
    print("Writing six plain text lists and twelve plots...", flush=True)
    outputs = []
    for method in METHODS:
        for sampler in SAMPLERS:
            selected = arrays[f"{method}_{sampler}_indices"]
            sources = sorted(table.iloc[selected]["source"].tolist())
            folder = args.plot_dir / sampler
            text_path = folder / f"{method}.txt"
            write_text(text_path, "".join(source + "\n" for source in sources))
            colored = folder / f"{method}_colored.png"
            if sampler == "FPS":
                draw_map(colored, arrays[method], grain_classes,
                         axis_labels=map_axis_labels(method, embeddings[method]), title=title)
            else:
                tmp = sc._temporary(colored)
                tmp.write_bytes((args.plot_dir / "FPS" / colored.name).read_bytes())
                os.replace(tmp, colored)
            highlighted = folder / f"{method}_selected.png"
            draw_map(highlighted, arrays[method], grain_classes, selected,
                     axis_labels=map_axis_labels(method, embeddings[method]),
                     selected_background=args.selected_background, title=title)
            outputs.extend((text_path, colored, highlighted))
    mapping = args.out_dir / "structures.csv"
    aliases = args.out_dir / "family_mapping.csv"
    npz = args.out_dir / "embeddings.npz"
    union = args.out_dir / "sample_list.csv"
    sc.write_csv(mapping, table.to_dict("records"), list(table.columns))
    sc.write_csv(aliases, census.to_dict("records"), list(census.columns))
    sc.write_npz(npz, **arrays)
    write_text(union, sample_list_text(table.iloc[sorted(selected_rows)]["source"]))
    outputs.extend((mapping, aliases, npz, union))
    manifest["outputs"] = {sc.relative_path(path): fingerprint(path) for path in outputs}
    print("Replaying sampling and checking structure paths, text lists, hashes, and PNGs...", flush=True)
    report = verify_outputs(args.plot_dir, args.out_dir, manifest)
    manifest["seconds"] = round(time.monotonic() - started, 3)
    sc.write_json(args.out_dir / "run.json", manifest)
    sc.write_json(args.out_dir / "validation.json", report)
    print(f"Verified {report['text_files']} lists and {report['png_files']} plots; "
          f"{report['unique_structures_across_lists']} distinct GBs across all six lists.", flush=True)


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def dry_run(args):
    """Count the subset from the census and the survey alone (no descriptors; login-node safe)."""
    import pandas as pd

    census = pd.read_csv(CENSUS, dtype={"plane": str})
    representatives = census.sort_values("family").drop_duplicates("structure_class").reset_index(drop=True)
    survey = None if args.min_pair is None else pd.read_csv(args.overlap_survey, sep="\t", dtype=str)
    mask, subset = overlap_filter(representatives, census, survey, args.planes, args.min_pair)
    kept = representatives[mask]
    print(f"planes {' '.join(subset['planes'])}, closest pair >= {args.min_pair} A: "
          f"{subset['families']} families, {subset['distinct_gbs']} distinct GBs; "
          f"{args.n_select} per list ({'too many' if args.n_select > len(kept) else 'ok'})")
    print("grain classes: " + ", ".join(f"{name} {count}" for name, count in
                                        kept["grain_class"].value_counts().items()))
    print(f"plots   -> {sc.relative_path(args.plot_dir)}\noutputs -> {sc.relative_path(args.out_dir)}")
    found = existing_outputs(args.plot_dir, args.out_dir)
    print("existing selection there: " + (", ".join(sc.relative_path(path) for path in found) or "none"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--descriptor-type", choices=mc.DESCRIPTOR_TYPES, default="soap",
                        help="soap (standard SOAP, default), covariant or ace: which descriptor the maps are built from")
    parser.add_argument("--descriptor", default=None,
                        help=f"descriptor run, tag or folder (default {mc.DEFAULT_DESCRIPTORS['soap']} for soap and "
                             f"covariant, {mc.DEFAULT_DESCRIPTORS['ace']} for ace)")
    parser.add_argument("--ace-blocks", nargs="+", choices=mc.ACE_BLOCKS, default=None,
                        help="ACE only: blocks joined into the vector (default every stored block)")
    parser.add_argument("--ace-scaling", choices=mc.ACE_SCALINGS, default="pristine",
                        help="ACE only: pristine (default) divides each block by its pristine-bulk length; none")
    parser.add_argument("--covariant-blocks", nargs="+", choices=mc.COVARIANT_BLOCKS, default=None,
                        help="covariant only: blocks joined into the vector (default the lambda >= 1 blocks)")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="default plots/CoGe/dim-reduction/SOAP/standard (covariant: .../SOAP/covariant, "
                             "ACE: .../ACE), plus /<tag> for a subset")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default calculations/ML_dscribe/CoGe/standard/dim-reduction (covariant: "
                             ".../CoGe/covariant/dim-reduction, ACE: calculations/ML_ACE/CoGe/dim-reduction), "
                             "plus /<tag> for a subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-background", choices=("gray", "grain_class"), default="gray",
                        help="background colors in selection views (default gray)")
    parser.add_argument("--planes", nargs="+", choices=sc.PLANES, default=list(sc.PLANES),
                        help="select only among GBs of these planes (default all)")
    parser.add_argument("--min-pair", type=float, default=None,
                        help="select only among GBs whose closest pair of atoms at 8x is >= this (A)")
    parser.add_argument("--overlap-survey", type=Path, default=SURVEY,
                        help="closest-pair table for --min-pair (default results/CoGe_interface/overlap_survey.tsv)")
    parser.add_argument("--n-select", type=int, default=N_SELECT, help="GBs per list (default %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing selection in the target folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="count the subset and show the target folders; read no descriptors")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.n_select < 1:
        parser.error("--n-select must be at least 1")
    args.planes = sorted(set(args.planes))
    if args.descriptor_type != "ace" and (args.ace_blocks is not None or args.ace_scaling != "pristine"):
        parser.error("--ace-blocks and --ace-scaling need --descriptor-type ace")
    if args.descriptor_type != "covariant" and args.covariant_blocks is not None:
        parser.error("--covariant-blocks needs --descriptor-type covariant")
    if args.ace_blocks is not None:
        args.ace_blocks = [name for name in mc.ACE_BLOCKS if name in args.ace_blocks]
    if args.covariant_blocks is not None:
        args.covariant_blocks = [name for name in mc.COVARIANT_BLOCKS if name in args.covariant_blocks]
    args.descriptor = args.descriptor or mc.DEFAULT_DESCRIPTORS[args.descriptor_type]
    args.descriptor_tag = mc.descriptor_tag(args.descriptor, args.descriptor_type)
    plot_dir, out_dir = default_folders(args.descriptor_type, args.descriptor_tag, args.ace_blocks, args.ace_scaling,
                                        args.planes, args.min_pair, args.n_select, args.covariant_blocks)
    args.plot_dir = project_path(args.plot_dir) if args.plot_dir else plot_dir
    args.out_dir = project_path(args.out_dir) if args.out_dir else out_dir
    args.overlap_survey = project_path(args.overlap_survey)
    if args.dry_run:
        dry_run(args)
        return
    found = existing_outputs(args.plot_dir, args.out_dir)
    if found and not args.verify_only and not args.overwrite:
        sys.exit("STOP: a selection already exists in the target folders (--overwrite replaces it): "
                 + ", ".join(sc.relative_path(path) for path in found))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.verify_only:
            manifest = json.loads((args.out_dir / "run.json").read_text())
            report = verify_outputs(args.plot_dir, args.out_dir, manifest)
            sc.write_json(args.out_dir / "validation.json", report)
            print(json.dumps(report, indent=2))
        else:
            run(args)


if __name__ == "__main__":
    main()
