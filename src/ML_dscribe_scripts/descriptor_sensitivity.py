#!/usr/bin/env python3
"""Descriptor study, part C: which SOAP settings change which GBs look alike?

Every setting is compared with a baseline on the 1,112 distinct GBs at 8x (the
pool select_gbs.py picks from). The baseline is the reference vector

  SOAP r_cut 6, n_max 8, l_max 6, sigma 0.5, no weighting; the mean of the Co-centred
  and of the Ge-centred atoms within 8 A of both interface planes; no scaling,

except for weighting runs at r_cut 8 and 12: they are compared with the unweighted
run at the same r_cut, so the weighting effect is not mixed up with the r_cut
effect (their scores against the reference are in the table too).

Scores of a setting against its baseline:
  spearman          rank correlation of all pairwise distances (1 = same ordering)
  knn10_kept        share of each GB's 10 nearest neighbours that stay among its 10 nearest
  gfre_from_base    feature-space reconstruction error (Goscinski et al. 2021, skmatter),
  gfre_to_base      baseline -> setting and setting -> baseline; 0 = one predicts the other
                    fully, 1 = not at all
  procrustes_pca    Procrustes disparity of the two 2D PCA maps (0 = the same map after
                    shifting, turning and scaling)
  fps_overlap       how many of the 30 FPS picks agree (full space, first pick farthest from bulk)
Also per setting: share of the variance in PC1 and PC2, PCs needed for 90% and 99%,
vector length and the CPU hours of its SOAP run.

Two more tests:
  thickness   all 13,868 structures at the reference: how much of PC1, PC2 and of the
              whole spread follows the thickness (and how much the GB), for the global
              and for the interface vector
  seeds       t-SNE and UMAP with 10 seeds at the reference (seed 42 is select_gbs.py's);
              then one map per setting, to see whether a setting moves the map more than
              a new seed does

Outputs in calculations/ML_dscribe/CoGe/descriptor_sensitivity/:
  sensitivity.csv      one row per setting
  knob_ranking.csv     per knob, the largest change over its values
  seed_robustness.csv  seed-to-seed pairs, and each setting's map against the reference map
  thickness_test.csv   thickness and GB shares per vector and axis
  thickness_pcs.npz    PC1 and PC2 of every structure, global and interface vector (for the figures)
  maps.npz             PCA maps (turned onto the reference), the t-SNE/UMAP maps and their 30 FPS and
                       k-means picks
  summary.json         settings, sanity checks, timing
  sensitivity.png, seed_robustness.png, pca_maps.png

Needs every SOAP run of soap_runs() (descriptor_grid.py makes them). --only runs a
subset, e.g. --only reference "r_cut 8" "region global".
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import ml_common as mc
from ml_common import sc

OUT_DIR = mc.OUTPUT_ROOT / "descriptor_sensitivity"
REFERENCE = {"r_cut": 6.0, "n_max": 8, "l_max": 6, "sigma": 0.5, "weighting": None}
N_PICKS = 30
N_NEIGHBOURS = 10
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
METHODS = ("tsne", "umap")
SCORES = ("spearman", "knn10_kept", "gfre_from_base", "gfre_to_base", "procrustes_pca", "fps_overlap")
KNOBS = ("r_cut", "n_max", "l_max", "sigma", "weighting", "region", "processing")


# --------------------------------------------------------------------------- settings

def pow_weighting(r_cut):
    """Willatt et al. 2018: w(r) = c / (d + (r / r0)^m)."""
    return {"function": "pow", "r0": 3.0, "c": 1.0, "d": 1.0, "m": 4.0}


def poly_weighting(r_cut):
    """Caro 2019: w(r) = c (1 + 2 (r / r0)^3 - 3 (r / r0)^2)^m, falling to 0 at r0 = r_cut."""
    return {"function": "poly", "r0": float(r_cut), "c": 1.0, "m": 2.0}


def soap_runs():
    """(knob, value, SOAP settings) of every SOAP run of the study; the reference first.

    Each run changes one setting away from the reference. Weighting is tried at
    r_cut 6, 8 and 12, because it matters more the larger the cutoff.
    """
    runs = [("reference", "reference", dict(REFERENCE))]
    runs += [("r_cut", f"{v:g}", {**REFERENCE, "r_cut": float(v)}) for v in (4, 5, 8, 10, 12)]
    runs += [("n_max", str(v), {**REFERENCE, "n_max": v}) for v in (4, 6, 10, 12)]
    runs += [("l_max", str(v), {**REFERENCE, "l_max": v}) for v in (2, 4, 8, 10)]
    runs += [("sigma", f"{v:g}", {**REFERENCE, "sigma": v}) for v in (0.3, 0.75, 1.0)]
    for r_cut in (6, 8, 12):
        for name, make in (("pow", pow_weighting), ("poly", poly_weighting)):
            runs.append(("weighting", f"{name} rc{r_cut}",
                         {**REFERENCE, "r_cut": float(r_cut), "weighting": make(r_cut)}))
    return runs


def soap_tag(soap):
    return sc.soap_tag({"rbf": "gto", **soap})


@dataclass
class Variant:
    """One GB vector: a SOAP run plus the region and processing that turn it into one vector per GB."""
    name: str
    knob: str
    value: str
    soap: dict = field(default_factory=lambda: dict(REFERENCE))
    kind: str = "interface"
    radius: float = 8.0
    planes: str = "both"
    mean: str = "species"        # species: [Co mean | Ge mean]; all: one mean over all atoms
    scale: str = "none"          # none, standard (each feature to mean 0, spread 1), unit (each vector to length 1)
    baseline: str = "reference"

    @property
    def tag(self):
        return soap_tag(self.soap)


def variants(only=None):
    """Every vector of the study, the reference first. only: names to keep."""
    out = []
    for knob, value, soap in soap_runs():
        name = "reference" if knob == "reference" else f"{knob} {value}"
        baseline = "reference"
        if knob == "weighting" and soap["r_cut"] != REFERENCE["r_cut"]:
            baseline = f"r_cut {soap['r_cut']:g}"
        out.append(Variant(name, knob, value, soap, baseline=baseline))
    out += [Variant("region interface r6", "region", "interface r6", radius=6.0),
            Variant("region interface r10", "region", "interface r10", radius=10.0),
            Variant("region edge plane only", "region", "edge plane only", planes="edge"),
            Variant("region mid plane only", "region", "mid plane only", planes="mid"),
            Variant("region nonbulk atoms", "region", "nonbulk atoms", kind="nonbulk"),
            Variant("region global", "region", "global", kind="global"),
            Variant("processing one mean", "processing", "one mean over all atoms", mean="all"),
            Variant("processing standardised", "processing", "standardised", scale="standard"),
            Variant("processing unit length", "processing", "unit length", scale="unit")]
    if only:
        unknown = set(only) - {v.name for v in out}
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}; known: {[v.name for v in out]}")
        out = [v for v in out if v.name in only or v.name == "reference"]
    names = {v.name for v in out}
    for v in out:
        if v.baseline not in names:
            v.baseline = "reference"
    return out


# --------------------------------------------------------------------------- vectors

def species_counts(tag, radius, families, thickness=8):
    """Co and Ge atom counts within `radius` of both planes, one row per family."""
    import pandas as pd

    folder = sc.run_dir("both_interfaces", tag)
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    position = {(f, t): i for i, (f, t) in enumerate(zip(index["family"], index["thickness"]))}
    with np.load(folder / sc.INTERFACE_FILE) as data:
        n = data[f"n_{sc.radius_key(radius)}"]
    return n[[position[(f, thickness)] for f in families]].sum(axis=1).astype(np.float64)


def process(X, bulk, scale):
    """Scale the GB vectors (fitted on the defined GB rows) and the bulk vector alike."""
    if scale == "none":
        return X, bulk
    if scale == "standard":
        defined = X[np.isfinite(X).all(axis=1)]
        mean, spread = defined.mean(axis=0), defined.std(axis=0)
        spread = np.where(spread > 0, spread, 1.0)
        return (X - mean) / spread, (bulk - mean) / spread
    if scale == "unit":
        return X / np.linalg.norm(X, axis=1, keepdims=True), bulk / np.linalg.norm(bulk)
    raise ValueError(f"unknown scale {scale!r}")


def load_variant(variant):
    """(X, table, bulk) for the 1,112 distinct GBs at 8x; undefined vectors are rows of NaN."""
    X, table, bulk = mc.load_distinct(variant.tag, variant.kind, variant.radius, variant.planes,
                                      allow_undefined=True)
    if variant.mean == "all":
        if variant.kind != "interface":
            raise ValueError("one mean over all atoms is only set up for the interface vector")
        counts = species_counts(variant.tag, variant.radius, table["family"])
        half = X.shape[1] // 2
        X = (counts[:, :1] * X[:, :half] + counts[:, 1:] * X[:, half:]) / counts.sum(axis=1, keepdims=True)
        bulk = 0.5 * (bulk[:half] + bulk[half:])  # B20 CoGe has as many Co as Ge atoms
    elif variant.mean != "species":
        raise ValueError(f"unknown mean {variant.mean!r}")
    X, bulk = process(X, bulk, variant.scale)
    return X, table, bulk


def soap_cpu_hours(tag):
    """CPU hours of the GB SOAP run (wall time x workers), from its settings.json."""
    path = sc.run_dir("both_interfaces", tag) / "settings.json"
    settings = json.loads(path.read_text())
    if settings.get("limit"):
        return None
    return settings["seconds"] * settings["workers"] / 3600.0


# --------------------------------------------------------------------------- scores

def square_distances(X):
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(X))


def nearest(D, k):
    """The k nearest other rows of every row, nearest first (ties: lower row first)."""
    masked = D.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argsort(masked, axis=1, kind="stable")[:, :k]


def neighbours_kept(D_a, D_b, k=N_NEIGHBOURS):
    """Mean share of each row's k nearest neighbours under D_a that are also among its k nearest under D_b."""
    a, b = nearest(D_a, k), nearest(D_b, k)
    return float(np.mean([len(set(x) & set(y)) for x, y in zip(a.tolist(), b.tolist())]) / k)


def distance_correlation(D_a, D_b):
    """Spearman correlation of all pairwise distances."""
    from scipy.stats import spearmanr

    upper = np.triu_indices(len(D_a), 1)
    return float(spearmanr(D_a[upper], D_b[upper]).statistic)


def reconstruction_error(X, Y, seed=0):
    """GFRE of Y from X: linear ridge regression on half the rows, error on the other half."""
    from skmatter.metrics import global_reconstruction_error

    order = np.random.default_rng(seed).permutation(len(X))
    half = len(X) // 2
    return float(global_reconstruction_error(X, Y, test_idx=order[half:], train_idx=order[:half]))


def pca_map(X):
    from sklearn.decomposition import PCA

    return PCA(n_components=2, svd_solver="full").fit_transform(X)


def aligned(reference_map, other_map):
    """Procrustes: (disparity, other map shifted, turned and scaled onto the reference map).

    Both maps are centred and scaled to unit size first, as scipy.spatial.procrustes does.
    """
    from scipy.spatial import procrustes

    first, second, disparity = procrustes(reference_map, other_map)
    return float(disparity), first, second


def variance_profile(X):
    from sklearn.decomposition import PCA

    ratio = PCA(svd_solver="full").fit(X).explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    return {"pc1": float(ratio[0]), "pc2": float(ratio[1]),
            "n_pc90": int(np.searchsorted(cumulative, 0.90) + 1),
            "n_pc99": int(np.searchsorted(cumulative, 0.99) + 1)}


def fps_picks(X, bulk, count=N_PICKS):
    return mc.farthest_point_sampling(X, count, mc.farthest_from(X, bulk))


def compare(A, bulk_A, B, bulk_B, families):
    """Scores of vectors B against vectors A (same GBs, same row order).

    Rows undefined in either are left out. Returns (scores, B's PCA map turned
    onto A's, rows used).
    """
    keep = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    A, B, families = A[keep], B[keep], np.asarray(families)[keep]
    D_a, D_b = square_distances(A), square_distances(B)
    disparity, _, turned = aligned(pca_map(A), pca_map(B))
    picks_a, picks_b = families[fps_picks(A, bulk_A)], families[fps_picks(B, bulk_B)]
    scores = {"n_gbs": int(keep.sum()),
              "spearman": distance_correlation(D_a, D_b),
              "knn10_kept": neighbours_kept(D_a, D_b),
              "gfre_from_base": reconstruction_error(A, B),
              "gfre_to_base": reconstruction_error(B, A),
              "procrustes_pca": disparity,
              "fps_overlap": len(set(picks_a) & set(picks_b))}
    return scores, turned, keep


# --------------------------------------------------------------------------- tasks (run in worker processes)

def variant_task(variant, baseline, reference):
    """One row of sensitivity.csv and the setting's PCA map turned onto the reference map."""
    started = time.time()
    X, table, bulk = load_variant(variant)
    defined = np.isfinite(X).all(axis=1)
    families = table["family"].to_numpy()
    row = {"name": variant.name, "knob": variant.knob, "value": variant.value, "baseline": baseline.name,
           "tag": variant.tag, "kind": variant.kind, "radius": variant.radius if variant.kind == "interface" else None,
           "planes": variant.planes, "mean": variant.mean, "scale": variant.scale,
           "vector_length": int(X.shape[1]), "n_undefined": int((~defined).sum()),
           "soap_cpu_hours": soap_cpu_hours(variant.tag)}
    row.update(variance_profile(X[defined]))
    loaded = {variant.name: (X, bulk)}
    for other in (baseline, reference):
        if other.name not in loaded:
            X_other, _, bulk_other = load_variant(other)
            loaded[other.name] = (X_other, bulk_other)
    scores, _, _ = compare(*loaded[baseline.name], X, bulk, families)
    row.update(scores)
    scores, turned, keep = compare(*loaded[reference.name], X, bulk, families)
    row.update({f"{key}_vs_reference": value for key, value in scores.items() if key != "n_gbs"})
    full_map = np.full((len(X), 2), np.nan)
    full_map[keep] = turned
    row["seconds"] = round(time.time() - started, 1)
    return row, full_map


def embedding_task(variant, method, seed):
    """One t-SNE or UMAP map (GBs + pristine bulk, as select_gbs.py) and what it keeps of the full space."""
    import select_gbs
    from sklearn.manifold import trustworthiness

    started = time.time()
    X, table, bulk = load_variant(variant)
    keep = np.isfinite(X).all(axis=1)
    families = table["family"].to_numpy()[keep]
    coordinates, _ = select_gbs.embed(np.vstack([X[keep], bulk]), method, seed)
    points = coordinates[:-1]
    full_map = np.full((len(X) + 1, 2), np.nan)
    full_map[np.append(keep, True)] = coordinates
    median, _, _, _ = select_gbs.cluster_centers(points, families, seed)
    return {"name": variant.name, "method": method, "seed": seed, "map": full_map,
            "trustworthiness": float(trustworthiness(X[keep], points, n_neighbors=N_NEIGHBOURS)),
            "knn10_kept": neighbours_kept(square_distances(X[keep]), square_distances(points)),
            "fps": families[select_gbs.farthest_points(points, families)].tolist(),
            "median": families[median].tolist(), "seconds": round(time.time() - started, 1)}


def thickness_share(values, labels):
    """Share of the spread of values (rows) that lies between the groups given by labels."""
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    centre = values.mean(axis=0)
    total = np.square(values - centre).sum()
    between = 0.0
    for label in np.unique(labels):
        members = labels == label
        between += members.sum() * np.square(values[members].mean(axis=0) - centre).sum()
    return float(between / total)


def thickness_task(tag):
    """Thickness test on every structure: does PC1 follow the thickness or the GB?"""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    started = time.time()
    rows, projections = [], {}
    for vector, kind in (("global", "global"), ("interface r8 both", "interface")):
        X, index = mc.load_all_vectors(tag, kind, 8.0, "both")
        X = np.asarray(X, dtype=np.float64)
        subsets = [("all structures", np.isfinite(X).all(axis=1))]
        if kind == "interface":
            subsets.append(("radius within a quarter cell", subsets[0][1] & (index["Lx"].to_numpy() / 4.0 >= 8.0)))
        for subset, mask in subsets:
            Z = X[mask]
            thickness = index["thickness"].to_numpy()[mask]
            family = index["family"].to_numpy()[mask]
            pca = PCA(n_components=2, svd_solver="full").fit(Z)
            scores = pca.transform(Z)
            if subset == "all structures":
                key = kind
                projections[f"{key}_pcs"] = np.full((len(X), 2), np.nan)
                projections[f"{key}_pcs"][mask] = scores
                projections[f"{key}_share"] = pca.explained_variance_ratio_[:2].copy()
                projections.setdefault("thickness", index["thickness"].to_numpy())
                projections.setdefault("family", index["family"].to_numpy().astype(str))
                projections.setdefault("plane", index["plane"].to_numpy().astype(str))
            for axis, values, share in (("PC1", scores[:, 0], pca.explained_variance_ratio_[0]),
                                        ("PC2", scores[:, 1], pca.explained_variance_ratio_[1]),
                                        ("all features", Z, 1.0)):
                rows.append({"vector": vector, "subset": subset, "n_structures": int(mask.sum()),
                             "n_thicknesses": int(len(np.unique(thickness))), "axis": axis,
                             "axis_share_of_variance": float(share),
                             "thickness_share": thickness_share(values, thickness),
                             "gb_share": thickness_share(values, family),
                             "spearman_with_thickness": (float(spearmanr(values, thickness).statistic)
                                                         if values.ndim == 1 else None)})
    return rows, projections, round(time.time() - started, 1)


# --------------------------------------------------------------------------- tables

def knob_ranking(table):
    """Per knob, the largest change over its values (scores against each setting's baseline)."""
    import pandas as pd

    rows = []
    for knob in KNOBS:
        part = table[table["knob"] == knob]
        if part.empty:
            continue
        rows.append({"knob": knob, "n_settings": len(part), "values": "; ".join(part["value"]),
                     "min_spearman": part["spearman"].min(), "min_knn10_kept": part["knn10_kept"].min(),
                     "max_gfre": part[["gfre_from_base", "gfre_to_base"]].to_numpy().max(),
                     "max_procrustes_pca": part["procrustes_pca"].max(),
                     "min_fps_overlap": int(part["fps_overlap"].min())})
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    # Rank 1 = the knob that changes the picture most, for each score; then the mean rank.
    ranks = pd.DataFrame({
        "spearman": ranking["min_spearman"].rank(), "knn": ranking["min_knn10_kept"].rank(),
        "gfre": ranking["max_gfre"].rank(ascending=False),
        "procrustes": ranking["max_procrustes_pca"].rank(ascending=False),
        "fps": ranking["min_fps_overlap"].rank()})
    ranking["mean_rank"] = ranks.mean(axis=1)
    return ranking.sort_values("mean_rank").reset_index(drop=True)


def overlap(a, b):
    return len(set(a) & set(b))


def seed_table(embeddings, families):
    """Seed-to-seed pairs at the reference, and each setting's map against the reference (seed 42)."""
    import pandas as pd

    rows = []
    for method in METHODS:
        runs = {e["seed"]: e for e in embeddings if e["method"] == method and e["name"] == "reference"}
        seeds = sorted(runs)
        for i, s in enumerate(seeds):
            for t in seeds[i + 1:]:
                a, b = runs[s]["map"][:-1], runs[t]["map"][:-1]
                rows.append({"method": method, "comparison": "seed pair", "name": "reference",
                             "seed_a": s, "seed_b": t, "procrustes": aligned(a, b)[0],
                             "fps_overlap": overlap(runs[s]["fps"], runs[t]["fps"]),
                             "median_overlap": overlap(runs[s]["median"], runs[t]["median"]),
                             "trustworthiness": runs[t]["trustworthiness"], "knn10_kept": runs[t]["knn10_kept"]})
        base = runs.get(SEEDS[0])
        for e in embeddings:
            if e["method"] != method or e["name"] == "reference" or base is None:
                continue
            a, b = base["map"][:-1], e["map"][:-1]
            keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            rows.append({"method": method, "comparison": "setting vs reference", "name": e["name"],
                         "seed_a": SEEDS[0], "seed_b": e["seed"], "procrustes": aligned(a[keep], b[keep])[0],
                         "fps_overlap": overlap(base["fps"], e["fps"]),
                         "median_overlap": overlap(base["median"], e["median"]),
                         "trustworthiness": e["trustworthiness"], "knn10_kept": e["knn10_kept"]})
        for e in runs.values():
            rows.append({"method": method, "comparison": "single map", "name": "reference", "seed_a": e["seed"],
                         "seed_b": e["seed"], "procrustes": None, "fps_overlap": None, "median_overlap": None,
                         "trustworthiness": e["trustworthiness"], "knn10_kept": e["knn10_kept"]})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Is a setting's change larger than the largest change a new seed makes?
    for method in METHODS:
        pairs = table[(table["method"] == method) & (table["comparison"] == "seed pair")]
        settings = (table["method"] == method) & (table["comparison"] == "setting vs reference")
        if pairs.empty:
            continue
        table.loc[settings, "beyond_seed_procrustes"] = table.loc[settings, "procrustes"] > pairs["procrustes"].max()
        table.loc[settings, "beyond_seed_fps"] = table.loc[settings, "fps_overlap"] < pairs["fps_overlap"].min()
    return table


# --------------------------------------------------------------------------- figures

INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"
BAR = "#2a78d6"
BAND = "#C9C9C9"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, path):
    tmp = sc._temporary(path)
    fig.savefig(tmp, dpi=200, facecolor="white")
    os.replace(tmp, path)


def draw_sensitivity(table):
    """One panel per score; one bar per setting (against its baseline), grouped by knob. Returns (plt, fig)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = table[table["name"] != "reference"]
    reference = table[table["name"] == "reference"].iloc[0]
    labels = [f"{k}: {v}" for k, v in zip(part["knob"], part["value"])]
    y = np.arange(len(part))[::-1]
    # The reconstruction error is not 0 for identical vectors: the test half holds detail that the
    # training half does not span. The reference against itself gives that floor.
    panels = [("spearman", "Distance rank correlation", (0, 1), 1.0),
              ("knn10_kept", "10 nearest neighbours kept", (0, 1), 1.0),
              ("gfre_from_base", "GFRE baseline -> setting", (0, None), reference["gfre_from_base"]),
              ("gfre_to_base", "GFRE setting -> baseline", (0, None), reference["gfre_to_base"]),
              ("procrustes_pca", "Procrustes disparity, PCA map", (0, None), 0.0),
              ("fps_overlap", f"FPS picks shared (of {N_PICKS})", (0, N_PICKS), N_PICKS)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 2.5, 0.22 * len(part) + 1.6),
                             sharey=True, dpi=200, facecolor="white")
    for ax, (column, title, limits, same) in zip(axes, panels):
        ax.barh(y, part[column], height=0.7, color=BAR, linewidth=0)
        ax.axvline(same, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlim(left=limits[0], right=limits[1])
        _style(ax)
        # Thin separators between knobs.
        knobs = part["knob"].to_numpy()
        for i in range(1, len(knobs)):
            if knobs[i] != knobs[i - 1]:
                ax.axhline(y[i] + 0.5, color=GRID, linewidth=0.8)
    axes[0].set_yticks(y, labels, fontsize=8)
    return plt, fig


def plot_sensitivity(path, table):
    plt, fig = draw_sensitivity(table)
    fig.suptitle("Change against the baseline (dashed line = no change: the reference against itself); "
                 "weighting at r_cut 8/12 against the unweighted run at that r_cut", fontsize=9, color=MUTED, y=0.995)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_seeds(seeds):
    """Seed-to-seed spread (gray band) against each setting's change, for t-SNE and UMAP.

    Returns (plt, fig) with the legend in the bottom 8% of the figure, or None without settings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = seeds[seeds["comparison"] == "setting vs reference"]
    names = list(dict.fromkeys(settings["name"]))
    if not names:
        return None
    y = np.arange(len(names))[::-1]
    columns = [("procrustes", "Procrustes disparity to the reference map"),
               ("fps_overlap", f"FPS picks shared with the reference map (of {N_PICKS})")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.22 * len(names) + 2.2), sharey=True, dpi=200, facecolor="white")
    for col, method in enumerate(METHODS):
        pairs = seeds[(seeds["method"] == method) & (seeds["comparison"] == "seed pair")]
        mine = settings[settings["method"] == method].set_index("name").reindex(names)
        for k, (column, title) in enumerate(columns):
            ax = axes[2 * col + k]
            if not pairs.empty:
                low, high = pairs[column].min(), pairs[column].max()
                ax.axvspan(low, high, color=BAND, alpha=0.6, linewidth=0, label="range between seeds")
                ax.axvline(pairs[column].median(), color=MUTED, linewidth=0.8, label="median between seeds")
            ax.scatter(mine[column], y, s=18, color=BAR, linewidths=0, zorder=3, label="setting vs reference")
            ax.set_title(f"{'t-SNE' if method == 'tsne' else 'UMAP'}\n{title}", fontsize=8.5, color=INK)
            _style(ax)
            if column == "fps_overlap":
                ax.set_xlim(0, N_PICKS)
    axes[0].set_yticks(y, names, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    return plt, fig


def plot_seeds(path, seeds):
    drawn = draw_seeds(seeds)
    if drawn is None:
        return
    plt, fig = drawn
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)
    plt.close(fig)


def plot_pca_maps(path, names, maps, grain_classes):
    """One small PCA map per setting, turned onto the reference map, coloured by grain class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from select_gbs import GRAIN_COLORS

    columns = 6
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.6 * rows + 0.6), dpi=200, facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    finite = np.vstack([m[np.isfinite(m).all(axis=1)] for m in maps])
    low, high = finite.min(axis=0), finite.max(axis=0)
    for ax, name, coordinates in zip(axes, names, maps):
        for grain_class, color in GRAIN_COLORS.items():
            members = grain_classes == grain_class
            ax.scatter(coordinates[members, 0], coordinates[members, 1], s=2, color=color, linewidths=0)
        ax.set_title(name, fontsize=7.5, color=INK)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color(GRID)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=5, label=g)
               for g, c in GRAIN_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False, title="Grain class",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- main

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def worker_pool(jobs):
    """Spawned worker processes with one BLAS/OpenMP/numba thread each, so `jobs` workers use `jobs` CPUs.

    env.sh sets these variables to the whole allocation; inherited unchanged, every worker would
    start that many threads. The workers read them when they import numpy, after the spawn."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    return ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=multiprocessing.get_context("spawn"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="+", help="setting names to run (the reference is always included)")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help=f"seeds of the seed test (default {len(SEEDS)})")
    parser.add_argument("--no-embeddings", action="store_true", help="skip the seed test and the per-setting maps")
    parser.add_argument("--no-thickness", action="store_true", help="skip the thickness test")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="worker processes, one thread each (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--dry-run", action="store_true", help="list the settings and their folders, then stop")
    args = parser.parse_args()

    chosen = variants(args.only)
    by_name = {v.name: v for v in chosen}
    if args.dry_run:
        for v in chosen:
            folder = sc.run_dir("global" if v.kind == "global" else "both_interfaces", v.tag)
            print(f"{v.name:28s} {v.tag:40s} baseline={v.baseline:12s} {'ok' if folder.is_dir() else 'MISSING'}")
        return 0
    missing = sorted({v.tag for v in chosen if not (sc.run_dir("both_interfaces", v.tag) / sc.INTERFACE_FILE).is_file()})
    if missing:
        raise SystemExit(f"SOAP runs missing: {missing}; run descriptor_grid.py first")

    started = time.time()
    seeds = SEEDS[:max(1, args.seeds)]
    reference = by_name["reference"]
    with worker_pool(args.jobs) as pool:
        thickness = None if args.no_thickness else pool.submit(thickness_task, reference.tag)
        embedding_jobs = []
        if not args.no_embeddings:
            embedding_jobs = [pool.submit(embedding_task, reference, m, s) for m in METHODS for s in seeds]
            embedding_jobs += [pool.submit(embedding_task, v, m, seeds[0]) for v in chosen if v.name != "reference"
                               for m in METHODS]
        variant_jobs = [pool.submit(variant_task, v, by_name[v.baseline], reference) for v in chosen]
        results = [job.result() for job in variant_jobs]
        print(f"[{time.time() - started:6.0f} s] {len(results)} settings compared", flush=True)
        embeddings = [job.result() for job in embedding_jobs]
        print(f"[{time.time() - started:6.0f} s] {len(embeddings)} t-SNE/UMAP maps", flush=True)
        thickness_rows, thickness_pcs, thickness_seconds = thickness.result() if thickness else ([], {}, 0)

    import pandas as pd

    _, table_ref, _ = mc.load_distinct(reference.tag, allow_undefined=True)
    families = table_ref["family"].to_numpy()
    grain_classes = table_ref["grain_class"].to_numpy()
    table = pd.DataFrame([row for row, _ in results])
    pca_maps = [m for _, m in results]
    ranking = knob_ranking(table)
    seed_rows = seed_table(embeddings, families)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    mc.write_table(out / "sensitivity.csv", table)
    mc.write_table(out / "knob_ranking.csv", ranking)
    if not seed_rows.empty:
        mc.write_table(out / "seed_robustness.csv", seed_rows)
    if thickness_rows:
        mc.write_table(out / "thickness_test.csv", pd.DataFrame(thickness_rows))
        tmp = sc._temporary(out / "thickness_pcs.npz")
        np.savez(tmp, **thickness_pcs)
        os.replace(tmp, out / "thickness_pcs.npz")
    # Plain strings, so the file loads without pickle.
    arrays = {"families": families.astype(str), "grain_class": grain_classes.astype(str),
              "names": np.array(table["name"], dtype=str),
              "pca_maps": np.stack(pca_maps)}
    for e in embeddings:
        key = f"{e['method']}__{e['name']}__seed{e['seed']}"
        arrays[key] = e["map"]
        arrays[f"{key}__fps"] = np.array(e["fps"], dtype=str)        # the 30 FPS picks on that map
        arrays[f"{key}__median"] = np.array(e["median"], dtype=str)  # the 30 k-means members
    tmp = sc._temporary(out / "maps.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out / "maps.npz")

    plot_sensitivity(out / "sensitivity.png", table)
    plot_pca_maps(out / "pca_maps.png", table["name"].tolist(), pca_maps, grain_classes)
    if not seed_rows.empty:
        plot_seeds(out / "seed_robustness.png", seed_rows)

    self_row = table[table["name"] == "reference"].iloc[0]
    summary = {
        "pool": "1,112 distinct GBs at 8x (one family per structure class, as select_gbs.py)",
        "reference": asdict(reference) | {"tag": reference.tag},
        "settings": [asdict(v) | {"tag": v.tag} for v in chosen],
        "n_picks": N_PICKS, "n_neighbours": N_NEIGHBOURS, "seeds": list(seeds),
        "sanity_reference_against_itself": {key: self_row[key] for key in SCORES},
        "knob_order_most_change_first": ranking["knob"].tolist() if not ranking.empty else [],
        "seconds": {"total": round(time.time() - started, 1), "thickness_test": thickness_seconds,
                    "settings": {row["name"]: row["seconds"] for row, _ in results},
                    "embeddings": round(sum(e["seconds"] for e in embeddings), 1)},
        "run": mc.run_metadata()}
    if not seed_rows.empty:
        summary["seed_test"] = {
            method: {"procrustes_between_seeds_max": float(part["procrustes"].max()),
                     "fps_overlap_between_seeds_min": int(part["fps_overlap"].min()),
                     "median_overlap_between_seeds_min": int(part["median_overlap"].min())}
            for method, part in seed_rows[seed_rows["comparison"] == "seed pair"].groupby("method")}
    mc.write_json(out / "summary.json", summary)

    print(f"reference against itself: " + ", ".join(f"{k} {self_row[k]:.3g}" for k in SCORES))
    if not ranking.empty:
        print("knobs, most change first: " + ", ".join(ranking["knob"]))
    print(f"done in {time.time() - started:.0f} s; outputs in {sc.relative_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
