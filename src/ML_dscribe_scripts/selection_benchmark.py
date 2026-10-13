#!/usr/bin/env python3
"""Descriptor study, part D: which way of choosing GBs teaches a model most from the fewest?

Pool: the distinct GBs at 8x (1,111: the 1,112 of Stage 1.5 without D0985, the two plain-bulk
NO_BOUNDARY families of ml_common.py; --keep-no-boundary keeps them). Ten times, 222 GBs are
set aside as a test set. A strategy picks n GBs (n = 5 ... 100) from the other GBs without
seeing any label, a model learns the labels of the picks, and is scored on the test set.

Strategies
  random        20 draws
  stratified    plane x grain-class groups; picks shared out in proportion to group size,
                random inside each group (20 draws)
  fps           farthest-point sampling in the full vector space, first pick farthest from bulk
  kmeans        the member nearest each k-means centre in the full space (select_gbs.py's
                "median" rule, in the full space instead of a 2D map)
  gp_variance   greedy largest predictive variance of a Gaussian process with the grain-class
                kernel: the label-free first step of Bayesian active learning
  map_fps_<m>, map_kmeans_<m>
                select_gbs.py's six rules on its PCA, t-SNE and UMAP maps (m); the maps are
                fitted once on all GBs + bulk with seed 42, as select_gbs.py does

Models (the same for every strategy)
  krr   kernel ridge regression with the most common Stage 1 settings of each label:
        grain class: standardised features, gamma 10, alpha 1e-4; plane: standardised,
        gamma 0.1, alpha 0.01; handedness: no scaling, gamma 10, alpha 1e-4.
        The feature scaling and the kernel width (median squared distance) come from the
        pool GBs, which needs no labels, so they do not change with n.
        Geometry labels: scaling, gamma and alpha chosen on the picks alone by the
        closed-form leave-one-out error of kernel ridge regression (hat-matrix shortcut),
        from standard/none x gamma 0.1, 1, 10, 100 x alpha 1e-6, 1e-4, 1e-2, 1.
  1nn   label (or value) of the nearest pick (Euclidean, unscaled); no settings
Labels: grain class (main), plane, handedness (relative to grain A; it stays near chance).
Grain class is also scored on each plane's test GBs alone (rows grain_class@100, @110,
@111), since a 100/110 GB has 4 possible classes and a 111 GB 8.
Geometry labels (geometry_labels.py, region "cell"): d_min, log1p_n_close, cn_excess,
cn_defect, which depend on the shift; scored by R^2 on the test set (columns r2, mae).
--no-geometry leaves them out.

Label-free scores of every set of picks, in the unscaled reference space, over all
GBs of the pool: coverage radius (largest distance from a GB to its nearest pick) and mean
distance to the nearest pick, both divided by the median distance between GBs;
plane x grain-class groups covered.

Existing lists: the same scores, and the accuracy on all other GBs, for the six lists of
30 (plots/CoGe/dim-reduction/{FPS,median}/{pca,tsne,umap}.txt), their union, and 30
picks of every strategy taken from all GBs.

Descriptor comparison: fps and random curves for every setting of
descriptor_sensitivity.py. The KRR settings were tuned on the reference, so the 1-NN
curves are the fairer comparison between descriptors.

Outputs in calculations/ML_dscribe/CoGe/selection_benchmark_v2/ (refused if summary.json is
there, unless --overwrite; v1 with 1,112 GBs and no geometry labels is selection_benchmark/):
  learning_curves.csv   descriptor x split x strategy x draw x n x label x model: accuracy
                        (class labels) or r2 and mae (geometry labels)
  coverage.csv          split x strategy x draw x n: label-free scores (reference vector)
  summary.csv           mean and spread over splits (draws averaged first); mean over all n
  existing_lists.csv    the six lists, their union, and 30 picks per strategy from all GBs
  picks.csv             the GBs behind every row of existing_lists.csv (list, draw, rank, family)
  maps.npz              select_gbs.py's PCA, t-SNE and UMAP maps of the pool GBs (last row: bulk),
                        with their families and labels
  summary.json, learning_curves.png, coverage.png, descriptor_efficiency.png
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import descriptor_sensitivity as ds
import geometry_labels as gl
import ml_common as mc
from ml_common import sc

OUT_DIR = mc.OUTPUT_ROOT / "selection_benchmark_v2"
EXISTING_DIR = mc.PROJECT_ROOT / "plots" / "CoGe" / "dim-reduction"
SIZES = (5, 10, 15, 20, 30, 40, 60, 80, 100)
N_SPLITS = 10
N_TEST = 222
N_DRAWS = 20
MAP_SEED = 42
LABELS = ("grain_class", "plane", "handedness")
# Most common choice of the Stage 1 nested cross-validation (krr_classify.py), per label.
KRR_SETTINGS = {"grain_class": ("standard", 10.0, 1e-4), "plane": ("standard", 0.1, 1e-2),
                "handedness": ("none", 10.0, 1e-4)}
# Grain class scored on each plane's test GBs alone.
PLANE_LABELS = tuple(f"grain_class@{plane}" for plane in sc.PLANES)
GEOMETRY = gl.TARGETS
REGRESSION_SCALES, REGRESSION_GAMMAS, REGRESSION_ALPHAS = ("standard", "none"), (0.1, 1.0, 10.0, 100.0), (1e-6, 1e-4, 1e-2, 1.0)
FULL_SPACE = ("random", "stratified", "fps", "kmeans", "gp_variance")
MAP_RULES = tuple(f"map_{rule}_{method}" for method in ("pca", "tsne", "umap") for rule in ("fps", "kmeans"))
RANDOM_STRATEGIES = ("random", "stratified")


# --------------------------------------------------------------------------- pieces

def load_pool(variant, keep_no_boundary=False, geometry=True):
    """(X, table, bulk) of the distinct GBs at 8x; table gains the geometry labels (region "cell").

    geometry: False (none), True (the default geometry_labels.csv) or the path of a geometry_labels.csv.
    """
    X, table, bulk = ds.load_variant(variant)
    if not keep_no_boundary:
        X, table = mc.drop_no_boundary(X, table)
    if geometry:
        values = gl.load_geometry(table.assign(thickness=8), "both", path=None if geometry is True else geometry)
        for name in GEOMETRY:
            table[name] = values[name].to_numpy()
    return X, table, bulk


def test_splits(n_rows, n_splits=N_SPLITS, n_test=N_TEST, seed=0):
    """(pool rows, test rows) for each split; a fresh random test set every time."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        order = rng.permutation(n_rows)
        out.append((np.sort(order[n_test:]), np.sort(order[:n_test])))
    return out


def largest_remainder(n, counts):
    """Share n picks among groups in proportion to their sizes (counts: {group: size}).

    Leftover picks go to the largest fractional parts (ties: larger group, then group
    name). No group gets more than its size.
    """
    total = sum(counts.values())
    exact = {g: n * c / total for g, c in counts.items()}
    quota = {g: min(int(np.floor(e)), counts[g]) for g, e in exact.items()}
    order = sorted(counts, key=lambda g: (-(exact[g] - np.floor(exact[g])), -counts[g], str(g)))
    left = n - sum(quota.values())
    while left > 0:
        progressed = False
        for g in order:
            if left and quota[g] < counts[g]:
                quota[g] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return quota


def stratified_picks(pool, groups, sizes, rng):
    """{n: rows}: in each group the first rows of one random order, as many as the group's share."""
    members = {g: rng.permutation(pool[groups[pool] == g]) for g in sorted(set(groups[pool]))}
    counts = {g: len(rows) for g, rows in members.items()}
    out = {}
    for n in sizes:
        quota = largest_remainder(n, counts)
        out[n] = np.concatenate([members[g][:quota[g]] for g in members])
    return out


def nested(order, sizes):
    return {n: np.asarray(order[:n]) for n in sizes}


class Spaces:
    """Squared distances between all rows, after scaling fitted on the pool rows (no labels)."""

    def __init__(self, X, pool):
        import krr_classify as krr

        self.D2, self.width = {}, {}
        for scale in ("none", "standard"):
            D2 = krr.scaled_distances(X, pool, scale)
            self.D2[scale] = D2
            self.width[scale] = krr.median_squared_distance(D2, pool)


def predict(spaces, label, model, codes, picks, test, n_classes):
    import krr_classify as krr

    if model == "1nn":
        D2 = spaces.D2["none"][np.ix_(test, picks)]
        return codes[picks][np.argmin(D2, axis=1)]
    scale, gamma, alpha = KRR_SETTINGS[label]
    D2, width = spaces.D2[scale], spaces.width[scale]
    present = np.zeros(n_classes, dtype=bool)
    present[np.unique(codes[picks])] = True
    if present.sum() == 1:
        return np.full(len(test), codes[picks][0])
    K_train = krr.kernel_matrix(D2[np.ix_(picks, picks)], gamma, width)
    K_test = krr.kernel_matrix(D2[np.ix_(test, picks)], gamma, width)
    scores = krr.krr_scores(K_train, krr.one_hot(codes[picks], n_classes), K_test, alpha)
    return krr.pick_class(scores, present)


def predict_values(spaces, model, Y, picks, test):
    """Predicted geometry labels of the test rows (Y: one column per label).

    1nn: the value of the nearest pick. krr: kernel ridge regression whose scaling, gamma and
    alpha are chosen for each label by the leave-one-out error on the picks alone.
    """
    if model == "1nn":
        D2 = spaces.D2["none"][np.ix_(test, picks)]
        return Y[picks][np.argmin(D2, axis=1)]
    import krr_classify as krr

    out = np.empty((len(test), Y.shape[1]))
    choices = loo_choice(spaces, Y[picks], picks)
    for setting in sorted(set(choices)):
        scale, gamma, alpha = setting
        columns = [c for c, chosen in enumerate(choices) if chosen == setting]
        D2, width = spaces.D2[scale], spaces.width[scale]
        K_train = krr.kernel_matrix(D2[np.ix_(picks, picks)], gamma, width)
        K_test = krr.kernel_matrix(D2[np.ix_(test, picks)], gamma, width)
        out[:, columns] = krr.krr_scores(K_train, Y[picks][:, columns], K_test, alpha)
    return out


def loo_errors(K, Y, alphas):
    """Leave-one-out mean squared error of mean-centred kernel ridge regression: (alphas, columns of Y).

    Hat-matrix shortcut e_i / (1 - H_ii) with H = J/n + G (I - J/n), G = K (K + alpha I)^-1,
    from one eigendecomposition of K.
    """
    Y = np.asarray(Y, dtype=float).reshape(len(K), -1)
    n = len(K)
    eigenvalues, vectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    mean = Y.mean(axis=0)
    projected = vectors.T @ (Y - mean)
    ones = vectors.T @ np.ones(n)
    errors = np.empty((len(alphas), Y.shape[1]))
    for position, alpha in enumerate(alphas):
        shrink = eigenvalues / (eigenvalues + alpha)
        fitted = mean + vectors @ (shrink[:, None] * projected)
        H_diag = 1.0 / n + (vectors ** 2) @ shrink - (vectors @ (shrink * ones)) / n
        errors[position] = np.mean(np.square((Y - fitted) / np.maximum(1.0 - H_diag, 1e-12)[:, None]), axis=0)
    return errors


def loo_choice(spaces, Y_picks, picks):
    """For each column of Y_picks (the labels of the rows `picks`), the (scale, gamma, alpha) with the
    smallest leave-one-out error over REGRESSION_SCALES x REGRESSION_GAMMAS x REGRESSION_ALPHAS."""
    import krr_classify as krr

    settings, errors = [], []
    for scale in REGRESSION_SCALES:
        for gamma in REGRESSION_GAMMAS:
            K = krr.kernel_matrix(spaces.D2[scale][np.ix_(picks, picks)], gamma, spaces.width[scale])
            errors.append(loo_errors(K, Y_picks, REGRESSION_ALPHAS))
            settings += [(scale, gamma, alpha) for alpha in REGRESSION_ALPHAS]
    errors = np.vstack(errors)  # (settings, columns); the first smallest wins ties
    return [settings[int(np.argmin(errors[:, c]))] for c in range(errors.shape[1])]


def regression_scores(true, predicted):
    """(R^2 against the mean of the test rows, mean absolute error)."""
    true, predicted = np.asarray(true, dtype=float), np.asarray(predicted, dtype=float)
    total = float(np.square(true - true.mean()).sum())
    r2 = 1.0 - float(np.square(true - predicted).sum()) / total if total > 0 else float("nan")
    return r2, float(np.abs(true - predicted).mean())


def accuracies(true, predicted):
    """(accuracy, balanced accuracy over the classes present in the test set)."""
    accuracy = float(np.mean(true == predicted))
    recalls = [np.mean(predicted[true == c] == c) for c in np.unique(true)]
    return accuracy, float(np.mean(recalls))


def label_codes(table):
    """{label: (codes per row, number of classes)} for the class labels, plus
    "_plane": the plane of every row and "_geometry": {label: values} when the table has them."""
    out = {}
    for label in LABELS:
        values = table[label].astype(str).to_numpy()
        classes = sorted(set(values))
        out[label] = (np.array([classes.index(v) for v in values]), len(classes))
    out["_plane"] = table["plane"].astype(str).to_numpy()
    out["_geometry"] = {name: table[name].to_numpy(dtype=float) for name in GEOMETRY if name in table}
    return out


def score_picks(spaces, codes, picks, test):
    """Rows of learning-curve results for one set of picks: every label and model."""
    rows = []
    plane = codes["_plane"]
    for label, entry in codes.items():
        if label.startswith("_"):
            continue
        values, n_classes = entry
        for model in ("krr", "1nn"):
            predicted = predict(spaces, label, model, values, picks, test, n_classes)
            accuracy, balanced = accuracies(values[test], predicted)
            rows.append({"label": label, "model": model, "accuracy": accuracy, "balanced_accuracy": balanced})
            if label == "grain_class":
                for p in sc.PLANES:
                    at = plane[test] == p
                    if at.any():
                        accuracy, balanced = accuracies(values[test][at], predicted[at])
                        rows.append({"label": f"grain_class@{p}", "model": model, "accuracy": accuracy,
                                     "balanced_accuracy": balanced})
    if codes["_geometry"]:
        names = list(codes["_geometry"])
        Y = np.column_stack([codes["_geometry"][name] for name in names])
        for model in ("krr", "1nn"):
            predicted = predict_values(spaces, model, Y, picks, test)
            for c, label in enumerate(names):
                r2, mae = regression_scores(Y[test, c], predicted[:, c])
                rows.append({"label": label, "model": model, "r2": r2, "mae": mae})
    return rows


def coverage_scores(D, median_distance, groups, picks):
    """Label-free scores of one set of picks (D: distances between all GBs, unscaled)."""
    nearest = D[:, picks].min(axis=1)
    return {"coverage_radius": float(nearest.max() / median_distance),
            "mean_nearest_pick": float(nearest.mean() / median_distance),
            "groups_covered": int(len(set(groups[picks])))}


def gb_groups(table):
    return (table["plane"].astype(str) + " " + table["grain_class"].astype(str)).to_numpy()


def strategy_picks(X, bulk, pool, groups, families, maps, sizes, kernel, split_seed, strategies, n_draws):
    """{strategy: [(draw, {n: rows})]} for one pool. Rows index the full table."""
    import select_gbs

    largest = max(sizes)
    out = {}
    start = int(pool[mc.farthest_from(X[pool], bulk)])
    if "random" in strategies:
        out["random"] = [(d, nested(np.random.default_rng([split_seed, 1, d]).permutation(pool), sizes))
                         for d in range(n_draws)]
    if "stratified" in strategies:
        out["stratified"] = [(d, stratified_picks(pool, groups, sizes, np.random.default_rng([split_seed, 2, d])))
                             for d in range(n_draws)]
    if "fps" in strategies:
        order = pool[mc.farthest_point_sampling(X[pool], largest, int(np.flatnonzero(pool == start)[0]))]
        out["fps"] = [(0, nested(order, sizes))]
    if "kmeans" in strategies:
        out["kmeans"] = [(0, {n: pool[mc.kmeans_members(X[pool], n, seed=0)] for n in sizes})]
    if "gp_variance" in strategies:
        Z, gamma, width = kernel
        order = pool[mc.gp_variance_selection(Z[pool], largest, gamma, width, int(np.flatnonzero(pool == start)[0]))]
        out["gp_variance"] = [(0, nested(order, sizes))]
    for method, coordinates in (maps or {}).items():
        points = coordinates[:-1][pool]
        if f"map_fps_{method}" in strategies:
            order = pool[select_gbs.farthest_points(points, families[pool], count=largest)]
            out[f"map_fps_{method}"] = [(0, nested(order, sizes))]
        if f"map_kmeans_{method}" in strategies:
            out[f"map_kmeans_{method}"] = [
                (0, {n: pool[select_gbs.cluster_centers(points, families[pool], MAP_SEED, count=n)[0]] for n in sizes})]
    return out


def gp_kernel(X, pool):
    """The grain-class KRR kernel: standardised features (pool), gamma 10, median squared distance."""
    import krr_classify as krr

    mean, spread = X[pool].mean(axis=0), X[pool].std(axis=0)
    Z = (X - mean) / np.where(spread > 0, spread, 1.0)
    scale, gamma, _ = KRR_SETTINGS["grain_class"]
    width = krr.median_squared_distance(krr.squared_distances(Z), pool)
    return Z, gamma, width


# --------------------------------------------------------------------------- tasks (worker processes)

def reference_split_task(split, pool, test, maps, sizes, n_draws, keep=False, geometry=True):
    """Every strategy on one split of the reference vector: learning curves and coverage."""
    from scipy.spatial.distance import pdist, squareform

    started = time.time()
    X, table, bulk = load_pool(ds.variants(["reference"])[0], keep, geometry)
    families, groups, codes = table["family"].to_numpy(), gb_groups(table), label_codes(table)
    distances = pdist(X)
    D, median_distance = squareform(distances), float(np.median(distances))
    spaces = Spaces(X, pool)
    picks = strategy_picks(X, bulk, pool, groups, families, maps, sizes, gp_kernel(X, pool), split,
                           FULL_SPACE + MAP_RULES, n_draws)
    curves, coverage = [], []
    for strategy, draws in picks.items():
        for draw, by_size in draws:
            for n, rows in by_size.items():
                key = {"descriptor": "reference", "split": split, "strategy": strategy, "draw": draw, "n": n}
                coverage.append(key | coverage_scores(D, median_distance, groups, rows))
                curves += [key | row for row in score_picks(spaces, codes, rows, test)]
    return curves, coverage, round(time.time() - started, 1)


def variant_task(variant, splits, sizes, n_draws, keep=False, geometry=True):
    """fps and random learning curves of one descriptor setting over every split."""
    started = time.time()
    X, table, bulk = load_pool(variant, keep, geometry)
    defined = np.isfinite(X).all(axis=1)
    families, groups, codes = table["family"].to_numpy(), gb_groups(table), label_codes(table)
    curves = []
    for split, (pool, test) in enumerate(splits):
        pool, test = pool[defined[pool]], test[defined[test]]
        spaces = Spaces(X, pool)
        picks = strategy_picks(X, bulk, pool, groups, families, None, sizes, None, split, ("random", "fps"), n_draws)
        for strategy, draws in picks.items():
            for draw, by_size in draws:
                for n, rows in by_size.items():
                    key = {"descriptor": variant.name, "split": split, "strategy": strategy, "draw": draw, "n": n}
                    curves += [key | row for row in score_picks(spaces, codes, rows, test)]
    return curves, round(time.time() - started, 1)


def read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def existing_task(maps, n_draws, count=30, keep=False, geometry=True):
    """The six existing lists, their union, and `count` picks of every strategy from all GBs."""
    from scipy.spatial.distance import pdist, squareform

    started = time.time()
    X, table, bulk = load_pool(ds.variants(["reference"])[0], keep, geometry)
    families, groups, codes = table["family"].to_numpy(), gb_groups(table), label_codes(table)
    distances = pdist(X)
    D, median_distance = squareform(distances), float(np.median(distances))
    everything = np.arange(len(X))
    spaces = Spaces(X, everything)
    row_of_source = {source: i for i, source in enumerate(table["source"])}
    lists = {}
    for folder, rule in (("FPS", "fps"), ("median", "kmeans")):
        for method in ("pca", "tsne", "umap"):
            path = EXISTING_DIR / folder / f"{method}.txt"
            if path.is_file():
                lists[f"existing {folder}/{method}.txt"] = np.array([row_of_source[s] for s in read_list(path)])
    if lists:
        lists["existing union of the six"] = np.unique(np.concatenate(list(lists.values())))
    picks = strategy_picks(X, bulk, everything, groups, families, maps, (count,), gp_kernel(X, everything), 999,
                           FULL_SPACE + MAP_RULES, n_draws)
    entries = [(name, 0, rows) for name, rows in lists.items()]
    entries += [(f"strategy {s}", d, by_size[count]) for s, draws in picks.items() for d, by_size in draws]
    rows_out = []
    picked = [{"list": name, "draw": draw, "rank": rank, "family": families[row]}
              for name, draw, rows in entries for rank, row in enumerate(rows)]
    for name, draw, rows in entries:
        rest = np.setdiff1d(everything, rows)
        record = {"list": name, "draw": draw, "n": len(rows)} | coverage_scores(D, median_distance, groups, rows)
        for row in score_picks(spaces, codes, rows, rest):
            for metric in ("accuracy", "balanced_accuracy", "r2", "mae"):
                if metric in row:
                    record[f"{row['label']}_{row['model']}_{metric}"] = row[metric]
        # Does the strategy re-create an existing list? (The PCA ones should: PCA is deterministic.)
        for method in ("pca", "tsne", "umap"):
            for folder, rule in (("FPS", "fps"), ("median", "kmeans")):
                if name == f"strategy map_{rule}_{method}" and f"existing {folder}/{method}.txt" in lists:
                    record["same_as_existing_list"] = bool(
                        set(rows.tolist()) == set(lists[f"existing {folder}/{method}.txt"].tolist()))
        rows_out.append(record)
    return rows_out, picked, round(time.time() - started, 1)


def maps_task(keep=False):
    """select_gbs.py's three maps of the reference vector (all GBs + bulk, seed 42)."""
    import select_gbs

    X, table, bulk = load_pool(ds.variants(["reference"])[0], keep, geometry=False)
    vectors = np.vstack([X, bulk])
    return {method: select_gbs.embed(vectors, method, MAP_SEED)[0] for method in ("pca", "tsne", "umap")}


# --------------------------------------------------------------------------- summaries and figures

def summarise(curves):
    """Mean and spread over splits; random strategies are averaged over their draws first."""
    keys = ["descriptor", "strategy", "label", "model", "n"]
    curves = curves.copy()
    for column in ("accuracy", "balanced_accuracy", "r2", "mae"):
        if column not in curves:
            curves[column] = np.nan
    metrics = ["accuracy", "balanced_accuracy", "r2", "mae"]
    per_split = curves.groupby(keys + ["split"], as_index=False)[metrics].mean()
    summary = per_split.groupby(keys, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean"),
        n_splits=("split", "nunique"))
    area = summary.groupby(keys[:-1], as_index=False)[["accuracy_mean", "r2_mean"]].mean().rename(
        columns={"accuracy_mean": "mean_accuracy_over_n", "r2_mean": "mean_r2_over_n"})
    return summary.merge(area, on=keys[:-1])


GRAY = "#8a8a8a"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"
FULL_STYLE = {"random": (GRAY, "-"), "stratified": (SLOTS[0], "-"), "fps": (SLOTS[1], "-"),
              "kmeans": (SLOTS[2], "-"), "gp_variance": (SLOTS[3], "-")}
MAP_STYLE = {"random": (GRAY, "-")}
for _method, _color in zip(("pca", "tsne", "umap"), SLOTS):
    MAP_STYLE[f"map_fps_{_method}"] = (_color, "-")
    MAP_STYLE[f"map_kmeans_{_method}"] = (_color, "--")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, path):
    tmp = sc._temporary(path)
    fig.savefig(tmp, dpi=200, facecolor="white")
    os.replace(tmp, path)


def plot_curves(path, summary, majority):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = summary[summary["descriptor"] == "reference"]
    columns = [("krr", FULL_STYLE, "KRR, full-space strategies"), ("krr", MAP_STYLE, "KRR, rules on 2D maps"),
               ("1nn", FULL_STYLE, "1-NN, full-space strategies"), ("1nn", MAP_STYLE, "1-NN, rules on 2D maps")]
    fig, axes = plt.subplots(len(LABELS), len(columns), figsize=(16, 10), sharex=True, sharey="row", dpi=200,
                             facecolor="white")
    for r, label in enumerate(LABELS):
        for c, (model, styles, title) in enumerate(columns):
            ax = axes[r, c]
            for strategy, (color, line) in styles.items():
                part = reference[(reference["label"] == label) & (reference["model"] == model)
                                 & (reference["strategy"] == strategy)].sort_values("n")
                if part.empty:
                    continue
                ax.plot(part["n"], part["accuracy_mean"], color=color, linestyle=line, linewidth=2, label=strategy)
                ax.fill_between(part["n"], part["accuracy_mean"] - part["accuracy_std"],
                                part["accuracy_mean"] + part["accuracy_std"], color=color, alpha=0.12, linewidth=0)
            ax.axhline(majority[label], color=MUTED, linewidth=0.8, linestyle=":", label="largest class")
            ax.set_xscale("log")
            ax.set_xticks(SIZES, [str(n) for n in SIZES])
            ax.minorticks_off()
            _style(ax)
            if r == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(f"{label.replace('_', ' ')}: test accuracy", fontsize=9, color=INK)
            if r == len(LABELS) - 1:
                ax.set_xlabel("GBs labelled (n)", fontsize=9, color=INK)
    for c in (0, 1):
        axes[0, c].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(f"Mean over {int(summary['n_splits'].max())} test sets of {N_TEST} GBs (band: one standard deviation)",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_coverage(coverage, existing):
    """Label-free coverage scores against n; returns (plt, fig)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [("coverage_radius", "Coverage radius / median distance"),
               ("mean_nearest_pick", "Mean distance to nearest pick / median distance"),
               ("groups_covered", "Plane x grain-class groups covered")]
    mean = coverage.groupby(["strategy", "n", "split"], as_index=False).mean(numeric_only=True)
    mean = mean.groupby(["strategy", "n"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(2, len(columns), figsize=(15, 8), sharex=True, dpi=200, facecolor="white")
    for r, styles in enumerate((FULL_STYLE, MAP_STYLE)):
        for c, (column, title) in enumerate(columns):
            ax = axes[r, c]
            for strategy, (color, line) in styles.items():
                part = mean[mean["strategy"] == strategy].sort_values("n")
                ax.plot(part["n"], part[column], color=color, linestyle=line, linewidth=2, label=strategy)
            if r == 1 and existing is not None:
                for _, row in existing[existing["list"].str.startswith("existing ")
                                       & ~existing["list"].str.contains("union")].iterrows():
                    method = row["list"].split("/")[1].replace(".txt", "")
                    rule = "fps" if "FPS" in row["list"] else "kmeans"
                    color = MAP_STYLE[f"map_{rule}_{method}"][0]
                    ax.scatter([row["n"]], [row[column]], marker="D" if rule == "fps" else "s", s=40, color=color,
                               edgecolors="white", linewidths=1.5, zorder=4)
            ax.set_xscale("log")
            ax.set_xticks(SIZES, [str(n) for n in SIZES])
            ax.minorticks_off()
            _style(ax)
            if r == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if r == 1:
                ax.set_xlabel("GBs picked (n)", fontsize=9, color=INK)
        axes[r, 0].legend(fontsize=7.5, frameon=False)
    return plt, fig


def plot_coverage(path, coverage, existing, n_pool, n_all):
    plt, fig = draw_coverage(coverage, existing)
    fig.suptitle(f"Mean over test splits (picks from the {n_pool:,} pool GBs, scores over all {n_all:,}); markers at "
                 "n = 30: the six existing lists (diamond FPS, square k-means)", fontsize=9, color=MUTED)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_descriptors(summary):
    """FPS and random grain-class learning per descriptor setting; returns (plt, fig), or None for one setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = summary[(summary["label"] == "grain_class") & summary["strategy"].isin(["fps", "random"])]
    area = part.groupby(["descriptor", "strategy", "model"], as_index=False)["mean_accuracy_over_n"].first()
    names = list(dict.fromkeys(part["descriptor"]))
    if len(names) < 2:
        return None
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.24 * len(names) + 1.6), sharey=True, dpi=200, facecolor="white")
    for ax, model in zip(axes, ("1nn", "krr")):
        for strategy, color in (("random", GRAY), ("fps", SLOTS[1])):
            values = area[(area["model"] == model) & (area["strategy"] == strategy)].set_index("descriptor")
            ax.scatter(values["mean_accuracy_over_n"].reindex(names), y, s=24, color=color, linewidths=0,
                       label=strategy, zorder=3)
            ref = values["mean_accuracy_over_n"].get("reference")
            if ref is not None:
                ax.axvline(ref, color=color, linewidth=0.8, linestyle="--")
        ax.set_title(f"{'1-NN' if model == '1nn' else 'KRR (settings tuned on the reference)'}: grain-class "
                     f"accuracy, mean over n = {part['n'].min()}-{part['n'].max()}", fontsize=8.5, color=INK)
        _style(ax)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    return plt, fig


def plot_descriptors(path, summary):
    drawn = draw_descriptors(summary)
    if drawn is None:
        return
    plt, fig = drawn
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="+", help="descriptor settings for the comparison (default: all of "
                                                  "descriptor_sensitivity.py; the reference is always included)")
    parser.add_argument("--no-descriptors", action="store_true", help="reference vector only")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--draws", type=int, default=N_DRAWS, help=f"draws of random strategies (default {N_DRAWS})")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="worker processes, one thread each (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--keep-no-boundary", action="store_true",
                        help="keep D0985 (111/sym00119 and 111/sym00175, plain bulk); left out by default")
    parser.add_argument("--no-geometry", action="store_true", help="no geometry-label learning curves")
    parser.add_argument("--labels", type=Path, default=gl.OUT_DIR / gl.CSV_NAME, help="geometry_labels.csv to use")
    parser.add_argument("--overwrite", action="store_true", help="replace the outputs of an earlier run")
    args = parser.parse_args()

    import pandas as pd

    started = time.time()
    if (args.out_dir / "summary.json").exists() and not args.overwrite:
        parser.error(f"{sc.relative_path(args.out_dir)} holds an earlier run; choose another --out-dir or pass --overwrite")
    keep, geometry = args.keep_no_boundary, (False if args.no_geometry else args.labels)
    chosen = [] if args.no_descriptors else [v for v in ds.variants(args.only) if v.name != "reference"]
    missing = sorted({v.tag for v in chosen + ds.variants(["reference"])
                      if not (sc.run_dir("both_interfaces", v.tag) / sc.INTERFACE_FILE).is_file()})
    if missing:
        raise SystemExit(f"SOAP runs missing: {missing}; run descriptor_grid.py first")
    sizes = sorted(set(args.sizes))
    _, table, _ = load_pool(ds.variants(["reference"])[0], keep, geometry)
    splits = test_splits(len(table), args.splits)
    with ds.worker_pool(args.jobs) as pool:
        maps_job = pool.submit(maps_task, keep)
        descriptor_jobs = [pool.submit(variant_task, v, splits, sizes, args.draws, keep, geometry) for v in chosen]
        maps = maps_job.result()
        print(f"[{time.time() - started:6.0f} s] maps fitted", flush=True)
        existing_job = pool.submit(existing_task, maps, args.draws, 30, keep, geometry)
        split_jobs = [pool.submit(reference_split_task, k, p, t, maps, sizes, args.draws, keep, geometry)
                      for k, (p, t) in enumerate(splits)]
        reference_results = [job.result() for job in split_jobs]
        print(f"[{time.time() - started:6.0f} s] {len(split_jobs)} reference splits", flush=True)
        existing_rows, picked, existing_seconds = existing_job.result()
        descriptor_results = [job.result() for job in descriptor_jobs]
        print(f"[{time.time() - started:6.0f} s] {len(descriptor_jobs)} descriptor settings", flush=True)

    curves = pd.DataFrame([row for result in reference_results for row in result[0]]
                          + [row for result in descriptor_results for row in result[0]])
    coverage = pd.DataFrame([row for result in reference_results for row in result[1]])
    existing = pd.DataFrame(existing_rows)
    summary = summarise(curves)
    majority = {label: float(table[label].astype(str).value_counts(normalize=True).iloc[0]) for label in LABELS}

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    mc.write_table(out / "learning_curves.csv", curves)
    mc.write_table(out / "coverage.csv", coverage)
    mc.write_table(out / "summary.csv", summary)
    mc.write_table(out / "existing_lists.csv", existing)
    mc.write_table(out / "picks.csv", pd.DataFrame(picked))
    tmp = sc._temporary(out / "maps.npz")
    np.savez(tmp, families=table["family"].to_numpy().astype(str),
             **{label: table[label].to_numpy().astype(str) for label in LABELS}, **maps)
    os.replace(tmp, out / "maps.npz")
    plot_curves(out / "learning_curves.png", summary, majority)
    plot_coverage(out / "coverage.png", coverage, existing, len(table) - N_TEST, len(table))
    plot_descriptors(out / "descriptor_efficiency.png", summary)

    ref = summary[(summary["descriptor"] == "reference") & (summary["label"] == "grain_class")]
    columns = ["strategy", "model", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean"]
    best = ref[ref["n"] == 30].sort_values("accuracy_mean", ascending=False)[columns].head(8)
    plane_majority = {p: float(table.loc[table["plane"] == p, "grain_class"].value_counts(normalize=True).iloc[0])
                      for p in sc.PLANES if (table["plane"] == p).any()}
    geometry_r2 = (summary[(summary["descriptor"] == "reference") & summary["label"].isin(GEOMETRY)
                           & (summary["n"] == 30)][["strategy", "label", "model", "r2_mean", "r2_std"]]
                   .to_dict("records"))
    mc.write_json(out / "summary.json", {
        "pool": f"{len(table):,} distinct GBs at 8x", "no_boundary_dropped": not keep,
        "test_sets": {"n": args.splits, "size": N_TEST, "seed": 0},
        "sizes": sizes, "draws_of_random_strategies": args.draws, "krr_settings": KRR_SETTINGS,
        "regression": None if not geometry else {"labels": list(GEOMETRY), "file": sc.relative_path(args.labels), "region": "cell",
                                                 "krr": "settings by leave-one-out on the picks",
                                                 "grid": {"scale": REGRESSION_SCALES, "gamma": REGRESSION_GAMMAS,
                                                          "alpha": REGRESSION_ALPHAS}, "r2_at_n30": geometry_r2},
        "majority_class_share": majority, "grain_class_majority_per_plane": plane_majority, "map_seed": MAP_SEED,
        "descriptors": [v.name for v in chosen], "grain_class_at_n30_best_first": best.to_dict("records"),
        "pca_rules_recreate_existing_lists": (existing.dropna(subset=["same_as_existing_list"])
                                              .set_index("list")["same_as_existing_list"].to_dict()
                                              if "same_as_existing_list" in existing else {}),
        "seconds": {"total": round(time.time() - started, 1), "existing_lists": existing_seconds,
                    "reference_splits": [r[2] for r in reference_results],
                    "descriptors": {v.name: r[1] for v, r in zip(chosen, descriptor_results)}},
        "run": mc.run_metadata()})
    n_classes = table["grain_class"].nunique()
    print(f"grain class at n = 30, best first (largest class {majority['grain_class']:.3f} of the GBs; "
          f"chance balanced accuracy {1 / n_classes:.3f}):")
    print(best.to_string(index=False))
    print(f"done in {time.time() - started:.0f} s; outputs in {sc.relative_path(out)}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
