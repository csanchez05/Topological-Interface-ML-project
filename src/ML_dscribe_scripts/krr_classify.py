#!/usr/bin/env python3
"""Step 1.4: can kernel ridge regression read the plane and the operation off the GB vector?

One vector per GB family (one thickness). Each label is turned into one column
per class (1 for the true class, 0 otherwise), scikit-learn's KernelRidge is
fitted to those columns, and the predicted class is the column with the largest
value. A nearest-neighbour classifier is run alongside for comparison.

Testing is by grouped cross-validation, done twice:

  family    the guide's rule: no family is split between training and testing
  distinct  stricter: families whose vectors coincide (the same GB built more
            than once) also stay together, so no test vector was seen in training

The kernel width, the ridge strength and the feature scaling are chosen by a
second cross-validation inside each training set; the test fold plays no part.

Reported for every label: accuracy, a majority-class baseline, and the ceiling
set by coinciding vectors that carry different labels (no function of the
vector can beat it).

Grain class per plane: a 100 or 110 GB can be one of 4 classes, a 111 GB one of 8,
so the pooled score mixes two questions. krr_by_plane.csv scores the same
out-of-fold predictions on each plane separately, against that plane's chance
(1/4, 1/4, 1/8) and largest class, for the model trained on all planes and for
the models trained on one plane. It also gives the share of predictions that
cannot be right (a 111-only class for a 100/110 GB) and, for grain class, the
share of errors that are a mirror class: grain B mirrored through a plane that
contains the GB normal and keeps the cell (2 per class; notes/grain_classes_20261005.md).

The two NO_BOUNDARY families of ml_common.py (111/sym00119 and 111/sym00175, plain
bulk labelled E) are left out unless --keep-no-boundary is given.

--regression: the same nested cross-validation for the continuous geometry labels
of geometry_labels.py (settings chosen by the inner-fold squared error), with
R^2 and mean absolute error against predicting the training mean; for the
pooled data set also per plane.

--descriptor-type ace uses the ACE vectors (src/ACE/CoGe) instead of SOAP and writes into
calculations/ML_ACE/CoGe/ instead of calculations/ML_dscribe/CoGe/.

Outputs, in calculations/ML_dscribe/CoGe/<descriptor tag>/<vector tag>/ (a folder that
already holds krr_summary.csv is refused unless --overwrite is given):
  krr_summary.csv         one row per data set, label and grouping
  krr_by_plane.csv        grain class and handedness scored on each plane separately
  krr_regression.csv      geometry labels (only with --regression)
  krr_predictions.csv     out-of-fold prediction for every family
  krr_confusions.csv      true label x predicted label counts
  krr_confused_pairs.csv  pairs of labels the model mixes up, most mixed first
  krr_summary.json        settings and the hyperparameters chosen in each fold
  krr_confusion_*.png     confusion matrices for plane and grain class, "distinct" grouping
                          (skip with --no-plots)
"""
import argparse
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import gb_labels
import geometry_labels
import ml_common as mc
from ml_common import sc

LABELS = ("handedness", "grain_class", "first_op", "rot_set")
NEIGHBOURS = (1, 3, 5)
# Labels scored per plane in krr_by_plane.csv.
PER_PLANE = ("grain_class", "handedness")


# --------------------------------------------------------------------------- pieces

def one_hot(codes, n_classes):
    Y = np.zeros((len(codes), n_classes))
    Y[np.arange(len(codes)), codes] = 1.0
    return Y


def squared_distances(Z):
    norms = (Z ** 2).sum(axis=1)
    return np.maximum(norms[:, None] + norms[None, :] - 2.0 * (Z @ Z.T), 0.0)


def scaled_distances(X, train, scale):
    """Squared distances between all rows, after scaling fitted on the training rows only."""
    mean = X[train].mean(axis=0)
    Z = X - mean
    if scale == "standard":
        spread = X[train].std(axis=0)
        Z = Z / np.where(spread > 0, spread, 1.0)
    elif scale != "none":
        raise ValueError(f"unknown scale {scale!r}")
    return squared_distances(Z)


def median_squared_distance(D2, rows):
    values = D2[np.ix_(rows, rows)][np.triu(np.ones((len(rows), len(rows)), dtype=bool), 1)]
    values = values[values > 0]
    return float(np.median(values)) if values.size else 1.0


def kernel_matrix(D2, gamma, width):
    """exp(-gamma d^2 / width), built in a single array."""
    K = -gamma * D2
    K /= width
    return np.exp(K, out=K)


def krr_scores(K_train, Y_train, K_test, alpha):
    """Kernel ridge regression on class-indicator columns; returns one score per class."""
    from sklearn.kernel_ridge import KernelRidge

    mean = Y_train.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = KernelRidge(alpha=alpha, kernel="precomputed").fit(K_train, Y_train - mean)
    return model.predict(K_test) + mean


def pick_class(scores, present):
    """Largest score among the classes that occur in the training set."""
    masked = np.where(present[None, :], scores, -np.inf)
    return masked.argmax(axis=1)


def nearest_rows(D2_test_train, k):
    """Indices of the k nearest training rows, nearest first; equal distances keep column order.

    Same result as a full stable sort of every row, without sorting what is not needed.
    """
    k = min(k, D2_test_train.shape[1])
    if k == D2_test_train.shape[1]:
        return np.argsort(D2_test_train, axis=1, kind="stable")
    kth = np.partition(D2_test_train, k - 1, axis=1)[:, k - 1]
    order = np.empty((len(D2_test_train), k), dtype=int)
    for row, distances in enumerate(D2_test_train):
        candidates = np.flatnonzero(distances <= kth[row])
        order[row] = candidates[np.argsort(distances[candidates], kind="stable")][:k]
    return order


def knn_vote(order, train_codes, k):
    """Majority vote of the first k columns of nearest_rows(); ties go to the nearest."""
    predicted = np.empty(len(order), dtype=int)
    for row, neighbours in enumerate(train_codes[order[:, :k]]):
        counts = Counter(neighbours.tolist())
        best = max(counts.values())
        predicted[row] = next(code for code in neighbours if counts[code] == best)
    return predicted


def knn_predict(D2_test_train, train_codes, k):
    """Majority vote of the k nearest training rows; ties go to the nearest."""
    return knn_vote(nearest_rows(D2_test_train, k), train_codes, k)


def group_folds(groups, n_splits, seed):
    """Folds that never split a group. Returns a list of (train rows, test rows)."""
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_splits, len(set(groups.tolist())))
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(groups)), groups=groups))
    for train, test in folds:
        if np.intersect1d(groups[train], groups[test]).size:
            raise RuntimeError("a group was split between training and testing")
    return folds


def label_ceiling(codes, groups):
    """Best accuracy any function of the vector can reach when coinciding vectors carry different labels."""
    best = 0
    for g in np.unique(groups):
        best += Counter(codes[groups == g].tolist()).most_common(1)[0][1]
    return best / len(codes)


def mirror_classes(gb_root=None, planes=sc.PLANES):
    """{plane: {grain class: set of its mirror classes}}.

    A mirror class is the class of m @ R, where R is a turn of that class and m a mirror
    (det -1, trace 1) among the plane's listed turns that keeps the GB normal (the
    plane's first cell axis): grain B mirrored through a plane that contains the normal.
    """
    from pathlib import Path

    gb_root = Path(gb_root or sc.GB_ROOT)
    group = gb_labels.crystal_rotations()
    out = {}
    for plane in planes:
        interfaces, _ = gb_labels.parse_information(gb_root / f"CoGe_{plane}_GBs" / "Interface" / "Information.txt")
        turns = {}
        for entry in interfaces:
            for operation in entry["operations"]:
                R = gb_labels.to_cubic(operation["matrix"][:, :3], plane)
                turns.setdefault(gb_labels.rotation_name(R), R)
        normal = gb_labels.FRAMES[plane][0] / np.linalg.norm(gb_labels.FRAMES[plane][0])
        mirrors = [R for R in turns.values() if np.linalg.det(R) < 0 and abs(np.trace(R) - 1.0) < 1e-6
                   and np.allclose(R @ normal, normal, atol=1e-6)]
        out[plane] = {}
        for R in turns.values():
            out[plane].setdefault(gb_labels.grain_class(R, group), set()).update(
                gb_labels.grain_class(m @ R, group) for m in mirrors)
    return out


def plane_scores(true, predicted, plane, possible, mirrors=None):
    """Class predictions scored on each plane separately: one dict per plane.

    true, predicted, plane: arrays of strings, one per row. possible: {plane: the classes
    that occur on that plane}. mirrors: optional {plane: {class: its mirror classes}}.
    Chance is 1 / (classes possible on the plane); majority is the share of the plane's
    largest class among the scored rows.
    """
    true, predicted, plane = (np.asarray(a).astype(str) for a in (true, predicted, plane))
    rows = []
    for p in sorted(set(plane.tolist())):
        at = plane == p
        t, q = true[at], predicted[at]
        classes = sorted(possible[p])
        correct = t == q
        row = {"plane": p, "n": int(at.sum()), "n_classes_possible": len(classes),
               "chance": round(1.0 / len(classes), 4),
               "majority": round(float(Counter(t.tolist()).most_common(1)[0][1] / len(t)), 4),
               "accuracy": round(float(correct.mean()), 4),
               "balanced_accuracy": round(float(np.mean([np.mean(q[t == c] == c) for c in np.unique(t)])), 4),
               "impossible_share": round(float(np.mean(~np.isin(q, classes))), 4)}
        if mirrors is not None:
            mirror = np.array([b in mirrors[p].get(a, ()) for a, b in zip(t, q)])
            wrong = ~correct
            row["mirror_share_of_errors"] = round(float(mirror[wrong].mean()), 4) if wrong.any() else ""
            row["mirror_share_if_random"] = round(float(np.mean(
                [len(mirrors[p].get(a, ())) / max(len(classes) - 1, 1) for a in t])), 4)
            row["correct_or_mirror"] = round(float((correct | mirror).mean()), 4)
        rows.append(row)
    return rows


def regression_scores(true, predicted):
    """(R^2 against the mean of the scored rows, mean absolute error)."""
    true, predicted = np.asarray(true, dtype=float), np.asarray(predicted, dtype=float)
    total = float(np.square(true - true.mean()).sum())
    r2 = 1.0 - float(np.square(true - predicted).sum()) / total if total > 0 else float("nan")
    return round(r2, 4), round(float(np.abs(true - predicted).mean()), 6)


# --------------------------------------------------------------------------- nested cross-validation

def cv_fold(X, targets, number, train, test, inner, gammas, alphas, scales):
    """One outer fold: choose the settings on the training rows, then predict the test rows.

    inner: the (fit rows, check rows) splits of the training set, as positions in `train`.
    Returns (predicted, entry): predicted[model][name] holds the class codes of
    the test rows, entry the settings chosen. Folds do not depend on each other,
    so they can run in separate processes.
    """
    names = list(targets)
    n_classes = {name: int(targets[name].max()) + 1 for name in names}
    columns, start = {}, 0
    for name in names:
        columns[name] = slice(start, start + n_classes[name])
        start += n_classes[name]
    Y = np.hstack([one_hot(targets[name], n_classes[name]) for name in names])
    krr_grid = [(scale, gamma, alpha) for scale in scales for gamma in sorted(gammas)
                for alpha in sorted(alphas, reverse=True)]
    knn_grid = [(scale, k) for scale in scales for k in NEIGHBOURS]
    present = {name: np.bincount(targets[name][train], minlength=n_classes[name]) > 0 for name in names}
    D2 = {scale: scaled_distances(X, train, scale) for scale in scales}
    width = {scale: median_squared_distance(D2[scale], train) for scale in scales}

    # ---- choose hyperparameters on the training rows only
    krr_correct = {name: np.zeros(len(krr_grid)) for name in names}
    knn_correct = {name: np.zeros(len(knn_grid)) for name in names}
    for fit_rows, check_rows in inner:
        fit_rows, check_rows = train[fit_rows], train[check_rows]
        inner_present = {name: np.bincount(targets[name][fit_rows], minlength=n_classes[name]) > 0
                         for name in names}
        K, built = None, None
        for position, (scale, gamma, alpha) in enumerate(krr_grid):
            if built != (scale, gamma):
                K = None  # release the previous kernel before building the next
                K, built = kernel_matrix(D2[scale], gamma, width[scale]), (scale, gamma)
            scores = krr_scores(K[np.ix_(fit_rows, fit_rows)], Y[fit_rows], K[np.ix_(check_rows, fit_rows)], alpha)
            for name in names:
                guess = pick_class(scores[:, columns[name]], inner_present[name])
                krr_correct[name][position] += (guess == targets[name][check_rows]).sum()
        K = None
        order = {scale: nearest_rows(D2[scale][np.ix_(check_rows, fit_rows)], max(NEIGHBOURS)) for scale in scales}
        for position, (scale, k) in enumerate(knn_grid):
            for name in names:
                guess = knn_vote(order[scale], targets[name][fit_rows], k)
                knn_correct[name][position] += (guess == targets[name][check_rows]).sum()

    # ---- refit on the whole training set with the chosen settings, predict the test fold
    predicted = {"krr": {}, "knn": {}, "baseline": {}}
    entry = {"fold": number, "n_train": len(train), "n_test": len(test), "krr": {}, "knn": {}}
    best_krr = {name: krr_grid[int(np.argmax(krr_correct[name]))] for name in names}
    for setting in sorted(set(best_krr.values())):
        scale, gamma, alpha = setting
        K = None
        K = kernel_matrix(D2[scale], gamma, width[scale])
        scores = krr_scores(K[np.ix_(train, train)], Y[train], K[np.ix_(test, train)], alpha)
        for name in names:
            if best_krr[name] == setting:
                predicted["krr"][name] = pick_class(scores[:, columns[name]], present[name])
    K = None
    order = {scale: nearest_rows(D2[scale][np.ix_(test, train)], max(NEIGHBOURS)) for scale in scales}
    for name in names:
        scale, gamma, alpha = best_krr[name]
        entry["krr"][name] = {"scale": scale, "gamma_times_median": gamma, "alpha": alpha,
                              "inner_accuracy": float(krr_correct[name].max() / len(train))}
        scale, k = knn_grid[int(np.argmax(knn_correct[name]))]
        predicted["knn"][name] = knn_vote(order[scale], targets[name][train], k)
        entry["knn"][name] = {"scale": scale, "k": k,
                              "inner_accuracy": float(knn_correct[name].max() / len(train))}
        predicted["baseline"][name] = int(np.bincount(targets[name][train]).argmax())
    return predicted, entry


def regression_fold(X, targets, number, train, test, inner, gammas, alphas, scales):
    """cv_fold() for continuous targets: settings chosen per target by the inner-fold squared error.

    targets: {name: float values}. Returns (predicted, entry) as cv_fold() does, with
    predicted values; the baseline is the training mean.
    """
    names = list(targets)
    Y = np.column_stack([np.asarray(targets[name], dtype=float) for name in names])
    krr_grid = [(scale, gamma, alpha) for scale in scales for gamma in sorted(gammas)
                for alpha in sorted(alphas, reverse=True)]
    knn_grid = [(scale, k) for scale in scales for k in NEIGHBOURS]
    D2 = {scale: scaled_distances(X, train, scale) for scale in scales}
    width = {scale: median_squared_distance(D2[scale], train) for scale in scales}

    # ---- choose hyperparameters on the training rows only
    krr_error, knn_error = np.zeros((len(krr_grid), len(names))), np.zeros((len(knn_grid), len(names)))
    spread = np.zeros(len(names))
    for fit_rows, check_rows in inner:
        fit_rows, check_rows = train[fit_rows], train[check_rows]
        spread += np.square(Y[check_rows] - Y[fit_rows].mean(axis=0)).sum(axis=0)
        K, built = None, None
        for position, (scale, gamma, alpha) in enumerate(krr_grid):
            if built != (scale, gamma):
                K = None
                K, built = kernel_matrix(D2[scale], gamma, width[scale]), (scale, gamma)
            guess = krr_scores(K[np.ix_(fit_rows, fit_rows)], Y[fit_rows], K[np.ix_(check_rows, fit_rows)], alpha)
            krr_error[position] += np.square(guess - Y[check_rows]).sum(axis=0)
        K = None
        order = {scale: nearest_rows(D2[scale][np.ix_(check_rows, fit_rows)], max(NEIGHBOURS)) for scale in scales}
        for position, (scale, k) in enumerate(knn_grid):
            guess = Y[fit_rows][order[scale][:, :k]].mean(axis=1)
            knn_error[position] += np.square(guess - Y[check_rows]).sum(axis=0)

    # ---- refit on the whole training set with the chosen settings, predict the test fold
    predicted = {"krr": {}, "knn": {}, "baseline": {}}
    entry = {"fold": number, "n_train": len(train), "n_test": len(test), "krr": {}, "knn": {}}
    best_krr = {name: krr_grid[int(np.argmin(krr_error[:, c]))] for c, name in enumerate(names)}
    for setting in sorted(set(best_krr.values())):
        scale, gamma, alpha = setting
        K = None
        K = kernel_matrix(D2[scale], gamma, width[scale])
        guess = krr_scores(K[np.ix_(train, train)], Y[train], K[np.ix_(test, train)], alpha)
        for c, name in enumerate(names):
            if best_krr[name] == setting:
                predicted["krr"][name] = guess[:, c]
    K = None
    order = {scale: nearest_rows(D2[scale][np.ix_(test, train)], max(NEIGHBOURS)) for scale in scales}
    for c, name in enumerate(names):
        scale, gamma, alpha = best_krr[name]
        inner_r2 = 1.0 - krr_error[:, c].min() / spread[c] if spread[c] > 0 else float("nan")
        entry["krr"][name] = {"scale": scale, "gamma_times_median": gamma, "alpha": alpha,
                              "inner_r2": float(inner_r2)}
        position = int(np.argmin(knn_error[:, c]))
        scale, k = knn_grid[position]
        predicted["knn"][name] = Y[train, c][order[scale][:, :k]].mean(axis=1)
        entry["knn"][name] = {"scale": scale, "k": k,
                              "inner_r2": float(1.0 - knn_error[position, c] / spread[c]) if spread[c] > 0 else None}
        predicted["baseline"][name] = float(Y[train, c].mean())
    return predicted, entry


def start_nested_cv(X, targets, groups, n_outer, n_inner, gammas, alphas, scales, seed, pool=None,
                    fold_function=None):
    """Lay out the folds of nested_cv(); with a process pool, start them all at once.

    Returns a function that runs the folds (or waits for the pool) and gives
    what nested_cv() returns. Starting several of these before finishing any
    lets one pool work on all of them together. fold_function: cv_fold (class
    labels, the default) or regression_fold (continuous targets).
    """
    fold_function = fold_function or cv_fold
    folds = []
    for number, (train, test) in enumerate(group_folds(groups, n_outer, seed)):
        folds.append((number, train, test, group_folds(groups[train], n_inner, seed + 1 + number)))
    settings = (list(gammas), list(alphas), list(scales))
    pending = None if pool is None else [pool.submit(fold_function, X, targets, *fold, *settings) for fold in folds]
    fill = np.nan if fold_function is regression_fold else -1

    def finish(log=None):
        names = list(targets)
        predictions = {model: {name: np.full(len(X), fill) for name in names} for model in ("krr", "knn", "baseline")}
        fold_of = np.full(len(X), -1)
        chosen = []
        for position, (number, train, test, inner) in enumerate(folds):
            if pending is None:
                predicted, entry = fold_function(X, targets, number, train, test, inner, *settings)
            else:
                predicted, entry = pending[position].result()
            fold_of[test] = number
            for model in predictions:
                for name in names:
                    predictions[model][name][test] = predicted[model][name]
            chosen.append(entry)
            if log:
                log(f"    fold {number + 1}: {len(train)} train, {len(test)} test")
        return predictions, fold_of, chosen

    return finish


def nested_cv(X, targets, groups, n_outer, n_inner, gammas, alphas, scales, seed, log=None, pool=None,
              fold_function=None):
    """Out-of-fold predictions for several labels at once.

    targets: {name: integer class codes} (or float values with fold_function=regression_fold).
    groups: one group number per row. Returns (predictions, fold, chosen): predictions[model][name]
    holds class codes (or values), fold the outer fold of each row, chosen the hyperparameters per fold.
    """
    return start_nested_cv(X, targets, groups, n_outer, n_inner, gammas, alphas, scales, seed, pool,
                           fold_function)(log)


# --------------------------------------------------------------------------- plots

def plot_confusion(path, table, title, subtitle):
    """Row-normalised confusion matrix: one blue ramp, counts written in the cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, secondary, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    ramp = LinearSegmentedColormap.from_list("blue", [surface, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    counts = table.to_numpy(dtype=float)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    n_rows, n_cols = counts.shape
    longest = max(len(str(label)) for label in list(table.index) + list(table.columns))
    cell = 0.62
    fig, ax = plt.subplots(figsize=(max(6.4, cell * n_cols + 0.11 * longest + 1.6),
                                    max(4.6, cell * n_rows + 0.11 * longest + 2.2)), facecolor=surface)
    ax.set_facecolor(surface)
    ax.pcolormesh(share, cmap=ramp, vmin=0.0, vmax=1.0, edgecolors=surface, linewidth=2)
    ax.invert_yaxis()
    for r in range(n_rows):
        for c in range(n_cols):
            if counts[r, c]:
                ax.text(c + 0.5, r + 0.5, f"{int(counts[r, c])}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if share[r, c] > 0.55 else ink)
    ax.set_xticks(np.arange(n_cols) + 0.5, table.columns, rotation=45 if longest > 4 else 0,
                  ha="right" if longest > 4 else "center", fontsize=9, color=secondary)
    ax.set_yticks(np.arange(n_rows) + 0.5, table.index, fontsize=9, color=secondary)
    ax.set_xlabel("predicted", fontsize=9, color=muted)
    ax.set_ylabel("true", fontsize=9, color=muted)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=11, color=ink)
    fig.text(0.02, 0.945, subtitle, ha="left", va="top", fontsize=8.5, color=secondary, linespacing=1.4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, dpi=160, facecolor=surface)
    plt.close(fig)
    tmp.replace(path)


# --------------------------------------------------------------------------- command line

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_vector_arguments(parser)
    parser.add_argument("--tol", type=float, default=mc.DEFAULT_TOL,
                        help=f"vectors closer than this (relative) count as the same GB (default {mc.DEFAULT_TOL:g})")
    parser.add_argument("--datasets", nargs="+", default=["all", *sc.PLANES], choices=["all", *sc.PLANES],
                        help="all three planes together and/or each plane on its own (default: all four)")
    parser.add_argument("--folds", type=int, default=5, help="outer folds (default 5)")
    parser.add_argument("--inner-folds", type=int, default=3, help="folds used to choose hyperparameters (default 3)")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0],
                        help="kernel exp(-gamma d^2 / median d^2): values of gamma to try (default 0.1 1 10 100)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1.0],
                        help="ridge strengths to try (default 1e-6 1e-4 1e-2 1)")
    parser.add_argument("--scales", nargs="+", default=["standard", "none"], choices=["standard", "none"],
                        help="feature scalings to try: standard (each feature to unit spread) and/or none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes; the folds of every data set and grouping are shared out "
                             "among them, one thread each (default 1: no extra processes)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--keep-no-boundary", action="store_true",
                        help="keep 111/sym00119 and 111/sym00175 (plain bulk, labelled E); left out by default")
    parser.add_argument("--regression", nargs="*", default=None, choices=geometry_labels.TARGETS,
                        help="also predict these geometry labels of geometry_labels.py (no value: all of "
                             f"{' '.join(geometry_labels.TARGETS)}); writes krr_regression.csv")
    parser.add_argument("--labels", type=Path, default=geometry_labels.OUT_DIR / geometry_labels.CSV_NAME,
                        help="geometry_labels.csv for --regression")
    parser.add_argument("--overwrite", action="store_true",
                        help="write into a folder that already holds krr_summary.csv")
    args = parser.parse_args()
    mc.resolve_vector_arguments(args, parser)
    import pandas as pd

    started = time.time()
    out = mc.output_dir(args)
    if (out / "krr_summary.csv").exists() and not args.overwrite:
        parser.error(f"{sc.relative_path(out)} already holds krr_summary.csv; choose another --out-dir "
                     "or pass --overwrite")
    X, index, notes = mc.load_family_vectors(args.descriptor, args.kind, args.radius, args.planes, args.thickness,
                                             **mc.vector_source(args))
    if not args.keep_no_boundary:
        gone = mc.no_boundary_rows(index)
        if gone.any():
            notes["dropped_no_boundary"] = sorted(set(index.loc[gone, "family"]))
            X, index = X[~gone], index.loc[~gone].reset_index(drop=True)
    if len(index) == 0:
        parser.error(f"no structure with thickness {args.thickness} in {args.descriptor}")
    regression_targets = None
    if args.regression is not None:
        names = list(args.regression) or list(geometry_labels.TARGETS)
        geometry = geometry_labels.load_geometry(index, args.planes if args.kind != "global" else "both", path=args.labels)
        regression_targets = {name: geometry[name].to_numpy(dtype=float) for name in names}
    labels = gb_labels.load_labels().set_index("family").loc[index["family"]].reset_index()
    family = index["family"].to_numpy()
    plane = index["plane"].to_numpy()
    listed = [set(text.split("|")) for text in labels["rotations"]]
    family_group = pd.factorize(family)[0]
    # "distinct" groups: coinciding vectors stay together, and so do all thicknesses of one family.
    duplicate = mc.family_groups(family_group, mc.coincidence_groups(mc.relative_distances(X), args.tol))
    n_families = int(family_group.max()) + 1
    which = "all thicknesses pooled" if args.thickness == "all" else f"{args.thickness}x"
    print(f"{len(family)} structures in {n_families} families ({which}), {len(set(duplicate.tolist()))} distinct GBs "
          f"(tolerance {args.tol:g}), vector {mc.vector_tag(args.kind, args.radius, args.planes, args.thickness)}",
          flush=True)

    pool = None
    if args.jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # The workers start after this, so each of them runs its fold on one thread.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        pool = ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"))

    # Lay out every data set and grouping first (largest first), so that a pool works on all of them at once.
    prepared, pending = {}, {}
    for dataset in args.datasets:
        rows = np.arange(len(family)) if dataset == "all" else np.flatnonzero(plane == dataset)
        if len(rows) < 2 * args.folds:
            print(f"  {dataset}: only {len(rows)} families, skipped")
            continue
        names = (["plane"] if dataset == "all" else []) + list(LABELS)
        names = [name for name in names if labels[name].iloc[rows].nunique() > 1]
        classes = {name: sorted(labels[name].iloc[rows].unique()) for name in names}
        targets = {name: np.array([classes[name].index(v) for v in labels[name].iloc[rows]]) for name in names}
        prepared[dataset] = (rows, names, classes, targets)
        for grouping, groups in (("family", family_group[rows]), ("distinct", duplicate[rows])):
            pending[dataset, grouping] = start_nested_cv(X[rows], targets, groups, args.folds, args.inner_folds,
                                                         args.gammas, args.alphas, args.scales, args.seed, pool)
            if regression_targets is not None:
                pending["regression", dataset, grouping] = start_nested_cv(
                    X[rows], {name: values[rows] for name, values in regression_targets.items()}, groups,
                    args.folds, args.inner_folds, args.gammas, args.alphas, args.scales, args.seed, pool,
                    fold_function=regression_fold)
    possible = {name: {p: set(labels.loc[plane == p, name]) for p in sc.PLANES if (plane == p).any()}
                for name in PER_PLANE}
    mirrors = mirror_classes()

    def progress(text):
        print(f"{text} ({time.time() - started:.0f} s)", flush=True)

    summary_rows, prediction_rows, confusion_rows, pair_rows, chosen_all = [], [], [], [], {}
    plane_rows, regression_rows = [], []
    for dataset, (rows, names, classes, targets) in prepared.items():
        # A structure has a "coinciding partner" when another family shares its distinct group.
        families_per_group = np.bincount(np.unique(np.stack([duplicate, family_group]), axis=1)[0])
        size_of_group = families_per_group[duplicate[rows]]
        for grouping, groups in (("family", family_group[rows]), ("distinct", duplicate[rows])):
            print(f"  {dataset}, grouped by {grouping}: {len(rows)} structures, {len(set(groups.tolist()))} groups", flush=True)
            predictions, fold, chosen = pending.pop((dataset, grouping))(progress)
            chosen_all[f"{dataset}/{grouping}"] = chosen
            # A test family is "seen" when a coinciding vector sat in its training set.
            seen = np.array([np.any((duplicate[rows] == duplicate[rows][i]) & (fold != fold[i]))
                             for i in range(len(rows))])
            for name in names:
                truth = targets[name]
                correct = {model: predictions[model][name] == truth for model in predictions}
                per_fold = [correct["krr"][fold == f].mean() for f in np.unique(fold)]
                counts = np.bincount(truth)
                row = {
                    "dataset": dataset, "label": name, "grouping": grouping,
                    "n_families": len(set(family_group[rows].tolist())), "n_structures": len(rows),
                    "n_classes": len(classes[name]), "smallest_class": int(counts.min()),
                    "baseline_majority": round(float(correct["baseline"].mean()), 4),
                    "krr_accuracy": round(float(correct["krr"].mean()), 4),
                    "krr_fold_std": round(float(np.std(per_fold)), 4),
                    "knn_accuracy": round(float(correct["knn"].mean()), 4),
                    "ceiling_from_coinciding_vectors": round(label_ceiling(truth, duplicate[rows]), 4),
                    "n_with_coinciding_partner": int((size_of_group > 1).sum()),
                    "krr_accuracy_with_partner": _mean(correct["krr"][size_of_group > 1]),
                    "krr_accuracy_without_partner": _mean(correct["krr"][size_of_group == 1]),
                    "n_test_vectors_seen_in_training": int(seen.sum()),
                    "krr_accuracy_seen": _mean(correct["krr"][seen]),
                    "krr_accuracy_unseen": _mean(correct["krr"][~seen]),
                }
                if name == "first_op":
                    hit = np.array([classes[name][code] in listed[i] for code, i in zip(predictions["krr"][name], rows)])
                    row["krr_accuracy_any_listed_op"] = round(float(hit.mean()), 4)
                summary_rows.append(row)
                if name in PER_PLANE:
                    names_of = np.array(classes[name], dtype=object)
                    for model in ("krr", "knn"):
                        for scored in plane_scores(names_of[truth], names_of[predictions[model][name]], plane[rows],
                                                   possible[name], mirrors if name == "grain_class" else None):
                            plane_rows.append({"trained_on": "all planes" if dataset == "all" else f"{dataset} only",
                                               "grouping": grouping, "label": name, "model": model} | scored)

                for model in ("krr", "knn"):
                    table = Counter(zip(truth.tolist(), predictions[model][name].tolist()))
                    for (t, p), count in sorted(table.items()):
                        confusion_rows.append({"dataset": dataset, "label": name, "grouping": grouping, "model": model,
                                               "true": classes[name][t], "predicted": classes[name][p], "count": count})
                table = Counter(zip(truth.tolist(), predictions["krr"][name].tolist()))
                for a in range(len(classes[name])):
                    for b in range(a + 1, len(classes[name])):
                        mixed = table.get((a, b), 0) + table.get((b, a), 0)
                        if mixed:
                            pair_rows.append({"dataset": dataset, "label": name, "grouping": grouping,
                                              "label_a": classes[name][a], "label_b": classes[name][b],
                                              "a_called_b": table.get((a, b), 0), "b_called_a": table.get((b, a), 0),
                                              "n_a": int(counts[a]), "n_b": int(counts[b]),
                                              "share_mixed": round(mixed / (counts[a] + counts[b]), 4)})
                if not args.no_plots and grouping == "distinct" and name in ("plane", "grain_class"):
                    matrix = pd.DataFrame(0, index=classes[name], columns=classes[name])
                    for (t, p), count in table.items():
                        matrix.iloc[t, p] = count
                    plot_confusion(out / f"krr_confusion_{dataset}_{name}_{grouping}.png", matrix,
                                   f"KRR: {name.replace('_', ' ')} from the GB vector",
                                   f"{'all planes' if dataset == 'all' else dataset + ' plane'}, {len(rows)} GBs, "
                                   f"coinciding vectors kept in one fold\n"
                                   f"accuracy {correct['krr'].mean():.1%} (baseline "
                                   f"{correct['baseline'].mean():.1%}); number = GBs, colour = share of each row")
            if ("regression", dataset, grouping) in pending:
                values, _, chosen_regression = pending.pop(("regression", dataset, grouping))(progress)
                chosen_all[f"regression {dataset}/{grouping}"] = chosen_regression
                scopes = [("all", np.ones(len(rows), dtype=bool))]
                if dataset == "all":
                    scopes += [(p, plane[rows] == p) for p in sc.PLANES if (plane[rows] == p).any()]
                for name, truth in regression_targets.items():
                    truth = truth[rows]
                    for scope, at in scopes:
                        record = {"dataset": dataset, "grouping": grouping, "target": name, "scored_on": scope,
                                  "n": int(at.sum()), "mean": round(float(truth[at].mean()), 6),
                                  "std": round(float(truth[at].std()), 6)}
                        for model in ("krr", "knn", "baseline"):
                            r2, mae = regression_scores(truth[at], values[model][name][at])
                            record[f"{model}_r2"], record[f"{model}_mae"] = r2, mae
                        regression_rows.append(record)
            for position, i in enumerate(rows):
                record = {"dataset": dataset, "grouping": grouping, "family": family[i],
                          "thickness": int(index["thickness"].iloc[i]), "fold": int(fold[position]),
                          "distinct_id": f"D{duplicate[i]:04d}", "seen_in_training": bool(seen[position])}
                for name in names:
                    record[f"{name}_true"] = classes[name][targets[name][position]]
                    record[f"{name}_krr"] = classes[name][predictions["krr"][name][position]]
                    record[f"{name}_knn"] = classes[name][predictions["knn"][name][position]]
                prediction_rows.append(record)
    if pool is not None:
        pool.shutdown()

    summary = pd.DataFrame(summary_rows)
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs = pairs.sort_values(["dataset", "label", "grouping", "share_mixed"],
                                  ascending=[True, True, True, False])
    mc.write_table(out / "krr_summary.csv", summary)
    mc.write_table(out / "krr_by_plane.csv", pd.DataFrame(plane_rows))
    if regression_targets is not None:
        mc.write_table(out / "krr_regression.csv", pd.DataFrame(regression_rows))
    mc.write_table(out / "krr_predictions.csv", pd.DataFrame(prediction_rows))
    mc.write_table(out / "krr_confusions.csv", pd.DataFrame(confusion_rows))
    mc.write_table(out / "krr_confused_pairs.csv", pairs)
    mc.write_json(out / "krr_summary.json", {
        "vector": mc.vector_settings(args),
        "n_families": n_families, "n_structures": len(family),
        "n_distinct_vectors": int(len(set(duplicate.tolist()))), "tolerance": args.tol,
        "method": {
            "model": "sklearn KernelRidge on class-indicator columns (centred on the training mean), "
                     "prediction = class with the largest value",
            "kernel": "exp(-gamma * d^2 / median d^2 of the training set)",
            "grid": {"gamma": args.gammas, "alpha": args.alphas, "scale": args.scales},
            "comparison": f"k nearest neighbours, k in {list(NEIGHBOURS)}",
            "outer_folds": args.folds, "inner_folds": args.inner_folds, "seed": args.seed,
            "groupings": {"family": "no family split between training and testing",
                          "distinct": "families with coinciding vectors also kept together"},
        },
        "results": summary_rows,
        "per_plane": {"rows": "krr_by_plane.csv", "mirror_classes": {p: {c: sorted(m) for c, m in by.items()}
                                                                     for p, by in mirrors.items()}},
        "regression": None if regression_targets is None else {
            "targets": list(regression_targets), "labels": sc.relative_path(args.labels),
            "region": geometry_labels.REGION_OF_PLANES.get(args.planes, "cell") if args.kind != "global" else "cell",
            "settings_chosen_by": "inner-fold squared error", "results": regression_rows},
        "hyperparameters_chosen_per_fold": chosen_all,
        "notes": notes,
        "seconds": round(time.time() - started, 1),
        "run": mc.run_metadata(),
    })

    print(f"\n{'data':5s} {'label':12s} {'grouping':9s} {'classes':>7s} {'baseline':>9s} {'KRR':>7s} {'kNN':>7s} {'ceiling':>8s}")
    for row in summary_rows:
        print(f"{row['dataset']:5s} {row['label']:12s} {row['grouping']:9s} {row['n_classes']:7d} "
              f"{row['baseline_majority']:9.3f} {row['krr_accuracy']:7.3f} {row['knn_accuracy']:7.3f} "
              f"{row['ceiling_from_coinciding_vectors']:8.3f}")
    print(f"tables in {out} ({time.time() - started:.0f} s)")
    return 0


def _mean(values):
    return round(float(values.mean()), 4) if len(values) else ""


if __name__ == "__main__":
    sys.exit(main())
