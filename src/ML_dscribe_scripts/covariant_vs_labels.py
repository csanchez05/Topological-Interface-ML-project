#!/usr/bin/env python3
"""Covariant SOAP vs residual-symmetry labels: which input predicts a GB property, with the same model,
rows and folds (SOAP guide Stage 1 follow-up, preparing Stages 2-3).

Inputs (--inputs, default all six):
  cov1           covariant SOAP, lambda >= 1 blocks: polar, axial (lambda 1), tensor, pseudotensor (lambda 2)
  cov01          cov1 plus the lambda = 0 (invariant) block
  cov1+labels    cov1 plus the label block, its weight chosen in the inner cross-validation
  cov01+labels   the same for cov01
  labels         0/1 columns: plane, handedness, lattice, motif, the four symmetry flags and the space
                 group (labels/gb_symmetry.csv, from gb_symmetry.py)
  plane+class    plane and grain class only

Covariant vectors: the reference run rc6.0_n8_l6_s0.5 (made as ..._lam2), interface radius 8 A, 8x
cells. The edge and mid interfaces are separate vectors; they are never averaged.

Common scale. Each block (the five covariant blocks and the label block) is brought to unit total
variance on the outer training rows: its squared distances are divided by
    v_b = sum over training pairs (i, j) of d_b(i, j)^2 / (2 n^2),
which is the sum of its column variances. The covariant blocks are then averaged, so each carries the
same share whatever its width (lambda = 2 holds 80% of the raw width):
    d_cov^2 = (1/B) sum_b d_b^2 / v_b,
and the labels are added with a weight chosen in the inner CV (w^2 = 0, 0.1, 0.3, 1, 3, 10; 0 falls
back to covariant alone):
    d^2 = d_cov^2 + w^2 d_L^2 / v_L.
Within a covariant block the columns are centred and, for scaling "standard", divided by their spread
on the training rows (krr_classify's choice, made in the inner CV). Columns whose spread is below
--std-floor x the block's largest are left unscaled, so that float32 noise in components that vanish
by symmetry is not blown up. Label columns are never rescaled. A categorical variable is one-hot / sqrt(2),
so two rows that differ in it are at squared distance 1, as for a flag.

Model and folds as krr_classify.py --regression: kernel exp(-gamma d^2 / median training d^2), gamma
in {0.1, 1, 10, 100}, alpha in {1e-6, 1e-4, 1e-2, 1}, chosen per target by the inner 3-fold squared
error; 5 outer folds (GroupKFold, shuffled, seed 0; inner seed 1 + fold). Groups are the census
distinct_id (Stage 1.5). Copies of one GB are turned in space, so their covariant vectors do not
coincide and krr_classify's grouping by coinciding vectors would miss them. Both interfaces of a GB
and all its copies stay in one fold, and the folds are the same for every input. D0985 (the plain-bulk
pair) is dropped. Scores: R^2 and MAE against the mean of the scored rows, beside a baseline that
predicts the training mean; pooled, per plane and (interface rows) per interface.

Rows (--rows):
  interface  one row per interface (edge, mid) of every family at 8x; the targets need a region column
  cell       one row per family that has a target (a GB-energy table may hold only a few). The kernel
             joins the two interface vectors of a cell and averages over swapping them,
               K = 1/2 [exp(-g s(ee', mm') / w) + exp(-g s(em', me') / w)],   s = mean of the two d^2,
             w = the median of s over training pairs (both matchings), so it does not depend on which
             interface is called edge; it stays positive semi-definite
Targets: any numeric columns of a table (--targets-table, --targets), joined on family (and region).

Known-answer checks (on failure the results are still written, run.json says "status": "fail", exit 1):
  1  plane+class R^2 <= the in-sample share of the plane x grain class group means on the scored rows
     + 0.02 (03_geometry_labels: about 0.19 at most); enforced on full runs
  2  cov1 is poor: its pooled R^2 on a geometry label <= the midpoint between phase 1's kNN R^2 of C
     and standard SOAP's KRR R^2 (both re-read from their files); enforced on full interface runs
  3  data path: phase 1's kNN R^2 of C (raw lambda >= 1, k = 10, 2,222 rows) is recomputed from the
     loaded vectors and must match the stored values to 1e-6; always run
Also reported: test interface rows whose lambda = 0 vector coincides with a training row of another GB
(an interface shared by different distinct GBs), per fold.

Output: calculations/ML_dscribe/CoGe/comparison/covariant_vs_labels/<rows>_<targets table stem>/
  results.csv      one row per input x target x scored_on
  run.json         settings, block variances, folds, chosen settings, checks, timing
  predictions.csv  out-of-fold predictions (--predictions)
An existing folder is refused unless --overwrite.

Run on a compute node (loading reads whole npz members, about 10 GB peak), never on the login node:
  sbatch src/ML_dscribe_scripts/covariant_vs_labels.slurm
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_labels as gl  # noqa: E402
import krr_classify as kc  # noqa: E402
import ml_common as mc  # noqa: E402
import representation_study as rs  # noqa: E402
from ml_common import sc  # noqa: E402

INPUTS = ("cov1", "cov01", "cov1+labels", "cov01+labels", "labels", "plane+class")
COVARIANT = {"cov1": rs.VECTORIAL, "cov01": rs.INVARIANT + rs.VECTORIAL}
FLAGS = ("inversion", "mirror_parallel", "twofold_normal", "twofold_inplane")
LABEL_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
OUT_ROOT = mc.COMPARISON_ROOT / "covariant_vs_labels"
SYMMETRY_TABLE = mc.LABELS_DIR / "gb_symmetry.csv"
TARGETS_TABLE = mc.LABELS_DIR / gl.CSV_NAME
GEOMETRY_JSON = mc.LABELS_DIR / gl.JSON_NAME
PHASE1_DIR = mc.COMPARISON_ROOT / "representation_study" / "phase1"
STANDARD_KRR = mc.OUTPUT_ROOT / mc.DEFAULT_DESCRIPTOR / "interface_r8_both_t4_no_boundary" / "krr_regression.csv"
PLANE_CLASS_TOL = 0.02
PHASE1_TOL = 1e-6
COINCIDE = 1e-4  # relative distance below which two interface vectors are the same interface (as Stage 1.5)


# --------------------------------------------------------------------------- tables

def catalogue(symmetry_path, limit=0):
    """Families of the cross-validation: the census without D0985, with their symmetry labels.

    limit: the first `limit` sym folders per plane (smoke tests)."""
    import pandas as pd

    census = pd.read_csv(mc.CENSUS, dtype={"plane": str})
    table = census.loc[~census["family"].isin(mc.NO_BOUNDARY), ["family", "plane", "sym", "distinct_id"]]
    table = table.sort_values("family")
    if limit:
        table = table.groupby("plane", sort=True).head(limit)
    symmetry = pd.read_csv(symmetry_path, dtype={"plane": str})
    table = table.merge(symmetry.drop(columns=["plane", "sym"]), on="family", how="left", validate="one_to_one")
    missing = table["space_group"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} families have no symmetry labels in {symmetry_path}, e.g. "
                         f"{table.loc[missing, 'family'].head(3).tolist()}; run gb_symmetry.py")
    return table.sort_values("family").reset_index(drop=True)


def load_targets(path, names, rows_mode, thickness):
    """(targets table restricted to the join keys and the named columns, keys)."""
    import pandas as pd

    table = pd.read_csv(path, dtype={"plane": str})
    absent = [c for c in ["family", *names] if c not in table]
    if absent:
        raise ValueError(f"{path} has no column(s) {absent}")
    if "thickness" in table:
        table = table[table["thickness"] == thickness]
    if rows_mode == "interface":
        if "region" not in table:
            raise ValueError(f"{path}: interface rows need a region column (edge, mid)")
        table, keys = table[table["region"].isin(sc.PLANE_NAMES)], ["family", "region"]
    else:
        if "region" in table:
            table = table[table["region"] == "cell"]
        keys = ["family"]
    table = table[keys + list(names)].copy()
    duplicated = table.duplicated(keys)
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate rows in {path}, e.g. "
                         f"{table.loc[duplicated, keys].head(3).to_dict('records')}")
    for name in names:
        table[name] = pd.to_numeric(table[name], errors="raise").astype(float)
    if table[list(names)].isna().any().any():
        raise ValueError(f"{path}: empty target values")
    return table, keys


def build_rows(families, targets, keys, names, rows_mode, limited):
    """Rows of the cross-validation, in family order (interface rows: edge then mid of each family)."""
    if rows_mode == "interface":
        rows = families.loc[families.index.repeat(2)].reset_index(drop=True)
        rows["region"] = np.tile(sc.PLANE_NAMES, len(families))
        rows = rows.merge(targets, on=keys, how="left", validate="one_to_one")
        missing = rows[list(names)].isna().any(axis=1)
        if missing.any():
            raise ValueError(f"no target for {int(missing.sum())} interface rows, e.g. "
                             f"{rows.loc[missing, keys].head(3).to_dict('records')}")
        return rows
    unknown = sorted(set(targets["family"]) - set(families["family"]) - set(mc.NO_BOUNDARY))
    if unknown and not limited:
        raise ValueError(f"{len(unknown)} target families are not in the catalogue, e.g. {unknown[:3]}")
    rows = families.merge(targets, on="family", how="inner", validate="one_to_one")
    if len(rows) < 10:
        raise ValueError(f"only {len(rows)} cells have a target")
    return rows.reset_index(drop=True)


def label_vectors(table, kind, space_group_column="space_group"):
    """(matrix, column names) of the label block. kind: 'labels' or 'plane+class'."""
    columns, names = [], []

    def one_hot(values, prefix):
        values = np.asarray(values).astype(str)
        for level in sorted(set(values)):
            columns.append((values == level) / np.sqrt(2))
            names.append(f"{prefix}={level}")

    def flag(values, name):
        columns.append(np.asarray(values, dtype=float))
        names.append(name)

    one_hot(table["plane"], "plane")
    if kind == "plane+class":
        one_hot(table["grain_class"], "grain_class")
    elif kind == "labels":
        flag(table["handedness"] == "opposite", "opposite_handed")
        flag(table["lattice"] == "twinned", "twinned_lattice")
        flag(table["motif"] == "turned", "turned_motif")
        for name in FLAGS:
            if not set(np.unique(table[name])) <= {0, 1}:
                raise ValueError(f"column {name} is not 0/1")
            flag(table[name], name)
        one_hot(table[space_group_column].astype(int), "space_group")
    else:
        raise ValueError(f"unknown label input {kind!r}")
    return np.column_stack(columns).astype(np.float64), names


def group_share(y, groups):
    """In-sample share of the variance of y explained by the means of the groups (eta^2)."""
    y = np.asarray(y, dtype=float)
    total = float(np.square(y - y.mean()).sum())
    if total <= 0:
        return float("nan")
    within = sum(float(np.square(y[groups == g] - y[groups == g].mean()).sum()) for g in np.unique(groups))
    return round(1.0 - within / total, 4)


# --------------------------------------------------------------------------- distances

def block_distances(X, train, scale, floor):
    """(squared distances of all rows, number of floored columns) after centring on the training rows and,
    for 'standard', dividing each column by its training spread unless that spread is below floor x the
    block's largest."""
    Z = np.array(X, dtype=np.float64)
    Z -= Z[train].mean(axis=0)
    floored = 0
    if scale == "standard":
        spread = Z[train].std(axis=0)
        small = spread <= floor * (spread.max() if spread.size else 0.0)
        floored = int(small.sum())
        Z /= np.where(small, 1.0, spread)
    elif scale != "none":
        raise ValueError(f"unknown scale {scale!r}")
    return kc.squared_distances(Z), floored


def total_variance(D2, train):
    """Sum of the column variances of the training rows, from their squared distances."""
    n = len(train)
    value = float(D2[np.ix_(train, train)].sum()) / (2.0 * n * n)
    return value if value > 0 else 1.0


def matchings(M, rows_mode):
    """Distance parts over the CV rows: [M] for interface rows; for cells (rows 2f, 2f + 1 = edge, mid of
    family f) the two matchings s(ee', mm') and s(em', me'), each the mean of its two squared distances."""
    if rows_mode == "interface":
        return [M]
    e, m = slice(0, None, 2), slice(1, None, 2)
    return [(M[e, e] + M[m, m]) / 2.0, (M[e, m] + M[m, e]) / 2.0]


def interface_train(train, rows_mode):
    return train if rows_mode == "interface" else np.sort(np.concatenate([2 * train, 2 * train + 1]))


def fold_matrices(blocks, labels, train, scales, floor, rows_mode, wanted):
    """Normalised distance parts of one outer fold.

    blocks: {block: interface-row matrix}; labels: {kind: CV-row matrix}; wanted: covariant inputs needed.
    Returns ({scale: {cov input: parts}}, {kind: parts}, info)."""
    rows = interface_train(train, rows_mode)
    cov, info = {}, {}
    if wanted:
        for scale in scales:
            totals, info[scale] = {}, {}
            for block, X in blocks.items():
                if not any(block in COVARIANT[name] for name in wanted):
                    continue
                D2, floored = block_distances(X, rows, scale, floor)
                variance = total_variance(D2, rows)
                D2 /= variance
                info[scale][block] = {"total_variance": variance, "floored_columns": floored}
                for name in wanted:
                    if block in COVARIANT[name]:
                        share = D2 / len(COVARIANT[name])
                        totals[name] = share if name not in totals else totals[name] + share
                del D2
            cov[scale] = {name: matchings(totals[name], rows_mode) for name in wanted}
    label_parts = {}
    for kind, L in labels.items():
        D2 = kc.squared_distances(L)
        variance = total_variance(D2, train)
        D2 /= variance
        info.setdefault("labels", {})[kind] = {"total_variance": variance}
        label_parts[kind] = [D2] if rows_mode == "interface" else [D2, D2]
    return cov, label_parts, info


def combined(parts, label, w2):
    if label is None or w2 == 0:
        return parts
    return [M + w2 * L for M, L in zip(parts, label)]


def pooled_width(parts, rows):
    """krr_classify's median training d^2; for two matchings, the median over both."""
    if len(parts) == 1:
        return kc.median_squared_distance(parts[0], rows)
    upper = np.triu(np.ones((len(rows), len(rows)), dtype=bool), 1)
    values = np.concatenate([M[np.ix_(rows, rows)][upper] for M in parts])
    values = values[values > 0]
    return float(np.median(values)) if values.size else 1.0


def kernel(parts, gamma, width):
    K = kc.kernel_matrix(parts[0], gamma, width)
    if len(parts) == 2:
        K += kc.kernel_matrix(parts[1], gamma, width)
        K *= 0.5
    return K


# --------------------------------------------------------------------------- nested cross-validation

def kernel_fold(parts, label, Y, train, test, inner, gammas, alphas, weights):
    """krr_classify.regression_fold()'s kernel ridge regression on precomputed distance parts.

    parts: {scale: [matrix, ...]} over all CV rows; label: label parts or None; weights: label weights w^2.
    Settings (scale, w^2, gamma, alpha) are chosen per target by the inner-fold squared error, then
    refitted on the whole training set. Returns (predictions of the test rows, entry)."""
    weights = sorted(weights) if label is not None else [0.0]
    grid = [(scale, w2, gamma, alpha) for scale in parts for w2 in weights for gamma in sorted(gammas)
            for alpha in sorted(alphas, reverse=True)]
    position = {setting: p for p, setting in enumerate(grid)}
    error, spread = np.zeros((len(grid), Y.shape[1])), np.zeros(Y.shape[1])
    splits = [(train[fit], train[check]) for fit, check in inner]
    for fit, check in splits:
        spread += np.square(Y[check] - Y[fit].mean(axis=0)).sum(axis=0)
    widths = {}
    for scale in parts:
        for w2 in weights:
            M = combined(parts[scale], label, w2)
            widths[(scale, w2)] = pooled_width(M, train)
            for gamma in sorted(gammas):
                K = kernel(M, gamma, widths[(scale, w2)])
                for fit, check in splits:
                    K_fit, K_check = K[np.ix_(fit, fit)], K[np.ix_(check, fit)]
                    for alpha in sorted(alphas, reverse=True):
                        guess = kc.krr_scores(K_fit, Y[fit], K_check, alpha)
                        error[position[(scale, w2, gamma, alpha)]] += np.square(guess - Y[check]).sum(axis=0)
                del K
    best = [grid[int(np.argmin(error[:, c]))] for c in range(Y.shape[1])]
    predicted = np.full((len(test), Y.shape[1]), np.nan)
    for setting in sorted(set(best)):
        scale, w2, gamma, alpha = setting
        K = kernel(combined(parts[scale], label, w2), gamma, widths[(scale, w2)])
        guess = kc.krr_scores(K[np.ix_(train, train)], Y[train], K[np.ix_(test, train)], alpha)
        for c, chosen in enumerate(best):
            if chosen == setting:
                predicted[:, c] = guess[:, c]
        del K
    entry = []
    for c, (scale, w2, gamma, alpha) in enumerate(best):
        inner_r2 = 1.0 - error[:, c].min() / spread[c] if spread[c] > 0 else float("nan")
        entry.append({"scale": scale, "label_weight": w2, "gamma_times_median": gamma, "alpha": alpha,
                      "inner_r2": float(inner_r2), "median_d2": widths[(scale, w2)]})
    return predicted, entry


def input_parts(name, cov, labels, weights):
    """(parts by scale, label parts, label weights) of one input."""
    if name in COVARIANT:
        return {scale: cov[scale][name] for scale in cov}, None, (0.0,)
    if name.endswith("+labels"):
        base = name[: -len("+labels")]
        return {scale: cov[scale][base] for scale in cov}, labels["labels"], tuple(weights)
    return {"none": labels[name]}, None, (0.0,)


def coinciding_pairs(X, groups, tol=COINCIDE):
    """Pairs of interface rows of different groups whose vectors coincide (relative distance <= tol)."""
    Z = np.asarray(X, dtype=np.float64)
    norms = np.sqrt(np.einsum("ij,ij->i", Z, Z))
    distance = np.sqrt(kc.squared_distances(Z))
    scale = (norms[:, None] + norms[None, :]) / 2.0
    same = distance <= tol * np.where(scale > 0, scale, 1.0)
    return same & (groups[:, None] != groups[None, :])


def make_folds(groups, n_outer, n_inner, seed):
    """krr_classify.start_nested_cv()'s folds: [(number, train, test, inner)], built once for every input."""
    return [(number, train, test, kc.group_folds(groups[train], n_inner, seed + 1 + number))
            for number, (train, test) in enumerate(kc.group_folds(groups, n_outer, seed))]


def run_cv(blocks, labels, Y, groups, folds, rows_mode, inputs, args, shared=None, log=print):
    """Nested CV of every input on the same folds. Returns (predictions, baseline, fold_of, chosen, info)."""
    wanted = sorted({name.split("+")[0] for name in inputs if name.split("+")[0] in COVARIANT})
    label_kinds = {k: v for k, v in labels.items() if k in inputs or (k == "labels" and any("+labels" in i for i in inputs))}
    predictions = {name: np.full(Y.shape, np.nan) for name in inputs}
    baseline, fold_of = np.full(Y.shape, np.nan), np.full(len(Y), -1)
    chosen, info = {name: [] for name in inputs}, []
    for number, train, test, inner in folds:
        started = time.time()
        cov, label_parts, fold_info = fold_matrices(blocks, label_kinds, train, args.scales, args.std_floor,
                                                    rows_mode, wanted)
        fold_info = {"fold": number, "n_train": len(train), "n_test": len(test), "blocks": fold_info}
        if shared is not None:
            hit = shared[np.ix_(test, train)].any(axis=1)
            fold_info["test_rows_sharing_an_interface_with_training"] = int(hit.sum())
        for name in inputs:
            parts, label, weights = input_parts(name, cov, label_parts, args.label_weights)
            predicted, entry = kernel_fold(parts, label, Y, train, test, inner, args.gammas, args.alphas, weights)
            predictions[name][test] = predicted
            chosen[name].append(entry)
        baseline[test] = Y[train].mean(axis=0)
        fold_of[test] = number
        fold_info["seconds"] = round(time.time() - started, 1)
        info.append(fold_info)
        del cov, label_parts
        log(f"  fold {number + 1}/{len(folds)}: {len(train)} train, {len(test)} test, {fold_info['seconds']} s")
    return predictions, baseline, fold_of, chosen, info


# --------------------------------------------------------------------------- scores and checks

def subsets(rows, rows_mode):
    out = {"all": np.ones(len(rows), dtype=bool)}
    for plane in sc.PLANES:
        mask = (rows["plane"] == plane).to_numpy()
        if mask.any():
            out[plane] = mask
    if rows_mode == "interface":
        for region in sc.PLANE_NAMES:
            out[region] = (rows["region"] == region).to_numpy()
    return out


def usable_fold(values, reference, rows=5, share=1e-6):
    """A fold's R^2 is reported only with enough rows and spread; a near-constant fold gives a
    meaningless huge negative R^2 (seen in the smoke test: -6e27)."""
    return len(values) >= rows and float(np.var(values)) > share * float(np.var(reference))


def joined(values):
    return "/".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


def score_table(rows, rows_mode, names, Y, predictions, baseline, fold_of, chosen):
    import pandas as pd

    plane_class = (rows["plane"].astype(str) + "|" + rows["grain_class"].astype(str)).to_numpy()
    records = []
    for name, predicted in predictions.items():
        for c, target in enumerate(names):
            picks = [entry[c] for entry in chosen[name]]
            for subset, mask in subsets(rows, rows_mode).items():
                true = Y[mask, c]
                krr_r2, krr_mae = kc.regression_scores(true, predicted[mask, c])
                base_r2, base_mae = kc.regression_scores(true, baseline[mask, c])
                per_fold = [kc.regression_scores(Y[rows_in, c], predicted[rows_in, c])[0]
                            for rows_in in (mask & (fold_of == f) for f in np.unique(fold_of))
                            if usable_fold(Y[rows_in, c], true)]
                records.append({"rows": rows_mode, "input": name, "target": target, "scored_on": subset,
                                "n": int(mask.sum()), "mean": round(float(true.mean()), 6),
                                "std": round(float(true.std()), 6), "krr_r2": krr_r2, "krr_mae": krr_mae,
                                "baseline_r2": base_r2, "baseline_mae": base_mae,
                                "n_folds_scored": len(per_fold),
                                "r2_fold_min": float(np.nanmin(per_fold)) if per_fold else float("nan"),
                                "r2_fold_max": float(np.nanmax(per_fold)) if per_fold else float("nan"),
                                "eta2_plane_class": group_share(true, plane_class[mask]),
                                "scale": joined(p["scale"] for p in picks),
                                "label_weight": joined(p["label_weight"] for p in picks),
                                "gamma": joined(p["gamma_times_median"] for p in picks),
                                "alpha": joined(p["alpha"] for p in picks)})
    return pd.DataFrame.from_records(records)


def check_plane_class(results, enforce):
    rows = results[(results["input"] == "plane+class") & results["scored_on"].isin(["all", *sc.PLANE_NAMES])]
    bad = rows[rows["krr_r2"] > rows["eta2_plane_class"] + PLANE_CLASS_TOL]
    stored = {}
    if GEOMETRY_JSON.exists():
        shares = json.loads(GEOMETRY_JSON.read_text()).get("variance_explained_by_plane_and_grain_class", {})
        stored = {region: {t: v["plane_and_grain_class"] for t, v in per.items() if isinstance(v, dict)}
                  for region, per in shares.items() if isinstance(per, dict)}
    return {"pass": bool(bad.empty) or not enforce, "enforced": enforce, "tolerance": PLANE_CLASS_TOL,
            "rule": "plane+class krr_r2 <= eta2_plane_class (in-sample, same rows) + tolerance",
            "rows": rows[["target", "scored_on", "krr_r2", "eta2_plane_class"]].to_dict("records"),
            "failing": bad[["target", "scored_on", "krr_r2", "eta2_plane_class"]].to_dict("records"),
            "stored_in_sample_shares_03_geometry_labels": stored}


def phase1_values(variable_prefix="knn_r2_"):
    import pandas as pd

    table = pd.read_csv(PHASE1_DIR / "results.csv", dtype={"subset": str, "phase": str})
    rows = table[(table["kind"] == "raw") & (table["method"] == "raw") & (table["subset"] == "pooled")
                 & (table["representation"] == "C") & table["variable"].str.startswith(variable_prefix)]
    return {v[len(variable_prefix):]: float(x) for v, x in zip(rows["variable"], rows["value"])}


def standard_krr_values():
    import pandas as pd

    table = pd.read_csv(STANDARD_KRR, dtype={"scored_on": str})
    rows = table[(table["dataset"] == "all") & (table["grouping"] == "distinct") & (table["scored_on"] == "all")]
    return dict(zip(rows["target"], rows["krr_r2"].astype(float)))


def check_cov1(results, enforce):
    knn, krr = phase1_values(), standard_krr_values()
    rows, ok = [], True
    for target in gl.TARGETS:
        got = results[(results["input"] == "cov1") & (results["target"] == target) & (results["scored_on"] == "all")]
        if got.empty or target not in knn or target not in krr:
            continue
        midpoint = (knn[target] + krr[target]) / 2.0
        value = float(got["krr_r2"].iloc[0])
        ok &= value <= midpoint
        rows.append({"target": target, "cov1_krr_r2": value, "phase1_knn_r2_C": round(knn[target], 4),
                     "standard_soap_krr_r2": krr[target], "midpoint": round(midpoint, 4), "pass": value <= midpoint})
    return {"pass": bool(ok) or not enforce, "enforced": enforce and bool(rows),
            "rule": "cov1 pooled krr_r2 <= (phase 1 kNN R^2 of C + standard SOAP KRR R^2) / 2", "rows": rows}


def check_phase1(vectors, family_position, labels_path=None):
    """Recompute phase 1's kNN R^2 of C on its own rows, from the vectors loaded here."""
    summary = json.loads((PHASE1_DIR / "summary.json").read_text())
    k = int(summary["settings"]["organise_k"])
    gbs = rs.gb_table(available=list(family_position))
    rows = rs.interface_rows(gbs)
    positions = [family_position[family] for family, _ in rs.structure_keys(rows)]
    C = np.concatenate([vectors[block][positions] for block in rs.VECTORIAL], axis=-1)
    C = C.reshape(-1, C.shape[-1]).astype(np.float64)
    Z = rs.exact_coordinates(C)
    del C
    Z = rs.exact_coordinates(Z) if Z.shape[1] > len(Z) else Z - Z.mean(axis=0)
    D = rs.square_distances(Z)
    geometry = gl.load_geometry(rows.assign(region=rows["interface"]), path=labels_path)
    got = rs.organisation(D, rs.partners(len(rows)), rs.label_codes(rows, geometry), k, True)
    stored = phase1_values()
    out = {}
    for target in gl.TARGETS:
        value, reference = got.get(f"knn_r2_{target}"), stored.get(target)
        out[target] = {"recomputed": value, "stored": reference,
                       "difference": None if value is None or reference is None else abs(value - reference)}
    ok = all(v["difference"] is not None and v["difference"] <= PHASE1_TOL for v in out.values())
    return {"pass": ok, "tolerance": PHASE1_TOL, "n_rows": len(rows), "k": k, "values": out,
            "expected_rows": summary.get("n_rows")}


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", choices=["interface", "cell"], default="interface")
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=list(INPUTS))
    parser.add_argument("--targets-table", type=Path, default=TARGETS_TABLE)
    parser.add_argument("--targets", nargs="+", default=list(gl.TARGETS), help="columns of the targets table")
    parser.add_argument("--symmetry-table", type=Path, default=SYMMETRY_TABLE)
    parser.add_argument("--space-group-column", default="space_group",
                        help="space-group column of the symmetry table used as labels (e.g. space_group_5e-3)")
    parser.add_argument("--covariant-run", default=mc.DEFAULT_DESCRIPTOR)
    parser.add_argument("--covariant-root", type=Path, default=sc.COVARIANT_ROOT)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--thickness", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1.0])
    parser.add_argument("--scales", nargs="+", choices=["standard", "none"], default=["standard", "none"])
    parser.add_argument("--label-weights", type=float, nargs="+", default=list(LABEL_WEIGHTS),
                        help="label weights w^2 tried in the inner CV (keep 0 so it can fall back)")
    parser.add_argument("--std-floor", type=float, default=1e-5,
                        help="'standard' scaling leaves a column unscaled when its spread is below this x the block's largest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: the first N sym folders per plane")
    parser.add_argument("--predictions", action="store_true", help="also write the out-of-fold predictions")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def fold_digest(folds, n):
    assignment = np.full(n, -1)
    for number, _, test, _ in folds:
        assignment[test] = number
    return hashlib.sha256(assignment.astype(np.int64).tobytes()).hexdigest()[:16]


def main(argv=None):
    args = parse_args(argv)
    args.label_weights = sorted(set(args.label_weights))
    started = time.time()
    import pandas as pd

    out_dir = args.out_dir or OUT_ROOT / f"{args.rows}_{Path(args.targets_table).stem}"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.dry_run:
        print(f"refused: {mc.relative_path(out_dir)} exists (use --overwrite)", file=sys.stderr)
        return 2
    families = catalogue(args.symmetry_table, args.limit)
    targets, keys = load_targets(args.targets_table, args.targets, args.rows, args.thickness)
    rows = build_rows(families, targets, keys, args.targets, args.rows, bool(args.limit))
    cv_families = rows["family"].drop_duplicates().tolist()
    groups = rows["distinct_id"].to_numpy()
    print(f"{args.rows} rows: {len(rows)} ({len(cv_families)} families, {len(set(groups))} distinct GBs), "
          f"targets {args.targets}, inputs {args.inputs}; output {mc.relative_path(out_dir)}", flush=True)
    if args.dry_run:
        return 0

    # ---- vectors: every 8x family is loaded (npz members are read whole anyway); check 3 needs 1,111 of them
    census = pd.read_csv(mc.CENSUS, dtype={"plane": str})
    load = sorted(set(census["family"]) - set(mc.NO_BOUNDARY))
    loaded = rs.covariant_vectors(args.covariant_run, [(f, args.thickness) for f in load], rs.INVARIANT + rs.VECTORIAL,
                                  radius=args.radius, root=args.covariant_root, dtype=np.float32)
    interface, columns = loaded["interface"], loaded["columns"]
    del loaded
    vectors = {block: np.ascontiguousarray(interface[:, :, lo:hi]) for block, (lo, hi) in columns.items()}
    del interface
    family_position = {f: i for i, f in enumerate(load)}
    print(f"loaded {len(load)} families, widths {{{', '.join(f'{b}: {v.shape[-1]}' for b, v in vectors.items())}}} "
          f"({time.time() - started:.0f} s)", flush=True)

    checks = {"3_phase1_data_path": check_phase1(vectors, family_position)}
    print(f"check 3 (phase 1 data path): {'pass' if checks['3_phase1_data_path']['pass'] else 'FAIL'} "
          f"{ {t: v['difference'] for t, v in checks['3_phase1_data_path']['values'].items()} }", flush=True)

    cv_position = np.array([family_position[f] for f in cv_families])
    blocks = {block: v[cv_position].reshape(-1, v.shape[-1]) for block, v in vectors.items()}
    del vectors
    # labels are per CV row: interface rows repeat their family's
    encoded = {kind: label_vectors(rows, kind, args.space_group_column) for kind in ("labels", "plane+class")}
    labels = {kind: matrix for kind, (matrix, _) in encoded.items()}
    label_columns = {kind: names for kind, (_, names) in encoded.items()}
    Y = rows[args.targets].to_numpy(dtype=float)
    folds = make_folds(groups, args.folds, args.inner_folds, args.seed)
    if args.rows == "interface":
        same_family = rows["family"].to_numpy()
        for _, train, test, _ in folds:
            if np.intersect1d(same_family[train], same_family[test]).size:
                raise RuntimeError("the two interfaces of a family were split")
    shared = coinciding_pairs(blocks["invariant"], groups) if args.rows == "interface" else None
    predictions, baseline, fold_of, chosen, info = run_cv(blocks, labels, Y, groups, folds, args.rows, args.inputs,
                                                          args, shared=shared)
    results = score_table(rows, args.rows, args.targets, Y, predictions, baseline, fold_of, chosen)
    full = not args.limit
    checks["1_plane_class_at_most_in_sample_share"] = check_plane_class(results, enforce=full) \
        if "plane+class" in args.inputs else {"pass": True, "enforced": False}
    checks["2_cov1_poor_midpoint_rule"] = check_cov1(results, enforce=full and args.rows == "interface") \
        if "cov1" in args.inputs else {"pass": True, "enforced": False}
    status = "pass" if all(c["pass"] for c in checks.values()) else "fail"

    out_dir.mkdir(parents=True, exist_ok=True)
    mc.write_table(out_dir / "results.csv", results)
    if args.predictions:
        records = []
        for name, predicted in predictions.items():
            for c, target in enumerate(args.targets):
                records.append(pd.DataFrame({"family": rows["family"], **({"region": rows["region"]} if "region" in rows else {}),
                                             "fold": fold_of, "input": name, "target": target, "true": Y[:, c],
                                             "predicted": predicted[:, c], "baseline": baseline[:, c]}))
        mc.write_table(out_dir / "predictions.csv", pd.concat(records, ignore_index=True))
    payload = {"what": "covariant SOAP vs residual-symmetry labels as inputs for the same targets (KRR, nested CV)",
               "status": status, "rows": args.rows, "n_rows": len(rows), "n_families": len(cv_families),
               "n_distinct_gbs": int(len(set(groups))), "limited_to_first_syms_per_plane": args.limit or None,
               "targets_table": mc.relative_path(args.targets_table), "targets": args.targets,
               "symmetry_table": mc.relative_path(args.symmetry_table), "space_group_column": args.space_group_column,
               "covariant_run": mc.relative_path(rs.covariant_dirs(args.covariant_run, args.covariant_root)[0]),
               "radius_A": args.radius, "thickness": args.thickness,
               "inputs": {name: {"covariant_blocks": list(COVARIANT.get(name.split("+")[0], ())),
                                 "labels": "labels" if name.endswith("+labels") or name == "labels" else
                                 ("plane+class" if name == "plane+class" else None)} for name in args.inputs},
               "block_widths": {block: int(X.shape[1]) for block, X in blocks.items()},
               "label_columns": label_columns,
               "common_scale": "each block / its training total variance (sum of column variances); covariant blocks "
                               "averaged; label block added with weight w^2 chosen in the inner CV",
               "settings": {"folds": args.folds, "inner_folds": args.inner_folds, "seed": args.seed,
                            "gammas": args.gammas, "alphas": args.alphas, "scales": args.scales,
                            "label_weights": args.label_weights, "std_floor": args.std_floor,
                            "grouping": "census distinct_id (Stage 1.5)", "no_boundary_dropped": list(mc.NO_BOUNDARY)},
               "folds": {"sizes": [len(test) for _, _, test, _ in folds], "assignment_sha256_16": fold_digest(folds, len(rows))},
               "fold_info": info, "chosen": chosen,
               "test_rows_sharing_an_interface_with_training": None if shared is None else
               int(sum(f.get("test_rows_sharing_an_interface_with_training", 0) for f in info)),
               "checks": checks, "seconds": round(time.time() - started, 1), "run": mc.run_metadata()}
    mc.write_json(out_dir / "run.json", payload)
    for name, check in checks.items():
        print(f"  check {name}: {'pass' if check['pass'] else 'FAIL'}{'' if check.get('enforced', True) else ' (not enforced)'}")
    pooled = results[results["scored_on"] == "all"].pivot(index="input", columns="target", values="krr_r2")
    print(pooled.reindex(args.inputs).to_string())
    print(f"status {status}; wrote {mc.relative_path(out_dir)} ({time.time() - started:.0f} s)")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
