#!/usr/bin/env python3
"""Fifth and sixth descriptor studies: how much does the kind of representation change the maps?

Fifth study (--study representation). The SOAP accuracy settings stay at the reference
(r_cut 6 A, n_max 8, l_max 6, sigma 0.5 A, no weighting); only the kind of representation
changes, in a 2 x 2 design:

  A   standard SOAP, atoms within 8 A of one interface     standard/both_interfaces/<tag>, X_r8[:, edge|mid]
  B   standard SOAP, whole-cell mean                       standard/global/<tag>
  C   covariant SOAP, lambda = 1 and 2 (polar, axial, tensor, pseudotensor), atoms within 8 A
      of one interface                                      covariant/both_interfaces/<run>
  D   covariant SOAP, lambda = 1 and 2, whole-cell mean    covariant/global/<run>
  C0  C plus the lambda = 0 block: a one-off check in phase 1 (pooled rows), outside the 2 x 2 design

Rows: one per interface, edge and mid kept apart. These are the 1,111 distinct GBs at 8x (the
Stage 1.5 census without D0985, ml_common.NO_BOUNDARY) x 2 = 2,222 rows. A whole-cell vector gives
both rows of a GB the same vector. So the neighbour-based scores (kNN, Jaccard, Spearman,
trustworthiness) leave out the other interface of the same GB, in every representation, and compare
how each representation arranges *different* GBs. Procrustes and Ward/ARI use all rows. There is no
feature scaling. Each method sees exact coordinates of the vectors (centred and projected on their own
span), so the pairwise distances are unchanged while memory and time drop.

Phases (each writes its own folder):
  1          bulk test (cell averages of lambda >= 1 for the bulk cells; D against the count-weighted
             sum of C; B against A plus bulk), then PCA and UMAP on A-D (+ C0)
  2          the other methods: kernel PCA, ICA, LDA, PLS, PCovR, metric MDS, Isomap, t-SNE,
             PaCMAP, diffusion maps, sketch-map
  thickness  every method on all 9 thicknesses of 300 GBs; PCA on every structure

For every method and subset (pooled, 100, 110, 111), each pair of representations is compared with:
  procrustes     Procrustes disparity of the 2D maps (0 = the same map after shifting, turning, scaling)
  jaccard<k>     mean Jaccard index of each row's k nearest rows (k = 15 and 50)
  spearman       rank correlation of pairwise distances, on a fixed random sample of row pairs
  ari            adjusted Rand index of the Ward clusters (8) of the two maps
"Change" means the disparity itself, or 1 - score for the others. Controls:
  noise floor    the same comparison between replicates of one representation: 10 seeds for the
                 stochastic methods, 10 subsamples of 80% of the GBs for the deterministic ones.
                 A change is significant if it is above the floor's 95th percentile.
  input change   jaccard and spearman between the raw descriptors; sensitivity = map change / input change
Each map also gets: what organises it (kNN accuracy for plane, grain class and edge/mid; kNN R^2
for the geometry labels), trustworthiness and continuity against its own input, and random-triplet
accuracy. The 2 x 2 effects (region, covariance, interaction) come from PERMANOVA on the
map-to-map changes of all replicate maps, and from a two-way ANOVA for the per-map variables.

Sixth study (--study settings_grid). PCA, metric MDS, UMAP and diffusion maps for 12 SOAP settings
(r_cut 5/6/8 x l_max 4/8 x sigma 0.3/0.5, n_max 8) x the 4 representations x 5 replicates, on
pooled rows. PERMANOVA splits the map-to-map changes into representation, settings, their
interaction and seed noise. The reference setting is an extra anchor, outside the split.

Outputs, one folder per run: calculations/ML_dscribe/CoGe/representation_study/<phase>/ or
calculations/ML_dscribe/CoGe/settings_grid/
  results.csv    tidy long table: study, phase, kind, method, subset, representation, variable,
                 value, sd, main, noise_floor, floor_median, input_change, sensitivity, significant,
                 factor, p_value, n
  maps.npz       the main 2D map of every method, subset and representation, and the row table
  summary.json   settings, row counts, bulk test, checks, timing
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import geometry_labels as gl
import ml_common as mc
from descriptor_sensitivity import THREAD_VARIABLES, thickness_share
from ml_common import sc

OUT_DIR = mc.OUTPUT_ROOT / "representation_study"
GRID_DIR = mc.OUTPUT_ROOT / "settings_grid"
SKETCHMAP = mc.PROJECT_ROOT / "src" / "third_party" / "sketchmap" / "bin" / "dimred"
REPRESENTATIONS = ("A", "B", "C", "D")
CHECK = "C0"
REGION = {"A": "interface", "B": "cell", "C": "interface", "D": "cell", "C0": "interface"}
COVARIANCE = {"A": "standard", "B": "standard", "C": "covariant", "D": "covariant", "C0": "covariant"}
INVARIANT = ("invariant",)
VECTORIAL = ("polar", "axial", "tensor", "pseudotensor")
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
PHASE_METHODS = {
    "1": ("pca", "umap"),
    "2": ("kpca", "ica", "lda_plane", "lda_grain", "pls_plane", "pls_grain", "pls_geometry",
          "pcovr_plane", "pcovr_grain", "pcovr_geometry", "mds", "isomap", "tsne", "pacmap",
          "diffusion", "sketchmap"),
}
ALL_METHODS = PHASE_METHODS["1"] + PHASE_METHODS["2"]
STOCHASTIC = {"ica", "mds", "tsne", "umap", "pacmap", "sketchmap"}
SUPERVISED = {"lda", "pls", "pcovr"}
SUBSETS = ("pooled",) + sc.PLANES
GRID_METHODS = ("pca", "mds", "umap", "diffusion")
GRID = {"r_cut": (5.0, 6.0, 8.0), "l_max": (4, 8), "sigma": (0.3, 0.5)}
REFERENCE_TAG = mc.DEFAULT_DESCRIPTOR
COLUMNS = ("study", "phase", "kind", "method", "subset", "representation", "variable", "value", "sd",
           "main", "noise_floor", "floor_median", "input_change", "sensitivity", "significant",
           "factor", "p_value", "n")


# --------------------------------------------------------------------------- rows

def first_syms(per_plane):
    """Families of the first `per_plane` sym folders of each plane (as --limit in the descriptor scripts)."""
    families = set()
    for plane in sc.PLANES:
        folders = sorted((sc.GB_ROOT / f"CoGe_{plane}_GBs" / "Interface").glob("sym[0-9]*"))[:per_plane]
        families.update(f"{plane}/{folder.name}" for folder in folders)
    return families


def gb_table(available=None, keep_no_boundary=False, per_plane=0):
    """One row per distinct GB (the Stage 1.5 census), in family order, numbered by column gb."""
    import pandas as pd

    census = pd.read_csv(mc.CENSUS, dtype={"plane": str})
    labels = pd.read_csv(mc.LABELS_DIR / "gb_labels.csv", dtype={"plane": str})
    gbs = census.sort_values("family").drop_duplicates("structure_class")[["family", "plane", "sym", "structure_class"]]
    gbs = gbs.merge(labels[["family", "grain_class", "handedness"]], on="family", validate="one_to_one")
    if not keep_no_boundary:
        gbs = gbs[~gbs["family"].isin(mc.NO_BOUNDARY)]
    if per_plane:
        gbs = gbs[gbs["family"].isin(first_syms(per_plane))]
    if available is not None:
        gbs = gbs[gbs["family"].isin(set(available))]
    gbs = gbs.sort_values("family").reset_index(drop=True)
    return gbs.assign(gb=np.arange(len(gbs)))


def interface_rows(gbs, thicknesses=(8,), skip=()):
    """Two rows (edge, mid) per GB and thickness, in GB, thickness, interface order.

    The other interface of a row's structure is therefore row ^ 1 (see partners()).
    skip: (family, thickness) pairs without a structure (the empty 100/sym00016 9x POSCAR).
    """
    import pandas as pd

    skip = set(skip)
    records = [dict(gb_row, thickness=t, interface=side)
               for gb_row in gbs.to_dict("records") for t in thicknesses if (gb_row["family"], t) not in skip
               for side in sc.PLANE_NAMES]
    return pd.DataFrame.from_records(records)


def partners(n):
    """The other interface of the same structure: rows 2s and 2s + 1."""
    return np.arange(n) ^ 1


def structure_keys(rows):
    """(family, thickness) of every structure, one per pair of rows."""
    return list(zip(rows["family"].iloc[::2], rows["thickness"].iloc[::2]))


def label_codes(rows, geometry=None):
    """Integer codes of the class labels and float arrays of the geometry labels, row order."""
    classes = {name: np.unique(rows[name].astype(str), return_inverse=True)[1]
               for name in ("plane", "grain_class", "interface")}
    values = {} if geometry is None else {name: geometry[name].to_numpy(dtype=np.float64) for name in gl.TARGETS}
    return {"classes": classes, "values": values, "gb": rows["gb"].to_numpy(), "plane": rows["plane"].to_numpy(str)}


# --------------------------------------------------------------------------- vectors

def positions(index, keys):
    lookup = {(f, int(t)): i for i, (f, t) in enumerate(zip(index["family"], index["thickness"]))}
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} structures missing from the run, e.g. {missing[:3]}")
    return np.array([lookup[key] for key in keys])


def species_mask(width):
    """True for the Co-centred half of one [Co mean | Ge mean] block."""
    return np.arange(width) < width // 2


def standard_vectors(tag, keys, radius=8.0, dtype=np.float64, root=sc.STANDARD_ROOT):
    """Representations A and B for the structures `keys`, plus what the bulk test needs.

    A: (2 S, 2F) edge and mid rows, interleaved; B: the whole-cell vector, repeated for both rows.
    """
    import pandas as pd

    folder = sc.run_dir("both_interfaces", tag, root=root)
    pos = positions(pd.read_csv(folder / "index.csv", dtype={"plane": str}), keys)
    key = sc.radius_key(radius)
    with np.load(folder / sc.INTERFACE_FILE) as data:
        X, n = data[f"X_{key}"][pos].astype(dtype), data[f"n_{key}"][pos]
    global_dir = sc.run_dir("global", tag, root=root)
    global_index = pd.read_csv(global_dir / "index.csv", dtype={"plane": str})
    gpos = positions(global_index, keys)
    with np.load(global_dir / sc.GLOBAL_FILE) as data:
        G = data["X"][gpos].astype(dtype)
    X_bulk, bulk_index = sc.load_bulk(global_dir)
    bulk = X_bulk[(bulk_index["kind"] == "pristine").to_numpy()][0].astype(np.float64)
    return {"A": X.reshape(-1, X.shape[-1]), "B": np.repeat(G, 2, axis=0), "interface": X, "cell": G,
            "n_interface": n, "n_species": global_index.loc[gpos, ["n_Co", "n_Ge"]].to_numpy(),
            "bulk": bulk, "is_co": species_mask(X.shape[-1])}


def covariant_dirs(run, root=sc.COVARIANT_ROOT):
    return sc.run_dir("both_interfaces", run, root=root), sc.run_dir("global", run, root=root)


def covariant_vectors(run, keys, blocks, radius=8.0, root=sc.COVARIANT_ROOT, dtype=np.float64):
    """Covariant interface (C) and whole-cell (D) rows: the chosen blocks flattened and joined.

    Each block (N, [plane,] components, 2P) is flattened component by component, so every component
    keeps its [Co mean | Ge mean] halves. Also returns the column range of every block.
    """
    import pandas as pd

    interface_dir, global_dir = covariant_dirs(run, root)
    index = pd.read_csv(interface_dir / "index.csv", dtype={"plane": str})
    pos = positions(index, keys)
    key = sc.radius_key(radius)
    parts, cell_parts, is_co, columns, start = [], [], [], {}, 0
    with np.load(interface_dir / "covariant_soap.npz") as data:
        n = data[f"n_{key}"][pos]
        for block in blocks:
            values = data[f"X_{block}_{key}"][pos].astype(dtype)
            parts.append(values.reshape(len(pos), 2, -1))
            is_co.append(np.tile(species_mask(values.shape[-1]), values.shape[-2]))
            columns[block] = (start, start + parts[-1].shape[-1])
            start = columns[block][1]
    global_index = pd.read_csv(global_dir / "index.csv", dtype={"plane": str})
    gpos = positions(global_index, keys)
    with np.load(global_dir / "covariant_soap.npz") as data:
        for block in blocks:
            cell_parts.append(data[f"X_{block}"][gpos].astype(dtype).reshape(len(gpos), -1))
    X, G = np.concatenate(parts, axis=-1), np.concatenate(cell_parts, axis=-1)
    if X.shape[-1] != G.shape[-1]:
        raise ValueError("covariant interface and whole-cell vectors differ in length")
    return {"C": X.reshape(-1, X.shape[-1]), "D": np.repeat(G, 2, axis=0), "interface": X, "cell": G,
            "n_interface": n, "n_species": global_index.loc[gpos, ["n_Co", "n_Ge"]].to_numpy(),
            "is_co": np.concatenate(is_co), "columns": columns}


def block_shares(X, columns):
    """Share of the total variance (sum of column variances) carried by each block."""
    variance = X.var(axis=0)
    total = variance.sum()
    return {block: float(variance[a:b].sum() / total) for block, (a, b) in columns.items()}


def exact_coordinates(X, tol=1e-12):
    """Coordinates with the same pairwise distances as the rows of X: centred, projected on their span.

    These are the principal-component scores, largest variance first. Directions with an eigenvalue
    below tol x the largest are dropped (rounding noise only).
    """
    X = np.asarray(X, dtype=np.float64)
    centred = X - X.mean(axis=0)
    if centred.shape[1] <= centred.shape[0]:
        return centred
    w, V = np.linalg.eigh(centred @ centred.T)
    w, V = w[::-1], V[:, ::-1]
    keep = w > tol * w[0]
    return np.ascontiguousarray(V[:, keep] * np.sqrt(w[keep]))


def distance_error(X, Z, count=2000, seed=0):
    """Largest relative change of a pairwise distance between X and its exact coordinates Z.

    Each change is divided by the distance in X, except for identical rows (distance 0 in X, e.g. the
    two interface rows of one GB in B and D), whose change is divided by the median distance: Z
    separates them by rounding only. (Until 2026-10-06 they were divided by 1e-300, so one such
    pair in the sample gave about 1e286; phase 1 of Job A 12895 recorded that for D.)
    """
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), count), rng.integers(0, len(X), count)
    keep = i != j
    d_x = np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1)
    d_z = np.linalg.norm(Z[i[keep]] - Z[j[keep]], axis=1)
    scale = np.where(d_x > 0, d_x, np.median(d_x))
    return float(np.max(np.abs(d_x - d_z) / np.maximum(scale, 1e-300)))


# --------------------------------------------------------------------------- distances and neighbours

def square_distances(Y):
    Y = np.asarray(Y, dtype=np.float64)
    squared = np.einsum("ij,ij->i", Y, Y)
    D2 = squared[:, None] + squared[None, :] - 2.0 * (Y @ Y.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def excluded(D, partner):
    """Copy of D with the row itself and its partner (-1: none) set to infinity."""
    M = np.array(D, dtype=np.float64, copy=True)
    n = len(M)
    M[np.arange(n), np.arange(n)] = np.inf
    has = partner >= 0
    M[np.flatnonzero(has), partner[has]] = np.inf
    return M


def usable_k(k, n):
    return max(1, min(int(k), n - 2))


def neighbour_sets(D, partner, k):
    """Each row's k nearest other rows (unordered), leaving out the row's partner."""
    k = usable_k(k, len(D))
    return np.argpartition(excluded(D, partner), k - 1, axis=1)[:, :k]


def jaccard_lists(D, partner, ks):
    """Sorted neighbour lists for the Jaccard scores, {k: lists or None}.

    None (score left undefined) unless some candidate stays outside the k neighbours, i.e.
    k <= n - 3 (n rows less the row itself and its partner): otherwise every list holds every
    candidate and the score is 1 by construction. Unlike neighbour_sets, k is never capped.
    """
    return {k: np.sort(neighbour_sets(D, partner, k), axis=1) if k <= len(D) - 3 else None for k in ks}


def jaccard(Na, Nb):
    """Mean Jaccard index |a & b| / |a | b| of two neighbour lists of the same rows."""
    n, k = Na.shape
    member = np.zeros((n, int(max(Na.max(), Nb.max())) + 1), dtype=bool)
    member[np.arange(n)[:, None], Na] = True
    shared = member[np.arange(n)[:, None], Nb].sum(axis=1)
    return float(np.mean(shared / (2 * k - shared)))


def pair_sample(n, partner, count, seed=0):
    """Up to `count` row pairs i < j (all if there are fewer), same-structure pairs left out."""
    i, j = np.triu_indices(n, 1)
    keep = partner[i] != j
    i, j = i[keep], j[keep]
    if len(i) > count:
        chosen = np.sort(np.random.default_rng(seed).choice(len(i), count, replace=False))
        i, j = i[chosen], j[chosen]
    return i, j


def ward_labels(Y, clusters):
    from scipy.cluster.hierarchy import fcluster, linkage

    return fcluster(linkage(np.asarray(Y, dtype=np.float64), "ward"), t=min(clusters, len(Y)), criterion="maxclust")


def rank_matrix(D, partner):
    """Rank of every row j among row i's neighbours (1 = nearest); self and partner rank last."""
    order = np.argsort(excluded(D, partner), axis=1, kind="stable")
    n = len(D)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n + 1)[None, :]
    return ranks, order


def trust_continuity(D_in, D_out, partner, k):
    """Trustworthiness and continuity (Venna & Kaski) of a map, partner left out of both neighbourhoods.

    trustworthiness penalises map neighbours that are far in the input; continuity, input neighbours
    that are far in the map. Without partners this equals sklearn.manifold.trustworthiness (and the
    same with input and map swapped).
    """
    n = len(D_in)
    n_eff = n - float(np.mean(partner >= 0))
    k = int(min(usable_k(k, n), np.floor((2 * n_eff - 2) / 3)))
    R_in, order_in = rank_matrix(D_in, partner)
    R_out, order_out = rank_matrix(D_out, partner)
    rows = np.arange(n)[:, None]
    in_set = np.zeros((n, n), dtype=bool)
    out_set = np.zeros((n, n), dtype=bool)
    in_set[rows, order_in[:, :k]] = True
    out_set[rows, order_out[:, :k]] = True
    norm = 2.0 / (n_eff * k * (2.0 * n_eff - 3.0 * k - 1.0))
    trust = 1.0 - norm * float(np.sum((R_in - k)[out_set & ~in_set]))
    continuity = 1.0 - norm * float(np.sum((R_out - k)[in_set & ~out_set]))
    return trust, continuity


def triplet_accuracy(D_in, D_out, count=10000, seed=0):
    """Share of random triplets (i, j, l) whose order d(i,j) < d(i,l) the map keeps."""
    rng = np.random.default_rng(seed)
    n = len(D_in)
    i, j, l = rng.integers(0, n, (3, count))
    keep = (i != j) & (i != l) & (j != l)
    i, j, l = i[keep], j[keep], l[keep]
    return float(np.mean(np.sign(D_in[i, j] - D_in[i, l]) == np.sign(D_out[i, j] - D_out[i, l])))


def organisation(D, partner, codes, k, pooled):
    """What organises the rows: kNN accuracy for the classes, kNN R^2 for the geometry labels.

    Each row is predicted from its k nearest other rows (partner left out): majority vote
    (ties go to the lower code) or the mean value.
    """
    N = neighbour_sets(D, partner, k)
    out = {}
    for name, y in codes["classes"].items():
        if len(np.unique(y)) < 2:
            continue
        counts = np.zeros((len(y), y.max() + 1))
        np.add.at(counts, (np.repeat(np.arange(len(y)), N.shape[1]), y[N].ravel()), 1)
        correct = counts.argmax(axis=1) == y
        out[f"knn_{name}"] = float(correct.mean())
        if name == "grain_class" and pooled:
            for plane in np.unique(codes["plane"]):
                out[f"knn_grain_class@{plane}"] = float(correct[codes["plane"] == plane].mean())
    for name, y in codes["values"].items():
        spread = np.square(y - y.mean()).sum()
        if spread > 0:
            out[f"knn_r2_{name}"] = float(1.0 - np.square(y - y[N].mean(axis=1)).sum() / spread)
    return out


def majority_baselines(codes, pooled):
    out = {}
    for name, y in codes["classes"].items():
        if len(np.unique(y)) >= 2:
            out[f"knn_{name}"] = float(np.bincount(y).max() / len(y))
            if name == "grain_class" and pooled:
                for plane in np.unique(codes["plane"]):
                    members = y[codes["plane"] == plane]
                    out[f"knn_grain_class@{plane}"] = float(np.bincount(members).max() / len(members))
    return out


# --------------------------------------------------------------------------- methods

def pca_scores(Z, count):
    """(scores of the first `count` components, explained-variance ratio of all components)."""
    centred = Z - Z.mean(axis=0)
    w, V = np.linalg.eigh(centred @ centred.T)
    w, V = np.clip(w[::-1], 0.0, None), V[:, ::-1]
    count = min(count, len(w))
    return V[:, :count] * np.sqrt(w[:count]), w / w.sum()


def median_distance(D):
    return float(np.median(D[np.triu_indices(len(D), 1)]))


def diffusion_map(D, width, count=2, alpha=1.0, t=1):
    """Diffusion map (Coifman & Lafon 2006) from pairwise distances.

    Gaussian kernel exp(-(d / width)^2); alpha = 1 removes the effect of the sampling density.
    Returns (coordinates of the first `count` non-trivial eigenvectors scaled by eigenvalue^t,
    eigenvalues without the trivial 1).
    """
    K = np.exp(-np.square(D / width))
    q = K.sum(axis=1)
    K = K / np.outer(q ** alpha, q ** alpha)
    d = K.sum(axis=1)
    S = K / np.sqrt(np.outer(d, d))
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    w, V = w[::-1], V[:, ::-1]
    psi = V / np.sqrt(d)[:, None]
    psi /= psi[:, :1]
    return psi[:, 1:count + 1] * w[1:count + 1] ** t, w[1:]


def parse_dimred(text, n):
    """(points, last 'Error in fitting LD points') from dimred -vv output."""
    error, points = np.nan, []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "Error in fitting LD points" in stripped:
                error = float(stripped.split()[-1])
            continue
        values = stripped.split()
        try:
            points.append([float(values[0]), float(values[1])])
        except (ValueError, IndexError):
            continue
    points = np.array(points)
    if points.shape != (n, 2):
        raise RuntimeError(f"dimred returned {points.shape} points for {n} rows")
    return points, error


def sketch_map(Z, sigma, seed, hd=(2, 6), ld=(1, 2), binary=SKETCHMAP, iterations=10):
    """2D sketch-map (Ceriotti et al. 2011) with cosmo-epfl's dimred, as in its utils/sketch-map.sh.

    Start: the first two principal components plus seeded Gaussian noise (5% of their spread).
    Then: iterative sketch-map, a loop of global grid optimisations with a shrinking mixing
    weight (until the error changes by less than 1%), and a final fit. Both sigmoids use sigma;
    hd = (a, b) of the high-dimensional sigmoid, ld = (a, b) of the low-dimensional one.
    Returns (map, error after each dimred call; the first is after one step from the start).
    """
    binary = Path(binary)
    if not binary.is_file():
        raise FileNotFoundError(f"sketch-map not built: {binary}")
    Z = np.asarray(Z, dtype=np.float64)
    n = len(Z)
    rng = np.random.default_rng(seed)
    start = pca_scores(Z, 2)[0]
    start = start + rng.normal(scale=0.05 * start.std(), size=start.shape)
    common = ["-vv", "-D", str(Z.shape[1]), "-d", "2", "-center",
              "-fun-hd", f"{sigma:.10g},{hd[0]:g},{hd[1]:g}", "-fun-ld", f"{sigma:.10g},{ld[0]:g},{ld[1]:g}"]
    history = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
        data, init = Path(folder) / "data", Path(folder) / "init"
        np.savetxt(data, Z, fmt="%.12e")

        def run(extra, points):
            np.savetxt(init, points, fmt="%.12e")
            # cwd: the grid optimisation writes a restart file global.<point> (the whole map) into the working
            # directory every time it moves a point; unused here, so it stays in the temporary folder.
            with open(data) as stdin:
                done = subprocess.run([str(binary), *common, "-init", str(init), *extra], stdin=stdin,
                                      capture_output=True, text=True, cwd=folder)
            if done.returncode != 0:
                raise RuntimeError(f"dimred failed ({done.returncode}): {done.stderr[-500:]}")
            points, error = parse_dimred(done.stdout, n)
            history.append(error)
            return points

        run(["-preopt", "1"], start)
        Y = run(["-preopt", "100"], start)
        first, previous, mix = history[-1], history[-1], 1.0
        for step in range(iterations):
            width = 1.2 * np.sqrt(np.square(Y).sum(axis=1)).max()
            Y = run(["-preopt", "50", "-grid", f"{width:.10g},21,201", "-gopt", "3", "-imix", f"{mix:.10g}"], Y)
            error = history[-1]
            mix *= min(max(first / (first + error), 0.1), 0.5)
            if step > 0 and ((previous - error) / error) ** 2 < 1e-4:
                break
            previous = error
        width = 1.2 * np.sqrt(np.square(Y).sum(axis=1)).max()
        Y = run(["-preopt", "100", "-grid", f"{width:.10g},21,201", "-gopt", "10"], Y)
    return Y, history


def grouped_folds(groups, folds=5):
    from sklearn.model_selection import GroupKFold

    return list(GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(np.zeros(len(groups)), groups=groups))


def target_matrix(kind, codes):
    """Targets of the supervised maps: one-hot plane or grain class, or the standardised geometry labels."""
    if kind == "geometry":
        Y = np.column_stack([codes["values"][name] for name in gl.TARGETS])
        spread = Y.std(axis=0)
        return (Y - Y.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    y = codes["classes"]["plane" if kind == "plane" else "grain_class"]
    return np.eye(y.max() + 1)[y]


def q_squared(make_model, X, Y, groups):
    """Cross-validated R^2 of Y (folds by GB), summed over all target columns."""
    predicted = np.zeros_like(Y)
    for train, test in grouped_folds(groups):
        model = make_model().fit(X[train], Y[train])
        predicted[test] = model.predict(X[test]).reshape(len(test), -1)
    return float(1.0 - np.square(Y - predicted).sum() / np.square(Y - Y.mean(axis=0)).sum())


def make_map(method, Z, D, codes, seed, settings, main):
    """2D map of the rows Z (exact coordinates; D their distances) and the method's own numbers."""
    n = len(Z)
    extras = {}
    family, _, target = method.partition("_")
    if method == "pca":
        scores, ratio = pca_scores(Z, 10)
        Y = scores[:, :2]
        cumulative = np.cumsum(ratio)
        extras.update(participation_ratio=float(1.0 / np.square(ratio).sum()),
                      n_pc90=int(np.searchsorted(cumulative, 0.90) + 1),
                      n_pc99=int(np.searchsorted(cumulative, 0.99) + 1),
                      **{f"variance_ratio_{i + 1}": float(r) for i, r in enumerate(ratio[:10])})
        if main:
            extras["scores10"] = scores
    elif method == "kpca":
        from sklearn.decomposition import KernelPCA

        width = settings["width_factor"] * median_distance(D)
        K = np.exp(-0.5 * np.square(D / width))
        model = KernelPCA(n_components=min(10, n - 1), kernel="precomputed", eigen_solver="dense")
        Y = model.fit_transform(K)[:, :2]
        trace = np.trace(K) - K.sum() / n  # trace of the centred kernel = sum of all its eigenvalues
        extras.update(width=width, **{f"eigenvalue_share_{i + 1}": float(v / trace)
                                      for i, v in enumerate(model.eigenvalues_)})
    elif method == "ica":
        from sklearn.decomposition import FastICA

        scores = pca_scores(Z, 10)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = FastICA(n_components=2, whiten="unit-variance", random_state=seed, max_iter=2000)
            Y = model.fit_transform(scores)
            ten = FastICA(n_components=min(10, scores.shape[1]), whiten="unit-variance", random_state=seed,
                          max_iter=2000).fit_transform(scores)
        extras.update(n_iter=int(model.n_iter_), sources10=ten)
    elif family == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        y = codes["classes"]["plane" if target == "plane" else "grain_class"]
        classes = len(np.unique(y))
        scores = pca_scores(Z, 50)[0]
        model = LinearDiscriminantAnalysis(n_components=min(2, classes - 1))
        Y = model.fit_transform(scores, y)
        if Y.shape[1] < 2:
            Y = np.column_stack([Y, np.zeros(n)])
        extras.update({f"discriminant_ratio_{i + 1}": float(r) for i, r in enumerate(model.explained_variance_ratio_)})
        if main:
            correct = np.zeros(n, dtype=bool)
            for train, test in grouped_folds(codes["gb"]):
                fold = LinearDiscriminantAnalysis().fit(scores[train], y[train])
                correct[test] = fold.predict(scores[test]) == y[test]
            extras["cv_accuracy"] = float(correct.mean())
    elif family == "pls":
        from sklearn.cross_decomposition import PLSRegression

        T = target_matrix(target, codes)
        model = PLSRegression(n_components=2, scale=False).fit(Z, T)
        Y = model.x_scores_
        if main:
            for count in (2, 10):
                usable = min(count, Z.shape[1], n - n // 5 - 1)  # PLS needs components <= features and training rows
                extras[f"q2_{count}"] = q_squared(lambda c=usable: PLSRegression(n_components=c, scale=False),
                                                  Z, T, codes["gb"])
    elif family == "pcovr":
        from skmatter.decomposition import PCovR

        X = (Z - Z.mean(axis=0)) / np.sqrt(np.square(Z - Z.mean(axis=0)).sum() / n)
        T = target_matrix(target, codes)
        T = (T - T.mean(axis=0)) / np.sqrt(np.square(T - T.mean(axis=0)).sum() / n)
        model = PCovR(mixing=settings["mixing"], n_components=2, random_state=0).fit(X, T)
        Y = model.transform(X)
        if main:
            for mixing in (0.0, 0.25, 0.5, 0.75, 1.0):
                scores = PCovR(mixing=mixing, n_components=2, random_state=0).fit(X, T).transform(X)
                fit, *_ = np.linalg.lstsq(scores, np.column_stack([T, X]), rcond=None)
                residual = np.column_stack([T, X]) - scores @ fit
                extras[f"loss_regression@{mixing:g}"] = float(np.square(residual[:, :T.shape[1]]).sum() / np.square(T).sum())
                extras[f"loss_reconstruction@{mixing:g}"] = float(np.square(residual[:, T.shape[1]:]).sum() / np.square(X).sum())
    elif method == "mds":
        Y, extras["stress1"] = mds_map(D, 2, seed)
        if main:
            for dimension in (1, 3, 5, 10):
                extras[f"stress1_d{dimension}"] = mds_map(D, min(dimension, n - 1), seed)[1]
            extras["stress1_d2"] = extras["stress1"]
    elif method == "isomap":
        from scipy.sparse.csgraph import connected_components
        from sklearn.manifold import Isomap
        from sklearn.neighbors import kneighbors_graph

        neighbours = min(15, n - 2)
        extras["graph_components"] = int(connected_components(kneighbors_graph(Z, neighbours), directed=False)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Isomap(n_neighbors=neighbours, n_components=min(10, n - 1) if main else 2)
            embedding = model.fit_transform(Z)
        Y = embedding[:, :2]
        if main:
            geodesic = model.dist_matrix_[np.triu_indices(n, 1)]
            for dimension in range(1, embedding.shape[1] + 1):
                mapped = square_distances(embedding[:, :dimension])[np.triu_indices(n, 1)]
                extras[f"residual_variance_d{dimension}"] = float(1.0 - np.corrcoef(geodesic, mapped)[0, 1] ** 2)
    elif method == "tsne":
        Y = tsne_map(Z, 30, seed)
        extras["kl_divergence"] = tsne_kl(D, Y, 30)
        if main:
            for perplexity in (15, 50):
                extras[f"kl_perplexity{perplexity}"] = tsne_kl(D, tsne_map(Z, perplexity, seed), perplexity)
            extras["kl_perplexity30"] = extras["kl_divergence"]
    elif method == "umap":
        from umap import UMAP

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, n_epochs=500, metric="euclidean",
                     random_state=seed, n_jobs=1).fit_transform(Z)
        if main:
            from scipy.sparse.csgraph import connected_components
            from umap.umap_ import fuzzy_simplicial_set

            for neighbours in (15, 50):
                graph = fuzzy_simplicial_set(Z, min(neighbours, n - 1), np.random.RandomState(seed), "euclidean")[0]
                extras[f"graph_components_n{neighbours}"] = int(connected_components(graph, directed=False)[0])
    elif method == "pacmap":
        import pacmap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pacmap.PaCMAP(n_components=2, n_neighbors=min(10, n - 2), random_state=seed,
                                  apply_pca=Z.shape[1] > 100, verbose=False)
            Y = model.fit_transform(Z, init="pca")
    elif method == "diffusion":
        width = settings["width_factor"] * median_distance(D)
        Y, eigenvalues = diffusion_map(D, width, 2)
        gaps = -np.diff(eigenvalues[:10])
        extras.update(width=width, spectral_gap_after=int(np.argmax(gaps) + 1),
                      **{f"eigenvalue_{i + 1}": float(v) for i, v in enumerate(eigenvalues[:10])})
    elif method == "sketchmap":
        sigma = settings["width_factor"] * median_distance(D)
        Y, history = sketch_map(Z, sigma, seed, settings["sketchmap_hd"], settings["sketchmap_ld"],
                                settings["sketchmap_bin"])
        extras.update(sigma=sigma, error_start=float(history[0]), error=float(history[-1]), dimred_calls=len(history))
    else:
        raise ValueError(f"unknown method {method!r}")
    Y = np.asarray(Y, dtype=np.float64)
    if Y.shape != (n, 2) or not np.isfinite(Y).all():
        raise ValueError(f"{method} produced an invalid map {Y.shape}")
    return Y, extras


def mds_map(D, dimension, seed):
    """Metric MDS (SMACOF, one random start) and its Kruskal stress-1."""
    from sklearn.manifold import MDS

    model = MDS(n_components=dimension, dissimilarity="precomputed", n_init=1, max_iter=300,
                random_state=seed, normalized_stress=False)
    Y = model.fit_transform(D)
    mapped = square_distances(Y)
    upper = np.triu_indices(len(D), 1)
    stress = float(np.sqrt(np.square(mapped[upper] - D[upper]).sum() / np.square(D[upper]).sum()))
    if Y.shape[1] < 2:
        Y = np.column_stack([Y, np.zeros(len(Y))])
    return Y[:, :2], stress


def tsne_map(Z, perplexity, seed):
    """2D t-SNE, exact method (the select_gbs.embed settings otherwise).

    Barnes-Hut misplaces exactly duplicated rows, which B and D have (both interface rows of a GB): on
    synthetic twins its map's KL was 5.0 against 0.2 for the exact method at 56 rows, 1.4 against 0.7 at
    1,000 rows, and close only at 2,222 (archive/representation_study_20261005/smoke/diagnostics/).
    Exact costs about 1 min per 2,222-row map.
    """
    from sklearn.manifold import TSNE

    model = TSNE(n_components=2, perplexity=min(perplexity, (len(Z) - 1) / 3.0), init="pca", learning_rate="auto",
                 max_iter=1000, metric="euclidean", method="exact", random_state=seed, n_jobs=1)
    return model.fit_transform(Z)


def tsne_kl(D, Y, perplexity):
    """Exact KL(P || Q) of a finished 2D t-SNE map, with the dense P of t-SNE's exact method.

    D: the input's Euclidean distance matrix. sklearn's kl_divergence_ is the Barnes-Hut estimate and can
    be negative: it was -2.3 and -3.0 for B and D (two identical rows per GB) in the smoke test.
    """
    from scipy.spatial.distance import pdist
    from sklearn.manifold._t_sne import _joint_probabilities  # private; sklearn is pinned in src/mlenv

    eps = np.finfo(np.double).eps
    P = _joint_probabilities(np.square(D), min(perplexity, (len(D) - 1) / 3.0), 0)
    weights = 1.0 / (1.0 + pdist(Y, "sqeuclidean"))  # Student t, one degree of freedom (2D map)
    Q = np.maximum(weights / (2.0 * weights.sum()), eps)
    return float(2.0 * np.dot(P, np.log(np.maximum(P, eps) / Q)))


# --------------------------------------------------------------------------- comparisons

def features(Y, rows, partner_of, pairs, ks, clusters):
    """What a comparison needs from one map on the rows `rows` (sorted global row numbers)."""
    lookup = np.full(int(partner_of.size), -1)
    lookup[rows] = np.arange(len(rows))
    partner = lookup[partner_of[rows]]
    gi, gj = pairs
    keep = (lookup[gi] >= 0) & (lookup[gj] >= 0)
    i, j = lookup[gi[keep]], lookup[gj[keep]]
    D = square_distances(Y)
    out = {"neighbours": jaccard_lists(D, partner, ks), "pair_distances": D[i, j], "Y": Y}
    if clusters:
        out["ward"] = ward_labels(Y, clusters)
    return out


def score_features(a, b, ks, clusters, coordinates=True):
    """Map-to-map scores from two feature sets on the same rows."""
    from scipy.stats import spearmanr

    out = {}
    if coordinates:
        out["procrustes"] = procrustes_disparity(a["Y"], b["Y"])
    for k in ks:
        Na, Nb = a["neighbours"][k], b["neighbours"][k]
        out[f"jaccard{k}"] = jaccard(Na, Nb) if Na is not None and Nb is not None else float("nan")
    out["spearman"] = float(spearmanr(a["pair_distances"], b["pair_distances"]).statistic)
    if clusters and coordinates:
        from sklearn.metrics import adjusted_rand_score

        out["ari"] = float(adjusted_rand_score(a["ward"], b["ward"]))
    return out


def procrustes_disparity(Ya, Yb):
    from scipy.spatial import procrustes

    if np.ptp(Ya, axis=0).max() == 0 or np.ptp(Yb, axis=0).max() == 0:
        return float("nan")
    return float(procrustes(Ya, Yb)[2])


def compare_maps(a, b, partner_of, pairs, ks, clusters):
    """Scores between two map records ({"rows", "Y"}), on the rows both contain."""
    shared = np.intersect1d(a["rows"], b["rows"], assume_unique=True)
    Ya = a["Y"][np.searchsorted(a["rows"], shared)]
    Yb = b["Y"][np.searchsorted(b["rows"], shared)]
    return score_features(features(Ya, shared, partner_of, pairs, ks, clusters),
                          features(Yb, shared, partner_of, pairs, ks, clusters), ks, clusters)


def change(variable, score):
    """Dissimilarity from a score: the Procrustes disparity itself, else 1 - score."""
    return score if variable == "procrustes" else 1.0 - score


def sensitivity(map_change, input_change):
    """Map change per unit of input change; NaN when the input does not change."""
    return float(map_change / input_change) if input_change > 0 else float("nan")


# --------------------------------------------------------------------------- statistics

def _within(D2, groups):
    """Sum over groups of (1 / n_g) x sum of squared distances within the group (each pair once)."""
    total = 0.0
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        total += D2[np.ix_(members, members)].sum() / (2.0 * len(members))
    return total


def permanova_shares(D, a, b):
    """Sums of squares of a two-way PERMANOVA (Anderson 2001) on the distance matrix D.

    Returns {factor: SS} for a, b, interaction and residual, plus the total. With one observation
    per cell the residual is the interaction and no interaction term is returned.
    """
    D2 = np.square(np.asarray(D, dtype=np.float64))
    a, b = np.asarray(a), np.asarray(b)
    cells = np.array([f"{x}|{y}" for x, y in zip(a, b)])
    total = _within(D2, np.zeros(len(D2)))
    ss_a = total - _within(D2, a)
    ss_b = total - _within(D2, b)
    residual = _within(D2, cells)
    out = {"a": ss_a, "b": ss_b, "total": total}
    if len(np.unique(cells)) == len(cells):
        out["residual"] = total - ss_a - ss_b
    else:
        out["interaction"] = total - ss_a - ss_b - residual
        out["residual"] = residual
    return out


def permanova(D, a, b, names=("a", "b"), permutations=999, seed=0):
    """Two-way PERMANOVA: share of the total sum of squares, pseudo-F and permutation p per term."""
    a, b = np.asarray(a), np.asarray(b)
    n, la, lb = len(a), len(np.unique(a)), len(np.unique(b))
    ss = permanova_shares(D, a, b)
    replicated = "interaction" in ss
    df = {"a": la - 1, "b": lb - 1}
    if replicated:
        df["interaction"] = (la - 1) * (lb - 1)
        df["residual"] = n - la * lb
    else:
        df["residual"] = n - 1 - df["a"] - df["b"]
    terms = [t for t in ("a", "b", "interaction") if t in ss]

    def pseudo_f(values):
        # Sums of squares below 1e-12 of the total are rounding noise and count as 0. With a zero residual
        # (every replicate map identical, e.g. ICA) F is infinite for a term that explains something and
        # NaN for one that explains nothing; NaN also when nothing varies at all (e.g. ARI 1 everywhere).
        clean = {t: np.float64(values[t] if abs(values[t]) > 1e-12 * values["total"] else 0.0)
                 for t in terms + ["residual"]}  # numpy floats: x / 0 gives inf or NaN, not an exception
        mean_residual = clean["residual"] / df["residual"] if df["residual"] > 0 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            return {t: (clean[t] / df[t]) / mean_residual if df[t] > 0 else np.nan for t in terms}

    observed = pseudo_f(ss)
    exceed = {t: 0 for t in terms}
    rng = np.random.default_rng(seed)
    if permutations and df["residual"] > 0:
        for _ in range(permutations):
            order = rng.permutation(n)
            trial = pseudo_f(permanova_shares(D, a[order], b[order]))
            for t in terms:
                exceed[t] += trial[t] >= observed[t] - 1e-12
    label = {"a": names[0], "b": names[1], "interaction": "interaction", "residual": "residual"}
    out = {}
    for t in terms + ["residual"]:
        # No p-value for an undefined statistic: nothing can exceed a NaN, which would give p = 1/(permutations+1).
        testable = t in observed and not bool(np.isnan(observed[t])) and permutations and df["residual"] > 0
        out[label[t]] = {"share": float(ss[t] / ss["total"]) if ss["total"] > 0 else np.nan,
                         "pseudo_f": float(observed[t]) if t in observed else np.nan,
                         "p_value": float((exceed[t] + 1) / (permutations + 1)) if testable else np.nan}
    return out


def anova_2x2(values, a, b, names=("a", "b")):
    """Two-way ANOVA of per-map values: share of the total sum of squares, F and p per term."""
    from scipy.stats import f as f_distribution

    values, a, b = np.asarray(values, dtype=np.float64), np.asarray(a), np.asarray(b)
    grand = values.mean()
    total = np.square(values - grand).sum()

    def between(groups):
        return sum((groups == g).sum() * (values[groups == g].mean() - grand) ** 2 for g in np.unique(groups))

    cells = np.array([f"{x}|{y}" for x, y in zip(a, b)])
    ss = {"a": between(a), "b": between(b)}
    ss_cells = between(cells)
    ss["interaction"] = ss_cells - ss["a"] - ss["b"]
    ss["residual"] = total - ss_cells
    df = {"a": len(np.unique(a)) - 1, "b": len(np.unique(b)) - 1}
    df["interaction"] = df["a"] * df["b"]
    df["residual"] = len(values) - len(np.unique(cells))
    label = {"a": names[0], "b": names[1], "interaction": "interaction", "residual": "residual"}
    # As in permanova: below 1e-12 of the total is rounding noise; a zero residual makes F infinite (p = 0)
    # for a term that explains something and leaves it undefined for one that explains nothing.
    clean = {t: v if abs(v) > 1e-12 * total else 0.0 for t, v in ss.items()}
    out = {}
    for t in ("a", "b", "interaction", "residual"):
        entry = {"share": float(ss[t] / total) if total > 0 else np.nan, "pseudo_f": np.nan, "p_value": np.nan}
        if t != "residual" and df["residual"] > 0 and df[t] > 0:
            if clean["residual"] > 0:
                F = (clean[t] / df[t]) / (clean["residual"] / df["residual"])
                entry.update(pseudo_f=float(F), p_value=float(f_distribution.sf(F, df[t], df["residual"])))
            elif clean[t] > 0:
                entry.update(pseudo_f=np.inf, p_value=0.0)
        out[label[t]] = entry
    return out


def linear_cka(Xa, Xb):
    """Linear centred kernel alignment of two representations of the same rows (1 = same up to rotation and scale)."""
    Ka = (Xa - Xa.mean(axis=0)) @ (Xa - Xa.mean(axis=0)).T
    Kb = (Xb - Xb.mean(axis=0)) @ (Xb - Xb.mean(axis=0)).T
    return float((Ka * Kb).sum() / np.sqrt((Ka * Ka).sum() * (Kb * Kb).sum()))


def subspace_overlap(Sa, Sb):
    """Mean squared cosine of the principal angles between the column spaces of two score matrices."""
    Qa = np.linalg.qr(Sa - Sa.mean(axis=0))[0]
    Qb = np.linalg.qr(Sb - Sb.mean(axis=0))[0]
    return float(np.mean(np.square(np.linalg.svd(Qa.T @ Qb, compute_uv=False))))


def reproducible_components(sources, threshold=0.95):
    """ICA components of the first seed whose best match (|correlation|) in every other seed is above threshold."""
    from scipy.optimize import linear_sum_assignment

    first = sources[0]
    kept = np.ones(first.shape[1], dtype=bool)
    matched = []
    for other in sources[1:]:
        c = np.abs(np.corrcoef(first.T, other.T)[:first.shape[1], first.shape[1]:])
        r, col = linear_sum_assignment(-c)
        best = np.zeros(first.shape[1])
        best[r] = c[r, col]
        kept &= best > threshold
        matched.append(best.mean())
    return int(kept.sum()), float(np.mean(matched)) if matched else float("nan")


# --------------------------------------------------------------------------- bulk test

def weighted_interfaces(X_interface, n_interface, n_species, is_co, only_interface=False):
    """Per species: (n_edge X_edge + n_mid X_mid) / N, with N all atoms of that species in the cell
    (or, with only_interface, only the atoms near the two planes)."""
    out = np.zeros((X_interface.shape[0], X_interface.shape[-1]))
    for s, half in enumerate((is_co, ~is_co)):
        weights = n_interface[:, :, s].astype(np.float64)
        total = weights.sum(axis=1) if only_interface else n_species[:, s].astype(np.float64)
        out[:, half] = (weights[:, :, None] * X_interface[:, :, half]).sum(axis=1) / total[:, None]
    return out


def cell_from_interfaces(cell, X_interface, n_interface, n_species, is_co, bulk=None):
    """How well the whole-cell vector follows from the two interface vectors, per GB.

    prediction = (n_edge X_edge + n_mid X_mid + n_rest bulk) / N per species, with bulk = 0 when not given
    (the covariant lambda >= 1 blocks, whose bulk average vanishes). Returns per-GB arrays.
    """
    predicted = weighted_interfaces(X_interface, n_interface, n_species, is_co)
    if bulk is not None:
        rest = n_species - n_interface.sum(axis=1)
        for s, half in enumerate((is_co, ~is_co)):
            predicted[:, half] += rest[:, s, None] * bulk[half][None, :] / n_species[:, s, None]
    inside = weighted_interfaces(X_interface, n_interface, n_species, is_co, only_interface=True)
    residual = np.linalg.norm(cell - predicted, axis=1)
    scale = (cell * inside).sum(axis=1) / (inside * inside).sum(axis=1)
    out = {"relative_residual": residual / np.linalg.norm(cell, axis=1),
           "cosine_with_interfaces": (cell * inside).sum(axis=1) / (np.linalg.norm(cell, axis=1) * np.linalg.norm(inside, axis=1)),
           "fitted_scale": scale,
           "share_of_atoms_near_planes": n_interface.sum(axis=(1, 2)) / n_species.sum(axis=1)}
    if bulk is not None:
        out["residual_over_gb_signal"] = residual / np.linalg.norm(cell - bulk[None, :], axis=1)
    return out


def bulk_averages(run, root=sc.COVARIANT_ROOT, blocks=VECTORIAL):
    """Length of each bulk cell's lambda >= 1 cell average over one atom's mean vector length
    (as check_covariant.py), summarised per kind of cell and plane."""
    import pandas as pd

    _, global_dir = covariant_dirs(run, root)
    index = pd.read_csv(global_dir / "bulk_index.csv", dtype={"plane": str})
    out = {}
    with np.load(global_dir / "bulk_covariant.npz") as data:
        for block in blocks:
            size = np.linalg.norm(data[f"X_{block}"].reshape(len(index), -1), axis=1) / index[f"atom_norm_{block}"].to_numpy()
            for kind in index["kind"].unique():
                for plane in [None] + sorted(index.loc[index["kind"] == kind, "plane"].dropna().unique()):
                    members = (index["kind"] == kind).to_numpy()
                    if plane is not None:
                        members &= (index["plane"] == plane).to_numpy()
                    out[f"{block}|{kind}|{plane or 'all'}"] = {"n": int(members.sum()), "max": float(size[members].max()),
                                                                "median": float(np.median(size[members]))}
    return out


def summarise_per_plane(values, planes):
    out = {}
    for plane in ["all"] + sorted(np.unique(planes)):
        members = np.ones(len(planes), dtype=bool) if plane == "all" else planes == plane
        v = values[members]
        out[plane] = {"n": int(members.sum()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                      "max": float(v.max()), "min": float(v.min())}
    return out


def bulk_test(standard, covariant, gbs, covariant_run, covariant_root, vectorial_columns):
    """Bulk symmetry test and the two whole-cell versus interface checks (phase 1, step 1)."""
    planes = gbs["plane"].to_numpy(str)
    lo, hi = vectorial_columns
    result = {"bulk_cell_averages": bulk_averages(covariant_run, covariant_root)}
    d_vs_c = cell_from_interfaces(covariant["cell"][:, lo:hi], covariant["interface"][:, :, lo:hi],
                                  covariant["n_interface"], covariant["n_species"], covariant["is_co"][lo:hi])
    b_vs_a = cell_from_interfaces(standard["cell"], standard["interface"], standard["n_interface"],
                                  standard["n_species"], standard["is_co"], bulk=standard["bulk"])
    result["D_vs_C"] = {name: summarise_per_plane(values, planes) for name, values in d_vs_c.items()}
    result["B_vs_A"] = {name: summarise_per_plane(values, planes) for name, values in b_vs_a.items()}
    return result


# --------------------------------------------------------------------------- workers

_STATE = {}


def _init_worker(paths, codes, settings):
    _STATE.update(paths=paths, codes=codes, settings=settings, cache={})


def _coordinates(key):
    cache = _STATE["cache"]
    if key not in cache:
        cache[key] = np.load(_STATE["paths"][key], mmap_mode="r")
    return cache[key]


def local_codes(codes, rows):
    return {"classes": {k: v[rows] for k, v in codes["classes"].items()},
            "values": {k: v[rows] for k, v in codes["values"].items()},
            "gb": codes["gb"][rows], "plane": codes["plane"][rows]}


def subset_coordinates(key, rows):
    Z = np.asarray(_coordinates(key)[rows], dtype=np.float64)
    return exact_coordinates(Z) if Z.shape[1] > len(rows) else Z - Z.mean(axis=0)


def map_task(task):
    """One map and its per-map variables. task: dict with key, method, coordinates, rows, seed, main."""
    started = time.time()
    settings = _STATE["settings"]
    rows = task["rows"]
    try:
        Z = subset_coordinates(task["coordinates"], rows)
        D = square_distances(Z)
        codes = local_codes(_STATE["codes"], rows)
        Y, extras = make_map(task["method"], Z, D, codes, task["seed"], settings, task["main"])
        variables = {}
        if task.get("variables", True):
            partner = local_partner(rows, settings["n_rows"])
            D_map = square_distances(Y)
            variables = organisation(D_map, partner, codes, settings["organise_k"], task["subset"] == "pooled")
            variables["trustworthiness"], variables["continuity"] = trust_continuity(D, D_map, partner, settings["trust_k"])
            variables["triplet_accuracy"] = triplet_accuracy(D, D_map, settings["triplets"])
        return {**task, "Y": Y, "extras": extras, "variables": variables, "error": None,
                "seconds": round(time.time() - started, 2)}
    except Exception as error:  # one failed map must not stop the run
        return {**task, "Y": None, "extras": {}, "variables": {}, "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.time() - started, 2)}


def local_partner(rows, n_rows):
    lookup = np.full(n_rows, -1)
    lookup[rows] = np.arange(len(rows))
    return lookup[partners(n_rows)[rows]]


def raw_task(task):
    """Raw-descriptor features of one representation on one subset: organisation and neighbour features."""
    settings = _STATE["settings"]
    rows = task["rows"]
    Z = subset_coordinates(task["coordinates"], rows)
    D = square_distances(Z)
    partner = local_partner(rows, settings["n_rows"])
    out = {**task, "variables": {}}
    if task.get("organise", True):
        out["variables"] = organisation(D, partner, local_codes(_STATE["codes"], rows), settings["organise_k"],
                                        task["subset"] == "pooled")
    lookup = np.full(settings["n_rows"], -1)
    lookup[rows] = np.arange(len(rows))
    gi, gj = task["pairs"]
    keep = (lookup[gi] >= 0) & (lookup[gj] >= 0)
    out["features"] = {"neighbours": jaccard_lists(D, partner, settings["ks"]),
                       "pair_distances": D[lookup[gi[keep]], lookup[gj[keep]]]}
    if task.get("gram", False):
        out["gram"] = None  # CKA is computed in the parent from the coordinates
    return out


def compare_task(task):
    """Scores for a chunk of map pairs: task = {maps: {id: record}, pairs: [(id_a, id_b)], ...}."""
    settings = _STATE["settings"]
    maps, ks, clusters = task["maps"], settings["ks"], task["clusters"]
    partner_of = partners(settings["n_rows"])
    own = {}

    def own_features(i):
        if i not in own:
            own[i] = features(maps[i]["Y"], maps[i]["rows"], partner_of, task["sample"], ks, clusters)
        return own[i]

    out = []
    for i, j in task["pairs"]:
        a, b = maps[i], maps[j]
        if len(a["rows"]) == len(b["rows"]) and np.array_equal(a["rows"], b["rows"]):
            scores = score_features(own_features(i), own_features(j), ks, clusters)
        else:
            scores = compare_maps(a, b, partner_of, task["sample"], ks, clusters)
        out.append((i, j, scores))
    return out


def run_pool(function, tasks, jobs, paths, codes, settings, label):
    """Run tasks in spawned workers (one thread each), in order of submission; prints progress."""
    started = time.time()
    results = []
    if jobs <= 1:
        _init_worker(paths, codes, settings)
        for done, task in enumerate(tasks, 1):
            results.append(function(task))
            _progress(label, done, len(tasks), started)
        return results
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # One BLAS/OpenMP/numba thread per worker (as descriptor_sensitivity.worker_pool); the
    # workers read these variables when they import numpy, after the spawn.
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(paths, codes, settings)) as pool:
        futures = {pool.submit(function, task): k for k, task in enumerate(tasks)}
        results = [None] * len(tasks)
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            _progress(label, done, len(tasks), started)
    return results


def _progress(label, done, total, started):
    if done == total or done % max(1, total // 20) == 0:
        print(f"  {label}: {done}/{total} ({time.time() - started:.0f} s)", flush=True)


# --------------------------------------------------------------------------- the fifth study

def subset_rows(rows, subsets):
    planes = rows["plane"].to_numpy(str)
    return {s: np.arange(len(rows)) if s == "pooled" else np.flatnonzero(planes == s) for s in subsets}


def gb_masks(rows, members, count, share, seed):
    """`count` subsamples of `share` of the GBs among `members` (both rows of a structure together)."""
    rng = np.random.default_rng(seed)
    structures = np.unique(members // 2)
    size = max(2, int(round(share * len(structures))))
    out = []
    for _ in range(count):
        chosen = np.sort(rng.choice(structures, size, replace=False))
        out.append(np.sort(np.concatenate([2 * chosen, 2 * chosen + 1])))
    return out


def seed_list(count):
    return list(SEEDS[:count]) + [100 + k for k in range(max(0, count - len(SEEDS)))]


def map_tasks(methods, subsets, members, masks, representations, replicates):
    """All map tasks: per method, subset and representation, a main map and `replicates` replicates.

    Stochastic methods: replicate r is seed r on all rows, and the main map is replicate 0.
    Deterministic methods: replicate r is GB subsample r, and the main map uses all rows (replicate -1).
    """
    tasks = []
    for method in methods:
        for subset in subsets:
            if method.endswith("_plane") and subset != "pooled":
                continue
            for rep in representations[subset]:
                base = {"method": method, "subset": subset, "representation": rep, "coordinates": rep}
                if method in STOCHASTIC:
                    for r, seed in enumerate(seed_list(replicates)):
                        tasks.append({**base, "replicate": r, "seed": seed, "rows": members[subset], "main": r == 0})
                else:
                    tasks.append({**base, "replicate": -1, "seed": 0, "rows": members[subset], "main": True})
                    for r in range(replicates):
                        tasks.append({**base, "replicate": r, "seed": 0, "rows": masks[subset][r], "main": False})
    return tasks


COST = {"sketchmap": 20.0, "tsne": 8.0, "mds": 6.0, "isomap": 3.0, "umap": 3.0, "pacmap": 2.0, "pcovr": 2.0}


def by_cost(tasks):
    """Expensive maps first, so the last tasks of a pool are short ones."""
    weight = lambda t: COST.get(t["method"], COST.get(t["method"].split("_")[0], 1.0)) * len(t["rows"]) ** 2
    return sorted(tasks, key=weight, reverse=True)


def chunked(items, size):
    return [items[k:k + size] for k in range(0, len(items), size)]


def comparison_tasks(maps, samples, clusters, chunk):
    """All map pairs within each (method, subset) group, among the replicate maps and the main maps."""
    groups = {}
    for key, record in maps.items():
        groups.setdefault((record["method"], record["subset"]), []).append(key)
    tasks = []
    for (method, subset), keys in sorted(groups.items()):
        keys = sorted(keys, key=lambda k: (maps[k]["representation"], maps[k]["replicate"]))
        pairs = [(a, b) for x, a in enumerate(keys) for b in keys[x + 1:]
                 if _wanted_pair(maps[a], maps[b])]
        for part in chunked(pairs, chunk):
            needed = {k for pair in part for k in pair}
            tasks.append({"maps": {k: {"rows": maps[k]["rows"], "Y": maps[k]["Y"]} for k in needed},
                          "pairs": part, "sample": samples[subset], "clusters": clusters})
    return tasks


def _wanted_pair(a, b):
    """Replicate maps are compared with each other; main maps only with main maps."""
    main_a = a["replicate"] < 0 or (a["method"] in STOCHASTIC and a["replicate"] == 0)
    main_b = b["replicate"] < 0 or (b["method"] in STOCHASTIC and b["replicate"] == 0)
    if a["replicate"] >= 0 and b["replicate"] >= 0:
        return True
    return main_a and main_b


def map_key(method, subset, rep, replicate):
    return f"{method}|{subset}|{rep}|{replicate}"


def add(results, **fields):
    results.append({column: fields.get(column) for column in COLUMNS})


def summarise_group(results, phase, method, subset, maps, scores, raw, raw_features, settings, study="representation"):
    """Rows of results.csv for one method and subset: pairs, per-map variables, effects."""
    keys = [k for k, m in maps.items() if m["method"] == method and m["subset"] == subset]
    main_key = {maps[k]["representation"]: k for k in keys if maps[k]["main"]}
    reps = sorted(main_key, key=lambda r: (r == CHECK, r))  # a representation whose main map failed is left out
    replicate_keys = {rep: sorted((k for k in keys if maps[k]["representation"] == rep and maps[k]["replicate"] >= 0),
                                  key=lambda k: maps[k]["replicate"]) for rep in reps}
    variables = sorted({v for s in scores.values() for v in s})

    def value(a, b, variable):
        s = scores.get((a, b)) or scores.get((b, a))
        return change(variable, s[variable]) if s is not None and variable in s else np.nan

    common = {"study": study, "phase": phase, "method": method, "subset": subset}
    for variable in variables:
        floors = {}
        for rep in reps:
            ks = replicate_keys[rep]
            floors[rep] = np.array([value(a, b, variable) for x, a in enumerate(ks) for b in ks[x + 1:]])
            add(results, **common, kind="floor", representation=rep, variable=variable,
                value=float(np.nanmean(floors[rep])) if floors[rep].size else np.nan,
                noise_floor=float(np.nanpercentile(floors[rep], 95)) if floors[rep].size else np.nan,
                floor_median=float(np.nanmedian(floors[rep])) if floors[rep].size else np.nan, n=int(floors[rep].size))
        for x, a in enumerate(reps):
            for b in reps[x + 1:]:
                between = np.array([value(ka, kb, variable) for ka, kb in zip(replicate_keys[a], replicate_keys[b])])
                floor = np.concatenate([floors[a], floors[b]])
                mean = float(np.nanmean(between)) if between.size else np.nan
                p95 = float(np.nanpercentile(floor, 95)) if floor.size else np.nan
                raw_change = raw.get((subset, a, b), {}).get(variable)
                add(results, **common, kind="pair", representation=f"{a}-{b}", variable=variable, value=mean,
                    sd=float(np.nanstd(between)) if between.size else np.nan, main=value(main_key[a], main_key[b], variable),
                    noise_floor=p95, floor_median=float(np.nanmedian(floor)) if floor.size else np.nan,
                    input_change=raw_change,
                    sensitivity=sensitivity(mean, raw_change) if raw_change is not None else None,
                    significant=bool(mean > p95) if np.isfinite(mean) and np.isfinite(p95) else None,
                    n=int(between.size))
        design = [rep for rep in REPRESENTATIONS if rep in reps]
        if len(design) == 4:
            ks = [k for rep in design for k in replicate_keys[rep]]
            if len(ks) > 4:
                D = np.array([[0.0 if a == b else value(a, b, variable) for b in ks] for a in ks])
                if variable == "procrustes":
                    D = np.sqrt(np.clip(D, 0.0, None))
                if np.isfinite(D).all():
                    region = [REGION[maps[k]["representation"]] for k in ks]
                    covariance = [COVARIANCE[maps[k]["representation"]] for k in ks]
                    effects = permanova(D, region, covariance, ("region", "covariance"), settings["permutations"])
                    for factor, entry in effects.items():
                        add(results, **common, kind="effect", representation="A-D", variable=variable, factor=factor,
                            value=entry["share"], main=entry["pseudo_f"], p_value=entry["p_value"], n=len(ks))
    per_map = sorted({v for k in keys for v in maps[k]["variables"]})
    for variable in per_map:
        for rep in reps:
            values = np.array([maps[k]["variables"].get(variable, np.nan) for k in replicate_keys[rep]])
            add(results, **common, kind="map", representation=rep, variable=variable,
                value=maps[main_key[rep]]["variables"].get(variable, np.nan),
                main=maps[main_key[rep]]["variables"].get(variable, np.nan),
                sd=float(np.nanstd(values)) if values.size else np.nan,
                floor_median=float(np.nanmean(values)) if values.size else np.nan, n=int(values.size))
        design = [rep for rep in REPRESENTATIONS if rep in reps]
        if len(design) == 4:
            ks = [k for rep in design for k in replicate_keys[rep]]
            values = np.array([maps[k]["variables"].get(variable, np.nan) for k in ks])
            if np.isfinite(values).all() and len(ks) > 4:
                effects = anova_2x2(values, [REGION[maps[k]["representation"]] for k in ks],
                                    [COVARIANCE[maps[k]["representation"]] for k in ks], ("region", "covariance"))
                for factor, entry in effects.items():
                    add(results, **common, kind="map_effect", representation="A-D", variable=variable, factor=factor,
                        value=entry["share"], main=entry["pseudo_f"], p_value=entry["p_value"], n=len(ks))
    for rep in reps:
        for name, extra in maps[main_key[rep]]["extras"].items():
            if np.isscalar(extra):
                add(results, **common, kind="method", representation=rep, variable=name, value=float(extra))
    if method == "ica":
        for rep in reps:
            sources = [maps[k]["extras"]["sources10"] for k in replicate_keys[rep]]
            if len(sources) > 1:
                count, mean = reproducible_components(sources)
                add(results, **common, kind="method", representation=rep, variable="reproducible_components", value=count,
                    n=len(sources))
                add(results, **common, kind="method", representation=rep, variable="mean_matched_correlation", value=mean,
                    n=len(sources))
    if method == "pca":
        for x, a in enumerate(reps):
            for b in reps[x + 1:]:
                Sa, Sb = maps[main_key[a]]["extras"]["scores10"], maps[main_key[b]]["extras"]["scores10"]
                add(results, **common, kind="method", representation=f"{a}-{b}", variable="subspace_overlap10",
                    value=subspace_overlap(Sa, Sb))
                add(results, **common, kind="method", representation=f"{a}-{b}", variable="cka_scores10",
                    value=linear_cka(Sa, Sb))


def check_rows(gbs, per_plane, keep_no_boundary):
    expected = len(mc.distinct_families()) - (0 if keep_no_boundary else len(mc.NO_BOUNDARY) // 2)
    if not per_plane and len(gbs) != expected:
        raise ValueError(f"{len(gbs)} GBs found, {expected} expected; a descriptor run is incomplete")


def load_representations(args, gbs, phase):
    """Vectors of A-D (+ C0) for the 8x rows, and what the bulk test needs."""
    rows = interface_rows(gbs)
    keys = structure_keys(rows)
    standard = standard_vectors(args.descriptor, keys, root=args.descriptor_root)
    blocks = INVARIANT + VECTORIAL if phase == "1" else VECTORIAL
    covariant = covariant_vectors(args.covariant, keys, blocks, root=args.covariant_root)
    columns = covariant["columns"]
    lo = columns[VECTORIAL[0]][0]
    hi = columns[VECTORIAL[-1]][1]
    vectors = {"A": standard["A"], "B": standard["B"], "C": covariant["C"][:, lo:hi], "D": covariant["D"][:, lo:hi]}
    if phase == "1":
        vectors[CHECK] = covariant["C"]
    return rows, vectors, standard, covariant, (lo, hi)


def check_a_equals_x_r8(descriptor, gbs, A):
    """Anchor: A's edge and mid rows are the stored X_r8 rows of load_distinct()."""
    out = {}
    for p, planes in enumerate(sc.PLANE_NAMES):
        X, table, _ = mc.load_distinct(descriptor, "interface", 8.0, planes)
        lookup = dict(zip(table["family"], range(len(table))))
        stored = X[[lookup[f] for f in gbs["family"]]]
        out[planes] = float(np.abs(stored - A[p::2]).max())
    return out


def fifth_study(args, phase, out_dir):
    started = time.time()
    import pandas as pd

    covariant_index = pd.read_csv(covariant_dirs(args.covariant, args.covariant_root)[0] / "index.csv", dtype={"plane": str})
    available = covariant_index.loc[covariant_index["thickness"] == 8, "family"]
    gbs = gb_table(available, args.keep_no_boundary, args.families_per_plane)
    check_rows(gbs, args.families_per_plane, args.keep_no_boundary)
    subsets = [s for s in args.subsets if s == "pooled" or (gbs["plane"] == s).any()]
    methods = default_methods(args)
    if args.dry_run:
        rows = interface_rows(gbs)
        reps = len(REPRESENTATIONS) + (1 if phase == "1" else 0)
        print(f"phase {phase}: {len(gbs)} GBs, {len(rows)} rows, subsets {subsets}, methods {list(methods)}, "
              f"{reps} representations, {args.replicates} replicates; output {out_dir}")
        return 0
    rows, vectors, standard, covariant, vectorial_columns = load_representations(args, gbs, phase)
    geometry = gl.load_geometry(rows.assign(region=rows["interface"]), path=args.labels)
    codes = label_codes(rows, geometry)
    summary = {"study": "representation", "phase": phase, "n_gbs": len(gbs), "n_rows": len(rows),
               "gbs_per_plane": gbs["plane"].value_counts().sort_index().to_dict(),
               "no_boundary_dropped": not args.keep_no_boundary, "subsets": subsets, "methods": list(methods),
               "descriptor": args.descriptor, "covariant_run": sc.relative_path(covariant_dirs(args.covariant, args.covariant_root)[0]),
               "vector_lengths": {rep: int(X.shape[1]) for rep, X in vectors.items()},
               "covariant_block_shares": {"C": block_shares(covariant["interface"].reshape(-1, covariant["interface"].shape[-1]),
                                                            covariant["columns"]),
                                          "D": block_shares(covariant["cell"], covariant["columns"])},
               "settings": settings_record(args)}
    results = []
    if phase == "1":
        print("bulk test", flush=True)
        summary["bulk_test"] = bulk_test(standard, covariant, gbs, args.covariant, args.covariant_root, vectorial_columns)
        if Path(args.descriptor_root) == sc.STANDARD_ROOT:
            summary["check_A_equals_X_r8_max_abs_difference"] = check_a_equals_x_r8(args.descriptor, gbs, vectors["A"])
        for name, entry in summary["bulk_test"]["bulk_cell_averages"].items():
            block, kind, plane = name.split("|")
            add(results, study="representation", phase=phase, kind="bulk", method=block, subset=plane,
                representation=kind, variable="cell_average_over_atom_norm", value=entry["max"], main=entry["median"],
                n=entry["n"])
        for test in ("D_vs_C", "B_vs_A"):
            for variable, per_plane in summary["bulk_test"][test].items():
                for plane, entry in per_plane.items():
                    add(results, study="representation", phase=phase, kind="bulk", method=test, subset=plane,
                        variable=variable, value=entry["median"], main=entry["max"], noise_floor=entry["p95"], n=entry["n"])
    del standard, covariant
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
        paths, checks = {}, {}
        for rep, X in vectors.items():
            Z = exact_coordinates(X)
            checks[rep] = {"rank": int(Z.shape[1]), "max_relative_distance_change": distance_error(X, Z)}
            paths[rep] = str(Path(folder) / f"{rep}.npy")
            np.save(paths[rep], Z)
        summary["exact_coordinates"] = checks
        del vectors
        output = run_maps(args, phase, rows, codes, paths, subsets, methods, results, summary)
    summary["seconds"] = round(time.time() - started, 1)
    write_outputs(out_dir, results, output, rows, geometry, summary)
    return 0 if not summary.get("failed_maps") else 1


def settings_record(args):
    keep = ("replicates", "subsample", "ks", "organise_k", "trust_k", "clusters", "spearman_pairs", "triplets",
            "width_factor", "mixing", "sketchmap_hd", "sketchmap_ld", "permutations", "jobs")
    record = {name: getattr(args, name) for name in keep if hasattr(args, name)}
    record["descriptor_root"] = str(sc.relative_path(Path(args.descriptor_root)))
    record["covariant_root"] = str(sc.relative_path(Path(args.covariant_root)))
    return record


def worker_settings(args, n_rows):
    return {"n_rows": n_rows, "ks": tuple(args.ks), "organise_k": args.organise_k, "trust_k": args.trust_k,
            "triplets": args.triplets, "width_factor": args.width_factor, "mixing": args.mixing,
            "sketchmap_hd": tuple(args.sketchmap_hd), "sketchmap_ld": tuple(args.sketchmap_ld),
            "sketchmap_bin": str(args.sketchmap_bin), "permutations": args.permutations}


def run_maps(args, phase, rows, codes, paths, subsets, methods, results, summary):
    n = len(rows)
    members = subset_rows(rows, subsets)
    masks = {s: gb_masks(rows, members[s], args.replicates, args.subsample, seed=1000 + k)
             for k, s in enumerate(subsets)}
    representations = {s: list(REPRESENTATIONS) + ([CHECK] if phase == "1" and s == "pooled" else []) for s in subsets}
    samples = {s: tuple(members[s][x] for x in pair_sample(len(members[s]), local_partner(members[s], n),
                                                           args.spearman_pairs, seed=7))
               for s in subsets}
    settings = worker_settings(args, n)
    pooled_names = {"pooled"}

    print("raw descriptors", flush=True)
    raw_tasks = [{"subset": s, "representation": rep, "coordinates": rep, "rows": members[s], "pairs": samples[s]}
                 for s in subsets for rep in representations[s]]
    raw = run_pool(raw_task, raw_tasks, args.jobs, paths, codes, settings, "raw")
    raw_by = {(t["subset"], t["representation"]): t for t in raw}
    raw_changes = {}
    for s in subsets:
        baseline = majority_baselines(local_codes(codes, members[s]), s in pooled_names)
        for variable, value in baseline.items():
            add(results, study="representation", phase=phase, kind="baseline", subset=s, variable=variable, value=value)
        reps = representations[s]
        for rep in reps:
            for variable, value in raw_by[(s, rep)]["variables"].items():
                add(results, study="representation", phase=phase, kind="raw", method="raw", subset=s,
                    representation=rep, variable=variable, value=value)
        for x, a in enumerate(reps):
            for b in reps[x + 1:]:
                scores = score_features(raw_by[(s, a)]["features"], raw_by[(s, b)]["features"], settings["ks"], 0,
                                        coordinates=False)
                raw_changes[(s, a, b)] = {v: change(v, score) for v, score in scores.items()}
                Za = np.load(paths[a], mmap_mode="r")[members[s]]
                Zb = np.load(paths[b], mmap_mode="r")[members[s]]
                raw_changes[(s, a, b)]["cka"] = 1.0 - linear_cka(np.asarray(Za), np.asarray(Zb))
                for variable, value in raw_changes[(s, a, b)].items():
                    add(results, study="representation", phase=phase, kind="raw_pair", method="raw", subset=s,
                        representation=f"{a}-{b}", variable=variable, value=value)

    tasks = by_cost(map_tasks(methods, subsets, members, masks, representations, args.replicates))
    print(f"{len(tasks)} maps", flush=True)
    done = run_pool(map_task, tasks, args.jobs, paths, codes, settings, "maps")
    failed = [{k: t[k] for k in ("method", "subset", "representation", "replicate", "error")} for t in done if t["error"]]
    summary["failed_maps"] = failed
    for entry in failed:
        print(f"  map failed: {entry}", file=sys.stderr)
    maps = {map_key(t["method"], t["subset"], t["representation"], t["replicate"]): t for t in done if not t["error"]}
    summary["map_seconds"] = {m: round(sum(t["seconds"] for t in done if t["method"] == m), 1) for m in methods}

    compare = comparison_tasks(maps, samples, args.clusters, args.chunk)
    print(f"{sum(len(t['pairs']) for t in compare)} map pairs in {len(compare)} chunks", flush=True)
    scores = {}
    for chunk in run_pool(compare_task, compare, args.jobs, paths, codes, settings, "pairs"):
        for i, j, s in chunk:
            scores[(i, j)] = s
    for method in methods:
        for s in subsets:
            if any(m["method"] == method and m["subset"] == s for m in maps.values()):
                group_scores = {pair: v for pair, v in scores.items() if maps[pair[0]]["method"] == method
                                and maps[pair[0]]["subset"] == s}
                summarise_group(results, phase, method, s, maps, group_scores, raw_changes, raw_by, settings)
    main_maps = {f"{m['method']}|{m['subset']}|{m['representation']}": m["Y"] for m in maps.values()
                 if m["main"]}
    main_maps.update({f"rows|{s}": members[s] for s in subsets})
    return main_maps


def write_outputs(out_dir, results, maps, rows, geometry, summary):
    import pandas as pd

    table = pd.DataFrame(results, columns=list(COLUMNS))
    mc.write_table(out_dir / "results.csv", table)
    arrays = {key: np.asarray(value) for key, value in maps.items()}
    arrays.update({f"table|{name}": rows[name].to_numpy().astype(str) for name in ("family", "plane", "grain_class", "interface")})
    arrays.update({"table|gb": rows["gb"].to_numpy(), "table|thickness": rows["thickness"].to_numpy()})
    if geometry is not None:
        arrays.update({f"table|{name}": geometry[name].to_numpy(dtype=np.float64) for name in gl.TARGETS})
    sc.write_npz(out_dir / "maps.npz", **arrays)
    mc.write_json(out_dir / "summary.json", {**summary, **mc.run_metadata()})
    print(f"wrote {out_dir}/results.csv ({len(table)} rows), maps.npz, summary.json")


# --------------------------------------------------------------------------- thickness check

def thickness_check(args, out_dir):
    """Every method on all thicknesses of a fixed subset of GBs; PCA on every structure."""
    started = time.time()
    import pandas as pd

    covariant_index = pd.read_csv(covariant_dirs(args.covariant, args.covariant_root)[0] / "index.csv", dtype={"plane": str})
    gbs = gb_table(covariant_index.loc[covariant_index["thickness"] == 8, "family"], args.keep_no_boundary,
                   args.families_per_plane)
    check_rows(gbs, args.families_per_plane, args.keep_no_boundary)
    rng = np.random.default_rng(11)
    chosen = []
    for plane, group in gbs.groupby("plane"):
        count = max(1, int(round(args.thickness_gbs * len(group) / len(gbs))))
        chosen.extend(rng.choice(group.index.to_numpy(), min(count, len(group)), replace=False).tolist())
    subset = gbs.loc[sorted(chosen)].reset_index(drop=True)
    present = set(zip(covariant_index["family"], covariant_index["thickness"]))
    thicknesses = sorted(covariant_index["thickness"].unique())
    skip = {(f, t) for f in subset["family"] for t in thicknesses if (f, t) not in present}
    rows = interface_rows(subset, thicknesses, skip)
    methods = default_methods(args)
    if args.dry_run:
        print(f"thickness check: {len(subset)} GBs, {len(rows)} rows ({len(thicknesses)} thicknesses), methods "
              f"{list(methods)}; output {out_dir}")
        return 0
    keys = structure_keys(rows)
    standard = standard_vectors(args.descriptor, keys, root=args.descriptor_root)
    covariant = covariant_vectors(args.covariant, keys, VECTORIAL, root=args.covariant_root)
    vectors = {"A": standard["A"], "B": standard["B"], "C": covariant["C"], "D": covariant["D"]}
    del standard, covariant
    geometry = gl.load_geometry(rows.assign(region=rows["interface"]), path=args.labels)
    codes = label_codes(rows, geometry)
    summary = {"study": "representation", "phase": "thickness", "n_gbs": len(subset), "n_rows": len(rows),
               "thicknesses": [int(t) for t in thicknesses], "methods": list(methods), "settings": settings_record(args)}
    results, maps = [], {}
    thickness = rows["thickness"].to_numpy()
    family = rows["family"].to_numpy()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
        paths = {}
        for rep, X in vectors.items():
            paths[rep] = str(Path(folder) / f"{rep}.npy")
            np.save(paths[rep], exact_coordinates(X))
        del vectors
        settings = worker_settings(args, len(rows))
        everything = np.arange(len(rows))
        tasks = by_cost([{"method": m, "subset": "pooled", "representation": rep, "coordinates": rep, "rows": everything,
                          "seed": SEEDS[0], "main": False, "replicate": 0, "variables": False}
                         for m in methods for rep in REPRESENTATIONS])
        print(f"{len(tasks)} maps on {len(rows)} rows", flush=True)
        done = run_pool(map_task, tasks, args.jobs, paths, codes, settings, "thickness maps")
        for rep in REPRESENTATIONS:
            Z = np.load(paths[rep])
            done.append({"method": "raw", "representation": rep, "Y": Z, "error": None})
        for t in done:
            if t["error"]:
                summary.setdefault("failed_maps", []).append({k: t[k] for k in ("method", "representation", "error")})
                continue
            for variable, value in thickness_variables(t["Y"], thickness, family).items():
                add(results, study="representation", phase="thickness", kind="thickness", method=t["method"],
                    subset="pooled", representation=t["representation"], variable=variable, value=value, n=len(rows))
            if t["method"] != "raw":
                maps[f"thickness|{t['method']}|{t['representation']}"] = t["Y"]
    print("PCA on every structure", flush=True)
    summary["all_structures_pca"] = all_structures_pca(args, results)
    summary["seconds"] = round(time.time() - started, 1)
    write_outputs(out_dir, results, maps, rows, geometry, summary)
    return 0 if not summary.get("failed_maps") else 1


def thickness_variables(Y, thickness, family, k=8):
    """How much of a map follows the thickness, and how much the GB."""
    Y = np.asarray(Y, dtype=np.float64)
    partner = partners(len(Y))
    N = neighbour_sets(square_distances(Y), partner, k)
    return {"thickness_share": thickness_share(Y, thickness), "gb_share": thickness_share(Y, family),
            "same_gb_neighbours": float(np.mean(family[N] == family[:, None]))}


def all_structures_pca(args, results):
    """Randomised PCA (float32) of every structure's interface rows, per representation."""
    import pandas as pd
    from sklearn.decomposition import PCA

    out = {}
    interface_dir, _ = covariant_dirs(args.covariant, args.covariant_root)
    index = pd.read_csv(interface_dir / "index.csv", dtype={"plane": str})
    if args.families_per_plane:
        index = index[index["family"].isin(first_syms(args.families_per_plane))]
    if not args.keep_no_boundary:
        index = index[~index["family"].isin(mc.NO_BOUNDARY)]
    keys = list(zip(index["family"], index["thickness"]))
    thickness = np.repeat(index["thickness"].to_numpy(), 2)
    family = np.repeat(index["family"].to_numpy(), 2)
    for rep in REPRESENTATIONS:
        if rep in ("A", "B"):
            data = standard_vectors(args.descriptor, keys, dtype=np.float32, root=args.descriptor_root)
        else:
            data = covariant_vectors(args.covariant, keys, VECTORIAL, root=args.covariant_root, dtype=np.float32)
        X = data[rep]
        del data
        model = PCA(n_components=min(10, len(X) - 1), svd_solver="randomized", random_state=0)
        scores = model.fit_transform(X)
        entry = {"n_rows": int(len(X)), "variance_ratio": model.explained_variance_ratio_.tolist()}
        for name, columns in (("PC1", scores[:, :1]), ("PC2", scores[:, 1:2]), ("PC1-10", scores)):
            entry[name] = {"thickness_share": thickness_share(columns, thickness), "gb_share": thickness_share(columns, family)}
            for variable in ("thickness_share", "gb_share"):
                add(results, study="representation", phase="thickness", kind="all_structures_pca", method="pca",
                    subset=name, representation=rep, variable=variable, value=entry[name][variable], n=len(X))
        out[rep] = entry
        del X, scores
    return out


# --------------------------------------------------------------------------- the sixth study

def grid_tags(args):
    return [f"rc{r:.1f}_n8_l{l}_s{s:g}" for r in args.grid_r_cut for l in args.grid_l_max for s in args.grid_sigma]


def settings_grid(args, out_dir):
    started = time.time()
    import pandas as pd

    tags = grid_tags(args)
    covariant_runs = {tag: f"{tag}_t8" for tag in tags}
    covariant_runs[REFERENCE_TAG] = args.covariant
    available = None
    for run in covariant_runs.values():
        index = pd.read_csv(covariant_dirs(run, args.covariant_root)[0] / "index.csv", dtype={"plane": str})
        families = set(index.loc[index["thickness"] == 8, "family"])
        available = families if available is None else available & families
    gbs = gb_table(available, args.keep_no_boundary, args.families_per_plane)
    check_rows(gbs, args.families_per_plane, args.keep_no_boundary)
    rows = interface_rows(gbs)
    keys = structure_keys(rows)
    methods = default_methods(args)
    if args.dry_run:
        print(f"settings grid: {len(tags)} settings + anchor, {len(gbs)} GBs, {len(rows)} rows, methods {list(methods)}, "
              f"{args.replicates} replicates; output {out_dir}")
        return 0
    n = len(rows)
    codes = label_codes(rows)
    settings = worker_settings(args, n)
    members = np.arange(n)
    masks = gb_masks(rows, members, args.replicates, args.subsample, seed=1000)
    sample = tuple(members[x] for x in pair_sample(n, partners(n), args.spearman_pairs, seed=7))
    summary = {"study": "settings_grid", "tags": tags, "anchor": REFERENCE_TAG, "covariant_runs": covariant_runs,
               "n_gbs": len(gbs), "n_rows": n, "methods": list(methods), "settings": settings_record(args)}
    results = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
        paths = {}
        for tag in tags + [REFERENCE_TAG]:
            standard = standard_vectors(tag, keys, root=args.descriptor_root)
            covariant = covariant_vectors(covariant_runs[tag], keys, VECTORIAL, root=args.covariant_root)
            for rep, X in (("A", standard["A"]), ("B", standard["B"]), ("C", covariant["C"]), ("D", covariant["D"])):
                paths[f"{tag}|{rep}"] = str(Path(folder) / f"{tag}_{rep}.npy")
                np.save(paths[f"{tag}|{rep}"], exact_coordinates(X))
            del standard, covariant
            print(f"  loaded {tag}", flush=True)
        sets = list(paths)
        raw = run_pool(raw_task, [{"subset": "pooled", "representation": key, "coordinates": key, "rows": members,
                                   "pairs": sample, "organise": False} for key in sets],
                       args.jobs, paths, codes, settings, "raw")
        raw_features = {t["representation"]: t["features"] for t in raw}
        tasks = []
        for method in methods:
            for key in sets:
                for r in range(args.replicates):
                    stochastic = method in STOCHASTIC
                    tasks.append({"method": method, "subset": "pooled", "representation": key, "coordinates": key,
                                  "replicate": r, "seed": seed_list(args.replicates)[r] if stochastic else 0,
                                  "rows": members if stochastic else masks[r], "main": r == 0, "variables": False})
        tasks = by_cost(tasks)
        print(f"{len(tasks)} maps", flush=True)
        done = run_pool(map_task, tasks, args.jobs, paths, codes, settings, "maps")
    failed = [{k: t[k] for k in ("method", "representation", "replicate", "error")} for t in done if t["error"]]
    summary["failed_maps"] = failed
    maps = {f"{t['method']}|{t['representation']}|{t['replicate']}": t for t in done if not t["error"]}
    pairs_by_method = {}
    for method in methods:
        keys_m = sorted(k for k in maps if maps[k]["method"] == method)
        pairs_by_method[method] = [(a, b) for x, a in enumerate(keys_m) for b in keys_m[x + 1:]]
    compare = []
    for method, pairs in pairs_by_method.items():
        for part in chunked(pairs, args.chunk):
            needed = {k for pair in part for k in pair}
            compare.append({"maps": {k: {"rows": maps[k]["rows"], "Y": maps[k]["Y"]} for k in needed},
                            "pairs": part, "sample": sample, "clusters": 0})
    print(f"{sum(len(t['pairs']) for t in compare)} map pairs", flush=True)
    scores = {}
    for chunk in run_pool(compare_task, compare, args.jobs, paths, codes, settings, "pairs"):
        for i, j, s in chunk:
            scores[(i, j)] = s
    arrays = summarise_grid(results, methods, maps, scores, raw_features, sets, tags, settings, args)
    arrays.update({f"map|{k}": m["Y"] for k, m in maps.items() if m["replicate"] == 0})
    summary["seconds"] = round(time.time() - started, 1)
    write_outputs(out_dir, results, arrays, rows, None, summary)
    return 0 if not failed else 1


def summarise_grid(results, methods, maps, scores, raw_features, sets, tags, settings, args):
    """PERMANOVA shares (representation x settings, replicated) per method and variable; anchor distances."""
    arrays = {}
    common = {"study": "settings_grid", "phase": "grid", "subset": "pooled"}
    grid_sets = [s for s in sets if s.split("|")[0] in tags]
    for method in methods + ["raw"]:
        if method == "raw":
            keys = grid_sets
            all_keys = sets  # the grid plus the reference anchor
            names = {k: k for k in all_keys}
            pair_scores = {}
            for x, a in enumerate(sets):
                for b in sets[x + 1:]:
                    pair_scores[(a, b)] = score_features(raw_features[a], raw_features[b], settings["ks"], 0, coordinates=False)
        else:
            all_keys = sorted(k for k in maps if maps[k]["method"] == method)
            keys = [k for k in all_keys if maps[k]["representation"].split("|")[0] in tags]
            names = {k: maps[k]["representation"] for k in all_keys}
            pair_scores = {pair: s for pair, s in scores.items() if pair[0] in maps and maps[pair[0]]["method"] == method}
        variables = sorted({v for s in pair_scores.values() for v in s})
        for variable in variables:
            def value(a, b):
                s = pair_scores.get((a, b)) or pair_scores.get((b, a))
                return change(variable, s[variable]) if s is not None else np.nan

            D = np.array([[0.0 if a == b else value(a, b) for b in keys] for a in keys])
            arrays[f"dissimilarity|{method}|{variable}"] = D
            rep = [names[k].split("|")[1] for k in keys]
            setting = [names[k].split("|")[0] for k in keys]
            metric = np.sqrt(np.clip(D, 0, None)) if variable == "procrustes" else D
            if np.isfinite(metric).all():
                effects = permanova(metric, rep, setting, ("representation", "settings"), args.permutations)
                for factor, entry in effects.items():
                    add(results, **common, kind="effect", method=method, representation="all", variable=variable,
                        factor="seed_noise" if factor == "residual" and method != "raw" else factor,
                        value=entry["share"], main=entry["pseudo_f"], p_value=entry["p_value"], n=len(keys))
            same_rep = [D[i, j] for i in range(len(keys)) for j in range(i + 1, len(keys)) if rep[i] == rep[j] and setting[i] != setting[j]]
            same_setting = [D[i, j] for i in range(len(keys)) for j in range(i + 1, len(keys)) if rep[i] != rep[j] and setting[i] == setting[j]]
            floor = [D[i, j] for i in range(len(keys)) for j in range(i + 1, len(keys)) if rep[i] == rep[j] and setting[i] == setting[j]]
            for label, values in (("between_settings", same_rep), ("between_representations", same_setting), ("replicates", floor)):
                if values:
                    add(results, **common, kind="grid_mean", method=method, representation=label, variable=variable,
                        value=float(np.mean(values)), sd=float(np.std(values)),
                        noise_floor=float(np.percentile(floor, 95)) if floor else np.nan, n=len(values))
            anchors = {}  # representation -> the reference setting's maps (all replicates)
            for k in all_keys:
                if names[k].split("|")[0] == REFERENCE_TAG:
                    anchors.setdefault(names[k].split("|")[1], []).append(k)
            for anchor_rep, anchor_keys in anchors.items():
                for tag in tags:
                    others = [k for k in keys if names[k] == f"{tag}|{anchor_rep}"]
                    values = [v for v in (value(a, k) for a in anchor_keys for k in others) if np.isfinite(v)]
                    if values:
                        add(results, **common, kind="anchor", method=method, representation=anchor_rep,
                            variable=variable, factor=tag, value=float(np.mean(values)), n=len(values))
    return arrays


# --------------------------------------------------------------------------- command line

def default_methods(args):
    if args.methods:
        return list(args.methods)
    if args.study == "settings_grid":
        return list(GRID_METHODS)
    return list(ALL_METHODS if args.phase == "thickness" else PHASE_METHODS[args.phase])


def missing_runs(args):
    """Descriptor runs this study needs that do not exist yet."""
    tags = grid_tags(args) + [REFERENCE_TAG] if args.study == "settings_grid" else [args.descriptor]
    missing = [str(sc.run_dir(region, tag, root=args.descriptor_root)) for tag in tags for region in sc.REGIONS
               if not sc.run_dir(region, tag, root=args.descriptor_root).is_dir()]
    runs = ([f"{tag}_t8" for tag in grid_tags(args)] + [args.covariant]) if args.study == "settings_grid" else [args.covariant]
    missing += [str(folder) for run in runs for folder in covariant_dirs(run, args.covariant_root) if not folder.is_dir()]
    return missing


def pair_of_numbers(text):
    a, b = (float(x) for x in text.split(","))
    return a, b


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=("representation", "settings_grid"), default="representation")
    parser.add_argument("--phase", choices=("1", "2", "thickness"), default="1",
                        help="fifth study only: 1 = bulk test + PCA/UMAP, 2 = the other methods, thickness (default 1)")
    parser.add_argument("--descriptor", default=REFERENCE_TAG, help=f"standard SOAP tag (default {REFERENCE_TAG})")
    parser.add_argument("--covariant", default=REFERENCE_TAG,
                        help=f"covariant run name or folder with lambda = 1 and 2 blocks (default {REFERENCE_TAG})")
    parser.add_argument("--descriptor-root", type=Path, default=sc.STANDARD_ROOT,
                        help="folder holding global/ and both_interfaces/ of the standard runs (smoke tests)")
    parser.add_argument("--covariant-root", type=Path, default=sc.COVARIANT_ROOT)
    parser.add_argument("--labels", type=Path, default=gl.OUT_DIR / gl.CSV_NAME, help="geometry_labels.csv to use")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=None,
                        help="default: the phase's methods (or PCA, MDS, UMAP, diffusion maps for the grid)")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--replicates", type=int, default=None, help="seeds or subsamples per map (default 10; grid 5)")
    parser.add_argument("--subsample", type=float, default=0.8, help="share of GBs in each subsample (default 0.8)")
    parser.add_argument("--ks", type=int, nargs="+", default=[15, 50], help="neighbour counts for Jaccard (default 15 50)")
    parser.add_argument("--organise-k", type=int, default=10, help="neighbours for kNN accuracy and R^2 (default 10)")
    parser.add_argument("--trust-k", type=int, default=15, help="neighbours for trustworthiness/continuity (default 15)")
    parser.add_argument("--clusters", type=int, default=8, help="Ward clusters for ARI (default 8)")
    parser.add_argument("--spearman-pairs", type=int, default=200000, help="row pairs sampled for Spearman (default 200000)")
    parser.add_argument("--triplets", type=int, default=10000, help="random triplets per map (default 10000)")
    parser.add_argument("--width-factor", type=float, default=1.0,
                        help="kernel width, diffusion bandwidth and sketch-map sigma = factor x median distance (default 1)")
    parser.add_argument("--mixing", type=float, default=0.5, help="PCovR mixing for the map (default 0.5)")
    parser.add_argument("--sketchmap-hd", type=pair_of_numbers, default=(2.0, 6.0), help="a,b of the high-dim sigmoid (default 2,6)")
    parser.add_argument("--sketchmap-ld", type=pair_of_numbers, default=(1.0, 2.0), help="a,b of the low-dim sigmoid (default 1,2)")
    parser.add_argument("--sketchmap-bin", type=Path, default=SKETCHMAP)
    parser.add_argument("--permutations", type=int, default=999, help="PERMANOVA permutations (default 999)")
    parser.add_argument("--thickness-gbs", type=int, default=300, help="GBs in the thickness check (default 300)")
    parser.add_argument("--grid-r-cut", type=float, nargs="+", default=list(GRID["r_cut"]))
    parser.add_argument("--grid-l-max", type=int, nargs="+", default=list(GRID["l_max"]))
    parser.add_argument("--grid-sigma", type=float, nargs="+", default=list(GRID["sigma"]))
    parser.add_argument("--families-per-plane", type=int, default=0,
                        help="only the first N sym folders per plane (smoke tests; the 1,111-GB check is skipped)")
    parser.add_argument("--keep-no-boundary", action="store_true", help="keep D0985 (111/sym00119, sym00175)")
    parser.add_argument("--chunk", type=int, default=60, help="map pairs per worker task (default 60)")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default representation_study/phase<N> or settings_grid/)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing results.csv")
    parser.add_argument("--dry-run", action="store_true", help="count rows and maps and stop")
    args = parser.parse_args(argv)
    if args.replicates is None:
        args.replicates = 5 if args.study == "settings_grid" else 10
    if args.study == "settings_grid":
        out_dir = args.out_dir or GRID_DIR
    else:
        out_dir = args.out_dir or OUT_DIR / ("thickness" if args.phase == "thickness" else f"phase{args.phase}")
    if (out_dir / "results.csv").exists() and not args.overwrite and not args.dry_run:
        print(f"{out_dir / 'results.csv'} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    missing = missing_runs(args)
    if missing:
        print("missing descriptor runs:\n  " + "\n  ".join(missing), file=sys.stderr)
        if args.dry_run:
            return 0
        return 2
    if "sketchmap" in default_methods(args) and not Path(args.sketchmap_bin).is_file():
        print(f"sketch-map binary {args.sketchmap_bin} not found; build it (src/third_party/README.md) or leave "
              "sketchmap out with --methods", file=sys.stderr)
        return 2
    if args.study == "settings_grid":
        return settings_grid(args, out_dir)
    if args.phase == "thickness":
        return thickness_check(args, out_dir)
    return fifth_study(args, args.phase, out_dir)


if __name__ == "__main__":
    sys.exit(main())
