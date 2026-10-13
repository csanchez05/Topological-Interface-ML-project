"""Shared helpers for the Stage 1 analysis scripts (steps 1.4 and 1.5).

Loads one SOAP vector per grain-boundary family through soap_common.load_gb,
measures relative distances between vectors, groups vectors that coincide,
and writes tables atomically.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOAP_SCRIPTS = PROJECT_ROOT / "src" / "dscribe" / "CoGe"
# soap_common lives in a plain folder (no package), so it is imported by path.
if str(SOAP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOAP_SCRIPTS))

import soap_common as sc  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "calculations" / "ML_dscribe" / "CoGe"
LABELS_DIR = OUTPUT_ROOT / "labels"
DEFAULT_DESCRIPTOR = "rc6.0_n8_l6_s0.5"
DEFAULT_TOL = 1e-4
# Stage 1 census: which families hold the same structure (structure_class), from step 1.5.
CENSUS = OUTPUT_ROOT / DEFAULT_DESCRIPTOR / "interface_r8_both_t4" / "distinct_gbs.csv"
KINDS = ("interface", "nonbulk", "global")
PLANE_CHOICES = ("both", "edge", "mid")
# Two 111 families whose operation is a lattice translation of the cubic cell: the cell is plain
# bulk with no boundary, yet labelled grain class E. Together they are one distinct GB (D0985).
# The loaders keep them (select_gbs.py's lists and the Stage 1.5 census count 1,112 GBs);
# krr_classify.py, selection_benchmark.py and the newer scripts drop them unless asked not to.
NO_BOUNDARY = ("111/sym00119", "111/sym00175")


# --------------------------------------------------------------------------- command line

def add_vector_arguments(parser):
    group = parser.add_argument_group("which vector")
    group.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR,
                       help="descriptor run: a tag under calculations/descriptors/dscribe/standard/ or a "
                            f"run folder of either region (default {DEFAULT_DESCRIPTOR})")
    group.add_argument("--kind", default="interface", choices=KINDS,
                       help="interface: atoms within --radius of a plane; nonbulk: atoms whose SOAP differs "
                            "from bulk; global: every atom (default interface)")
    group.add_argument("--radius", type=float, default=8.0,
                       help="interface radius in A, one of the stored radii (default 8)")
    group.add_argument("--planes", default="both", choices=PLANE_CHOICES,
                       help="both: mean of the two interfaces in the cell; edge or mid: one of them (default both)")
    group.add_argument("--thickness", type=thickness_argument, default=4,
                       help="the one thickness used for every GB (default 4, the smallest), or 'all' "
                            "to pool every thickness (krr_classify.py and pooled_duplicates.py only)")
    group.add_argument("--out-dir", type=Path, default=None,
                       help="default: calculations/ML_dscribe/CoGe/<descriptor tag>/<vector tag>/")


def thickness_argument(text):
    """A thickness (4..12) or the word all."""
    return "all" if str(text) == "all" else int(text)


def descriptor_tag(descriptor):
    """Folder name of the descriptor run, e.g. rc6.0_n8_l6_s0.5."""
    return sc.run_dir("both_interfaces", descriptor).resolve().name


def region_of(kind):
    """The descriptor folder that holds a vector kind."""
    return "global" if kind == "global" else "both_interfaces"


def vector_tag(kind, radius, planes, thickness):
    """Name for one choice of vector, e.g. interface_r8_both_t4."""
    parts = [kind]
    if kind == "interface":
        parts.append(f"r{radius:g}")
    if kind != "global":
        parts.append(planes)
    parts.append(f"t{thickness}")
    return "_".join(parts)


def output_dir(args):
    if args.out_dir is not None:
        return Path(args.out_dir)
    return (OUTPUT_ROOT / descriptor_tag(args.descriptor)
            / vector_tag(args.kind, args.radius, args.planes, args.thickness))


def vector_settings(args):
    return {"descriptor": descriptor_tag(args.descriptor),
            "descriptor_dir": sc.relative_path(sc.run_dir(region_of(args.kind), args.descriptor)),
            "kind": args.kind, "radius": args.radius if args.kind == "interface" else None,
            "planes": args.planes if args.kind != "global" else None, "thickness": args.thickness}


# --------------------------------------------------------------------------- vectors

def load_all_vectors(descriptor, kind="interface", radius=8.0, planes="both"):
    """Every structure of the run: (X as stored, float32; index DataFrame)."""
    folder = sc.run_dir(region_of(kind), descriptor)
    name = sc.GLOBAL_FILE if kind == "global" else sc.INTERFACE_FILE
    if not (folder / name).is_file():
        raise FileNotFoundError(f"no {name} in {folder}; run gb_descriptor.py first or pass --descriptor")
    X, index = sc.load_gb(folder, kind=kind, radius=radius if kind == "interface" else None, planes=planes)
    return X, index.reset_index(drop=True)


def select_thickness(X, index, thickness, kind="interface", radius=8.0):
    """One row per family at the chosen thickness, or every row when thickness is "all".

    Returns (X, index, notes). Families without that thickness, and rows with
    an undefined vector, are dropped and listed in notes.
    """
    notes = {}
    if thickness == "all":
        chosen = np.ones(len(index), dtype=bool)
    else:
        chosen = (index["thickness"] == thickness).to_numpy()
    missing = sorted(set(index["family"]) - set(index.loc[chosen, "family"]))
    if missing:
        notes["families_without_this_thickness"] = missing
    finite = np.isfinite(X).all(axis=1)
    undefined = index.loc[chosen & ~finite, "family"].tolist()
    if undefined:
        notes["families_with_undefined_vector"] = undefined
    keep = chosen & finite
    X, index = np.asarray(X[keep], dtype=np.float64), index.loc[keep].reset_index(drop=True)
    if thickness != "all" and index["family"].duplicated().any():
        raise ValueError("a family appears more than once at one thickness")
    if kind == "interface":
        # Past Lx/4 the slabs around the two planes meet and the vector stops being interface-centred.
        too_thin = index.loc[index["Lx"] / 4.0 < radius, "family"].tolist()
        if too_thin:
            notes["families_where_radius_exceeds_quarter_cell"] = len(too_thin)
    return X, index, notes


def load_family_vectors(descriptor, kind="interface", radius=8.0, planes="both", thickness=4):
    """(X, index, notes): one float64 vector per family at one thickness."""
    X, index = load_all_vectors(descriptor, kind, radius, planes)
    return select_thickness(X, index, thickness, kind, radius)


def thickness_noise(X, index, thickness, kind="interface", radius=8.0):
    """How much one family's vector changes between the chosen thickness and the others.

    Only meaningful for interface-centred vectors. Thicknesses whose cell is too
    thin for the radius are left out. Returns None when nothing can be compared.
    """
    if kind == "global":
        return None
    usable = np.isfinite(X).all(axis=1)
    if kind == "interface":
        usable &= (index["Lx"].to_numpy() / 4.0 >= radius)
    family = index["family"].to_numpy()
    thick = index["thickness"].to_numpy()
    reference = {family[i]: i for i in np.flatnonzero(usable & (thick == thickness))}
    differences = []
    for i in np.flatnonzero(usable & (thick != thickness)):
        j = reference.get(family[i])
        if j is not None:
            differences.append(relative_distance(X[i], X[j]))
    if not differences:
        return None
    differences = np.array(differences)
    return {"n_comparisons": int(differences.size), "max": float(differences.max()),
            "p99": float(np.percentile(differences, 99)), "median": float(np.median(differences))}


def load_distinct(descriptor=DEFAULT_DESCRIPTOR, kind="interface", radius=8.0, planes="both", thickness=8,
                  allow_undefined=False):
    """One vector per distinct GB (1,112): the alphabetically first family of each structure class.

    Returns (X float64, table, bulk). Rows are in family order, as in select_gbs.py.
    table has family, plane, sym, structure_class, grain_class, handedness and source;
    bulk is the pristine vector of the same SOAP setting. An undefined vector (a
    nonbulk vector with no nonbulk atoms) is an error, or with allow_undefined a row of NaN.
    """
    import pandas as pd

    X_all, index_all = load_all_vectors(descriptor, kind, radius, planes)
    X, index, notes = select_thickness(X_all, index_all, thickness, kind, radius)
    undefined = notes.get("families_with_undefined_vector", [])
    if notes.get("families_without_this_thickness") or (undefined and not allow_undefined):
        raise ValueError(f"incomplete {thickness}x catalogue: {notes}")
    if undefined:
        rows = ((index_all["thickness"] == thickness) & index_all["family"].isin(undefined)).to_numpy()
        X = np.vstack([X, np.full((rows.sum(), X.shape[1]), np.nan)])
        index = pd.concat([index, index_all.loc[rows]], ignore_index=True)
    census = pd.read_csv(CENSUS, dtype={"plane": str})
    labels = pd.read_csv(LABELS_DIR / "gb_labels.csv", dtype={"plane": str})
    keep = census.sort_values("family").drop_duplicates("structure_class")[["family", "structure_class"]]
    rows = index.reset_index(drop=True).assign(row=np.arange(len(index))).merge(keep, on="family", validate="one_to_one")
    rows = rows.merge(labels[["family", "grain_class", "handedness"]], on="family", validate="one_to_one")
    rows = rows.sort_values("family").reset_index(drop=True)
    table = rows[["family", "plane", "sym", "structure_class", "grain_class", "handedness", "source"]]
    X_bulk, bulk_index = sc.load_bulk(sc.run_dir("global", descriptor))
    bulk = X_bulk[(bulk_index["kind"] == "pristine").to_numpy()][0].astype(np.float64)
    return X[rows["row"].to_numpy()], table, bulk


def distinct_families():
    """The family that stands for each distinct GB in load_distinct(): the alphabetically first of its class."""
    import pandas as pd

    census = pd.read_csv(CENSUS, dtype={"plane": str})
    return census.sort_values("family").drop_duplicates("structure_class")["family"].tolist()


def no_boundary_rows(table):
    """True for the rows of the NO_BOUNDARY families (table: anything with a family column)."""
    return table["family"].isin(NO_BOUNDARY).to_numpy()


def drop_no_boundary(X, table):
    """(X, table) without the NO_BOUNDARY families; the table is renumbered from 0."""
    keep = ~no_boundary_rows(table)
    return X[keep], table.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- selection in the full space

def squared_distances_to(X, x):
    return np.square(X - x).sum(axis=1)


def farthest_point_sampling(X, count, start):
    """Farthest-point sampling in the full vector space (Euclidean).

    start is the first pick; every next pick is the row farthest from all picks
    so far. Ties go to the lowest row. Returns row indices in pick order.
    """
    X = np.asarray(X, dtype=np.float64)
    if not 1 <= count <= len(X):
        raise ValueError("count must be between 1 and the number of rows")
    chosen = [int(start)]
    nearest = np.full(len(X), np.inf)
    for _ in range(count - 1):
        nearest = np.minimum(nearest, squared_distances_to(X, X[chosen[-1]]))
        nearest[chosen] = -np.inf
        chosen.append(int(np.argmax(nearest)))
    return np.array(chosen, dtype=int)


def farthest_from(X, reference):
    """Row farthest from a reference vector (the usual FPS start: farthest from bulk)."""
    return int(np.argmax(squared_distances_to(np.asarray(X, dtype=np.float64), np.asarray(reference, dtype=np.float64))))


def kmeans_members(X, count, seed=0, n_init=10):
    """K-means with `count` clusters; for each cluster the member nearest its centre.

    This is select_gbs.py's "median" rule, in the full vector space. Ties go to the lowest row.
    """
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=np.float64)
    model = KMeans(n_clusters=count, n_init=n_init, random_state=seed).fit(X)
    chosen = []
    for cluster, centre in enumerate(model.cluster_centers_):
        members = np.flatnonzero(model.labels_ == cluster)
        chosen.append(int(members[np.argmin(squared_distances_to(X[members], centre))]))
    return np.array(chosen, dtype=int)


def gp_variance_selection(X, count, gamma, width, start):
    """Greedy uncertainty sampling for a Gaussian process with kernel exp(-gamma d^2 / width).

    Each pick is the row whose predicted value is most uncertain given the picks so
    far. The uncertainty does not depend on any label, so this is the label-free
    first step of Bayesian active learning. Computed as a pivoted Cholesky
    factorisation. Ties go to the lowest row.
    """
    X = np.asarray(X, dtype=np.float64)
    variance = np.ones(len(X))
    factors = np.zeros((count, len(X)))
    chosen = []
    for j in range(count):
        i = int(start) if j == 0 else int(np.argmax(variance))
        chosen.append(i)
        column = np.exp(-gamma * squared_distances_to(X, X[i]) / width)
        factors[j] = (column - factors[:j].T @ factors[:j, i]) / np.sqrt(max(variance[i], 1e-12))
        variance = variance - factors[j] ** 2
        variance[chosen] = -np.inf
    return np.array(chosen, dtype=int)


# --------------------------------------------------------------------------- distances and groups

def relative_distance(a, b):
    """|a - b| divided by the mean of |a| and |b|."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b))))


def relative_distances(X):
    """Matrix of relative distances between the rows of X.

    The fast formula loses precision for nearly equal rows (it cannot resolve
    relative distances below about 1e-7), so every pair under 1e-3 is recomputed
    from the difference of the two rows.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    squared = norms[:, None] ** 2 + norms[None, :] ** 2 - 2.0 * (X @ X.T)
    scale = 0.5 * (norms[:, None] + norms[None, :])
    D = np.sqrt(np.maximum(squared, 0.0)) / scale
    close_i, close_j = np.nonzero(np.triu(D < 1e-3, 1))
    for i, j in zip(close_i, close_j):
        D[i, j] = D[j, i] = np.linalg.norm(X[i] - X[j]) / scale[i, j]
    np.fill_diagonal(D, 0.0)
    return D


def coincidence_groups(D, tol):
    """Group rows linked by a chain of pairs closer than tol. Returns one group number per row.

    Groups are numbered by their first member, so the numbering follows row order.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    _, raw = connected_components(csr_matrix(D <= tol), directed=False)
    order = {}
    return np.array([order.setdefault(g, len(order)) for g in raw])


def family_groups(family, vector_groups):
    """Groups that never split a family and never separate two coinciding vectors.

    family, vector_groups: one integer per row. Two rows end up together when a
    chain of "same family" and "same vector group" links joins them. With one
    row per family this returns the vector groups unchanged. Groups are numbered
    by their first member, so the numbering follows row order.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    family, vector_groups = np.asarray(family), np.asarray(vector_groups)
    n = len(family)
    n_family = int(family.max()) + 1 if n else 0
    # Rows are linked through one extra node per family and one per vector group.
    rows = np.concatenate([np.arange(n), np.arange(n)])
    columns = np.concatenate([n + family, n + n_family + vector_groups])
    size = n + n_family + (int(vector_groups.max()) + 1 if n else 0)
    graph = csr_matrix((np.ones(2 * n, dtype=bool), (rows, columns)), shape=(size, size))
    _, raw = connected_components(graph, directed=False)
    order = {}
    return np.array([order.setdefault(g, len(order)) for g in raw[:n]])


def group_diameters(D, groups):
    """Largest distance inside each group (0 for single rows)."""
    diameters = np.zeros(groups.max() + 1)
    for g in range(groups.max() + 1):
        members = np.flatnonzero(groups == g)
        if members.size > 1:
            diameters[g] = D[np.ix_(members, members)].max()
    return diameters


def nearest_other(D, exclude=None):
    """For each row, the nearest other row and its distance.

    exclude: optional group numbers; rows of the same group are skipped, which
    gives the nearest vector that is not a coinciding one.
    """
    masked = D.copy()
    np.fill_diagonal(masked, np.inf)
    if exclude is not None:
        masked[exclude[:, None] == exclude[None, :]] = np.inf
    nearest = masked.argmin(axis=1)
    return nearest, masked[np.arange(len(D)), nearest]


# --------------------------------------------------------------------------- output

def write_table(path, frame):
    """Write a DataFrame as CSV through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    tmp.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    os.replace(tmp, path)


def _plain(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def run_metadata():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas", "ase"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version.split()[0]
    return {"command": sys.argv, "host": socket.gethostname(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "versions": versions}


def relative_path(path):
    return sc.relative_path(path)
