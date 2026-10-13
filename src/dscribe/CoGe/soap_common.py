"""Shared helpers for the CoGe SOAP descriptor scripts (bulk cells and grain boundaries).

Vectors are always laid out as [mean of Co-centred SOAP | mean of Ge-centred SOAP],
so every stored vector has 2 * n_features entries.

Outputs live in calculations/descriptors/dscribe/standard/, one folder per SOAP
setting in each region:
  global/<tag>/           global_soap.npz (mean over every atom), the bulk cells, index.csv
  both_interfaces/<tag>/  interface_soap.npz (atoms near each interface plane), index.csv
"""
from __future__ import annotations

import csv
import json
import os
import socket
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BULK_ROOT = PROJECT_ROOT / "calculations" / "CoGe_Bulk"
GB_ROOT = PROJECT_ROOT / "structures" / "CoGeGrainBoundaries"
DESCRIPTOR_ROOT = PROJECT_ROOT / "calculations" / "descriptors" / "dscribe"
STANDARD_ROOT = DESCRIPTOR_ROOT / "standard"
COVARIANT_ROOT = DESCRIPTOR_ROOT / "covariant"
LOG_DIR = DESCRIPTOR_ROOT / "logs"
REGIONS = ("global", "both_interfaces")
GLOBAL_FILE = "global_soap.npz"
INTERFACE_FILE = "interface_soap.npz"
PRISTINE_POSCAR = GB_ROOT / "POSCAR"
PLANES = ("100", "110", "111")
SPECIES = ("Co", "Ge")
# Order of the plane axis in the grain-boundary arrays.
PLANE_NAMES = ("edge", "mid")


# --------------------------------------------------------------------------- structures

def read_structure(path):
    """Read a POSCAR. Returns (atoms, scale_corrected).

    Several bulk POSCARs carry the scale 4.6294 on lattice vectors that are
    already in Angstrom. When the scale is not 1 and the raw vectors are
    already Angstrom-sized, the scale is ignored.
    """
    from ase.io import read

    path = Path(path)
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    raw_cell = np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    atoms = read(path, format="vasp")
    try:
        header_count = sum(int(v) for v in lines[6].split())
    except ValueError:
        header_count = len(atoms)
    if len(atoms) != header_count:
        raise ValueError(f"read {len(atoms)} atoms but the header lists {header_count}")
    unknown = set(atoms.get_chemical_symbols()) - set(SPECIES)
    if unknown:
        raise ValueError(f"unexpected species {sorted(unknown)}")
    corrected = False
    if abs(scale - 1.0) > 1e-6 and np.linalg.norm(raw_cell, axis=1).max() > 3.0:
        atoms.set_cell(raw_cell, scale_atoms=True)
        corrected = True
    return atoms, corrected


def min_pair_distance(atoms, cutoff=3.0):
    """Smallest interatomic distance (periodic). Raises if nothing is within `cutoff`."""
    from ase.neighborlist import neighbor_list

    distances = neighbor_list("d", atoms, cutoff)
    if distances.size == 0:
        raise ValueError(f"no atom pair within {cutoff} A; the lattice scale is probably wrong")
    return float(distances.min())


def plane_distances(x, lx):
    """Distance along x to the nearest interface plane, and which plane that is.

    The cell holds two interfaces: plane 0 ("edge") at x = 0, which is the same
    plane as x = lx by periodicity, and plane 1 ("mid") at x = lx / 2.
    """
    x = np.mod(np.asarray(x, dtype=float), lx)
    d_edge = np.minimum(x, lx - x)
    d_mid = np.abs(x - lx / 2.0)
    plane = (d_mid < d_edge).astype(int)
    return np.minimum(d_edge, d_mid), plane


# --------------------------------------------------------------------------- SOAP

def add_soap_arguments(parser):
    group = parser.add_argument_group("SOAP settings")
    group.add_argument("--r-cut", type=float, default=6.0, help="cutoff radius in A (default 6.0)")
    group.add_argument("--n-max", type=int, default=8, help="radial basis functions (default 8)")
    group.add_argument("--l-max", type=int, default=6, help="maximum angular momentum (default 6)")
    group.add_argument("--sigma", type=float, default=0.5,
                       help="Gaussian width of each atom in A (default 0.5; DScribe's own default is 1.0)")
    group.add_argument("--rbf", default="gto", choices=("gto", "polynomial"), help="radial basis (default gto)")
    group.add_argument("--weighting", type=json.loads, default=None,
                       help='optional DScribe radial weighting as JSON, e.g. \'{"function":"pow","r0":5,"m":6,"c":1,"d":1}\'')


def soap_settings(args):
    return {
        "r_cut": args.r_cut,
        "n_max": args.n_max,
        "l_max": args.l_max,
        "sigma": args.sigma,
        "rbf": args.rbf,
        "weighting": args.weighting,
        "species": list(SPECIES),
        "periodic": True,
    }


# Short names of the weighting parameters in folder names, in this order.
WEIGHTING_KEYS = (("r0", "r"), ("c", "c"), ("d", "d"), ("m", "m"), ("w0", "w0"), ("threshold", "t"))


def soap_tag(settings):
    """Folder name for one set of SOAP settings, e.g. rc6.0_n8_l6_s0.5.

    A radial weighting is spelled out: {"function": "pow", "r0": 3, "c": 1, "d": 1, "m": 4}
    gives the suffix _wpow-r3-c1-d1-m4.
    """
    tag = f"rc{settings['r_cut']:.1f}_n{settings['n_max']}_l{settings['l_max']}_s{settings['sigma']:g}"
    if settings["rbf"] != "gto":
        tag += f"_{settings['rbf']}"
    weighting = settings["weighting"]
    if weighting:
        unknown = set(weighting) - {"function"} - {key for key, _ in WEIGHTING_KEYS}
        if unknown:
            raise ValueError(f"unknown weighting parameters {sorted(unknown)}")
        tag += "_w" + weighting["function"] + "".join(
            f"-{short}{weighting[key]:g}" for key, short in WEIGHTING_KEYS if key in weighting)
    return tag


def make_soap(settings):
    from dscribe.descriptors import SOAP

    return SOAP(
        species=settings["species"],
        r_cut=settings["r_cut"],
        n_max=settings["n_max"],
        l_max=settings["l_max"],
        sigma=settings["sigma"],
        rbf=settings["rbf"],
        # DScribe adds keys to the dict it is given, so it gets a copy.
        weighting=dict(settings["weighting"]) if settings["weighting"] else None,
        periodic=settings["periodic"],
        average="off",
        sparse=False,
    )


def species_mean(X, is_co, mask=None):
    """[mean over Co-centred rows | mean over Ge-centred rows] and the two atom counts.

    A species with no selected atom gives NaN in its half and a count of zero.
    """
    n_features = X.shape[1]
    vector = np.full(2 * n_features, np.nan)
    counts = np.zeros(2, dtype=np.int32)
    for s, selected in enumerate((is_co, ~is_co)):
        if mask is not None:
            selected = selected & mask
        counts[s] = int(selected.sum())
        if counts[s]:
            vector[s * n_features:(s + 1) * n_features] = X[selected].mean(axis=0)
    return vector, counts


def within_species_spread(X, is_co):
    """Largest relative distance of any atom's SOAP from its species mean."""
    spread = 0.0
    for selected in (is_co, ~is_co):
        rows = X[selected]
        mean = rows.mean(axis=0)
        spread = max(spread, float(np.linalg.norm(rows - mean, axis=1).max() / np.linalg.norm(mean)))
    return spread


def bulk_reference(soap):
    """Per-species SOAP of the pristine cell, shape (2, n_features): row 0 Co, row 1 Ge."""
    atoms, _ = read_structure(PRISTINE_POSCAR)
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    X = soap.create(atoms)
    return np.stack([X[is_co].mean(axis=0), X[~is_co].mean(axis=0)])


def combine_planes(X, n):
    """Count-weighted mean over the plane axis.

    X: (N, 2 planes, 2F); n: (N, 2 planes, 2 species). Returns (N, 2F).
    """
    n_features = X.shape[-1] // 2
    out = np.full((X.shape[0], X.shape[-1]), np.nan, dtype=X.dtype)
    for s in (0, 1):
        weights = n[:, :, s].astype(np.float64)
        part = np.nan_to_num(X[:, :, s * n_features:(s + 1) * n_features].astype(np.float64))
        total = weights.sum(axis=1)
        filled = total > 0
        summed = (part * weights[:, :, None]).sum(axis=1)
        out[filled, s * n_features:(s + 1) * n_features] = summed[filled] / total[filled, None]
    return out


# --------------------------------------------------------------------------- output

def _temporary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")


def write_npz(path, **arrays):
    tmp = _temporary(path)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_json(path, payload):
    tmp = _temporary(path)
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows, fieldnames):
    tmp = _temporary(path)
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_metadata():
    versions = {name: metadata.version(name) for name in ("dscribe", "ase", "numpy")}
    versions["python"] = sys.version.split()[0]
    return {
        "command": sys.argv,
        "host": socket.gethostname(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": versions,
    }


def run_dir(region, tag_or_dir, root=STANDARD_ROOT):
    """Folder of one run in one region ("global" or "both_interfaces").

    tag_or_dir is a tag (rc6.0_n8_l6_s0.5, looked up under root) or a run folder.
    A folder of either region gives the matching folder of the other one, so
    .../both_interfaces/<tag> and .../global/<tag> both work.
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    path = Path(tag_or_dir)
    if path.parent.name in REGIONS:
        return path.parent.parent / region / path.name
    if path.is_dir():
        return path
    return Path(root) / region / str(tag_or_dir)


def radius_key(radius):
    """Name of one interface radius inside interface_soap.npz, e.g. r8 or r6.5."""
    return f"r{float(radius):g}"


def write_gb_arrays(global_dir, interface_dir, X_global, X_interface, n_interface,
                    X_nonbulk, n_nonbulk, n_nonbulk_within, radii):
    """Write global_soap.npz and interface_soap.npz.

    X_interface: (N, radius, plane, 2F) and n_interface: (N, radius, plane, species)
    are stored as one X_r<R> / n_r<R> pair per radius, so a reader can load one
    radius without touching the others.
    """
    per_radius = {}
    for r, radius in enumerate(radii):
        per_radius[f"X_{radius_key(radius)}"] = X_interface[:, r]
        per_radius[f"n_{radius_key(radius)}"] = n_interface[:, r]
    write_npz(Path(global_dir) / GLOBAL_FILE, X=X_global, species=np.array(SPECIES))
    write_npz(Path(interface_dir) / INTERFACE_FILE, **per_radius, radii=np.array(radii, dtype=float),
              X_nonbulk=X_nonbulk, n_nonbulk=n_nonbulk, n_nonbulk_within=n_nonbulk_within,
              plane_names=np.array(PLANE_NAMES), species=np.array(SPECIES))


# --------------------------------------------------------------------------- loaders

def load_bulk(tag_or_dir):
    """Returns (X, index): X is (n_cells, 2F); index is a DataFrame in the same row order.

    The bulk cells are stored with the global vectors of the same SOAP setting.
    """
    import pandas as pd

    folder = run_dir("global", tag_or_dir)
    with np.load(folder / "bulk_soap.npz") as data:
        X = data["X"]
    return X, pd.read_csv(folder / "bulk_index.csv", dtype={"plane": str})


def load_gb(tag_or_dir, kind="interface", radius=None, planes="both"):
    """Returns (X, index) for the grain boundaries.

    kind:   "global"    mean over every atom in the cell
            "interface" atoms within `radius` (A) of an interface plane
            "nonbulk"   atoms whose SOAP differs from bulk (guide section 2.4)
    planes: "both" (count-weighted mean of the two interfaces), "edge" or "mid".
    """
    import pandas as pd

    if kind == "global":
        folder = run_dir("global", tag_or_dir)
        index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
        with np.load(folder / GLOBAL_FILE) as data:
            return data["X"], index
    folder = run_dir("both_interfaces", tag_or_dir)
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    with np.load(folder / INTERFACE_FILE) as data:
        if kind == "interface":
            radii = data["radii"]
            if radius is None:
                raise ValueError(f"choose a radius from {radii.tolist()}")
            matches = np.flatnonzero(np.isclose(radii, radius))
            if matches.size == 0:
                raise ValueError(f"radius {radius} not stored; available: {radii.tolist()}")
            # Only the arrays of this radius are read from the file.
            key = radius_key(radii[matches[0]])
            X, n = data[f"X_{key}"], data[f"n_{key}"]
        elif kind == "nonbulk":
            X, n = data["X_nonbulk"], data["n_nonbulk"]
        else:
            raise ValueError(f"unknown kind {kind!r}")
    if planes == "both":
        return combine_planes(X, n), index
    return X[:, PLANE_NAMES.index(planes)], index
