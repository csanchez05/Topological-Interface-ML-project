"""Loaders for the covariant (lambda-) SOAP runs, for the Stage 1/2 analysis scripts.

covariant_descriptor.py writes the runs; this module only reads them, so it can be imported by the
ML scripts without pinning their threads (covariant_descriptor.py sets OMP_NUM_THREADS=1 on import).

The analysis vector of one GB cell, as for standard SOAP, is built from [Co-centred mean |
Ge-centred mean] over the atoms near the interfaces, but the two interfaces are never averaged:
under the cell's own geometry the edge interface (x = 0, grain B | grain A) and the mid interface
(x = Lx/2, grain A | grain B) face opposite ways, so their lambda = 1 parts would partly cancel.

  kind "interface", planes "both"   [C_edge | C_mid]: the two interface vectors joined
  kind "interface", planes "edge"   C_edge only (or "mid": C_mid only)
  kind "global"                     D: the whole-cell mean

C of one interface is the chosen blocks joined in storage order, each flattened component by
component (study 06's C), so every component keeps its [Co | Ge] halves. The default blocks are the
four with lambda >= 1 (polar, axial, tensor, pseudotensor); the invariant (lambda = 0) block is
standard-SOAP-like and can be added with blocks=[..., "invariant"]. Blocks are not rescaled.

Bulk: the lambda >= 1 whole-cell means of every bulk cell vanish (checks.json), so the bulk point of
these vectors is (close to) zero. load_bulk_vectors() returns the bulk cells' whole-cell means laid
out like the GB vectors ([x | x] for planes "both").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOAP_CODE = PROJECT_ROOT / "src" / "dscribe" / "CoGe"
if str(SOAP_CODE) not in sys.path:
    sys.path.insert(0, str(SOAP_CODE))

import soap_common as sc  # noqa: E402

# name: (lambda, sigma), storage order; must match covariant_descriptor.BLOCKS (tested)
BLOCKS = {"invariant": (0, 1), "polar": (1, 1), "axial": (1, -1), "tensor": (2, 1), "pseudotensor": (2, -1)}
DEFAULT_BLOCKS = ("polar", "axial", "tensor", "pseudotensor")
FILE = "covariant_soap.npz"
BULK_FILE = "bulk_covariant.npz"
SETTINGS_FILE = "settings.json"


def run_dir(region, tag_or_dir):
    """As soap_common.run_dir, with tags looked up under calculations/descriptors/dscribe/covariant."""
    return sc.run_dir(region, tag_or_dir, root=sc.COVARIANT_ROOT)


def stored_blocks(data):
    """Blocks present in an open npz file, in storage order."""
    present = {key[2:].split("_r")[0] for key in data.files if key.startswith("X_")}
    return [name for name in BLOCKS if name in present]


def choose_blocks(available, blocks=None):
    """The requested blocks in storage order (default: the lambda >= 1 blocks that are stored)."""
    wanted = list(DEFAULT_BLOCKS) if blocks is None else list(blocks)
    unknown = [name for name in wanted if name not in BLOCKS]
    if unknown:
        raise ValueError(f"unknown covariant blocks {unknown}; known: {list(BLOCKS)}")
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(f"blocks {missing} not stored in this run; stored: {list(available)}")
    return [name for name in BLOCKS if name in wanted]


def blocks_suffix(blocks=None):
    """Tag suffix for a block choice other than the default, e.g. _polar+axial or _all5."""
    if blocks is None or list(blocks) == list(DEFAULT_BLOCKS):
        return ""
    chosen = [name for name in BLOCKS if name in blocks]
    return "_all5" if chosen == list(BLOCKS) else "_" + "+".join(chosen)


def member_shape(data, name):
    """Shape of one array in an open npz file, read from its header (the array itself is not loaded)."""
    from numpy.lib import format as npy

    with data.zip.open(name + ".npy") as handle:
        version = npy.read_magic(handle)
        reader = npy.read_array_header_1_0 if version == (1, 0) else npy.read_array_header_2_0
        return reader(handle)[0]


def _radius_key(data, radius):
    radii = data["radii"]
    if radius is None:
        raise ValueError(f"choose a radius from {radii.tolist()}")
    matches = np.flatnonzero(np.isclose(radii, radius))
    if matches.size == 0:
        raise ValueError(f"radius {radius} not stored; available: {radii.tolist()}")
    return sc.radius_key(radii[matches[0]])


def flatten(arrays, kind="interface", planes="both"):
    """Join {block: X} into one float32 array, blocks in storage order.

    Interface arrays are (N, 2 planes, components, 2P): planes "both" gives [C_edge | C_mid], "edge"
    or "mid" one plane. Global arrays are (N, components, 2P). Each block is flattened component by
    component.
    """
    names = [name for name in BLOCKS if name in arrays]
    if not names:
        raise ValueError("no blocks to join")
    n = len(arrays[names[0]])
    if kind == "global":
        widths = [int(np.prod(arrays[name].shape[1:])) for name in names]
        out = np.empty((n, sum(widths)), dtype=np.float32)
        start = 0
        for name, width in zip(names, widths):
            out[:, start:start + width] = arrays[name].reshape(n, width)
            start += width
        return out
    if kind != "interface":
        raise ValueError(f"unknown kind {kind!r} (covariant runs store interface and global vectors)")
    widths = [int(np.prod(arrays[name].shape[2:])) for name in names]
    sides = [0, 1] if planes == "both" else [sc.PLANE_NAMES.index(planes)]
    total = sum(widths)
    out = np.empty((n, total * len(sides)), dtype=np.float32)
    for k, side in enumerate(sides):
        start = k * total
        for name, width in zip(names, widths):
            out[:, start:start + width] = arrays[name][:, side].reshape(n, width)
            start += width
    return out


def load_gb_vectors(tag_or_dir, kind="interface", radius=None, planes="both", blocks=None):
    """(X float32, index DataFrame) for every GB structure of the run.

    Blocks are read one at a time into the output, so the peak memory is the output plus one block.
    """
    import pandas as pd

    region = "global" if kind == "global" else "both_interfaces"
    if kind not in ("global", "interface"):
        raise ValueError(f"unknown kind {kind!r} (covariant runs store interface and global vectors)")
    if kind == "interface" and planes not in ("both",) + tuple(sc.PLANE_NAMES):
        raise ValueError(f"planes must be both, edge or mid, not {planes!r}")
    folder = run_dir(region, tag_or_dir)
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    with np.load(folder / FILE) as data:
        names = choose_blocks(stored_blocks(data), blocks)
        key = _radius_key(data, radius) if kind == "interface" else None
        shapes = {name: member_shape(data, f"X_{name}_{key}" if key else f"X_{name}") for name in names}
        n = len(index)
        if any(shape[0] != n for shape in shapes.values()):
            raise ValueError(f"{folder}: arrays and index.csv differ in length")
        sides = ([0, 1] if planes == "both" else [sc.PLANE_NAMES.index(planes)]) if key else [None]
        skip = 2 if key else 1
        widths = [int(np.prod(shapes[name][skip:])) for name in names]
        total = sum(widths)
        out = np.empty((n, total * len(sides)), dtype=np.float32)
        for b, name in enumerate(names):
            block = data[f"X_{name}_{key}" if key else f"X_{name}"]
            for k, side in enumerate(sides):
                start = k * total + sum(widths[:b])
                out[:, start:start + widths[b]] = (block if side is None else block[:, side]).reshape(n, widths[b])
            del block
    return out, index


def load_bulk_vectors(tag_or_dir, blocks=None, kind="interface", planes="both"):
    """(X float32, bulk index) for the bulk cells: their whole-cell means of the chosen blocks, laid
    out like load_gb_vectors(kind, planes), i.e. [x | x] for interface vectors of both planes."""
    import pandas as pd

    folder = run_dir("global", tag_or_dir)
    index = pd.read_csv(folder / "bulk_index.csv", dtype={"plane": str})
    with np.load(folder / BULK_FILE) as data:
        names = choose_blocks(stored_blocks(data), blocks)
        X = flatten({name: data[f"X_{name}"] for name in names}, "global")
    if kind == "interface" and planes == "both":
        X = np.concatenate([X, X], axis=1)
    return X, index


def vector_length(tag_or_dir, blocks=None, kind="interface", planes="both"):
    """Length of the analysis vector, from the stored property lists (no arrays are read)."""
    folder = run_dir("global" if kind == "global" else "both_interfaces", tag_or_dir)
    with np.load(folder / FILE) as data:
        names = choose_blocks(stored_blocks(data), blocks)
        length = sum((2 * BLOCKS[name][0] + 1) * 2 * len(data[f"properties_{name}"]) for name in names)
    return length * (2 if kind == "interface" and planes == "both" else 1)


def settings(tag_or_dir, region="both_interfaces"):
    """The run's descriptor settings (r_cut, n_max, l_max, sigma, smoothing)."""
    import json

    return json.loads((run_dir(region, tag_or_dir) / SETTINGS_FILE).read_text())["settings"]
