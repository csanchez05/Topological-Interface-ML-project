"""Covariant ACE: the lambda = 1 and 2 companions of the invariant ACE blocks of ace_common.py.

The invariant blocks keep only lambda = 0. The same density products also come in versions that
turn with the structure (featomic DensityCorrelations with selected keys lambda = 1, 2):

  nu<k>_polar         lambda 1, sigma +1   turns like an ordinary vector, flips under inversion
  nu<k>_axial         lambda 1, sigma -1   turns like a vector, unchanged by inversion (as a spin does)
  nu<k>_tensor        lambda 2, sigma +1   symmetric traceless rank-2 tensor (5 components), unchanged
                                           by inversion
  nu<k>_pseudotensor  lambda 2, sigma -1   turns like tensor, flips under inversion

for body orders k = 1 (the density expansion itself, which has only sigma +1), 2 (the covariant
power spectrum, lambda-SOAP) and 3. Under inversion a block is multiplied by sigma (-1)^lambda,
as in src/dscribe/CoGe/covariant/. nu1 and nu2 use the (n_max, l_max) expansion and nu3 the smaller
(n3, l3) one: the expansions of the invariant run with the same tag, so a covariant run is the
lambda >= 1 counterpart of that run. lambda = 1 components are stored in x, y, z order, lambda = 2
in featomic's order m = -2 ... 2 (real spherical harmonics).

Averaging as for the invariant blocks: [Co mean | Ge mean] for every component, over the whole cell
and over the atoms within R of each interface plane (edge and mid apart). The atoms go through
featomic `chunk` at a time and only running sums are kept, so a worker's memory does not grow with
the cell (one nu3 atom carries about 285,000 numbers at n3 = l3 = 4).

The cell average of every lambda >= 1 block vanishes in bulk B20 (its point group 23 allows no
vector and no rank-2 tensor). The loaders therefore scale each block by the mean length of one
pristine atom's block vector (atom_norm_<block> of the pristine row of bulk_index.csv), the
covariant counterpart of the invariant runs' pristine scaling.

Outputs, under calculations/descriptors/ACE/covariant/:
  global/<name>/           covariant_ace.npz (X_<block>: N, C, 2P), bulk_covariant_ace.npz, index.csv,
                           bulk_index.csv, settings.json, skipped.csv
  both_interfaces/<name>/  covariant_ace.npz (X_<block>_r<R>: N, plane, C, 2P; n_r<R>), index.csv,
                           settings.json, skipped.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ace_common as ac  # noqa: E402

sc = ac.sc
COVARIANT_ROOT = ac.ACE_ROOT / "covariant"
FILE = "covariant_ace.npz"
BULK_FILE = "bulk_covariant_ace.npz"
# kind: (lambda, sigma)
KINDS = {"polar": (1, 1), "axial": (1, -1), "tensor": (2, 1), "pseudotensor": (2, -1)}
# block: (nu, lambda, sigma); the density expansion (nu = 1) has only sigma = +1.
BLOCKS = {f"nu{nu}_{kind}": (nu, lam, sig) for nu in (1, 2, 3) for kind, (lam, sig) in KINDS.items()
          if nu > 1 or sig == 1}
# featomic's real spherical harmonics for lambda = 1 are ordered m = -1, 0, 1, i.e. y, z, x.
TO_XYZ = [2, 0, 1]
SCALINGS = ("pristine_atom", "none")


def blocks_for(settings):
    """Covariant blocks of one ACE setting, in storage order."""
    return [name for name, (nu, _, _) in BLOCKS.items() if nu <= settings["nu_max"]]


def inversion_sign(name):
    """Factor a block picks up under r -> -r: sigma (-1)^lambda."""
    _, lam, sig = BLOCKS[name]
    return float(sig * (-1) ** lam)


def components(name):
    return 2 * BLOCKS[name][1] + 1


def run_name(settings):
    """Default run name: the invariant tag plus _lam1-2."""
    return ac.ace_tag(settings) + "_lam1-2"


def run_dir(region, tag_or_dir):
    return sc.run_dir(region, tag_or_dir, root=COVARIANT_ROOT)


# --------------------------------------------------------------------------- calculation

class CovariantCalculator:
    """featomic calculators for the covariant blocks of one ACE setting.

    chunks(atoms) yields (atom indices, {block: (n, C, P) float64}) for `chunk` atoms at a time.
    The property labels of every block are fixed by the first structure and checked afterwards.
    """

    def __init__(self, settings, chunk=128):
        import featomic
        from featomic.clebsch_gordan import DensityCorrelations
        from metatensor import Labels

        self._labels = Labels
        self.settings = dict(settings)
        self.blocks = blocks_for(settings)
        self.chunk = int(chunk)
        self.neighbors = Labels(["neighbor_type"],
                                np.array([[ac.ATOMIC_NUMBERS[s]] for s in sc.SPECIES], dtype=np.int32))
        self.keys = Labels(["o3_lambda", "o3_sigma"], np.array(list(KINDS.values()), dtype=np.int32))
        self.expansion = featomic.SphericalExpansion(
            **ac.calculator_hypers(settings, settings["n_max"], settings["l_max"]))
        self.nu2 = DensityCorrelations(n_correlations=1, max_angular=settings["l_max"])
        if settings["nu_max"] >= 3:
            self.expansion3 = featomic.SphericalExpansion(
                **ac.calculator_hypers(settings, settings["n3"], settings["l3"]))
            self.nu3 = DensityCorrelations(n_correlations=2, max_angular=settings["l3"])
        self.properties = {}
        self.property_names = {}

    def _density(self, expansion, atoms, selected):
        return expansion.compute(atoms, selected_samples=selected).keys_to_properties(self.neighbors)

    def _gather(self, name, tensor, atoms_in_chunk, start):
        """One block for the atoms of the chunk, (n, C, P), centre types joined, lambda = 1 as x, y, z."""
        _, lam, sig = BLOCKS[name]
        out = None
        for species in sc.SPECIES:
            block = tensor.block(o3_lambda=lam, o3_sigma=sig, center_type=ac.ATOMIC_NUMBERS[species])
            properties = np.asarray(block.properties.values)
            if name not in self.properties:
                self.properties[name] = properties.copy()
                self.property_names[name] = list(block.properties.names)
            elif not np.array_equal(properties, self.properties[name]):
                raise ValueError(f"block {name}: property labels differ from the first structure's")
            if out is None:
                out = np.full((atoms_in_chunk, 2 * lam + 1, properties.shape[0]), np.nan)
            rows = np.asarray(block.samples.column("atom")) - start
            if rows.size:
                out[rows] = block.values
        if np.isnan(out).any():
            raise ValueError(f"block {name}: featomic returned no value for some atoms")
        return out[:, TO_XYZ] if lam == 1 else out

    def chunks(self, atoms):
        n_atoms = len(atoms)
        for start in range(0, n_atoms, self.chunk):
            stop = min(start + self.chunk, n_atoms)
            selected = self._labels(["atom"], np.arange(start, stop, dtype=np.int32).reshape(-1, 1))
            rho = self._density(self.expansion, atoms, selected)
            tensors = {1: rho, 2: ac.Calculator._merge_cg_keys(self.nu2.compute(rho, selected_keys=self.keys))}
            if self.settings["nu_max"] >= 3:
                rho3 = self._density(self.expansion3, atoms, selected)
                tensors[3] = ac.Calculator._merge_cg_keys(
                    self.nu3.compute(rho3, angular_cutoff=self.settings["l3"], selected_keys=self.keys))
            yield np.arange(start, stop), {name: self._gather(name, tensors[BLOCKS[name][0]], stop - start, start)
                                           for name in self.blocks}

    def per_atom(self, atoms):
        """{block: (n_atoms, C, P)} for the whole structure (tests and small cells)."""
        parts = {name: [] for name in self.blocks}
        for _, values in self.chunks(atoms):
            for name, block in values.items():
                parts[name].append(block)
        return {name: np.concatenate(blocks) for name, blocks in parts.items()}

    def widths(self):
        """Features per atom and component of each block (known after the first call)."""
        return {name: int(self.properties[name].shape[0]) for name in self.blocks}


def averages(calculator, atoms, masks):
    """[Co mean | Ge mean] of every block over each atom set, from running sums.

    masks: (S, n_atoms) bool, one row per atom set. Returns ({block: (S, C, 2P) float64},
    counts (S, 2), {block: mean length of one atom's block vector}). A set without atoms of one
    species gives NaN in that half, as soap_common.species_mean.
    """
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    species = np.stack([is_co, ~is_co])                              # (2, n_atoms)
    selection = masks[:, None, :] & species[None, :, :]              # (S, 2, n_atoms)
    counts = selection.sum(axis=2)
    sums, norm_sums = {}, {}
    for idx, values in calculator.chunks(atoms):
        for name, block in values.items():
            weights = selection[:, :, idx].astype(np.float64)        # (S, 2, n)
            part = np.einsum("tsn,ncp->tscp", weights, block)        # (S, 2, C, P)
            sums[name] = part if name not in sums else sums[name] + part
            norm_sums[name] = norm_sums.get(name, 0.0) + float(np.linalg.norm(block.reshape(len(idx), -1), axis=1).sum())
    out = {}
    for name, total in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / counts[:, :, None, None]
        mean[counts == 0] = np.nan
        # (S, 2, C, P) -> (S, C, 2P): each component's [Co | Ge] halves.
        out[name] = np.concatenate([mean[:, 0], mean[:, 1]], axis=-1)
    return out, counts.astype(np.int32), {name: value / len(atoms) for name, value in norm_sums.items()}


def describe_structure(path, calculator, radii):
    """Whole-cell and interface averages of every block for one GB cell (float32)."""
    atoms, _ = sc.read_structure(path)
    cell = atoms.cell.array
    lengths = np.diag(cell).copy()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6:
        raise ValueError("cell is not orthogonal")
    if lengths[0] < lengths[1:].max():
        raise ValueError("x is not the longest axis, so the interface normal is unclear")
    lx = lengths[0]
    distance, plane = sc.plane_distances(atoms.get_scaled_positions(wrap=True)[:, 0] * lx, lx)
    masks = [np.ones(len(atoms), dtype=bool)]
    masks += [(distance <= radius) & (plane == p) for radius in radii for p in (0, 1)]
    vectors, counts, norms = averages(calculator, atoms, np.array(masks))
    n_radii = len(radii)
    result = {"global": {name: values[0].astype(np.float32) for name, values in vectors.items()},
              "interface": {name: values[1:].reshape((n_radii, 2) + values.shape[1:]).astype(np.float32)
                            for name, values in vectors.items()},
              "n_interface": counts[1:].reshape(n_radii, 2, 2)}
    result["meta"] = {"n_atoms": len(atoms), "n_Co": int(counts[0, 0]), "n_Ge": int(counts[0, 1]),
                      "Lx": f"{lengths[0]:.8f}", "Ly": f"{lengths[1]:.8f}", "Lz": f"{lengths[2]:.8f}",
                      **{f"atom_norm_{name}": f"{value:.6e}" for name, value in norms.items()}}
    return result


def describe_bulk(path, calculator):
    """Whole-cell average of every block of one bulk cell (float64) and the atom norms."""
    atoms, _ = sc.read_structure(path)
    vectors, _, norms = averages(calculator, atoms, np.ones((1, len(atoms)), dtype=bool))
    return {"vectors": {name: values[0] for name, values in vectors.items()},
            "meta": {"n_atoms": len(atoms), **{f"atom_norm_{name}": f"{value:.6e}" for name, value in norms.items()}}}


def label_arrays(calculator):
    """Property labels of every block as plain arrays (stored next to the vectors)."""
    return ac.label_arrays(calculator.properties, calculator.property_names)


# --------------------------------------------------------------------------- loaders

def stored_blocks(data):
    """Covariant blocks present in an opened covariant_ace.npz, in storage order."""
    keys = set(data.files)
    return [name for name in BLOCKS if f"X_{name}" in keys or any(k.startswith(f"X_{name}_r") for k in keys)]


def pristine_atom_norms(run, blocks):
    """{block: mean length of one pristine atom's block vector}, from global/<run>/bulk_index.csv."""
    import pandas as pd

    index = pd.read_csv(run_dir("global", run) / "bulk_index.csv", dtype={"plane": str})
    rows = index[index["kind"] == "pristine"]
    if len(rows) != 1:
        raise ValueError("expected exactly one pristine row in bulk_index.csv")
    return {name: float(rows[f"atom_norm_{name}"].iloc[0]) for name in blocks}


def _positions(index, keys):
    lookup = {(f, int(t)): i for i, (f, t) in enumerate(zip(index["family"], index["thickness"]))}
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} structures missing from the covariant run, e.g. {missing[:3]}")
    return np.array([lookup[key] for key in keys])


def load_rows(run, keys, blocks=None, radius=8.0, scaling="pristine_atom", dtype=np.float64):
    """Covariant vectors of the structures `keys` ((family, thickness) pairs), blocks flattened and joined.

    Returns {"interface": (S, 2 planes, F), "cell": (S, F), "n_interface": (S, 2, 2), "n_species": (S, 2),
    "is_co": (F,) True for the Co-centred half of every component, "columns": {block: (start, stop)}}.
    Each block (C, 2P) is flattened component by component, so every component keeps its [Co | Ge]
    halves. scaling "pristine_atom" divides each block by pristine_atom_norms(); "none" keeps it raw.
    """
    import pandas as pd

    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}")
    interface_dir, global_dir = run_dir("both_interfaces", run), run_dir("global", run)
    index = pd.read_csv(interface_dir / "index.csv", dtype={"plane": str})
    pos = _positions(index, keys)
    global_index = pd.read_csv(global_dir / "index.csv", dtype={"plane": str})
    gpos = _positions(global_index, keys)
    key = sc.radius_key(radius)
    parts, cell_parts, is_co, columns, start = [], [], [], {}, 0
    with np.load(interface_dir / FILE) as data, np.load(global_dir / FILE) as cell_data:
        chosen = stored_blocks(data) if blocks is None else [name for name in BLOCKS if name in blocks]
        norms = pristine_atom_norms(run, chosen) if scaling == "pristine_atom" else dict.fromkeys(chosen, 1.0)
        n = data[f"n_{key}"][pos]
        for name in chosen:
            values = data[f"X_{name}_{key}"][pos].astype(dtype) / norms[name]
            parts.append(values.reshape(len(pos), 2, -1))
            cell_parts.append(cell_data[f"X_{name}"][gpos].astype(dtype).reshape(len(gpos), -1) / norms[name])
            width = values.shape[-1]
            is_co.append(np.tile(np.arange(width) < width // 2, values.shape[-2]))
            columns[name] = (start, start + parts[-1].shape[-1])
            start = columns[name][1]
    return {"interface": np.concatenate(parts, axis=-1), "cell": np.concatenate(cell_parts, axis=-1),
            "n_interface": n, "n_species": global_index.loc[gpos, ["n_Co", "n_Ge"]].to_numpy(),
            "is_co": np.concatenate(is_co), "columns": columns, "norms": norms}


def bulk_averages(run, blocks=None):
    """(X {block: (n_cells, C, 2P)}, bulk_index DataFrame) of the bulk cells of a run."""
    import pandas as pd

    global_dir = run_dir("global", run)
    index = pd.read_csv(global_dir / "bulk_index.csv", dtype={"plane": str})
    with np.load(global_dir / BULK_FILE) as data:
        chosen = stored_blocks(data) if blocks is None else [name for name in BLOCKS if name in blocks]
        return {name: data[f"X_{name}"] for name in chosen}, index
