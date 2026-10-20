"""Shared helpers for the CoGe ACE descriptor scripts (bulk cells and grain boundaries).

ACE (atomic cluster expansion) builds rotation invariants body order by body order from one
atomic density expansion A_nlm = sum_j R_nl(r_ij) Y_lm(r_ij), computed here with featomic:

  nu1       2-body:  A_n00, the radial density                        (featomic SphericalExpansion)
  nu2       3-body:  products of two A's, the SOAP power spectrum      (DensityCorrelations, 1 product)
  nu3_even  4-body:  products of three A's, unchanged by inversion     (DensityCorrelations, 2 products)
  nu3_odd   4-body:  products of three A's with l1 + l2 + l3 odd; they change sign under
            inversion, so they tell the two B20 enantiomers apart (SOAP cannot)

The neighbour species goes into the properties, so the products mix Co and Ge neighbours. The
centre species stays separate: every stored vector is [mean over Co centres | mean over Ge
centres], as in soap_common, so a block with P features per atom gives 2P entries. Each block is
stored on its own; the loaders join the chosen blocks, by default each divided by the length of
its pristine-bulk vector so that no body order dominates.

Structure reading, pooling, plane assignment and writers are imported from soap_common, so the
ACE vectors average exactly the same atoms as the SOAP ones.

Outputs live in calculations/descriptors/ACE/standard/, one folder per setting in each region:
  global/<tag>/           global_ace.npz (mean over every atom), bulk_ace.npz, index.csv
  both_interfaces/<tag>/  interface_ace.npz (atoms near each interface plane), index.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOAP_CODE = PROJECT_ROOT / "src" / "dscribe" / "CoGe"
sys.path.insert(0, str(SOAP_CODE))

import soap_common as sc  # noqa: E402

ACE_ROOT = PROJECT_ROOT / "calculations" / "descriptors" / "ACE"
STANDARD_ROOT = ACE_ROOT / "standard"
LOG_DIR = ACE_ROOT / "logs"
GLOBAL_FILE = "global_ace.npz"
INTERFACE_FILE = "interface_ace.npz"
BULK_FILE = "bulk_ace.npz"
# Descriptor-free labels shared with SOAP (handedness of each bulk cell's operation).
GB_LABELS = PROJECT_ROOT / "calculations" / "ML_dscribe" / "CoGe" / "labels" / "gb_labels.csv"
# name: (body order nu, inversion parity sigma)
BLOCKS = {"nu1": (1, 1), "nu2": (2, 1), "nu3_even": (3, 1), "nu3_odd": (3, -1)}
ATOMIC_NUMBERS = {"Co": 27, "Ge": 32}
SCALINGS = ("pristine", "none")


def _soap_script(relative, name):
    """Import one of the SOAP step scripts under its own module name.

    The ACE step scripts have the same file names, so a plain `import gb_descriptor` from an
    ACE script would find the ACE script itself.
    """
    spec = importlib.util.spec_from_file_location(name, SOAP_CODE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_cells(bulk_root, planes, limit=0):
    """The SOAP bulk cell list (kind, plane, sym, path); with limit, only the first `limit`
    transformed cells per plane (the pristine cell and the references are always kept)."""
    cells = _soap_script("bulk/bulk_descriptor.py", "soap_bulk_descriptor").collect_cells(bulk_root, planes)
    if not limit:
        return cells
    seen, kept = {}, []
    for cell in cells:
        if cell[0] == "transformed":
            seen[cell[1]] = seen.get(cell[1], 0) + 1
            if seen[cell[1]] > limit:
                continue
        kept.append(cell)
    return kept


def collect_structures(gb_root, planes, thicknesses, limit):
    """The SOAP GB structure list: (jobs, skipped), a job being (plane, sym, thickness, path)."""
    module = _soap_script("gb/gb_descriptor.py", "soap_gb_descriptor")
    return module.collect_structures(gb_root, planes, thicknesses, limit)


def all_thicknesses():
    return _soap_script("gb/gb_descriptor.py", "soap_gb_descriptor").ALL_THICKNESSES


# --------------------------------------------------------------------------- settings

def add_ace_arguments(parser):
    group = parser.add_argument_group("ACE settings")
    group.add_argument("--r-cut", type=float, default=6.0, help="cutoff radius in A (default 6.0, as SOAP)")
    group.add_argument("--n-max", type=int, default=8, help="radial functions for nu1 and nu2 (default 8)")
    group.add_argument("--l-max", type=int, default=6, help="largest angular order for nu2 (default 6)")
    group.add_argument("--sigma", type=float, default=0.5,
                       help="Gaussian width of each atom in A (default 0.5, as SOAP)")
    group.add_argument("--smoothing", type=float, default=0.5,
                       help="width of the shifted-cosine cutoff smoothing in A (default 0.5)")
    group.add_argument("--density", default="gaussian", choices=("gaussian", "dirac"),
                       help="atom density: gaussian (smeared, as SOAP; default); dirac (textbook ACE) is refused, featomic 0.6.7 cannot expand it on the GTO basis")
    group.add_argument("--nu-max", type=int, default=3, choices=(2, 3),
                       help="highest body order minus one: 2 stops at the SOAP power spectrum, "
                            "3 adds the 4-body invariants (default 3)")
    group.add_argument("--n3", type=int, default=4, help="radial functions of the smaller nu3 expansion (default 4)")
    group.add_argument("--l3", type=int, default=4, help="largest angular order of the nu3 expansion (default 4)")
    group.add_argument("--weighting", default="none", choices=("none", "pow"),
                       help="radial weighting of the neighbour density: none (default) or pow, "
                            "w(r) = 1 / (1 + (r / r0)^m) (Willatt et al. 2018; DScribe's pow with c = d = 1)")
    group.add_argument("--weighting-r0", type=float, default=3.0, help="r0 of the pow weighting in A (default 3)")
    group.add_argument("--weighting-m", type=int, default=4, help="exponent m of the pow weighting (default 4)")


def pow_weighting(r0=3.0, m=4):
    """The pow weighting as a soap_common weighting dict (DScribe's pow with c = d = 1, as the SOAP study)."""
    return {"function": "pow", "r0": float(r0), "c": 1.0, "d": 1.0, "m": int(m)}


def ace_settings(args):
    gaussian = args.density == "gaussian"
    settings = {
        "r_cut": args.r_cut, "n_max": args.n_max, "l_max": args.l_max,
        "sigma": args.sigma if gaussian else None, "smoothing": args.smoothing, "density": args.density,
        "nu_max": args.nu_max, "n3": args.n3 if args.nu_max >= 3 else None,
        "l3": args.l3 if args.nu_max >= 3 else None,
        "radial_basis": "gto", "species": list(sc.SPECIES), "periodic": True,
    }
    weighting = getattr(args, "weighting", "none")
    if weighting == "pow":
        settings["weighting"] = pow_weighting(args.weighting_r0, args.weighting_m)
    elif weighting != "none":
        raise ValueError(f"unknown weighting {weighting!r}")
    return settings


def blocks_for(settings):
    """Block names computed for one setting, in storage order."""
    return [name for name, (nu, _) in BLOCKS.items() if nu <= settings["nu_max"]]


def ace_tag(settings):
    """Folder name for one ACE setting, e.g. rc6.0_n8_l6_s0.5_nu3-n4-l4.

    The first part is the SOAP tag of the same density (so the nu2 block of rc6.0_n8_l6_s0.5_nu3-n4-l4
    is the SOAP power spectrum of rc6.0_n8_l6_s0.5); a Dirac density replaces _s<sigma> by _dirac, a
    smoothing other than 0.5 A adds _sm<width>, and the suffix gives the body order and nu3 basis.
    A weighting adds the SOAP suffix, e.g. _wpow-r3-c1-d1-m4. Settings without a "weighting" key (the
    runs made before the option existed) are unweighted.
    """
    if settings["density"] == "gaussian":
        tag = sc.soap_tag({**settings, "rbf": "gto", "weighting": None})
    elif settings["density"] == "dirac":
        tag = f"rc{settings['r_cut']:.1f}_n{settings['n_max']}_l{settings['l_max']}_dirac"
    else:
        raise ValueError(f"unknown density {settings['density']!r}")
    weighting = settings.get("weighting")
    if weighting:
        plain = sc.soap_tag({**settings, "sigma": 1.0, "rbf": "gto", "weighting": None})
        tag += sc.soap_tag({**settings, "sigma": 1.0, "rbf": "gto", "weighting": weighting})[len(plain):]
    if settings["smoothing"] != 0.5:
        tag += f"_sm{settings['smoothing']:g}"
    if settings["nu_max"] >= 3:
        tag += f"_nu3-n{settings['n3']}-l{settings['l3']}"
    else:
        tag += f"_nu{settings['nu_max']}"
    return tag


def hypers(settings, n_max, l_max):
    """featomic SphericalExpansion hypers for one expansion of this setting."""
    density = ({"type": "Gaussian", "width": settings["sigma"]} if settings["density"] == "gaussian"
               else {"type": "DiracDelta"})
    weighting = settings.get("weighting")
    if weighting:
        if weighting["function"] != "pow" or weighting["c"] != weighting["d"]:
            raise ValueError("featomic has only the pow weighting with c = d (Willatt2018)")
        density["scaling"] = {"type": "Willatt2018", "exponent": int(weighting["m"]), "rate": float(weighting["c"]),
                              "scale": float(weighting["r0"])}
    return {"cutoff": {"radius": settings["r_cut"],
                       "smoothing": {"type": "ShiftedCosine", "width": settings["smoothing"]}},
            "density": density,
            "basis": {"type": "TensorProduct", "max_angular": l_max,
                      "radial": {"type": "Gto", "max_radial": n_max - 1}}}


def calculator_hypers(settings, n_max, l_max):
    """What featomic.SphericalExpansion gets: hypers(), for a Gaussian density.

    A Dirac density is refused: featomic 0.6.7 has no built-in radial integral for it on the GTO
    basis ("this combination of basis and density is not supported"), and its SoapSpliner gives NaN
    derivatives for that combination (tried 2026-10-13).
    """
    if settings["density"] == "dirac":
        raise ValueError("--density dirac is not available: featomic 0.6.7 cannot expand a Dirac density "
                         "on the GTO basis (neither built in nor splined); use --density gaussian")
    return hypers(settings, n_max, l_max)


def featomic_hypers(settings):
    out = {"nu1_nu2": hypers(settings, settings["n_max"], settings["l_max"])}
    if settings["nu_max"] >= 3:
        out["nu3"] = hypers(settings, settings["n3"], settings["l3"])
    return out


# --------------------------------------------------------------------------- calculation

class Calculator:
    """featomic calculators for one ACE setting.

    per_atom(atoms) returns {block: (n_atoms, P) float64}. Atoms go through featomic `chunk` at a time
    (the results are identical; only the memory of one call is bounded).
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
                                np.array([[ATOMIC_NUMBERS[s]] for s in sc.SPECIES], dtype=np.int32))
        self.expansion = featomic.SphericalExpansion(
            **calculator_hypers(settings, settings["n_max"], settings["l_max"]))
        self.nu2 = DensityCorrelations(n_correlations=1, max_angular=settings["l_max"])
        self.even = Labels(["o3_lambda", "o3_sigma"], np.array([[0, 1]], dtype=np.int32))
        if settings["nu_max"] >= 3:
            self.expansion3 = featomic.SphericalExpansion(
                **calculator_hypers(settings, settings["n3"], settings["l3"]))
            self.nu3 = DensityCorrelations(n_correlations=2, max_angular=settings["l3"])
            self.both = Labels(["o3_lambda", "o3_sigma"], np.array([[0, 1], [0, -1]], dtype=np.int32))
        self.properties = {}
        self.property_names = {}

    def _density(self, expansion, atoms, selected):
        return expansion.compute(atoms, selected_samples=selected).keys_to_properties(self.neighbors)

    @staticmethod
    def _merge_cg_keys(tensor):
        names = [name for name in tensor.keys.names if name not in ("o3_lambda", "o3_sigma", "center_type")]
        return tensor.keys_to_properties(names) if names else tensor

    def _store(self, out, name, block, n_atoms):
        """Copy one centre type's lambda = 0 block into out[name] and check its property labels."""
        properties = np.asarray(block.properties.values)
        if name not in self.properties:
            self.properties[name] = properties.copy()
            self.property_names[name] = list(block.properties.names)
        elif not np.array_equal(properties, self.properties[name]):
            raise ValueError(f"block {name}: property labels differ from the first structure's")
        if name not in out:
            out[name] = np.full((n_atoms, properties.shape[0]), np.nan)
        atoms = np.asarray(block.samples.column("atom"))
        if atoms.size:
            out[name][atoms] = block.values[:, 0, :]

    def per_atom(self, atoms):
        n_atoms = len(atoms)
        out = {}
        for start in range(0, n_atoms, self.chunk):
            selected = self._labels(["atom"], np.arange(start, min(start + self.chunk, n_atoms),
                                                        dtype=np.int32).reshape(-1, 1))
            rho = self._density(self.expansion, atoms, selected)
            for species in sc.SPECIES:
                center = ATOMIC_NUMBERS[species]
                self._store(out, "nu1", rho.block(o3_lambda=0, o3_sigma=1, center_type=center), n_atoms)
            nu2 = self._merge_cg_keys(self.nu2.compute(rho, selected_keys=self.even))
            for species in sc.SPECIES:
                self._store(out, "nu2", nu2.block(o3_lambda=0, o3_sigma=1, center_type=ATOMIC_NUMBERS[species]),
                            n_atoms)
            if self.settings["nu_max"] >= 3:
                rho3 = self._density(self.expansion3, atoms, selected)
                nu3 = self._merge_cg_keys(self.nu3.compute(rho3, angular_cutoff=self.settings["l3"],
                                                           selected_keys=self.both))
                for name, sigma in (("nu3_even", 1), ("nu3_odd", -1)):
                    for species in sc.SPECIES:
                        self._store(out, name, nu3.block(o3_lambda=0, o3_sigma=sigma,
                                                         center_type=ATOMIC_NUMBERS[species]), n_atoms)
        for name, values in out.items():
            if np.isnan(values).any():
                raise ValueError(f"block {name}: featomic returned no value for some atoms")
        return {name: out[name] for name in self.blocks}

    def widths(self):
        """Features per atom of each block (known after the first call)."""
        return {name: int(self.properties[name].shape[0]) for name in self.blocks}


def atom_norms(per_atom):
    """Mean length of one atom's vector in each block, to judge how small an average is."""
    return {f"atom_norm_{name}": f"{np.linalg.norm(values, axis=1).mean():.6e}" for name, values in per_atom.items()}


def species_means(per_atom, is_co, mask=None):
    """{block: [Co mean | Ge mean]} and the two atom counts (the same for every block)."""
    vectors, counts = {}, None
    for name, values in per_atom.items():
        vectors[name], counts = sc.species_mean(values, is_co, mask)
    return vectors, counts


# --------------------------------------------------------------------------- output

def label_arrays(properties, property_names):
    """properties_<block> and property_names_<block> for an npz file."""
    out = {}
    for name, values in properties.items():
        out[f"properties_{name}"] = np.asarray(values, dtype=np.int32)
        out[f"property_names_{name}"] = np.array(property_names[name])
    return out


def stored_blocks(data):
    """Blocks present in an open npz file, in storage order."""
    present = {key[2:].split("_r")[0] if key.startswith("X_") else None for key in data.files}
    return [name for name in BLOCKS if name in present]


def run_dir(region, tag_or_dir):
    """As soap_common.run_dir, with tags looked up under calculations/descriptors/ACE/standard."""
    return sc.run_dir(region, tag_or_dir, root=STANDARD_ROOT)


# --------------------------------------------------------------------------- loaders

def _choose(available, blocks):
    if blocks is None:
        return list(available)
    unknown = [name for name in blocks if name not in available]
    if unknown:
        raise ValueError(f"blocks {unknown} not stored; available: {list(available)}")
    return [name for name in BLOCKS if name in blocks]


def load_bulk(tag_or_dir, blocks=None):
    """Returns ({block: X (n_cells, 2P)}, index DataFrame in the same row order)."""
    import pandas as pd

    folder = run_dir("global", tag_or_dir)
    with np.load(folder / BULK_FILE) as data:
        chosen = _choose(stored_blocks(data), blocks)
        arrays = {name: data[f"X_{name}"] for name in chosen}
    return arrays, pd.read_csv(folder / "bulk_index.csv", dtype={"plane": str})


def load_gb(tag_or_dir, kind="interface", radius=None, planes="both", blocks=None):
    """Returns ({block: X (N, 2P)}, index) for the grain boundaries.

    kind:   "global"    mean over every atom in the cell
            "interface" atoms within `radius` (A) of an interface plane
    planes: "both" (count-weighted mean of the two interfaces), "edge" or "mid".
    """
    import pandas as pd

    if kind == "global":
        folder = run_dir("global", tag_or_dir)
        index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
        with np.load(folder / GLOBAL_FILE) as data:
            chosen = _choose(stored_blocks(data), blocks)
            return {name: data[f"X_{name}"] for name in chosen}, index
    if kind != "interface":
        raise ValueError(f"unknown kind {kind!r} (ACE stores global and interface vectors)")
    folder = run_dir("both_interfaces", tag_or_dir)
    index = pd.read_csv(folder / "index.csv", dtype={"plane": str})
    with np.load(folder / INTERFACE_FILE) as data:
        radii = data["radii"]
        if radius is None:
            raise ValueError(f"choose a radius from {radii.tolist()}")
        matches = np.flatnonzero(np.isclose(radii, radius))
        if matches.size == 0:
            raise ValueError(f"radius {radius} not stored; available: {radii.tolist()}")
        key = sc.radius_key(radii[matches[0]])
        chosen = _choose(stored_blocks(data), blocks)
        n = data[f"n_{key}"]
        arrays = {name: data[f"X_{name}_{key}"] for name in chosen}
    if planes == "both":
        return {name: sc.combine_planes(X, n) for name, X in arrays.items()}, index
    p = sc.PLANE_NAMES.index(planes)
    return {name: X[:, p] for name, X in arrays.items()}, index


def pristine_norms(tag_or_dir, blocks=None):
    """{block: length of the pristine cell's [Co | Ge] vector}, from bulk_ace.npz of the same setting."""
    arrays, index = load_bulk(tag_or_dir, blocks)
    rows = np.flatnonzero(index["kind"].to_numpy() == "pristine")
    if rows.size != 1:
        raise ValueError("expected exactly one pristine row in bulk_index.csv")
    return {name: float(np.linalg.norm(X[rows[0]])) for name, X in arrays.items()}


# A pristine block shorter than this is zero by symmetry, not small: at l3 <= 2 every B20 atom (on a 3-fold
# axis) sees an axially symmetric density, so nu3_odd vanishes in bulk (3e-19 at l3 = 2; the pristine blocks of
# the grid runs are otherwise 0.2-60 long). Dividing by it would blow the block up by ~1e19.
VANISHING_NORM = 1e-6


def assemble(arrays, scaling="pristine", norms=None):
    """Join the blocks of {block: X} into one (N, sum 2P) array, in storage order.

    scaling "pristine" divides each block by the length of its pristine-bulk vector (norms), so the
    pristine cell has length 1 in every block; "none" joins the raw blocks. A block that vanishes in
    pristine bulk (length below VANISHING_NORM) cannot be scaled that way and is refused.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}")
    parts = []
    for name in (name for name in BLOCKS if name in arrays):
        X = np.asarray(arrays[name], dtype=np.float64)
        if scaling == "pristine":
            if not norms or name not in norms or not norms[name] > 0:
                raise ValueError(f"no pristine norm for block {name}")
            if norms[name] < VANISHING_NORM:
                raise ValueError(f"block {name} vanishes in pristine bulk (length {norms[name]:.1e}), so it cannot be "
                                 "divided by it; leave it out (--ace-blocks) or use --ace-scaling none")
            X = X / norms[name]
        parts.append(X)
    return np.concatenate(parts, axis=1)


def load_gb_vectors(tag_or_dir, kind="interface", radius=None, planes="both", blocks=None, scaling="pristine"):
    """(X, index): load_gb() joined by assemble(); the pristine norms come from the same setting."""
    arrays, index = load_gb(tag_or_dir, kind, radius, planes, blocks)
    norms = pristine_norms(tag_or_dir, list(arrays)) if scaling == "pristine" else None
    return assemble(arrays, scaling, norms), index


def load_bulk_vectors(tag_or_dir, blocks=None, scaling="pristine"):
    """(X, index) for the bulk cells, joined and scaled as load_gb_vectors()."""
    arrays, index = load_bulk(tag_or_dir, blocks)
    norms = pristine_norms(tag_or_dir, list(arrays)) if scaling == "pristine" else None
    return assemble(arrays, scaling, norms), index


def run_metadata():
    """soap_common.run_metadata() plus the featomic and metatensor versions."""
    from importlib import metadata

    meta = sc.run_metadata()
    for name in ("featomic", "metatensor-core"):
        try:
            meta["versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return meta
