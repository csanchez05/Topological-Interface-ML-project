"""Direct comparison of two grain-boundary cells, atom by atom.

SOAP cannot see rotations, mirrors or inversion, so "the same interface built
twice" is tested as: is there a rigid motion, proper or improper, that puts
every atom of one structure on an atom of the same species in the other?

Two tests:

  match_cells       the whole periodic cell (both interfaces and both grains)
  match_interfaces  the slab of atoms within a half-width of one interface plane

Both try every rigid motion that keeps the interface normal (x) along x or -x,
combined with an in-plane rotation by a multiple of 60 or 90 degrees, an
optional in-plane mirror, and any shift. A motion counts as a match when

  1. it maps the in-plane periods of each structure onto translations that
     really are symmetries of the other (so the cells need not look alike),
  2. every atom of the first lands on an atom of the same species in the second,
  3. and every atom of the second lands on an atom of the first under the
     reverse motion.

Together these make the two infinite periodic structures equal to within the
position tolerance. The tolerance must stay below half the shortest distance
between two atoms of one species, otherwise two atoms could share a partner.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TOL = 0.02       # A; coordinates are written to 3-6 decimals, i.e. to about 0.003 A
DEFAULT_HALFWIDTH = 12.0  # A; longer than the bulk period along x on every plane (4.6, 6.5, 8.0 A)
PLANE_NAMES = ("edge", "mid")
_FINGERPRINT_CUTOFF = 4.0


def point_operations():
    """Orthogonal matrices that send x to +x or -x: 2 x 8 in-plane rotations x optional mirror = 32."""
    operations = []
    for x_sign in (1.0, -1.0):
        for angle in (0, 60, 90, 120, 180, 240, 270, 300):
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            for mirror in (1.0, -1.0):
                operations.append(np.array([[x_sign, 0.0, 0.0],
                                            [0.0, c, -s * mirror],
                                            [0.0, s, c * mirror]]))
    return operations


_OPERATIONS = point_operations()


def describe_operation(operation):
    det = int(round(np.linalg.det(operation)))
    angle = np.degrees(np.arctan2(operation[2, 1], operation[1, 1])) % 360.0
    mirrored = np.linalg.det(operation[1:, 1:]) < 0
    return (f"x->{'+' if operation[0, 0] > 0 else '-'}x, in-plane "
            f"{'mirror then ' if mirrored else ''}rotation {angle:.0f} deg, {'proper' if det > 0 else 'improper'}")


class Structure:
    """An orthogonal periodic cell prepared for matching."""

    def __init__(self, atoms, name=""):
        from scipy.spatial import cKDTree

        cell = atoms.cell.array
        self.lengths = np.diag(cell).copy()
        if np.abs(cell - np.diag(self.lengths)).max() > 1e-6:
            raise ValueError(f"{name}: cell is not orthogonal")
        self.name = name
        self.atoms = atoms
        self.species = np.array(atoms.get_chemical_symbols())
        self.positions = self._wrap(atoms.get_positions())
        self.index = {s: np.flatnonzero(self.species == s) for s in sorted(set(self.species))}
        self.trees = {s: cKDTree(self.positions[rows], boxsize=self.lengths) for s, rows in self.index.items()}
        self._fingerprint = None
        self._translations = {}

    @classmethod
    def from_file(cls, path):
        from ml_common import sc

        atoms, _ = sc.read_structure(path)
        return cls(atoms, name=str(path))

    def _wrap(self, positions):
        wrapped = np.mod(positions, self.lengths)
        wrapped[wrapped >= self.lengths] = 0.0
        return wrapped

    def locate(self, positions, species, tol):
        """Index of the nearest atom of the right species within tol of each position.

        Returns None when some position has no such atom.
        """
        found = np.empty(len(positions), dtype=int)
        wrapped = self._wrap(positions)
        for s in set(species):
            if s not in self.trees:
                return None
            rows = np.flatnonzero(species == s)
            distance, nearest = self.trees[s].query(wrapped[rows], k=1, distance_upper_bound=tol)
            if not np.all(np.isfinite(distance)):
                return None
            found[rows] = self.index[s][nearest]
        return found

    def is_translation(self, vector, tol, rows=None, key=None):
        """True when shifting the atoms by `vector` puts each one on an atom of its species.

        rows: restrict the test to these atoms (a slab); key names that choice for the cache.
        """
        cache_key = (key, tol, tuple(np.round(vector, 6)))
        if cache_key not in self._translations:
            rows = slice(None) if rows is None else rows
            self._translations[cache_key] = (
                self.locate(self.positions[rows] + vector, self.species[rows], tol) is not None)
        return self._translations[cache_key]

    def shortest_same_species_distance(self):
        distances = [d[0] for key, d in self.fingerprint().items() if len(d) and key[:2] == key[2:]]
        return float(min(distances)) if distances else float("inf")

    def plane_offsets(self, plane):
        """Signed distance along x from each atom to an interface plane ('edge' at 0, 'mid' at Lx/2)."""
        lx = self.lengths[0]
        origin = 0.0 if plane == "edge" else lx / 2.0
        return (self.positions[:, 0] - origin + lx / 2.0) % lx - lx / 2.0

    def fingerprint(self):
        """Sorted interatomic distances up to a cutoff, per pair of species."""
        if self._fingerprint is None:
            from ase.neighborlist import neighbor_list

            i, j, d = neighbor_list("ijd", self.atoms, _FINGERPRINT_CUTOFF)
            keep = i < j
            first, second = self.species[i], self.species[j]
            pair = np.char.add(np.where(first <= second, first, second), np.where(first <= second, second, first))
            self._fingerprint = {key: np.sort(d[keep & (pair == key)]) for key in sorted(set(pair.tolist()))}
        return self._fingerprint


def fingerprints_agree(a, b, tol):
    """A cheap necessary condition for two cells to be related by a rigid motion.

    Moving every atom by at most tol changes each distance by at most 2 tol, so
    the k-th smallest distances of the two cells must agree to 2 tol.
    """
    fa, fb = a.fingerprint(), b.fingerprint()
    if set(fa) != set(fb):
        return False
    for key in fa:
        n = int(np.searchsorted(fa[key], _FINGERPRINT_CUTOFF - 4.0 * tol))
        if len(fb[key]) < n or np.abs(fa[key][:n] - fb[key][:n]).max(initial=0.0) > 2.0 * tol:
            return False
    return True


def _lattice_compatible(operation, a, b, tol, rows_a=None, rows_b=None, key_a=None, key_b=None):
    """The in-plane cell vectors of each structure, moved by the operation (or its inverse),
    must be translations of the other. rows/key restrict the test to slabs."""
    for axis in (1, 2):
        vector = np.zeros(3)
        vector[axis] = a.lengths[axis]
        if not b.is_translation(operation @ vector, tol, rows_b, key_b):
            return False
        vector[axis] = b.lengths[axis]
        if not a.is_translation(operation.T @ vector, tol, rows_a, key_a):
            return False
    return True


def match_cells(a, b, tol=DEFAULT_TOL):
    """Whole-cell test. Returns a dict describing the rigid motion, or None when there is none."""
    if abs(a.lengths[0] - b.lengths[0]) > tol:
        return None
    if not fingerprints_agree(a, b, tol):
        return None
    reference = 0
    candidates = b.index.get(a.species[reference], ())
    for operation in _OPERATIONS:
        if not _lattice_compatible(operation, a, b, tol):
            continue
        moved = a.positions @ operation.T
        for j in candidates:
            shift = b.positions[j] - moved[reference]
            if b.locate(moved + shift, a.species, tol) is None:
                continue
            if a.locate((b.positions - shift) @ operation, b.species, tol) is None:
                continue
            return {"operation": describe_operation(operation), "proper": bool(np.linalg.det(operation) > 0)}
    return None


def match_interfaces(a, plane_a, b, plane_b, halfwidth=DEFAULT_HALFWIDTH, tol=DEFAULT_TOL):
    """Slab test: do the atoms within `halfwidth` of plane_a in `a` coincide, after a rigid motion
    that puts plane_a on plane_b, with the atoms within `halfwidth` of plane_b in `b`?

    The motion may flip the normal and move freely in the plane, but it may not
    shift along the normal: the two dividing planes must coincide. Atoms within
    2 tol of the slab edge need not have a partner.
    Returns a dict describing the motion, or None.
    """
    halfwidth = min(halfwidth, a.lengths[0] / 2.0 - 1.0, b.lengths[0] / 2.0 - 1.0)
    offset_a, offset_b = a.plane_offsets(plane_a), b.plane_offsets(plane_b)
    slab_a = np.flatnonzero(np.abs(offset_a) <= halfwidth)
    slab_b = np.flatnonzero(np.abs(offset_b) <= halfwidth)
    core_a = np.flatnonzero(np.abs(offset_a) <= halfwidth - 2.0 * tol)
    core_b = np.flatnonzero(np.abs(offset_b) <= halfwidth - 2.0 * tol)
    if core_a.size == 0 or core_b.size == 0:
        return None
    relative_a = np.column_stack([offset_a[core_a], a.positions[core_a, 1:]])
    relative_b = np.column_stack([offset_b[core_b], b.positions[core_b, 1:]])
    origin_a = np.array([0.0 if plane_a == "edge" else a.lengths[0] / 2.0, 0.0, 0.0])
    origin_b = np.array([0.0 if plane_b == "edge" else b.lengths[0] / 2.0, 0.0, 0.0])
    reference = int(np.argmin(np.abs(relative_a[:, 0])))
    same_species_b = b.index.get(a.species[core_a[reference]], np.array([], dtype=int))
    key_a, key_b = (plane_a, round(halfwidth, 6)), (plane_b, round(halfwidth, 6))
    for operation in _OPERATIONS:
        if not _lattice_compatible(operation, a, b, tol, slab_a, slab_b, key_a, key_b):
            continue
        moved = relative_a @ operation.T
        candidates = same_species_b[np.abs(offset_b[same_species_b] - moved[reference, 0]) <= tol]
        for j in candidates:
            shift = np.array([0.0, b.positions[j, 1] - moved[reference, 1], b.positions[j, 2] - moved[reference, 2]])
            if b.locate(moved + shift + origin_b, a.species[core_a], tol) is None:
                continue
            if a.locate((relative_b - shift) @ operation + origin_a, b.species[core_b], tol) is None:
                continue
            return {"operation": describe_operation(operation), "proper": bool(np.linalg.det(operation) > 0),
                    "n_atoms": int(core_a.size)}
    return None


def classify(items, same):
    """Sort items into classes of mutually equal ones. `same(i, j)` compares two items.

    Each item is compared with one representative per class. Returns the class number of each item.
    """
    representatives, classes = [], []
    for i in range(len(items)):
        for number, representative in enumerate(representatives):
            if same(representative, i):
                classes.append(number)
                break
        else:
            representatives.append(i)
            classes.append(len(representatives) - 1)
    return classes


def compare_group(structures, halfwidth=DEFAULT_HALFWIDTH, tol=DEFAULT_TOL):
    """Compare every structure of a group.

    Returns a dict with, per structure, its whole-cell class (0, 1, ...) and the
    classes of its two interfaces (letters, edge then mid), plus how the first
    match to each class representative was made.
    """
    cell_how = {}

    def same_cell(i, j):
        result = match_cells(structures[i], structures[j], tol)
        if result is not None:
            cell_how[j] = result["operation"]
        return result is not None

    cell_classes = classify(structures, same_cell)
    interfaces = [(i, plane) for i in range(len(structures)) for plane in PLANE_NAMES]

    def same_interface(m, n):
        (i, plane_i), (j, plane_j) = interfaces[m], interfaces[n]
        return match_interfaces(structures[i], plane_i, structures[j], plane_j, halfwidth, tol) is not None

    interface_classes = classify(interfaces, same_interface)
    letters = [_letter(c) for c in interface_classes]
    return {
        "cell_class": cell_classes,
        "cell_operation": [cell_how.get(i, "") for i in range(len(structures))],
        "edge_interface": letters[0::2],
        "mid_interface": letters[1::2],
    }


def _letter(number):
    text = ""
    number += 1
    while number:
        number, remainder = divmod(number - 1, 26)
        text = chr(ord("a") + remainder) + text
    return text
