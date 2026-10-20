"""Unit tests of gb_symmetry.py on synthetic cells and tables.

  src/dft_plots/.venv/bin/python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_gb_symmetry.py'

The spglib tests need the dft_plots environment; under mlenv (no spglib) they are skipped.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gb_symmetry as gs  # noqa: E402

try:
    import spglib  # noqa: F401
    HAVE_SPGLIB = True
except ImportError:
    HAVE_SPGLIB = False

A = 4.6294498444  # B20 CoGe lattice constant (structures/CoGeGrainBoundaries/POSCAR)


def b20_sites(u):
    """The 4a positions of P2_13 at parameter u."""
    return np.array([[u, u, u], [0.5 + u, 0.5 - u, -u], [-u, 0.5 + u, 0.5 - u], [0.5 - u, -u, 0.5 + u]]) % 1.0


def b20_cell():
    frac = np.vstack([b20_sites(0.140), b20_sites(0.841)])
    return np.diag([A, A, A]), frac, np.repeat([0, 1], 4)


def two_grain_cell(m=2):
    """Grain A (m cubic cells along x) joined to its inversion image: an inversion centre at x = 0."""
    _, frac, kinds = b20_cell()
    grain = np.vstack([(frac + [i, 0, 0]) / [2 * m, 1, 1] for i in range(m)])
    grain_kinds = np.tile(kinds, m)
    joined = np.vstack([grain, (-grain) % 1.0])
    return np.diag([2 * m * A, A, A]), joined % 1.0, np.concatenate([grain_kinds, grain_kinds])


def rotated_supercell():
    """B20 in the orthogonal cell a(1,0,0), a(0,1,1), a(0,-1,1), turned so that this cell is diagonal.

    The cubic 3-fold axes do not keep this cell, so spglib's list for the input cell misses them."""
    _, frac, kinds = b20_cell()
    S = A * np.array([[1.0, 0, 0], [0, 1, 1], [0, -1, 1]])
    shifts = np.array([[i, j, k] for i in range(-1, 3) for j in range(-1, 3) for k in range(-1, 3)])
    points, point_kinds = [], []
    for f, kind in zip(frac, kinds):
        cand = ((f + shifts) * A) @ np.linalg.inv(S)
        cand = cand[np.all((cand > -1e-9) & (cand < 1 - 1e-9), axis=1)]
        points.extend(cand)
        point_kinds.extend([kind] * len(cand))
    lattice = np.diag([A, np.sqrt(2) * A, np.sqrt(2) * A])
    order = np.argsort(point_kinds, kind="stable")
    return lattice, np.array(points)[order] % 1.0, np.array(point_kinds)[order]


class OperationTests(unittest.TestCase):
    n = np.array([1.0, 0, 0])

    def test_classify(self):
        cases = {"inversion": -np.eye(3), "mirror_parallel": np.diag([-1.0, 1, 1]),
                 "twofold_normal": np.diag([1.0, -1, -1]), "twofold_inplane": np.diag([-1.0, -1, 1])}
        for name, R in cases.items():
            flags = gs.classify(R, self.n)
            self.assertTrue(flags[name], name)
            self.assertEqual(sum(flags[f] for f in gs.FLAGS), 1, name)
        self.assertTrue(gs.classify(-np.eye(3), self.n)["improper"])
        threefold = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.assertFalse(any(gs.classify(threefold, self.n)[f] for f in gs.FLAGS + ("improper",)))
        mirror_containing_x = np.diag([1.0, -1, 1])
        flags = gs.classify(mirror_containing_x, self.n)
        self.assertTrue(flags["improper"])
        self.assertFalse(any(flags[f] for f in gs.FLAGS))

    def test_cartesian_on_a_hexagonal_basis(self):
        lattice = np.array([[1.0, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 1]])
        W = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])  # a1 -> a2, a2 -> -a1 - a2
        c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        np.testing.assert_allclose(gs.cartesian(W, lattice), [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-12)

    def test_centre_offset_and_edge_image(self):
        L = 40.0
        self.assertAlmostEqual(gs.centre_offset(0.0, L), 0.0)
        self.assertAlmostEqual(gs.centre_offset(L, L), 0.0)
        self.assertAlmostEqual(gs.centre_offset(L / 2, L), 0.5)  # fixed plane at Lx/4: middle of a grain
        self.assertAlmostEqual(gs.centre_offset(-0.4, L), 0.01)
        self.assertAlmostEqual(gs.edge_image(0.0, L), 0.0)
        self.assertAlmostEqual(gs.edge_image(L / 2, L), 1.0)
        self.assertAlmostEqual(gs.edge_image(-1.0, L), 0.05)

    def test_tol_name_and_enantiomorphs(self):
        self.assertEqual([gs.tol_name(x) for x in (1e-3, 5e-3, 2e-2)], ["1e-3", "5e-3", "2e-2"])
        self.assertEqual(gs.canonical_group(78), 76)
        self.assertEqual(gs.canonical_group(213), 212)
        self.assertEqual(gs.canonical_group(198), 198)

    def test_misfit_of_a_true_and_a_false_operation(self):
        lattice, frac, kinds = b20_cell()
        screw = (np.diag([1.0, -1, -1]), A * np.array([0.5, 0.5, 0.0]))  # 2_1 along x of P2_13
        self.assertLess(gs.misfit(lattice, frac, kinds, *screw), 1e-9)
        self.assertGreater(gs.misfit(lattice, frac, kinds, -np.eye(3), np.zeros(3)), 0.1)


@unittest.skipUnless(HAVE_SPGLIB, "spglib is in src/dft_plots/.venv")
class SpglibTests(unittest.TestCase):
    def test_b20_bulk(self):
        labels = gs.cell_labels(*b20_cell(), symprec=1e-3)
        self.assertEqual(labels["space_group"], 198)
        self.assertEqual(labels["n_rotations"], 12)
        self.assertEqual(labels["point_group_order"], 12)
        self.assertFalse(labels["improper"] or labels["inversion"] or labels["mirror_parallel"])
        self.assertTrue(labels["twofold_normal"] and labels["twofold_inplane"])
        self.assertLess(labels["max_misfit_A"], 1e-6)
        self.assertTrue(labels["tilts_normal"])  # the 3-fold axes of bulk do not keep x

    def test_table_route_finds_every_rotation(self):
        labels = gs.cell_labels(*rotated_supercell(), symprec=1e-3)
        self.assertEqual(labels["space_group"], 198)
        self.assertEqual(labels["n_rotations"], 12)
        self.assertEqual(labels["point_group_order"], 12)
        self.assertLess(labels["n_rotations_input_cell"], 12)  # what the input cell alone would give
        self.assertLess(labels["max_misfit_A"], 1e-6)

    def test_operations_fit_a_shifted_origin(self):
        lattice, frac, kinds = two_grain_cell()
        shifted = (frac + [0.113, 0.271, 0.402]) % 1.0
        labels = gs.cell_labels(lattice, shifted, kinds, symprec=1e-3)
        self.assertTrue(labels["inversion"])
        self.assertLess(labels["max_misfit_A"], 1e-6)
        self.assertGreater(labels["improper_centre_offset"], 0.0)  # the centre moved off x = 0 with the shift

    def test_two_grain_cell_has_an_inversion_at_the_interface(self):
        labels = gs.cell_labels(*two_grain_cell(), symprec=1e-3)
        self.assertTrue(labels["inversion"] and labels["improper"])
        self.assertLess(labels["improper_centre_offset"], 0.05)
        self.assertEqual(labels["n_rotations"], labels["point_group_order"])
        self.assertFalse(labels["tilts_normal"])
        table = pd.DataFrame([{"family": "100/sym99999", "handedness": "opposite", **labels}]).assign(
            improper=int(labels["improper"]))
        self.assertTrue(gs.check_handedness(table)["pass"])
        self.assertFalse(gs.check_handedness(table.assign(handedness="same"))["pass"])
        self.assertTrue(gs.check_centres(table)["pass"])
        self.assertFalse(gs.check_centres(table.assign(improper_centre_offset=0.5))["pass"])


class TableTests(unittest.TestCase):
    def test_recode(self):
        table = pd.DataFrame({"family": [f"f{i}" for i in range(8)], "grain_class": list(gs.CLASSES),
                              "handedness": [v[0] for v in gs.CLASSES.values()]})
        out = gs.recode(table)
        self.assertEqual(out.set_index("grain_class").loc["R70.53[1-10]", "lattice"], "twinned")
        self.assertEqual(out.set_index("grain_class").loc["I.R90[100]", "motif"], "turned")
        self.assertEqual(out.set_index("grain_class").loc["I", "lattice"], "shared")
        self.assertEqual(len(out[["handedness", "lattice", "motif"]].drop_duplicates()), 8)
        with self.assertRaises(ValueError):
            gs.recode(table.assign(grain_class=["E"] * 7 + ["R45[001]"]))
        with self.assertRaises(ValueError):
            gs.recode(table.assign(handedness="same"))

    def test_groups_share_labels(self):
        flags = dict.fromkeys(gs.FLAGS, 0)
        table = pd.DataFrame([{"family": "a", "space_group": 76, **flags}, {"family": "b", "space_group": 78, **flags},
                              {"family": "c", "space_group": 4, **flags}, {"family": "d", "space_group": 4, **flags},
                              {"family": "e", "space_group": 1, **flags}])
        census = pd.DataFrame({"family": list("abcde"), "distinct_id": ["D1", "D1", "D2", "D2", "D3"]})
        check = gs.check_groups(table, census)
        self.assertTrue(check["pass"])
        self.assertEqual(check["n_groups_tested"], 2)
        self.assertEqual(check["n_groups_differing_only_by_enantiomorph"], 1)
        table.loc[table["family"] == "d", "twofold_normal"] = 1
        self.assertFalse(gs.check_groups(table, census)["pass"])

    def test_thickness_check(self):
        row = {"family": "a", "space_group": 4, **dict.fromkeys(gs.FLAGS, 0)}
        table = pd.DataFrame([row])
        self.assertTrue(gs.check_thickness(table, {("a", 4): dict(row), ("a", 12): dict(row)})["pass"])
        self.assertFalse(gs.check_thickness(table, {("a", 4): {**row, "inversion": 1}})["pass"])

    def test_plain_bulk_check(self):
        table = pd.DataFrame({"family": list(gs.mc.NO_BOUNDARY), "space_group": [198, 198]})
        self.assertTrue(gs.check_plain_bulk(table)["pass"])
        self.assertFalse(gs.check_plain_bulk(table.assign(space_group=[198, 19]))["pass"])
        self.assertFalse(gs.check_plain_bulk(table.iloc[:1])["pass"])

    def test_family_selection_and_sample(self):
        rows = [{"family": f"{p}/sym{i:05d}", "plane": p, "grain_class": c}
                for p in gs.PLANES for i, c in enumerate(["E", "I", "R90[100]", "I.R90[100]"] * 25)]
        rows += [{"family": f, "plane": "111", "grain_class": "E"} for f in gs.mc.NO_BOUNDARY]
        labels = pd.DataFrame(rows)
        picked = gs.select_families(labels, 3)
        self.assertEqual(len(picked), 3 * 3 + 2)
        self.assertTrue(set(gs.mc.NO_BOUNDARY) <= set(picked["family"]))
        sample = gs.check_sample(labels, 8, seed=0)
        self.assertEqual(sample, gs.check_sample(labels, 8, seed=0))
        for plane in gs.PLANES:
            mine = labels[labels["family"].isin(sample) & (labels["plane"] == plane)
                          & ~labels["family"].isin(gs.mc.NO_BOUNDARY)]
            self.assertEqual(len(mine), 8)
            self.assertEqual(mine["grain_class"].value_counts().min(), 2)  # round robin over the 4 classes


if __name__ == "__main__":
    unittest.main()
