"""Synthetic tests for soap_common. Run from the project root with the mlenv environment:

    python -B -m unittest discover -s src/dscribe/CoGe -p 'test_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import soap_common as sc

POSCAR = """test cell
{scale}
{a} 0.0 0.0
0.0 {a} 0.0
0.0 0.0 {a}
Co Ge
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


class ReadStructure(unittest.TestCase):
    def read(self, scale, a):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "POSCAR"
            path.write_text(POSCAR.format(scale=scale, a=a))
            return sc.read_structure(path)

    def test_scale_on_angstrom_vectors_is_ignored(self):
        atoms, corrected = self.read("4.6294498444", "4.0")
        self.assertTrue(corrected)
        np.testing.assert_allclose(atoms.cell.lengths(), 4.0)
        np.testing.assert_allclose(atoms.positions[1], 2.0)

    def test_scale_on_unit_vectors_is_kept(self):
        atoms, corrected = self.read("2.8", "1.0")
        self.assertFalse(corrected)
        np.testing.assert_allclose(atoms.cell.lengths(), 2.8)

    def test_unit_scale_is_untouched(self):
        atoms, corrected = self.read("1.0", "2.8")
        self.assertFalse(corrected)
        np.testing.assert_allclose(atoms.cell.lengths(), 2.8)

    def test_min_pair_distance(self):
        atoms, _ = self.read("1.0", "2.8")
        self.assertAlmostEqual(sc.min_pair_distance(atoms), 2.8 * np.sqrt(3) / 2)
        far, _ = self.read("1.0", "12.0")
        with self.assertRaises(ValueError):
            sc.min_pair_distance(far)


class PlaneDistances(unittest.TestCase):
    def test_edge_plane_is_periodic(self):
        distance, plane = sc.plane_distances([0.99 * 40, 0.01 * 40, 40.0], 40.0)
        np.testing.assert_allclose(distance, [0.4, 0.4, 0.0], atol=1e-12)
        np.testing.assert_array_equal(plane, [0, 0, 0])

    def test_mid_plane_has_both_sides(self):
        distance, plane = sc.plane_distances([18.0, 22.0, 20.0], 40.0)
        np.testing.assert_allclose(distance, [2.0, 2.0, 0.0])
        np.testing.assert_array_equal(plane, [1, 1, 1])

    def test_each_atom_goes_to_its_nearer_plane(self):
        distance, plane = sc.plane_distances([9.0, 11.0, 29.0, 31.0], 40.0)
        np.testing.assert_allclose(distance, 9.0)
        np.testing.assert_array_equal(plane, [0, 1, 1, 0])


class Pooling(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
        self.is_co = np.array([True, True, False, False])

    def test_species_mean_layout(self):
        vector, counts = sc.species_mean(self.X, self.is_co)
        np.testing.assert_allclose(vector, [2.0, 3.0, 20.0, 30.0])
        np.testing.assert_array_equal(counts, [2, 2])

    def test_mask_and_empty_species(self):
        vector, counts = sc.species_mean(self.X, self.is_co, np.array([True, False, False, False]))
        np.testing.assert_allclose(vector[:2], [1.0, 2.0])
        self.assertTrue(np.isnan(vector[2:]).all())
        np.testing.assert_array_equal(counts, [1, 0])

    def test_combine_planes_is_count_weighted(self):
        edge, n_edge = sc.species_mean(self.X, self.is_co, np.array([True, False, True, True]))
        mid, n_mid = sc.species_mean(self.X, self.is_co, np.array([False, True, False, False]))
        combined = sc.combine_planes(np.stack([edge, mid])[None], np.stack([n_edge, n_mid])[None])
        np.testing.assert_allclose(combined[0], sc.species_mean(self.X, self.is_co)[0])

    def test_within_species_spread_is_zero_for_identical_atoms(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [5.0, 6.0]])
        self.assertEqual(sc.within_species_spread(X, self.is_co), 0.0)


class Tags(unittest.TestCase):
    def settings(self, **changes):
        base = {"r_cut": 6.0, "n_max": 8, "l_max": 6, "sigma": 0.5, "rbf": "gto", "weighting": None}
        return {**base, **changes}

    def test_tag(self):
        self.assertEqual(sc.soap_tag(self.settings()), "rc6.0_n8_l6_s0.5")
        self.assertEqual(sc.soap_tag(self.settings(r_cut=10.0, n_max=12, l_max=9)), "rc10.0_n12_l9_s0.5")

    def test_weighting_is_spelled_out(self):
        weighted = sc.soap_tag(self.settings(weighting={"function": "pow", "r0": 3, "m": 4, "c": 1, "d": 1}))
        self.assertEqual(weighted, "rc6.0_n8_l6_s0.5_wpow-r3-c1-d1-m4")
        poly = sc.soap_tag(self.settings(r_cut=8.0, weighting={"function": "poly", "r0": 8.0, "c": 1, "m": 2}))
        self.assertEqual(poly, "rc8.0_n8_l6_s0.5_wpoly-r8-c1-m2")
        with self.assertRaises(ValueError):
            sc.soap_tag(self.settings(weighting={"function": "pow", "r1": 3}))

    def test_make_soap_leaves_the_weighting_unchanged(self):
        weighting = {"function": "pow", "r0": 3, "m": 4, "c": 1, "d": 1}
        settings = {**self.settings(weighting=dict(weighting)), "species": ["Co", "Ge"], "periodic": True}
        sc.make_soap(settings)
        self.assertEqual(settings["weighting"], weighting)


class RunFolders(unittest.TestCase):
    def test_tag_is_looked_up_under_the_root(self):
        self.assertEqual(sc.run_dir("global", "rc6.0_n8_l6_s0.5"), sc.STANDARD_ROOT / "global" / "rc6.0_n8_l6_s0.5")
        self.assertEqual(sc.run_dir("both_interfaces", "x", root=Path("/r")), Path("/r/both_interfaces/x"))

    def test_a_folder_of_one_region_finds_the_other(self):
        folder = Path("/data/standard/both_interfaces/rc6.0_n8_l6_s0.5")
        self.assertEqual(sc.run_dir("global", folder), Path("/data/standard/global/rc6.0_n8_l6_s0.5"))
        self.assertEqual(sc.run_dir("both_interfaces", folder), folder)
        with self.assertRaises(ValueError):
            sc.run_dir("gb", folder)

    def test_radius_key(self):
        self.assertEqual(sc.radius_key(8.0), "r8")
        self.assertEqual(sc.radius_key(np.float64(6.5)), "r6.5")


class RoundTrip(unittest.TestCase):
    def test_load_gb(self):
        with tempfile.TemporaryDirectory() as root:
            global_dir = Path(root) / "standard" / "global" / "tag"
            interface_dir = Path(root) / "standard" / "both_interfaces" / "tag"
            X_interface = np.arange(2 * 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 2, 4)
            n_interface = np.ones((2, 2, 2, 2), dtype=np.int32)
            X_global = np.ones((2, 4), dtype=np.float32)
            sc.write_gb_arrays(global_dir, interface_dir, X_global, X_interface, n_interface,
                               X_interface[:, 0], n_interface[:, 0], np.zeros((2, 2), dtype=np.int32), [4.0, 6.0])
            for folder in (global_dir, interface_dir):
                sc.write_csv(folder / "index.csv", [{"index": 0, "plane": "100"}, {"index": 1, "plane": "110"}],
                             ["index", "plane"])
            with np.load(interface_dir / sc.INTERFACE_FILE) as data:
                self.assertEqual(sorted(k for k in data.files if k.startswith(("X_r", "n_r"))),
                                 ["X_r4", "X_r6", "n_r4", "n_r6"])
            X, index = sc.load_gb(interface_dir, radius=6.0, planes="mid")
            np.testing.assert_array_equal(X, X_interface[:, 1, 1])
            self.assertEqual(index["plane"].tolist(), ["100", "110"])
            both, _ = sc.load_gb(global_dir, radius=4.0)
            np.testing.assert_allclose(both, X_interface[:, 0].mean(axis=1))
            nonbulk, _ = sc.load_gb(interface_dir, kind="nonbulk", planes="edge")
            np.testing.assert_array_equal(nonbulk, X_interface[:, 0, 0])
            whole, _ = sc.load_gb(interface_dir, kind="global")
            np.testing.assert_array_equal(whole, X_global)
            with self.assertRaises(ValueError):
                sc.load_gb(interface_dir, radius=5.0)


if __name__ == "__main__":
    unittest.main()
