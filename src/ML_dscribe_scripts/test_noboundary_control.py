"""Tests for noboundary_control.py: synthetic data, plus one small featomic check on the pristine cell.

Run: python3 -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_noboundary_control.py'
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import noboundary_control as nc


def toy_cell():
    from ase import Atoms

    # Two atoms in each half of a 10 x 2 x 2 cell (fractional x 0.1, 0.3 | 0.6, 0.85).
    return Atoms("CoGeCoGe", scaled_positions=[[0.1, 0, 0], [0.3, 0.5, 0.5], [0.6, 0, 0.5], [0.85, 0.5, 0]],
                 cell=[10.0, 2.0, 2.0], pbc=True)


class Construction(unittest.TestCase):
    def test_halves_and_doubled(self):
        atoms = toy_cell()
        in_a, in_b = nc.halves(atoms)
        self.assertEqual(in_a.tolist(), [True, True, False, False])
        crystal = nc.doubled(atoms, in_b)
        self.assertEqual(len(crystal), 4)
        np.testing.assert_allclose(crystal.positions[:2], atoms.positions[2:])  # originals first, in order
        np.testing.assert_allclose(crystal.get_scaled_positions()[2:, 0], [0.1, 0.35])
        np.testing.assert_allclose(crystal.cell.array, atoms.cell.array)

    def test_own_grain_features(self):
        masks = (np.array([True, False, True, False]), np.array([False, True, False, True]))
        a = {"polar": np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)}
        b = {"polar": -np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2) - 100}
        out = nc.own_grain_features(4, masks, (a, b))["polar"]
        np.testing.assert_array_equal(out[[0, 2]], a["polar"][:2])
        np.testing.assert_array_equal(out[[1, 3]], b["polar"][:2])

    def test_region_means(self):
        import covariant_descriptor as cd

        lx = 40.0
        x = np.array([1.0, 39.0, 21.0, 10.5, 30.0])          # edge, edge, mid, edge (10.5 from 0), mid
        is_co = np.array([True, False, True, False, True])
        values = {"polar": np.arange(5, dtype=float)[:, None, None] * np.ones((5, 3, 2))}
        out, counts = nc.region_means(values, is_co, x, lx, 8.0, cd.split_mean)
        np.testing.assert_array_equal(counts, [[1, 1], [1, 0]])   # 10.5 and 30.0 are beyond 8 A
        np.testing.assert_allclose(out["polar"][0, 0], [0, 0, 1, 1])  # [Co mean | Ge mean]
        self.assertTrue(np.isnan(out["polar"][1, 0, 2:]).all())

    def test_failed_checks(self):
        good = {"min_pair_A": 2.39, "min_pair_B": 2.39, "cell_average_A": 1e-7, "cell_average_B": 1e-7}
        self.assertEqual(nc.failed_checks(good), [])
        self.assertEqual(nc.failed_checks({**good, "min_pair_B": 1.0, "gb_vs_noboundary": 1e-3}), ["min_pair_B"])
        self.assertEqual(nc.failed_checks({**good, "cell_average_A": float("nan")}), ["cell_average_A"])


class Identity(unittest.TestCase):
    def test_identity_holds_by_construction(self):
        rng = np.random.default_rng(0)
        S, F = 6, 8
        is_co = np.arange(F) < F // 2
        n_interface = rng.integers(5, 20, size=(S, 2, 2))
        n_species = n_interface.sum(axis=1) + rng.integers(10, 30, size=(S, 2))
        interface = rng.normal(size=(S, 2, F))
        interface_nb = rng.normal(size=(S, 2, F))
        w = [nc.species_weights(n_interface[:, p], is_co) for p in (0, 1)]
        cell = sum(w[p] * (interface[:, p] - interface_nb[:, p]) for p in (0, 1)) / nc.species_weights(n_species, is_co)
        out = nc.identity_residuals(cell, interface, interface_nb, n_interface, n_species, is_co)
        np.testing.assert_allclose(out["identity_with_C_nb"], 0.0, atol=1e-12)
        self.assertTrue((out["identity_without_C_nb"] > 0.1).all())

    def test_fit_06_is_zero_for_a_scaled_sum(self):
        rng = np.random.default_rng(1)
        interface = rng.normal(size=(4, 2, 5))
        cell = 0.3 * (interface[:, 0] + interface[:, 1])
        ones = np.ones((4, 2), dtype=int)
        out = nc.identity_residuals(cell, interface, np.zeros_like(interface), np.ones((4, 2, 2), dtype=int), ones,
                                    np.arange(5) < 2)
        np.testing.assert_allclose(out["fit_06"], 0.0, atol=1e-12)

    def test_size_measures(self):
        C = np.array([[3.0, 4.0], [1.0, 0.0]])
        out = nc.size_measures(C, np.array([[0.0, 0.0], [2.0, 0.0]]))
        np.testing.assert_allclose(out["size_ratio"], [0.0, 2.0])
        np.testing.assert_allclose(out["cosine"], [0.0, 1.0])
        np.testing.assert_allclose(out["relative_difference"], [1.0, 1.0])


class PristineWithoutBoundary(unittest.TestCase):
    """A 'GB' cell that is pristine bulk repeated 16 times has no boundary, so C_nb must equal C."""

    def test_pristine_16x(self):
        try:
            import featomic  # noqa: F401
        except ImportError:
            self.skipTest("featomic not installed (use src/mlenv)")
        import covariant_descriptor as cd
        from ase.io import write

        atoms, _ = nc.sc.read_structure(nc.sc.PRISTINE_POSCAR)
        settings = {"r_cut": 4.0, "n_max": 3, "l_max": 2, "sigma": 0.5, "smoothing": 0.5}
        calculator, keys = cd.make_calculator(settings, ["polar", "tensor"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "POSCAR"
            write(path, atoms.repeat((16, 1, 1)), format="vasp", direct=True)
            result = nc.describe_family(path, calculator, keys, 8.0, check_gb=True)
        self.assertEqual(nc.failed_checks(result["checks"]), [])
        self.assertLess(result["checks"]["gb_vs_noboundary"], 1e-6)
        self.assertAlmostEqual(result["checks"]["min_pair_A"], 2.3975, places=3)
        # Lx/2 is 8 whole periods, so the two regions hold the same atoms up to a translation.
        np.testing.assert_array_equal(result["n"][0], result["n"][1])
        self.assertGreater(result["n"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
