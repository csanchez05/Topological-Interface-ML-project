"""Synthetic tests for ace_common. Run from the project root with the mlenv environment:

    python -B -m unittest discover -s src/ACE/CoGe -p 'test_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from ase import Atoms

import ace_common as ac

sc = ac.sc
# Small expansions keep the tests fast; l3 = 2 already has odd (l1 + l2 + l3) couplings.
SETTINGS = {"r_cut": 4.0, "n_max": 3, "l_max": 2, "sigma": 0.5, "smoothing": 0.5, "density": "gaussian",
            "nu_max": 3, "n3": 2, "l3": 2}


def random_cell(seed=0):
    rng = np.random.default_rng(seed)
    # Atoms on a jittered grid, so none sit on top of each other.
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    fractional = (grid + 0.5 + rng.uniform(-0.2, 0.2, size=grid.shape)) / 2.0
    return Atoms(["Co", "Ge"] * 4, scaled_positions=fractional, cell=np.diag([4.6, 4.8, 5.0]), pbc=True)


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def relative(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


class Symmetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator = ac.Calculator(SETTINGS)
        cls.atoms = random_cell()
        cls.reference = cls.calculator.per_atom(cls.atoms)

    def test_blocks_and_shapes(self):
        self.assertEqual(list(self.reference), list(ac.BLOCKS))
        widths = self.calculator.widths()
        self.assertEqual(widths["nu1"], 2 * SETTINGS["n_max"])          # neighbour species x n
        self.assertEqual(widths["nu2"], (SETTINGS["l_max"] + 1) * (2 * SETTINGS["n_max"]) ** 2)
        for name, values in self.reference.items():
            self.assertEqual(values.shape, (len(self.atoms), widths[name]))
            self.assertGreater(np.abs(values).max(), 1e-6, name)

    def test_rotation_leaves_every_block_unchanged(self):
        rotation = random_rotation(3)
        turned = self.atoms.copy()
        turned.set_cell(self.atoms.cell.array @ rotation.T, scale_atoms=False)
        turned.positions = self.atoms.positions @ rotation.T
        blocks = self.calculator.per_atom(turned)
        for name in ac.BLOCKS:
            self.assertLess(relative(blocks[name], self.reference[name]), 1e-8, name)

    def test_inversion_flips_only_the_odd_block(self):
        inverted = self.atoms.copy()
        inverted.positions = -self.atoms.positions
        inverted.wrap()
        blocks = self.calculator.per_atom(inverted)
        for name, (_, sigma) in ac.BLOCKS.items():
            self.assertLess(relative(blocks[name], sigma * self.reference[name]), 1e-8, name)
        # The odd block really changes: it is not zero.
        self.assertGreater(relative(blocks["nu3_odd"], self.reference["nu3_odd"]), 1.0)

    def test_chunks_give_the_same_numbers(self):
        chunked = ac.Calculator(SETTINGS, chunk=3).per_atom(self.atoms)
        for name in ac.BLOCKS:
            np.testing.assert_array_equal(chunked[name], self.reference[name])

    def test_nu2_is_the_soap_power_spectrum(self):
        """nu2 equals featomic's invariant lambda-SOAP block (same features, possibly reordered)."""
        import featomic
        from featomic.clebsch_gordan import EquivariantPowerSpectrum
        from metatensor import Labels

        hypers = ac.hypers(SETTINGS, SETTINGS["n_max"], SETTINGS["l_max"])
        power = EquivariantPowerSpectrum(featomic.SphericalExpansion(**hypers), neighbor_types=[27, 32])
        keys = Labels(["o3_lambda", "o3_sigma"], np.array([[0, 1]], dtype=np.int32))
        A, B = [], []
        for seed in range(4):
            atoms = random_cell(seed)
            tensor = power.compute(atoms, selected_keys=keys, neighbors_to_properties=True)
            is_co = np.array(atoms.get_chemical_symbols()) == "Co"
            values = np.zeros((len(atoms), self.calculator.widths()["nu2"]))
            for center in (27, 32):
                block = tensor.block(o3_lambda=0, o3_sigma=1, center_type=center)
                values[np.asarray(block.samples.column("atom"))] = block.values[:, 0, :]
            A.append(sc.species_mean(values, is_co)[0])
            B.append(ac.species_means(self.calculator.per_atom(atoms), is_co)[0]["nu2"])
        A, B = np.array(A), np.array(B)
        self.assertEqual(A.shape, B.shape)
        self.assertLess(relative(B @ B.T, A @ A.T), 1e-10)
        np.testing.assert_allclose(np.sort(B, axis=1), np.sort(A, axis=1), rtol=1e-10, atol=1e-14)

    def test_nu2_only_setting(self):
        calculator = ac.Calculator({**SETTINGS, "nu_max": 2})
        blocks = calculator.per_atom(self.atoms)
        self.assertEqual(list(blocks), ["nu1", "nu2"])
        np.testing.assert_array_equal(blocks["nu2"], self.reference["nu2"])


class Tags(unittest.TestCase):
    def settings(self, **changes):
        base = {"r_cut": 6.0, "n_max": 8, "l_max": 6, "sigma": 0.5, "smoothing": 0.5, "density": "gaussian",
                "nu_max": 3, "n3": 4, "l3": 4}
        return {**base, **changes}

    def test_reference_tag_starts_with_the_soap_tag(self):
        self.assertEqual(ac.ace_tag(self.settings()), "rc6.0_n8_l6_s0.5_nu3-n4-l4")
        soap = sc.soap_tag({"r_cut": 6.0, "n_max": 8, "l_max": 6, "sigma": 0.5, "rbf": "gto", "weighting": None})
        self.assertTrue(ac.ace_tag(self.settings()).startswith(soap + "_"))

    def test_variants(self):
        self.assertEqual(ac.ace_tag(self.settings(nu_max=2, n3=None, l3=None)), "rc6.0_n8_l6_s0.5_nu2")
        self.assertEqual(ac.ace_tag(self.settings(density="dirac", sigma=None)), "rc6.0_n8_l6_dirac_nu3-n4-l4")
        self.assertEqual(ac.ace_tag(self.settings(smoothing=1.0, n3=6, l3=3)), "rc6.0_n8_l6_s0.5_sm1_nu3-n6-l3")

    def test_blocks_for(self):
        self.assertEqual(ac.blocks_for(self.settings()), ["nu1", "nu2", "nu3_even", "nu3_odd"])
        self.assertEqual(ac.blocks_for(self.settings(nu_max=2)), ["nu1", "nu2"])

    def test_dirac_hypers(self):
        self.assertEqual(ac.hypers(self.settings(density="dirac"), 4, 4)["density"], {"type": "DiracDelta"})
        self.assertEqual(ac.hypers(self.settings(), 8, 6)["basis"]["radial"], {"type": "Gto", "max_radial": 7})

    def test_soap_scripts_are_reachable(self):
        self.assertEqual(tuple(ac.all_thicknesses()), tuple(range(4, 13)))


class RoundTrip(unittest.TestCase):
    """write as gb_descriptor.py / bulk_descriptor.py do, read back with the loaders."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "standard"
        self.global_dir, self.interface_dir = root / "global" / "tag", root / "both_interfaces" / "tag"
        rng = np.random.default_rng(0)
        self.widths = {"nu1": 2, "nu2": 3, "nu3_even": 4, "nu3_odd": 1}
        self.inter = {name: rng.normal(size=(2, 2, 2 * w)).astype(np.float32) for name, w in self.widths.items()}
        self.glob = {name: rng.normal(size=(2, 2 * w)).astype(np.float32) for name, w in self.widths.items()}
        self.bulk = {name: rng.normal(size=(3, 2 * w)) for name, w in self.widths.items()}
        n = np.array([[[3, 1], [1, 3]], [[2, 2], [2, 2]]], dtype=np.int32)
        self.n = n
        sc.write_npz(self.global_dir / ac.GLOBAL_FILE, **{f"X_{k}": v for k, v in self.glob.items()})
        sc.write_npz(self.global_dir / ac.BULK_FILE, **{f"X_{k}": v for k, v in self.bulk.items()})
        sc.write_npz(self.interface_dir / ac.INTERFACE_FILE, **{f"X_{k}_r8": v for k, v in self.inter.items()},
                     n_r8=n, radii=np.array([8.0]))
        for folder in (self.global_dir, self.interface_dir):
            sc.write_csv(folder / "index.csv", [{"index": 0, "plane": "100"}, {"index": 1, "plane": "110"}],
                         ["index", "plane"])
        sc.write_csv(self.global_dir / "bulk_index.csv",
                     [{"index": k, "kind": kind} for k, kind in enumerate(("pristine", "reference", "transformed"))],
                     ["index", "kind"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_gb(self):
        arrays, index = ac.load_gb(self.interface_dir, radius=8.0, planes="mid")
        self.assertEqual(list(arrays), list(ac.BLOCKS))
        np.testing.assert_array_equal(arrays["nu2"], self.inter["nu2"][:, 1])
        self.assertEqual(index["plane"].tolist(), ["100", "110"])
        both, _ = ac.load_gb(self.global_dir, radius=8.0, blocks=["nu3_odd", "nu1"])
        self.assertEqual(list(both), ["nu1", "nu3_odd"])
        np.testing.assert_allclose(both["nu1"], sc.combine_planes(self.inter["nu1"], self.n))
        whole, _ = ac.load_gb(self.interface_dir, kind="global")
        np.testing.assert_array_equal(whole["nu3_even"], self.glob["nu3_even"])
        with self.assertRaises(ValueError):
            ac.load_gb(self.interface_dir, radius=6.0)
        with self.assertRaises(ValueError):
            ac.load_gb(self.interface_dir, kind="nonbulk")
        with self.assertRaises(ValueError):
            ac.load_gb(self.interface_dir, radius=8.0, blocks=["nu4"])

    def test_scaling(self):
        X, _ = ac.load_gb_vectors(self.interface_dir, radius=8.0, planes="edge")
        expected = np.concatenate([self.inter[name][:, 0] / np.linalg.norm(self.bulk[name][0])
                                   for name in ac.BLOCKS], axis=1)
        np.testing.assert_allclose(X, expected, rtol=1e-6)
        raw, _ = ac.load_gb_vectors(self.interface_dir, radius=8.0, planes="edge", blocks=["nu2"], scaling="none")
        np.testing.assert_array_equal(raw, self.inter["nu2"][:, 0])
        bulk, index = ac.load_bulk_vectors(self.global_dir)
        start = 0
        for name, width in self.widths.items():
            # The pristine row has length 1 in every block after scaling.
            self.assertAlmostEqual(np.linalg.norm(bulk[0, start:start + 2 * width]), 1.0)
            start += 2 * width
        self.assertEqual(bulk.shape, (3, start))
        with self.assertRaises(ValueError):
            ac.assemble(self.glob, scaling="unit")

    def test_stored_blocks(self):
        with np.load(self.interface_dir / ac.INTERFACE_FILE) as data:
            self.assertEqual(ac.stored_blocks(data), list(ac.BLOCKS))
        self.assertEqual(ac.run_dir("global", self.interface_dir), self.global_dir)
        self.assertEqual(ac.run_dir("both_interfaces", "x"), ac.STANDARD_ROOT / "both_interfaces" / "x")


if __name__ == "__main__":
    unittest.main()
