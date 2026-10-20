"""Synthetic tests for covariant_ace and covariant_descriptor. Run from the project root with the mlenv environment:

    python -B -m unittest discover -s src/ACE/CoGe/covariant -p 'test_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from ase import Atoms

import covariant_ace as cov
import covariant_descriptor as cd

ac, sc = cov.ac, cov.sc
# Small expansions keep the tests fast; l3 = 2 already gives every lambda <= 2 block.
SETTINGS = {"r_cut": 4.0, "n_max": 3, "l_max": 2, "sigma": 0.5, "smoothing": 0.5, "density": "gaussian",
            "nu_max": 3, "n3": 2, "l3": 2}
NAMES = ["nu1_polar", "nu1_tensor", "nu2_polar", "nu2_axial", "nu2_tensor", "nu2_pseudotensor",
         "nu3_polar", "nu3_axial", "nu3_tensor", "nu3_pseudotensor"]


def random_cell(seed=0):
    rng = np.random.default_rng(seed)
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    fractional = (grid + 0.5 + rng.uniform(-0.2, 0.2, size=grid.shape)) / 2.0
    return Atoms(["Co", "Ge"] * 4, scaled_positions=fractional, cell=np.diag([4.6, 4.8, 5.0]), pbc=True)


def b20_cell(u_co=0.135, u_ge=0.844, a=4.63):
    """Cubic B20 CoGe (space group P2_13): 4 Co and 4 Ge on 4a sites."""
    def sites(u):
        return [[u, u, u], [0.5 + u, 0.5 - u, -u], [-u, 0.5 + u, 0.5 - u], [0.5 - u, -u, 0.5 + u]]
    return Atoms(["Co"] * 4 + ["Ge"] * 4, scaled_positions=np.mod(sites(u_co) + sites(u_ge), 1.0),
                 cell=np.eye(3) * a, pbc=True)


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def turned(atoms, rotation):
    out = atoms.copy()
    out.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    out.positions = atoms.positions @ rotation.T
    return out


def relative(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


class Blocks(unittest.TestCase):
    def test_names_signs_and_components(self):
        self.assertEqual(list(cov.BLOCKS), NAMES)
        self.assertEqual(cov.blocks_for(SETTINGS), NAMES)
        self.assertEqual(cov.blocks_for({**SETTINGS, "nu_max": 2}), NAMES[:6])
        signs = {name: cov.inversion_sign(name) for name in NAMES}
        self.assertEqual(signs["nu2_polar"], -1.0)
        self.assertEqual(signs["nu2_axial"], 1.0)
        self.assertEqual(signs["nu3_tensor"], 1.0)
        self.assertEqual(signs["nu3_pseudotensor"], -1.0)
        self.assertEqual([cov.components(name) for name in ("nu1_polar", "nu1_tensor")], [3, 5])
        self.assertEqual(cov.run_name({**SETTINGS, "r_cut": 6.0, "n_max": 8, "l_max": 6, "n3": 4, "l3": 4}),
                         "rc6.0_n8_l6_s0.5_nu3-n4-l4_lam1-2")


class Symmetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator = cov.CovariantCalculator(SETTINGS)
        cls.atoms = random_cell()
        cls.reference = cls.calculator.per_atom(cls.atoms)

    def test_shapes(self):
        self.assertEqual(list(self.reference), NAMES)
        widths = self.calculator.widths()
        self.assertEqual(widths["nu1_polar"], 2 * SETTINGS["n_max"])   # neighbour species x n
        for name, values in self.reference.items():
            self.assertEqual(values.shape, (len(self.atoms), cov.components(name), widths[name]))
            self.assertGreater(np.abs(values).max(), 1e-6, name)

    def test_rotation(self):
        """lambda = 1 blocks turn as x, y, z vectors; lambda = 2 blocks by one orthogonal 5 x 5 matrix."""
        rotation = random_rotation(3)
        blocks = self.calculator.per_atom(turned(self.atoms, rotation))
        for name in NAMES:
            before, after = self.reference[name], blocks[name]
            if cov.BLOCKS[name][1] == 1:
                expected = np.einsum("ij,njp->nip", rotation, before)
                self.assertLess(relative(after, expected), 1e-8, name)
                continue
            # Fit after = D before over every atom and feature; D must be orthogonal and exact.
            flat_before = before.transpose(1, 0, 2).reshape(5, -1)
            flat_after = after.transpose(1, 0, 2).reshape(5, -1)
            D = np.linalg.lstsq(flat_before.T, flat_after.T, rcond=None)[0].T
            self.assertLess(relative(D @ flat_before, flat_after), 1e-8, name)
            np.testing.assert_allclose(D @ D.T, np.eye(5), atol=1e-8, err_msg=name)

    def test_inversion(self):
        inverted = self.atoms.copy()
        inverted.positions = -self.atoms.positions
        inverted.wrap()
        blocks = self.calculator.per_atom(inverted)
        for name in NAMES:
            sign = cov.inversion_sign(name)
            self.assertLess(relative(blocks[name], sign * self.reference[name]), 1e-8, name)
            self.assertGreater(relative(blocks[name], -sign * self.reference[name]), 1.0, name)

    def test_chunks_give_the_same_numbers(self):
        chunked = cov.CovariantCalculator(SETTINGS, chunk=3).per_atom(self.atoms)
        for name in NAMES:
            self.assertLess(relative(chunked[name], self.reference[name]), 1e-12, name)

    def test_nu2_is_the_covariant_soap(self):
        """nu2_* equal featomic's lambda-SOAP blocks of the same expansion (same Gram matrix)."""
        import featomic
        from featomic.clebsch_gordan import EquivariantPowerSpectrum
        from metatensor import Labels

        hypers = ac.hypers(SETTINGS, SETTINGS["n_max"], SETTINGS["l_max"])
        power = EquivariantPowerSpectrum(featomic.SphericalExpansion(**hypers), neighbor_types=[27, 32])
        kinds = [name.split("_", 1)[1] for name in NAMES if name.startswith("nu2_")]
        keys = Labels(["o3_lambda", "o3_sigma"], np.array([cov.KINDS[k] for k in kinds], dtype=np.int32))
        tensor = power.compute(self.atoms, selected_keys=keys, neighbors_to_properties=True)
        for kind in kinds:
            lam, sig = cov.KINDS[kind]
            values = np.zeros_like(self.reference[f"nu2_{kind}"])
            for center in (27, 32):
                block = tensor.block(o3_lambda=lam, o3_sigma=sig, center_type=center)
                values[np.asarray(block.samples.column("atom"))] = block.values[:, cov.TO_XYZ] if lam == 1 else block.values
            mine = self.reference[f"nu2_{kind}"]
            self.assertEqual(values.shape, mine.shape)
            A, B = values.reshape(-1, values.shape[-1]), mine.reshape(-1, mine.shape[-1])
            self.assertLess(relative(B @ B.T, A @ A.T), 1e-10, kind)

    def test_averages(self):
        rng = np.random.default_rng(1)
        is_co = np.array(self.atoms.get_chemical_symbols()) == "Co"
        masks = np.stack([np.ones(8, bool), rng.random(8) < 0.5, is_co])
        vectors, counts, norms = cov.averages(self.calculator, self.atoms, masks)
        for name in NAMES:
            values = self.reference[name]
            for s, mask in enumerate(masks):
                for half, species in enumerate((is_co, ~is_co)):
                    selected = mask & species
                    self.assertEqual(counts[s, half], selected.sum())
                    part = vectors[name][s][:, half * values.shape[2]:(half + 1) * values.shape[2]]
                    if selected.any():
                        np.testing.assert_allclose(part, values[selected].mean(axis=0), rtol=1e-12, atol=1e-15)
                    else:
                        self.assertTrue(np.isnan(part).all())
            self.assertAlmostEqual(norms[name], np.linalg.norm(values.reshape(8, -1), axis=1).mean(), places=12)

    def test_bulk_averages_vanish(self):
        """Point group 23 allows no vector and no rank-2 tensor: every species mean is zero.

        Each atom sits on a 3-fold axis, so its l <= 2 densities are all parallel and their
        antisymmetric (axial) products vanish atom by atom; l_max 4 and l3 3 give every block a
        non-zero single-atom value, which the cell average must cancel.
        """
        calculator = cov.CovariantCalculator({**SETTINGS, "l_max": 4, "l3": 3})
        atoms = b20_cell()
        per_atom = calculator.per_atom(atoms)
        vectors, _, norms = cov.averages(calculator, atoms, np.ones((1, 8), bool))
        for name in NAMES:
            self.assertGreater(norms[name], 1e-6, name)        # single atoms do carry the block
            self.assertLess(np.abs(vectors[name]).max() / norms[name], 1e-10, name)
            self.assertGreater(np.abs(per_atom[name]).max(), 1e-6, name)


class RoundTrip(unittest.TestCase):
    """write as covariant_descriptor.py does, read back with load_rows() and bulk_averages()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "covariant"
        self.global_dir, self.interface_dir = root / "global" / "run", root / "both_interfaces" / "run"
        rng = np.random.default_rng(0)
        self.blocks = ["nu1_polar", "nu2_tensor"]
        widths = {"nu1_polar": 2, "nu2_tensor": 3}
        self.inter = {b: rng.normal(size=(3, 2, cov.components(b), 2 * w)).astype(np.float32) for b, w in widths.items()}
        self.glob = {b: rng.normal(size=(3, cov.components(b), 2 * w)).astype(np.float32) for b, w in widths.items()}
        self.n = rng.integers(1, 9, size=(3, 2, 2)).astype(np.int32)
        sc.write_npz(self.interface_dir / cov.FILE, **{f"X_{b}_r8": v for b, v in self.inter.items()}, n_r8=self.n,
                     radii=np.array([8.0]))
        sc.write_npz(self.global_dir / cov.FILE, **{f"X_{b}": v for b, v in self.glob.items()})
        sc.write_npz(self.global_dir / cov.BULK_FILE, **{f"X_{b}": v[:2].astype(float) for b, v in self.glob.items()})
        families = [("100/sym00001", 8), ("110/sym00002", 8), ("100/sym00001", 4)]
        rows = [{"index": i, "plane": f.split("/")[0], "family": f, "thickness": t, "n_Co": 10 + i, "n_Ge": 20 + i}
                for i, (f, t) in enumerate(families)]
        fields = ["index", "plane", "family", "thickness", "n_Co", "n_Ge"]
        sc.write_csv(self.interface_dir / "index.csv", rows, fields)
        sc.write_csv(self.global_dir / "index.csv", rows[::-1], fields)     # another order on purpose
        self.norms = {"nu1_polar": 2.0, "nu2_tensor": 4.0}
        sc.write_csv(self.global_dir / "bulk_index.csv",
                     [{"index": 0, "kind": "pristine", **{f"atom_norm_{b}": v for b, v in self.norms.items()}},
                      {"index": 1, "kind": "reference", **{f"atom_norm_{b}": 1.0 for b in self.blocks}}],
                     ["index", "kind"] + [f"atom_norm_{b}" for b in self.blocks])

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_rows(self):
        keys = [("100/sym00001", 4), ("110/sym00002", 8)]
        out = cov.load_rows(self.interface_dir, keys)
        self.assertEqual(list(out["columns"]), self.blocks)
        expected = np.concatenate([self.inter[b][[2, 1]].reshape(2, 2, -1) / self.norms[b] for b in self.blocks], axis=-1)
        np.testing.assert_allclose(out["interface"], expected, rtol=1e-6)
        # the global index is in reverse order: row 2 of the interface index is row 0 there
        cell = np.concatenate([self.glob[b][[0, 1]].reshape(2, -1) / self.norms[b] for b in self.blocks], axis=-1)
        np.testing.assert_allclose(out["cell"], cell, rtol=1e-6)
        np.testing.assert_array_equal(out["n_interface"], self.n[[2, 1]])
        np.testing.assert_array_equal(out["n_species"], [[12, 22], [11, 21]])
        # each component keeps its [Co | Ge] halves: True for the first half of every component
        self.assertEqual(out["is_co"][:8].tolist(), [True, True, False, False] * 2)
        self.assertEqual(out["columns"]["nu2_tensor"], (12, 12 + 30))
        raw = cov.load_rows(self.interface_dir, keys, blocks=["nu2_tensor"], scaling="none")
        np.testing.assert_allclose(raw["interface"], self.inter["nu2_tensor"][[2, 1]].reshape(2, 2, -1), rtol=1e-6)
        with self.assertRaises(KeyError):
            cov.load_rows(self.interface_dir, [("111/sym00003", 8)])
        with self.assertRaises(ValueError):
            cov.load_rows(self.interface_dir, keys, scaling="unit")

    def test_bulk_averages(self):
        arrays, index = cov.bulk_averages(self.global_dir)
        self.assertEqual(list(arrays), self.blocks)
        self.assertEqual(arrays["nu2_tensor"].shape, (2, 5, 6))
        self.assertEqual(index["kind"].tolist(), ["pristine", "reference"])


class Driver(unittest.TestCase):
    def test_structure_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.csv"
            sc.write_csv(path, [{"family": "100/sym00001", "thickness": 8}, {"family": "110/sym00002", "thickness": 4}],
                         ["family", "thickness"])
            wanted = cd.read_structure_list(path)
            bad = Path(tmp) / "bad.csv"
            sc.write_csv(bad, [{"sym": "x"}], ["sym"])
            with self.assertRaises(ValueError):
                cd.read_structure_list(bad)
        self.assertEqual(wanted, {("100/sym00001", 8), ("110/sym00002", 4)})
        jobs = [("100", "sym00001", 8, Path("a")), ("100", "sym00001", 4, Path("b")), ("111", "sym00009", 8, Path("c"))]
        kept, missing = cd.select_jobs(jobs, wanted)
        self.assertEqual(kept, jobs[:1])
        self.assertEqual(missing, [("110/sym00002", 4)])

    def test_block_arrays(self):
        text = cd.block_arrays(["nu1_polar", "nu3_pseudotensor"], radii=[8.0])
        self.assertIn("X_nu1_polar_r<R>", text)
        self.assertTrue(text["X_nu3_pseudotensor_r<R>"].startswith("(N, plane, 5, 2P) body order 4, lambda 2, sigma -1"))
        self.assertEqual(text["radii"], [8.0])


if __name__ == "__main__":
    unittest.main()
