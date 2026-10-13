"""Symmetry tests of covariant_descriptor.py on a small random Co-Ge cell.

    python -B -m unittest discover -s src/dscribe/CoGe/covariant -p 'test_*.py'
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from ase import Atoms

from check_covariant import fitted_transformation, gram_change, random_rotation, relative
from covariant_descriptor import (BLOCKS, bulk_fields, index_fields, inversion_sign, make_calculator,
                                  per_atom_blocks, split_mean)

SETTINGS = {"r_cut": 4.0, "n_max": 3, "l_max": 2, "sigma": 0.5, "smoothing": 0.5}


def random_cell(seed=0):
    rng = np.random.default_rng(seed)
    # Atoms on a jittered grid, so none sit on top of each other.
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    fractional = (grid + 0.5 + rng.uniform(-0.2, 0.2, size=grid.shape)) / 2.0
    cell = np.diag([4.6, 4.8, 5.0])
    return Atoms(["Co", "Ge"] * 4, scaled_positions=fractional, cell=cell, pbc=True)


class Covariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator, cls.keys = make_calculator(SETTINGS)
        cls.atoms = random_cell()
        cls.reference, cls.properties = per_atom_blocks(cls.atoms, cls.calculator, cls.keys)

    def test_shapes(self):
        n = len(self.atoms)
        self.assertEqual(list(self.reference), list(BLOCKS))
        self.assertEqual(self.reference["invariant"].shape[:2], (n, 1))
        for name in ("polar", "axial"):
            self.assertEqual(self.reference[name].shape[:2], (n, 3))
            self.assertGreater(np.abs(self.reference[name]).max(), 1e-6)
        for name in ("tensor", "pseudotensor"):
            self.assertEqual(self.reference[name].shape[:2], (n, 5))
            self.assertGreater(np.abs(self.reference[name]).max(), 1e-6)

    def test_chosen_blocks_only(self):
        calculator, keys = make_calculator(SETTINGS, ["axial", "tensor"])
        blocks, _ = per_atom_blocks(self.atoms, calculator, keys)
        self.assertEqual(list(blocks), ["axial", "tensor"])
        # The same numbers as when every block is computed.
        for name in blocks:
            np.testing.assert_array_equal(blocks[name], self.reference[name])
        self.assertEqual(index_fields(["axial", "tensor"])[-3:], ["atom_norm_axial", "atom_norm_tensor", "source"])
        self.assertIn("atom_norm_pseudotensor", bulk_fields(BLOCKS))

    def test_rotation(self):
        rotation = random_rotation(3)
        turned = self.atoms.copy()
        turned.set_cell(self.atoms.cell.array @ rotation.T, scale_atoms=False)
        turned.positions = self.atoms.positions @ rotation.T
        blocks, _ = per_atom_blocks(turned, self.calculator, self.keys)
        self.assertLess(relative(blocks["invariant"], self.reference["invariant"]), 1e-8)
        for name in ("polar", "axial"):
            expected = np.einsum("ij,ajp->aip", rotation, self.reference[name])
            self.assertLess(relative(blocks[name], expected), 1e-8, name)
            # A wrong component order would not pass: the vectors do change.
            self.assertGreater(relative(blocks[name], self.reference[name]), 1e-3, name)
        for name in ("tensor", "pseudotensor"):
            D, residual, orthogonality = fitted_transformation(self.reference[name], blocks[name])
            self.assertLess(gram_change(self.reference[name], blocks[name]), 1e-8, name)
            self.assertLess(residual, 1e-8, name)
            self.assertLess(orthogonality, 1e-8, name)
            self.assertGreater(relative(blocks[name], self.reference[name]), 1e-3, name)
            # The same D for both lambda = 2 blocks: it depends only on the rotation.
            if name == "tensor":
                D_tensor = D
        D_pseudo, _, _ = fitted_transformation(self.reference["pseudotensor"], blocks["pseudotensor"])
        np.testing.assert_allclose(D_pseudo, D_tensor, atol=1e-8)

    def test_gram_change_equals_the_direct_product(self):
        rng = np.random.default_rng(3)
        for n_prop in (3, 40):  # fewer and more properties than 2 x 5 components
            before = rng.normal(size=(6, 5, n_prop))
            after = before + 0.1 * rng.normal(size=before.shape)
            direct = relative(np.einsum("acp,acq->apq", after, after), np.einsum("acp,acq->apq", before, before))
            self.assertAlmostEqual(gram_change(before, after), direct, places=12)
            Q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
            self.assertLess(gram_change(before, np.einsum("ij,ajp->aip", Q, before)), 1e-13)

    def test_fitted_transformation_finds_a_known_matrix(self):
        rng = np.random.default_rng(1)
        D, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        before = rng.normal(size=(6, 5, 4))
        after = np.einsum("ij,ajp->aip", D, before)
        fitted, residual, orthogonality = fitted_transformation(before, after)
        np.testing.assert_allclose(fitted, D, atol=1e-10)
        self.assertLess(residual, 1e-10)
        self.assertLess(orthogonality, 1e-10)
        # A different matrix for each atom is not one transformation.
        after[0] = np.einsum("ij,jp->ip", np.linalg.qr(rng.normal(size=(5, 5)))[0], before[0])
        self.assertGreater(fitted_transformation(before, after)[1], 1e-3)

    def test_inversion(self):
        inverted = self.atoms.copy()
        inverted.positions = -self.atoms.positions
        inverted.wrap()
        blocks, _ = per_atom_blocks(inverted, self.calculator, self.keys)
        signs = {"invariant": 1.0, "polar": -1.0, "axial": 1.0, "tensor": 1.0, "pseudotensor": -1.0}
        for name in BLOCKS:
            self.assertEqual(inversion_sign(name), signs[name])
            self.assertLess(relative(blocks[name], signs[name] * self.reference[name]), 1e-8, name)

    def test_split_mean(self):
        is_co = np.array([s == "Co" for s in self.atoms.get_chemical_symbols()])
        vector, counts = split_mean(self.reference["polar"], is_co)
        P = self.reference["polar"].shape[-1]
        self.assertEqual(vector.shape, (3, 2 * P))
        np.testing.assert_allclose(vector[:, :P], self.reference["polar"][is_co].mean(axis=0))
        np.testing.assert_allclose(vector[:, P:], self.reference["polar"][~is_co].mean(axis=0))


if __name__ == "__main__":
    unittest.main()
