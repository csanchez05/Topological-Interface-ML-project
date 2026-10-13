"""Tests of overlap_survey.py on synthetic cells.

  src/dft_plots/.venv/bin/python -B -m unittest discover -s src/dft_plots/coge_interface -p 'test_*.py'
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_survey as ov  # noqa: E402


def poscar(lattice, coords, counts=(1, 1), species="Co Ge"):
    rows = "".join(" ".join(f"{x:.6f}" for x in c) + "\n" for c in coords)
    lat = "".join(" ".join(f"{x:.8f}" for x in v) + "\n" for v in lattice)
    return f"synthetic\n1.0\n{lat}{species}\n{counts[0]} {counts[1]}\nDirect\n{rows}"


BOX = np.diag([20.0, 4.6294, 4.6294])


class Survey(unittest.TestCase):

    def test_close_pair_across_the_periodic_boundary(self):
        # Co at x = 0.01, Ge at x = 0.985 of a 20 A edge: 0.5 A apart through the boundary
        lat, frac, kinds = ov.read_poscar(poscar(BOX, [(0.01, 0.5, 0.5), (0.985, 0.5, 0.5)]))
        r = ov.gb_pairs(lat, frac, kinds, (0.4, 1.0, 2.0))
        self.assertAlmostEqual(r["dmin"], 0.5, places=6)
        self.assertEqual((r["pair"], r["below"]), ("Co-Ge", [0, 1, 1]))

    def test_threshold_beyond_exact_range_is_refused(self):
        lat, frac, kinds = ov.read_poscar(poscar(BOX, [(0.1, 0.5, 0.5), (0.6, 0.5, 0.5)]))
        with self.assertRaisesRegex(ValueError, "half the shortest edge"):
            ov.gb_pairs(lat, frac, kinds, (2.4,))

    def test_non_orthogonal_and_bad_files_are_refused(self):
        skew = BOX.copy()
        skew[1, 0] = 1.0
        lat, frac, kinds = ov.read_poscar(poscar(skew, [(0.1, 0.5, 0.5), (0.6, 0.5, 0.5)]))
        with self.assertRaisesRegex(ov.PoscarError, "non-orthogonal"):
            ov.gb_pairs(lat, frac, kinds, (1.0,))
        for text, message in (("", "empty"), (poscar(BOX, [(0, 0, 0), (0.5, 0, 0)], species="Ge Co"), "species"),
                              (poscar(BOX, [(0, 0, 0)]), "truncated")):
            with self.subTest(message=message), self.assertRaisesRegex(ov.PoscarError, message):
                ov.read_poscar(text)

    def test_bulk_reference_uses_periodic_images(self):
        # Two atoms in a 2 A cube: Co-Ge 1.0 A along x, and each atom 2 A from its own images
        cube = np.diag([2.0, 2.0, 2.0])
        lat, frac, kinds = ov.read_poscar(poscar(cube, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]))
        ref = ov.bulk_reference(lat, frac, kinds)
        self.assertAlmostEqual(ref["Co-Ge"], 1.0)
        self.assertAlmostEqual(ref["Co-Co"], 2.0)
        self.assertAlmostEqual(ref["Ge-Ge"], 2.0)


if __name__ == "__main__":
    unittest.main()
