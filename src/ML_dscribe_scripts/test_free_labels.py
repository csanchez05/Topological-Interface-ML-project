"""Synthetic tests for the grain-class follow-ups: NO_BOUNDARY, per-plane scores, mirror classes,
regression KRR (krr_classify.py, selection_benchmark.py) and geometry_labels.py.

    python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_*.py'
"""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import geometry_labels as gl
import krr_classify
import ml_common as mc
import selection_benchmark as sb


def cubic_group():
    group = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            group.append(matrix)
    return group


def information_text(rotations):
    """An Information.txt with one interface built by every given rotation (no shift)."""
    lines = [f"Interface 0 : Num of Syms = {len(rotations)}", "x distance = 0.1", "total distance = 1.0", "-----"]
    for k, R in enumerate(rotations):
        lines.append(f"op{k} : x, y, z [[0 0 0]]")
        for row in R:
            lines.append("[" + " ".join(f"{v:g}" for v in row) + " 0]")
        lines.append("")
    return "\n".join(lines) + "\n"


def cscl(n_cells, a=1.0):
    """A CsCl-type Co/Ge crystal, n_cells along x (the interface normal), one cell along y and z."""
    from ase import Atoms

    positions, symbols = [], []
    for k in range(n_cells):
        positions += [(k * a, 0, 0), ((k + 0.5) * a, 0.5 * a, 0.5 * a)]
        symbols += ["Co", "Ge"]
    return Atoms(symbols, positions=positions, cell=np.diag([n_cells * a, a, a]), pbc=True)


class NoBoundary(unittest.TestCase):
    def test_drop_no_boundary(self):
        table = pd.DataFrame({"family": ["111/sym00118", "111/sym00119", "100/sym00001", "111/sym00175"]})
        X = np.arange(8.0).reshape(4, 2)
        X2, table2 = mc.drop_no_boundary(X, table)
        self.assertEqual(table2["family"].tolist(), ["111/sym00118", "100/sym00001"])
        self.assertEqual(table2.index.tolist(), [0, 1])
        np.testing.assert_array_equal(X2, X[[0, 2]])


class PerPlane(unittest.TestCase):
    def test_plane_scores(self):
        true = np.array(["E", "E", "I", "I", "E", "T"])
        pred = np.array(["E", "I", "I", "T", "E", "E"])
        plane = np.array(["100", "100", "100", "100", "111", "111"])
        possible = {"100": {"E", "I", "R", "IR"}, "111": {"E", "I", "R", "IR", "T", "IT", "S", "IS"}}
        mirrors = {"100": {"E": {"I", "IR"}, "I": {"E", "R"}}, "111": {"T": {"E", "S"}, "E": {"I", "IT"}}}
        rows = {r["plane"]: r for r in krr_classify.plane_scores(true, pred, plane, possible, mirrors)}
        self.assertEqual(rows["100"]["n"], 4)
        self.assertEqual(rows["100"]["chance"], 0.25)
        self.assertEqual(rows["111"]["chance"], 0.125)
        self.assertEqual(rows["100"]["accuracy"], 0.5)
        self.assertEqual(rows["100"]["impossible_share"], 0.25)       # T cannot occur on 100
        self.assertEqual(rows["100"]["mirror_share_of_errors"], 0.5)  # E->I is a mirror, I->T is not
        self.assertEqual(rows["100"]["correct_or_mirror"], 0.75)
        self.assertAlmostEqual(rows["100"]["mirror_share_if_random"], round(2 / 3, 4))
        self.assertEqual(rows["111"]["mirror_share_of_errors"], 1.0)  # T->E
        self.assertEqual(rows["111"]["balanced_accuracy"], 0.5)

    def test_mirror_classes_of_the_cube_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "CoGe_100_GBs" / "Interface" / "Information.txt"
            path.parent.mkdir(parents=True)
            path.write_text(information_text(cubic_group()))
            mirrors = krr_classify.mirror_classes(folder, planes=("100",))["100"]
        self.assertEqual(sorted(mirrors), ["E", "I", "I.R90[100]", "R90[100]"])
        for grain, partners in mirrors.items():
            self.assertEqual(len(partners), 2, grain)
            self.assertNotIn(grain, partners)
        self.assertEqual(mirrors["E"], {"I", "I.R90[100]"})


class Regression(unittest.TestCase):
    def data(self, n=80, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        return X, {"linear": X @ np.array([1.0, -2.0, 0.5, 0.0]), "noise": rng.normal(size=n)}

    def test_regression_cv_learns_a_smooth_target_only(self):
        X, targets = self.data()
        values, fold, chosen = krr_classify.nested_cv(X, targets, np.arange(len(X)), 4, 3, [0.1, 1.0, 10.0],
                                                      [1e-4, 1e-2, 1.0], ["none"], 0,
                                                      fold_function=krr_classify.regression_fold)
        self.assertTrue(np.isfinite(values["krr"]["linear"]).all())
        r2, _ = krr_classify.regression_scores(targets["linear"], values["krr"]["linear"])
        self.assertGreater(r2, 0.9)
        r2, _ = krr_classify.regression_scores(targets["noise"], values["krr"]["noise"])
        self.assertLess(r2, 0.2)
        r2, _ = krr_classify.regression_scores(targets["linear"], values["baseline"]["linear"])
        self.assertLessEqual(r2, 0.0)
        self.assertIn("inner_r2", chosen[0]["krr"]["linear"])

    def test_regression_scores(self):
        self.assertEqual(krr_classify.regression_scores([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), (1.0, 0.0))
        self.assertEqual(krr_classify.regression_scores([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])[0], 0.0)

    def test_leave_one_out_shortcut_is_close_to_refitting(self):
        X, targets = self.data(n=30, seed=3)
        y = targets["linear"] + 0.1 * np.random.default_rng(4).normal(size=30)
        D2 = krr_classify.squared_distances(X)
        K = krr_classify.kernel_matrix(D2, 1.0, float(np.median(D2)))
        alphas = [1e-3, 1e-1]
        shortcut = sb.loo_errors(K, y, alphas)[:, 0]
        for alpha, fast in zip(alphas, shortcut):
            errors = []
            for i in range(len(y)):
                rest = np.delete(np.arange(len(y)), i)
                guess = krr_classify.krr_scores(K[np.ix_(rest, rest)], y[rest][:, None], K[[i]][:, rest], alpha)
                errors.append((guess[0, 0] - y[i]) ** 2)
            self.assertLess(abs(fast - np.mean(errors)) / np.mean(errors), 0.1)

    def test_benchmark_rows_for_planes_and_geometry(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(90, 3))
        plane = np.array(["100", "110", "111"] * 30)
        grain = np.where(X[:, 0] > 0, "E", "I")
        table = pd.DataFrame({"grain_class": grain, "plane": plane, "handedness": grain,
                              "d_min": X[:, 1] * 2.0 + 1.0, "log1p_n_close": X[:, 2], "cn_excess": X[:, 1] - X[:, 2],
                              "cn_defect": np.abs(X[:, 0])})
        codes = sb.label_codes(table)
        pool, test = np.arange(0, 90, 2), np.arange(1, 90, 2)
        spaces = sb.Spaces(X, pool)
        rows = sb.score_picks(spaces, codes, pool, test)
        labels = {(r["label"], r["model"]) for r in rows}
        for name in ("grain_class@100", "grain_class@110", "grain_class@111", *gl.TARGETS):
            self.assertIn((name, "krr"), labels)
        r2 = {r["label"]: r["r2"] for r in rows if r["model"] == "krr" and "r2" in r}
        self.assertGreater(r2["d_min"], 0.9)
        self.assertGreater(r2["cn_excess"], 0.9)


class Geometry(unittest.TestCase):
    def test_pristine_shells_of_cscl(self):
        shells, cutoff, neighbours = gl.pristine_shells(cscl(1))
        self.assertAlmostEqual(shells[0]["distance"], round(np.sqrt(3) / 2, 4))
        self.assertEqual(shells[0]["pairs"], ["Co-Ge", "Ge-Co"])
        self.assertAlmostEqual(cutoff, 0.5 * (1.0 + np.sqrt(2)), places=4)  # widest gap: 1.0 to 1.414
        self.assertEqual(neighbours, {"Co": 14, "Ge": 14})

    def test_perfect_crystal_gives_bulk_values(self):
        _, cutoff, neighbours = gl.pristine_shells(cscl(1))
        labels = gl.structure_labels(cscl(8), 2.0, 0.5, cutoff, neighbours)
        for region in gl.REGIONS:
            self.assertAlmostEqual(labels[region]["d_min"], np.sqrt(3) / 2)
            self.assertEqual(labels[region]["n_close_pairs"], 0.0)
            self.assertEqual(labels[region]["cn_defect"], 0.0)
        self.assertIsNone(gl.bulk_cell_check(cscl(3), 0.5, cutoff, neighbours, round(np.sqrt(3) / 2, 4)))

    def perturbed(self, n_cells):
        atoms = cscl(n_cells)
        positions = atoms.get_positions()
        positions[1] = (0.2, 0.0, 0.0)  # a Ge atom 0.2 A from the Co atom at the edge plane
        atoms.set_positions(positions)
        return atoms

    def test_a_close_pair_at_the_edge(self):
        _, cutoff, neighbours = gl.pristine_shells(cscl(1))
        labels = gl.structure_labels(self.perturbed(8), 2.0, 0.5, cutoff, neighbours)
        area_nm2 = 1.0 / 100.0
        self.assertEqual(labels["edge"]["n_close_pairs"], 1.0)
        self.assertAlmostEqual(labels["edge"]["n_close"], 1.0 / area_nm2)
        self.assertAlmostEqual(labels["edge"]["d_min"], 0.2)
        self.assertEqual(labels["mid"]["n_close_pairs"], 0.0)
        self.assertAlmostEqual(labels["cell"]["n_close"], 0.5 / area_nm2)
        self.assertAlmostEqual(labels["cell"]["d_min"], 0.2)
        self.assertGreater(labels["edge"]["cn_defect"], 0.0)
        self.assertEqual(labels["mid"]["cn_defect"], 0.0)
        self.assertIsNotNone(gl.bulk_cell_check(self.perturbed(3), 0.5, cutoff, neighbours, round(np.sqrt(3) / 2, 4)))

    def test_labels_do_not_depend_on_thickness(self):
        _, cutoff, neighbours = gl.pristine_shells(cscl(1))
        thin = gl.structure_labels(self.perturbed(8), 2.0, 0.5, cutoff, neighbours)
        thick = gl.structure_labels(self.perturbed(12), 2.0, 0.5, cutoff, neighbours)
        for region in gl.REGIONS:
            for label in gl.LABELS:
                self.assertAlmostEqual(thin[region][label], thick[region][label], places=10)

    def test_variance_shares(self):
        groups = np.repeat(["a", "b"], 10)
        self.assertAlmostEqual(gl.variance_shares(np.where(groups == "a", 1.0, 3.0), groups), 1.0)
        values = np.tile(np.arange(10.0), 2)
        self.assertAlmostEqual(gl.variance_shares(values, groups), 0.0)
        self.assertIsNone(gl.variance_shares(np.ones(4), np.array(["a", "a", "b", "b"])))

    def test_load_geometry_matches_rows(self):
        table = pd.DataFrame([{"family": f, "plane": "100", "sym": "x", "thickness": t, "region": r,
                               **{label: k for label in gl.LABELS}}
                              for k, (f, t, r) in enumerate(itertools.product(["100/a", "100/b"], [4, 8],
                                                                              gl.REGIONS))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "g.csv"
            table.to_csv(path, index=False)
            index = pd.DataFrame({"family": ["100/b", "100/a"], "thickness": [8, 4]})
            rows = gl.load_geometry(index, "mid", path)
            self.assertEqual(rows["region"].tolist(), ["mid", "mid"])
            self.assertEqual(rows["d_min"].tolist(), [10, 1])
            rows = gl.load_geometry(index.assign(region=["edge", "cell"]), "both", path)
            self.assertEqual(rows["d_min"].tolist(), [9, 2])
            with self.assertRaises(ValueError):
                gl.load_geometry(pd.DataFrame({"family": ["100/c"], "thickness": [8]}), "both", path)


if __name__ == "__main__":
    unittest.main()
