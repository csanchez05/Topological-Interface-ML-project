"""Unit tests of covariant_vs_labels.py on synthetic data.

  src/mlenv/.venv/bin/python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_covariant_vs_labels.py'
"""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import covariant_vs_labels as cl  # noqa: E402
import krr_classify as kc  # noqa: E402

GAMMAS, ALPHAS = [0.1, 1.0, 10.0], [1e-4, 1e-2, 1.0]


def label_table(n, seed=0):
    rng = np.random.default_rng(seed)
    classes = ["E", "I", "R90[100]", "I.R90[100]"]
    grain = rng.choice(classes, n)
    return pd.DataFrame({"plane": rng.choice(["100", "110", "111"], n), "grain_class": grain,
                         "handedness": np.where(np.char.startswith(grain.astype(str), "I"), "opposite", "same"),
                         "lattice": "shared", "motif": np.where(np.char.find(grain.astype(str), "R90") >= 0, "turned", "aligned"),
                         "inversion": rng.integers(0, 2, n), "mirror_parallel": rng.integers(0, 2, n),
                         "twofold_normal": rng.integers(0, 2, n), "twofold_inplane": rng.integers(0, 2, n),
                         "space_group": rng.choice([1, 4, 5, 14], n)})


class DistanceTests(unittest.TestCase):
    def test_total_variance_is_the_summed_column_variance(self):
        X = np.random.default_rng(1).normal(size=(40, 7))
        train = np.arange(0, 40, 2)
        D2 = kc.squared_distances(X)
        self.assertAlmostEqual(cl.total_variance(D2, train), X[train].var(axis=0).sum(), places=10)

    def test_block_distances(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 5))
        X[:, 4] = 1e-9 * rng.normal(size=30)  # float noise in a component that should vanish
        train = np.arange(20)
        D2, floored = cl.block_distances(X, train, "none", 1e-5)
        np.testing.assert_allclose(D2, kc.squared_distances(X), atol=1e-10)
        self.assertEqual(floored, 0)
        D2, floored = cl.block_distances(X, train, "standard", 1e-5)
        Z = (X - X[train].mean(axis=0)) / np.r_[X[train, :4].std(axis=0), 1.0]
        np.testing.assert_allclose(D2, kc.squared_distances(Z), atol=1e-10)
        self.assertEqual(floored, 1)
        with self.assertRaises(ValueError):
            cl.block_distances(X, train, "minmax", 1e-5)

    def test_cell_kernel_is_swap_invariant_and_psd(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(2 * 25, 6))  # rows 2f, 2f + 1: edge, mid of family f
        swapped = X.copy()
        for f in (0, 7, 11):
            swapped[[2 * f, 2 * f + 1]] = swapped[[2 * f + 1, 2 * f]]
        train = np.arange(20)
        kernels = []
        for data in (X, swapped):
            parts = cl.matchings(kc.squared_distances(data), "cell")
            kernels.append(cl.kernel(parts, 1.0, cl.pooled_width(parts, train)))
        np.testing.assert_allclose(kernels[0], kernels[1], atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(kernels[0]).min(), -1e-10)
        self.assertEqual(len(cl.matchings(np.eye(3), "interface")), 1)

    def test_coinciding_pairs(self):
        X = np.random.default_rng(4).normal(size=(6, 4))
        X[3] = X[0]
        X[5] = X[4] * (1 + 1e-6)
        groups = np.array(["a", "b", "c", "d", "e", "e"])
        pairs = cl.coinciding_pairs(X, groups)
        self.assertTrue(pairs[0, 3] and pairs[3, 0])
        self.assertFalse(pairs[4, 5])  # same group: not a leak
        self.assertEqual(int(pairs.sum()), 2)


class LabelTests(unittest.TestCase):
    def test_encoding(self):
        table = label_table(50)
        L, names = cl.label_vectors(table, "labels")
        self.assertEqual(L.shape, (50, len(names)))
        self.assertIn("twofold_normal", names)
        D2 = kc.squared_distances(L)
        a, b = table.index[(table["plane"] == "100")][0], table.index[(table["plane"] == "110")][0]
        only_plane = L[a].copy()
        only_plane[: 3] = L[b, : 3]
        self.assertAlmostEqual(float(np.square(L[a] - only_plane).sum()), 1.0)  # one categorical change = 1
        self.assertTrue(np.all(D2 >= 0))
        P, names = cl.label_vectors(table, "plane+class")
        self.assertEqual(len(names), 3 + 4)
        with self.assertRaises(ValueError):
            cl.label_vectors(table.assign(inversion=2), "labels")
        with self.assertRaises(ValueError):
            cl.label_vectors(table, "everything")

    def test_usable_fold(self):
        reference = np.arange(100.0)
        self.assertTrue(cl.usable_fold(np.arange(10.0), reference))
        self.assertFalse(cl.usable_fold(np.ones(10), reference))
        self.assertFalse(cl.usable_fold(np.arange(3.0), reference))

    def test_group_share(self):
        y = np.array([1.0, 1.0, 3.0, 3.0])
        self.assertEqual(cl.group_share(y, np.array(["a", "a", "b", "b"])), 1.0)
        self.assertEqual(cl.group_share(y, np.array(["a", "b", "a", "b"])), 0.0)


class FoldTests(unittest.TestCase):
    def test_folds_keep_groups_and_repeat(self):
        groups = np.repeat(np.arange(40), 2)
        folds = cl.make_folds(groups, 5, 3, 0)
        again = cl.make_folds(groups, 5, 3, 0)
        for (n, train, test, inner), (_, train2, test2, _) in zip(folds, again):
            self.assertFalse(np.intersect1d(groups[train], groups[test]).size)
            np.testing.assert_array_equal(test, test2)
            for fit, check in inner:
                self.assertFalse(np.intersect1d(groups[train][fit], groups[train][check]).size)
        self.assertEqual(cl.fold_digest(folds, len(groups)), cl.fold_digest(again, len(groups)))

    def test_kernel_fold_matches_regression_fold(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(90, 6))
        targets = {"a": np.sin(X[:, 0]) + 0.1 * rng.normal(size=90), "b": X[:, 1] ** 2}
        Y = np.column_stack(list(targets.values()))
        groups = np.repeat(np.arange(45), 2)
        number, train, test, inner = cl.make_folds(groups, 5, 3, 0)[0]
        scales = ["standard", "none"]
        expected, entry = kc.regression_fold(X, targets, number, train, test, inner, GAMMAS, ALPHAS, scales)
        parts = {scale: [kc.scaled_distances(X, train, scale)] for scale in scales}
        predicted, chosen = cl.kernel_fold(parts, None, Y, train, test, inner, GAMMAS, ALPHAS, (0.0,))
        for c, name in enumerate(targets):
            np.testing.assert_allclose(predicted[:, c], expected["krr"][name], rtol=1e-10, atol=1e-12)
            self.assertEqual(chosen[c]["scale"], entry["krr"][name]["scale"])
            self.assertEqual(chosen[c]["alpha"], entry["krr"][name]["alpha"])

    def test_label_weight_follows_the_target(self):
        rng = np.random.default_rng(6)
        n = 160
        X = rng.normal(size=(n, 4))
        L = rng.integers(0, 2, size=(n, 3)).astype(float)
        Y = np.column_stack([L[:, 0] + 2 * L[:, 1] + 0.01 * rng.normal(size=n), np.sin(2 * X[:, 0])])
        groups = np.arange(n)
        number, train, test, inner = cl.make_folds(groups, 5, 3, 0)[0]
        cov = kc.squared_distances(X)
        cov /= cl.total_variance(cov, train)
        lab = kc.squared_distances(L)
        lab /= cl.total_variance(lab, train)
        _, chosen = cl.kernel_fold({"none": [cov]}, [lab], Y, train, test, inner, GAMMAS, ALPHAS, cl.LABEL_WEIGHTS)
        self.assertGreaterEqual(chosen[0]["label_weight"], 1.0)
        self.assertLessEqual(chosen[1]["label_weight"], 0.1)

    def test_plane_class_r2_stays_below_the_in_sample_share(self):
        table = label_table(240, seed=7)
        rng = np.random.default_rng(7)
        effect = {c: v for c, v in zip(["E", "I", "R90[100]", "I.R90[100]"], [0.0, 0.5, 1.0, -0.4])}
        y = table["grain_class"].map(effect).to_numpy() + rng.normal(size=240)
        P, _ = cl.label_vectors(table, "plane+class")
        D2 = kc.squared_distances(P)
        groups = np.arange(240)
        predicted = np.full(240, np.nan)
        for number, train, test, inner in cl.make_folds(groups, 5, 3, 0):
            parts = {"none": [D2 / cl.total_variance(D2, train)]}
            predicted[test] = cl.kernel_fold(parts, None, y[:, None], train, test, inner, GAMMAS, ALPHAS, (0.0,))[0][:, 0]
        r2 = kc.regression_scores(y, predicted)[0]
        share = cl.group_share(y, (table["plane"] + "|" + table["grain_class"]).to_numpy())
        self.assertLessEqual(r2, share + cl.PLANE_CLASS_TOL)

    def test_run_cv_end_to_end(self):
        rng = np.random.default_rng(8)
        families = 30
        rows = label_table(families, seed=8).loc[np.repeat(np.arange(families), 2)].reset_index(drop=True)
        rows["region"] = np.tile(["edge", "mid"], families)
        blocks = {b: rng.normal(size=(2 * families, 3)) for b in cl.COVARIANT["cov01"]}
        labels = {kind: cl.label_vectors(rows, kind)[0] for kind in ("labels", "plane+class")}
        Y = rng.normal(size=(2 * families, 2))
        groups = np.repeat(np.arange(families), 2)
        folds = cl.make_folds(groups, 3, 2, 0)
        args = types.SimpleNamespace(scales=["standard", "none"], std_floor=1e-5, gammas=[1.0], alphas=[1e-2],
                                     label_weights=[0.0, 1.0])
        shared = cl.coinciding_pairs(blocks["invariant"], groups)
        predictions, baseline, fold_of, chosen, info = cl.run_cv(blocks, labels, Y, groups, folds, "interface",
                                                                 list(cl.INPUTS), args, shared=shared, log=lambda _: None)
        for name in cl.INPUTS:
            self.assertTrue(np.isfinite(predictions[name]).all(), name)
        self.assertTrue((fold_of >= 0).all())
        table = cl.score_table(rows, "interface", ["t0", "t1"], Y, predictions, baseline, fold_of, chosen)
        self.assertEqual(len(table), len(cl.INPUTS) * 2 * 6)
        self.assertEqual(set(table["scored_on"]), {"all", "100", "110", "111", "edge", "mid"})
        cell_blocks = blocks
        cell_rows = rows.iloc[::2].reset_index(drop=True)
        cell_labels = {kind: cl.label_vectors(cell_rows, kind)[0] for kind in ("labels", "plane+class")}
        cell_folds = cl.make_folds(np.arange(families), 3, 2, 0)
        out = cl.run_cv(cell_blocks, cell_labels, Y[::2], np.arange(families), cell_folds, "cell", list(cl.INPUTS),
                        args, log=lambda _: None)
        self.assertTrue(all(np.isfinite(out[0][name]).all() for name in cl.INPUTS))


class TableAndCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def families(self):
        return pd.DataFrame({"family": ["100/sym00000", "100/sym00001"], "plane": ["100", "100"],
                             "distinct_id": ["D1", "D2"]})

    def test_targets_join(self):
        path = self.dir / "t.csv"
        pd.DataFrame({"family": ["100/sym00000"] * 2 + ["100/sym00001"] * 2, "thickness": 8,
                      "region": ["edge", "mid"] * 2, "y": [1.0, 2.0, 3.0, 4.0]}).to_csv(path, index=False)
        targets, keys = cl.load_targets(path, ["y"], "interface", 8)
        rows = cl.build_rows(self.families(), targets, keys, ["y"], "interface", False)
        self.assertEqual(rows["y"].tolist(), [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            cl.load_targets(path, ["z"], "interface", 8)
        pd.read_csv(path).iloc[:3].to_csv(path, index=False)
        targets, keys = cl.load_targets(path, ["y"], "interface", 8)
        with self.assertRaises(ValueError):
            cl.build_rows(self.families(), targets, keys, ["y"], "interface", False)
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            cl.load_targets(path, ["y"], "interface", 8)
        pd.DataFrame({"family": ["100/sym00000"], "y": [1.0]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            cl.load_targets(path, ["y"], "interface", 8)

    def test_cell_targets_may_cover_few_families(self):
        path = self.dir / "e.csv"
        names = [f"100/sym{i:05d}" for i in range(12)]
        pd.DataFrame({"family": names + ["999/sym00000"], "E_GB": np.arange(13.0)}).to_csv(path, index=False)
        families = pd.DataFrame({"family": [f"100/sym{i:05d}" for i in range(40)], "plane": "100",
                                 "distinct_id": [f"D{i}" for i in range(40)]})
        targets, keys = cl.load_targets(path, ["E_GB"], "cell", 8)
        with self.assertRaises(ValueError):  # a family outside the catalogue
            cl.build_rows(families, targets, keys, ["E_GB"], "cell", False)
        rows = cl.build_rows(families, targets[targets["family"] != "999/sym00000"], keys, ["E_GB"], "cell", False)
        self.assertEqual(len(rows), 12)

    def test_plane_class_check(self):
        results = pd.DataFrame({"input": "plane+class", "target": "d_min", "scored_on": ["all", "edge", "mid"],
                                "krr_r2": [0.10, 0.25, 0.10], "eta2_plane_class": [0.12, 0.19, 0.11]})
        self.assertFalse(cl.check_plane_class(results, enforce=True)["pass"])
        self.assertTrue(cl.check_plane_class(results, enforce=False)["pass"])
        self.assertTrue(cl.check_plane_class(results.assign(krr_r2=[0.1, 0.2, 0.1]), enforce=True)["pass"])

    def test_cov1_check(self):
        phase1 = self.dir / "phase1"
        phase1.mkdir()
        pd.DataFrame({"study": "representation", "phase": 1, "kind": "raw", "method": "raw", "subset": "pooled",
                      "representation": "C", "variable": "knn_r2_d_min", "value": [0.34]}).to_csv(phase1 / "results.csv", index=False)
        standard = self.dir / "krr_regression.csv"
        pd.DataFrame({"dataset": "all", "grouping": "distinct", "target": "d_min", "scored_on": "all",
                      "krr_r2": [0.92]}).to_csv(standard, index=False)
        results = pd.DataFrame({"input": "cov1", "target": "d_min", "scored_on": "all", "krr_r2": [0.70]})
        with mock.patch.object(cl, "PHASE1_DIR", phase1), mock.patch.object(cl, "STANDARD_KRR", standard):
            check = cl.check_cov1(results, enforce=True)
            self.assertFalse(check["pass"])
            self.assertAlmostEqual(check["rows"][0]["midpoint"], 0.63)
            self.assertTrue(cl.check_cov1(results.assign(krr_r2=0.5), enforce=True)["pass"])
            self.assertTrue(cl.check_cov1(results, enforce=False)["pass"])


if __name__ == "__main__":
    unittest.main()
