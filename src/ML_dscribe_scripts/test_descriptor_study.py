"""Synthetic tests for the descriptor study (descriptor_grid.py, descriptor_sensitivity.py,
selection_benchmark.py) and the full-space selection helpers in ml_common.py.

    python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import descriptor_sensitivity as ds
import ml_common as mc
import rcut_scan_all
import selection_benchmark as sb


def clusters(n_per=20, centres=((0, 0), (10, 0), (0, 10)), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.5, size=(n_per, 2)) for c in centres])
    labels = np.repeat(np.arange(len(centres)), n_per)
    return X, labels


class FullSpaceSelection(unittest.TestCase):
    def test_fps_starts_where_told_then_takes_the_farthest(self):
        X = np.array([[0.0], [1.0], [5.0], [9.0], [10.0]])
        np.testing.assert_array_equal(mc.farthest_point_sampling(X, 3, 4), [4, 0, 2])
        self.assertEqual(mc.farthest_from(X, np.array([10.0])), 0)

    def test_fps_never_repeats_a_row(self):
        chosen = mc.farthest_point_sampling(np.zeros((4, 3)), 4, 2)
        self.assertEqual(sorted(chosen.tolist()), [0, 1, 2, 3])

    def test_kmeans_takes_one_member_per_cluster(self):
        X, labels = clusters()
        chosen = mc.kmeans_members(X, 3)
        self.assertEqual(sorted(labels[chosen].tolist()), [0, 1, 2])

    def test_gp_variance_matches_the_exact_posterior(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(25, 3))
        gamma, width = 1.0, 2.0
        chosen = mc.gp_variance_selection(X, 6, gamma, width, start=3)
        K = np.exp(-gamma * np.square(X[:, None] - X[None]).sum(-1) / width)
        picked = [3]
        for _ in range(5):
            S = np.array(picked)
            variance = 1.0 - np.einsum("ij,ij->i", K[:, S] @ np.linalg.inv(K[np.ix_(S, S)]), K[:, S])
            variance[S] = -np.inf
            picked.append(int(np.argmax(variance)))
        np.testing.assert_array_equal(chosen, picked)

    def test_gp_variance_skips_a_copy_of_a_pick(self):
        X = np.array([[0.0], [0.0], [3.0], [6.0]])
        chosen = mc.gp_variance_selection(X, 3, 1.0, 1.0, start=0)
        self.assertNotIn(1, chosen.tolist())


class SensitivitySettings(unittest.TestCase):
    def test_grid_has_one_change_per_run(self):
        runs = ds.soap_runs()
        self.assertEqual(runs[0][0], "reference")
        self.assertEqual(len(runs), 23)
        tags = [ds.soap_tag(settings) for _, _, settings in runs]
        self.assertEqual(len(set(tags)), len(tags))
        for knob, _, settings in runs[1:]:
            changed = {k for k in ds.REFERENCE if settings[k] != ds.REFERENCE[k]}
            if knob == "weighting":
                self.assertIn("weighting", changed)
                self.assertLessEqual(changed, {"weighting", "r_cut"})
            else:
                self.assertEqual(changed, {knob})

    def test_weighting_tags_are_readable(self):
        self.assertEqual(ds.soap_tag({**ds.REFERENCE, "weighting": ds.pow_weighting(6)}),
                         "rc6.0_n8_l6_s0.5_wpow-r3-c1-d1-m4")
        self.assertEqual(ds.soap_tag({**ds.REFERENCE, "r_cut": 12.0, "weighting": ds.poly_weighting(12)}),
                         "rc12.0_n8_l6_s0.5_wpoly-r12-c1-m2")

    def test_weighting_at_a_larger_cutoff_is_compared_with_that_cutoff(self):
        by_name = {v.name: v for v in ds.variants()}
        self.assertEqual(by_name["weighting pow rc8"].baseline, "r_cut 8")
        self.assertEqual(by_name["weighting poly rc6"].baseline, "reference")
        alone = {v.name: v for v in ds.variants(["weighting pow rc8"])}
        self.assertEqual(set(alone), {"reference", "weighting pow rc8"})
        self.assertEqual(alone["weighting pow rc8"].baseline, "reference")
        with self.assertRaises(ValueError):
            ds.variants(["no such setting"])

    def test_processing(self):
        rng = np.random.default_rng(2)
        X, bulk = rng.normal(3.0, 2.0, size=(40, 5)), rng.normal(size=5)
        X[7] = np.nan
        Z, z = ds.process(X, bulk, "standard")
        defined = np.isfinite(Z).all(axis=1)
        np.testing.assert_allclose(Z[defined].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Z[defined].std(axis=0), 1, atol=1e-12)
        mean, spread = X[defined].mean(axis=0), X[defined].std(axis=0)
        np.testing.assert_allclose(z, (bulk - mean) / spread)
        U, u = ds.process(X, bulk, "unit")
        np.testing.assert_allclose(np.linalg.norm(U[defined], axis=1), 1)
        self.assertAlmostEqual(np.linalg.norm(u), 1)


class SensitivityScores(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(80, 12))
        self.bulk = np.zeros(12)
        self.families = np.array([f"f{i:03d}" for i in range(80)])

    def test_a_vector_against_itself(self):
        scores, turned, keep = ds.compare(self.X, self.bulk, self.X, self.bulk, self.families)
        self.assertAlmostEqual(scores["spearman"], 1.0)
        self.assertEqual(scores["knn10_kept"], 1.0)
        self.assertLess(scores["procrustes_pca"], 1e-12)
        self.assertEqual(scores["fps_overlap"], ds.N_PICKS)
        self.assertLess(scores["gfre_from_base"], 0.1)
        self.assertTrue(keep.all())

    def test_a_scaled_and_turned_copy_is_the_same_picture(self):
        q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(12, 12)))
        scores, _, _ = ds.compare(self.X, self.bulk, 3.0 * self.X @ q, self.bulk, self.families)
        self.assertAlmostEqual(scores["spearman"], 1.0)
        self.assertEqual(scores["knn10_kept"], 1.0)
        self.assertLess(scores["procrustes_pca"], 1e-10)
        self.assertEqual(scores["fps_overlap"], ds.N_PICKS)

    def test_unrelated_vectors_score_low(self):
        other = np.random.default_rng(5).normal(size=self.X.shape)
        scores, _, _ = ds.compare(self.X, self.bulk, other, self.bulk, self.families)
        self.assertLess(abs(scores["spearman"]), 0.2)
        self.assertLess(scores["knn10_kept"], 0.4)
        self.assertGreater(scores["gfre_from_base"], 0.5)

    def test_undefined_rows_are_left_out(self):
        other = self.X.copy()
        other[[3, 9]] = np.nan
        scores, turned, keep = ds.compare(self.X, self.bulk, other, self.bulk, self.families)
        self.assertEqual(scores["n_gbs"], 78)
        self.assertEqual(int(keep.sum()), 78)
        self.assertEqual(turned.shape, (78, 2))

    def test_thickness_share(self):
        thickness = np.repeat([4, 8, 12], 10)
        self.assertAlmostEqual(ds.thickness_share(thickness.astype(float), thickness), 1.0)
        noise = np.tile(np.linspace(-1, 1, 10), 3)
        self.assertAlmostEqual(ds.thickness_share(noise, thickness), 0.0)

    def test_knob_ranking_puts_the_largest_change_first(self):
        table = pd.DataFrame([
            {"knob": "n_max", "value": "4", "spearman": 0.99, "knn10_kept": 0.9, "gfre_from_base": 0.1,
             "gfre_to_base": 0.1, "procrustes_pca": 0.01, "fps_overlap": 25},
            {"knob": "r_cut", "value": "12", "spearman": 0.5, "knn10_kept": 0.3, "gfre_from_base": 0.6,
             "gfre_to_base": 0.5, "procrustes_pca": 0.4, "fps_overlap": 5}])
        self.assertEqual(ds.knob_ranking(table)["knob"].tolist(), ["r_cut", "n_max"])


class Benchmark(unittest.TestCase):
    def test_splits_are_disjoint_and_complete(self):
        for pool, test in sb.test_splits(50, n_splits=3, n_test=10):
            self.assertEqual(len(test), 10)
            self.assertEqual(sorted(np.concatenate([pool, test]).tolist()), list(range(50)))

    def test_largest_remainder(self):
        quota = sb.largest_remainder(10, {"a": 50, "b": 30, "c": 20})
        self.assertEqual(quota, {"a": 5, "b": 3, "c": 2})
        quota = sb.largest_remainder(5, {"a": 1, "b": 1, "c": 98})
        self.assertEqual(sum(quota.values()), 5)
        self.assertLessEqual(quota["a"], 1)
        self.assertEqual(sb.largest_remainder(4, {"a": 1, "b": 1}), {"a": 1, "b": 1})

    def test_stratified_picks_follow_the_groups(self):
        groups = np.array(["a"] * 30 + ["b"] * 10)
        pool = np.arange(40)
        picks = sb.stratified_picks(pool, groups, (4, 8), np.random.default_rng(0))
        self.assertEqual(list(groups[picks[4]]).count("a"), 3)
        self.assertEqual(list(groups[picks[8]]).count("b"), 2)
        self.assertEqual(len(set(picks[8].tolist())), 8)

    def test_every_strategy_picks_distinct_pool_rows(self):
        X, labels = clusters(n_per=30)
        X = np.hstack([X, np.random.default_rng(6).normal(size=(len(X), 3))])
        bulk = np.zeros(X.shape[1])
        pool = np.arange(10, len(X))
        groups = np.array([f"g{label}" for label in labels])
        families = np.array([f"f{i:03d}" for i in range(len(X))])
        maps = {"pca": np.vstack([X[:, :2], np.zeros(2)])}
        sizes = (3, 6, 12)
        picks = sb.strategy_picks(X, bulk, pool, groups, families, maps, sizes, sb.gp_kernel(X, pool), 0,
                                  sb.FULL_SPACE + ("map_fps_pca", "map_kmeans_pca"), n_draws=2)
        self.assertEqual(set(picks), set(sb.FULL_SPACE) | {"map_fps_pca", "map_kmeans_pca"})
        for strategy, draws in picks.items():
            self.assertEqual(len(draws), 2 if strategy in sb.RANDOM_STRATEGIES else 1)
            for _, by_size in draws:
                for n in sizes:
                    rows = by_size[n]
                    self.assertEqual(len(set(rows.tolist())), n, strategy)
                    self.assertTrue(np.isin(rows, pool).all(), strategy)
        fps = picks["fps"][0][1]
        np.testing.assert_array_equal(fps[12][:6], fps[6])

    def test_models_learn_separable_classes(self):
        X, labels = clusters(n_per=20)
        X = np.hstack([X, np.random.default_rng(7).normal(scale=0.1, size=(len(X), 3))])
        table = pd.DataFrame({"grain_class": labels.astype(str), "plane": labels.astype(str),
                              "handedness": labels.astype(str)})
        codes = sb.label_codes(table)
        pool, test = np.arange(0, 60, 2), np.arange(1, 60, 2)
        spaces = sb.Spaces(X, pool)
        picks = pool[[0, 1, 10, 11, 20, 21]]
        for row in sb.score_picks(spaces, codes, picks, test):
            self.assertEqual(row["accuracy"], 1.0, row)

    def test_balanced_accuracy(self):
        accuracy, balanced = sb.accuracies(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]))
        self.assertEqual(accuracy, 0.75)
        self.assertEqual(balanced, 0.5)

    def test_coverage_scores(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0]])
        D = np.abs(X - X.T)
        scores = sb.coverage_scores(D, 1.0, np.array(["a", "a", "b", "b"]), np.array([0, 3]))
        self.assertEqual(scores["coverage_radius"], 2.0)
        self.assertEqual(scores["mean_nearest_pick"], 0.75)
        self.assertEqual(scores["groups_covered"], 2)


class MemoryBudget(unittest.TestCase):
    def test_tasks_wait_for_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / "trace.txt"
            tasks = []
            for name in ("a", "b"):
                marker = Path(folder) / f"{name}.done"
                code = (f"import pathlib, time; t = pathlib.Path({str(trace)!r}); "
                        f"t.open('a').write('start {name}\\n'); time.sleep(0.3); "
                        f"t.open('a').write('end {name}\\n'); pathlib.Path({str(marker)!r}).write_text('x')")
                tasks.append(rcut_scan_all.Task(name=name, kind="thickness", cpus=1, mem_gb=6.0,
                                                commands=[([sys.executable, "-c", code], False)], outputs=[marker]))
            rcut_scan_all.run(tasks, 4, Path(folder), Path(folder) / "status.json", total_mem_gb=10.0)
            lines = trace.read_text().split()
            self.assertEqual(lines[1::2], ["a", "a", "b", "b"])
            self.assertEqual([task.state for task in tasks], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
