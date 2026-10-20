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


class CovariantSensitivity(unittest.TestCase):
    """--descriptor-type covariant (2026-10-13): settings, wide vectors, the vector cache."""

    def test_covariant_settings(self):
        out = ds.variants(descriptor_type="covariant")
        by_name = {v.name: v for v in out}
        self.assertEqual(out[0].name, "reference")
        self.assertEqual(out[0].tag, "rc6.0_n8_l6_s0.5")
        self.assertFalse(any(v.knob == "weighting" for v in out))
        self.assertTrue(all(v.descriptor_type == "covariant" for v in out))
        self.assertEqual(by_name["r_cut 12"].tag, "rc12.0_n8_l6_s0.5_t8")
        self.assertEqual(by_name["l_max 4"].tag, "rc6.0_n8_l4_s0.5_t8")  # the settings-grid run
        self.assertEqual(by_name["blocks lambda 2"].tag, "rc6.0_n8_l6_s0.5")
        self.assertEqual(by_name["blocks lambda 2"].blocks, ("tensor", "pseudotensor"))
        self.assertEqual(by_name["region global"].kind, "global")
        self.assertEqual(by_name["region edge plane only"].planes, "edge")
        self.assertEqual({v.knob for v in out}, set(ds.COVARIANT_KNOBS) | {"reference"})
        self.assertNotIn("l_max 2", by_name)  # fails the bulk check (COVARIANT_LEFT_OUT)
        self.assertEqual(len(out), 1 + 15 + 3 + 3 + 2)
        self.assertEqual(ds.variants(["n_max 12"], "covariant")[1].tag, "rc6.0_n12_l6_s0.5_t8")
        with self.assertRaises(ValueError):
            ds.variants(["weighting pow rc6"], "covariant")
        with self.assertRaises(ValueError):
            ds.variants(descriptor_type="no such type")
        # the standard settings are unchanged
        self.assertEqual(len(ds.variants()), 1 + 5 + 4 + 4 + 3 + 6 + 9)

    def test_narrow_keeps_distances_pca_and_reconstruction_errors(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 300)) + rng.normal(scale=0.1, size=(40, 300))
        B = np.tanh(A @ rng.normal(size=(300, 250)) / 10)
        old = ds.WIDE
        try:
            ds.WIDE = 1_000
            wide_d, wide_v = ds.square_distances(A), ds.variance_profile(A)
            wide_e = (ds.reconstruction_error(A, B), ds.reconstruction_error(B, A))
            ds.WIDE = 100
            self.assertEqual(ds.narrow(A).shape, (40, 40))
            self.assertIs(ds.narrow(ds.narrow(A)).shape[1] <= ds.WIDE, True)
            np.testing.assert_allclose(ds.square_distances(A), wide_d, atol=1e-9)
            narrow_v = ds.variance_profile(A)
            self.assertAlmostEqual(narrow_v["pc1"], wide_v["pc1"], places=10)
            self.assertEqual(narrow_v["n_pc90"], wide_v["n_pc90"])
            np.testing.assert_allclose((ds.reconstruction_error(A, B), ds.reconstruction_error(B, A)), wide_e,
                                       rtol=1e-6)
            disparity, _, _ = ds.aligned(ds.pca_map(A), ds.pca_map(ds.narrow(A)))
            self.assertLess(disparity, 1e-12)
        finally:
            ds.WIDE = old

    def test_thickness_share_in_chunks(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(30, 50)).astype(np.float32)
        labels = np.repeat(np.arange(6), 5)
        whole = ds.thickness_share(values, labels)
        old = ds.WIDE
        try:
            ds.WIDE = 10
            self.assertAlmostEqual(ds.thickness_share(values, labels, chunk=7), whole, places=6)
        finally:
            ds.WIDE = old

    def test_cache_returns_what_the_loader_gave(self):
        X = np.array([[0.5, 1.25], [np.nan, np.nan], [2.0, 3.0]])
        table = pd.DataFrame({"family": ["100/sym00001", "110/sym00002", "111/sym00003"],
                              "plane": ["100", "110", "111"], "grain_class": ["E", "I", "E"]})
        calls = []

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return X.copy(), table.copy(), np.array([0.0, 0.1])

        variant = ds.variants(["blocks lambda 1"], "covariant")[1]
        old_loader = ds.mc.load_distinct
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ds.mc.load_distinct = fake
                ds.os.environ[ds.CACHE_ENV] = tmp
                first = ds.distinct_vectors(variant)
                second = ds.distinct_vectors(variant)
                self.assertEqual(ds.warm_task(variant)[1], (3, 2))
            finally:
                ds.mc.load_distinct = old_loader
                ds.os.environ.pop(ds.CACHE_ENV, None)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["covariant_blocks"], ("polar", "axial"))
            self.assertEqual(calls[0][1]["descriptor_type"], "covariant")
            for got in (first, second):
                np.testing.assert_array_equal(got[0], X)
                self.assertEqual(got[0].dtype, np.float64)
                pd.testing.assert_frame_equal(got[1], table)
                np.testing.assert_array_equal(got[2], [0.0, 0.1])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             sorted([ds.cache_key(variant) + ".npz", ds.cache_key(variant) + ".csv"]))
            # values that float32 cannot hold are stored as float64
            X[0, 0] = 1 / 3
            ds.os.environ[ds.CACHE_ENV] = tmp
            other = ds.variants(["blocks lambda 2"], "covariant")[1]
            try:
                ds.mc.load_distinct = fake
                ds.distinct_vectors(other)
                self.assertEqual(ds.distinct_vectors(other)[0][0, 0], 1 / 3)
            finally:
                ds.mc.load_distinct = old_loader
                ds.os.environ.pop(ds.CACHE_ENV, None)

    def test_one_mean_is_standard_soap_only(self):
        variant = ds.Variant("x", "processing", "x", mean="all", descriptor_type="covariant")
        old = ds.distinct_vectors
        try:
            ds.distinct_vectors = lambda v: (np.zeros((2, 4)), pd.DataFrame({"family": ["a", "b"]}), np.zeros(4))
            with self.assertRaises(ValueError):
                ds.load_variant(variant)
        finally:
            ds.distinct_vectors = old


class AceSensitivity(unittest.TestCase):
    """--descriptor-type ace (2026-10-13): settings, baselines, what reaches the loader."""

    REFERENCE = "rc6.0_n8_l6_s0.5_nu3-n4-l4"

    def test_ace_settings(self):
        saved = list(sys.path)
        out = ds.variants(descriptor_type="ace")
        self.assertEqual(sys.path, saved)  # importing ace_grid leaves sys.path as it was
        by_name = {v.name: v for v in out}
        self.assertEqual(out[0].name, "reference")
        self.assertEqual(out[0].tag, self.REFERENCE)
        self.assertTrue(all(v.descriptor_type == "ace" for v in out))
        self.assertEqual({v.knob for v in out}, set(ds.ACE_KNOBS) | {"reference"})
        # 27 one-knob runs + the reference, then 4 blocks, 5 regions, 3 processing
        self.assertEqual(len(out), 28 + 4 + 5 + 3)
        self.assertEqual(by_name["r_cut 12"].tag, "rc12.0_n8_l6_s0.5_nu3-n4-l4_t8")
        self.assertEqual(by_name["n3 2"].tag, "rc6.0_n8_l6_s0.5_nu3-n2-l4_t8")
        self.assertEqual(by_name["l3 6"].tag, "rc6.0_n8_l6_s0.5_nu3-n4-l6_t8")
        for v in out:
            self.assertEqual(v.ace_scaling, "none" if v.name == "processing blocks unscaled" else "pristine", v.name)
            self.assertTrue(v.tag == self.REFERENCE or v.tag.endswith("_t8"), v.name)
        # weighting: pow only; at r_cut 8 and 12 against the unweighted run at that r_cut
        self.assertEqual(sorted(v.value for v in out if v.knob == "weighting"), ["pow rc12", "pow rc6", "pow rc8"])
        self.assertEqual(by_name["weighting pow rc6"].baseline, "reference")
        self.assertEqual(by_name["weighting pow rc8"].baseline, "r_cut 8")
        self.assertEqual(by_name["weighting pow rc12"].baseline, "r_cut 12")
        # l3 2: nu3_odd vanishes in pristine bulk, so the run is compared without it
        self.assertEqual(by_name["l3 2"].blocks, ds.ACE_NO_ODD)
        self.assertEqual(by_name["l3 2"].baseline, "blocks without nu3_odd")
        self.assertEqual(by_name["blocks without nu3_odd"].tag, self.REFERENCE)
        self.assertIsNone(by_name["l3 3"].blocks)
        self.assertEqual(by_name["blocks nu2 only"].blocks, ("nu2",))
        # radii 6 and 10 are stored in the 8x copy of the reference
        self.assertEqual((by_name["region interface r6"].tag, by_name["region interface r6"].radius),
                         (self.REFERENCE + "_t8", 6.0))
        self.assertEqual(by_name["region global"].kind, "global")
        self.assertEqual(by_name["region mid plane only"].planes, "mid")
        self.assertFalse(any(v.kind == "nonbulk" or v.mean == "all" for v in out))
        self.assertEqual([v.name for v in ds.variants(["l3 2"], "ace")], ["reference", "l3 2"])

    def test_records_and_cache_keys(self):
        soap, ace = ds.variants()[0], ds.variants(descriptor_type="ace")[0]
        self.assertNotIn("ace_scaling", ds.variant_record(soap))
        self.assertEqual(ds.variant_record(ace)["ace_scaling"], "pristine")
        self.assertEqual(ds.variant_record(ace)["tag"], self.REFERENCE)
        unscaled = ds.variants(["processing blocks unscaled"], "ace")[1]
        self.assertNotEqual(ds.cache_key(ace), ds.cache_key(unscaled))
        self.assertEqual(ds.cache_key(soap), "soap__rc6.0_n8_l6_s0.5__interface__r8__both__default")  # as before

    def test_loader_gets_the_ace_blocks_and_scaling(self):
        calls = []

        def fake(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros((2, 3)), pd.DataFrame({"family": ["a", "b"]}), np.zeros(3)

        old = ds.mc.load_distinct
        try:
            ds.mc.load_distinct = fake
            for name in ("l3 2", "processing blocks unscaled", "reference"):
                ds.distinct_vectors(ds.variants([name], "ace")[-1])
        finally:
            ds.mc.load_distinct = old
        self.assertEqual(calls[0], {"allow_undefined": True, "descriptor_type": "ace",
                                    "ace_blocks": list(ds.ACE_NO_ODD), "ace_scaling": "pristine"})
        self.assertEqual((calls[1]["ace_blocks"], calls[1]["ace_scaling"]), (None, "none"))
        self.assertEqual((calls[2]["ace_blocks"], calls[2]["ace_scaling"]), (None, "pristine"))
        self.assertNotIn("covariant_blocks", calls[0])
        self.assertEqual(ds.block_source(ds.variants()[0]), {"descriptor_type": "soap", "covariant_blocks": None})

    def test_ace_outputs(self):
        self.assertEqual(sb.OUT_DIRS["ace"], mc.OUTPUT_ROOTS["ace"] / "selection_benchmark")
        self.assertEqual(ds.RUNS["ace"]()[0][0], "reference")


class BenchmarkKrrSettings(unittest.TestCase):
    def test_most_common_choice_per_label(self):
        def fold(grain, plane):
            return {"krr": {"grain_class": {"scale": grain[0], "gamma_times_median": grain[1], "alpha": grain[2]},
                            "plane": {"scale": plane[0], "gamma_times_median": plane[1], "alpha": plane[2]}}}

        chosen = {"all/family": [fold(("none", 1.0, 1.0), ("none", 1.0, 1.0))] * 5,
                  "all/distinct": [fold(("none", 100.0, 1e-6), ("standard", 0.1, 0.01)),
                                   fold(("none", 100.0, 1e-6), ("standard", 1.0, 0.01)),
                                   fold(("standard", 10.0, 1e-4), ("standard", 1.0, 0.01))]}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "krr_summary.json").write_text(__import__("json").dumps({"hyperparameters_chosen_per_fold": chosen}))
            got = sb.krr_settings_from(tmp)
        self.assertEqual(got["grain_class"], ("none", 100.0, 1e-6))
        self.assertEqual(got["plane"], ("standard", 1.0, 0.01))
        self.assertEqual(got["handedness"], sb.KRR_SETTINGS["handedness"])  # not in the run: the default

    def test_spaces_carry_the_settings(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        self.assertEqual(sb.Spaces(X, np.arange(6)).krr_settings, sb.KRR_SETTINGS)
        mine = {**sb.KRR_SETTINGS, "grain_class": ("none", 1.0, 1.0)}
        self.assertEqual(sb.Spaces(X, np.arange(6), mine).krr_settings["grain_class"], ("none", 1.0, 1.0))


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

    def test_existing_lists_outside_the_pool(self):
        # The ACE PCA lists pick 111/sym00119, which the default pool drops (NO_BOUNDARY): left out and reported.
        inside = [f"structures/CoGeGrainBoundaries/CoGe_100_GBs/Interface/sym{i:05d}/8x/POSCAR" for i in range(4)]
        bulk_like = "structures/CoGeGrainBoundaries/CoGe_111_GBs/Interface/sym00119/8x/POSCAR"
        row_of_source = {s: i for i, s in enumerate(inside)}
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("FPS", "median"):
                (Path(tmp) / folder).mkdir()
            (Path(tmp) / "FPS" / "pca.txt").write_text("\n".join([inside[2], bulk_like, inside[0]]) + "\n")
            (Path(tmp) / "median" / "umap.txt").write_text("\n".join(inside[1:]) + "\n")
            lists, outside = sb.read_existing_lists(tmp, row_of_source)
        self.assertEqual(sorted(lists), ["existing FPS/pca.txt", "existing median/umap.txt"])
        self.assertEqual(lists["existing FPS/pca.txt"].tolist(), [2, 0])
        self.assertEqual(lists["existing median/umap.txt"].tolist(), [1, 2, 3])
        self.assertEqual(outside, {"existing FPS/pca.txt": [bulk_like]})


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



class CovariantStudyTasks(unittest.TestCase):
    def tasks(self, tmp, only=None):
        import argparse

        import descriptor_grid as dg
        root = Path(tmp)
        args = argparse.Namespace(
            only=only, no_analyses=False, no_plots=False, force=False, limit=0, thicknesses=list(range(4, 13)),
            covariant_root=root / "covariant", covariant_out_root=root / "ml", covariant_plot_dir=root / "plots",
            comparison_plot_dir=root / "comparison", plot_dir=root / "standard", labels_dir=root / "labels",
            covariant_cpus=4, covariant_mem=8.0, grid_covariant_cpus=2, grid_covariant_mem=4.0, krr_cpus=2, krr_mem=8.0,
            scan_cpus=2, scan_mem=8.0, duplicates_mem=8.0, sensitivity_mem=8.0, analysis_cpus=2, analysis_mem=8.0)
        return {task.name: task for task in dg.covariant_study_tasks(args)}

    @staticmethod
    def argv(task, k=0):
        return [str(a) for a in task.commands[k][0]]

    def test_every_standard_analysis_is_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = self.tasks(tmp)
        runs = [n for n in tasks if n.startswith("covariant ")]
        self.assertEqual(len(runs), 16)  # reference + 15 settings (l_max 2 left out: COVARIANT_LEFT_OUT)
        self.assertNotIn("covariant l_max 2", tasks)
        self.assertEqual(set(tasks) - set(runs), {"duplicates", "krr", "select", "select plane100", "scan rc4", "scan rc5",
                                                  "scan rc6", "scan rc8", "scan rc10", "scan rc12", "scan summary",
                                                  "sensitivity", "benchmark", "plots"})
        reference = self.argv(tasks["covariant reference"])
        self.assertNotIn("--blocks", reference)
        self.assertEqual(reference[reference.index("--thickness") + 1:reference.index("--thickness") + 10],
                         [str(t) for t in range(4, 13)])
        far = self.argv(tasks["covariant r_cut 12"])
        self.assertIn("rc12.0_n8_l6_s0.5_t8", far)
        self.assertEqual(far[far.index("--blocks") + 1:far.index("--blocks") + 5], ["polar", "axial", "tensor", "pseudotensor"])
        krr = self.argv(tasks["krr"])
        self.assertTrue({"--census-grouping", "--regression", "covariant"} <= set(krr))
        self.assertEqual(tasks["krr"].needs, ["covariant reference", "duplicates"])
        scan6, scan10 = self.argv(tasks["scan rc6"], 1), self.argv(tasks["scan rc10"], 1)
        self.assertEqual(scan6[scan6.index("--descriptor") + 1], "rc6.0_n8_l6_s0.5")
        self.assertEqual(scan10[scan10.index("--descriptor") + 1], "rc10.0_n8_l6_s0.5_t8")
        self.assertTrue({"--census-grouping", "--keep-no-boundary"} <= set(scan10))
        self.assertEqual(tasks["scan rc6"].needs, ["covariant reference"])
        self.assertTrue({"krr", "select"} <= set(tasks["benchmark"].needs))
        self.assertIn("--census-grouping", self.argv(tasks["scan rc4"], 1))
        plots = self.argv(tasks["plots"])
        self.assertIn("grain_class_overview", plots)
        self.assertEqual(self.argv(tasks["plots"], 1)[-4:-2], ["--only", "stage1"])
        select = self.argv(tasks["select plane100"])
        self.assertEqual(select[select.index("--plot-dir") + 1], str(Path(tmp) / "plots" / "plane100_dmin1.8"))
        for name, task in tasks.items():  # every task writes only into the folders the driver was given
            for k in range(len(task.commands)):
                argv = self.argv(task, k)
                for i, flag in enumerate(argv[:-1]):
                    if flag in ("--out-dir", "--plot-dir", "--scan-dir", "--out-root", "--comparison-out-dir",
                                "--existing-dir"):
                        self.assertTrue(argv[i + 1].startswith(tmp), (name, flag, argv[i + 1]))
                self.assertTrue(name.startswith("covariant") or "--out-dir" in argv or "--scan-dir" in argv, name)

    def test_only_limits_runs_scan_points_and_analyses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = self.tasks(tmp, only=["r_cut 4"])
            with self.assertRaises(SystemExit):
                self.tasks(tmp, only=["weighting pow rc6"])  # no weighting in covariant SOAP
        self.assertEqual(sorted(n for n in tasks if n.startswith(("covariant ", "scan rc"))),
                         ["covariant r_cut 4", "covariant reference", "scan rc4", "scan rc6"])
        sensitivity = self.argv(tasks["sensitivity"])
        self.assertEqual(sensitivity[sensitivity.index("--only") + 1:], ["r_cut 4"])


if __name__ == "__main__":
    unittest.main()
