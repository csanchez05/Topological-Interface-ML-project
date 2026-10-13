"""Tests for representation_study.py on synthetic data (no descriptor files needed).

Run: python3 -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_representation_study.py'
"""
import os
import tempfile
import unittest

import numpy as np

import representation_study as rs


def circle(n=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    points = np.column_stack([np.cos(theta), np.sin(theta)]) + noise * rng.normal(size=(n, 2))
    return theta, points


class Rows(unittest.TestCase):
    def test_partners_and_masks(self):
        import pandas as pd

        gbs = pd.DataFrame({"family": ["100/sym00000", "110/sym00001", "111/sym00002"], "plane": ["100", "110", "111"],
                            "grain_class": ["E", "I", "E"], "gb": [0, 1, 2]})
        rows = rs.interface_rows(gbs, thicknesses=(8, 9), skip={("110/sym00001", 9)})
        self.assertEqual(len(rows), 10)
        partner = rs.partners(len(rows))
        same = (rows["family"].to_numpy() == rows["family"].to_numpy()[partner]) \
            & (rows["thickness"].to_numpy() == rows["thickness"].to_numpy()[partner])
        self.assertTrue(same.all())
        self.assertTrue((rows["interface"].to_numpy() != rows["interface"].to_numpy()[partner]).all())
        for mask in rs.gb_masks(rows, np.arange(len(rows)), 3, 0.6, seed=1):
            self.assertTrue(np.isin(mask ^ 1, mask).all(), "a subsample keeps both rows of a structure")

    def test_pair_sample_leaves_out_partners(self):
        partner = rs.partners(10)
        i, j = rs.pair_sample(10, partner, 1000)
        self.assertEqual(len(i), 45 - 5)
        self.assertFalse(np.any(partner[i] == j))
        i, j = rs.pair_sample(10, partner, 7)
        self.assertEqual(len(i), 7)


class Scores(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(80, 5))
        self.Y = self.X[:, :2] + 0.3 * rng.normal(size=(80, 2))
        self.none = np.full(80, -1)

    def test_identical_maps(self):
        partner = rs.partners(80)
        pairs = rs.pair_sample(80, partner, 10 ** 6)
        a = rs.features(self.Y, np.arange(80), partner, pairs, (5, 15), 8)
        scores = rs.score_features(a, a, (5, 15), 8)
        self.assertAlmostEqual(scores["procrustes"], 0.0, places=12)
        for name in ("jaccard5", "jaccard15", "spearman", "ari"):
            self.assertAlmostEqual(scores[name], 1.0, places=12)

    def test_jaccard_undefined_when_k_covers_every_row(self):
        # 10 rows: 8 candidates per row (itself and its partner left out); k = 8 would list them all.
        partner = rs.partners(10)
        lists = rs.jaccard_lists(rs.square_distances(self.X[:10]), partner, (7, 8))
        self.assertEqual(lists[7].shape, (10, 7))
        self.assertIsNone(lists[8])
        pairs = rs.pair_sample(10, partner, 100)
        a = rs.features(self.Y[:10], np.arange(10), partner, pairs, (7, 8), 0)
        scores = rs.score_features(a, a, (7, 8), 0, coordinates=False)
        self.assertEqual(scores["jaccard7"], 1.0)
        self.assertTrue(np.isnan(scores["jaccard8"]))

    def test_jaccard_known_value(self):
        Na = np.array([[1, 2, 3, 4]])
        Nb = np.array([[3, 4, 5, 6]])
        self.assertAlmostEqual(rs.jaccard(np.vstack([Na, Na]), np.vstack([Nb, Na])), (2 / 6 + 1) / 2)

    def test_trustworthiness_matches_sklearn(self):
        from sklearn.manifold import trustworthiness

        D_in, D_out = rs.square_distances(self.X), rs.square_distances(self.Y)
        trust, continuity = rs.trust_continuity(D_in, D_out, self.none, 7)
        self.assertAlmostEqual(trust, trustworthiness(self.X, self.Y, n_neighbors=7), places=10)
        self.assertAlmostEqual(continuity, trustworthiness(self.Y, self.X, n_neighbors=7), places=10)
        self.assertEqual(rs.trust_continuity(D_in, D_in, rs.partners(80), 7), (1.0, 1.0))

    def test_partner_left_out(self):
        # Two copies of every point (as a whole-cell vector gives both interface rows): with the
        # partner left out, each row's nearest neighbour is a different structure.
        X = np.repeat(self.X, 2, axis=0)
        N = rs.neighbour_sets(rs.square_distances(X), rs.partners(len(X)), 1)[:, 0]
        self.assertFalse(np.any(N == rs.partners(len(X))))

    def test_organisation(self):
        rows = 60
        y = np.repeat([0, 1, 2], rows // 3)
        X = np.column_stack([10.0 * y, np.random.default_rng(0).normal(size=rows)])
        values = X[:, 0] + 0.01 * np.random.default_rng(1).normal(size=rows)
        codes = {"classes": {"plane": y, "grain_class": y, "interface": np.arange(rows) % 2},
                 "values": {"d_min": values}, "gb": np.arange(rows) // 2, "plane": y.astype(str)}
        out = rs.organisation(rs.square_distances(X), rs.partners(rows), codes, 5, pooled=True)
        self.assertEqual(out["knn_plane"], 1.0)
        self.assertEqual(out["knn_grain_class@1"], 1.0)
        self.assertGreater(out["knn_r2_d_min"], 0.99)
        base = rs.majority_baselines(codes, pooled=True)
        self.assertAlmostEqual(base["knn_plane"], 1 / 3)

    def test_sensitivity(self):
        self.assertEqual(rs.sensitivity(0.2, 0.1), 2.0)
        self.assertTrue(np.isnan(rs.sensitivity(0.2, 0.0)))
        self.assertEqual(rs.change("procrustes", 0.3), 0.3)
        self.assertAlmostEqual(rs.change("jaccard15", 0.3), 0.7)

    def test_exact_coordinates_keep_distances(self):
        X = np.random.default_rng(4).normal(size=(30, 200))
        Z = rs.exact_coordinates(X)
        self.assertLessEqual(Z.shape[1], 30)
        self.assertLess(rs.distance_error(X, Z), 1e-10)
        self.assertTrue(np.allclose(rs.square_distances(X), rs.square_distances(Z)))

    def test_distance_error_with_identical_rows(self):
        """Twin rows (B, D) have distance 0 in X; rounding in Z must not count as a relative change of 1e286."""
        X = np.repeat(np.random.default_rng(6).normal(size=(20, 300)), 2, axis=0)
        Z = rs.exact_coordinates(X)
        self.assertEqual(Z.shape[1], 19)
        self.assertLess(rs.distance_error(X, Z, count=5000), 1e-10)
        Z_off = Z.copy()
        Z_off[0, 0] += 1e-3 * np.median(rs.square_distances(X))
        self.assertGreater(rs.distance_error(X, Z_off, count=5000), 1e-5)  # a real change is still seen

    def test_cka_and_subspace(self):
        Q = np.linalg.qr(np.random.default_rng(5).normal(size=(5, 5)))[0]
        self.assertAlmostEqual(rs.linear_cka(self.X, 3.0 * self.X @ Q), 1.0, places=12)
        self.assertAlmostEqual(rs.subspace_overlap(self.X[:, :3], self.X[:, :3] @ Q[:3, :3]), 1.0, places=12)


class Statistics(unittest.TestCase):
    def test_permanova_equals_anova_on_euclidean_data(self):
        rng = np.random.default_rng(6)
        a = np.repeat(["x", "y"], 20)
        b = np.tile(np.repeat(["p", "q"], 10), 2)
        points = rng.normal(size=(40, 3))
        points[a == "y", 0] += 3.0
        points[b == "q", 1] += 1.0
        result = rs.permanova(rs.square_distances(points), a, b, ("a", "b"), permutations=99)
        total = np.square(points - points.mean(axis=0)).sum()
        between_a = sum(20 * np.square(points[a == g].mean(axis=0) - points.mean(axis=0)).sum() for g in ("x", "y"))
        self.assertAlmostEqual(result["a"]["share"], between_a / total, places=10)
        self.assertAlmostEqual(sum(entry["share"] for entry in result.values()), 1.0, places=10)
        self.assertLess(result["a"]["p_value"], 0.05)
        self.assertGreater(result["a"]["share"], result["b"]["share"])

    def test_permanova_without_replicates(self):
        D = rs.square_distances(np.arange(6.0)[:, None])
        result = rs.permanova(D, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], permutations=0)
        self.assertEqual(set(result), {"a", "b", "residual"})

    def test_permanova_without_variation_has_no_p_value(self):
        a = np.repeat(["x", "y"], 6)
        b = np.tile(np.repeat(["p", "q"], 3), 2)
        for D in (np.zeros((12, 12)), np.full((12, 12), np.nan)):
            result = rs.permanova(D, a, b, ("a", "b"), permutations=19)
            self.assertTrue(all(np.isnan(result[name]["p_value"]) for name in ("a", "b", "interaction")))

    def test_identical_replicates_are_significant(self):
        """Zero residual (every replicate identical, as ICA in the smoke test): a term that explains something
        is significant, one that explains nothing has no p-value."""
        a = np.repeat(["x", "y"], 6)
        b = np.tile(np.repeat(["p", "q"], 3), 2)
        points = np.column_stack([(a == "y") * 2.0, np.zeros(12)])  # only a matters, no noise
        result = rs.permanova(rs.square_distances(points), a, b, ("a", "b"), permutations=99)
        self.assertAlmostEqual(result["a"]["share"], 1.0, places=10)
        self.assertLessEqual(result["a"]["p_value"], 0.05)
        self.assertTrue(np.isnan(result["b"]["p_value"]) and np.isnan(result["interaction"]["p_value"]))
        anova = rs.anova_2x2(points[:, 0], a, b, ("a", "b"))
        self.assertEqual(anova["a"]["p_value"], 0.0)
        self.assertTrue(np.isnan(anova["b"]["p_value"]) and np.isnan(anova["interaction"]["p_value"]))

    def test_anova(self):
        rng = np.random.default_rng(7)
        a = np.repeat([0, 1], 20)
        b = np.tile(np.repeat([0, 1], 10), 2)
        values = 5.0 * a + 0.1 * rng.normal(size=40)
        result = rs.anova_2x2(values, a, b, ("region", "covariance"))
        self.assertGreater(result["region"]["share"], 0.99)
        self.assertLess(result["region"]["p_value"], 1e-10)
        self.assertAlmostEqual(sum(entry["share"] for entry in result.values()), 1.0, places=10)

    def test_reproducible_components(self):
        rng = np.random.default_rng(8)
        S = rng.normal(size=(100, 3))
        count, mean = rs.reproducible_components([S, -S[:, [2, 0, 1]], S + 1e-3 * rng.normal(size=S.shape)])
        self.assertEqual(count, 3)
        self.assertGreater(mean, 0.99)


class Methods(unittest.TestCase):
    def test_diffusion_map_on_a_circle(self):
        theta, points = circle(300)
        Y, eigenvalues = rs.diffusion_map(rs.square_distances(points), width=0.5, count=2)
        target = np.column_stack([np.cos(theta), np.sin(theta)])
        fit = np.linalg.lstsq(np.column_stack([Y, np.ones(len(Y))]), target, rcond=None)[0]
        residual = target - np.column_stack([Y, np.ones(len(Y))]) @ fit
        self.assertLess(np.square(residual).sum() / np.square(target - target.mean(axis=0)).sum(), 0.01)
        self.assertTrue(np.all(np.diff(eigenvalues[:5]) <= 1e-12))

    def test_bulk_combination(self):
        rng = np.random.default_rng(9)
        G, F = 4, 6
        X = rng.normal(size=(G, 2, F))
        n = rng.integers(3, 9, size=(G, 2, 2))
        totals = n.sum(axis=1) + 5
        is_co = rs.species_mask(F)
        bulk = rng.normal(size=F)
        cell = rs.weighted_interfaces(X, n, totals, is_co)
        for s, half in enumerate((is_co, ~is_co)):
            cell[:, half] += 5 * bulk[half] / totals[:, s, None]
        result = rs.cell_from_interfaces(cell, X, n, totals, is_co, bulk=bulk)
        self.assertLess(result["relative_residual"].max(), 1e-12)
        without_bulk = rs.cell_from_interfaces(rs.weighted_interfaces(X, n, totals, is_co), X, n, totals, is_co)
        self.assertLess(without_bulk["relative_residual"].max(), 1e-12)
        share = n.sum(axis=1) / totals
        self.assertTrue(np.allclose(without_bulk["fitted_scale"][:, None] * np.ones(2), share, atol=0.5))

    def test_tsne_kl_is_exact(self):
        """tsne_kl equals sklearn's exact-method KL, and stays >= 0 with two identical rows per GB (as in B, D)."""
        from sklearn.manifold import TSNE
        from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence

        rng = np.random.default_rng(11)
        Z = np.repeat(rng.normal(size=(30, 5)), 2, axis=0)
        D = rs.square_distances(Z)
        model = TSNE(n_components=2, perplexity=10, method="exact", init="pca", learning_rate="auto",
                     random_state=0)
        Y = model.fit_transform(Z)
        kl = rs.tsne_kl(D, Y, 10)
        P = _joint_probabilities(np.square(D), 10, 0)
        self.assertAlmostEqual(kl, _kl_divergence(Y.ravel(), P, 1, len(Z), 2)[0], places=10)
        self.assertAlmostEqual(kl, model.kl_divergence_, delta=1e-3 * model.kl_divergence_)
        self.assertGreater(kl, 0.0)
        # tsne_map uses the exact method: on these twins Barnes-Hut's map has a KL several times larger.
        self.assertAlmostEqual(rs.tsne_kl(D, rs.tsne_map(Z, 10, 0), 10), kl, places=6)

    def test_every_method_on_a_toy(self):
        """Each method returns a finite 2D map on a small labelled toy (sketch-map only if built)."""
        rng = np.random.default_rng(10)
        rows = 60
        plane = np.repeat([0, 1, 2], rows // 3)
        Z = rs.exact_coordinates(rng.normal(size=(rows, 8)) + plane[:, None])
        codes = {"classes": {"plane": plane, "grain_class": np.arange(rows) % 4, "interface": np.arange(rows) % 2},
                 "values": {name: rng.normal(size=rows) for name in rs.gl.TARGETS},
                 "gb": np.arange(rows) // 2, "plane": plane.astype(str)}
        settings = {"width_factor": 1.0, "mixing": 0.5, "sketchmap_hd": (2, 6), "sketchmap_ld": (1, 2),
                    "sketchmap_bin": str(rs.SKETCHMAP)}
        D = rs.square_distances(Z)
        for method in rs.ALL_METHODS:
            if method == "sketchmap" and not rs.SKETCHMAP.is_file():
                continue
            with self.subTest(method=method):
                Y, extras = rs.make_map(method, Z, D, codes, 42, settings, main=True)
                self.assertEqual(Y.shape, (rows, 2))
                self.assertTrue(np.isfinite(Y).all())


@unittest.skipUnless(rs.SKETCHMAP.is_file(), "sketch-map is not built")
class Grid(unittest.TestCase):
    def test_summarise_grid_with_anchor(self):
        """Sixth study summary: representation outweighs settings; anchor rows for the maps and the raw vectors."""
        from types import SimpleNamespace

        import pandas as pd

        rng = np.random.default_rng(5)
        n, ks = 40, (5,)
        partner = rs.partners(n)
        pairs = rs.pair_sample(n, partner, 10 ** 6)
        tags = ["rc5.0_n8_l4_s0.5", "rc5.0_n8_l8_s0.5"]
        sets = [f"{tag}|{rep}" for tag in tags + [rs.REFERENCE_TAG] for rep in "AB"]
        base = {rep: rng.normal(size=(n, 2)) for rep in "AB"}
        raw_features, maps = {}, {}
        for key in sets:
            X = base[key.split("|")[1]] + 0.1 * rng.normal(size=(n, 2))
            raw_features[key] = rs.features(X, np.arange(n), partner, pairs, ks, 0)
            for r in range(3):
                maps[f"pca|{key}|{r}"] = {"method": "pca", "representation": key, "replicate": r, "rows": np.arange(n),
                                          "Y": X + 0.01 * rng.normal(size=(n, 2))}
        names = sorted(maps)
        scores = {(a, b): rs.compare_maps(maps[a], maps[b], partner, pairs, ks, 0)
                  for x, a in enumerate(names) for b in names[x + 1:]}
        results = []
        arrays = rs.summarise_grid(results, ["pca"], maps, scores, raw_features, sets, tags, {"ks": ks},
                                   SimpleNamespace(permutations=49))
        table = pd.DataFrame(results)
        self.assertEqual(arrays["dissimilarity|pca|spearman"].shape, (12, 12))  # 2 settings x 2 reps x 3 replicates
        for method in ("pca", "raw"):
            part = table[table["method"] == method]
            anchor = part[part["kind"] == "anchor"]
            self.assertEqual(set(anchor["factor"]), set(tags))
            self.assertEqual(set(anchor["representation"]), {"A", "B"})
            self.assertFalse(anchor.duplicated(["variable", "representation", "factor"]).any(), "one row per setting")
        anchor = table[(table["kind"] == "anchor") & (table["method"] == "pca")]
        self.assertTrue((anchor["n"] == 9).all(), "3 reference replicates x 3 replicates of the setting")
        effect = table[(table["method"] == "pca") & (table["kind"] == "effect") & (table["variable"] == "spearman")]
        share = effect.set_index("factor")["value"]
        self.assertGreater(share["representation"], 10 * share["settings"])


class SketchMap(unittest.TestCase):
    def test_curve_unfolds(self):
        """A helix in 10 dimensions: the error falls and neighbours along the curve stay neighbours."""
        t = np.linspace(0, 4 * np.pi, 150)
        X = np.column_stack([np.cos(t), np.sin(t), 0.3 * t] + [0.0 * t] * 7)
        D = rs.square_distances(X)
        here = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                Y, history = rs.sketch_map(X, sigma=rs.median_distance(D), seed=1)
            finally:
                os.chdir(here)
            # dimred's restart files (global.<point>) must stay in the wrapper's own temporary folder.
            self.assertEqual(os.listdir(folder), [])
        self.assertEqual(Y.shape, (150, 2))
        self.assertLess(history[-1], history[0])
        trust, _ = rs.trust_continuity(D, rs.square_distances(Y), np.full(150, -1), 5)
        self.assertGreater(trust, 0.9)


if __name__ == "__main__":
    unittest.main()
