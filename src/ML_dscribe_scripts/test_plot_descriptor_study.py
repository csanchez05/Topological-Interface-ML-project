"""Synthetic tests for plot_descriptor_study.py: every section draws its figures from study-shaped
outputs (all settings of descriptor_sensitivity.py, ten seeds, nine thicknesses, every selection strategy).

    python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_plot_descriptor_study.py'
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import descriptor_sensitivity as ds
import geometry_labels as gl
import plot_descriptor_study as pds
import representation_study as rst
import selection_benchmark as sb
from select_gbs import GRAIN_COLORS

N_GBS = 60
SCORE_COLUMNS = ("spearman", "knn10_kept", "gfre_from_base", "gfre_to_base", "procrustes_pca", "fps_overlap")


def sensitivity_outputs(folder, rng, descriptor_type="soap"):
    """sensitivity.csv, knob_ranking.csv, maps.npz, seed_robustness.csv and the thickness files."""
    rows = []
    for v in ds.variants(descriptor_type=descriptor_type):
        same = v.name == "reference"
        row = {"name": v.name, "knob": v.knob, "value": v.value, "baseline": v.baseline,
               "vector_length": 1904, "soap_cpu_hours": rng.uniform(1, 5), "pc1": rng.uniform(0.2, 0.4),
               "pc2": rng.uniform(0.1, 0.2), "n_pc90": rng.integers(5, 15), "n_pc99": rng.integers(30, 60),
               "spearman": 1.0 if same else rng.uniform(0.8, 1), "knn10_kept": 1.0 if same else rng.uniform(0.5, 1),
               "gfre_from_base": 0.0066, "gfre_to_base": 0.0066, "procrustes_pca": 0.0 if same else rng.uniform(0, 0.5),
               "fps_overlap": 30 if same else int(rng.integers(5, 30))}
        row.update({f"{c}_vs_reference": row[c] for c in SCORE_COLUMNS})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "sensitivity.csv", index=False)
    knobs = table[table["knob"] != "reference"]
    ranking = knobs.groupby("knob", sort=False).agg(
        n_settings=("name", "size"), values=("value", lambda v: ", ".join(v)), min_spearman=("spearman", "min"),
        min_knn10_kept=("knn10_kept", "min"), max_gfre=("gfre_from_base", "max"),
        max_procrustes_pca=("procrustes_pca", "max"), min_fps_overlap=("fps_overlap", "min")).reset_index()
    ranking["mean_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(folder / "knob_ranking.csv", index=False)

    families = np.array([f"100/sym{i:05d}" for i in range(N_GBS)])
    names = table["name"].to_numpy().astype(str)
    arrays = {"names": names, "families": families,
              "grain_class": rng.choice(list(GRAIN_COLORS), size=N_GBS).astype(str),
              "pca_maps": rng.normal(size=(len(names), N_GBS, 2))}
    seed_rows = []
    for method in ds.METHODS:
        for name in names:
            for seed in (ds.SEEDS if name == "reference" else ds.SEEDS[:1]):
                key = f"{method}__{name}__seed{seed}"
                arrays[key] = rng.normal(size=(N_GBS + 1, 2))
                arrays[f"{key}__fps"] = rng.choice(families, size=ds.N_PICKS, replace=False)
                arrays[f"{key}__median"] = rng.choice(families, size=ds.N_PICKS, replace=False)
            if name != "reference":
                seed_rows.append({"method": method, "comparison": "setting vs reference", "name": name,
                                  "procrustes": rng.uniform(0, 0.5), "fps_overlap": int(rng.integers(5, 30))})
        for k in range(len(ds.SEEDS) - 1):
            seed_rows.append({"method": method, "comparison": "seed pair", "name": "reference",
                              "procrustes": rng.uniform(0, 0.1), "fps_overlap": int(rng.integers(10, 30))})
    np.savez(folder / "maps.npz", **arrays)
    pd.DataFrame(seed_rows).to_csv(folder / "seed_robustness.csv", index=False)

    thickness = np.tile(np.arange(4, 13), N_GBS)
    family = np.repeat(families, 9)
    pcs = {f"{kind}_pcs": rng.normal(size=(len(thickness), 2)) for kind in ("global", "interface")}
    pcs["interface_pcs"][:3] = np.nan  # undefined vectors stay out
    np.savez(folder / "thickness_pcs.npz", **pcs, global_share=np.array([0.42, 0.13]),
             interface_share=np.array([0.32, 0.17]), thickness=thickness, family=family,
             plane=np.full(len(thickness), "100"))
    pd.DataFrame([{"vector": vector, "subset": "all structures", "axis": axis,
                   "thickness_share": rng.uniform(0, 1), "gb_share": rng.uniform(0, 1)}
                  for vector in ("global", "interface r8 both") for axis in ("PC1", "PC2", "all features")]
                 ).to_csv(folder / "thickness_test.csv", index=False)


def benchmark_outputs(folder, rng):
    """summary.csv/json, maps.npz, coverage.csv, existing_lists.csv and picks.csv of selection_benchmark.py."""
    families = np.array([f"100/sym{i:05d}" for i in range(N_GBS)])
    labels = {"grain_class": rng.choice(list(GRAIN_COLORS), size=N_GBS), "plane": rng.choice(["100", "110"], N_GBS),
              "handedness": rng.choice(["left", "right"], N_GBS)}
    np.savez(folder / "maps.npz", families=families, **{k: v.astype(str) for k, v in labels.items()},
             **{m: rng.normal(size=(N_GBS + 1, 2)) for m in ("pca", "tsne", "umap")})
    strategies = sb.FULL_SPACE + sb.MAP_RULES
    rows = [{"descriptor": descriptor, "strategy": strategy, "label": label, "model": model, "n": n,
             "accuracy_mean": rng.uniform(0.2, 0.3), "accuracy_std": 0.02, "balanced_accuracy_mean": rng.uniform(0.1, 0.2),
             "balanced_accuracy_std": 0.02, "n_splits": 10, "mean_accuracy_over_n": rng.uniform(0.2, 0.3)}
            for descriptor in ("reference", "r_cut 8", "sigma 0.3") for strategy in strategies for label in sb.LABELS
            for model in ("krr", "1nn") for n in sb.SIZES]
    rows += [{"descriptor": "reference", "strategy": strategy, "label": f"grain_class@{plane}", "model": model, "n": n,
              "accuracy_mean": rng.uniform(0.2, 0.6), "accuracy_std": 0.03, "balanced_accuracy_mean": rng.uniform(0.1, 0.5),
              "balanced_accuracy_std": 0.03, "n_splits": 10}
             for strategy in sb.FULL_SPACE for plane in ("100", "110", "111") for model in ("krr", "1nn") for n in sb.SIZES]
    rows += [{"descriptor": "reference", "strategy": strategy, "label": label, "model": model, "n": n,
              "r2_mean": rng.uniform(-0.5, 0.8), "r2_std": 0.1, "mae_mean": rng.uniform(0, 1), "n_splits": 10}
             for strategy in sb.FULL_SPACE for label in gl.TARGETS for model in ("krr", "1nn") for n in sb.SIZES]
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    (folder / "summary.json").write_text('{"majority_class_share": {"grain_class": 0.238, "plane": 0.45, '
                                         '"handedness": 0.5}, "grain_class_majority_per_plane": '
                                         '{"100": 0.3, "110": 0.28, "111": 0.2}}')
    pd.DataFrame([{"descriptor": "reference", "split": split, "strategy": strategy, "draw": 0, "n": n,
                   "coverage_radius": rng.uniform(1, 2), "mean_nearest_pick": rng.uniform(0.4, 1),
                   "groups_covered": int(rng.integers(3, 17))}
                  for split in range(2) for strategy in strategies for n in sb.SIZES]).to_csv(folder / "coverage.csv",
                                                                                              index=False)
    lists = [f"existing {rule}/{m}.txt" for rule in ("FPS", "median") for m in ("pca", "tsne", "umap")]
    lists += [f"strategy {s}" for s in strategies]
    pd.DataFrame([{"list": name, "draw": 0, "n": 30, "coverage_radius": rng.uniform(1, 2),
                   "mean_nearest_pick": rng.uniform(0.4, 1), "groups_covered": int(rng.integers(3, 17)),
                   "grain_class_1nn_balanced_accuracy": rng.uniform(0.1, 0.2), "plane_1nn_accuracy": rng.uniform(0.5, 1)}
                  for name in lists]).to_csv(folder / "existing_lists.csv", index=False)
    pd.DataFrame([{"list": name, "draw": 0, "rank": r, "family": f}
                  for name in lists for r, f in enumerate(rng.choice(families, size=ds.N_PICKS, replace=False))]
                 ).to_csv(folder / "picks.csv", index=False)


def krr_outputs(folder, rng):
    """krr_by_plane.csv and krr_regression.csv of krr_classify.py --regression."""
    pd.DataFrame([{"trained_on": trained, "grouping": "distinct", "label": "grain_class", "model": model, "plane": plane,
                   "n": 100, "accuracy": rng.uniform(0.1, 0.7), "chance": 0.25}
                  for plane in ("100", "110", "111") for trained in ("all planes", f"{plane} only")
                  for model in ("krr", "knn")]).to_csv(folder / "krr_by_plane.csv", index=False)
    pd.DataFrame([{"dataset": "all", "grouping": "distinct", "target": target, "scored_on": scope, "n": 100,
                   "krr_r2": rng.uniform(0, 1), "knn_r2": rng.uniform(0, 1), "baseline_r2": 0.0}
                  for target in gl.TARGETS for scope in ("all", "100", "110", "111")]
                 ).to_csv(folder / "krr_regression.csv", index=False)


def krr_summary_outputs(folder, rng, groupings=("family", "distinct")):
    """krr_summary.csv/json, krr_by_plane.csv and krr_predictions.csv of krr_classify.py (all GBs, 4x)."""
    rows = [{"dataset": dataset, "label": label, "grouping": grouping, "n_families": 120,
             "baseline_majority": 0.2, "krr_accuracy": rng.uniform(0.3, 0.9), "knn_accuracy": rng.uniform(0.3, 0.9)}
            for grouping in groupings for dataset in ("all", "100", "110", "111")
            for label in (("plane",) if dataset == "all" else ()) + ("handedness", "grain_class")]
    pd.DataFrame(rows).to_csv(folder / "krr_summary.csv", index=False)
    pd.DataFrame([{"trained_on": trained, "grouping": grouping, "label": "grain_class", "model": model, "plane": plane,
                   "n": 40, "accuracy": rng.uniform(0.1, 0.7), "chance": pds.CHANCE[plane], "majority": 0.3}
                  for grouping in groupings for plane in ("100", "110", "111")
                  for trained in ("all planes", f"{plane} only") for model in ("krr", "knn")]
                 ).to_csv(folder / "krr_by_plane.csv", index=False)
    census = pds.census_classes()
    pair = census.groupby("structure_class")["family"].apply(list)
    pair = next(members for members in pair if len(members) > 1)[:2]
    others = census[census["structure_class"] != census.set_index("family").loc[pair[0], "structure_class"]]
    families = pair + others.drop_duplicates("structure_class")["family"].tolist()[:38]  # no other census copies
    pd.DataFrame([{"dataset": "all", "grouping": grouping, "family": f, "fold": i % 5,  # the pair sits in folds 0 and 1
                   "grain_class_true": "E", "grain_class_krr": "E" if i % 3 else "I"}
                  for grouping in groupings for i, f in enumerate(families)]).to_csv(folder / "krr_predictions.csv",
                                                                                      index=False)
    (folder / "krr_summary.json").write_text(json.dumps({"vector": {"thickness": 4}}))


def scan_outputs(folder, rng):
    """rcut_scan_summary.csv of an 8x scan with interface radii 8 A and r_cut + 2."""
    rows = []
    for r_cut in (4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        for radius in sorted({8.0, r_cut + 2.0}):
            rows.append({"r_cut": r_cut, "interface_radius": radius, "thickness": 8, "n_families": 1541,
                         "n_distinct": int(rng.integers(1100, 1300)), "n_distinct_by_structure": 1112,
                         "acc_plane": rng.uniform(0.9, 1), "baseline_plane": 0.4, "knn_plane": rng.uniform(0.8, 1),
                         "acc_grain_class": rng.uniform(0.4, 0.8), "baseline_grain_class": 0.19,
                         "knn_grain_class": rng.uniform(0.4, 0.8), "acc_grain_class_100": rng.uniform(0.4, 0.8),
                         "acc_grain_class_110": rng.uniform(0.4, 0.8), "acc_grain_class_111": rng.uniform(0.1, 0.3),
                         "grouping": "census"})
    pd.DataFrame(rows).to_csv(folder / "rcut_scan_summary.csv", index=False)


def labels_output(path, families, rng):
    """geometry_labels.csv for the given families at 8x (edge, mid, cell)."""
    rows = []
    for family in families:
        for region in gl.REGIONS:
            rows.append({"family": family, "plane": family[:3], "sym": family[4:], "thickness": 8, "region": region,
                         **{label: rng.uniform(0, 3) for label in gl.LABELS}})
    pd.DataFrame(rows).to_csv(path, index=False)


def representation_outputs(folder, rng, n_gbs=12):
    """phase1/, phase2/ and thickness/ of representation_study.py with every row kind the figures read."""
    reps = list("ABCD")
    pairs = [f"{a}-{b}" for i, a in enumerate(reps) for b in reps[i + 1:]]
    plane = np.repeat(["100", "110", "111"], 2 * n_gbs // 3)
    grain = rng.choice(list(GRAIN_COLORS), size=len(plane))
    for phase, methods in (("phase1", rst.PHASE_METHODS["1"]), ("phase2", rst.PHASE_METHODS["2"][:4] + ("diffusion",))):
        out = folder / phase
        out.mkdir(parents=True)
        rows = []
        def add(**fields):
            rows.append({c: fields.get(c) for c in rst.COLUMNS})
        for subset in rst.SUBSETS:
            for variable in ("knn_plane", "knn_grain_class", "knn_r2_d_min"):
                add(study="representation", phase=phase[-1], kind="baseline", subset=subset, variable=variable, value=0.3)
            for rep in reps:
                for variable in ("knn_plane", "knn_grain_class", "knn_interface", "knn_r2_d_min"):
                    add(kind="raw", method="raw", subset=subset, representation=rep, variable=variable, value=rng.uniform())
            for pair in pairs:
                for variable in ("jaccard15", "jaccard50", "spearman", "cka"):
                    add(kind="raw_pair", method="raw", subset=subset, representation=pair, variable=variable,
                        value=rng.uniform(0, 0.5))
            for method in methods:
                for variable, _ in pds.PAIR_VARIABLES:
                    for pair in pairs:
                        value = rng.uniform(0, 0.6)
                        add(kind="pair", method=method, subset=subset, representation=pair, variable=variable, value=value,
                            noise_floor=0.2, input_change=0.3, sensitivity=value / 0.3, significant=value > 0.2)
                    for factor in ("region", "covariance", "interaction", "residual"):
                        add(kind="effect", method=method, subset=subset, representation="A-D", variable=variable,
                            factor=factor, value=0.25, p_value=0.01)
                for rep in reps:
                    for variable in ("knn_plane", "knn_grain_class", "knn_interface", "knn_grain_class@100", "knn_r2_d_min",
                                     "trustworthiness", "continuity", "triplet_accuracy"):
                        add(kind="map", method=method, subset=subset, representation=rep, variable=variable,
                            value=rng.uniform(), sd=0.02)
                    add(kind="method", method=method, subset=subset, representation=rep, variable="participation_ratio",
                        value=rng.uniform(1, 5))
                for factor in ("region", "covariance", "interaction", "residual"):
                    add(kind="map_effect", method=method, subset=subset, representation="A-D", variable="knn_plane",
                        factor=factor, value=0.25, p_value=0.2)
        if phase == "phase1":
            for block in ("polar", "axial", "tensor", "pseudotensor"):
                for kind in ("pristine", "transformed"):
                    add(kind="bulk", method=block, subset="all", representation=kind,
                        variable="cell_average_over_atom_norm", value=1e-9, main=1e-10, n=10)
            for test, variables in (("D_vs_C", ("relative_residual", "cosine_with_interfaces")),
                                    ("B_vs_A", ("relative_residual", "residual_over_gb_signal"))):
                for variable in variables:
                    for subset in ("all", "100", "110", "111"):
                        add(kind="bulk", method=test, subset=subset, variable=variable, value=0.01, main=0.02,
                            noise_floor=0.015, n=10)
        pd.DataFrame(rows, columns=list(rst.COLUMNS)).to_csv(out / "results.csv", index=False)
        arrays = {"table|plane": plane, "table|grain_class": grain}
        for subset in rst.SUBSETS:
            members = np.arange(len(plane)) if subset == "pooled" else np.where(plane == subset)[0]
            arrays[f"rows|{subset}"] = members
            for method in methods:
                for rep in reps:
                    arrays[f"{method}|{subset}|{rep}"] = rng.normal(size=(len(members), 2))
        np.savez(out / "maps.npz", **arrays)
        (out / "summary.json").write_text("{}")
    out = folder / "thickness"
    out.mkdir()
    rows = [{"kind": "thickness", "method": m, "subset": "pooled", "representation": r, "variable": v, "value": rng.uniform()}
            for m in ("pca", "umap", "raw") for r in reps for v in ("thickness_share", "gb_share", "same_gb_neighbours")]
    rows += [{"kind": "all_structures_pca", "method": "pca", "subset": a, "representation": r, "variable": v,
              "value": rng.uniform(), "n": 500} for a in ("PC1", "PC2", "PC1-10") for r in reps
             for v in ("thickness_share", "gb_share")]
    pd.DataFrame(rows, columns=list(rst.COLUMNS)).to_csv(out / "results.csv", index=False)


def grid_outputs(folder, rng):
    """results.csv of representation_study.py --study settings_grid."""
    tags = ["rc5.0_n8_l4_s0.3", "rc5.0_n8_l8_s0.5", "rc8.0_n8_l4_s0.5"]
    rows = []
    for method in rst.GRID_METHODS + ("raw",):
        for variable, _ in pds.PAIR_VARIABLES:
            for factor in ("representation", "settings", "interaction", "seed_noise" if method != "raw" else "residual"):
                rows.append({"kind": "effect", "method": method, "representation": "all", "variable": variable,
                             "factor": factor, "value": 0.25, "p_value": 0.01})
            for label in ("between_settings", "between_representations", "replicates"):
                rows.append({"kind": "grid_mean", "method": method, "representation": label, "variable": variable,
                             "value": rng.uniform(0, 0.5), "sd": 0.05, "noise_floor": 0.1})
            for rep in "ABCD":
                for tag in tags:
                    rows.append({"kind": "anchor", "method": method, "representation": rep, "variable": variable,
                                 "factor": tag, "value": rng.uniform(0, 0.5)})
    rows[0]["value"] = np.nan  # an undefined share is drawn as n/a
    pd.DataFrame(rows, columns=list(rst.COLUMNS)).to_csv(folder / "results.csv", index=False)
    (folder / "summary.json").write_text(json.dumps({"tags": tags, "settings": {"replicates": 5}}))


class Helpers(unittest.TestCase):
    def test_grid_description(self):
        tags = [f"rc{r}_n8_l{l}_s{s}" for r in ("5.0", "6.0", "8.0") for l in (4, 8) for s in (0.3, 0.5)]
        self.assertEqual(pds._grid_description(tags), "r_cut 5/6/8 x l_max 4/8 x sigma 0.3/0.5")
        self.assertEqual(pds._grid_description(tags[:1]), "r_cut 5 x l_max 4 x sigma 0.3")

    def test_pick_frequency(self):
        counts = pds.pick_frequency([["a", "b", "c"], ["a", "b", "d"], ["a", "e", "e"]])
        # a in all three lists, b in two, c d e in one (a repeat within one list counts once).
        np.testing.assert_array_equal(counts, [3, 1, 1])

    def test_turned_recovers_a_rotated_map_and_keeps_missing_rows(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(40, 2))
        angle = 0.7
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        other = 3.0 * base @ turn + 5.0
        other[0] = np.nan
        first, second = pds._turned(base, other)
        self.assertTrue(np.isnan(second[0]).all())
        np.testing.assert_allclose(first[1:], second[1:], atol=1e-10)

    def test_refusal_keeps_marked_and_archived_figures_out_of_results(self):
        root = pds.sc.PROJECT_ROOT
        real = [root / "calculations" / "ML_dscribe" / "CoGe" / "descriptor_sensitivity"]
        smoke = [root / "archive" / "descriptor_study_20261003" / "smoke" / "descriptor_sensitivity_v2"]
        for out in (pds.mc.plot_root("soap"), root / "results" / "dscribe"):
            self.assertIsNone(pds.refusal(out, "", real))
            self.assertIn("--title-prefix", pds.refusal(out, "SMOKE TEST", real))
            self.assertIn("archive/", pds.refusal(out, "", smoke))
        elsewhere = root / "archive" / "descriptor_study_20261003" / "smoke" / "plots"
        self.assertIsNone(pds.refusal(elsewhere, "SMOKE TEST", smoke))

    def test_covariant_sections_follow_out_dir(self):
        # --descriptor-type covariant: vector_sizes goes to --out-dir unless --covariant-out-dir is given,
        # so a smoke run with its own --out-dir is not refused for the real plots folder
        from unittest import mock
        names = ("TITLE_PREFIX", "ARCHIVE_ROOT", "DESCRIPTOR_TYPE", "KIND")  # globals main() sets
        saved = {name: getattr(pds, name) for name in names}
        self.addCleanup(lambda: [setattr(pds, name, value) for name, value in saved.items()])
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["plot_descriptor_study.py", "--title-prefix", "SMOKE TEST", "--only", "vector_sizes",
                    "--out-dir", tmp, "--dry-run"]
            with mock.patch.object(sys, "argv", argv[:1] + ["--descriptor-type", "covariant"] + argv[1:]), \
                    mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(pds.main(), 0)
            with mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(pds.main(), 2)  # standard SOAP: still the real covariant plots folder

    def test_title_prefix_reaches_every_title(self):
        import matplotlib
        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as tmp:
            pds.TITLE_PREFIX = "SYNTHETIC DATA"
            try:
                plt, fig, _ = pds._figure(1, 1, 4, 3)
                texts = []
                def savefig(target, **kwargs):  # read the texts before _finish closes the figure
                    texts.extend(t.get_text() for t in fig.texts)
                    Path(target).write_bytes(b"")
                fig.savefig = savefig
                pds._finish(plt, fig, Path(tmp) / "a.png", "What does it show?", "note")
            finally:
                pds.TITLE_PREFIX = ""
        self.assertTrue(texts[0].startswith("SYNTHETIC DATA: What does it show?"))

    def test_keep_old_never_overwrites(self):
        with tempfile.TemporaryDirectory(dir=pds.sc.PROJECT_ROOT / "archive") as tmp:
            figure = Path(tmp) / "figure" / "a.png"
            figure.parent.mkdir()
            root = Path(tmp) / "kept"
            for content in (b"first", b"second"):
                figure.write_bytes(content)
                pds.keep_old(figure, root)
            kept = sorted(p.read_bytes() for p in root.rglob("*.png"))
            self.assertEqual(kept, [b"first", b"second"])
            self.assertFalse(figure.exists())

    def test_pool_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.json").write_text('{"pool": "1,111 distinct GBs at 8x"}')
            self.assertEqual(pds.pool_count(Path(tmp)), 1111)
            self.assertIsNone(pds.pool_count(Path(tmp) / "missing"))

    def test_legend_grid_fills_rows_evenly(self):
        self.assertEqual(pds.legend_grid(8, 10.8), (2, 4))  # 7 fit in one row: 2 rows of 4, not 7 + 1
        self.assertEqual(pds.legend_grid(8, 16.2), (1, 8))
        self.assertEqual(pds.legend_grid(7, 5.6), (2, 4))
        self.assertEqual(pds.legend_grid(8, 1.0), (4, 2))  # at least two columns


class Sections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data, cls.benchmark, cls.out = root / "sensitivity", root / "benchmark", root / "plots"
        cls.krr, cls.representation, cls.grid = root / "krr", root / "representation", root / "grid"
        cls.census_krr, cls.scan, cls.missing = root / "census_krr", root / "scan", root / "missing"
        for folder in (cls.data, cls.benchmark, cls.krr, cls.representation, cls.grid, cls.census_krr, cls.scan):
            folder.mkdir()
        sensitivity_outputs(cls.data, np.random.default_rng(1))
        benchmark_outputs(cls.benchmark, np.random.default_rng(2))
        krr_outputs(cls.krr, np.random.default_rng(3))
        krr_summary_outputs(cls.krr, np.random.default_rng(7))
        krr_summary_outputs(cls.census_krr, np.random.default_rng(8), groupings=("family", "distinct", "census"))
        scan_outputs(cls.scan, np.random.default_rng(9))
        representation_outputs(cls.representation, np.random.default_rng(4))
        grid_outputs(cls.grid, np.random.default_rng(5))
        cls.labels = root / "geometry_labels.csv"
        distinct = sorted(set(pds.mc.distinct_families()) - set(pds.mc.NO_BOUNDARY))
        labels_output(cls.labels, distinct[:20] + distinct[-20:], np.random.default_rng(6))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def drawn(self, folder):
        return sorted(p.name for p in (self.out / folder).glob("*.png"))

    def test_descriptor_settings(self):
        pds.descriptor_settings(self.data, self.out)
        self.assertEqual(self.drawn("descriptor_settings"),
                         ["cost.png", "knob_ranking.png", "knob_trends.png", "overview.png", "pca_dimensions.png",
                          "region_and_processing.png", "weighting.png"])

    def test_maps_by_setting(self):
        pds.maps_by_setting(self.data, self.out)
        expected = sorted(f"{method}_{knob}.png" for method in ("pca", "tsne", "umap") for knob in ds.KNOBS)
        self.assertEqual(self.drawn("maps_by_setting"), expected)

    def test_seed_stability(self):
        pds.seed_stability(self.data, self.out)
        self.assertEqual(self.drawn("seed_stability"),
                         ["pick_frequency.png", "seed_vs_setting.png", "tsne_seeds.png", "umap_seeds.png"])

    def test_thickness(self):
        pds.thickness(self.data, self.out)
        self.assertEqual(self.drawn("thickness"), ["pc1_by_thickness.png", "variance_shares.png"])

    def test_selection(self):
        pds.selection(self.benchmark, self.out, self.krr)
        self.assertEqual(self.drawn("selection"),
                         ["coverage.png", "descriptor_efficiency.png", "existing_lists.png", "grain_class_per_plane.png",
                          "learning_curves_free_labels.png", "learning_curves_grain_class.png",
                          "learning_curves_handedness.png", "learning_curves_plane.png", "picks_full_space.png",
                          "picks_map_rules.png"])

    def test_free_labels(self):
        pds.free_labels(self.labels, self.krr, self.out)
        self.assertEqual(self.drawn("free_labels"), ["by_grain_class.png", "edge_vs_mid.png", "krr_r2.png"])

    def test_representation_response(self):
        pds.representation_response(self.representation, self.out)
        methods = rst.PHASE_METHODS["1"] + rst.PHASE_METHODS["2"][:4] + ("diffusion",)
        self.assertEqual(self.drawn("representation_response"), sorted(
            ["bulk_test.png", "pair_changes.png", "sensitivity.png", "sensitivity_by_plane.png", "organisation.png",
             "faithfulness.png", "effects.png", "map_effects.png", "method_numbers.png", "thickness.png"]
            + [f"maps_{m}.png" for m in methods]))

    def test_stage1(self):
        pds.stage1(self.census_krr, self.scan, self.out)
        self.assertEqual(self.drawn("stage1"), ["krr_by_plane.png", "rcut_scan.png"])

    def test_grain_class_overview(self):
        pds.grain_class_overview(self.out, {"standard SOAP": self.krr, "covariant SOAP": self.census_krr,
                                            "ACE": self.missing},
                                 self.representation, {"standard SOAP": self.benchmark})
        self.assertEqual(self.drawn("grain_class_overview"), ["grain_class_overview.png"])
        table = pd.read_csv(self.out / "grain_class_overview" / "grain_class_overview.csv", dtype={"scope": str})
        self.assertEqual(set(table["descriptor"]), {"standard SOAP", "covariant SOAP", "standard SOAP, whole cell",
                                                     "covariant SOAP, whole cell"})
        stage = table[table["evaluation"] == "Stage 1.4 KRR, 5-fold"].set_index("descriptor")
        self.assertEqual(stage.loc["covariant SOAP", "grouping"], "census")
        self.assertEqual(stage.loc["standard SOAP", "grouping"], "distinct")
        alone = table[table["evaluation"].str.contains("no census copy")]
        self.assertEqual(alone["descriptor"].tolist(), ["standard SOAP"])  # census folds need no such row
        self.assertEqual(int(alone["n"].iloc[0]), 38)  # the pair straddles folds 0 and 1
        self.assertTrue((table["source"].str.len() > 0).all())

    def test_covariant_texts(self):
        table = pd.read_csv(self.data / "sensitivity.csv", dtype={"value": str})
        blocks = table[table["knob"] == "region"].head(2).assign(knob="blocks", value=["lambda 1", "lambda 2"])
        table = pd.concat([table, blocks], ignore_index=True)
        pds.DESCRIPTOR_TYPE, pds.KIND = "covariant", "Covariant SOAP"
        try:
            pds.region_and_processing(table, self.out / "covariant" / "region_and_processing.png")
            plt, fig, _ = pds._figure(1, 1, 4, 3)
            texts = []
            def savefig(target, **kwargs):
                texts.extend(t.get_text() for t in fig.texts)
                Path(target).write_bytes(b"")
            fig.savefig = savefig
            pds._finish(plt, fig, self.out / "covariant" / "a.png", "What does it show?", "note")
        finally:
            pds.DESCRIPTOR_TYPE, pds.KIND = "soap", ""
        self.assertTrue((self.out / "covariant" / "region_and_processing.png").is_file())
        self.assertTrue(texts[0].startswith("Covariant SOAP: What does it show?"))

    def test_headline_rows_prefer_census(self):
        table = pd.DataFrame({"grouping": ["family", "distinct", "census"], "value": [1, 2, 3]})
        self.assertEqual(pds.headline_rows(table)["value"].tolist(), [3])
        self.assertEqual(pds.headline_rows(table[table["grouping"] != "census"])["value"].tolist(), [2])
        self.assertIn("census", pds.fold_rule(pds.headline_rows(table)))

    def test_settings_grid(self):
        pds.settings_grid_figures(self.grid, self.out)
        self.assertEqual(self.drawn("settings_grid"),
                         ["anchor_distance.png", "between_settings_and_representations.png", "variance_shares.png"])


class AceSections(unittest.TestCase):
    """--descriptor-type ace (2026-10-13): the nu3 knobs, pow-only weighting and the ACE titles."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data, cls.out = root / "sensitivity", root / "plots"
        cls.data.mkdir()
        sensitivity_outputs(cls.data, np.random.default_rng(11), descriptor_type="ace")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.titles = {}
        self.saved = pds.DESCRIPTOR_TYPE, pds.KIND, pds._finish
        original = pds._finish

        def finish(plt, fig, path, title=None, note=None, rect_bottom=0.0):
            self.titles[Path(path).name] = (title, note)
            original(plt, fig, path, title, note, rect_bottom)

        pds.DESCRIPTOR_TYPE, pds.KIND, pds._finish = "ace", "ACE", finish

    def tearDown(self):
        pds.DESCRIPTOR_TYPE, pds.KIND, pds._finish = self.saved

    def drawn(self, folder):
        return sorted(p.name for p in (self.out / folder).glob("*.png"))

    def test_descriptor_settings(self):
        pds.descriptor_settings(self.data, self.out)
        self.assertEqual(self.drawn("descriptor_settings"),
                         ["cost.png", "knob_ranking.png", "knob_trends.png", "overview.png", "pca_dimensions.png",
                          "region_and_processing.png", "weighting.png"])
        table = pd.read_csv(self.data / "sensitivity.csv", dtype={"value": str})
        self.assertEqual(pds.present_knobs(table), ["r_cut", "n_max", "l_max", "sigma", "n3", "l3"])
        self.assertEqual((pds.reference_value("n3"), pds.reference_value("l3"), pds.reference_value("r_cut")),
                         (4.0, 4.0, 6.0))
        title, note = self.titles["knob_trends.png"]
        self.assertIn("each ACE setting", title)
        self.assertIn("n3 4, l3 4", title)
        self.assertIn("without nu3_odd", note)
        self.assertIn("ACE time is not shown", self.titles["cost.png"][1])
        self.assertIn("each block divided by its length in pristine bulk", self.titles["region_and_processing.png"][1])

    def test_maps_by_setting(self):
        pds.maps_by_setting(self.data, self.out)
        expected = sorted(f"{method}_{knob}.png" for method in ("pca", "tsne", "umap") for knob in ds.ACE_KNOBS)
        self.assertEqual(self.drawn("maps_by_setting"), expected)

    def test_existing_list_with_members_outside_the_pool(self):
        # selection_benchmark.py leaves 111/sym00119 out of the ACE PCA lists: they hold 29 GBs and stay in the figure.
        rng = np.random.default_rng(5)
        names = [f"existing {rule}/{m}.txt" for rule in ("FPS", "median") for m in ("pca", "tsne", "umap")]
        names += ["existing union of the six", "strategy fps", "strategy random"]
        sizes = {"existing FPS/pca.txt": 29, "existing median/pca.txt": 29, "existing union of the six": 160}
        dropped = "structures/CoGeGrainBoundaries/CoGe_111_GBs/Interface/sym00119/8x/POSCAR"
        existing = pd.DataFrame([{"list": name, "draw": 0, "n": sizes.get(name, 30),
                                  "left_out_outside_pool": dropped if sizes.get(name) == 29 else np.nan,
                                  "coverage_radius": rng.uniform(1, 2), "mean_nearest_pick": rng.uniform(0.4, 1),
                                  "groups_covered": int(rng.integers(3, 17)),
                                  "grain_class_1nn_balanced_accuracy": rng.uniform(0.1, 0.2),
                                  "plane_1nn_accuracy": rng.uniform(0.5, 1)} for name in names])
        labels = {}
        original = pds._figure

        def figure(*args, **kwargs):
            plt, fig, axes = original(*args, **kwargs)
            labels["axes"] = axes
            return plt, fig, axes

        pds._figure = figure
        try:
            (self.out / "selection").mkdir(parents=True, exist_ok=True)
            pds.existing_lists(existing, 16, self.out / "selection" / "existing_lists.png", 1110)
        finally:
            pds._figure = original
        ticks = [t.get_text() for t in labels["axes"][0, 0].get_yticklabels()]
        self.assertEqual(len(ticks), 8)  # the six lists and two strategies; not the union
        self.assertIn("existing: FPS/pca.txt (29 GBs)", ticks)
        self.assertIn("existing: FPS/tsne.txt", ticks)
        title, note = self.titles["existing_lists.png"]
        self.assertIn("The union of the six lists (160 GBs) is left out", note)
        self.assertIn("1 structure(s) outside the pool, left out here: 111/sym00119 (no boundary)", note)
        self.assertEqual(pds._family_of(dropped), "111/sym00119")

    def test_sections_and_folders(self):
        import contextlib
        import io

        self.assertEqual(pds.DEFAULT_DIRS["ace"]["out"], pds.mc.plot_root("ace"))
        self.assertEqual(pds.DEFAULT_DIRS["ace"]["sensitivity"], pds.mc.OUTPUT_ROOTS["ace"] / "descriptor_sensitivity")
        self.assertNotIn("vector_sizes", pds.ACE_TYPE_SECTIONS)
        argv = sys.argv
        try:
            sys.argv = ["plot_descriptor_study.py", "--descriptor-type", "ace", "--only", "vector_sizes", "--dry-run"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pds.main()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
