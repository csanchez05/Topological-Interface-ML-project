"""Small numerical fixtures for GB sampling and census alignment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_gbs as sampling


class FarthestPointSampling(unittest.TestCase):
    def test_bulk_seed_then_cover_both_ends_and_middle(self):
        points = np.array([[0, 0], [1, 0], [5, 0], [9, 0], [10, 0]], dtype=float)
        chosen = sampling.farthest_points(points, list("abcde"), count=3)
        np.testing.assert_array_equal(chosen, [4, 0, 2])

    def test_ties_follow_family_name_after_row_permutation(self):
        points = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        names = np.array(["d", "a", "c", "b"])
        first = sampling.farthest_points(points, names, count=4)
        self.assertEqual(names[first].tolist(), ["a", "d", "b", "c"])
        order = [2, 0, 3, 1]
        second = sampling.farthest_points(points[order], names[order], count=4)
        self.assertEqual(names[order][second].tolist(), names[first].tolist())

    def test_coincident_points_do_not_repeat_a_structure(self):
        chosen = sampling.farthest_points(np.zeros((3, 2)), ["c", "a", "b"], count=3)
        np.testing.assert_array_equal(chosen, [1, 2, 0])

    def test_invalid_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            sampling.farthest_points([[np.nan, 0]], ["a"], count=1)
        with self.assertRaises(ValueError):
            sampling.farthest_points([[0, 0]], ["a"], count=2)


class CentroidSampling(unittest.TestCase):
    def test_real_member_nearest_each_center_and_tie(self):
        points = np.array([[0, 0], [2, 0], [9, 0], [10, 0], [13, 0]], dtype=float)
        names = ["z", "a", "b", "c", "d"]
        chosen = sampling.nearest_centroids(points, names, [0, 0, 1, 1, 1], [[1, 0], [10.5, 0]])
        np.testing.assert_array_equal(chosen, [1, 3])

    def test_empty_cluster_is_rejected(self):
        with self.assertRaises(ValueError):
            sampling.nearest_centroids([[0, 0], [2, 0]], ["a", "b"], [0, 0], [[0, 0], [2, 0]])

    def test_fitted_clusters_choose_known_central_structures(self):
        points = np.array([[0, 0], [0, 1], [0, 2], [20, 0], [20, 1], [20, 2]], dtype=float)
        chosen, _, _, _ = sampling.cluster_centers(points, list("abcdef"), seed=42, count=2)
        self.assertEqual(set(chosen), {1, 4})


class CensusAlignment(unittest.TestCase):
    def test_alphabetical_representative_retains_original_vector_row(self):
        index = pd.DataFrame({"family": ["100/z", "110/b", "100/a"], "source": ["z", "b", "a"]})
        census = pd.DataFrame({"family": ["100/a", "100/z", "110/b"],
                               "structure_class": ["S1", "S1", "S2"]})
        positions, chosen = sampling.representative_rows(index, census)
        np.testing.assert_array_equal(positions, [2, 1])
        self.assertEqual(chosen["source"].tolist(), ["a", "b"])

    def test_mismatched_census_is_rejected(self):
        with self.assertRaises(ValueError):
            sampling.representative_rows(pd.DataFrame({"family": ["a"]}),
                                         pd.DataFrame({"family": ["b"], "structure_class": ["S0"]}))


class GrainClassAlignment(unittest.TestCase):
    def test_labels_follow_family_when_label_rows_are_shuffled(self):
        index = pd.DataFrame({"family": ["110/b", "100/a"], "source": ["b/8x/POSCAR", "a/8x/POSCAR"]})
        labels = pd.DataFrame({"family": ["100/a", "110/b"], "grain_class": ["E", "I"]})
        joined = sampling.attach_grain_classes(index, labels)
        self.assertEqual(joined["family"].tolist(), index["family"].tolist())
        self.assertEqual(joined["grain_class"].tolist(), ["I", "E"])
        self.assertEqual(joined["source"].tolist(), index["source"].tolist())

    def test_missing_and_unknown_classes_are_rejected(self):
        index = pd.DataFrame({"family": ["a"]})
        with self.assertRaises(ValueError):
            sampling.attach_grain_classes(index, pd.DataFrame({"family": ["b"], "grain_class": ["E"]}))
        with self.assertRaises(ValueError):
            sampling.attach_grain_classes(index, pd.DataFrame({"family": ["a"], "grain_class": ["unknown"]}))


class SubsetFilter(unittest.TestCase):
    # S1 = two [100] families, S2 = one [100] family, S3 = one [110] family
    census = pd.DataFrame({"family": ["100/a", "100/b", "100/c", "110/d"], "plane": ["100", "100", "100", "110"],
                           "sym": ["a", "b", "c", "d"], "structure_class": ["S1", "S1", "S2", "S3"]})
    index = census.drop_duplicates("structure_class").reset_index(drop=True)

    @staticmethod
    def survey(rows):
        return pd.DataFrame([{"plane": plane, "sym": sym, "thickness": thickness, "status": "ok", "dmin_A": dmin}
                             for plane, sym, thickness, dmin in rows], dtype=str)

    def test_keeps_classes_of_the_plane_at_or_above_the_limit(self):
        # c is clean at 4x but not at 8x; d is another plane and has no survey row
        survey = self.survey([("100", "a", "8x", "1.900"), ("100", "b", "8x", "1.800"),
                              ("100", "c", "8x", "1.700"), ("100", "c", "4x", "2.000")])
        mask, info = sampling.overlap_filter(self.index, self.census, survey, ["100"], 1.8)
        np.testing.assert_array_equal(mask, [True, False, False])
        self.assertEqual((info["families"], info["distinct_gbs"], info["planes"]), (2, 1, ["100"]))

    def test_class_straddling_the_limit_stops(self):
        survey = self.survey([("100", "a", "8x", "1.900"), ("100", "b", "8x", "1.799"), ("100", "c", "8x", "2.0")])
        with self.assertRaisesRegex(ValueError, "straddling"):
            sampling.overlap_filter(self.index, self.census, survey, ["100"], 1.8)

    def test_missing_or_unreadable_value_in_a_selected_plane_stops(self):
        survey = self.survey([("100", "a", "8x", "1.900"), ("100", "b", "8x", "1.900")])
        with self.assertRaisesRegex(ValueError, "100/c"):
            sampling.overlap_filter(self.index, self.census, survey, ["100"], 1.8)
        survey.loc[len(survey)] = {"plane": "100", "sym": "c", "thickness": "8x", "status": "empty", "dmin_A": np.nan}
        with self.assertRaisesRegex(ValueError, "empty"):
            sampling.overlap_filter(self.index, self.census, survey, ["100"], 1.8)

    def test_plane_filter_without_a_limit_needs_no_survey(self):
        mask, info = sampling.overlap_filter(self.index, self.census, None, ["110"], None)
        np.testing.assert_array_equal(mask, [False, False, True])
        self.assertEqual((info["families"], info["distinct_gbs"]), (1, 1))

    def test_class_spanning_planes_stops(self):
        census = self.census.assign(structure_class=["S1", "S1", "S2", "S2"])
        with self.assertRaisesRegex(ValueError, "spanning planes"):
            sampling.overlap_filter(census.drop_duplicates("structure_class"), census, None, ["100"], None)


class SubsetOutputs(unittest.TestCase):
    def test_tags_name_every_departure_from_the_stage2_default(self):
        self.assertIsNone(sampling.selection_tag(["100", "110", "111"], None, 30))
        self.assertEqual(sampling.selection_tag(["100"], 1.8, 30), "plane100_dmin1.8")
        self.assertEqual(sampling.selection_tag(["110", "100"], None, 12), "plane100-110_n12")
        self.assertEqual(sampling.selection_tag(["111", "110", "100"], 2.0, 30), "allplanes_dmin2")

    def test_existing_selection_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            plots, out = Path(tmp) / "plots", Path(tmp) / "out"
            self.assertEqual(sampling.existing_outputs(plots, out), [])
            (plots / "FPS").mkdir(parents=True)
            (plots / "FPS" / "umap.txt").write_text("x\n")
            out.mkdir()
            (out / "run.json").write_text("{}")
            self.assertEqual(sorted(p.name for p in sampling.existing_outputs(plots, out)), ["run.json", "umap.txt"])

    def test_union_list_is_sorted_unique_with_header(self):
        self.assertEqual(sampling.sample_list_text(["b/POSCAR", "a/POSCAR", "b/POSCAR"]),
                         "Structure\na/POSCAR\nb/POSCAR\n")

    def test_map_legend_names_only_present_classes_and_keeps_the_title(self):
        from PIL import Image
        coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.png"
            sampling.draw_map(path, coordinates, ["E", "I", "E"], [0, 2], axis_labels=("x", "y"),
                              selected_background="grain_class", title="[100] GBs (3 distinct)")
            with Image.open(path) as img:
                self.assertEqual(json.loads(img.info["Legend"]), ["Selected (2)", "E", "I", "Pristine bulk"])
                self.assertEqual(img.info["Title"], "[100] GBs (3 distinct)")
                self.assertEqual(img.size, (1600, 1200))


if __name__ == "__main__":
    unittest.main()
