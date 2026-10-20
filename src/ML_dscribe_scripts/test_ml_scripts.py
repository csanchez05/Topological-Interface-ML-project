"""Synthetic tests for the step 1.4 / 1.5 scripts. Run from the project root with the mlenv environment:

    python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_*.py'
"""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import compare_duplicates
import find_duplicates
import gb_labels
import krr_classify
import ml_common as mc
import pooled_duplicates
import rcut_scan_all
import rcut_scan_summary
import structure_match as sm

INFORMATION = """Information for Test

Number of Applied Symmetry Operations: 8
Number of Unique Simple Interfaces: 2

Number of Symmetry Operations that Create Each Interface

0: 2 symmetry operations, x distance = 0.035, total distance = 1.33
1: 1 symmetry operations, x distance = 0.028, total distance = 1.42

Symmetry Operations that Create Each Interface & Distances


Interface 0 : Num of Syms = 2
x distance = 0.035
total distance = 1.33
-----------------------------------
Inversion : -x, -y, -z [[0 0 0]]
[[-1.  0.  0.  0.]
 [ 0. -1.  0.  0.]
 [ 0.  0. -1.  0.]]

-3+ [111] : -z -x, -y [[0.  0.5 0.5]]
[[ 0.   0.  -1.   0. ]
 [-1.   0.   0.   0.5]
 [ 0.  -1.   0.   0.5]]



Interface 1 : Num of Syms = 1
x distance = 0.028
total distance = 1.42
-----------------------------------
4+ [001] : -y, x, z [[0.33333333 0.66666667 0.        ]]
[[ 0.         -1.          0.          0.33333333]
 [ 1.          0.          0.          0.66666667]
 [ 0.          0.          1.          0.        ]]
"""


def cubic_group():
    """The 48 signed permutation matrices."""
    group = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            group.append(matrix)
    return group


class Labels(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Information.txt"
            path.write_text(text)
            return gb_labels.parse_information(path)

    def test_parse_information(self):
        interfaces, header = self.parse(INFORMATION)
        self.assertEqual([entry["number"] for entry in interfaces], [0, 1])
        self.assertEqual([len(entry["operations"]) for entry in interfaces], [2, 1])
        self.assertEqual(header["Number of Unique Simple Interfaces"], 2)
        self.assertAlmostEqual(interfaces[1]["total_distance"], 1.42)
        second = interfaces[0]["operations"][1]
        self.assertEqual(second["name"], "-3+ [111]")
        np.testing.assert_allclose(second["matrix"][:, 3], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(interfaces[1]["operations"][0]["matrix"][0], [0, -1, 0, 1 / 3], atol=1e-8)

    def test_parse_rejects_a_wrong_count(self):
        with self.assertRaises(ValueError):
            self.parse(INFORMATION.replace("Interface 1 : Num of Syms = 1", "Interface 1 : Num of Syms = 2"))
        with self.assertRaises(ValueError):
            self.parse(INFORMATION.replace("Number of Unique Simple Interfaces: 2",
                                           "Number of Unique Simple Interfaces: 3"))

    def test_crystal_rotations_form_a_group_of_12(self):
        group = gb_labels.crystal_rotations()
        self.assertEqual(len(group), 12)
        for a in group:
            self.assertAlmostEqual(np.linalg.det(a), 1.0)
            for b in group:
                self.assertTrue(any(np.allclose(a @ b, c) for c in group))

    def test_rotation_names(self):
        self.assertEqual(gb_labels.rotation_name(np.eye(3)), "E")
        self.assertEqual(gb_labels.rotation_name(-np.eye(3)), "I")
        self.assertEqual(gb_labels.rotation_name(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])), "R120[111]")
        self.assertEqual(gb_labels.rotation_name(np.diag([1.0, 1.0, -1.0])), "I.R180[001]")
        self.assertEqual(gb_labels.rotation_name(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])), "R90[001]")

    def test_48_cubic_operations_give_four_grain_classes(self):
        classes = [gb_labels.grain_class(matrix) for matrix in cubic_group()]
        counts = {name: classes.count(name) for name in set(classes)}
        self.assertEqual(counts, {"E": 12, "I": 12, "R90[100]": 12, "I.R90[100]": 12})

    def test_grain_class_ignores_a_crystal_rotation(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        for symmetry in gb_labels.crystal_rotations():
            self.assertEqual(gb_labels.grain_class(rotation @ symmetry), gb_labels.grain_class(rotation))

    def test_to_cubic(self):
        # "-y, -x, -z" in the 110 cell is the two-fold axis along cubic x.
        swap = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])
        self.assertEqual(gb_labels.rotation_name(gb_labels.to_cubic(swap, "110")), "R180[100]")
        # A two-fold axis along the 111 cell's x is a 180 degree turn about cubic [111]: not a cubic operation.
        turn = np.diag([1.0, -1.0, -1.0])
        cubic = gb_labels.to_cubic(turn, "111")
        np.testing.assert_allclose(cubic @ cubic.T, np.eye(3), atol=1e-12)
        self.assertEqual(gb_labels.rotation_name(cubic), "R180[111]")
        self.assertEqual(gb_labels.grain_class(cubic), "R60[111]")
        np.testing.assert_allclose(gb_labels.to_cubic(swap, "100"), swap)

    def test_same_periodic_set(self):
        frac = np.array([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
        species = np.array(["Co", "Ge"])
        self.assertTrue(gb_labels.same_periodic_set(frac, species, frac[::-1] + [1.0, 0.0, -1.0], species[::-1]))
        self.assertFalse(gb_labels.same_periodic_set(frac, species, frac, species[::-1]))
        self.assertFalse(gb_labels.same_periodic_set(frac, species, frac + [0.01, 0, 0], species))


class Common(unittest.TestCase):
    def test_vector_tag(self):
        self.assertEqual(mc.vector_tag("interface", 8.0, "both", 4), "interface_r8_both_t4")
        self.assertEqual(mc.vector_tag("nonbulk", 8.0, "mid", 5), "nonbulk_mid_t5")
        self.assertEqual(mc.vector_tag("global", 8.0, "both", 4), "global_t4")

    def test_relative_distances_resolve_tiny_differences(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=50) + 10.0
        X = np.array([base, base, base * (1 + 1e-9), base + rng.normal(size=50)])
        D = mc.relative_distances(X)
        self.assertEqual(D[0, 1], 0.0)
        self.assertAlmostEqual(D[0, 2] / 1e-9, 1.0, places=3)
        self.assertAlmostEqual(D[0, 3], mc.relative_distance(X[0], X[3]), places=10)
        np.testing.assert_allclose(D, D.T)

    def test_coincidence_groups_follow_chains(self):
        D = np.array([[0.0, 0.5, 0.5, 0.5],
                      [0.5, 0.0, 0.01, 0.5],
                      [0.5, 0.01, 0.0, 0.01],
                      [0.5, 0.5, 0.01, 0.0]])
        groups = mc.coincidence_groups(D, 0.02)
        self.assertEqual(groups.tolist(), [0, 1, 1, 1])
        np.testing.assert_allclose(mc.group_diameters(D, groups), [0.0, 0.5])
        self.assertEqual(mc.coincidence_groups(D, 0.001).tolist(), [0, 1, 2, 3])

    def test_nearest_other_skips_the_own_group(self):
        D = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.2], [0.3, 0.2, 0.0]])
        nearest, distance = mc.nearest_other(D, exclude=np.array([0, 0, 1]))
        self.assertEqual(nearest.tolist(), [2, 2, 1])
        np.testing.assert_allclose(distance, [0.3, 0.2, 0.2])

    def test_select_thickness_and_noise(self):
        import pandas as pd

        index = pd.DataFrame({"family": ["100/a", "100/a", "100/b", "100/b", "100/c"],
                              "plane": ["100"] * 5, "thickness": [4, 5, 4, 5, 5], "Lx": [40.0, 50.0, 40.0, 50.0, 50.0]})
        X = np.array([[1.0, 0.0], [1.0, 1e-5], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        chosen, table, notes = mc.select_thickness(X, index, 4, "interface", 8.0)
        self.assertEqual(table["family"].tolist(), ["100/a", "100/b"])
        self.assertEqual(chosen.dtype, np.float64)
        self.assertEqual(notes["families_without_this_thickness"], ["100/c"])
        noise = mc.thickness_noise(X, index, 4, "interface", 8.0)
        self.assertEqual(noise["n_comparisons"], 2)
        self.assertAlmostEqual(noise["max"], 1e-5, places=7)
        # A radius above Lx / 4 leaves nothing to compare.
        self.assertIsNone(mc.thickness_noise(X, index, 4, "interface", 11.0))
        self.assertIn("families_where_radius_exceeds_quarter_cell",
                      mc.select_thickness(X, index, 4, "interface", 11.0)[2])


def random_cell(seed, lengths=(30.0, 5.0, 7.0), n=24):
    from ase import Atoms

    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1, size=(n, 3)) * lengths
    return Atoms(["Co"] * (n // 2) + ["Ge"] * (n - n // 2), positions=positions, cell=lengths, pbc=True)


def moved(atoms, operation, shift):
    from ase import Atoms

    positions = atoms.get_positions() @ np.asarray(operation, dtype=float).T + shift
    copy = Atoms(atoms.get_chemical_symbols(), positions=positions, cell=atoms.cell, pbc=True)
    copy.wrap()
    return copy[np.random.default_rng(1).permutation(len(copy))]


class StructureMatch(unittest.TestCase):
    def test_point_operations(self):
        operations = sm.point_operations()
        self.assertEqual(len(operations), 32)
        for operation in operations:
            np.testing.assert_allclose(operation @ operation.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(abs(operation[0, 0]), 1.0)

    def test_shifted_mirrored_and_inverted_copies_match(self):
        atoms = random_cell(0)
        a = sm.Structure(atoms)
        for operation, proper in ((np.eye(3), True), (np.diag([-1.0, 1.0, 1.0]), False),
                                  (-np.eye(3), False), (np.diag([1.0, -1.0, -1.0]), True)):
            result = sm.match_cells(a, sm.Structure(moved(atoms, operation, [1.3, 0.4, -2.2])))
            self.assertIsNotNone(result)
            self.assertEqual(result["proper"], proper)

    def test_quarter_turn_in_a_square_cell(self):
        atoms = random_cell(1, lengths=(30.0, 6.0, 6.0))
        turn = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
        self.assertIsNotNone(sm.match_cells(sm.Structure(atoms), sm.Structure(moved(atoms, turn, [0.0, 1.0, 2.0]))))
        # The same turn does not map a rectangular cell onto itself.
        other = random_cell(1, lengths=(30.0, 5.0, 7.0))
        rotated = other.copy()
        rotated.set_positions(other.get_positions() @ turn.T)
        rotated.wrap()
        self.assertIsNone(sm.match_cells(sm.Structure(other), sm.Structure(rotated)))

    def test_third_turn_in_a_hexagonal_lattice(self):
        from ase import Atoms

        a = 4.0
        lengths = np.array([30.0, a, a * np.sqrt(3.0)])
        rng = np.random.default_rng(2)
        motif = rng.uniform(0, 1, size=(6, 3)) * [30.0, a, a]
        symbols = ["Co", "Co", "Co", "Ge", "Ge", "Ge"]
        centring = np.array([0.0, a / 2.0, a * np.sqrt(3.0) / 2.0])

        def build(points):
            cell = Atoms(symbols * 2, positions=np.vstack([points, points + centring]), cell=lengths, pbc=True)
            cell.wrap()
            return cell

        c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        original = sm.Structure(build(motif))
        self.assertIsNotNone(sm.match_cells(original, sm.Structure(build(motif @ turn.T + [0.0, 0.7, 0.2]))))
        # Without the centring the rotated motif is a different structure.
        quarter = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
        self.assertIsNone(sm.match_cells(original, sm.Structure(build(motif @ quarter.T))))

    def test_different_structures_do_not_match(self):
        atoms = random_cell(3)
        self.assertIsNone(sm.match_cells(sm.Structure(atoms), sm.Structure(random_cell(4))))
        nudged = atoms.copy()
        nudged.positions[5] += [0.3, 0.0, 0.0]
        self.assertIsNone(sm.match_cells(sm.Structure(atoms), sm.Structure(nudged)))
        within = atoms.copy()
        within.positions[5] += [0.005, 0.0, 0.0]
        self.assertIsNotNone(sm.match_cells(sm.Structure(atoms), sm.Structure(within)))

    def test_interface_slabs(self):
        # Two cells that agree near the middle plane (x = 15) and differ near the edge plane (x = 0).
        first = random_cell(5, n=40)
        second = first.copy()
        far = np.flatnonzero(np.abs(first.positions[:, 0] - 15.0) > 9.0)
        self.assertGreater(far.size, 0)
        second.positions[far, 1] = (second.positions[far, 1] + 1.1) % 5.0
        a, b = sm.Structure(first), sm.Structure(second)
        self.assertIsNone(sm.match_cells(a, b))
        self.assertIsNotNone(sm.match_interfaces(a, "mid", b, "mid", halfwidth=8.0))
        self.assertIsNone(sm.match_interfaces(a, "edge", b, "edge", halfwidth=8.0))
        self.assertIsNone(sm.match_interfaces(a, "mid", b, "edge", halfwidth=8.0))
        result = sm.compare_group([a, b, sm.Structure(moved(first, np.diag([-1.0, 1.0, 1.0]), [0.0, 2.0, 3.0]))],
                                  halfwidth=8.0)
        self.assertEqual(result["cell_class"], [0, 1, 0])
        self.assertEqual(result["mid_interface"][0], result["mid_interface"][1])
        self.assertNotEqual(result["edge_interface"][0], result["edge_interface"][1])

    def test_swapped_interfaces_are_recognised(self):
        # Shifting by half a cell along x exchanges the edge and the middle interface.
        atoms = random_cell(6, n=40)
        a = sm.Structure(atoms)
        b = sm.Structure(moved(atoms, np.eye(3), [15.0, 0.0, 0.0]))
        self.assertIsNotNone(sm.match_interfaces(a, "edge", b, "mid", halfwidth=6.0))
        self.assertIsNone(sm.match_interfaces(a, "edge", b, "edge", halfwidth=6.0))
        result = sm.compare_group([a, b], halfwidth=6.0)
        self.assertEqual(result["cell_class"], [0, 0])
        self.assertEqual(result["edge_interface"], result["mid_interface"][::-1])

    def test_classify(self):
        items = [1, 2, 11, 12, 21]
        self.assertEqual(sm.classify(items, lambda i, j: items[i] // 10 == items[j] // 10), [0, 0, 1, 1, 2])


class Duplicates(unittest.TestCase):
    def test_group_verdict(self):
        same = {"cell_class": [0, 0], "edge_interface": ["a", "b"], "mid_interface": ["b", "a"]}
        self.assertEqual(find_duplicates.group_verdict(same, "interface", "both"), "same cell")
        swapped = {"cell_class": [0, 1], "edge_interface": ["a", "b"], "mid_interface": ["b", "a"]}
        self.assertEqual(find_duplicates.group_verdict(swapped, "interface", "both"), "same interfaces")
        self.assertEqual(find_duplicates.group_verdict(swapped, "interface", "edge"), "different")
        self.assertEqual(find_duplicates.group_verdict(swapped, "global", "both"), "different")
        other = {"cell_class": [0, 1], "edge_interface": ["a", "a"], "mid_interface": ["b", "c"]}
        self.assertEqual(find_duplicates.group_verdict(other, "interface", "both"), "different")
        self.assertEqual(find_duplicates.group_verdict(other, "interface", "edge"), "same interfaces")

    def test_partition_differences(self):
        first = np.array([0, 0, 0, 1, 2])
        second = np.array([0, 0, 1, 1, 2])
        self.assertEqual(find_duplicates.partition_differences(first, second), (2, 1))
        self.assertEqual(find_duplicates.partition_differences(first, first), (0, 0))

    def test_compare_partitions(self):
        def groups(*sets):
            return {family: frozenset(group) for group in sets for family in group}

        first = groups({"a", "b"}, {"c", "d", "e"}, {"f"}, {"g"}, {"h", "i"})
        second = groups({"a", "b"}, {"c", "d"}, {"e"}, {"f", "g"}, {"h"}, {"i"})
        rows = {row["families"]: row for row in compare_duplicates.compare_partitions(first, second)}
        self.assertEqual(rows["a|b"]["status"], "unchanged")
        self.assertEqual(rows["c|d|e"]["status"], "split")
        self.assertEqual(rows["c|d|e"]["n_pieces_in_second"], 2)
        self.assertEqual(rows["h|i"]["status"], "split")
        self.assertEqual(rows["f|g"]["status"], "merged")
        self.assertEqual(compare_duplicates.count_pairs(first), 1 + 3 + 1)


class Classification(unittest.TestCase):
    def blobs(self, n_per_class=30, n_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(n_classes, 6)) * 4.0
        X = np.vstack([centre + rng.normal(size=(n_per_class, 6)) for centre in centres])
        return X, np.repeat(np.arange(n_classes), n_per_class)

    def run_cv(self, X, targets, groups):
        return krr_classify.nested_cv(X, targets, groups, 4, 3, [0.1, 1.0, 10.0], [1e-4, 1e-2], ["standard", "none"], 0)

    def test_separable_classes_are_learned(self):
        X, y = self.blobs()
        coarse = (y > 0).astype(int)
        predictions, fold, chosen = self.run_cv(X, {"fine": y, "coarse": coarse}, np.arange(len(y)))
        self.assertTrue((fold >= 0).all())
        self.assertEqual(len(chosen), 4)
        self.assertGreater((predictions["krr"]["fine"] == y).mean(), 0.95)
        self.assertGreater((predictions["knn"]["coarse"] == coarse).mean(), 0.95)
        self.assertLess((predictions["baseline"]["fine"] == y).mean(), 0.5)

    def test_random_labels_are_not_learned(self):
        X, _ = self.blobs()
        y = np.random.default_rng(5).integers(0, 3, size=len(X))
        predictions, _, _ = self.run_cv(X, {"noise": y}, np.arange(len(y)))
        self.assertLess((predictions["krr"]["noise"] == y).mean(), 0.55)

    def test_folds_in_a_process_pool_give_the_same_result(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        X, y = self.blobs(n_per_class=20)
        targets, groups = {"fine": y, "coarse": (y > 0).astype(int)}, np.arange(len(y)) // 2
        serial = self.run_cv(X, targets, groups)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            # Two cross-validations started before either is collected, as krr_classify.py --jobs does.
            first = krr_classify.start_nested_cv(X, targets, groups, 4, 3, [0.1, 1.0, 10.0], [1e-4, 1e-2],
                                                 ["standard", "none"], 0, pool)
            second = krr_classify.start_nested_cv(X, {"fine": y}, groups, 4, 3, [1.0], [1e-2], ["none"], 0, pool)
            pooled, alone = first(), second()
        for model in serial[0]:
            for name in targets:
                np.testing.assert_array_equal(pooled[0][model][name], serial[0][model][name])
        np.testing.assert_array_equal(pooled[1], serial[1])
        self.assertEqual(pooled[2], serial[2])
        self.assertTrue((alone[0]["krr"]["fine"] >= 0).all())

    def test_kernel_matrix_matches_the_plain_formula(self):
        D2 = np.random.default_rng(2).random((7, 7))
        np.testing.assert_array_equal(krr_classify.kernel_matrix(D2, 10.0, 0.37), np.exp(-10.0 * D2 / 0.37))
        rows = np.array([0, 2, 3, 6])
        values = D2[np.ix_(rows, rows)][np.triu_indices(4, 1)]
        self.assertEqual(krr_classify.median_squared_distance(D2, rows), float(np.median(values)))

    def test_groups_stay_together(self):
        groups = np.repeat(np.arange(20), 3)
        for train, test in krr_classify.group_folds(groups, 5, 0):
            self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(len(krr_classify.group_folds(np.array([0, 0, 1, 1, 2, 2]), 5, 0)), 3)

    def test_coinciding_vectors_with_different_labels_cap_the_accuracy(self):
        X, y = self.blobs(n_per_class=20, n_classes=2)
        X = np.vstack([X, X])                         # every vector appears twice ...
        y = np.concatenate([y, 1 - y])                # ... with opposite labels
        twin = np.tile(np.arange(40), 2)
        self.assertAlmostEqual(krr_classify.label_ceiling(y, twin), 0.5)
        self.assertAlmostEqual(krr_classify.label_ceiling(np.concatenate([y[:40], y[:40]]), twin), 1.0)
        # Split by family, the twin with the other label sits in training: the model is led astray.
        by_family, _, _ = self.run_cv(X, {"label": y}, np.arange(80))
        self.assertLess((by_family["krr"]["label"] == y).mean(), 0.5)

    def test_pick_class_ignores_absent_classes(self):
        scores = np.array([[0.1, 0.9, 0.3], [-0.5, 0.0, -0.2]])
        self.assertEqual(krr_classify.pick_class(scores, np.array([True, False, True])).tolist(), [2, 2])

    def test_knn_ties_go_to_the_nearest(self):
        distances = np.array([[0.1, 0.2, 0.3, 0.9]])
        self.assertEqual(krr_classify.knn_predict(distances, np.array([2, 1, 1, 0]), 1).tolist(), [2])
        self.assertEqual(krr_classify.knn_predict(distances, np.array([2, 1, 1, 0]), 3).tolist(), [1])
        self.assertEqual(krr_classify.knn_predict(distances, np.array([2, 1, 0, 0]), 2).tolist(), [2])

    def test_scaling_uses_training_rows_only(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 100.0]])
        D2 = krr_classify.scaled_distances(X, np.array([0, 1]), "standard")
        # Feature 2 has no spread in the training rows, so it is left unscaled.
        self.assertAlmostEqual(D2[0, 1], 4.0)
        self.assertAlmostEqual(D2[0, 2], 100.0 ** 2)


class PooledThicknesses(unittest.TestCase):
    """Everything that lets the nine thicknesses of a GB be analysed together."""

    def test_thickness_argument(self):
        self.assertEqual(mc.thickness_argument("7"), 7)
        self.assertEqual(mc.thickness_argument("all"), "all")
        self.assertEqual(mc.vector_tag("interface", 8.0, "both", "all"), "interface_r8_both_tall")

    def test_select_all_keeps_every_finite_row(self):
        import pandas as pd
        X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
        index = pd.DataFrame({"family": ["100/a", "100/a", "100/b", "100/b"], "plane": ["100"] * 4,
                              "thickness": [4, 5, 4, 5], "Lx": [40.0, 50.0, 40.0, 50.0]})
        chosen, table, notes = mc.select_thickness(X, index, "all", "interface", 8.0)
        self.assertEqual(table["family"].tolist(), ["100/a", "100/a", "100/b"])
        self.assertEqual(chosen.dtype, np.float64)
        self.assertEqual(notes["families_with_undefined_vector"], ["100/b"])

    def test_family_groups_join_families_and_coinciding_vectors(self):
        family = np.array([0, 0, 1, 1, 2, 2, 3])
        # Family 0 and family 1 share a vector at one thickness; family 2 is split by thickness.
        vectors = np.array([0, 1, 1, 2, 3, 4, 5])
        self.assertEqual(mc.family_groups(family, vectors).tolist(), [0, 0, 0, 0, 1, 1, 2])

    def test_family_groups_with_one_row_per_family_are_the_vector_groups(self):
        vectors = np.array([0, 1, 0, 2, 1])
        self.assertEqual(mc.family_groups(np.arange(5), vectors).tolist(), vectors.tolist())

    def test_nearest_rows_matches_a_full_stable_sort(self):
        rng = np.random.default_rng(0)
        D = rng.integers(0, 6, size=(40, 25)).astype(float)   # many exact ties
        full = np.argsort(D, axis=1, kind="stable")
        for k in (1, 3, 5, 25, 40):
            np.testing.assert_array_equal(krr_classify.nearest_rows(D, k), full[:, :min(k, 25)])
        codes = rng.integers(0, 3, size=25)
        for k in (1, 3, 5):
            np.testing.assert_array_equal(krr_classify.knn_predict(D, codes, k),
                                          krr_classify.knn_vote(full, codes, k))

    def test_pooled_analysis(self):
        # Three families at two thicknesses. a and b are one structure; c is another.
        base = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "c": [0.0, 1.0, 0.0]}
        rows, family, thickness = [], [], []
        for code, name in enumerate("abc"):
            for t in (4, 5):
                vector = np.array(base[name])
                if name == "c" and t == 5:
                    vector = np.array([0.0, 0.9, 0.3])       # c is not thickness-consistent
                rows.append(vector + 1e-6 * t)
                family.append(code)
                thickness.append(t)
        result = pooled_duplicates.analyse(np.array(rows), np.array(family), np.array(thickness),
                                           np.array([0, 0, 1]), 1e-4)
        self.assertEqual(result["groups_per_family"].tolist(), [1, 1, 2])
        self.assertEqual(len(set(result["merged_group"].tolist())), 2)
        self.assertEqual(result["merges"], {})
        self.assertGreater(result["spread"][2], 0.1)
        self.assertEqual([entry["n_distinct_vectors"] for entry in result["per_thickness"]], [2, 2])
        self.assertLess(result["largest_same_structure"], result["smallest_different_structure"] + 1.0)

    def test_pooled_analysis_reports_merged_structures(self):
        X = np.array([[1.0, 0.0], [1.0, 1e-6], [0.0, 1.0]])
        result = pooled_duplicates.analyse(X, np.array([0, 1, 2]), np.array([4, 4, 4]), np.array([0, 1, 2]), 1e-4)
        self.assertEqual(list(result["merges"]), [(0, 1)])
        self.assertEqual(result["merges"][(0, 1)]["thicknesses"], {(4, 4)})

    def test_group_folds_refuse_a_split_group(self):
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        for train, test in krr_classify.group_folds(groups, 4, 0):
            self.assertFalse(set(groups[train]) & set(groups[test]))


class ScanDriver(unittest.TestCase):
    def tasks(self, folder, outcomes):
        """Three tasks: b needs a, c is independent. outcomes: name -> exit status."""
        made = {}
        for name, needs, cpus in (("a", [], 2), ("b", ["a"], 1), ("c", [], 1)):
            marker = Path(folder) / f"{name}.done"
            code = f"import pathlib, sys; pathlib.Path({str(marker)!r}).write_text('x'); sys.exit({outcomes[name]})"
            made[name] = rcut_scan_all.Task(name=name, kind="thickness", cpus=cpus, needs=needs,
                                            commands=[([sys.executable, "-c", code], False)], outputs=[marker])
        return list(made.values())

    def test_all_tasks_run_in_dependency_order(self):
        with tempfile.TemporaryDirectory() as folder:
            tasks = rcut_scan_all.run(self.tasks(folder, {"a": 0, "b": 0, "c": 0}), 2, Path(folder),
                                      Path(folder) / "status.json")
            self.assertEqual([task.state for task in tasks], ["ok", "ok", "ok"])
            self.assertTrue((Path(folder) / "status.json").is_file())

    def test_failure_skips_only_dependents(self):
        with tempfile.TemporaryDirectory() as folder:
            tasks = rcut_scan_all.run(self.tasks(folder, {"a": 3, "b": 0, "c": 0}), 2, Path(folder),
                                      Path(folder) / "status.json")
            self.assertEqual([task.state for task in tasks], ["failed", "skipped", "ok"])
            self.assertFalse((Path(folder) / "b.done").exists())



class CensusGrouping(unittest.TestCase):
    def test_census_keeps_turned_copies_together(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "distinct_gbs.csv"
            pd.DataFrame({"family": ["a", "b", "c", "d", "e"],
                          "structure_class": ["S0", "S0", "S1", "S2", "S3"]}).to_csv(path, index=False)
            # a, b: one structure, different vectors (turned copies); c, d: coinciding vectors; f: not in the census
            groups = krr_classify.census_groups(np.array(["a", "b", "c", "d", "e", "f"]), np.array([0, 1, 2, 2, 3, 4]),
                                                path)
        self.assertEqual(groups.tolist(), [0, 0, 1, 1, 2, 3])


class ScanSummaryTable(unittest.TestCase):
    def test_missing_values_print_as_a_dash(self):
        # covariant SOAP: no coinciding pair, so largest_coinciding_distance is None
        self.assertEqual(rcut_scan_summary.cell(None, 6, ".2e"), "     -")
        self.assertEqual(rcut_scan_summary.cell("", 4, ".3f"), "   -")
        self.assertEqual(rcut_scan_summary.cell(1.0e-3, 9, ".2e"), " 1.00e-03")


if __name__ == "__main__":
    unittest.main()
