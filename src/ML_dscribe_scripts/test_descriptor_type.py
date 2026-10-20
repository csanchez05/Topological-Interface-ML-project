"""Synthetic tests for --descriptor-type soap|covariant|ace in ml_common.py and select_gbs.py.

    python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_*.py'
"""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import ml_common as mc
import select_gbs

sc = mc.sc
sys.path.insert(0, str(mc.COVARIANT_SCRIPTS))
import test_covariant_common as tcc  # noqa: E402  (its synthetic covariant run)

cc = mc.covariant_module()


def parse(*argv):
    parser = argparse.ArgumentParser()
    mc.add_vector_arguments(parser)
    return mc.resolve_vector_arguments(parser.parse_args(list(argv)))


def write_ace_run(root, widths, n=3):
    """A small ACE run: global and both_interfaces folders of one tag, with bulk vectors."""
    rng = np.random.default_rng(1)
    global_dir, interface_dir = root / "global" / "tag", root / "both_interfaces" / "tag"
    interface = {name: rng.normal(size=(n, 2, 2 * w)).astype(np.float32) for name, w in widths.items()}
    counts = np.full((n, 2, 2), 2, dtype=np.int32)
    bulk = {name: rng.normal(size=(2, 2 * w)) for name, w in widths.items()}
    sc.write_npz(interface_dir / "interface_ace.npz", **{f"X_{k}_r8": v for k, v in interface.items()},
                 n_r8=counts, radii=np.array([8.0]))
    sc.write_npz(global_dir / "global_ace.npz", **{f"X_{k}": v[:, 0] for k, v in interface.items()})
    sc.write_npz(global_dir / "bulk_ace.npz", **{f"X_{k}": v for k, v in bulk.items()})
    rows = [{"index": i, "plane": "100", "sym": f"sym{i:05d}", "thickness": 4, "family": f"100/sym{i:05d}",
             "Lx": 40.0} for i in range(n)]
    for folder in (global_dir, interface_dir):
        sc.write_csv(folder / "index.csv", rows, list(rows[0]))
    sc.write_csv(global_dir / "bulk_index.csv", [{"index": 0, "kind": "pristine"}, {"index": 1, "kind": "transformed"}],
                 ["index", "kind"])
    return interface_dir, interface, bulk


class Arguments(unittest.TestCase):
    def test_soap_default_is_unchanged(self):
        args = parse()
        self.assertEqual((args.descriptor_type, args.descriptor), ("soap", "rc6.0_n8_l6_s0.5"))
        self.assertEqual(mc.output_dir(args), mc.OUTPUT_ROOT / "rc6.0_n8_l6_s0.5" / "interface_r8_both_t4")
        self.assertEqual(mc.vector_source(args), {"descriptor_type": "soap", "ace_blocks": None,
                                                  "ace_scaling": "pristine", "covariant_blocks": None})

    def test_layout_since_2026_10_13(self):
        ml = mc.PROJECT_ROOT / "calculations" / "ML_dscribe" / "CoGe"
        self.assertEqual(mc.OUTPUT_ROOT, ml / "standard")
        self.assertEqual(mc.OUTPUT_ROOTS["covariant"], ml / "covariant")
        self.assertEqual(mc.COMPARISON_ROOT, ml / "comparison")
        self.assertEqual(mc.LABELS_DIR, ml / "labels")
        self.assertEqual(mc.CENSUS, ml / "standard" / "rc6.0_n8_l6_s0.5" / "interface_r8_both_t4" / "distinct_gbs.csv")
        soap = mc.PROJECT_ROOT / "plots" / "CoGe" / "dim-reduction" / "SOAP"
        self.assertEqual((mc.plot_root("soap"), mc.plot_root("covariant"), mc.COMPARISON_PLOTS),
                         (soap / "standard", soap / "covariant", soap / "comparison"))
        self.assertEqual(list(mc.COVARIANT_BLOCKS), list(cc.BLOCKS))

    def test_covariant_default_and_folders(self):
        args = parse("--descriptor-type", "covariant", "--thickness", "8")
        self.assertEqual(args.descriptor, "rc6.0_n8_l6_s0.5")
        self.assertEqual(mc.output_dir(args), mc.OUTPUT_ROOTS["covariant"] / "rc6.0_n8_l6_s0.5" / "interface_r8_both_t8")
        self.assertEqual(mc.vector_source(args)["covariant_blocks"], None)
        blocks = parse("--descriptor-type", "covariant", "--covariant-blocks", "tensor", "polar")
        self.assertEqual(blocks.covariant_blocks, ["polar", "tensor"])       # storage order
        self.assertEqual(mc.output_dir(blocks).name, "interface_r8_both_t4_polar+tensor")
        every = parse("--descriptor-type", "covariant", "--covariant-blocks", *mc.COVARIANT_BLOCKS)
        self.assertEqual(mc.output_dir(every).name, "interface_r8_both_t4_all5")

    def test_ace_default_and_folders(self):
        args = parse("--descriptor-type", "ace")
        self.assertEqual(args.descriptor, "rc6.0_n8_l6_s0.5_nu3-n4-l4")
        root = mc.PROJECT_ROOT / "calculations" / "ML_ACE" / "CoGe"
        self.assertEqual(mc.output_dir(args), root / "rc6.0_n8_l6_s0.5_nu3-n4-l4" / "interface_r8_both_t4")
        blocks = parse("--descriptor-type", "ace", "--ace-blocks", "nu3_odd", "nu2", "--ace-scaling", "none")
        self.assertEqual(blocks.ace_blocks, ["nu2", "nu3_odd"])       # storage order
        self.assertEqual(mc.output_dir(blocks).name, "interface_r8_both_t4_nu2+nu3_odd_noscale")
        every = parse("--descriptor-type", "ace", "--ace-blocks", *mc.ACE_BLOCKS)
        self.assertEqual(mc.output_dir(every).name, "interface_r8_both_t4")

    def test_options_that_do_not_fit_the_type(self):
        with self.assertRaises(ValueError):
            parse("--ace-blocks", "nu2")
        with self.assertRaises(ValueError):
            parse("--ace-scaling", "none")
        with self.assertRaises(ValueError):
            parse("--descriptor-type", "ace", "--kind", "nonbulk")
        with self.assertRaises(ValueError):
            parse("--covariant-blocks", "polar")
        with self.assertRaises(ValueError):
            parse("--descriptor-type", "ace", "--covariant-blocks", "polar")
        with self.assertRaises(ValueError):
            parse("--descriptor-type", "covariant", "--ace-blocks", "nu2")
        with self.assertRaises(ValueError):
            parse("--descriptor-type", "covariant", "--kind", "nonbulk")


class CovariantLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder, self.interface, self.glob, self.bulk = tcc.write_run(Path(self.tmp.name) / "covariant")

    def tearDown(self):
        self.tmp.cleanup()

    def test_vectors_join_edge_and_mid(self):
        X, index = mc.load_all_vectors(self.folder, descriptor_type="covariant")
        edge = np.concatenate([self.interface[k][:, 0].reshape(3, -1) for k in cc.DEFAULT_BLOCKS], axis=1)
        mid = np.concatenate([self.interface[k][:, 1].reshape(3, -1) for k in cc.DEFAULT_BLOCKS], axis=1)
        np.testing.assert_array_equal(X, np.concatenate([edge, mid], axis=1))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(len(index), 3)
        polar, _ = mc.load_all_vectors(self.folder, kind="global", descriptor_type="covariant",
                                       covariant_blocks=["polar"])
        np.testing.assert_array_equal(polar, self.glob["polar"].reshape(3, -1))
        with self.assertRaises(ValueError):
            mc.load_all_vectors(self.folder, kind="nonbulk", descriptor_type="covariant")

    def test_settings_record_blocks_and_join(self):
        args = parse("--descriptor-type", "covariant", "--descriptor", str(self.folder))
        settings = mc.vector_settings(args)
        self.assertEqual((settings["descriptor_type"], settings["descriptor"]), ("covariant", "tag"))
        self.assertEqual(settings["covariant_blocks"], list(cc.DEFAULT_BLOCKS))
        self.assertEqual(settings["planes_joined"], "[edge | mid], not averaged")
        self.assertNotIn("covariant_blocks", mc.vector_settings(parse()))

    def test_a_covariant_run_read_as_soap_gets_a_hint(self):
        with self.assertRaisesRegex(FileNotFoundError, "--descriptor-type covariant"):
            mc.load_all_vectors(self.folder, descriptor_type="soap")


class Loading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.widths = {"nu1": 2, "nu2": 3, "nu3_even": 4, "nu3_odd": 2}
        self.folder, self.interface, self.bulk = write_ace_run(Path(self.tmp.name) / "ACE" / "standard", self.widths)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ace_vectors_are_scaled_blocks(self):
        X, index = mc.load_all_vectors(self.folder, planes="edge", descriptor_type="ace")
        expected = np.concatenate([self.interface[name][:, 0] / np.linalg.norm(self.bulk[name][0])
                                   for name in self.widths], axis=1)
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_allclose(X, expected, rtol=1e-6)
        self.assertEqual(index["family"].tolist(), ["100/sym00000", "100/sym00001", "100/sym00002"])
        odd, _ = mc.load_all_vectors(self.folder, planes="mid", descriptor_type="ace", ace_blocks=["nu3_odd"],
                                     ace_scaling="none")
        np.testing.assert_array_equal(odd, self.interface["nu3_odd"][:, 1])

    def test_settings_record_the_type(self):
        args = parse("--descriptor-type", "ace", "--descriptor", str(self.folder))
        settings = mc.vector_settings(args)
        self.assertEqual((settings["descriptor_type"], settings["descriptor"]), ("ace", "tag"))
        self.assertEqual(settings["ace_blocks"], "all stored")
        self.assertNotIn("ace_blocks", mc.vector_settings(parse()))

    def test_a_run_of_the_other_type_is_refused_with_a_hint(self):
        with self.assertRaisesRegex(FileNotFoundError, "--descriptor-type ace"):
            mc.load_all_vectors(self.folder, descriptor_type="soap")
        soap_dir = Path(self.tmp.name) / "standard" / "both_interfaces" / "soap"
        soap_dir.mkdir(parents=True)
        (soap_dir / sc.INTERFACE_FILE).write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "--descriptor-type soap"):
            mc.load_all_vectors(soap_dir, descriptor_type="ace")


class SelectionFolders(unittest.TestCase):
    def folders(self, descriptor_type="soap", tag="rc6.0_n8_l6_s0.5", blocks=None, scaling="pristine",
                planes=("100", "110", "111"), min_pair=None, n_select=30, covariant_blocks=None):
        return select_gbs.default_folders(descriptor_type, tag, blocks, scaling, list(planes), min_pair, n_select,
                                          covariant_blocks)

    def test_soap_folders_are_unchanged(self):
        self.assertEqual(self.folders(), (select_gbs.PLOT_DIR, select_gbs.OUT_DIR))
        plot, out = self.folders(planes=["100"], min_pair=1.8)
        self.assertEqual((plot.name, out.name), ("plane100_dmin1.8", "plane100_dmin1.8"))
        self.assertEqual(plot.parent, mc.plot_root("soap"))

    def test_ace_folders(self):
        ace = mc.DEFAULT_DESCRIPTORS["ace"]
        plot, out = self.folders("ace", ace)
        self.assertEqual(plot, mc.plot_root("ace"))
        self.assertEqual(out, mc.OUTPUT_ROOTS["ace"] / "dim-reduction")
        plot, out = self.folders("ace", ace, ["nu3_odd"], "none", ["100"], 1.8)
        self.assertEqual(plot, mc.plot_root("ace") / "nu3_odd_noscale_plane100_dmin1.8")
        plot, _ = self.folders("ace", "rc6.0_n8_l6_dirac_nu3-n4-l4")
        self.assertEqual(plot.name, "rc6.0_n8_l6_dirac_nu3-n4-l4")

    def test_covariant_folders(self):
        self.assertEqual(self.folders("covariant"),
                         (mc.plot_root("covariant"), mc.OUTPUT_ROOTS["covariant"] / "dim-reduction"))
        plot, out = self.folders("covariant", planes=["100"], min_pair=1.8, covariant_blocks=list(mc.COVARIANT_BLOCKS))
        self.assertEqual((plot.name, out.name), ("all5_plane100_dmin1.8", "all5_plane100_dmin1.8"))
        plot, _ = self.folders("covariant", "rc8.0_n8_l4_s0.3_t8")
        self.assertEqual(plot, mc.plot_root("covariant") / "rc8.0_n8_l4_s0.3_t8")


if __name__ == "__main__":
    unittest.main()
