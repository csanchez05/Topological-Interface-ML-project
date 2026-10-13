"""Synthetic tests for --descriptor-type soap|ace in ml_common.py and select_gbs.py.

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
                                                  "ace_scaling": "pristine"})

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
                planes=("100", "110", "111"), min_pair=None, n_select=30):
        return select_gbs.default_folders(descriptor_type, tag, blocks, scaling, list(planes), min_pair, n_select)

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


if __name__ == "__main__":
    unittest.main()
