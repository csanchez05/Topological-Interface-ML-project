"""Tests of covariant_common.py on a synthetic run folder (no featomic needed).

    python -B -m unittest discover -s src/dscribe/CoGe/covariant -p 'test_covariant_common.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import covariant_common as cc

N_PROPS = {"invariant": 4, "polar": 3, "axial": 2, "tensor": 5, "pseudotensor": 1}  # P per block


def comps(name):
    return 2 * cc.BLOCKS[name][0] + 1


def write_run(root, blocks=tuple(cc.BLOCKS), n=3, seed=0):
    """A fake covariant run with n GB structures and 2 bulk cells; returns the arrays written."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    interface, glob, bulk = {}, {}, {}
    for name in blocks:
        p = N_PROPS[name]
        interface[name] = rng.normal(size=(n, 2, comps(name), 2 * p)).astype(np.float32)
        glob[name] = rng.normal(size=(n, comps(name), 2 * p)).astype(np.float32)
        bulk[name] = rng.normal(size=(2, comps(name), 2 * p)).astype(np.float32)
    props = {f"properties_{name}": np.zeros((N_PROPS[name], 6), dtype=np.int32) for name in blocks}
    gb_dir, global_dir = root / "both_interfaces" / "tag", root / "global" / "tag"
    gb_dir.mkdir(parents=True)
    global_dir.mkdir(parents=True)
    np.savez(gb_dir / cc.FILE, **{f"X_{k}_r8": v for k, v in interface.items()}, n_r8=np.ones((n, 2, 2), np.int32),
             radii=np.array([8.0]), **props)
    np.savez(global_dir / cc.FILE, **{f"X_{k}": v for k, v in glob.items()}, **props)
    np.savez(global_dir / cc.BULK_FILE, **{f"X_{k}": v for k, v in bulk.items()}, **props)
    index = pd.DataFrame({"index": range(n), "plane": ["100"] * n, "family": [f"100/sym{i:05d}" for i in range(n)]})
    for folder in (gb_dir, global_dir):
        index.to_csv(folder / "index.csv", index=False)
    pd.DataFrame({"index": [0, 1], "kind": ["pristine", "transformed"]}).to_csv(global_dir / "bulk_index.csv", index=False)
    return gb_dir, interface, glob, bulk


class Loaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gb_dir, self.interface, self.glob, self.bulk = write_run(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def expected_interface(self, side, names=cc.DEFAULT_BLOCKS):
        return np.concatenate([self.interface[k][:, side].reshape(3, -1) for k in names], axis=1)

    def test_both_planes_joins_edge_then_mid(self):
        X, index = cc.load_gb_vectors(self.gb_dir, "interface", 8.0, "both")
        expected = np.concatenate([self.expected_interface(0), self.expected_interface(1)], axis=1)
        np.testing.assert_array_equal(X, expected)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(len(index), 3)
        self.assertEqual(X.shape[1], cc.vector_length(self.gb_dir))

    def test_single_plane_and_block_choice(self):
        X, _ = cc.load_gb_vectors(self.gb_dir, "interface", 8.0, "mid", blocks=["tensor", "polar"])
        np.testing.assert_array_equal(X, self.expected_interface(1, ["polar", "tensor"]))  # storage order
        self.assertEqual(X.shape[1], cc.vector_length(self.gb_dir, ["polar", "tensor"], planes="mid"))

    def test_global(self):
        X, _ = cc.load_gb_vectors(self.gb_dir, "global")
        expected = np.concatenate([self.glob[k].reshape(3, -1) for k in cc.DEFAULT_BLOCKS], axis=1)
        np.testing.assert_array_equal(X, expected)
        self.assertEqual(X.shape[1], cc.vector_length(self.gb_dir, kind="global"))

    def test_bulk_layout_matches_gb_layout(self):
        X, index = cc.load_bulk_vectors(self.gb_dir)
        one = np.concatenate([self.bulk[k].reshape(2, -1) for k in cc.DEFAULT_BLOCKS], axis=1)
        np.testing.assert_array_equal(X, np.concatenate([one, one], axis=1))
        self.assertEqual(index["kind"].tolist(), ["pristine", "transformed"])
        X_edge, _ = cc.load_bulk_vectors(self.gb_dir, planes="edge")
        np.testing.assert_array_equal(X_edge, one)

    def test_errors(self):
        with self.assertRaises(ValueError):
            cc.load_gb_vectors(self.gb_dir, "interface", 6.0)          # radius not stored
        with self.assertRaises(ValueError):
            cc.load_gb_vectors(self.gb_dir, "nonbulk", 8.0)            # not stored for covariant runs
        with self.assertRaises(ValueError):
            cc.choose_blocks(["polar"], ["polar", "spin"])             # unknown block
        with self.assertRaises(ValueError):
            cc.choose_blocks(["polar", "axial"], None)                 # default blocks not all stored

    def test_member_shape_reads_header_only(self):
        with np.load(self.gb_dir / cc.FILE) as data:
            self.assertEqual(cc.member_shape(data, "X_tensor_r8"), (3, 2, 5, 2 * N_PROPS["tensor"]))

    def test_suffix(self):
        self.assertEqual(cc.blocks_suffix(None), "")
        self.assertEqual(cc.blocks_suffix(list(cc.DEFAULT_BLOCKS)), "")
        self.assertEqual(cc.blocks_suffix(list(cc.BLOCKS)), "_all5")
        self.assertEqual(cc.blocks_suffix(["tensor", "polar"]), "_polar+tensor")


class SameBlocksAsTheWriter(unittest.TestCase):
    def test_blocks_match_covariant_descriptor(self):
        try:
            import covariant_descriptor as cd
        except ImportError as exc:  # featomic missing
            self.skipTest(str(exc))
        self.assertEqual(list(cd.BLOCKS.items()), list(cc.BLOCKS.items()))


if __name__ == "__main__":
    unittest.main()
