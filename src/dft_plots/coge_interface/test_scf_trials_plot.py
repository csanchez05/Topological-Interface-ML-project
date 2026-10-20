"""Tests of scf_trials_plot.py on synthetic trial records (no real data, no drawing except one smoke)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scf_trials_plot as sp  # noqa: E402


def run(ispin, steps, elapsed, converged=True):
    return {"run": "run1", "incar": {"ISPIN": str(ispin)}, "steps": steps, "elapsed_s": elapsed,
            "converged": converged, "trace": {"rmsc": [None, None] + [1.0] * (steps - 2),
                                              "mag": [] if ispin == 1 else [64.0] * (steps + 2)}}


class Series(unittest.TestCase):

    def test_seed_then_magnetic_run_on_one_time_axis(self):
        trial = {"runs": [run(1, 10, 3600), run(2, 20, 7200)]}
        segs = sp.series(trial, sp.fallbacks([trial]))
        self.assertTrue(segs[0]["dashed"] and not segs[1]["dashed"])
        self.assertAlmostEqual(segs[0]["end"], 1.0)
        self.assertAlmostEqual(segs[1]["end"], 3.0)
        self.assertAlmostEqual(segs[1]["rmsc"][0][0], 1.0 + 3 * 0.1)  # first rms(c) is at step 3
        self.assertEqual(segs[0]["mag"], [])
        self.assertAlmostEqual(segs[1]["mag"][-1][0], 3.0)

    def test_killed_run_uses_the_median_step_time(self):
        timed = {"runs": [run(2, 10, 2000)]}
        killed = {"runs": [run(2, 5, None, converged=False)]}
        segs = sp.series(killed, sp.fallbacks([timed, killed]))
        self.assertAlmostEqual(segs[0]["end"], 5 * 200 / 3600)

    def test_label_positions_keep_a_gap(self):
        pos = sp.label_positions([0.0, 0.01, 0.02, 2.0], -3, 3, 0.5)
        self.assertEqual(pos[0], 0.0)
        self.assertAlmostEqual(pos[1], 0.5)
        self.assertAlmostEqual(pos[2], 1.0)
        self.assertEqual(pos[3], 2.0)

    def test_smoke_figure_is_marked_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp.main(["--smoke", tmp])
            png = Path(tmp) / "SYNTHETIC_trace.png"
            self.assertTrue(png.stat().st_size > 10000)
            self.assertIn(b"SYNTHETIC TEST DATA", png.read_bytes())


if __name__ == "__main__":
    unittest.main()
