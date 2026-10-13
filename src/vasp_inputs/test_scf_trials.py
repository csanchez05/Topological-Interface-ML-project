"""Tests of scf_trials.py on a temporary project tree with synthetic VASP output.

  python3 -B -m unittest discover -s src/vasp_inputs -p 'test_*.py'

Nothing here runs VASP or Slurm. The OUTCAR/OSZICAR snippets copy the line formats of the v0.2.0 run
12919 (VASP 6.4.3); the POTCAR is a text stand-in.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402
import scf_trials as st  # noqa: E402
import test_make_vasp_inputs as tmv  # noqa: E402
import vasp_task as vt  # noqa: E402

ENV = {"SLURM_NTASKS": "16", "SLURM_JOB_ID": "52", "SLURM_ARRAY_JOB_ID": "51", "SLURM_ARRAY_TASK_ID": "2"}


def outcar(ispin=2, icharg=2, mags=(64.0, 64.0, 7.0), converged=True, finished=True):
    lines = [f"   ICHARG =      {icharg}    charge: 1-file 2-atom 10-const",
             f"   ISPIN  =      {ispin}    spin polarized calculation?"]
    for m in mags:
        lines.append(f" number of electron     832.0000000 magnetization      {m:.7f}" if ispin == 2
                     else " number of electron     832.0000000 magnetization ")
    if finished:
        lines.append("------------------------ aborting loop because EDIFF is reached ------------" if converged
                     else "------------------------ aborting loop EDIFF was not reached (unconverged)  ----")
        lines += ["  free  energy   TOTEN  =      -764.71866258 eV",
                  "  energy  without entropy=     -764.73019212  energy(sigma->0) =     -764.72250576",
                  " General timing and accounting informations for this job:",
                  "                         Elapsed time (sec):    3600.000"]
    return "\n".join(lines) + "\n"


def oszicar(steps=3, ispin=2):
    rows = ["       N       E                     dE             d eps       ncg     rms          rms(c)",
            "DAV:   1     0.698925486079E+04    0.69893E+04   -0.32548E+05 10264   0.176E+03"]
    for n in range(2, steps + 1):
        rows.append(f"DAV: {n:3d}    -0.764718662577E+03   -0.43770E-06   -0.67970E-07 15416   0.428E-03    0.1{n}0E-02")
    rows.append("   1 F= -.76471866E+03 E0= -.76472251E+03  d E =0.115295E-01" + ("  mag=     7.2402" if ispin == 2 else ""))
    return "\n".join(rows) + "\n"


def fake(run, **kw):
    osz = kw.pop("oszicar", None)
    (run / "OUTCAR").write_text(outcar(**kw))
    (run / "OSZICAR").write_text(osz if osz is not None else oszicar(ispin=kw.get("ispin", 2)))
    (run / "rss.json").write_text(json.dumps({"peak_rss_bytes": 9 * 10**9}))


class Tree(unittest.TestCase):
    """A temporary project root holding the two trial structures (synthetic POSCARs)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        for target, name, value in ((vt, "MIN_FREE_GIB", 0), (st, "PROJECT_ROOT", self.root)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        r = mvi.recipe(st.FAMILY, st.STAGE)
        (self.root / r["potcar"]).parent.mkdir(parents=True)
        (self.root / r["potcar"]).write_text("synthetic stand-in, not a potential\n")
        for sym in st.STRUCTURES:
            for template in (r["poscar"], r["catalogue"]):
                path = self.root / template.format(sym=sym, thickness="8x")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(tmv.synthetic_poscar())
        self.records = self.root / "results/trials"
        self.scratch_base = self.root / "scratch"
        self.scratch_base.mkdir()

    def prepare(self, index=2, environ=ENV):
        return st.prepare(index, self.records, environ, self.scratch_base, self.root)


class Inputs(Tree):

    def test_eight_trials(self):
        self.assertEqual(len(st.TRIALS), 8)
        self.assertEqual({v for _, v in st.TRIALS}, {"linmag", "wiki", "nmseed", "nmseed_linmag"})

    def test_each_run_differs_from_v020_by_exactly_the_declared_set(self):
        for sym, variant in st.TRIALS:
            files, _ = st.base_files(sym, self.root)
            base = mvi.tags(files["INCAR"])
            for run, text in zip(st.VARIANTS[variant], st.trial_incars(sym, variant, self.root)):
                expected = {**st.COMMON, **run.get("set", {})}
                diff = st.incar_diff(files["INCAR"], text)
                self.assertEqual({k for k, (_, b) in diff.items() if b is not None}, set(expected))
                self.assertEqual({k for k, (_, b) in diff.items() if b is None}, set(run.get("drop", ())))
                for key, value in expected.items():
                    self.assertEqual(mvi.tags(text)[key], value)
                for key in ("ALGO", "ISMEAR", "SIGMA", "ENCUT", "EDIFF", "PREC", "LREAL", "NBANDS", "KPAR"):
                    self.assertEqual(mvi.tags(text)[key], base[key], key)
                self.assertTrue(text.startswith(f"# scf_trials.py trial={sym}_{variant} run="))

    def test_seeded_run_keeps_magmom_and_reads_a_chgcar(self):
        seed, spin = st.trial_incars("sym00177", "nmseed", self.root)
        self.assertEqual(mvi.tags(seed)["ISPIN"], "1")
        self.assertNotIn("MAGMOM", mvi.tags(seed))
        self.assertEqual(mvi.tags(spin)["ICHARG"], "1")
        self.assertEqual(mvi.tags(spin)["MAGMOM"], "64*1.0 64*0.0")

    def test_trials_refuse_another_generator_version(self):
        with mock.patch.object(mvi, "VERSION", "0.3.0"):
            with self.assertRaises(vt.TaskError):
                st.base_files("sym00177", self.root)


class Prepare(Tree):

    def test_two_run_trial(self):
        scratch = self.prepare(2)
        self.assertEqual([p.name for p in st.run_dirs(scratch)], ["run1", "run2"])
        for run in st.run_dirs(scratch):
            self.assertTrue((run / "POTCAR").is_symlink())
            self.assertEqual({p.name for p in run.iterdir()}, {"INCAR", "KPOINTS", "POSCAR", "POTCAR"})
        prepared = json.loads((scratch / "prepare.json").read_text())
        self.assertEqual((prepared["trial"], prepared["job"]), ("sym00177_nmseed", "51_2"))
        self.assertTrue(scratch.name.startswith("coge-trial-sym00177-nmseed-51_2-"))

    def test_one_run_trial(self):
        self.assertEqual([p.name for p in st.run_dirs(self.prepare(4))], ["run1"])

    def test_refusals(self):
        with self.assertRaises(vt.TaskError):
            self.prepare(2, {**ENV, "SLURM_NTASKS": "8"})
        with self.assertRaises(vt.TaskError):
            self.prepare(8)
        self.assertEqual(list(self.scratch_base.iterdir()), [])


class Stage(Tree):

    def test_seed_must_converge_first(self):
        scratch = self.prepare(2)
        self.assertIn("atomic", st.stage(scratch, "run1"))
        fake(scratch / "run1", ispin=1, converged=False)
        (scratch / "run1/CHGCAR").write_text("density\n")
        with self.assertRaises(vt.TaskError):
            st.stage(scratch, "run2")
        self.assertFalse((scratch / "run2/CHGCAR").exists())

    def test_converged_seed_is_copied(self):
        scratch = self.prepare(2)
        fake(scratch / "run1", ispin=1)
        (scratch / "run1/CHGCAR").write_text("density\n")
        st.stage(scratch, "run2")
        self.assertEqual((scratch / "run2/CHGCAR").read_text(), "density\n")

    def test_empty_chgcar_is_refused(self):
        scratch = self.prepare(2)
        fake(scratch / "run1", ispin=1)
        (scratch / "run1/CHGCAR").write_text("")
        with self.assertRaises(vt.TaskError):
            st.stage(scratch, "run2")


class Parse(Tree):

    def run_dir(self, index=0, **kw):
        run = st.run_dirs(self.prepare(index))[-1]
        fake(run, **kw)
        return run

    def test_converged(self):
        r = st.parse_run(self.run_dir())
        self.assertTrue(r["converged"])
        self.assertEqual((r["steps"], r["final_mag_muB"], r["elapsed_s"]), (3, 7.2402, 3600.0))
        self.assertEqual(r["trace"]["mag"], [64.0, 64.0, 7.0])
        self.assertEqual(r["last_rmsc"], 0.13e-2)

    def test_unconverged_and_killed(self):
        self.assertFalse(st.parse_run(self.run_dir(converged=False))["converged"])
        killed = st.parse_run(self.run_dir(finished=False))
        self.assertFalse(killed["converged"])
        self.assertIsNone(killed["toten_eV"])

    def test_nelm_reached_is_not_converged(self):
        run = self.run_dir(oszicar=oszicar(steps=200))
        self.assertFalse(st.parse_run(run)["converged"])

    def test_non_magnetic_run(self):
        r = st.parse_run(self._seed_run())
        self.assertTrue(r["converged"])
        self.assertIsNone(r["final_mag_muB"])
        self.assertEqual(r["trace"]["mag"], [])

    def _seed_run(self):
        run = st.run_dirs(self.prepare(2))[0]
        fake(run, ispin=1)
        return run

    def test_seed_check(self):
        run = st.run_dirs(self.prepare(2))[1]
        fake(run, icharg=1)
        self.assertTrue(st.parse_run(run)["seed_check"]["ok"])
        fake(run, icharg=1, mags=(0.0, 0.0, 0.0))
        self.assertFalse(st.parse_run(run)["seed_check"]["ok"])
        fake(run, icharg=2)
        self.assertFalse(st.parse_run(run)["seed_check"]["ok"])


class Summaries(Tree):

    def test_two_run_trial_and_table(self):
        scratch = self.prepare(2)
        fake(scratch / "run1", ispin=1)
        fake(scratch / "run2", icharg=1)
        t = st.summarize(scratch, self.records)
        self.assertTrue(t["converged"])
        self.assertEqual(t["core_hours"], 32.0)
        self.assertTrue((self.records / "trials/sym00177_nmseed.json").is_file())
        text = st.table(self.records)
        header, row = text.splitlines()
        cells = dict(zip(header.split("\t"), row.split("\t")))
        self.assertEqual((cells["converged"], cells["nm_steps"], cells["seed_ok"]), ("True", "3", "True"))

    def test_seed_only_is_not_a_converged_trial(self):
        scratch = self.prepare(2)
        fake(scratch / "run1", ispin=1)
        self.assertFalse(st.summarize(scratch, self.records)["converged"])

    def test_past_runner_scratch_needs_a_label(self):
        past = self.scratch_base / "coge-gb-sym00177-e500_k1x3x3-12919_0-abc"
        past.mkdir()
        (past / "INCAR").write_text(st.base_files("sym00177", self.root)[0]["INCAR"])
        fake(past, converged=False)
        with self.assertRaises(vt.TaskError):
            st.summarize(past, self.records)
        t = st.summarize(past, self.records, label="sym00177_v0.2.0_12919")
        self.assertEqual((t["sym"], t["converged"], len(t["runs"])), ("sym00177", False, 1))


if __name__ == "__main__":
    unittest.main()
