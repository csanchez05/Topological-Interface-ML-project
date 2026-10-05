"""Tests of vasp_task.py and submit_vasp.py on a temporary project tree with synthetic VASP output.

  python3 -B -m unittest discover -s src/vasp_inputs -p 'test_*.py'

Nothing here runs VASP or Slurm. The OUTCAR/OSZICAR snippets copy the line formats of the finished
VASP 6.4.3 bulk runs; the POTCAR is a text stand-in.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402
import submit_vasp  # noqa: E402
import test_make_vasp_inputs as tmv  # noqa: E402
import vasp_task  # noqa: E402

FAMILY, STAGE, SYM = "CoGe_100_GB", "convergence-nonsoc", "sym00001"
ENV = {"SLURM_NTASKS": "16", "SLURM_JOB_ID": "42", "SLURM_ARRAY_JOB_ID": "41", "SLURM_ARRAY_TASK_ID": "0"}

# A finished e500_k1x9x9 run: 16 ranks, KPAR 4, NCORE 4, 128 ions, 832 electrons.
OUTCAR = """\
 vasp.6.4.3 19Mar24 (build Oct 19 2024 00:28:57) complex

 executed on             LinuxIFC date 2026.10.05  12:00:00
 running   16 mpi-ranks, with    1 threads/rank, on    1 nodes
 distrk:  each k-point on    4 cores,    4 groups
 distr:  one band on NCORE=   4 cores,    1 groups
   ENCUT = 500
   NELM = 100
   TITEL  = PAW_PBE Co 02Aug2007
   TITEL  = PAW_PBE Ge 05Jan2001
 generate k-points for:    1    9    9
   k-points           NKPTS =     25   k-points in BZ     NKDIM =     25   number of bands    NBANDS=    512
   number of dos      NEDOS =    301   number of ions     NIONS =    128
   total plane-waves  NPLWV = 1036800
   NELECT =     832.0000    total number of electrons
   ISTART =      0    job   : 0-new  1-cont  2-samecut
   ICHARG =      2    charge: 1-file 2-atom 10-const
   ISPIN  =      2    spin polarized calculation?
   LSORBIT =      F    spin-orbit coupling
   ENCUT  =  500.0 eV  36.75 Ry    6.06 a.u.
   NELM   =    100;   NELMIN=  2; NELMDL= -5     # of ELM steps
   EDIFF  = 0.1E-05   stopping-criterion for ELM
   LREAL  =      F    real-space projection
   LMAXMIX     =    4 max onsite mixed and CHGCAR
   NSW    =      0    number of steps for IOM
   IBRION =     -1    ionic relax: 0-MD 1-quasi-New 2-CG
   ISYM   =      2    0-nonsym 1-usesym 2-fastsym
   ISMEAR =     1;   SIGMA  =   0.05  broadening in eV -4-tet -1-fermi 0-gaus
   LWAVE        =      F    write WAVECAR
   LCHARG       =      F    write CHGCAR
 E-fermi :   8.1234     XC(G=0): -12.3678     alpha+bet :-16.1116
------------------------ aborting loop because EDIFF is reached ----------------------------------------
  free  energy   TOTEN  =      -812.34567890 eV
  energy  without entropy=     -812.30000000  energy(sigma->0) =     -812.32283945
  entropy T*S    EENTRO =        -0.04567890
 General timing and accounting informations for this job:
                         Elapsed time (sec):     9000.123
                   Maximum memory used (kb):     4152280.
"""
OSZICAR = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1     0.105600000000E+04    0.10560E+04   -0.81000E+04 25600   0.123E+03
DAV:  30    -0.812345678800E+03   -0.20000E-05   -0.30000E-07 25600   0.200E-03   0.100E-03
DAV:  31    -0.812345678900E+03   -0.50000E-06   -0.10000E-07 25600   0.100E-03
   1 F= -.81234568E+03 E0= -.81232284E+03  d E =-.456789E-01  mag=     0.0012
"""
VASP_OUT = """\
 running   16 mpi-ranks, with    1 threads/rank, on    1 nodes
 -----------------------------------------------------------------------------
|                                                                             |
|               ----> ADVICE to this user running VASP <----                  |
|                                                                             |
|     You have a (more or less) 'large supercell' and for larger cells it     |
|     might be more efficient to use real-space projection operators.         |
|                                                                             |
 -----------------------------------------------------------------------------

 DAV:   1     0.105600000000E+04
"""


def fake_run(scratch, **replace):
    """Write a finished run's outputs into scratch; replace={file: [(old, new), ...]} breaks one."""
    texts = {"OUTCAR": OUTCAR, "OSZICAR": OSZICAR, "vasp.out": VASP_OUT,
             "vasprun.xml": "<?xml version=\"1.0\"?>\n<modeling>\n</modeling>\n",
             "EIGENVAL": "synthetic\n", "IBZKPT": "synthetic\n", "DOSCAR": "synthetic\n",
             "rss.json": json.dumps({"peak_rss_bytes": 50 * 1024**3, "max_ranks_seen": 16, "samples": 300})}
    for name, pairs in replace.items():
        for old, new in pairs:
            assert old in texts[name], (name, old)
            texts[name] = texts[name].replace(old, new)
    for name, text in texts.items():
        (Path(scratch) / name).write_text(text)


class Tree(unittest.TestCase):
    """A temporary project root holding the 9 generated folders of one synthetic GB."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        for target, value in ((vasp_task, self.root), (submit_vasp, self.root)):
            patcher = mock.patch.object(target, "PROJECT_ROOT", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(vasp_task, "MIN_FREE_GIB", 0)
        patcher.start()
        self.addCleanup(patcher.stop)
        r = mvi.recipe(FAMILY, STAGE)
        (self.root / r["potcar"]).parent.mkdir(parents=True)
        (self.root / r["potcar"]).write_text("synthetic stand-in, not a potential\n")
        for template in (r["poscar"], r["catalogue"]):
            path = self.root / template.format(sym=SYM, thickness="8x")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tmv.synthetic_poscar())
        counts, problems, _, _ = mvi.generate(FAMILY, STAGE, [SYM], "8x", None, False, "abc1234", self.root)
        assert counts["written"] == 9 and not problems, (counts, problems)
        self.records = self.root / "results/CoGe_interface/100/convergence_8x"
        self.scratch_base = self.root / "scratch"
        self.scratch_base.mkdir()

    def folder(self, point="e500_k1x9x9"):
        return self.root / f"calculations/CoGe_interface/100/{SYM}/8x/convergence/{point}"

    def prepare(self, point="e500_k1x9x9", environ=ENV):
        return vasp_task.prepare(self.folder(point), self.records, environ, self.scratch_base)

    def claim(self, point="e500_k1x9x9"):
        return self.records / f"claims/{SYM}_8x_{point}.json"


class Prepare(Tree):

    def test_identify(self):
        info = vasp_task.identify(self.folder())
        self.assertEqual((info["sym"], info["thickness"], info["point"], info["version"], info["git"]),
                         (SYM, "8x", "e500_k1x9x9", mvi.VERSION, "abc1234"))

    def test_prepare_stages_inputs_and_claims(self):
        scratch = self.prepare()
        self.assertEqual(scratch.parent, self.scratch_base)
        self.assertTrue(scratch.name.startswith(f"coge-gb-{SYM}-e500_k1x9x9-41_0-"))
        self.assertEqual(sorted(os.listdir(scratch)), ["INCAR", "KPOINTS", "POSCAR", "POTCAR", "prepare.json"])
        self.assertEqual(os.readlink(scratch / "POTCAR"), str(self.root / "calculations/CoGe_interface/POTCAR"))
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            self.assertEqual((scratch / name).read_bytes(), (self.folder() / name).read_bytes())
        prepared = json.loads((scratch / "prepare.json").read_text())
        self.assertEqual((prepared["ranks"], prepared["kpar"], prepared["ncore"], prepared["job"]), (16, 4, 4, "41_0"))
        self.assertEqual(prepared["inputs_sha256"]["INCAR"], vasp_task.digest(self.folder() / "INCAR"))
        self.assertTrue(self.claim().exists())

    def test_small_mesh_profile(self):
        prepared = json.loads((self.prepare("e500_k1x3x3") / "prepare.json").read_text())
        self.assertEqual((prepared["ranks"], prepared["kpar"], prepared["ncore"]), (16, 1, 4))

    def test_second_prepare_is_refused_by_the_claim(self):
        self.prepare()
        with self.assertRaisesRegex(vasp_task.TaskError, "exists: running elsewhere"):
            self.prepare()

    def test_wrong_rank_count_is_refused_before_claiming(self):
        with self.assertRaisesRegex(vasp_task.TaskError, "needs 16 ranks"):
            self.prepare(environ={**ENV, "SLURM_NTASKS": "8"})
        self.assertFalse(self.claim().exists())

    def test_claim_released_when_scratch_fails(self):
        with mock.patch.object(vasp_task, "MIN_FREE_GIB", 10**9):
            with self.assertRaisesRegex(vasp_task.TaskError, "GiB free"):
                self.prepare()
        self.assertFalse(self.claim().exists())

    def test_folder_with_output_is_refused(self):
        (self.folder() / "OUTCAR").write_text("finished\n")
        with self.assertRaisesRegex(vasp_task.TaskError, r"\(refused\)"):
            self.prepare()
        self.assertFalse(self.claim().exists())

    def test_hand_edited_inputs_are_refused(self):
        incar = self.folder() / "INCAR"
        incar.write_text(incar.read_text().replace("NELM = 100", "NELM = 200"))
        with self.assertRaisesRegex(vasp_task.TaskError, "different: INCAR"):
            self.prepare()

    def test_folder_outside_interface_tree_is_refused(self):
        with self.assertRaisesRegex(vasp_task.TaskError, "not under calculations/CoGe_interface"):
            vasp_task.resolve_folder(self.root / "results")


class Validate(Tree):

    def setUp(self):
        super().setUp()
        self.scratch = self.prepare()
        self.prepared = json.loads((self.scratch / "prepare.json").read_text())

    def test_good_run(self):
        fake_run(self.scratch)
        rec = vasp_task.validate(self.scratch, self.prepared, 0)
        self.assertTrue(rec["valid"])
        self.assertEqual((rec["nions"], rec["nelect"], rec["mesh"], rec["nkpts"], rec["nbands"], rec["scf_steps"]),
                         (128, 832.0, [1, 9, 9], 25, 512, 31))
        self.assertEqual((rec["encut_eV"], rec["ismear"], rec["sigma_eV"], rec["ediff_eV"]), (500.0, 1, 0.05, 1e-6))
        self.assertAlmostEqual(rec["toten_eV"], -812.34567890)
        self.assertAlmostEqual(rec["toten_eV_atom"], -812.34567890 / 128)
        self.assertAlmostEqual(rec["e0_eV"], -812.32283945)
        self.assertAlmostEqual(rec["eentro_eV"], -0.04567890)
        self.assertAlmostEqual(rec["efermi_eV"], 8.1234)
        self.assertAlmostEqual(rec["magnetization_muB"], 0.0012)
        self.assertEqual((rec["nplwv"], rec["elapsed_s"], rec["rank0_max_memory_kb"]), (1036800, 9000.123, 4152280.0))
        self.assertEqual((rec["peak_rss_bytes"], rec["max_ranks_seen"]), (50 * 1024**3, 16))
        self.assertEqual([w["title"] for w in rec["warnings"]], ["ADVICE to this user running VASP"])

    def test_broken_runs_fail(self):
        cases = {
            "VASP did not finish": {"OUTCAR": [(" General timing", " Gen timing")]},
            "did not reach EDIFF": {"OUTCAR": [("aborting loop because EDIFF", "aborting loop EDIFF")]},
            "SCF steps": {"OSZICAR": [("DAV:  31", "DAV: 100")]},
            "final F= line": {"OSZICAR": [("   1 F=", "   F=")]},
            "k-mesh": {"OUTCAR": [("for:    1    9    9", "for:    1    7    7")]},
            "MPI decomposition": {"OUTCAR": [("4 cores,    4 groups", "16 cores,    1 groups")]},
            "NBANDS 500": {"OUTCAR": [("NBANDS=    512", "NBANDS=    500")]},
            "POTCAR TITELs": {"OUTCAR": [("PAW_PBE Co 02Aug2007", "PAW_PBE Co 06Sep2000")]},
            "NELECT": {"OUTCAR": [("832.0000", "830.0000")]},
            "ENCUT": {"OUTCAR": [("500.0 eV", "520.0 eV")]},
            "EDIFF": {"OUTCAR": [("0.1E-05", "0.1E-07")]},
            "ISMEAR/SIGMA": {"OUTCAR": [("ISMEAR =     1;", "ISMEAR =    -1;")]},
            "OUTCAR ICHARG": {"OUTCAR": [("ICHARG =      2", "ICHARG =      1")]},
            "OUTCAR LCHARG": {"OUTCAR": [("LCHARG       =      F", "LCHARG       =      T")]},
            "not vasp.6.4.3": {"OUTCAR": [("vasp.6.4.3", "vasp.6.2.0")]},
            "vasprun.xml incomplete": {"vasprun.xml": [("</modeling>", "")]},
        }
        for message, replace in cases.items():
            with self.subTest(message=message):
                fake_run(self.scratch, **replace)
                with self.assertRaisesRegex(vasp_task.TaskError, message):
                    vasp_task.validate(self.scratch, self.prepared, 0)

    def test_exit_code_and_changed_inputs_fail(self):
        fake_run(self.scratch)
        with self.assertRaisesRegex(vasp_task.TaskError, "exit code 137"):
            vasp_task.validate(self.scratch, self.prepared, 137)
        (self.scratch / "KPOINTS").write_text("changed\n")
        with self.assertRaisesRegex(vasp_task.TaskError, "KPOINTS changed during the run"):
            vasp_task.validate(self.scratch, self.prepared, 0)


class Finish(Tree):

    def setUp(self):
        super().setUp()
        self.scratch = self.prepare()

    def finish(self, exit_code=0):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            status = vasp_task.finish(self.folder(), self.scratch, exit_code, self.records)
        return status, out.getvalue()

    def test_success_exports_and_cleans_up(self):
        fake_run(self.scratch)
        status, out = self.finish()
        self.assertEqual(status, 0, out)
        self.assertTrue(out.startswith("OK calculations/CoGe_interface/100/sym00001/8x/convergence/e500_k1x9x9"))
        names = set(os.listdir(self.folder()))
        self.assertEqual(names, set(mvi.INPUTS) | set(vasp_task.EXPORT_ALWAYS) | {"result.json"})
        self.assertTrue((self.folder() / "POTCAR").is_symlink())
        self.assertFalse(self.scratch.exists())
        self.assertFalse(self.claim().exists())
        rec = json.loads((self.folder() / "result.json").read_text())
        self.assertEqual((rec["valid"], rec["sym"], rec["point"], rec["job"], rec["git"]),
                         (True, SYM, "e500_k1x9x9", "41_0", "abc1234"))
        self.assertEqual(rec["exported_sha256"]["OUTCAR"], vasp_task.digest(self.folder() / "OUTCAR"))
        with self.assertRaisesRegex(vasp_task.TaskError, r"\(refused\)"):  # never rerun in place
            self.prepare()

    def test_failure_keeps_scratch_and_claim(self):
        fake_run(self.scratch, OUTCAR=[(" General timing", " Gen timing")])
        status, out = self.finish()
        self.assertEqual(status, 1)
        self.assertIn("FAILED", out)
        self.assertEqual(set(os.listdir(self.folder())), set(mvi.INPUTS))
        self.assertTrue(self.scratch.exists())
        self.assertTrue(self.claim().exists())
        failed = json.loads((self.records / "failed" / f"{SYM}_8x_e500_k1x9x9_41_0.json").read_text())
        self.assertIn("VASP did not finish", failed["error"])
        self.assertEqual(failed["exit_code"], 0)

    def test_export_refuses_a_folder_that_gained_output(self):
        fake_run(self.scratch)
        (self.folder() / "OSZICAR").write_text("someone else's run\n")
        status, out = self.finish()
        self.assertEqual(status, 1)
        self.assertIn("gained VASP output", out)
        self.assertEqual((self.folder() / "OSZICAR").read_text(), "someone else's run\n")

    def test_export_names(self):
        self.assertEqual(vasp_task.export_names({"LCHARG": ".FALSE."}), list(vasp_task.EXPORT_ALWAYS))
        self.assertEqual(vasp_task.export_names({"LCHARG": ".TRUE.", "LORBIT": "11"})[-2:], ["CHGCAR", "PROCAR"])

    def test_scratch_removal_guard(self):
        prepared = json.loads((self.scratch / "prepare.json").read_text())
        with self.assertRaisesRegex(vasp_task.TaskError, "unsafe scratch removal"):
            vasp_task.safe_remove_scratch(self.scratch_base, prepared)
        self.assertTrue(self.scratch.exists())


class Parsing(unittest.TestCase):

    def test_job_rss_sums_only_this_jobs_vasp_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            procs = {"100": ("vasp_std", "42", "VmRSS:\t 1000 kB\n"), "101": ("vasp_std", "43", "VmRSS:\t 5000 kB\n"),
                     "102": ("python3", "42", "VmRSS:\t 7000 kB\n"), "103": ("vasp_std", "42", "State: Z\n"),
                     "104": ("vasp_ncl", "42", "VmRSS:\t 2000 kB\n"), "self": ("vasp_std", "42", "VmRSS:\t 9 kB\n")}
            for pid, (comm, job, status) in procs.items():
                (proc / pid).mkdir()
                (proc / pid / "comm").write_text(comm + "\n")
                (proc / pid / "environ").write_bytes(f"A=1\0SLURM_JOB_ID={job}\0B=2\0".encode())
                (proc / pid / "status").write_text("Name:\tx\n" + status)
            (proc / "105").mkdir()  # vanished process: no files
            self.assertEqual(vasp_task.job_rss("42", proc), (3000 * 1024, 2))

    def test_warning_boxes(self):
        boxes = vasp_task.warning_boxes(VASP_OUT)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["title"], "ADVICE to this user running VASP")
        self.assertTrue(boxes[0]["text"].startswith("You have a (more or less) 'large supercell'"))
        self.assertEqual(vasp_task.warning_boxes("no boxes here\n"), [])


class Submit(Tree):

    def rel(self, point):
        return str(self.folder(point).relative_to(self.root))

    def test_pending(self):
        (self.folder("e350_k1x9x9") / "OUTCAR").write_text("finished\n")
        self.claim("e400_k1x9x9").parent.mkdir(parents=True)
        self.claim("e400_k1x9x9").write_text("{}")
        missing = self.rel("e500_k1x9x9").replace(SYM, "sym00002")
        folders = [self.rel(p) for p in ("e350_k1x9x9", "e400_k1x9x9", "e450_k1x9x9")] + [missing]
        todo, skipped = submit_vasp.pending(folders, self.records)
        self.assertEqual(todo, [self.rel("e450_k1x9x9")])
        reasons = dict(skipped)
        self.assertEqual(reasons[self.rel("e350_k1x9x9")], "holds VASP output")
        self.assertTrue(reasons[self.rel("e400_k1x9x9")].startswith("claimed"))
        self.assertEqual(reasons[missing], "no INCAR (not generated)")

    def test_dry_run_command(self):
        tasks = self.records / "tasks.txt"
        tasks.parent.mkdir(parents=True)
        tasks.write_text("".join(self.rel(p) + "\n" for p in mvi.recipe(FAMILY, STAGE)["points"]))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            submit_vasp.main(["--tasks", str(tasks), "--only", "e500_k1x9x9", "e500_k1x3x3", "--mem", "64G",
                              "--dry-run"])
        text = out.getvalue()
        self.assertIn("2 pending folder(s), 16 ranks each", text)
        self.assertIn("--array=0-1%8 --mem=64G --ntasks=16 --ntasks-per-node=16", text)
        self.assertFalse((self.records / "submissions").exists())


if __name__ == "__main__":
    unittest.main()
