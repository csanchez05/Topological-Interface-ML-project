"""Tests of make_vasp_inputs.py (CLAUDE.md section 7.4).

  python3 -B -m unittest discover -s src/vasp_inputs -p 'test_*.py'
  python3 -B src/vasp_inputs/test_make_vasp_inputs.py --update-goldens   # after a deliberate recipe change

(a) goldens: the script's own INCAR/KPOINTS for fixed structures, byte for byte
(b) the bulk anchor reproduces the finished CoGe_100 non-soc/scf runs; the GB recipe differs from it by
    exactly the declared overrides
(c) refusal / skip / stop on a temporary tree
(d) POSCAR checks
Reads the real GB POSCARs and bulk INCARs (read-only); writes only to temporary folders.
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "tests/golden"
GOLDEN_CASES = [("CoGe_100_GB", "convergence-nonsoc", sym, "8x", point)
                for sym in ("sym00177", "sym00174")
                for point in ("e500_k1x9x9", "e350_k1x9x9", "e500_k1x3x3")]
GOLDEN_HASH = "GOLDEN"
BULK_LIVE = ("sym00000", "sym00311", "sym00622")


def golden_dir(family, stage, sym, thickness, point):
    return GOLDEN / family / stage / sym / thickness / point


def synthetic_poscar(counts=(64, 64), scale="1.0", species="Co Ge", rows=None):
    n = sum(counts) if rows is None else rows
    coords = "".join(f"{(i % 16) / 16:.6f} {(i // 16 % 4) / 4:.6f} {(i // 64) / 2:.6f}\n" for i in range(n))
    return (f"Synthetic GB\n{scale}\n74.07119751 0.0 0.0\n0.0 4.62944984 0.0\n0.0 0.0 4.62944984\n"
            f"{species}\n{counts[0]} {counts[1]}\nDirect\n{coords}")


def update_goldens():
    for case in GOLDEN_CASES:
        _, files, _ = mvi.plan_folder(*case, GOLDEN_HASH)
        out = golden_dir(*case)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("INCAR", "KPOINTS"):
            (out / name).write_text(files[name])
        print("wrote", out.relative_to(mvi.PROJECT_ROOT))


class Goldens(unittest.TestCase):
    """(a) The script's own output for fixed structures, byte for byte."""

    def test_goldens_match(self):
        for case in GOLDEN_CASES:
            with self.subTest(case=case):
                _, files, _ = mvi.plan_folder(*case, GOLDEN_HASH)
                for name in ("INCAR", "KPOINTS"):
                    self.assertEqual(files[name], (golden_dir(*case) / name).read_text(), name)

    def test_sweep_has_nine_unique_points(self):
        points = mvi.recipe("CoGe_100_GB", "convergence-nonsoc")["points"]
        self.assertEqual(len(points), 9)
        self.assertEqual(points["e500_k1x9x9"], (500, (1, 9, 9)))
        self.assertEqual(sorted(e for e, m in points.values() if m == (1, 9, 9)), [350, 400, 450, 500, 600])
        self.assertEqual(sorted(m[1] for e, m in points.values() if e == 500), [3, 5, 7, 9, 11])

    def test_small_mesh_uses_kpar_1(self):
        r = mvi.recipe("CoGe_100_GB", "convergence-nonsoc")
        self.assertEqual(mvi.kpar_for(r, (1, 3, 3)), 1)
        self.assertEqual(mvi.kpar_for(r, (1, 5, 5)), 4)


class BulkAnchor(unittest.TestCase):
    """(b) Tags equal the finished bulk runs; the GB recipe differs by exactly the declared overrides."""

    def test_anchor_reproduces_live_bulk_runs(self):
        incar = mvi.tags(mvi.render_incar("CoGe_100", "scf-nonsoc", "scf", (4, 4), "X"))
        kpoints = mvi.render_kpoints("CoGe_100", "scf-nonsoc", "scf")
        for sym in BULK_LIVE:
            live = mvi.PROJECT_ROOT / f"calculations/CoGe_Bulk/CoGe_100/{sym}/non-soc/scf"
            with self.subTest(sym=sym):
                self.assertEqual(incar, mvi.tags((live / "INCAR").read_text()))
                self.assertEqual(kpoints, (live / "KPOINTS").read_text())

    def test_gb_recipe_differs_only_by_declared_overrides(self):
        anchor = mvi.tags(mvi.render_incar("CoGe_100", "scf-nonsoc", "scf", (64, 64), "X"))
        gb = mvi.tags(mvi.render_incar("CoGe_100_GB", "convergence-nonsoc", "e500_k1x9x9", (64, 64), "X"))
        changed = {k: (anchor[k], gb[k]) for k in anchor.keys() & gb.keys() if anchor[k] != gb[k]}
        self.assertEqual(changed, {
            "SYSTEM": ("CoGe B20 SCF", "CoGe B20 [100] GB convergence"),
            "ENCUT": ("600", "500"), "EDIFF": ("1E-8", "1E-6"), "ISMEAR": ("-1", "1"),
            "LCHARG": (".TRUE.", ".FALSE."), "ICHARG": ("1", "2"), "KPAR": ("3", "4"), "NCORE": ("1", "4")})
        added = {k: gb[k] for k in gb.keys() - anchor.keys()}
        self.assertEqual(added, {"NBANDS": "512", "AMIX": "0.2", "AMIX_MAG": "0.8", "AMIN": "0.01"})
        self.assertEqual(anchor.keys() - gb.keys(), {"LORBIT"})
        self.assertEqual(gb["MAGMOM"], "64*1.0 64*0.0")
        self.assertEqual(gb["SIGMA"], "0.05")

    def test_stamp_mask_ignores_only_the_hash(self):
        a = mvi.render_incar("CoGe_100_GB", "convergence-nonsoc", "e500_k1x9x9", (64, 64), "abc1234")
        b = mvi.render_incar("CoGe_100_GB", "convergence-nonsoc", "e500_k1x9x9", (64, 64), "fff9999")
        self.assertTrue(mvi.same_incar(a, b))
        self.assertFalse(mvi.same_incar(a, b.replace(f"v{mvi.VERSION}", "v9.9.9")))
        self.assertFalse(mvi.same_incar(a, b.replace("ENCUT = 500", "ENCUT = 501")))

    def test_base_recipe_is_never_written(self):
        with self.assertRaisesRegex(mvi.RecipeError, "base recipe"):
            mvi.generate("CoGe_100", "scf-nonsoc", ["sym00000"], "8x", None, True, "X")


class TempTree(unittest.TestCase):
    """(c) Refused / identical / stop on a temporary project tree."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        r = mvi.recipe("CoGe_100_GB", "convergence-nonsoc")
        (self.root / r["potcar"]).parent.mkdir(parents=True)
        (self.root / r["potcar"]).write_text("synthetic stand-in, not a potential\n")
        for template in (r["poscar"], r["catalogue"]):
            path = self.root / template.format(sym="sym00001", thickness="8x")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(synthetic_poscar())

    def run_gen(self, dry_run=False, git_hash="abc1234", points=None):
        return mvi.generate("CoGe_100_GB", "convergence-nonsoc", ["sym00001"], "8x", points, dry_run,
                            git_hash, self.root)

    def folder(self, point="e500_k1x9x9"):
        return self.root / f"calculations/CoGe_interface/100/sym00001/8x/convergence/{point}"

    def test_write_then_identical(self):
        counts, problems, ready, _ = self.run_gen()
        self.assertEqual((counts["written"], problems, len(ready)), (9, [], 9))
        f = self.folder()
        self.assertEqual(sorted(os.listdir(f)), ["INCAR", "KPOINTS", "POSCAR", "POTCAR"])
        self.assertEqual(os.readlink(f / "POTCAR"), str(self.root / "calculations/CoGe_interface/POTCAR"))
        self.assertTrue((f / "INCAR").read_text().startswith(
            f"# make_vasp_inputs.py v{mvi.VERSION} (git abc1234) family=CoGe_100_GB stage=convergence-nonsoc\n"))
        counts, problems, _, _ = self.run_gen(git_hash="fff9999")  # only the stamp hash differs
        self.assertEqual((counts["identical"], counts["written"], problems), (9, 0, []))

    def test_dry_run_writes_nothing(self):
        counts, problems, _, _ = self.run_gen(dry_run=True)
        self.assertEqual((counts["new (dry-run)"], problems), (9, []))
        self.assertFalse(self.folder().exists())

    def test_folder_with_vasp_output_is_refused(self):
        self.run_gen()
        (self.folder() / "OUTCAR").write_text("finished run\n")
        (self.folder() / "INCAR").write_text("ENCUT = 1\n")  # even different inputs: refused, not stop
        counts, problems, _, refused = self.run_gen()
        self.assertEqual((counts["refused"], counts["identical"], problems), (1, 8, []))
        self.assertEqual(refused, [self.folder()])

    def test_different_inputs_stop(self):
        self.run_gen()
        (self.folder("e350_k1x9x9") / "KPOINTS").write_text("changed\n")
        (self.folder("e400_k1x9x9") / "notes.txt").write_text("extra\n")
        counts, problems, _, _ = self.run_gen()
        self.assertEqual(counts["identical"], 7)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("different: KPOINTS" in p for p in problems))
        self.assertTrue(any("unexpected file" in p for p in problems))

    def test_a_problem_blocks_every_write(self):
        self.run_gen()
        new = self.folder("e600_k1x9x9")
        for name in os.listdir(new):
            os.remove(new / name)
        new.rmdir()
        (self.folder("e350_k1x9x9") / "KPOINTS").write_text("changed\n")
        counts, problems, ready, _ = self.run_gen()
        self.assertEqual((counts["written"], counts["not written"], counts["identical"], len(problems), ready),
                         (0, 1, 7, 1, []))
        self.assertFalse(new.exists())

    def test_catalogue_mismatch_stops(self):
        r = mvi.recipe("CoGe_100_GB", "convergence-nonsoc")
        (self.root / r["catalogue"].format(sym="sym00001", thickness="8x")).write_text(
            synthetic_poscar().replace("Synthetic GB", "Other"))
        counts, problems, _, _ = self.run_gen()
        self.assertEqual(counts["written"], 0)
        self.assertTrue(all("not a byte copy of the catalogue" in p for p in problems))


class Poscar(unittest.TestCase):
    """(d) Scale fix, empty file, Co-then-Ge order, atom count."""

    def test_good(self):
        self.assertEqual(mvi.check_poscar(synthetic_poscar(), (64, 64))["atoms"], 128)

    def test_rejects(self):
        cases = {
            "double-scaled": synthetic_poscar(scale="4.6294498444"),
            "scale must be 1.0": synthetic_poscar(scale="2.0"),
            "empty POSCAR": "",
            "species must be 'Co Ge'": synthetic_poscar(species="Ge Co"),
            "atom counts": synthetic_poscar(counts=(63, 65)),
            "coordinate rows": synthetic_poscar(rows=100),
            "unreadable scale": synthetic_poscar(scale="x"),
            "scale must be 1.0, got 74": synthetic_poscar().replace("Synthetic GB\n1.0", "Synthetic GB 1.0"),
        }
        for message, text in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(mvi.RecipeError, message):
                    mvi.check_poscar(text, (64, 64))

    def test_selected_real_cells_pass(self):
        for sym in ("sym00485", "sym00174", "sym00016", "sym00177"):
            text = (mvi.PROJECT_ROOT / f"calculations/CoGe_interface/100/{sym}/8x/POSCAR").read_text()
            self.assertEqual(mvi.check_poscar(text, (64, 64))["atoms"], 128)


if __name__ == "__main__":
    if "--update-goldens" in sys.argv:
        update_goldens()
    else:
        unittest.main()
