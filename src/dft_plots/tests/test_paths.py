"""Regression checks for stage selection in the flat calculation layout."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dft_plots.coge_bulk.paths import (
    PlotInputError, band_reference_stage, resolve_stage,
)


class StagePathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.structure = (self.root / "calculations" / "CoGe_Bulk"
                          / "CoGe_100" / "sym00001")
        self.branch = self.structure / "non-soc"

    def run_directory(self, path):
        path.mkdir(parents=True)
        (path / "vasprun.xml").write_text("<modeling />")
        return path

    def dos_referenced_bands(self, branch):
        bands = self.run_directory(branch / "bandstructure")
        (bands / "energy_reference.json").write_text(json.dumps({"fermi_eV": 9.0}))
        return bands

    def test_stages_resolve_to_their_own_folders(self):
        for stage in ("scf", "dos", "bandstructure"):
            selected = self.run_directory(self.branch / stage)
            self.assertEqual(resolve_stage(self.branch, stage), selected)

    def test_missing_or_empty_vasprun_is_an_error(self):
        with self.assertRaisesRegex(PlotInputError, "Missing completed dos"):
            resolve_stage(self.branch, "dos")
        dos = self.run_directory(self.branch / "dos")
        (dos / "vasprun.xml").write_text("")
        with self.assertRaisesRegex(PlotInputError, "Missing completed dos"):
            resolve_stage(self.branch, "dos")

    def test_dos_referenced_bands_use_their_own_branch_dos(self):
        self.run_directory(self.branch / "scf")
        non_soc_dos = self.run_directory(self.branch / "dos")
        self.dos_referenced_bands(self.branch)
        soc = self.structure / "soc"
        self.run_directory(soc / "scf")
        soc_dos = self.run_directory(soc / "dos")
        self.dos_referenced_bands(soc)
        self.assertEqual(band_reference_stage(self.branch), non_soc_dos)
        self.assertEqual(band_reference_stage(soc), soc_dos)

    def test_missing_dos_reference_does_not_fall_back_to_scf(self):
        self.run_directory(self.branch / "scf")
        self.dos_referenced_bands(self.branch)
        with self.assertRaisesRegex(PlotInputError, "Missing completed dos"):
            band_reference_stage(self.branch)

    def test_bands_without_energy_reference_keep_scf(self):
        scf = self.run_directory(self.branch / "scf")
        self.run_directory(self.branch / "dos")
        self.run_directory(self.branch / "bandstructure")
        self.assertEqual(band_reference_stage(self.branch), scf)

    def test_final_link_is_followed(self):
        selected = self.run_directory(self.branch / "scf" / "ismear_m1")
        (self.branch / "scf" / "final").symlink_to("ismear_m1")
        self.assertEqual(resolve_stage(self.branch, "scf"), selected)
        (selected / "vasprun.xml").unlink()
        with self.assertRaises(PlotInputError):
            resolve_stage(self.branch, "scf")

    def test_cli_relocation_and_output_override_from_another_directory(self):
        selected = self.run_directory(self.branch / "dos")
        script = Path(__file__).resolve().parents[1] / "coge_bulk" / "plot_dos.py"
        output = self.root / "custom-figures"
        result = subprocess.run(
            [sys.executable, "-B", str(script), "--project-root", str(self.root),
             "--output-dir", str(output), "--show-paths"],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["inputs"]["dos"], str(selected))
        self.assertEqual(data["output_dir"], str(output))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
