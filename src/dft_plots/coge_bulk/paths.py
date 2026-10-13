"""Resolve plotting inputs in the CoGe bulk calculation tree.

Each stage folder (``scf``, ``dos``, ``bandstructure``) holds its VASP files
directly, for example ``CoGe_100/sym00000/non-soc/dos/vasprun.xml``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PlotInputError(RuntimeError):
    """A selected plotting input is missing or incomplete."""


def _selected_run(stage: Path) -> Path:
    final = stage / "final"
    return final if final.exists() or final.is_symlink() else stage


def resolve_stage(branch: Path, stage: str) -> Path:
    """Return the folder holding a stage's VASP files; never guess another run.

    A ``final`` link inside the stage is still followed for other B20 data.
    A stage without a non-empty vasprun.xml is an error, not a fallback.
    """
    selected = _selected_run(Path(branch) / stage)
    vasprun = selected / "vasprun.xml"
    if not vasprun.is_file() or vasprun.stat().st_size == 0:
        raise PlotInputError(f"Missing completed {stage} input: {vasprun}")
    return selected.resolve()


def band_reference_stage(branch: Path) -> Path:
    """Bands with an energy_reference.json use the branch's DOS Fermi level.

    The CoGe_100 bands were referenced to the 15³ DOS run of the same
    branch. Bands without that record keep the SCF Fermi level.
    """
    branch = Path(branch)
    bands = _selected_run(branch / "bandstructure")
    stage = "dos" if (bands / "energy_reference.json").is_file() else "scf"
    return resolve_stage(branch, stage)


@dataclass
class PlotPaths:
    project_root: Path
    orientation: str
    structure: str
    category: str
    output_override: Path | None = None
    _stages: dict[str, Path] = field(default_factory=dict)

    @property
    def calculation_dir(self) -> Path:
        return (self.project_root / "calculations" / "CoGe_Bulk"
                / f"CoGe_{self.orientation}" / self.structure)

    @property
    def output_dir(self) -> Path:
        if self.output_override is not None:
            return self.output_override
        return (self.project_root / "plots" / "dft_plots" / "CoGe" / "bulk"
                / self.orientation / self.structure / self.category)

    def branch(self, soc: bool = False) -> Path:
        return self.calculation_dir / ("soc" if soc else "non-soc")

    def stage(self, key: str) -> Path:
        if key not in self._stages:
            soc = key.startswith("soc_")
            name = key.removeprefix("soc_")
            branch = self.branch(soc)
            self._stages[key] = (
                band_reference_stage(branch) if name == "reference"
                else resolve_stage(branch, name)
            )
        return self._stages[key]


def add_path_arguments(parser: argparse.ArgumentParser, *, structure: str | None,
                       orientation: str = "100") -> None:
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Project containing calculations/, src/, and plots/")
    parser.add_argument("--orientation", choices=("100", "110", "111"),
                        default=orientation)
    parser.add_argument("--structure", default=structure,
                        help=f"Symmetry directory (default: {structure or 'selected by --variant'})")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the figure destination directory")
    parser.add_argument("--show-paths", action="store_true",
                        help="Check and print input/output paths without plotting")


def paths_from_args(args: argparse.Namespace, category: str) -> PlotPaths:
    if not re.fullmatch(r"sym\d{5}", args.structure):
        raise PlotInputError("--structure must have the form sym00001")
    return PlotPaths(args.project_root.expanduser().resolve(), args.orientation,
                     args.structure, category,
                     args.output_dir.expanduser().resolve() if args.output_dir else None)


def check_paths(paths: PlotPaths, required: tuple[str, ...], *, show: bool) -> None:
    resolved = {}
    errors = []
    for key in required:
        try:
            resolved[key] = str(paths.stage(key))
        except PlotInputError as exc:
            errors.append(str(exc))
    if show:
        print(json.dumps({"calculation_dir": str(paths.calculation_dir),
                          "inputs": resolved, "output_dir": str(paths.output_dir),
                          "missing": errors}, indent=2))
        raise SystemExit(1 if errors else 0)
    if errors:
        raise PlotInputError("\n".join(errors))


def configure_plot(*, orientation: str, structure: str, category: str,
                   required: tuple[str, ...]) -> PlotPaths:
    parser = argparse.ArgumentParser(description="Plot CoGe calculation results.")
    add_path_arguments(parser, structure=structure, orientation=orientation)
    try:
        args = parser.parse_args()
        paths = paths_from_args(args, category)
        check_paths(paths, required, show=args.show_paths)
    except PlotInputError as exc:
        parser.error(str(exc))
    return paths
