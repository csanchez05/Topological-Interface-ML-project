"""Reusable, non-interactive plotting for explicit bulk B20 VASP results.

The public :func:`generate_b20_plots` entry point intentionally accepts
repository paths instead of importing calculation-workflow internals.  The
CoGe plotting conventions are therefore reusable without imposing a shared
calculation driver or directory revision scheme.  POTCAR files are never
parsed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.electronic_structure.plotter import BSDOSPlotter, BSPlotter
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

from .relaxation import plot_relaxation_histories
from .coge_bulk.paths import PlotInputError, band_reference_stage, resolve_stage


_NORMAL_END_MARKER = b"General timing and accounting informations for this job"
_ORIENTATIONS = (("100", "100_regular"), ("110", "110"), ("111", "111"))
_ORBITALS = (
    OrbitalType.s,
    OrbitalType.p,
    OrbitalType.d,
    OrbitalType.f,
)
_ORBITALS_BY_NAME = {orbital.name: orbital for orbital in _ORBITALS}
_PYPROCAR_ORBITAL_INDICES = {
    "s": (0,),
    "p": (1, 2, 3),
    "d": (4, 5, 6, 7, 8),
}
_BLOCK_ORBITAL_ORDER = {
    "s": ("s",),
    "p": ("s", "p"),
    "d": ("d", "s"),
    "f": ("f", "d", "s"),
}
_REVIEWED_ORBITAL_SHELLS = {
    "Co": ("3d", "4s"),
    "Fe": ("3d", "4s"),
    "Cr": ("3d", "4s"),
    "Rh": ("4d", "5s"),
    "Ru": ("4d", "5s"),
    "Pd": ("4d", "5s"),
    "Os": ("5d", "6s"),
    "Au": ("5d", "6s"),
    "Si": ("3s", "3p"),
    "Ge": ("4s", "4p"),
    "Al": ("3s", "3p"),
    "Be": ("2s", "2p"),
}


class PlotUnavailable(RuntimeError):
    """Raised internally when a plot does not yet have validated inputs."""


def _stage(branch: Path, name: str) -> Path:
    try:
        return resolve_stage(branch, name)
    except PlotInputError as exc:
        raise PlotUnavailable(str(exc)) from exc


def _band_efermi(cache: _RunCache, branch: Path) -> float:
    try:
        reference = band_reference_stage(branch)
    except PlotInputError as exc:
        raise PlotUnavailable(str(exc)) from exc
    return float(cache.scf(reference).efermi)


def _branch_root(root: Path, soc: bool) -> Path:
    if root.name.startswith("sym") or (root / "non-soc").is_dir():
        return root / ("soc" if soc else "non-soc")
    return root / "SOC" if soc else root


class _RunCache:
    """Parse each potentially large VASP result at most once per invocation."""

    def __init__(self) -> None:
        self._scf: dict[Path, Vasprun] = {}
        self._dos: dict[Path, Vasprun] = {}
        self._bands: dict[tuple[Path, float], Any] = {}

    def scf(self, stage: Path) -> Vasprun:
        stage = stage.resolve()
        if stage not in self._scf:
            _require_completed_stage(stage)
            run = Vasprun(
                str(stage / "vasprun.xml"),
                parse_dos=True,
                parse_eigen=False,
                parse_potcar_file=False,
                exception_on_bad_xml=True,
            )
            if not run.converged_electronic:
                raise PlotUnavailable(
                    f"{stage}: electronic convergence was not reached"
                )
            if run.efermi is None:
                raise PlotUnavailable(f"{stage}: no SCF Fermi level is available")
            self._scf[stage] = run
            if run.complete_dos is not None:
                self._dos[stage] = run
        return self._scf[stage]

    def dos(self, stage: Path) -> Vasprun:
        stage = stage.resolve()
        if stage not in self._dos:
            _require_completed_stage(stage)
            run = Vasprun(
                str(stage / "vasprun.xml"),
                parse_dos=True,
                parse_eigen=False,
                parse_potcar_file=False,
                exception_on_bad_xml=True,
            )
            if not run.converged_electronic:
                raise PlotUnavailable(
                    f"{stage}: electronic convergence was not reached"
                )
            if run.efermi is None or run.complete_dos is None:
                raise PlotUnavailable(f"{stage}: complete DOS data are unavailable")
            self._dos[stage] = run
            self._scf[stage] = run
        return self._dos[stage]

    def bands(self, stage: Path, efermi: float) -> Any:
        stage = stage.resolve()
        key = (stage, float(efermi))
        if key not in self._bands:
            _require_completed_stage(stage)
            kpoints = stage / "KPOINTS"
            if not kpoints.is_file():
                raise PlotUnavailable(f"{stage}: KPOINTS is missing")
            validation_run = Vasprun(
                str(stage / "vasprun.xml"),
                parse_dos=False,
                parse_eigen=False,
                parse_potcar_file=False,
                exception_on_bad_xml=True,
            )
            if not validation_run.converged_electronic:
                raise PlotUnavailable(
                    f"{stage}: electronic convergence was not reached"
                )
            run = BSVasprun(
                str(stage / "vasprun.xml"),
                parse_projected_eigen=False,
                parse_potcar_file=False,
            )
            self._bands[key] = run.get_band_structure(
                kpoints_filename=str(kpoints),
                line_mode=True,
                efermi=efermi,
            )
        return self._bands[key]


def _outcar_ended_normally(path: Path) -> bool:
    if not path.is_file():
        return False
    chunk_size = 256 * 1024
    with path.open("rb") as handle:
        try:
            handle.seek(-chunk_size, 2)
        except OSError:
            handle.seek(0)
        return _NORMAL_END_MARKER in handle.read()


def _require_completed_stage(stage: Path) -> None:
    vasprun = stage / "vasprun.xml"
    if not vasprun.is_file() or vasprun.stat().st_size == 0:
        raise PlotUnavailable(f"{stage}: vasprun.xml is missing or empty")
    if not _outcar_ended_normally(stage / "OUTCAR"):
        raise PlotUnavailable(f"{stage}: normal VASP termination is not recorded")


def _save_figure(figure: Any, destination: Path, *, dpi: int = 300,
                 figure_size: tuple[float, float] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}")
    try:
        if figure_size is not None:
            figure.set_size_inches(*figure_size)
            figure.tight_layout()
        figure.savefig(temporary, dpi=dpi,
                       bbox_inches=None if figure_size is not None else "tight")
        os.replace(temporary, destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return destination


def _densities(dos: Any) -> np.ndarray:
    """Return total density for either one- or two-spin calculations."""

    return np.asarray(dos.get_densities(), dtype=float)


def _element_projection_metadata(element: Element) -> dict[str, Any]:
    """Return valence-shell labels for one element.

    VASP reports projected DOS by angular-momentum channel, without a
    principal quantum number.  Use pymatgen's ground-state electron
    configuration together with the element's period and block to assign the
    chemically relevant valence shells.  The period/block fallback retains
    an empty outer ``s`` shell for cases such as Pd, where hybridized 5s
    weight can still occur even though the atomic ground state is 4d10.
    """

    period = int(element.row)
    block = str(element.block)
    reviewed_shells = _REVIEWED_ORBITAL_SHELLS.get(element.symbol)
    orbital_order = (
        tuple(shell[-1] for shell in reviewed_shells)
        if reviewed_shells is not None
        else _BLOCK_ORBITAL_ORDER.get(block)
    )
    if orbital_order is None:
        raise PlotUnavailable(
            f"{element}: unsupported periodic-table block {block!r}"
        )

    shell_number = {
        "s": period,
        "p": period,
        "d": max(1, period - 1),
        "f": max(1, period - 2),
    }
    occupancies = {
        (int(principal), str(orbital)): float(occupancy)
        for principal, orbital, occupancy
        in element.full_electronic_structure
    }
    shell_labels = (
        {
            shell[-1]: shell
            for shell in reviewed_shells
        }
        if reviewed_shells is not None
        else {
            orbital: f"{shell_number[orbital]}{orbital}"
            for orbital in orbital_order
        }
    )
    return {
        "atomic_number": int(element.Z),
        "block": block,
        "electron_configuration": str(element.electronic_structure),
        "orbital_order": list(orbital_order),
        "shell_labels": shell_labels,
        "plot_labels": {
            orbital: f"{element.symbol}-{shell_labels[orbital]}"
            for orbital in orbital_order
        },
        "ground_state_occupancies": {
            orbital: occupancies.get(
                (shell_number[orbital], orbital),
                0.0,
            )
            for orbital in orbital_order
        },
    }


def _material_projection_metadata(formula: str) -> dict[str, Any]:
    """Build deterministic orbital-projection metadata for a material."""

    try:
        elements = Composition(formula).elements
    except Exception as exc:
        raise PlotUnavailable(
            f"{formula!r}: formula cannot be parsed for orbital projections"
        ) from exc
    if not elements:
        raise PlotUnavailable(
            f"{formula!r}: formula has no elements for orbital projections"
        )

    return {
        "schema_version": 1,
        "material": formula,
        "source": (
            "pymatgen.core.Element electronic_structure and "
            "full_electronic_structure"
        ),
        "species": {
            element.symbol: _element_projection_metadata(element)
            for element in elements
        },
    }


def _write_projection_metadata(
    formula: str,
    results_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Atomically write the per-material orbital-projection metadata."""

    metadata = _material_projection_metadata(formula)
    results_root.mkdir(parents=True, exist_ok=True)
    destination = results_root / f"{formula}_orbital_projection_metadata.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, metadata


def _plot_convergence(
    csv_path: Path,
    *,
    formula: str,
    kind: str,
    destination: Path,
) -> list[Path]:
    if not csv_path.is_file():
        raise PlotUnavailable(f"{csv_path}: convergence summary is missing")
    frame = pd.read_csv(csv_path)
    if "status" in frame:
        frame = frame[frame["status"].astype(str) == "completed_converged"]
    for column in ("electronic_converged", "normal_termination"):
        if column in frame:
            truthy = frame[column].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
            frame = frame[truthy]

    x_candidates = (
        ("candidate", "encut_eV")
        if kind == "encut"
        else ("candidate", "kmesh")
    )
    x_column = next((name for name in x_candidates if name in frame), None)
    if x_column is None:
        raise PlotUnavailable(
            f"{csv_path}: no supported {kind} candidate column is present"
        )

    frame = frame.copy()
    frame["_x"] = pd.to_numeric(frame[x_column], errors="coerce")
    if kind == "kpoints":
        mesh_number = (
            frame[x_column]
            .astype(str)
            .str.extract(r"^\s*(\d+)", expand=False)
        )
        frame["_x"] = frame["_x"].fillna(
            pd.to_numeric(mesh_number, errors="coerce")
        )
    frame = frame.dropna(subset=["_x"]).sort_values("_x")
    if len(frame) < 2:
        raise PlotUnavailable(
            f"{csv_path}: fewer than two completed, converged candidates"
        )

    if "delta_meV_per_atom" in frame:
        delta = pd.to_numeric(frame["delta_meV_per_atom"], errors="coerce")
    elif "energy_eV_per_atom" in frame:
        energy = pd.to_numeric(frame["energy_eV_per_atom"], errors="coerce")
        delta = 1000.0 * (energy - energy.iloc[-1])
    elif "energy_per_atom_meV" in frame:
        energy = pd.to_numeric(frame["energy_per_atom_meV"], errors="coerce")
        delta = energy - energy.iloc[-1]
    else:
        raise PlotUnavailable(
            f"{csv_path}: no supported per-atom energy column is present"
        )
    valid = delta.notna()
    frame = frame[valid]
    delta = delta[valid]
    if len(frame) < 2:
        raise PlotUnavailable(f"{csv_path}: converged energies are incomplete")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frame["_x"], delta, "o-")
    axis.axhline(0, color="black", linewidth=0.5)
    axis.axhline(1, linestyle="--", color="gray")
    axis.axhline(
        -1,
        linestyle="--",
        color="gray",
        label="±1 meV/atom",
    )
    if kind == "encut":
        axis.set_xlabel("ENCUT (eV)")
        axis.set_title(f"{formula} ENCUT convergence")
    else:
        axis.set_xlabel("k-mesh (N×N×N)")
        axis.set_title(f"{formula} k-point convergence")
    axis.set_ylabel("Energy/atom − densest setting (meV)")
    axis.legend()
    figure.tight_layout()
    return [_save_figure(figure, destination)]


def _plot_dos_total(
    cache: _RunCache,
    dos_stage: Path,
    *,
    formula: str,
    branch_label: str,
    destination: Path,
) -> list[Path]:
    run = cache.dos(dos_stage)
    complete_dos = run.complete_dos
    reference_efermi = float(run.efermi)
    energies = np.asarray(complete_dos.energies) - reference_efermi

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(energies, _densities(complete_dos), color="black", linewidth=1.5)
    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(-8, 4)
    axis.set_xlabel("E − E$_F$ (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(f"{formula} {branch_label} total DOS")
    figure.tight_layout()
    return [_save_figure(figure, destination)]


def _plot_dos_elements(
    cache: _RunCache,
    dos_stage: Path,
    *,
    formula: str,
    branch_label: str,
    destination: Path,
) -> list[Path]:
    run = cache.dos(dos_stage)
    complete_dos = run.complete_dos
    reference_efermi = float(run.efermi)
    energies = np.asarray(complete_dos.energies) - reference_efermi
    element_dos = complete_dos.get_element_dos()
    if not element_dos:
        raise PlotUnavailable(f"{dos_stage}: element-projected DOS is unavailable")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        energies,
        _densities(complete_dos),
        color="black",
        linewidth=1.5,
        label="Total",
    )
    for element, dos in element_dos.items():
        axis.plot(energies, _densities(dos), linewidth=1.5, label=str(element))
    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(-8, 4)
    axis.set_xlabel("E − E$_F$ (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(f"{formula} {branch_label} total and element DOS")
    axis.legend()
    figure.tight_layout()
    return [_save_figure(figure, destination)]


def _plot_dos_orbitals(
    cache: _RunCache,
    dos_stage: Path,
    *,
    formula: str,
    branch_label: str,
    projection_metadata: dict[str, Any],
    destination: Path,
) -> list[Path]:
    run = cache.dos(dos_stage)
    complete_dos = run.complete_dos
    reference_efermi = float(run.efermi)
    energies = np.asarray(complete_dos.energies) - reference_efermi
    element_dos = complete_dos.get_element_dos()
    if not element_dos:
        raise PlotUnavailable(f"{dos_stage}: projected DOS is unavailable")

    species_metadata = projection_metadata.get("species", {})
    projections: list[tuple[Any, list[tuple[Any, Any, str]]]] = []
    for element in element_dos:
        symbol = getattr(element, "symbol", str(element))
        configured = species_metadata.get(symbol)
        if not configured:
            raise PlotUnavailable(
                f"{dos_stage}: projection metadata is missing for {symbol}"
            )
        spd = complete_dos.get_element_spd_dos(element)
        curves: list[tuple[Any, Any, str]] = []
        for orbital_name in configured["orbital_order"]:
            orbital = _ORBITALS_BY_NAME[orbital_name]
            if orbital in spd:
                curves.append(
                    (
                        orbital,
                        spd[orbital],
                        configured["plot_labels"][orbital_name],
                    )
                )
        if curves:
            projections.append((element, curves))
    if not projections:
        raise PlotUnavailable(
            f"{dos_stage}: orbital-projected DOS is unavailable"
        )

    figure, axes = plt.subplots(
        len(projections),
        1,
        figsize=(7, max(4, 3.5 * len(projections))),
        sharex=True,
        squeeze=False,
    )
    for axis, (element, curves) in zip(axes[:, 0], projections):
        for _orbital, dos, label in curves:
            axis.plot(
                energies,
                _densities(dos),
                linewidth=1.5,
                label=label,
            )
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_ylabel("DOS (states/eV/cell)")
        axis.set_title(f"{element} orbital-projected DOS")
        axis.legend()
    axes[-1, 0].set_xlabel("E − E$_F$ (eV)")
    axes[-1, 0].set_xlim(-8, 4)
    figure.suptitle(f"{formula} {branch_label}", y=1.0)
    figure.tight_layout()
    return [_save_figure(figure, destination)]


def _band_structure(
    cache: _RunCache,
    branch: Path,
) -> Any:
    return cache.bands(_stage(branch, "bandstructure"), _band_efermi(cache, branch))


def _plot_bands(
    cache: _RunCache,
    branch: Path,
    *,
    formula: str,
    branch_label: str,
    destination: Path,
    dpi: int = 800,
    figure_size: tuple[float, float] | None = None,
) -> list[Path]:
    bands = _band_structure(cache, branch)
    axis = BSPlotter(bands).get_plot(
        zero_to_efermi=True,
        ylim=(-3, 3),
    )
    axis.set_title(f"{formula} {branch_label} band structure")
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(labelsize=12)
    axis.set_xticks(axis.get_xticks(), [label.get_text().replace("GAMMA", "Γ")
                                       for label in axis.get_xticklabels()])
    handles, _labels = axis.get_legend_handles_labels()
    if len(bands.bands) == 2 and len(handles) >= 2:
        axis.legend(handles[:2], ["Spin up", "Spin down"])
    elif axis.get_legend() is not None:
        axis.get_legend().remove()
    return [_save_figure(axis.get_figure(), destination, dpi=dpi, figure_size=figure_size)]


def _plot_bands_dos(
    cache: _RunCache,
    branch: Path,
    *,
    formula: str,
    branch_label: str,
    destination: Path,
    dpi: int = 800,
    figure_size: tuple[float, float] | None = None,
) -> list[Path]:
    bands = _band_structure(cache, branch)
    # Campaign bands use the matching DOS reference; direct stages use SCF.
    # BSDOSPlotter applies each object's own reference.
    dos = cache.dos(_stage(branch, "dos")).complete_dos
    plotter = BSDOSPlotter(
        bs_projection=None,
        dos_projection="elements",
        vb_energy_range=3,
        cb_energy_range=3,
        egrid_interval=1,
        font="DejaVu Sans",
    )
    plotted = plotter.get_plot(bands, dos=dos)
    if hasattr(plotted, "get_figure"):
        figure = plotted.get_figure()
    elif hasattr(plotted, "figure"):
        figure = plotted.figure
    else:
        figure = plt.gcf()
    band_axis = figure.axes[0]
    band_axis.set_xticks(band_axis.get_xticks(),
                        [label.get_text().replace("GAMMA", "Γ")
                         for label in band_axis.get_xticklabels()])
    # A noncollinear SOC result has spinor bands, not separate up/down bands.
    # BSDOSPlotter otherwise adds both spin legend entries unconditionally.
    if len(bands.bands) == 1 and band_axis.get_legend() is not None:
        band_axis.get_legend().remove()
    if len(figure.axes) > 1:
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        dos_axis = figure.axes[1]
        dos_axis.set_xlabel("DOS (states/eV/cell)", fontsize=13)
        dos_axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        dos_axis.xaxis.set_major_formatter(ScalarFormatter())
        dos_axis.tick_params(axis="x", labelsize=11)
    figure.suptitle(f"{formula} {branch_label} bands and DOS")
    return [_save_figure(figure, destination, dpi=dpi, figure_size=figure_size)]


def _plot_soc_overlay(
    cache: _RunCache,
    regular: Path,
    soc: Path,
    *,
    formula: str,
    destination: Path,
) -> list[Path]:
    regular_bands = _band_structure(cache, regular)
    soc_bands = _band_structure(cache, soc)
    plotter = BSPlotter(regular_bands)
    plotter.add_bs(soc_bands)
    axis = plotter.get_plot(zero_to_efermi=True, ylim=(-2, 2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colors[0],
                linewidth=1.25,
                label="non-SOC",
            ),
            Line2D(
                [],
                [],
                color=colors[1],
                linewidth=1.25,
                label="SOC",
            ),
        ],
        loc="upper right",
    )
    axis.set_title(f"{formula} SOC versus non-SOC bands")
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(labelsize=12)
    return [_save_figure(axis.get_figure(), destination, dpi=800)]


def _load_pyprocar() -> Any:
    """Import PyProcar with the compatibility shim used by the CoGe plots."""

    import pyvista.core.utilities as pyvista_utilities

    if not hasattr(pyvista_utilities, "NORMALS"):
        from pyvista.core.utilities.helpers import _NORMALS as pyvista_normals
        pyvista_utilities.NORMALS = pyvista_normals

    import pyprocar

    logging.getLogger("user").setLevel(logging.CRITICAL)
    for name, logger in logging.root.manager.loggerDict.items():
        if name == "pyprocar" or name.startswith("pyprocar."):
            if isinstance(logger, logging.Logger):
                logger.setLevel(logging.CRITICAL)
                for handler in logger.handlers:
                    handler.setLevel(logging.CRITICAL)

    return pyprocar


def _plot_soc_spin_projections(
    cache: _RunCache,
    branch: Path,
    *,
    formula: str,
    orientation: str,
    destination: Path,
    dpi: int = 300,
    figure_size: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot the noncollinear Sx, Sy, and Sz band projections with PyProcar.

    PyProcar writes parsed pickle caches beside the input files.  Run it from a
    temporary symlink mirror (node-local scratch when configured) so completed
    VASP directories remain untouched.
    """

    band_stage = _stage(branch, "bandstructure")
    _require_completed_stage(band_stage)
    efermi = _band_efermi(cache, branch)

    required = ("PROCAR", "OUTCAR", "POSCAR", "KPOINTS")
    for filename in required:
        source = band_stage / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise PlotUnavailable(f"{band_stage}: {filename} is missing or empty")

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=".pyprocar-",
        dir=os.environ.get("COGE_PLOT_SCRATCH", str(destination)),
    ) as temporary:
        mirror = Path(temporary)
        for filename in required:
            (mirror / filename).symlink_to(
                (band_stage / filename).resolve()
            )
        try:
            os.chdir(mirror)
            pyprocar = _load_pyprocar()
            for spin_index, label in ((1, "Sx"), (2, "Sy"), (3, "Sz")):
                output = (destination / f"soc_spin_{label}.png").resolve()
                figure, _axis = pyprocar.bandsplot(
                    code="vasp",
                    dirname=str(mirror),
                    mode="parametric",
                    fermi=efermi,
                    spins=[spin_index],
                    cmap="coolwarm",
                    clim=[-1, 1],
                    colorbar_title=f"Spin {label} projection",
                    elimit=[-2, 2],
                    legend=False,
                    title=(
                        f"{formula} {orientation} SOC bands - spin {label}"
                    ),
                    show=False,
                    dpi=dpi,
                    figure_size=figure_size or (9, 6),
                    use_cache=True,
                    verbose=0,
                    quiet_welcome=True,
                )
                _save_figure(figure, output, dpi=dpi, figure_size=figure_size)
                if not output.is_file() or output.stat().st_size == 0:
                    raise PlotUnavailable(
                        f"PyProcar did not create {output}"
                    )
                written.append(output)
        finally:
            os.chdir(previous_cwd)

    return written


def _plot_orbital_projected_bands(
    cache: _RunCache,
    branch: Path,
    *,
    formula: str,
    orientation: str,
    destination: Path,
    dpi: int = 300,
    figure_size: tuple[float, float] | None = None,
) -> list[Path]:
    """Reproduce the pristine-CoGe element/orbital band projections.

    The CoGe 100 plot suite contains one non-SOC parametric band figure for
    every element and each of the VASP s, p, and d projection channels.  Keep
    the same filenames and plotting convention for all binary B20 materials.
    PyProcar cache files are confined to a temporary mirror outside completed
    VASP directories, using node-local scratch when configured.
    """

    band_stage = _stage(branch, "bandstructure")
    _require_completed_stage(band_stage)
    efermi = _band_efermi(cache, branch)

    required = ("PROCAR", "OUTCAR", "POSCAR", "KPOINTS")
    for filename in required:
        source = band_stage / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise PlotUnavailable(f"{band_stage}: {filename} is missing or empty")

    try:
        structure = Structure.from_file(band_stage / "POSCAR")
    except Exception as exc:
        raise PlotUnavailable(
            f"{band_stage}: POSCAR cannot be parsed for atom projections"
        ) from exc

    atoms_by_symbol: dict[str, list[int]] = {}
    for index, site in enumerate(structure):
        if not site.is_ordered:
            raise PlotUnavailable(
                f"{band_stage}: disordered sites cannot be mapped to PROCAR"
            )
        symbol = site.specie.symbol
        atoms_by_symbol.setdefault(symbol, []).append(index)

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=".pyprocar-orbitals-",
        dir=os.environ.get("COGE_PLOT_SCRATCH", str(destination)),
    ) as temporary:
        mirror = Path(temporary)
        for filename in required:
            (mirror / filename).symlink_to((band_stage / filename).resolve())
        try:
            os.chdir(mirror)
            pyprocar = _load_pyprocar()
            for element in Composition(formula).elements:
                symbol = element.symbol
                atoms = atoms_by_symbol.get(symbol)
                if not atoms:
                    raise PlotUnavailable(
                        f"{band_stage}: no {symbol} atoms were found in POSCAR"
                    )
                for orbital, indices in _PYPROCAR_ORBITAL_INDICES.items():
                    output = (
                        destination / f"proj_{symbol}_{orbital}.png"
                    ).resolve()
                    figure, _axis = pyprocar.bandsplot(
                        code="vasp",
                        dirname=str(mirror),
                        mode="parametric",
                        fermi=efermi,
                        atoms=atoms,
                        orbitals=list(indices),
                        cmap="jet",
                        elimit=[-3, 3],
                        legend=False,
                        title=(
                            f"{formula} {orientation} {symbol}-{orbital} "
                            "projected bands"
                        ),
                        show=False,
                        dpi=dpi,
                        figure_size=figure_size or (9, 6),
                        use_cache=True,
                        verbose=0,
                        quiet_welcome=True,
                    )
                    _save_figure(figure, output, dpi=dpi, figure_size=figure_size)
                    if not output.is_file() or output.stat().st_size == 0:
                        raise PlotUnavailable(
                            f"PyProcar did not create {output}"
                        )
                    written.append(output)
        finally:
            os.chdir(previous_cwd)

    return written


def _first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _relaxation_stages(
    calculation_root: Path,
    configured_roots: Mapping[str, str | Path] | None,
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    """Resolve explicit relaxation roots without consulting workflow state."""

    stages: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    roots: Mapping[str, str | Path] = (
        {
            "preliminary": calculation_root / "relaxation" / "preliminary",
            "final": calculation_root / "relaxation" / "final",
        }
        if configured_roots is None
        else configured_roots
    )
    for name, configured_stage in roots.items():
        stage = Path(configured_stage)
        try:
            _require_completed_stage(stage)
        except PlotUnavailable as exc:
            skipped.append(
                {
                    "plot": f"relaxation.{name}",
                    "reason": str(exc),
                }
            )
            continue
        stages[name] = stage
    return stages, skipped


def generate_b20_plots(
    formula: str,
    calculation_root: str | Path,
    results_root: str | Path,
    output_root: str | Path,
    *,
    orientation_roots: Mapping[str, str | Path] | None = None,
    relaxation_roots: Mapping[str, str | Path] | None = None,
    convergence_files: Mapping[str, str | Path] | None = None,
    structure: str | None = None,
) -> dict[str, Any]:
    """Generate every currently available validated B20 plot.

    Parameters
    ----------
    formula
        Formula used by calculation directories, for example ``"CoSi"``.
    calculation_root
        Material directory, such as
        ``calculations/CoGe_Bulk``.
    results_root
        Directory containing campaign CSV/JSON summaries, usually
        ``results/b20_campaign``.
    output_root
        Shared DFT plot root, usually ``plots/dft_plots``.  The function
        appends ``<FORMULA>/bulk`` and creates only directories receiving a
        figure.
    orientation_roots
        Optional mapping for accepted ``"100"``, ``"110"``, and ``"111"``
        roots. A root may be a symmetry directory with non-soc/soc children
        or a direct non-SOC directory with a SOC child. When omitted, CoGe
        uses sym00001 for [100] and sym00000 for [110]/[111]. Completed
        campaign records select DOS/bands; final links select SCF results.
    relaxation_roots
        Optional explicit mapping of plot labels to accepted relaxation-stage
        directories.  When omitted, direct ``preliminary`` and ``final``
        paths are checked.
    convergence_files
        Optional explicit ``{"encut": ..., "kpoints": ...}`` mapping.  A
        supplied mapping is authoritative; omitted entries are skipped.
    structure
        Optional symmetry directory, such as ``"sym00000"``, applied to all
        default orientation roots. Explicit orientation_roots take precedence.

    Returns
    -------
    dict
        JSON-serializable report with ``plots``, ``skipped``, and ``errors``.
        Missing or incomplete stages are reported under ``skipped`` and never
        abort plots whose inputs are ready.
    """

    calculation_root = Path(calculation_root)
    results_root = Path(results_root)
    material_output = Path(output_root) / formula / "bulk"
    if orientation_roots is None:
        resolved_orientation_roots = {
            orientation: calculation_root / f"{formula}_{orientation}"
            for orientation, _output_name in _ORIENTATIONS
        }
        if structure is not None or formula == "CoGe":
            if structure is not None and (
                len(structure) != 8 or not structure.startswith("sym")
                or not structure[3:].isdigit()
            ):
                raise ValueError("structure must have the form sym00001")
            resolved_orientation_roots = {
                orientation: root / (structure or (
                    "sym00001" if orientation == "100" else "sym00000"
                ))
                for orientation, root in resolved_orientation_roots.items()
            }
    else:
        unsupported = sorted(set(orientation_roots) - {"100", "110", "111"})
        if unsupported:
            raise ValueError(
                f"unsupported plotting orientations: {unsupported}"
            )
        resolved_orientation_roots = {
            orientation: Path(path)
            for orientation, path in orientation_roots.items()
        }
    projection_metadata_path, projection_metadata = (
        _write_projection_metadata(formula, results_root)
    )
    report: dict[str, Any] = {
        "formula": formula,
        "calculation_root": str(calculation_root),
        "output_root": str(material_output),
        "orientation_roots": {
            orientation: str(path)
            for orientation, path in resolved_orientation_roots.items()
        },
        "relaxation_roots": (
            {
                label: str(path)
                for label, path in relaxation_roots.items()
            }
            if relaxation_roots is not None
            else {
                "preliminary": str(
                    calculation_root / "relaxation" / "preliminary"
                ),
                "final": str(calculation_root / "relaxation" / "final"),
            }
        ),
        "orbital_projection_metadata": str(projection_metadata_path),
        "soc_spin_convention": {
            "SAXIS": [0, 0, 1],
            "components": ["Sx", "Sy", "Sz"],
            "interpretation": (
                "k-resolved time-reversal partners can carry opposite spin; "
                "these projections are not bulk magnetization"
            ),
        },
        "plots": [],
        "skipped": [],
        "errors": [],
    }
    cache = _RunCache()

    def attempt(
        name: str,
        operation: Callable[[], list[Path]],
    ) -> None:
        try:
            written = operation()
        except PlotUnavailable as exc:
            report["skipped"].append({"plot": name, "reason": str(exc)})
        except Exception as exc:  # One malformed result must not block the rest.
            report["errors"].append(
                {
                    "plot": name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            if written:
                report["plots"].extend(str(path) for path in written)
            else:
                report["skipped"].append(
                    {"plot": name, "reason": "no plottable data were found"}
                )

    canonical = resolved_orientation_roots.get(
        "100",
        calculation_root / f"{formula}_100",
    )
    canonical_output = (
        material_output / "100" / canonical.name
        if canonical.name.startswith("sym") else material_output / "100_regular"
    )
    convergence_output = canonical_output / "convergence"
    if convergence_files is None:
        encut_csv = _first_existing(
            (
                results_root / f"{formula}_encut_convergence.csv",
                _branch_root(canonical, False) / "convergence" / "encut_convergence.csv",
            )
        )
        kpoints_csv = _first_existing(
            (
                results_root / f"{formula}_kpoints_convergence.csv",
                results_root / f"{formula}_kpoint_convergence.csv",
                _branch_root(canonical, False) / "convergence" / "kpoints_convergence.csv",
                _branch_root(canonical, False) / "convergence" / "kpoint_convergence.csv",
            )
        )
    else:
        unsupported = sorted(set(convergence_files) - {"encut", "kpoints"})
        if unsupported:
            raise ValueError(
                f"unsupported convergence file labels: {unsupported}"
            )
        encut_csv = Path(
            convergence_files.get(
                "encut",
                results_root / f"{formula}_encut.not_configured",
            )
        )
        kpoints_csv = Path(
            convergence_files.get(
                "kpoints",
                results_root / f"{formula}_kpoints.not_configured",
            )
        )
    attempt(
        "convergence.encut",
        lambda: _plot_convergence(
            encut_csv,
            formula=formula,
            kind="encut",
            destination=convergence_output / "encut_convergence.png",
        ),
    )
    attempt(
        "convergence.kpoints",
        lambda: _plot_convergence(
            kpoints_csv,
            formula=formula,
            kind="kpoints",
            destination=convergence_output / "kpoint_convergence.png",
        ),
    )

    relaxations, relaxation_skips = _relaxation_stages(
        calculation_root,
        relaxation_roots,
    )
    report["skipped"].extend(relaxation_skips)
    if relaxations:
        attempt(
            "relaxation.history",
            lambda: plot_relaxation_histories(
                relaxations,
                canonical_output / "relaxation",
            ),
        )
    else:
        report["skipped"].append(
            {
                "plot": "relaxation.history",
                "reason": "no completed accepted relaxation root is available",
            }
        )

    for orientation, output_name in _ORIENTATIONS:
        official_root = resolved_orientation_roots.get(orientation)
        if official_root is None:
            report["skipped"].append(
                {
                    "plot": orientation,
                    "reason": (
                        "no explicit accepted orientation root was supplied"
                    ),
                }
            )
            continue
        if official_root.name.startswith("sym"):
            output_name = str(Path(orientation) / official_root.name)
        for soc_enabled in (False, True):
            branch_name = "soc" if soc_enabled else (
                "non-soc" if official_root.name.startswith("sym") else "non_soc"
            )
            branch_label = "SOC" if soc_enabled else "non-SOC"
            branch = _branch_root(official_root, soc_enabled)
            destination = material_output / output_name / branch_name
            prefix = f"{orientation}.{branch_name}"
            attempt(
                f"{prefix}.dos_total",
                lambda branch=branch,
                branch_label=branch_label,
                destination=destination: _plot_dos_total(
                    cache,
                    _stage(branch, "dos"),
                    formula=formula,
                    branch_label=branch_label,
                    destination=destination / "dos_total_only.png",
                ),
            )
            attempt(
                f"{prefix}.dos_elements",
                lambda branch=branch,
                branch_label=branch_label,
                destination=destination: _plot_dos_elements(
                    cache,
                    _stage(branch, "dos"),
                    formula=formula,
                    branch_label=branch_label,
                    destination=destination / "dos_total_elements.png",
                ),
            )
            attempt(
                f"{prefix}.dos_orbitals",
                lambda branch=branch,
                branch_label=branch_label,
                destination=destination: _plot_dos_orbitals(
                    cache,
                    _stage(branch, "dos"),
                    formula=formula,
                    branch_label=branch_label,
                    projection_metadata=projection_metadata,
                    destination=destination / "dos_orbitals.png",
                ),
            )
            band_filename = (
                "bands_soc.png" if soc_enabled else "bands_nosoc.png"
            )
            attempt(
                f"{prefix}.bands",
                lambda branch=branch,
                branch_label=branch_label,
                destination=destination,
                band_filename=band_filename: _plot_bands(
                    cache,
                    branch,
                    formula=formula,
                    branch_label=branch_label,
                    destination=destination / band_filename,
                ),
            )
            attempt(
                f"{prefix}.bands_dos",
                lambda branch=branch,
                branch_label=branch_label,
                destination=destination: _plot_bands_dos(
                    cache,
                    branch,
                    formula=formula,
                    branch_label=branch_label,
                    destination=destination / "bands_dos_combined.png",
                ),
            )
            if soc_enabled:
                attempt(
                    f"{prefix}.spin_projections",
                    lambda branch=branch,
                    orientation=orientation,
                    destination=destination: _plot_soc_spin_projections(
                        cache,
                        branch,
                        formula=formula,
                        orientation=orientation,
                        destination=destination,
                    ),
                )
            elif orientation == "100":
                attempt(
                    f"{prefix}.orbital_projected_bands",
                    lambda branch=branch,
                    orientation=orientation,
                    destination=destination: _plot_orbital_projected_bands(
                        cache,
                        branch,
                        formula=formula,
                        orientation=orientation,
                        destination=destination,
                    ),
                )

        attempt(
            f"{orientation}.comparison.bands_soc_vs_nosoc",
            lambda official_root=official_root,
            output_name=output_name: _plot_soc_overlay(
                cache,
                _branch_root(official_root, False),
                _branch_root(official_root, True),
                formula=formula,
                destination=material_output
                / output_name
                / "comparison"
                / "bands_soc_vs_nosoc.png",
            ),
        )

    return report


__all__ = ["generate_b20_plots"]
