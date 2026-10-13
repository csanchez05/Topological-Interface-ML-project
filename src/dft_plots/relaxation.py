"""Relaxation-history plotting shared by B20 materials."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp import Vasprun


def _history(stage: Path) -> dict[str, list[float]] | None:
    vasprun = stage / "vasprun.xml"
    if not vasprun.is_file() or vasprun.stat().st_size == 0:
        return None
    run = Vasprun(
        str(vasprun),
        parse_potcar_file=False,
        parse_dos=False,
        parse_eigen=False,
        exception_on_bad_xml=False,
    )
    if not run.ionic_steps:
        return None
    energies: list[float] = []
    forces: list[float] = []
    volumes: list[float] = []
    stresses: list[float] = []
    moments: list[float] = []
    for step in run.ionic_steps:
        energies.append(float(step.get("e_0_energy", step.get("e_fr_energy", np.nan))))
        force = np.asarray(step.get("forces", []), dtype=float)
        forces.append(
            float(np.max(np.linalg.norm(force, axis=1))) if force.size else np.nan
        )
        structure = step.get("structure")
        volumes.append(float(structure.volume) if structure is not None else np.nan)
        stress = np.asarray(step.get("stress", []), dtype=float)
        stresses.append(
            float(np.max(np.abs(np.linalg.eigvalsh(stress))))
            if stress.size
            else np.nan
        )
        electronic = step.get("electronic_steps", [])
        moment = np.nan
        if electronic:
            moment = float(electronic[-1].get("mag", np.nan))
        moments.append(moment)
    return {
        "energy_eV": energies,
        "maximum_force_eV_per_A": forces,
        "volume_A3": volumes,
        "maximum_principal_stress_kbar": stresses,
        "magnetic_moment_muB": moments,
    }


def plot_relaxation_histories(
    stages: Mapping[str, Path], output_dir: Path
) -> list[Path]:
    """Plot available histories without reading a POTCAR."""

    histories = {
        label: history
        for label, path in stages.items()
        if (history := _history(path)) is not None
    }
    if not histories:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("energy_eV", "Total energy (eV)", "relaxation_energy.png"),
        (
            "maximum_force_eV_per_A",
            "Maximum force (eV/Å)",
            "relaxation_max_force.png",
        ),
        ("volume_A3", "Cell volume (Å³)", "relaxation_volume.png"),
        (
            "maximum_principal_stress_kbar",
            "Maximum principal stress (kbar)",
            "relaxation_stress.png",
        ),
        (
            "magnetic_moment_muB",
            "Magnetic moment (μB)",
            "relaxation_magnetic_moment.png",
        ),
    ]
    written: list[Path] = []
    for key, ylabel, filename in specs:
        if all(np.all(np.isnan(history[key])) for history in histories.values()):
            continue
        figure, axis = plt.subplots(figsize=(6, 4))
        for label, history in histories.items():
            values = np.asarray(history[key], dtype=float)
            axis.plot(np.arange(1, len(values) + 1), values, "o-", label=label)
        axis.set_xlabel("Ionic step")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        destination = output_dir / filename
        figure.savefig(destination, dpi=300)
        plt.close(figure)
        written.append(destination)
    return written
