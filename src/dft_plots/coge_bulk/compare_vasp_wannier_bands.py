#!/usr/bin/env python3
"""Compare matching direct VASP and Wannier90 bands for non-SOC B20 CoGe."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.io.vasp.inputs import Kpoints

if __package__:
    from .paths import PlotInputError, add_path_arguments, paths_from_args
else:
    from paths import PlotInputError, add_path_arguments, paths_from_args


ENERGY_LIMITS = (-10.0, 10.0)
IMAGE_SUFFIXES = frozenset({".png", ".pdf", ".svg"})

VASP_COLOR = "#000000"
WANNIER_COLOR = "#FF002A"


class BandComparisonError(RuntimeError):
    """Raised when a required input or expected band-path detail is invalid."""


def find_repository_root(script_path: Path) -> Path:
    """Find the repository root independently of the caller's working directory."""
    for candidate in script_path.resolve().parents:
        if (
            (candidate / "calculations" / "CoGe_Bulk").is_dir()
            and (candidate / "src" / "dft_plots").is_dir()
        ):
            return candidate

    raise BandComparisonError(
        "Could not locate the repository root above "
        f"{script_path.resolve()}. Expected both calculations/CoGe_Bulk "
        "and src/dft_plots directories."
    )


def require_nonempty_file(path: Path, description: str) -> Path:
    """Return *path* after checking that it is a nonempty regular file."""
    if not path.is_file():
        raise BandComparisonError(
            f"Required {description} file was not found: {path}"
        )
    if path.stat().st_size == 0:
        raise BandComparisonError(f"Required {description} file is empty: {path}")
    return path


def read_wannier_band_blocks(path: Path) -> list[np.ndarray]:
    """Read blank-line-separated bands from a Wannier90 ``*_band.dat`` file."""
    import numpy as np

    bands: list[np.ndarray] = []
    current_band: list[tuple[float, float]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                if current_band:
                    bands.append(np.asarray(current_band, dtype=float))
                    current_band = []
                continue

            fields = stripped.split()
            if len(fields) < 2:
                raise BandComparisonError(
                    f"Malformed Wannier band row at {path}:{line_number}; "
                    "expected at least two numeric columns."
                )
            try:
                current_band.append((float(fields[0]), float(fields[1])))
            except ValueError as exc:
                raise BandComparisonError(
                    f"Non-numeric Wannier band row at {path}:{line_number}: "
                    f"{stripped!r}"
                ) from exc

    if current_band:
        bands.append(np.asarray(current_band, dtype=float))

    if not bands:
        raise BandComparisonError(f"No band blocks were found in {path}")

    for band_number, band in enumerate(bands, start=1):
        if band.shape[0] < 2 or band.shape[1] != 2:
            raise BandComparisonError(
                f"Wannier band {band_number} in {path} does not contain a valid "
                "two-column path."
            )
        if not np.isfinite(band).all():
            raise BandComparisonError(
                f"Wannier band {band_number} in {path} contains non-finite values."
            )

    return bands


def canonical_label(label: str) -> str:
    """Normalize common Gamma spellings and path-separator whitespace."""
    cleaned = label.strip().replace("$", "").replace("{", "").replace("}", "")
    cleaned = cleaned.replace(r"\Gamma", "G").replace("Γ", "G")
    cleaned = re.sub(r"(?i)gamma", "G", cleaned)
    return "|".join(part.strip().upper() for part in cleaned.split("|"))


def read_wannier_ticks(path: Path) -> tuple[list[float], list[str]]:
    """Extract high-symmetry tick positions and labels from Wannier90 GNU output."""
    text = path.read_text(encoding="utf-8")
    xtics_match = re.search(r"set\s+xtics\s*\((.*?)\)", text, flags=re.DOTALL)
    if xtics_match is None:
        raise BandComparisonError(
            f"Could not find a 'set xtics (...)' definition in {path}"
        )

    number = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?"
    matches = re.findall(rf'"([^"]+)"\s+({number})', xtics_match.group(1))
    if not matches:
        raise BandComparisonError(
            f"No high-symmetry labels and positions could be parsed from {path}"
        )

    positions: list[float] = []
    labels: list[str] = []
    for label, position_text in matches:
        position = float(position_text)
        if positions and abs(position - positions[-1]) < 1e-10:
            continue
        positions.append(position)
        labels.append(canonical_label(label))

    if len(positions) < 2 or any(
        right <= left for left, right in zip(positions, positions[1:])
    ):
        raise BandComparisonError(
            f"High-symmetry positions in {path} are not strictly increasing: "
            f"{positions}"
        )

    return positions, labels


def vasp_path_labels(kpoints: Kpoints, path: Path) -> list[str]:
    """Convert paired VASP line-mode endpoints into combined plot tick labels."""
    labels = kpoints.labels
    if not labels or len(labels) != len(kpoints.kpts):
        raise BandComparisonError(
            f"VASP line-mode labels could not be parsed from {path}"
        )
    if len(labels) % 2:
        raise BandComparisonError(
            f"VASP line-mode KPOINTS must contain paired segment endpoints: {path}"
        )
    if any(label is None for label in labels):
        raise BandComparisonError(
            f"Every VASP path endpoint must have a label in {path}"
        )

    segments = [
        (canonical_label(labels[index]), canonical_label(labels[index + 1]))
        for index in range(0, len(labels), 2)
    ]
    nodes = [segments[0][0]]
    previous_end: str | None = None

    for segment_number, (start, end) in enumerate(segments):
        if segment_number and previous_end != start:
            nodes[-1] = f"{previous_end}|{start}"
        nodes.append(end)
        previous_end = end

    return nodes


def load_vasp_bands(vasprun_path: Path, kpoints_path: Path, efermi: float):
    """Parse VASP bands with the same pymatgen logic used by existing plots."""
    from pymatgen.io.vasp.outputs import BSVasprun

    try:
        vasprun = BSVasprun(str(vasprun_path), parse_potcar_file=False)
        return vasprun.get_band_structure(
            kpoints_filename=str(kpoints_path),
            line_mode=True,
            efermi=efermi,
        )
    except Exception as exc:
        raise BandComparisonError(
            "Failed to parse the direct non-SOC VASP band structure from "
            f"{vasprun_path} with KPOINTS {kpoints_path}: {exc}"
        ) from exc


def shared_vasp_coordinate(
    band_structure,
    kpoints: Kpoints,
    tick_positions: list[float],
) -> np.ndarray:
    """Map each VASP line segment onto the Wannier90 high-symmetry positions."""
    import numpy as np

    segment_count = len(kpoints.kpts) // 2
    if len(tick_positions) != segment_count + 1:
        raise BandComparisonError(
            "VASP and Wannier paths have different segment counts: "
            f"{segment_count} VASP segments versus "
            f"{len(tick_positions) - 1} Wannier segments."
        )

    point_count = len(band_structure.kpoints)
    coordinate = np.full(point_count, np.nan, dtype=float)
    branches = getattr(band_structure, "branches", [])

    if len(branches) == segment_count:
        for index, branch in enumerate(branches):
            start = int(branch["start_index"])
            end = int(branch["end_index"])
            if start < 0 or end < start or end >= point_count:
                break
            values = np.linspace(
                tick_positions[index],
                tick_positions[index + 1],
                end - start + 1,
            )
            existing = coordinate[start : end + 1]
            overlap = np.isfinite(existing)
            if overlap.any() and not np.allclose(existing[overlap], values[overlap]):
                break
            coordinate[start : end + 1] = values
        else:
            if np.isfinite(coordinate).all():
                return coordinate

    points_per_segment = int(kpoints.num_kpts)
    expected_count = segment_count * points_per_segment
    if point_count != expected_count:
        raise BandComparisonError(
            "Could not map the VASP k-points onto the shared path: parsed "
            f"{point_count} k-points, but {segment_count} segments with "
            f"{points_per_segment} points each imply {expected_count}."
        )

    for index in range(segment_count):
        start = index * points_per_segment
        end = start + points_per_segment
        coordinate[start:end] = np.linspace(
            tick_positions[index],
            tick_positions[index + 1],
            points_per_segment,
        )
    return coordinate


def latex_tick_label(label: str) -> str:
    """Format a canonical path label for Matplotlib math text."""
    symbols = [r"\Gamma" if part == "G" else part for part in label.split("|")]
    if len(symbols) == 1:
        return f"${symbols[0]}$"
    return rf"${symbols[0]}\,|\,{symbols[1]}$"


def plot_comparison(
    band_structure,
    vasp_coordinate: np.ndarray,
    wannier_bands: list[np.ndarray],
    tick_positions: list[float],
    tick_labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot all VASP and Wannier90 bands with one style per dataset."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    plotted_vasp_bands = 0
    for spin_bands in band_structure.bands.values():
        if spin_bands.shape[1] != len(vasp_coordinate):
            raise BandComparisonError(
                "A parsed VASP band has a different k-point count from the "
                "shared path coordinate."
            )
        for energies in spin_bands:
            ax.plot(
                vasp_coordinate,
                energies - band_structure.efermi,
                color=VASP_COLOR,
                linewidth=0.9,
                linestyle="-",
                alpha=0.78,
                zorder=2,
            )
            plotted_vasp_bands += 1

    if plotted_vasp_bands == 0:
        raise BandComparisonError("No direct VASP bands were parsed.")

    for band in wannier_bands:
        ax.plot(
            band[:, 0],
            band[:, 1] - band_structure.efermi,
            color=WANNIER_COLOR,
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            alpha=0.9,
            zorder=3,
        )

    for position in tick_positions[1:-1]:
        ax.axvline(
            position,
            color="#737373",
            linewidth=0.7,
            alpha=0.65,
            zorder=0,
        )
    ax.axhline(
        0.0,
        color="#000000",
        linewidth=1.0,
        alpha=0.65,
        zorder=1,
    )

    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_ylim(*ENERGY_LIMITS)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([latex_tick_label(label) for label in tick_labels])
    ax.set_xlabel("Wave-vector path", fontsize=14)
    ax.set_ylabel(r"$E-E_F$ (eV)", fontsize=14)
    ax.set_title(title, fontsize=15, pad=12)
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.yaxis.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.65)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [],
            [],
            color=VASP_COLOR,
            linewidth=1.5,
            linestyle="-",
            label="Direct VASP",
        ),
        Line2D(
            [],
            [],
            color=WANNIER_COLOR,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label="Wannier90 interpolation",
        ),
        Line2D(
            [],
            [],
            color="#000000",
            linewidth=1.0,
            label=r"$E_F$",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    require_nonempty_file(output_path, "generated comparison image")


def available_destination(source: Path, destination_dir: Path) -> Path:
    """Return a collision-free destination without overwriting an existing image."""
    destination = destination_dir / source.name
    if not destination.exists():
        return destination

    for index in range(1, 10_000):
        candidate = destination_dir / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate

    raise BandComparisonError(
        f"Could not find a collision-free destination for {source.name} in "
        f"{destination_dir}"
    )


def move_wannier_images(
    wannier_dir: Path,
    destination_dir: Path,
) -> list[tuple[Path, Path]]:
    """Move only top-level generated image files out of the calculation directory."""
    image_sources = sorted(
        path
        for path in wannier_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    for source in image_sources:
        require_nonempty_file(source, "Wannierization image")

    destination_dir.mkdir(parents=True, exist_ok=True)
    moves: list[tuple[Path, Path]] = []
    for source in image_sources:
        destination = available_destination(source, destination_dir)
        expected_size = source.stat().st_size
        try:
            shutil.move(str(source), str(destination))
        except OSError as exc:
            raise BandComparisonError(
                f"Failed to move Wannierization image {source} to {destination}: "
                f"{exc}"
            ) from exc

        require_nonempty_file(destination, "moved Wannierization image")
        if destination.stat().st_size != expected_size:
            raise BandComparisonError(
                f"Moved image size changed unexpectedly: {source} -> {destination}"
            )
        moves.append((source, destination))

    return moves


def relative_to_repository(path: Path, repository_root: Path) -> str:
    """Format a path relative to the repository for concise run reporting."""
    return str(path.relative_to(repository_root)) if path.is_relative_to(repository_root) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("regular", "inverted"),
        default="regular",
        help="CoGe calculation branch to compare (default: regular)",
    )
    add_path_arguments(parser, structure=None)
    parser.add_argument("--wannier-dir", type=Path,
                        help="Directory containing the matching Wannier90 band data")
    args = parser.parse_args()

    args.structure = args.structure or ("sym00001" if args.variant == "regular" else "sym00000")
    paths = paths_from_args(args, "comparison")
    repository_root = paths.project_root
    vasp_dir = paths.stage("bandstructure")
    reference_dir = paths.stage("reference")
    wannier_dir = (args.wannier_dir.expanduser().resolve() if args.wannier_dir
                   else paths.branch() / "wannier")
    if args.show_paths:
        missing = [str(wannier_dir / name)
                   for name in ("wannier90_band.dat", "wannier90_band.gnu")
                   if not (wannier_dir / name).is_file()]
        print(json.dumps({"bandstructure": str(vasp_dir), "reference": str(reference_dir),
                          "wannier_dir": str(wannier_dir), "output_dir": str(paths.output_dir),
                          "missing": missing}, indent=2))
        return 1 if missing else 0

    vasp_xml = require_nonempty_file(
        vasp_dir / "vasprun.xml", "direct VASP vasprun.xml"
    )
    vasp_kpoints_path = require_nonempty_file(
        vasp_dir / "KPOINTS", "direct VASP KPOINTS"
    )
    wannier_dat = require_nonempty_file(
        wannier_dir / "wannier90_band.dat", "Wannier90 band data"
    )
    wannier_gnu = require_nonempty_file(
        wannier_dir / "wannier90_band.gnu", "Wannier90 GNU plot metadata"
    )

    # Path inspection and missing-input errors also work without plotting dependencies.
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from pymatgen.io.vasp.inputs import Kpoints
    from pymatgen.io.vasp.outputs import Vasprun

    reference = Vasprun(str(reference_dir / "vasprun.xml"), parse_dos=True,
                        parse_eigen=False, parse_potcar_file=False)
    if reference.efermi is None:
        raise BandComparisonError(f"No Fermi energy in {reference_dir}")

    try:
        vasp_kpoints = Kpoints.from_file(str(vasp_kpoints_path))
    except Exception as exc:
        raise BandComparisonError(
            f"Failed to parse VASP line-mode path from {vasp_kpoints_path}: {exc}"
        ) from exc

    wannier_bands = read_wannier_band_blocks(wannier_dat)
    tick_positions, tick_labels = read_wannier_ticks(wannier_gnu)
    direct_path_labels = vasp_path_labels(vasp_kpoints, vasp_kpoints_path)

    if tick_labels != direct_path_labels:
        raise BandComparisonError(
            f"Wannier path in {wannier_gnu} is {tick_labels}, but the direct "
            f"VASP path in {vasp_kpoints_path} is {direct_path_labels}."
        )

    for band_number, band in enumerate(wannier_bands, start=1):
        if not np.isclose(band[0, 0], tick_positions[0], atol=1e-5) or not np.isclose(
            band[-1, 0], tick_positions[-1], atol=1e-5
        ):
            raise BandComparisonError(
                f"Wannier band {band_number} does not span the GNU path range "
                f"{tick_positions[0]} to {tick_positions[-1]}."
            )

    band_structure = load_vasp_bands(vasp_xml, vasp_kpoints_path, float(reference.efermi))
    vasp_coordinate = shared_vasp_coordinate(
        band_structure,
        vasp_kpoints,
        tick_positions,
    )

    comparison_image = paths.output_dir / "vasp_vs_wannier_non_soc.png"
    wannierization_dir = paths.output_dir.parent / "wannierization"

    plot_comparison(
        band_structure,
        vasp_coordinate,
        wannier_bands,
        tick_positions,
        tick_labels,
        comparison_image,
        f"CoGe ({paths.orientation}) {paths.structure} non-SOC: Direct VASP vs Wannier90",
    )
    moved_images = move_wannier_images(wannier_dir, wannierization_dir)

    print(
        "Created:",
        relative_to_repository(comparison_image, repository_root),
        f"({comparison_image.stat().st_size} bytes)",
    )
    if moved_images:
        for source, destination in moved_images:
            print(
                "Moved:",
                relative_to_repository(source, repository_root),
                "->",
                relative_to_repository(destination, repository_root),
                f"({destination.stat().st_size} bytes)",
            )
    else:
        print("Moved: no Wannierization image files were present.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BandComparisonError, PlotInputError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
