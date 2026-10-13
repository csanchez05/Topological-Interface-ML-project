#!/usr/bin/env python3
"""Labels for every grain boundary, read from the three Information.txt files.

"Interface N" in Information.txt is folder symNNNNN. Each interface lists every
operation (rotation part plus translation) that builds it, written in the frame
of that plane's bulk cell. This script turns the rotation parts into the cubic
frame, so that labels mean the same thing on all three planes, and derives:

  grain_class  which second grain the operation makes: the rotation modulo the
               12 rotations of the B20 crystal itself
  handedness   whether the second grain has the same or the opposite chirality
  first_op     the first listed rotation
  rot_set      the full set of listed rotations

Writes gb_labels.csv (one row per GB), gb_operations.csv (one row per listed
operation) and gb_labels.json into calculations/ML_dscribe/CoGe/labels/.
"""
import argparse
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import ml_common as mc
from ml_common import sc

# Rows are the bulk-cell axes of each plane in cubic lattice coordinates.
# verify_frame() checks them against the reference bulk cells.
FRAMES = {
    "100": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "110": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float),
    "111": np.array([[1, 1, 1], [-1, 1, 0], [-1, -1, 2]], dtype=float),
}
LABEL_SCHEMES = ("handedness", "grain_class", "first_op", "rot_set")
_INTERFACE = re.compile(r"^Interface (\d+) : Num of Syms = (\d+)\s*$")
_SUMMARY = re.compile(r"^(\d+): (\d+) symmetry operations, x distance = (\S+), total distance = (\S+)\s*$")
_OPERATION = re.compile(r"^(.+?) : (.+?) \[\[(.*?)\]\]\s*$")
_NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- parsing

def _read_operation(lines, i):
    """Parse one operation starting at its header line. Returns (operation, next line)."""
    match = _OPERATION.match(lines[i])
    name, xyz = match.group(1).strip(), match.group(2).strip()
    numbers, j = [], i + 1
    while len(numbers) < 12 and j < len(lines):
        numbers.extend(float(v) for v in _NUMBER.findall(lines[j]))
        j += 1
    if len(numbers) != 12:
        raise ValueError(f"line {i + 1}: expected a 3 x 4 matrix after '{lines[i].strip()}'")
    matrix = np.array(numbers).reshape(3, 4)
    header_shift = np.array([float(v) for v in _NUMBER.findall(match.group(3))])
    if header_shift.shape != (3,) or not np.allclose(header_shift, matrix[:, 3], atol=1e-6):
        raise ValueError(f"line {i + 1}: translation in the header does not match the matrix")
    return {"name": name, "xyz": xyz, "matrix": matrix}, j


def parse_operations(path):
    """Every operation in a file such as Sym_of_Crystal.txt."""
    lines = Path(path).read_text().splitlines()
    operations, i = [], 0
    while i < len(lines):
        if _OPERATION.match(lines[i]):
            operation, i = _read_operation(lines, i)
            operations.append(operation)
        else:
            i += 1
    return operations


def parse_information(path):
    """Parse one Information.txt.

    Returns (interfaces, header). interfaces is a list ordered by interface
    number; each entry has number, x_distance, total_distance and operations.
    Raises ValueError when the file does not agree with itself.
    """
    lines = Path(path).read_text().splitlines()
    header, summary, interfaces = {}, {}, []
    current, i = None, 0
    while i < len(lines):
        line = lines[i]
        match = _INTERFACE.match(line)
        if match:
            current = {"number": int(match.group(1)), "n_declared": int(match.group(2)),
                       "x_distance": float(lines[i + 1].split("=")[1]),
                       "total_distance": float(lines[i + 2].split("=")[1]), "operations": []}
            interfaces.append(current)
            i += 3
        elif current is not None and _OPERATION.match(line):
            operation, i = _read_operation(lines, i)
            current["operations"].append(operation)
        elif current is None and _SUMMARY.match(line):
            number, count = int(_SUMMARY.match(line).group(1)), int(_SUMMARY.match(line).group(2))
            summary[number] = count
            i += 1
        else:
            if current is None and line.startswith("Number of") and ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = int(value)
            i += 1

    numbers = [entry["number"] for entry in interfaces]
    if numbers != list(range(len(numbers))):
        raise ValueError(f"{path}: interface numbers are not 0..{len(numbers) - 1}")
    for entry in interfaces:
        if len(entry["operations"]) != entry["n_declared"]:
            raise ValueError(f"{path}: interface {entry['number']} declares {entry['n_declared']} operations "
                             f"but {len(entry['operations'])} were read")
        if summary and summary.get(entry["number"]) != entry["n_declared"]:
            raise ValueError(f"{path}: interface {entry['number']} disagrees with the summary list")
    declared = header.get("Number of Unique Simple Interfaces")
    if declared is not None and declared != len(interfaces):
        raise ValueError(f"{path}: header declares {declared} interfaces but {len(interfaces)} were read")
    if summary and len(summary) != len(interfaces):
        raise ValueError(f"{path}: summary lists {len(summary)} interfaces but {len(interfaces)} were read")
    return interfaces, header


# --------------------------------------------------------------------------- rotations

def crystal_rotations():
    """The 12 proper rotations of B20 (point group 23), in the cubic frame."""
    generators = [np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float), np.diag([-1.0, -1.0, 1.0])]
    group = [np.eye(3)]
    grew = True
    while grew:
        grew = False
        for member in list(group):
            for generator in generators:
                candidate = member @ generator
                if not any(np.allclose(candidate, other) for other in group):
                    group.append(candidate)
                    grew = True
    return group


def to_cubic(rotation, plane):
    """Rotation written for fractional coordinates of a plane's bulk cell -> cubic Cartesian frame."""
    frame = FRAMES[plane]
    lengths = np.linalg.norm(frame, axis=1)
    axes = frame / lengths[:, None]
    cartesian = np.diag(lengths) @ rotation @ np.diag(1.0 / lengths)
    return axes.T @ cartesian @ axes


def _axis_text(axis):
    smallest = np.abs(axis)[np.abs(axis) > 1e-6].min()
    scaled = axis / smallest
    if np.allclose(scaled, np.round(scaled), atol=1e-6):
        return "[" + "".join(str(int(round(v))) for v in scaled) + "]"
    return "[" + ",".join(f"{v:.3f}" for v in axis) + "]"


def rotation_parts(rotation):
    """(determinant, angle in degrees, unit axis) of an orthogonal matrix.

    For an improper matrix the angle and axis are those of the matrix times -1.
    The axis of a 180 degree rotation is given with its first non-zero entry positive.
    """
    det = int(round(np.linalg.det(rotation)))
    proper = rotation * det
    angle = float(np.degrees(np.arccos(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0))))
    if angle < 1e-4:
        return det, 0.0, np.zeros(3)
    if abs(angle - 180.0) < 1e-4:
        projector = (proper + np.eye(3)) / 2.0
        axis = projector[:, np.argmax(np.linalg.norm(projector, axis=0))]
        axis = axis / np.linalg.norm(axis)
        axis = axis * np.sign(axis[np.flatnonzero(np.abs(axis) > 1e-6)[0]])
        return det, 180.0, axis
    axis = np.array([proper[2, 1] - proper[1, 2], proper[0, 2] - proper[2, 0], proper[1, 0] - proper[0, 1]])
    return det, angle, axis / np.linalg.norm(axis)


def rotation_name(rotation):
    """Short name in the cubic frame: E, I, R120[111], I.R180[001] (a mirror), ..."""
    det, angle, axis = rotation_parts(rotation)
    prefix = "" if det > 0 else "I."
    if angle == 0.0:
        return "E" if det > 0 else "I"
    return f"{prefix}R{angle:.4g}{_axis_text(axis)}"


def grain_class(rotation, group=None):
    """Name of the coset rotation * (crystal rotations): which second grain this rotation makes.

    Two rotations give the same second grain, up to a shift, exactly when they
    differ by a rotation of the crystal. The class is named after its member
    with the smallest rotation angle.
    """
    group = group or crystal_rotations()
    members = []
    for symmetry in group:
        det, angle, axis = rotation_parts(rotation @ symmetry)
        members.append((round(angle, 4), tuple(-np.round(axis, 4) + 0.0), rotation @ symmetry))
    members.sort(key=lambda member: member[:2])
    return rotation_name(members[0][2])


# --------------------------------------------------------------------------- tables

def build_labels(gb_root=None):
    """Returns (labels, operations, headers).

    labels: DataFrame, one row per GB family. operations: DataFrame, one row per
    listed operation. headers: the counts printed at the top of each file.
    """
    import pandas as pd

    gb_root = Path(gb_root or sc.GB_ROOT)
    group = crystal_rotations()
    family_rows, operation_rows, headers = [], [], {}
    for plane in sc.PLANES:
        interfaces, headers[plane] = parse_information(gb_root / f"CoGe_{plane}_GBs" / "Interface" / "Information.txt")
        for entry in interfaces:
            sym = f"sym{entry['number']:05d}"
            family = f"{plane}/{sym}"
            names, classes, dets = [], [], []
            for position, operation in enumerate(entry["operations"]):
                cubic = to_cubic(operation["matrix"][:, :3], plane)
                if not np.allclose(cubic @ cubic.T, np.eye(3), atol=1e-8):
                    raise ValueError(f"{family}: operation {operation['name']} is not a rotation in the cubic frame")
                names.append(rotation_name(cubic))
                classes.append(grain_class(cubic, group))
                dets.append(int(round(np.linalg.det(cubic))))
                shift = operation["matrix"][:, 3]
                operation_rows.append({
                    "family": family, "plane": plane, "sym": sym, "position": position,
                    "name_as_written": operation["name"], "xyz_as_written": operation["xyz"],
                    "rotation_cubic": names[-1], "grain_class": classes[-1], "det": dets[-1],
                    "t1": f"{shift[0]:.8g}", "t2": f"{shift[1]:.8g}", "t3": f"{shift[2]:.8g}",
                    "rotation_cell_frame": " ".join(f"{v:g}" for v in operation["matrix"][:, :3].ravel()),
                })
            if len(set(classes)) != 1:
                raise ValueError(f"{family}: listed operations fall into several grain classes {sorted(set(classes))}")
            family_rows.append({
                "family": family, "plane": plane, "sym": sym, "n_ops": len(names),
                "n_rotations": len(set(names)),
                "handedness": "same" if dets[0] > 0 else "opposite",
                "grain_class": classes[0],
                "first_op": names[0],
                "rot_set": "|".join(sorted(set(names))),
                "first_op_as_written": entry["operations"][0]["name"],
                "x_distance": entry["x_distance"], "total_distance": entry["total_distance"],
            })
    labels = pd.DataFrame(family_rows)
    # Short identifiers for the rotation sets; the members stay in their own column.
    sets = sorted(labels["rot_set"].unique())
    identifier = {members: f"RS{i:03d}" for i, members in enumerate(sets)}
    labels.insert(labels.columns.get_loc("rot_set") + 1, "rotations", labels["rot_set"])
    labels["rot_set"] = labels["rot_set"].map(identifier)
    return labels, pd.DataFrame(operation_rows), headers


def load_labels(gb_root=None):
    """The label table, built from the Information.txt files each time (fast)."""
    return build_labels(gb_root)[0]


# --------------------------------------------------------------------------- verification

def same_periodic_set(frac_a, species_a, frac_b, species_b, tol=2e-3):
    """True when two sets of fractional coordinates in the same cell hold the same atoms."""
    if len(frac_a) != len(frac_b):
        return False
    used = np.zeros(len(frac_b), dtype=bool)
    for position, species in zip(frac_a, species_a):
        delta = frac_b - position
        delta -= np.round(delta)
        hits = np.flatnonzero((np.abs(delta).max(axis=1) < tol) & (species_b == species) & ~used)
        if hits.size == 0:
            return False
        used[hits[0]] = True
    return True


def _fractional(atoms):
    return atoms.get_scaled_positions(wrap=True), np.array(atoms.get_chemical_symbols())


def verify_frame(plane, gb_root=None):
    """True when the pristine cubic cell, re-expressed with FRAMES[plane], is that plane's bulk cell."""
    gb_root = Path(gb_root or sc.GB_ROOT)
    pristine, _ = sc.read_structure(gb_root / "POSCAR")
    reference, _ = sc.read_structure(gb_root / f"CoGe_{plane}_GBs" / "Bulk" / f"CoGe_{plane}_POSCAR.vasp")
    frac, species = _fractional(pristine)
    frame = FRAMES[plane]
    if not np.allclose(reference.cell.lengths(), np.linalg.norm(frame, axis=1) * pristine.cell.lengths()[0], atol=1e-4):
        return False
    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=float)
    points = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ np.linalg.inv(frame)
    inside = np.all((points > -1e-6) & (points < 1 - 1e-6), axis=1)
    return same_periodic_set(points[inside], np.tile(species, len(shifts))[inside], *_fractional(reference))


def verify_crystal_symmetries(gb_root=None):
    """Checks on Sym_of_Crystal.txt: the 100 file must list exactly the 12 crystal rotations,
    and the rotations listed for 110 and 111 must be crystal rotations once in the cubic frame."""
    gb_root = Path(gb_root or sc.GB_ROOT)
    group = crystal_rotations()
    result = {}
    for plane in sc.PLANES:
        operations = parse_operations(gb_root / f"CoGe_{plane}_GBs" / "Bulk" / "Sym_of_Crystal.txt")
        cubic = [to_cubic(op["matrix"][:, :3], plane) for op in operations]
        inside = all(any(np.allclose(r, g, atol=1e-8) for g in group) for r in cubic)
        distinct = {rotation_name(r) for r in cubic}
        result[plane] = {"n_listed": len(operations), "n_distinct_rotations": len(distinct),
                         "all_are_crystal_rotations": bool(inside)}
    result["100"]["is_the_full_group"] = result["100"]["n_distinct_rotations"] == len(group)
    return result


def verify_bulk_cells(gb_root=None, tol=2e-3):
    """Apply every listed operation to the plane's bulk cell and compare with Bulk/symNNNNN/POSCAR.

    Returns {plane: {"n_families", "n_all_operations_match", "failures"}}. A match means
    the operation, read in the bulk cell's fractional frame, reproduces the stored
    second-grain cell exactly as a periodic set of atoms.
    """
    gb_root = Path(gb_root or sc.GB_ROOT)
    result = {}
    for plane in sc.PLANES:
        bulk = gb_root / f"CoGe_{plane}_GBs" / "Bulk"
        reference, _ = sc.read_structure(bulk / f"CoGe_{plane}_POSCAR.vasp")
        frac, species = _fractional(reference)
        interfaces, _ = parse_information(gb_root / f"CoGe_{plane}_GBs" / "Interface" / "Information.txt")
        failures = []
        for entry in interfaces:
            sym = f"sym{entry['number']:05d}"
            path = bulk / sym / "POSCAR"
            if not path.is_file():
                failures.append(f"{plane}/{sym}: no bulk cell")
                continue
            cell, _ = sc.read_structure(path)
            if not np.allclose(cell.cell.lengths(), reference.cell.lengths(), atol=1e-4):
                failures.append(f"{plane}/{sym}: cell differs from the reference")
                continue
            target = _fractional(cell)
            for operation in entry["operations"]:
                matrix = operation["matrix"]
                if not same_periodic_set(frac @ matrix[:, :3].T + matrix[:, 3], species, *target, tol=tol):
                    failures.append(f"{plane}/{sym}: {operation['name']} does not reproduce the bulk cell")
                    break
        result[plane] = {"n_families": len(interfaces), "n_all_operations_match": len(interfaces) - len(failures),
                         "failures": failures[:20]}
    return result


# --------------------------------------------------------------------------- command line

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gb-root", type=Path, default=sc.GB_ROOT)
    parser.add_argument("--out-dir", type=Path, default=mc.LABELS_DIR)
    parser.add_argument("--skip-bulk-check", action="store_true",
                        help="do not compare every operation with the stored second-grain cells")
    args = parser.parse_args()

    labels, operations, headers = build_labels(args.gb_root)
    counts = labels.groupby("plane").size().to_dict()
    folders = {plane: len(list((args.gb_root / f"CoGe_{plane}_GBs" / "Interface").glob("sym[0-9]*")))
               for plane in sc.PLANES}
    checks = {
        "interfaces_parsed": counts,
        "interface_folders": folders,
        "parsed_equals_folders": counts == folders,
        "frames_verified": {plane: bool(verify_frame(plane, args.gb_root)) for plane in sc.PLANES},
        "crystal_symmetry_files": verify_crystal_symmetries(args.gb_root),
    }
    if not args.skip_bulk_check:
        checks["operations_reproduce_bulk_cells"] = verify_bulk_cells(args.gb_root)

    classes = {}
    for scheme in ("plane",) + LABEL_SCHEMES:
        classes[scheme] = {"all": int(labels[scheme].nunique()),
                           **{plane: int(labels.loc[labels["plane"] == plane, scheme].nunique())
                              for plane in sc.PLANES}}
    summary = {
        "gb_root": mc.relative_path(args.gb_root),
        "n_families": len(labels), "n_operations": len(operations),
        "file_headers": headers,
        "frames_rows_are_cell_axes_in_cubic_coordinates": {plane: FRAMES[plane].astype(int).tolist()
                                                           for plane in sc.PLANES},
        "checks": checks,
        "n_classes": classes,
        "grain_class_counts": {plane: labels.loc[labels["plane"] == plane, "grain_class"].value_counts().to_dict()
                               for plane in sc.PLANES},
        "run": mc.run_metadata(),
    }
    mc.write_table(args.out_dir / "gb_labels.csv", labels)
    mc.write_table(args.out_dir / "gb_operations.csv", operations)
    mc.write_json(args.out_dir / "gb_labels.json", summary)

    print(f"{len(labels)} interfaces parsed: " + ", ".join(f"{plane}: {counts[plane]}" for plane in sc.PLANES)
          + f" (folders: {', '.join(str(folders[plane]) for plane in sc.PLANES)})")
    print("frame of each bulk cell confirmed against the pristine cell: "
          + ", ".join(f"{plane}: {'yes' if ok else 'NO'}" for plane, ok in checks["frames_verified"].items()))
    if not args.skip_bulk_check:
        for plane, entry in checks["operations_reproduce_bulk_cells"].items():
            print(f"  {plane}: every listed operation reproduces Bulk/symNNNNN for "
                  f"{entry['n_all_operations_match']} of {entry['n_families']} families")
    print(f"{'label':12s} {'all':>5s} {'100':>5s} {'110':>5s} {'111':>5s}   (number of classes)")
    for scheme, entry in classes.items():
        print(f"{scheme:12s} {entry['all']:5d} {entry['100']:5d} {entry['110']:5d} {entry['111']:5d}")
    for plane in sc.PLANES:
        print(f"grain classes on {plane}: " + ", ".join(f"{name} {count}" for name, count in
                                                         sorted(Counter(labels.loc[labels['plane'] == plane, 'grain_class']).items())))
    print(f"tables in {args.out_dir}")
    failed = (not checks["parsed_equals_folders"] or not all(checks["frames_verified"].values())
              or any(entry["failures"] for entry in checks.get("operations_reproduce_bulk_cells", {}).values()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
