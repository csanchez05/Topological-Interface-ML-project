#!/usr/bin/env python3
"""Independent Stage 1 audit of the eight catalogue grain-orientation labels.

Run on a compute node with src/dft_plots/.venv/bin/python (spglib required).
This does not import the label generator, its crystal group, or its matcher.
It discovers bulk symmetry from atoms, partitions raw construction matrices,
independently groups stored bulk cells by translation-only atom matching, and
checks the interiors of the actual GB cells. One JSON contains all evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import spglib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAMES = {
    "100": np.eye(3),
    "110": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float),
    "111": np.array([[1, 1, 1], [-1, 1, 0], [-1, -1, 2]], dtype=float),
}
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def read_poscar(path, hashes):
    raw = path.read_bytes()
    hashes.update(str(path.relative_to(PROJECT_ROOT)).encode() + b"\0" + raw)
    lines = raw.decode().splitlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in line.split()[:3]] for line in lines[2:5]])
    corrected = abs(scale - 1) > 1e-8 and np.max(np.linalg.norm(cell, axis=1)) > 3
    if not corrected:
        cell *= scale
    if scale <= 0 or not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-9, rtol=0):
        raise ValueError(f"This audit expects positive-scale orthogonal source cells: {path}")
    species = np.repeat(lines[5].split(), [int(x) for x in lines[6].split()])
    index = 7
    if lines[index].lower().startswith("s"):
        index += 1
    if not lines[index].lower().startswith("d"):
        raise ValueError(f"This audit expects Direct coordinates: {path}")
    frac = np.array([[float(x) for x in line.split()[:3]]
                     for line in lines[index + 1:index + 1 + len(species)]])
    if frac.shape != (len(species), 3):
        raise ValueError(f"Atom-count mismatch: {path}")
    return cell, frac % 1, species, corrected


def read_operations(path):
    text = path.read_text()
    headers = list(re.finditer(r"^Interface (\d+) : Num of Syms = (\d+)\s*$", text, re.M))
    entries = []
    for index, match in enumerate(headers):
        lines = text[match.end():headers[index + 1].start() if index + 1 < len(headers) else None].splitlines()
        matrices = []
        for k, line in enumerate(lines):
            if line.lstrip().startswith("[["):
                numbers = [float(x) for x in NUMBER.findall(" ".join(lines[k:k + 3]))]
                if len(numbers) != 12:
                    raise ValueError(f"Malformed operation in {path}, interface {index}")
                matrices.append(np.array(numbers).reshape(3, 4))
        if int(match[1]) != index or len(matrices) != int(match[2]):
            raise ValueError(f"Operation count/index mismatch in {path}, interface {index}")
        entries.append(matrices)
    declared = int(re.search(r"Number of Unique Simple Interfaces: (\d+)", text)[1])
    if len(entries) != declared:
        raise ValueError(f"Interface count mismatch in {path}")
    return entries


def distances(a, b, lengths):
    delta = a[:, None, :] - b[None, :, :]
    delta -= np.rint(delta)
    return np.linalg.norm(delta * lengths, axis=-1)


def atom_match(a, sa, b, sb, lengths):
    """Species-preserving bijection at the supplied shift; returns max/RMS error."""
    if len(a) != len(b) or Counter(sa) != Counter(sb):
        return float("inf"), float("inf")
    ds = distances(a, b, lengths)
    ds[sa[:, None] != sb[None, :]] = np.inf
    rows, cols = linear_sum_assignment(ds ** 2)
    errors = ds[rows, cols]
    return float(errors.max()), float(np.sqrt(np.mean(errors ** 2)))


def translation_match(a, sa, b, sb, lengths, tolerance):
    """Test all shifts that align atom a[0] with an equal-species atom of b.

    Returned lower bound applies to the best maximum atom error over ALL
    continuous translations, not just those tested. If a mapping has error e,
    snapping its anchor to its assigned target changes the shift by <=e;
    every atom then has a neighbour within 2e. Thus half the smallest maximum
    nearest-neighbour error over anchor shifts is a rigorous lower bound.
    An accepted upper bound always uses a species-preserving bijection.
    """
    if len(a) != len(b) or Counter(sa) != Counter(sb):
        return False, float("inf"), float("inf")
    shifts = b[sb == sa[0]] - a[0]
    moved = a[None, :, :] + shifts[:, None, :]
    delta = moved[:, :, None, :] - b[None, None, :, :]
    delta -= np.rint(delta)
    ds = np.linalg.norm(delta * lengths, axis=-1)
    ds[:, sa[:, None] != sb[None, :]] = np.inf
    scores = ds.min(axis=2).max(axis=1)
    lower = float(scores.min() / 2)
    for index in np.argsort(scores):
        if scores[index] > tolerance:
            break
        rows, cols = linear_sum_assignment(ds[index] ** 2)
        error = float(ds[index, rows, cols].max())
        if error <= tolerance:
            return True, error, lower
    return False, float("inf"), lower


def matrix_distance(a, b, group, both=False):
    if both:
        variants = np.array([s @ b @ t for s in group for t in group])
    else:
        variants = np.array([b @ t for t in group])
    return float(np.min(np.linalg.norm(variants - a, axis=(1, 2))))


def unique_matrices(matrices):
    return list({tuple(np.round(m.ravel(), 12)): m for m in matrices}.values())


def partition(matrices, group, tolerance):
    representatives, ids = [], []
    for matrix in matrices:
        hits = [i for i, ref in enumerate(representatives)
                if matrix_distance(matrix, ref, group) < tolerance]
        if not hits:
            hits = [len(representatives)]
            representatives.append(matrix)
        if len(hits) != 1:
            raise ValueError("Ambiguous matrix equivalence class")
        ids.append(hits[0])
    return representatives, ids


def signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            yield np.eye(3)[list(perm)] * np.array(signs)[:, None]


def named_representative(name):
    """Decode existing names only AFTER discovering the classes independently."""
    sign = -1 if name == "I" or name.startswith("I.") else 1
    base = name.removeprefix("I.")
    if base in ("E", "I"):
        return sign * np.eye(3)
    definitions = {
        "R90[100]": (np.pi / 2, [1, 0, 0]),
        "R60[111]": (np.pi / 3, [1, 1, 1]),
        "R70.53[1-10]": (np.arccos(1 / 3), [1, -1, 0]),
    }
    angle, axis = definitions[base]
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return sign * (np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew)


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None  # A failed atom-count/matching check has no finite error.
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gb-root", type=Path, default=PROJECT_ROOT / "structures/CoGeGrainBoundaries")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "calculations/ML_dscribe/CoGe/labels/gb_labels.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/dscribe/grain_class_verification.json")
    parser.add_argument("--atom-tolerance", type=float, default=0.002, help="Angstrom; accommodates rounded source coordinates")
    parser.add_argument("--matrix-tolerance", type=float, default=1e-8)
    parser.add_argument("--thickness", type=int, default=8)
    parser.add_argument("--max-families-per-plane", type=int, default=0, help="0 means every family; otherwise a smoke run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.atom_tolerance <= 0 or args.matrix_tolerance <= 0 or args.thickness < 4:
        parser.error("Positive tolerances and thickness >=4 are required")
    entries = {plane: read_operations(args.gb_root / f"CoGe_{plane}_GBs/Interface/Information.txt") for plane in FRAMES}
    if args.dry_run:
        print(json.dumps({"families": {p: len(e) for p, e in entries.items()}, "thickness": args.thickness,
                          "out": str(args.out), "host": socket.gethostname()}, indent=2))
        return 0
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}; choose a new --out")
    started = time.monotonic()
    hashes = hashlib.sha256()
    pristine_cell, pristine, species, _ = read_poscar(args.gb_root / "POSCAR", hashes)
    numbers = np.array([{"Co": 27, "Ge": 32}[s] for s in species])
    symmetry_runs = []
    for tol in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        data = spglib.get_symmetry_dataset((pristine_cell, pristine, numbers), symprec=tol)
        symmetry_runs.append({"tolerance_A": tol, "spacegroup": data.international,
                              "number": data.number, "pointgroup": data.pointgroup,
                              "n_rotations": len(unique_matrices(data.rotations))})
    group = unique_matrices(data.rotations)
    direct_group, excluded_bounds = [], []
    direct_errors = []
    for q in signed_permutations():
        ok, error, bound = translation_match(pristine @ q.T, species, pristine, species,
                                             np.diag(pristine_cell), 1e-8)
        if ok:
            direct_group.append(q)
            direct_errors.append(error)
        else:
            excluded_bounds.append(bound)
    group_agreement = len(group) == len(direct_group) and all(
        any(np.linalg.norm(q - s) < 1e-10 for s in direct_group) for q in group)
    with args.labels.open() as stream:
        old_labels = {r["family"]: r["grain_class"] for r in csv.DictReader(stream)}

    all_unique, matrices_by_plane = [], {}
    for plane, listing in entries.items():
        frame = FRAMES[plane]
        qs = [frame.T @ op[:, :3] @ np.linalg.inv(frame.T) for ops in listing for op in ops]
        if max(np.linalg.norm(q.T @ q - np.eye(3)) for q in qs) > args.matrix_tolerance:
            raise ValueError(f"Non-orthogonal construction matrix for plane {plane}")
        matrices_by_plane[plane] = unique_matrices(qs)
        all_unique += matrices_by_plane[plane]
    all_unique = unique_matrices(all_unique)
    representatives, _ = partition(all_unique, group, args.matrix_tolerance)
    matrix_names = {i: set() for i in range(len(representatives))}
    family_matrix_ids = {}
    for plane, listing in entries.items():
        frame = FRAMES[plane]
        for index, ops in enumerate(listing):
            family = f"{plane}/sym{index:05d}"
            ids = []
            for op in ops:
                q = frame.T @ op[:, :3] @ np.linalg.inv(frame.T)
                hits = [i for i, ref in enumerate(representatives)
                        if matrix_distance(q, ref, group) < args.matrix_tolerance]
                if len(hits) != 1:
                    raise ValueError(f"Matrix partition failure: {family}")
                ids.append(hits[0])
            if len(set(ids)) != 1:
                raise ValueError(f"Operations of {family} do not give the same orientation")
            family_matrix_ids[family] = ids[0]
            matrix_names[ids[0]].add(old_labels[family])
    names = {i: "|".join(sorted(v)) for i, v in matrix_names.items()}
    label_partition_matches = all(len(v) == 1 for v in matrix_names.values()) and len(set(names.values())) == len(names)
    representative_rows = []
    name_errors = []
    if label_partition_matches:
        for i, q in enumerate(representatives):
            canonical = named_representative(names[i])
            error = matrix_distance(canonical, q, group)
            name_errors.append(error)
            representative_rows.append({"name": names[i], "matrix": canonical.tolist(),
                                        "determinant": int(round(np.linalg.det(canonical))),
                                        "name_to_observed_class_error": error})
    pairwise = []
    for i, j in itertools.combinations(range(len(representatives)), 2):
        a, b = representatives[i], representatives[j]
        d1 = matrix_distance(a, b, group)
        d2 = matrix_distance(a, b, group, both=True)
        same_hand = np.linalg.det(a) * np.linalg.det(b) > 0
        pairwise.append({"class_a": names[i], "class_b": names[j], "same_handedness": bool(same_hand),
                         "right_coset_Frobenius_distance": d1, "both_grain_symmetries_Frobenius_distance": d2,
                         "both_grain_symmetries_angle_deg": float(np.degrees(2 * np.arcsin(np.clip(d2 / (2 * np.sqrt(2)), 0, 1)))) if same_hand else None,
                         "distance_allowing_grain_exchange": min(d2, matrix_distance(a, b.T, group, both=True))})

    result = {
        "definition": "Second-grain orientation and relative handedness with the first crystal frame fixed, modulo translations and the crystal point group. Not a count of full grain-boundary structures or all possible orientations.",
        "run": {"command": sys.argv, "host": socket.gethostname(), "utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "numpy": np.__version__, "spglib": spglib.__version__, "complete_catalogue": not bool(args.max_families_per_plane),
                "atom_tolerance_A": args.atom_tolerance, "matrix_tolerance": args.matrix_tolerance},
        "pristine_symmetry": {"spglib_tolerance_sweep": symmetry_runs, "exhaustive_48_lattice_isometries_accepted": len(direct_group),
                              "agrees_with_spglib": group_agreement, "accepted_max_atom_error_A": max(direct_errors),
                              "excluded_isometries_minimum_lower_bound_A": min(excluded_bounds),
                              "proper_operations": sum(np.linalg.det(q) > 0 for q in group),
                              "rotations": [q.tolist() for q in group]},
        "matrix_partition": {"raw_unique_rotations_all_planes": len(all_unique), "n_classes": len(representatives),
                             "matches_stored_labels": label_partition_matches,
                             "named_representatives": representative_rows,
                             "tolerance_sweep": {str(t): len(partition(all_unique, group, t)[0]) for t in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]},
                             "pairwise_separation": pairwise},
        "planes": {}, "failures": [],
    }
    # A commensurate orientation omitted by the catalogue: cos(theta)=4/5,
    # sin(theta)=3/5 about [001]. diag(5,5,1) is a common lattice supercell.
    extra = np.array([[.8, -.6, 0], [.6, .8, 0], [0, 0, 1.]])
    result["counterexample_to_universal_eight"] = {
        "rotation": extra.tolist(), "angle_deg": float(np.degrees(np.arctan2(.6, .8))), "axis": "[001]",
        "common_supercell_in_first_lattice": np.diag([5, 5, 1]).tolist(),
        "common_supercell_in_rotated_lattice": np.round(extra.T @ np.diag([5, 5, 1]), 12).tolist(),
        "distance_to_each_existing_class_allowing_both_grain_symmetries": {
            names[i]: matrix_distance(extra, q, group, both=True) for i, q in enumerate(representatives)},
    }

    for plane, listing in entries.items():
        folder = args.gb_root / f"CoGe_{plane}_GBs"
        ref_cell, reference, ref_species, _ = read_poscar(folder / f"Bulk/CoGe_{plane}_POSCAR.vasp", hashes)
        lengths = np.diag(ref_cell)
        frame = FRAMES[plane]
        shifts = np.array(list(itertools.product(range(-3, 4), repeat=3)))
        expanded = (pristine[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ np.linalg.inv(frame)
        mask = np.all((expanded >= -1e-10) & (expanded < 1 - 1e-10), axis=1)
        expanded_species = np.tile(species, len(shifts))[mask]
        frame_error, _ = atom_match(expanded[mask], expanded_species, reference, ref_species, lengths)
        expected_lengths = np.linalg.norm(frame, axis=1) * pristine_cell[0, 0]
        if frame_error > args.atom_tolerance or np.max(np.abs(expected_lengths - lengths)) > 1e-5:
            raise ValueError(f"Reference frame verification failed: {plane}")
        raw = matrices_by_plane[plane]
        intersection = sum(any(np.linalg.norm(q - s) < args.matrix_tolerance for s in group) for q in raw)
        closure_error = max(min(np.linalg.norm(a @ b - c) for c in raw) for a in raw for b in raw)
        atomic_reps, atomic_names, family_rows = [], [], []
        maximum_operation_error = maximum_class_error = maximum_interface_error = 0.0
        op_count = corrected_count = 0
        maximum_frame_error = frame_error
        for index, ops in enumerate(listing[:args.max_families_per_plane or None]):
            family = f"{plane}/sym{index:05d}"
            cell, frac, sa, corrected = read_poscar(folder / f"Bulk/sym{index:05d}/POSCAR", hashes)
            corrected_count += int(corrected)
            if np.max(np.abs(cell - ref_cell)) > 1e-5:
                raise ValueError(f"Unexpected bulk lattice: {family}")
            errors = [atom_match(reference @ op[:, :3].T + op[:, 3], ref_species, frac, sa, lengths)[0] for op in ops]
            operation_error = max(errors)
            maximum_operation_error = max(maximum_operation_error, operation_error)
            op_count += len(ops)
            if operation_error > args.atom_tolerance:
                result["failures"].append({"family": family, "check": "operation_reproduces_bulk", "error_A": operation_error})
            # Discover the atomic class WITHOUT consulting names or operations.
            chosen, atomic_error = None, 0.0
            for k, (other, other_species) in enumerate(atomic_reps):
                ok, error, _ = translation_match(frac, sa, other, other_species, lengths, args.atom_tolerance)
                if ok:
                    chosen, atomic_error = k, error
                    break
            if chosen is None:
                chosen = len(atomic_reps)
                atomic_reps.append((frac.copy(), sa.copy()))
                atomic_names.append(set())
            atomic_names[chosen].add(old_labels[family])
            maximum_class_error = max(maximum_class_error, atomic_error)
            gb_cell, gb, gb_species, _ = read_poscar(folder / f"Interface/sym{index:05d}/{args.thickness}x/POSCAR", hashes)
            expected_cell = ref_cell.copy()
            expected_cell[0] *= 2 * args.thickness
            if np.max(np.abs(gb_cell - expected_cell)) > 1e-4 or len(gb) != len(frac) * 2 * args.thickness:
                raise ValueError(f"Unexpected GB cell or atom count: {family}")
            u = gb[:, 0] * 2 * args.thickness
            interface_errors = []
            for half, (bulk_positions, bulk_species) in enumerate([(reference, ref_species), (frac, sa)]):
                start = args.thickness / 2 - 0.877 + half * args.thickness
                interior = (u >= start) & (u < start + 1)
                interior_positions = gb[interior].copy()
                interior_positions[:, 0] = u[interior] % 1
                error, _ = atom_match(interior_positions, gb_species[interior], bulk_positions, bulk_species, lengths)
                interface_errors.append(error)
                if error > args.atom_tolerance:
                    result["failures"].append({"family": family, "check": f"grain_{half + 1}_interior", "error_A": error})
            maximum_interface_error = max(maximum_interface_error, *interface_errors)
            family_rows.append({"family": family, "label": old_labels[family], "atomic_class": chosen,
                                "operation_max_error_A": operation_error, "class_match_max_error_A": atomic_error,
                                "first_grain_max_error_A": interface_errors[0], "second_grain_max_error_A": interface_errors[1]})
            if (index + 1) % 100 == 0:
                print(f"{plane}: checked {index + 1} bulk/GB pairs; {len(atomic_reps)} atomic classes", flush=True)
        separations = []
        for i, j in itertools.combinations(range(len(atomic_reps)), 2):
            (a, sa), (b, sb) = atomic_reps[i], atomic_reps[j]
            _, _, bound = translation_match(a, sa, b, sb, lengths, args.atom_tolerance)
            separations.append({"class_a": "|".join(sorted(atomic_names[i])), "class_b": "|".join(sorted(atomic_names[j])),
                                "representatives_translation_only_lower_bound_A": bound,
                                "any_members_translation_only_lower_bound_A": max(0.0, bound - 2 * maximum_class_error)})
        names_agree = all(len(v) == 1 for v in atomic_names) and len(set(next(iter(v)) for v in atomic_names)) == len(atomic_names)
        if not names_agree:
            result["failures"].append({"plane": plane, "check": "independent_atomic_partition_matches_labels"})
        result["planes"][plane] = {
            "families_checked": len(family_rows), "operations_checked": op_count, "bulk_scale_headers_corrected_in_memory": corrected_count,
            "raw_unique_rotations": len(raw), "raw_rotation_group_closure_error": closure_error,
            "intersection_with_crystal_point_group": int(intersection), "predicted_classes_H_over_intersection": len(raw) // intersection,
            "frame_max_atom_error_A": maximum_frame_error, "operation_max_atom_error_A": maximum_operation_error,
            "independent_atomic_classes": len(atomic_reps), "atomic_partition_matches_labels": names_agree,
            "max_atomic_class_match_error_A": maximum_class_error, "max_grain_interior_error_A": maximum_interface_error,
            "class_counts": dict(Counter(row["label"] for row in family_rows)),
            "atomic_class_pairwise_lower_bounds": separations, "families": family_rows,
        }
        print(f"{plane}: finished {len(family_rows)} families, {len(atomic_reps)} atomic classes; max class error {maximum_class_error:.6g} A", flush=True)

    # Same orientation label does not determine the actual boundary geometry.
    # A distance between unlike species is invariant under ANY rigid motion.
    example = []
    for family in ["100/sym00001", "100/sym00002"]:
        plane, sym = family.split("/")
        path = args.gb_root / f"CoGe_{plane}_GBs/Interface/{sym}/{args.thickness}x/POSCAR"
        cell, frac, sa, _ = read_poscar(path, hashes)
        separation = distances(frac[sa == "Co"], frac[sa == "Ge"], np.diag(cell))
        example.append({"family": family, "label": old_labels[family],
                        "source": str(path.relative_to(PROJECT_ROOT)), "minimum_Co_Ge_distance_A": float(separation.min())})
    result["same_class_different_boundary_example"] = example
    result["checks"] = {
        "atomic_and_spglib_point_groups_agree": group_agreement,
        "matrix_partition_matches_stored_labels": label_partition_matches,
        "label_names_describe_the_discovered_classes": bool(name_errors) and max(name_errors) < args.matrix_tolerance,
        "class_pairs_distinct_even_with_both_grain_symmetries_and_exchange": all(p["distance_allowing_grain_exchange"] > args.matrix_tolerance for p in pairwise),
        "constructed_ninth_orientation_is_distinct": all(v > args.matrix_tolerance for v in result["counterexample_to_universal_eight"]["distance_to_each_existing_class_allowing_both_grain_symmetries"].values()),
        "same_label_can_have_different_boundary_geometry": example[0]["label"] == example[1]["label"] and abs(example[0]["minimum_Co_Ge_distance_A"] - example[1]["minimum_Co_Ge_distance_A"]) > 2 * args.atom_tolerance,
        "all_atomic_checks_pass": not result["failures"],
    }
    result["status"] = "passed" if all(result["checks"].values()) else "failed"
    result["run"]["elapsed_seconds"] = time.monotonic() - started
    result["run"]["structure_stream_sha256"] = hashes.hexdigest()
    result["run"]["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["run"]["metadata_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.labels] + [args.gb_root / f"CoGe_{plane}_GBs/Interface/Information.txt" for plane in FRAMES]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {args.out}; elapsed {result['run']['elapsed_seconds']:.1f}s", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
