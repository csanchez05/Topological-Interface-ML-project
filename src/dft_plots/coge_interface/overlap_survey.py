#!/usr/bin/env python3
"""Closest-pair survey of the CoGe grain-boundary (GB) catalogue.

For every GB POSCAR
  structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/Interface/symNNNNN/<k>x/POSCAR
it finds the closest pair of atoms (periodic, minimum image) and counts the pairs closer than each
threshold. The bulk reference (the shortest Co-Ge, Co-Co and Ge-Ge distances of B20 CoGe) comes from
calculations/CoGe_Bulk/CoGe_100/sym00000/non-soc/scf/POSCAR.

All GB cells are orthogonal boxes whose shortest edge is 4.63 A, so the per-axis minimum image is
exact for every distance below half the shortest edge (2.31 A). Thresholds above that are refused.

Output: one TSV with a row per POSCAR, and a summary on stdout: for each plane and threshold, how many
GB families clear it (closest pair >= threshold) in the catalogue and in the Stage 2 list
(results/dscribe/total_sample_list.csv).

Run on a compute node (it reads 13,869 POSCARs; ~5 min with 16 reading threads):
  srun -p general --exclude=qcg-docker -n 1 -c 4 --mem=4G -t 00:45:00 bash -lc \\
      'cd <project root> && src/dft_plots/.venv/bin/python -B src/dft_plots/coge_interface/overlap_survey.py'
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = "structures/CoGeGrainBoundaries/CoGe_{plane}_GBs/Interface"
BULK_POSCAR = "calculations/CoGe_Bulk/CoGe_100/sym00000/non-soc/scf/POSCAR"
STAGE2_LIST = "results/dscribe/total_sample_list.csv"
PLANES = ("100", "110", "111")
THICKNESSES = tuple(f"{k}x" for k in range(4, 13))
THRESHOLDS = (1.0, 1.5, 1.8, 2.0, 2.2)
SPECIES = ("Co", "Ge")


class PoscarError(Exception):
    pass


def read_poscar(text):
    """(lattice 3x3 in A, fractional coordinates Nx3, species index per atom)."""
    lines = text.splitlines()
    if not text.strip():
        raise PoscarError("empty")
    try:
        scale = float(lines[1].split()[0])
        lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
    except (IndexError, ValueError):
        raise PoscarError("malformed header")
    if species != list(SPECIES):
        raise PoscarError("species " + " ".join(species))
    if not lines[7].strip().lower().startswith("d"):
        raise PoscarError("not Direct")
    n = sum(counts)
    try:
        frac = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n)])
    except (IndexError, ValueError):
        raise PoscarError("truncated coordinates")
    return lattice, frac, np.repeat(np.arange(len(counts)), counts)


def pair_name(a, b):
    return "-".join(sorted((SPECIES[a], SPECIES[b])))


def gb_pairs(lattice, frac, kinds, thresholds):
    """Closest pair and pair counts below each threshold, for an orthogonal cell."""
    if np.abs(lattice - np.diag(np.diag(lattice))).max() > 1e-8:
        raise PoscarError("non-orthogonal cell")
    lengths = np.diag(lattice)
    exact_below = lengths.min() / 2
    if max(thresholds) >= exact_below:
        raise ValueError(f"threshold {max(thresholds)} A >= half the shortest edge ({exact_below:.2f} A)")
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.rint(d)
    dist = np.sqrt(((d * lengths) ** 2).sum(axis=-1))
    iu = np.triu_indices(len(frac), 1)
    flat = dist[iu]
    k = int(np.argmin(flat))
    i, j = iu[0][k], iu[1][k]
    return {"dmin": float(flat[k]), "pair": pair_name(kinds[i], kinds[j]),
            "below": [int((flat < t).sum()) for t in thresholds]}


def bulk_reference(lattice, frac, kinds):
    """Shortest distance per pair type in a small periodic cell (all 27 neighbouring images)."""
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    best = {}
    for i, j in itertools.product(range(len(frac)), repeat=2):
        cart = (frac[j] - frac[i] + shifts) @ lattice
        r = np.sqrt((cart ** 2).sum(axis=1))
        r = r[r > 1e-8]
        name = pair_name(kinds[i], kinds[j])
        best[name] = min(best.get(name, np.inf), float(r.min()))
    return best


def stage2_set(root):
    found = set()
    for line in (root / STAGE2_LIST).read_text().splitlines()[1:]:
        parts = line.strip().split("/")
        if len(parts) >= 4 and parts[-1] == "POSCAR":
            found.add((parts[-5].split("_")[1], parts[-3], parts[-2]))  # (plane, sym, thickness)
    return found


def read_or_none(path):
    return path.read_text() if path.is_file() else None


def survey(root, planes, thicknesses, thresholds, workers=16):
    """Reads run in threads: each POSCAR read costs ~0.09 s of file-system latency, the pairs ~0.01 s."""
    lattice, frac, kinds = read_poscar((root / BULK_POSCAR).read_text())
    bulk = bulk_reference(lattice, frac, kinds)
    stage2 = stage2_set(root)
    keys = [(plane, sym_dir.name, thickness, sym_dir / thickness / "POSCAR")
            for plane in planes
            for sym_dir in sorted((root / CATALOGUE.format(plane=plane)).glob("sym[0-9][0-9][0-9][0-9][0-9]"))
            for thickness in thicknesses]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (plane, sym, thickness, _), text in zip(keys, pool.map(read_or_none, [k[3] for k in keys])):
            row = {"plane": plane, "sym": sym, "thickness": thickness,
                   "stage2": (plane, sym, thickness) in stage2}
            if text is None:
                row["status"] = "missing"
            else:
                try:
                    lat, fr, kd = read_poscar(text)
                    row.update(gb_pairs(lat, fr, kd, thresholds), atoms=len(fr), status="ok")
                    row["ratio"] = row["dmin"] / bulk[row["pair"]]
                except PoscarError as exc:
                    row["status"] = str(exc)
            rows.append(row)
    return bulk, rows


def write_tsv(rows, thresholds):
    head = ["plane", "sym", "thickness", "status", "atoms", "dmin_A", "pair", "dmin_over_bulk_pair",
            *[f"pairs_lt_{t:g}A" for t in thresholds], "in_stage2_list"]
    lines = ["\t".join(head)]
    for r in rows:
        ok = r["status"] == "ok"
        lines.append("\t".join([r["plane"], r["sym"], r["thickness"], r["status"],
                                str(r["atoms"]) if ok else "", f"{r['dmin']:.3f}" if ok else "",
                                r["pair"] if ok else "", f"{r['ratio']:.3f}" if ok else "",
                                *([str(n) for n in r["below"]] if ok else [""] * len(thresholds)),
                                "yes" if r["stage2"] else "no"]))
    return "\n".join(lines) + "\n"


def summary(bulk, rows, thresholds, reference_thickness="8x"):
    out = ["bulk B20 shortest distances: " + ", ".join(f"{k} {v:.3f} A" for k, v in sorted(bulk.items()))]
    bad = [r for r in rows if r["status"] not in ("ok", "missing")]
    out.append(f"POSCARs read: {sum(r['status'] == 'ok' for r in rows)}; unreadable: "
               + (", ".join(f"{r['plane']}/{r['sym']}/{r['thickness']} ({r['status']})" for r in bad) or "none"))
    by_family = {}
    for r in rows:
        if r["status"] == "ok":
            by_family.setdefault((r["plane"], r["sym"]), []).append(r["dmin"])
    spread = max((max(v) - min(v) for v in by_family.values()), default=0.0)
    varying = sum(max(v) - min(v) > 0.01 for v in by_family.values())
    out.append(f"closest pair across thicknesses: largest spread within a family {spread:.3f} A; "
               f"families varying by > 0.01 A: {varying} of {len(by_family)}")
    out.append(f"families whose {reference_thickness} closest pair is >= threshold "
               "(catalogue | Stage 2 list):")
    out.append("plane  families | stage2   " + "  ".join(f">={t:g}A" for t in thresholds))
    for plane in sorted({r["plane"] for r in rows}):
        ref = [r for r in rows if r["plane"] == plane and r["thickness"] == reference_thickness
               and r["status"] == "ok"]
        s2 = [r for r in ref if r["stage2"]]
        cells = "  ".join(f"{sum(r['dmin'] >= t for r in ref):>4}|{sum(r['dmin'] >= t for r in s2):<3}"
                          for t in thresholds)
        out.append(f"{plane:<6} {len(ref):>8} | {len(s2):<6}   {cells}")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planes", nargs="+", default=PLANES, choices=PLANES)
    parser.add_argument("--thicknesses", nargs="+", default=THICKNESSES, choices=THICKNESSES)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS, help="A (default %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("results/CoGe_interface/overlap_survey.tsv"),
                        help="project-relative TSV (default %(default)s)")
    parser.add_argument("--workers", type=int, default=16, help="file-reading threads (default %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing, different TSV")
    parser.add_argument("--dry-run", action="store_true", help="count the POSCARs only")
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    if args.dry_run:
        n = sum(len(list((root / CATALOGUE.format(plane=p)).glob(f"sym*/{t}/POSCAR")))
                for p in args.planes for t in args.thicknesses)
        print(f"{n} POSCARs would be read")
        return 0
    thresholds = sorted(args.thresholds)
    bulk, rows = survey(root, args.planes, args.thicknesses, thresholds, args.workers)
    text = write_tsv(rows, thresholds)
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists() and out.read_text() != text and not args.overwrite:
        sys.exit(f"STOP: {out} exists with different content (use --overwrite to replace it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(summary(bulk, rows, thresholds))
    print(f"wrote {out.relative_to(root) if out.is_relative_to(root) else out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
