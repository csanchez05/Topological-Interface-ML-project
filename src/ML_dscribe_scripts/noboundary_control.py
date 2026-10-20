#!/usr/bin/env python3
"""No-boundary control for covariant SOAP (SOAP guide Stage 1, a follow-up of the fifth study).

Question: covariant interface vectors (C) sort the GBs by grain class (study 06: raw kNN 0.77,
against 0.26 for standard SOAP). Do they need the boundary for that, or would the same 8 A regions
cut from the two perfect grains, with no boundary at all, already tell the grain classes apart?

Construction (checked on every family): an 8x GB cell is grain A (the plane's reference crystal) in
x < Lx/2 and grain B (Bulk/symNNNNN) in x >= Lx/2, each 8 whole periods thick, with no atom removed.
Each half is doubled into a perfect crystal (the half plus a copy shifted by Lx/2). Every atom's
covariant features are computed in its own grain's crystal and averaged over exactly the GB's
interface regions (atoms within 8 A of x = 0 or Lx/2, each atom with its nearer plane). The result,
C_nb, is the "no-boundary" version of C: what those regions would give without a boundary.

Identity tested: atoms farther than 8 A from both planes see only their own grain (r_cut 6 A < 8 A),
and each half is whole periods (its lambda >= 1 sum is zero), so the whole-cell vector D satisfies,
for each species s,
    N_s D_s = n_edge,s (C_edge,s - C_nb,edge,s) + n_mid,s (C_mid,s - C_nb,mid,s).
Study 06 fitted D = s (C_edge + C_mid) and found residuals 0.39 / 0.25 / 3.6e-4 (100 / 110 / 111);
its hypothesis was that the grain interior adds covariant signal on 100 and 110. If the identity
holds, that signal is exactly -C_nb.

Steps (--step):
  describe  per family at 8x (the distinct GBs plus the two plain-bulk families): C_nb for the
            lambda >= 1 blocks of the reference covariant run, with the checks below. Writes
            noboundary_vectors.npz, index.csv, describe.json.
  analyse   loads C and D of the reference covariant run for the rows of study 06 (distinct GBs
            without the plain-bulk ones, edge and mid) and C_nb. kNN (k = 10, the other interface of
            the same cell left out, as in study 06) for grain class, plane and edge/mid on C, D, C_nb
            and C - C_nb; size of C_nb against C; the identity residual next to the study-06 fit.
            Writes results.csv and summary.json.
Checks: (describe) the doubled cells have no pair closer than 2.3 A and a whole-cell lambda >= 1
average below 1e-4 of the mean atom norm (GB coordinate rounding leaves 3e-5 to 6e-5); for the two
plain-bulk families |C_nb - C| is recorded. (analyse) the region atom counts equal those of the
reference run; C and D reproduce study 06's kNN values (to 1e-3, two of 2,222 rows); for the
plain-bulk families |C_nb - stored C| is below 1% of a typical GB's |C|.

Output: calculations/ML_dscribe/CoGe/comparison/representation_study/noboundary_control/
  python -B src/ML_dscribe_scripts/noboundary_control.py --step describe --workers 12
  python -B src/ML_dscribe_scripts/noboundary_control.py --step analyse
Slurm: sbatch src/ML_dscribe_scripts/noboundary_control.slurm (both steps).
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

import ml_common as mc
import representation_study as rs
from ml_common import sc

COVARIANT_CODE = mc.PROJECT_ROOT / "src" / "dscribe" / "CoGe" / "covariant"
if str(COVARIANT_CODE) not in sys.path:
    sys.path.insert(0, str(COVARIANT_CODE))  # covariant_descriptor is imported lazily (it pins threads to 1)
OUT_DIR = rs.OUT_DIR / "noboundary_control"
PHASE1_DIR = rs.OUT_DIR / "phase1"
RADIUS = 8.0
MIN_PAIR = 2.3        # A; bulk nearest neighbours are 2.40 A
# Largest accepted whole-cell lambda >= 1 average / mean atom norm of a doubled cell. The GB POSCARs are
# rounded more coarsely than the bulk ones (shortest pair 2.397493 A in the 100 GB cells, 2.397514 A in
# pristine), which leaves 3e-5 to 6e-5 (smoke test 2026-10-07; the bulk cells give <= 9.4e-7).
CELL_AVERAGE = 1e-4
# Plain-bulk families: largest accepted |C_nb - stored C| / median |C| of the analysed GBs. Both are
# noise-sized there (<= 8e-4 against a median near 1 in the smoke test), so this is a sanity check only;
# the identity residual over all GBs is the real test of the construction.
PLAIN_BULK = 1e-2
# Study 06's kNN values are reproduced when they differ by at most this much (1e-3 is two of its 2,222
# rows): neighbour near-ties can flip with last-bit BLAS differences between CPU types.
KNN_MATCH = 1e-3
_STATE = {}


# --------------------------------------------------------------------------- construction

def halves(atoms):
    """(grain A mask, grain B mask): x < Lx/2 is grain A, x >= Lx/2 grain B (fractional x)."""
    in_b = atoms.get_scaled_positions(wrap=True)[:, 0] >= 0.5
    return ~in_b, in_b


def doubled(atoms, mask):
    """The atoms of one half plus a copy shifted by Lx/2, i.e. a perfect crystal of that grain when the
    half is whole periods thick. The first mask.sum() atoms are the originals, in their order."""
    part = atoms[mask]
    copy = part.copy()
    copy.positions = copy.positions + np.array([atoms.cell.array[0, 0] / 2.0, 0.0, 0.0])
    copy.wrap()
    return part + copy


def own_grain_features(n_atoms, masks, per_cell):
    """Per-atom arrays of the GB cell assembled from the doubled cells.

    masks: (mask A, mask B); per_cell: ({block: (2 nA, C, P)}, {block: (2 nB, C, P)}) of the doubled
    cells, originals first. Returns {block: (n_atoms, C, P)}.
    """
    out = {}
    for mask, blocks in zip(masks, per_cell):
        for block, values in blocks.items():
            if block not in out:
                out[block] = np.zeros((n_atoms,) + values.shape[1:])
            out[block][mask] = values[:mask.sum()]
    return out


def region_means(blocks, is_co, x, lx, radius, split_mean):
    """{block: (2 planes, C, 2P)} means over each interface region and the (2 planes, 2 species) counts."""
    distance, plane = sc.plane_distances(x, lx)
    counts = np.zeros((2, 2), dtype=np.int32)
    out = {}
    for block, values in blocks.items():
        arr = np.empty((2, values.shape[1], 2 * values.shape[2]))
        for p in (0, 1):
            arr[p], counts[p] = split_mean(values, is_co, (distance <= radius) & (plane == p))
        out[block] = arr
    return out, counts


def min_pair(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def describe_family(path, calculator, keys, radius, check_gb):
    import covariant_descriptor as cd

    atoms, _ = sc.read_structure(path)
    cell = atoms.cell.array
    lengths = np.diag(cell).copy()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6 or lengths[0] < lengths[1:].max():
        raise ValueError("cell is not orthogonal with x the longest axis")
    lx = lengths[0]
    masks = halves(atoms)
    if masks[0].sum() != masks[1].sum():
        raise ValueError(f"halves differ in size ({masks[0].sum()} and {masks[1].sum()} atoms)")
    checks, per_cell = {}, []
    for name, mask in zip("AB", masks):
        crystal = doubled(atoms, mask)
        blocks, _ = cd.per_atom_blocks(crystal, calculator, keys)
        is_co_crystal = np.array(crystal.get_chemical_symbols()) == "Co"
        checks[f"min_pair_{name}"] = min_pair(crystal)
        checks[f"cell_average_{name}"] = max(
            float(np.linalg.norm(cd.split_mean(values, is_co_crystal)[0])
                  / np.linalg.norm(values.reshape(len(values), -1), axis=1).mean())
            for values in blocks.values())
        per_cell.append(blocks)
    own = own_grain_features(len(atoms), masks, per_cell)
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    x = atoms.get_scaled_positions(wrap=True)[:, 0] * lx
    vectors, counts = region_means(own, is_co, x, lx, radius, cd.split_mean)
    if check_gb:
        gb = cd.describe_structure(path, calculator, keys, [radius])
        # absolute |C_nb - C|, recorded only (see PLAIN_BULK)
        checks["gb_vs_noboundary"] = max(float(np.linalg.norm(vectors[block] - gb["interface"][block][0]))
                                         for block in vectors)
    meta = {"n_atoms": len(atoms), "n_Co": int(is_co.sum()), "n_Ge": int((~is_co).sum()), "Lx": f"{lx:.8f}"}
    return {"vectors": vectors, "n": counts, "checks": checks, "meta": meta}


def failed_checks(checks):
    """Names of the describe checks that fail."""
    bad = [k for k in ("min_pair_A", "min_pair_B") if not checks[k] >= MIN_PAIR]
    bad += [k for k in ("cell_average_A", "cell_average_B") if not checks[k] <= CELL_AVERAGE]
    return bad


def _init_worker(settings, blocks, radius):
    import covariant_descriptor as cd

    calculator, keys = cd.make_calculator(settings, blocks)
    _STATE.update(calculator=calculator, keys=keys, radius=radius)


def _work(task):
    i, path, check_gb = task
    try:
        return i, describe_family(path, _STATE["calculator"], _STATE["keys"], _STATE["radius"], check_gb), None
    except Exception as exc:
        return i, None, f"{type(exc).__name__}: {exc}"


def families_to_describe(per_plane):
    """The distinct GBs of the census (one family each) plus both plain-bulk families, family order."""
    gbs = rs.gb_table(keep_no_boundary=False, per_plane=per_plane)
    return sorted(set(gbs["family"]) | set(mc.NO_BOUNDARY))


def reference_settings(run, root):
    settings = json.loads((sc.run_dir("both_interfaces", run, root=root) / "settings.json").read_text())
    return {k: settings["settings"][k] for k in ("r_cut", "n_max", "l_max", "sigma", "smoothing")}


def describe(args):
    families = families_to_describe(args.limit)
    paths = [sc.GB_ROOT / f"CoGe_{f.split('/')[0]}_GBs" / "Interface" / f.split("/")[1] / "8x" / "POSCAR" for f in families]
    settings = reference_settings(args.covariant, args.covariant_root)
    if args.dry_run:
        print(f"describe: {len(families)} families at 8x, settings {settings}, blocks {list(rs.VECTORIAL)}, "
              f"{args.workers} worker(s), output {args.out_dir}")
        return 0
    if (args.out_dir / "noboundary_vectors.npz").exists() and not args.overwrite:
        print(f"{args.out_dir / 'noboundary_vectors.npz'} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    start = time.time()
    tasks = sorted(((i, p, families[i] in mc.NO_BOUNDARY) for i, p in enumerate(paths)),
                   key=lambda t: -t[1].stat().st_size)
    results, errors = [None] * len(families), []
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, initializer=_init_worker, initargs=(settings, list(rs.VECTORIAL), RADIUS)) as pool:
        for done, (i, result, error) in enumerate(pool.imap_unordered(_work, tasks), 1):
            if error:
                errors.append(f"{families[i]}: {error}")
            results[i] = result
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} families, {time.time() - start:.0f} s", flush=True)
    keep = [i for i, r in enumerate(results) if r is not None]
    rows, failures = [], {}
    for i in keep:
        r = results[i]
        bad = failed_checks(r["checks"])
        if bad:
            failures[families[i]] = bad
        plane, sym = families[i].split("/")
        rows.append({"index": len(rows), "family": families[i], "plane": plane, "sym": sym, **r["meta"],
                     **{k: f"{v:.6e}" for k, v in r["checks"].items()}, "checks_failed": " ".join(bad),
                     "source": sc.relative_path(paths[i])})
    fields = ["index", "family", "plane", "sym", "n_atoms", "n_Co", "n_Ge", "Lx", "min_pair_A", "min_pair_B",
              "cell_average_A", "cell_average_B", "gb_vs_noboundary", "checks_failed", "source"]
    for row in rows:
        row.setdefault("gb_vs_noboundary", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc.write_npz(args.out_dir / "noboundary_vectors.npz",
                 **{f"X_{b}": np.stack([results[i]["vectors"][b] for i in keep]).astype(np.float32) for b in rs.VECTORIAL},
                 n=np.stack([results[i]["n"] for i in keep]))
    sc.write_csv(args.out_dir / "index.csv", rows, fields)
    checks = {k: [float(results[i]["checks"][k]) for i in keep] for k in ("min_pair_A", "min_pair_B", "cell_average_A", "cell_average_B")}
    record = {"step": "describe", "status": "pass" if not (errors or failures) else "fail",
              "n_families": len(families), "n_written": len(keep), "errors": errors, "failed_checks": failures,
              "covariant_run": args.covariant, "settings": settings, "blocks": list(rs.VECTORIAL), "radius_A": RADIUS,
              "limits": {"min_pair_A": MIN_PAIR, "cell_average": CELL_AVERAGE},
              "smallest_min_pair_A": min(checks["min_pair_A"] + checks["min_pair_B"]),
              "largest_cell_average": max(checks["cell_average_A"] + checks["cell_average_B"]),
              "plain_bulk_gb_vs_noboundary": {families[i]: results[i]["checks"].get("gb_vs_noboundary") for i in keep
                                              if families[i] in mc.NO_BOUNDARY},
              "arrays": {"X_<block>": "(F, 2 planes, components, 2P) C_nb, float32, rows as index.csv; "
                                      "planes 0 = edge (x = 0), 1 = mid; last axis [Co mean | Ge mean]",
                         "n": "(F, 2 planes, 2 species) atoms behind each vector"},
              "workers": args.workers, "seconds": round(time.time() - start, 1), **sc.run_metadata()}
    sc.write_json(args.out_dir / "describe.json", record)
    print(f"describe: {record['status']}; {len(keep)} of {len(families)} families written to {args.out_dir} "
          f"in {record['seconds']} s; smallest pair {record['smallest_min_pair_A']:.4f} A, "
          f"largest cell average {record['largest_cell_average']:.2e}")
    for line in errors:
        print("  error " + line)
    for family, bad in failures.items():
        print(f"  check failed {family}: {bad}")
    return 0 if record["status"] == "pass" else 1


# --------------------------------------------------------------------------- analysis

def species_weights(counts, is_co):
    """(S, F) per-column weights from (S, 2 species) counts: the Co count on Co columns, else Ge."""
    return np.where(is_co[None, :], counts[:, :1], counts[:, 1:2]).astype(np.float64)


def identity_residuals(cell, interface, interface_nb, n_interface, n_species, is_co):
    """Relative residuals per GB of N_s D_s = sum_p n_p,s (C_p,s - C_nb_p,s), with and without C_nb,
    and of study 06's fit D = s (C_edge + C_mid) with the best scale s."""
    lhs = species_weights(n_species, is_co) * cell
    parts = [species_weights(n_interface[:, p], is_co) for p in (0, 1)]
    without = sum(w * interface[:, p] for p, w in enumerate(parts))
    with_nb = sum(w * (interface[:, p] - interface_nb[:, p]) for p, w in enumerate(parts))
    size = np.linalg.norm(lhs, axis=1)
    summed = interface[:, 0] + interface[:, 1]
    scale = np.einsum("ij,ij->i", cell, summed) / np.einsum("ij,ij->i", summed, summed)
    fit = np.linalg.norm(cell - scale[:, None] * summed, axis=1) / np.linalg.norm(cell, axis=1)
    return {"identity_with_C_nb": np.linalg.norm(lhs - with_nb, axis=1) / size,
            "identity_without_C_nb": np.linalg.norm(lhs - without, axis=1) / size,
            "fit_06": fit}


def size_measures(C, C_nb):
    """Per row: |C_nb| / |C|, cosine of C and C_nb, |C - C_nb| / |C|."""
    norm_c, norm_nb = np.linalg.norm(C, axis=1), np.linalg.norm(C_nb, axis=1)
    cosine = np.einsum("ij,ij->i", C, C_nb) / np.maximum(norm_c * norm_nb, 1e-300)
    return {"size_ratio": norm_nb / norm_c, "cosine": cosine,
            "relative_difference": np.linalg.norm(C - C_nb, axis=1) / norm_c}


def phase1_values(folder=PHASE1_DIR):
    """Study 06's raw kNN values {(representation, variable): value} and its number of GBs."""
    import pandas as pd

    n_gbs = json.loads((folder / "summary.json").read_text())["n_gbs"]
    table = pd.read_csv(folder / "results.csv")
    raw = table[(table["kind"] == "raw") & (table["subset"] == "pooled") & table["variable"].str.startswith("knn_")]
    return {(r, v): float(x) for r, v, x in zip(raw["representation"], raw["variable"], raw["value"])}, n_gbs


def analyse(args):
    import pandas as pd

    record = json.loads((args.out_dir / "describe.json").read_text())
    if record["status"] != "pass" and not args.allow_failed_checks:
        print("describe.json reports failed checks; fix them or pass --allow-failed-checks", file=sys.stderr)
        return 2
    if (args.out_dir / "results.csv").exists() and not args.overwrite:
        print(f"{args.out_dir / 'results.csv'} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    start = time.time()
    index = pd.read_csv(args.out_dir / "index.csv", dtype={"plane": str})
    position = {f: i for i, f in enumerate(index["family"])}
    gbs = rs.gb_table(available=set(index["family"]))
    rows = rs.interface_rows(gbs)
    keys = rs.structure_keys(rows)
    cov = rs.covariant_vectors(args.covariant, keys, rs.VECTORIAL, root=args.covariant_root)
    pos = np.array([position[f] for f, _ in keys])
    with np.load(args.out_dir / "noboundary_vectors.npz") as data:
        parts = [data[f"X_{b}"][pos].astype(np.float64).reshape(len(pos), 2, -1) for b in rs.VECTORIAL]
        n_nb = data["n"][pos]
        plain = np.array([position[f] for f in mc.NO_BOUNDARY])
        plain_nb = np.concatenate([data[f"X_{b}"][plain].astype(np.float64).reshape(len(plain), 2, -1)
                                   for b in rs.VECTORIAL], axis=-1)
    widths = [p.shape[-1] for p in parts]
    if widths != [b - a for a, b in cov["columns"].values()]:
        raise ValueError("C_nb and C have different block widths")
    if not np.array_equal(n_nb, cov["n_interface"]):
        raise ValueError("region atom counts differ from the reference run")
    interface_nb = np.concatenate(parts, axis=-1)
    C, D, C_nb = cov["C"], cov["D"], interface_nb.reshape(-1, interface_nb.shape[-1])
    plain_c = rs.covariant_vectors(args.covariant, [(f, 8) for f in mc.NO_BOUNDARY], rs.VECTORIAL,
                                   root=args.covariant_root)["interface"]
    plain_check = float(np.max(np.linalg.norm(plain_nb - plain_c, axis=-1)) / np.median(np.linalg.norm(cov["C"], axis=1)))

    codes = rs.label_codes(rows)
    partner = rs.partners(len(rows))
    majority = rs.majority_baselines(codes, True)
    reference, phase1_gbs = phase1_values()
    if len(gbs) != phase1_gbs:  # a --limit test: other rows, so no comparison
        reference = {}
    results, anchors = [], {}
    for name, X in (("C", C), ("D", D), ("C_nb", C_nb), ("C_minus_C_nb", C - C_nb)):
        distances = rs.square_distances(rs.exact_coordinates(X))
        for variable, value in rs.organisation(distances, partner, codes, args.k, True).items():
            results.append({"kind": "knn", "variant": name, "plane": "pooled", "variable": variable,
                            "value": value, "majority": majority.get(variable, "")})
            if (name, variable) in reference:
                anchors[f"{name} {variable}"] = abs(value - reference[(name, variable)])
        print(f"  kNN {name} done, {time.time() - start:.0f} s", flush=True)

    planes = rows["plane"].to_numpy(str)
    measures = size_measures(C, C_nb)
    residuals = identity_residuals(cov["cell"], cov["interface"], interface_nb, cov["n_interface"],
                                   cov["n_species"], cov["is_co"])
    gb_planes = planes[::2]
    for kind, table, which in (("size", measures, planes), ("identity", residuals, gb_planes)):
        for variable, values in table.items():
            for plane in sc.PLANES:
                chosen = values[which == plane]
                for stat, value in (("median", np.median(chosen)), ("max", np.max(chosen)), ("min", np.min(chosen))):
                    results.append({"kind": kind, "variant": stat, "plane": plane, "variable": variable,
                                    "value": float(value), "majority": ""})
    worst_anchor = max(anchors.values()) if anchors else None
    reproduced = None if worst_anchor is None else worst_anchor <= KNN_MATCH
    summary = {"step": "analyse", "n_gbs": len(gbs), "n_rows": len(rows), "k": args.k,
               "covariant_run": args.covariant, "describe_status": record["status"],
               "checks": {"region_counts_equal": True, "plain_bulk_noboundary_vs_stored_C": plain_check,
                          "plain_bulk_limit": PLAIN_BULK, "plain_bulk_ok": plain_check <= PLAIN_BULK,
                          "phase1_knn_largest_difference": worst_anchor, "phase1_knn_tolerance": KNN_MATCH,
                          "phase1_knn_reproduced": reproduced},
               "anchors": anchors, "seconds": round(time.time() - start, 1), **sc.run_metadata()}
    pd.DataFrame(results).to_csv(args.out_dir / "results.csv", index=False)
    sc.write_json(args.out_dir / "summary.json", summary)
    ok = summary["checks"]["plain_bulk_ok"] and reproduced is not False
    print(f"analyse: wrote {args.out_dir / 'results.csv'} in {summary['seconds']} s; checks "
          f"{'pass' if ok else 'FAIL'} (plain bulk {plain_check:.2e}, phase-1 kNN difference "
          f"{'not compared (subset)' if worst_anchor is None else f'{worst_anchor:.2e}'})")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=("describe", "analyse"), required=True)
    parser.add_argument("--covariant", default=rs.REFERENCE_TAG, help="reference covariant run (default %(default)s)")
    parser.add_argument("--covariant-root", type=Path, default=sc.COVARIANT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--limit", type=int, default=0, help="describe: only the first N sym folders per plane (testing)")
    parser.add_argument("--k", type=int, default=10, help="analyse: neighbours for kNN (default 10, as study 06)")
    parser.add_argument("--allow-failed-checks", action="store_true", help="analyse even if describe failed a check")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="describe: count the families and stop")
    args = parser.parse_args()
    return describe(args) if args.step == "describe" else analyse(args)


if __name__ == "__main__":
    sys.exit(main())
