#!/usr/bin/env python3
"""Step 1.1 (ACE): ACE vectors for the pristine cell and every transformed bulk cell.

Writes bulk_ace.npz, bulk_index.csv and bulk_settings.json into the global folder of the ACE
setting, calculations/descriptors/ACE/standard/global/<tag>/, next to the grain-boundary global
vectors of the same setting. bulk_ace.npz holds one X_<block> array per block (cells x 2P, float64)
and the property labels of each block.
"""
import os

# One thread per worker process; the parallelism is over cells.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ace_common as ac

sc = ac.sc
INDEX_FIELDS = ["index", "kind", "plane", "sym", "n_atoms", "n_Co", "n_Ge", "a", "b", "c",
                "min_pair_distance", "scale_corrected"]
_STATE = {}


def index_fields(blocks):
    return (INDEX_FIELDS + [f"within_species_spread_{name}" for name in blocks]
            + [f"atom_norm_{name}" for name in blocks] + ["source"])


def describe_cell(path, calculator):
    atoms, corrected = sc.read_structure(path)
    is_co = np.array(atoms.get_chemical_symbols()) == "Co"
    per_atom = calculator.per_atom(atoms)
    vectors, counts = ac.species_means(per_atom, is_co)
    lengths = atoms.cell.lengths()
    meta = {"n_atoms": len(atoms), "n_Co": int(counts[0]), "n_Ge": int(counts[1]),
            "a": f"{lengths[0]:.8f}", "b": f"{lengths[1]:.8f}", "c": f"{lengths[2]:.8f}",
            "min_pair_distance": f"{sc.min_pair_distance(atoms):.6f}", "scale_corrected": int(corrected),
            **{f"within_species_spread_{name}": f"{sc.within_species_spread(values, is_co):.6e}"
               for name, values in per_atom.items()},
            **ac.atom_norms(per_atom)}
    return {"vectors": vectors, "meta": meta}


def _init_worker(settings, chunk):
    _STATE["calculator"] = ac.Calculator(settings, chunk)


def _work(task):
    k, path = task
    try:
        return k, describe_cell(path, _STATE["calculator"]), None
    except Exception as exc:  # reported at the end; nothing is written then
        return k, None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ac.add_ace_arguments(parser)
    parser.add_argument("--bulk-root", type=Path, default=sc.BULK_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N transformed cells per plane (testing; the pristine cell and "
                             "the references are always included)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: calculations/descriptors/ACE/standard/global/<tag> (with _p<planes> / _lim<N> "
                             "when filtered)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="worker processes (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--chunk", type=int, default=128,
                        help="atoms per featomic call; bounds memory, does not change the numbers (default 128)")
    parser.add_argument("--overwrite", action="store_true", help="recompute even if the output exists")
    parser.add_argument("--dry-run", action="store_true", help="list the cells and stop")
    args = parser.parse_args()

    settings = ac.ace_settings(args)
    blocks = ac.blocks_for(settings)
    name = ac.ace_tag(settings)
    # A partial bulk set goes to its own folder, so it cannot stand in for the full one.
    if set(args.planes) != set(sc.PLANES):
        name += "_p" + "-".join(sorted(args.planes))
    if args.limit:
        name += f"_lim{args.limit}"
    out_dir = args.out_dir or ac.run_dir("global", name)
    if (out_dir / ac.BULK_FILE).exists() and not args.overwrite:
        print(f"{out_dir / ac.BULK_FILE} already exists; reusing it (pass --overwrite to recompute)")
        return 0
    cells = ac.collect_cells(args.bulk_root, args.planes, args.limit)
    if args.dry_run:
        print(f"{len(cells)} bulk cells, blocks {blocks}, output {sc.relative_path(out_dir)}")
        return 0

    start = time.time()
    probe = ac.Calculator(settings, args.chunk)
    first = describe_cell(cells[0][3], probe)
    widths = probe.widths()
    print(f"{len(cells)} bulk cells, features per atom "
          + ", ".join(f"{name} {width}" for name, width in widths.items())
          + f", {args.workers} worker(s)", flush=True)
    X = {name: np.zeros((len(cells), 2 * width)) for name, width in widths.items()}
    meta, errors = [None] * len(cells), []

    def store(k, result, error):
        if error:
            errors.append(f"{sc.relative_path(cells[k][3])}: {error}")
            return
        for name in blocks:
            X[name][k] = result["vectors"][name]
        meta[k] = result["meta"]

    store(0, first, None)
    tasks = [(k, cell[3]) for k, cell in enumerate(cells)][1:]
    if args.workers > 1:
        # featomic has already run in this process, and forking after that can hang the workers.
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker, initargs=(settings, args.chunk)) as pool:
            for outcome in pool.imap_unordered(_work, tasks, chunksize=4):
                store(*outcome)
    else:
        _STATE["calculator"] = probe
        for task in tasks:
            store(*_work(task))
    if errors:
        print("bulk cells failed; nothing written:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    rows = [{"index": k, "kind": kind, "plane": plane, "sym": sym, **meta[k], "source": sc.relative_path(path)}
            for k, (kind, plane, sym, path) in enumerate(cells)]
    layout = {name: f"[0:{width}] mean of Co-centred {name}, [{width}:{2 * width}] mean of Ge-centred {name}"
              for name, width in widths.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.write_npz(out_dir / ac.BULK_FILE, **{f"X_{name}": X[name] for name in blocks},
                 **ac.label_arrays(probe.properties, probe.property_names), species=np.array(sc.SPECIES))
    sc.write_csv(out_dir / "bulk_index.csv", rows, index_fields(blocks))
    sc.write_json(out_dir / "bulk_settings.json", {
        "ace": settings, "featomic_hypers": ac.featomic_hypers(settings), "blocks": blocks,
        "n_features_per_atom": widths, "vector_length": {name: 2 * width for name, width in widths.items()},
        "vector_layout": layout, "n_cells": len(cells), "limit": args.limit, "workers": args.workers,
        "chunk": args.chunk, "seconds": round(time.time() - start, 2), **ac.run_metadata(),
    })
    print(f"wrote {sc.relative_path(out_dir)} in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
