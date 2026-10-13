#!/usr/bin/env python3
"""Step 1.1: SOAP vectors for the pristine cell and every transformed bulk cell.

Writes bulk_soap.npz, bulk_index.csv and bulk_settings.json into the global
folder of the SOAP setting, calculations/descriptors/dscribe/standard/global/<tag>/,
next to the grain-boundary global vectors of the same setting.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import soap_common as sc

INDEX_FIELDS = ["index", "kind", "plane", "sym", "n_atoms", "n_Co", "n_Ge", "a", "b", "c",
                "min_pair_distance", "scale_corrected", "within_species_spread", "source"]


def collect_cells(bulk_root, planes):
    """(kind, plane, sym, path) for the pristine cell, each orientation reference and each sym cell."""
    cells = [("pristine", "", "pristine", sc.PRISTINE_POSCAR)]
    for plane in planes:
        folder = bulk_root / f"CoGe_{plane}"
        for name in (f"CoGe_{plane}_POSCAR", f"CoGe_{plane}_POSCAR.vasp"):
            if (folder / name).is_file():
                cells.append(("reference", plane, f"ref_{plane}", folder / name))
                break
        for sym_dir in sorted(folder.glob("sym[0-9]*")):
            if (sym_dir / "POSCAR").is_file():
                cells.append(("transformed", plane, sym_dir.name, sym_dir / "POSCAR"))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_soap_arguments(parser)
    parser.add_argument("--bulk-root", type=Path, default=sc.BULK_ROOT)
    parser.add_argument("--planes", nargs="+", default=list(sc.PLANES), choices=sc.PLANES)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: calculations/descriptors/dscribe/standard/global/<tag>")
    parser.add_argument("--overwrite", action="store_true", help="recompute even if the output exists")
    args = parser.parse_args()

    settings = sc.soap_settings(args)
    out_dir = args.out_dir or sc.run_dir("global", sc.soap_tag(settings))
    if (out_dir / "bulk_soap.npz").exists() and not args.overwrite:
        print(f"{out_dir / 'bulk_soap.npz'} already exists; reusing it (pass --overwrite to recompute)")
        return 0

    soap = sc.make_soap(settings)
    n_features = soap.get_number_of_features()
    cells = collect_cells(args.bulk_root, args.planes)
    print(f"{len(cells)} bulk cells, {n_features} SOAP features per atom", flush=True)

    start = time.time()
    X = np.empty((len(cells), 2 * n_features))
    rows = []
    for i, (kind, plane, sym, path) in enumerate(cells):
        atoms, corrected = sc.read_structure(path)
        is_co = np.array(atoms.get_chemical_symbols()) == "Co"
        per_atom = soap.create(atoms)
        X[i], counts = sc.species_mean(per_atom, is_co)
        lengths = atoms.cell.lengths()
        rows.append({
            "index": i, "kind": kind, "plane": plane, "sym": sym,
            "n_atoms": len(atoms), "n_Co": int(counts[0]), "n_Ge": int(counts[1]),
            "a": f"{lengths[0]:.8f}", "b": f"{lengths[1]:.8f}", "c": f"{lengths[2]:.8f}",
            "min_pair_distance": f"{sc.min_pair_distance(atoms):.6f}",
            "scale_corrected": int(corrected),
            "within_species_spread": f"{sc.within_species_spread(per_atom, is_co):.6e}",
            "source": sc.relative_path(path),
        })

    sc.write_npz(out_dir / "bulk_soap.npz", X=X,
                 feature_layout=np.array(f"[0:{n_features}] mean of Co-centred SOAP, "
                                         f"[{n_features}:{2 * n_features}] mean of Ge-centred SOAP"))
    sc.write_csv(out_dir / "bulk_index.csv", rows, INDEX_FIELDS)
    sc.write_json(out_dir / "bulk_settings.json", {
        "soap": settings, "n_features_per_atom": n_features, "vector_length": 2 * n_features,
        "n_cells": len(cells), "seconds": round(time.time() - start, 2), **sc.run_metadata(),
    })
    print(f"wrote {out_dir} in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
