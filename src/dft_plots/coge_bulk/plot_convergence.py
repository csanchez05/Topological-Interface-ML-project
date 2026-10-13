import argparse
import json
from pathlib import Path

if __package__:
    from .paths import PlotInputError, add_path_arguments, paths_from_args
else:
    from paths import PlotInputError, add_path_arguments, paths_from_args

parser = argparse.ArgumentParser(description="Plot existing CoGe convergence CSVs.")
add_path_arguments(parser, structure="sym00001")
parser.add_argument("--convergence-dir", type=Path,
                    help="Directory containing kpoint_convergence.csv and encut_convergence.csv")
args = parser.parse_args()
try:
    paths = paths_from_args(args, "convergence")
except PlotInputError as exc:
    parser.error(str(exc))
CONV = (args.convergence_dir.expanduser().resolve() if args.convergence_dir
        else paths.branch() / "convergence")
output_dir = paths.output_dir
missing = [str(CONV / name) for name in ("kpoint_convergence.csv", "encut_convergence.csv")
           if not (CONV / name).is_file()]
if args.show_paths:
    print(json.dumps({"convergence_dir": str(CONV), "output_dir": str(output_dir),
                      "missing": missing}, indent=2))
    raise SystemExit(1 if missing else 0)
if missing:
    parser.error("Missing convergence CSVs: " + ", ".join(missing)
                 + ". Supply --convergence-dir for an existing convergence study.")

import pandas as pd
import matplotlib.pyplot as plt

output_dir.mkdir(parents=True, exist_ok=True)

for fname, xcol, xlabel, title in [
    ("kpoint_convergence.csv", "kmesh",    "k-mesh (N×N×N)", "k-point convergence — CoGe SCF"),
    ("encut_convergence.csv",  "encut_eV", "ENCUT (eV)",     "ENCUT convergence — CoGe SCF"),
]:
    df = pd.read_csv(CONV / fname)

    # energy relative to the densest/highest setting (the reference), in meV/atom
    df["dE"] = df["energy_per_atom_meV"] - df["energy_per_atom_meV"].iloc[-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[xcol], df["dE"], "o-")
    ax.axhline(0,  color="k",    lw=0.5)
    ax.axhline(1,  ls="--", color="gray")
    ax.axhline(-1, ls="--", color="gray", label="±1 meV/atom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy/atom − converged (meV)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out = output_dir / fname.replace(".csv", ".png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)
