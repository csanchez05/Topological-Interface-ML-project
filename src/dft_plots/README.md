# DFT plots

## Grain-boundary cells

`coge_interface/`: analysis of the GB cells, starting with the closest-pair survey that sets the
overlap limit (1.8 Å) for GB calculations. See `coge_interface/README.md`.

## Complete CoGe_100 batch

The CoGe_100 batch writes exactly 13 PNGs per structure to
`plots/CoGe/bulk/100/symXXXXX/{non_soc,soc}/`, using 160 DPI. MACE is deferred.

Create/reproduce the isolated rendering environment with `bash src/dft_plots/setup.sh`.
Activate it with `source src/dft_plots/env.sh`. From the project root, the batch
module is `python -B -m src.dft_plots.coge_bulk.plot_all --index N --dpi 160`.
The Slurm worker is `src/dft_plots/coge_bulk/plot_all.slurm`; use at most four
concurrent array tasks. Completed figures are reused after validating their
sources and hashes. Use `--verify-all` in place of `--index N` to validate the
whole collection; submit it with `sbatch src/dft_plots/coge_bulk/verify_all.slurm`.
Do not run it on the login node (light work only, see `CLAUDE.md`), even though
file reads are faster there: `verify_workspace.py` is the old login-node wrapper
for the same audit and is kept only for its records.
Reports are in `results/CoGe_100_plots/`, Slurm logs in its `logs/` folder.

## Standalone and reusable plotters

The CoGe scripts resolve paths relative to this project, independently of the
working directory. No cluster-specific absolute paths are required.

Calculation inputs live under:

```text
calculations/CoGe_Bulk/CoGe_<100|110|111>/symXXXXX/<non-soc|soc>/<scf|dos|bandstructure>/
```

Each stage folder holds its VASP files directly; see
`calculations/CoGe_Bulk/README.md`.

For [100], the structure manifest identifies `sym00001` as regular and
`sym00000` as inverted. The [110] scripts default to `sym00000`.
Figures are written to:

```text
plots/dft_plots/CoGe/bulk/<orientation>/symXXXXX/<non-soc|soc|comparison>/
```

Existing figure filenames are retained. Outputs for different structures and
orientations have separate directories.

## Usage

From the project root, inspect paths or plot with the same command:

```bash
python3 src/dft_plots/coge_bulk/plot_dos.py --show-paths
python3 src/dft_plots/coge_bulk/plot_dos.py
python3 src/dft_plots/coge_bulk/plot_bands.py --structure sym00001
python3 src/dft_plots/coge_bulk/plot_soc_overlay.py --structure sym00000
python3 src/dft_plots/coge_bulk/plot_dos.py --orientation 111 --structure sym00000 --show-paths
```

Every standalone script accepts `--project-root`, `--orientation`,
`--structure`, `--output-dir`, and `--show-paths`. Path inspection needs only
Python's standard library, creates no figures, and exits with status 1 when
required inputs are missing. `--help` lists the options. Absolute script paths
also work from another working directory; module invocation works with
`python3 -m src.dft_plots.coge_bulk.plot_dos` from the project root.

| Scripts | Default input |
| --- | --- |
| `plot_bands.py`, `plot_projected.py` | [100] inverted, non-SOC |
| `plot_dos.py`, `plot_bands_dos_combined_soc.py` | [100] regular, non-SOC |
| `plot_spin_texture.py` | [100] regular, SOC |
| `plot_*_inverted.py` | [100] inverted, SOC |
| `plot_*_110.py` | [110] `sym00000`, spin branch indicated by the script |
| `plot_soc_overlay.py` | [100] regular, both spin branches |

The historical filename `plot_bands_dos_combined_soc.py` continues to produce
the regular **non-SOC** combined plot.

## Input selection

Each stage folder holds the files of the one run used downstream, so a stage
resolves to its own folder. A stage without a non-empty `vasprun.xml` is
reported as missing rather than replaced by another run, and input-only
directories cannot satisfy a path check. A `final` link inside a stage is
still followed, for other datasets that use one.

Bands whose folder has an `energy_reference.json` (all CoGe_100 bands) use the
15³ DOS Fermi level of the same branch, as specified in
[the campaign notes](../../notes/dos_bands_reproducibility.md). Bands without
that record keep the SCF reference. SOC and non-SOC use their own references.

The JSON records inside the stage folders are kept byte-for-byte, so the
absolute paths they contain are historical locations;
`archive/reorganization_20261002/moves.tsv` maps every moved path to its new
location. The docstrings in `b20.py` still describe the earlier campaign
layout; the module is kept byte-identical because its hash is recorded in every
plot report.

At migration time, the [110]/[111] trees have no completed DOS/band results,
and this workspace has no convergence CSVs or Wannier band files. The
corresponding scripts report those missing inputs. To use separate existing
data, `plot_convergence.py` accepts `--convergence-dir`, and
`compare_vasp_wannier_bands.py` accepts `--wannier-dir` and
`--variant regular|inverted`. Wannier and direct VASP paths must match.

## Reusable plotter and dependencies

`dft_plots.b20.generate_b20_plots` accepts `calculations/CoGe_Bulk` as its
calculation root, supports [100]/[110]/[111], and uses the same stage
selection and output layout. Pass `structure="sym00000"` to select a specific
symmetry index across orientations, or `orientation_roots` for explicit
symmetry directories. Other materials can still use direct stage directories
with an uppercase `SOC/` child. Missing data are listed in the returned report.
`relaxation.py` already accepts explicit input/output paths and needs no path
migration.

Rendering requires NumPy, Matplotlib, and pymatgen; convergence plots also use
pandas, and band/spin projection scripts use PyProcar and PyVista. Use an
environment containing these packages. Path checks do not need them.

Run the path resolver regression checks with:

```bash
python3 -B -m unittest discover -s src/dft_plots/tests -v
```
