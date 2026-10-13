# CoGe_100 DOS and direct bands — September 2026

> Layout note (2026-10-02): run paths below are as they were during the campaign. The selected runs now sit directly in each `scf/`, `dos/` and `bandstructure/` folder; the other runs and the campaign scripts and records are under `archive/` at the same relative path. See [calculations/CoGe_Bulk/README.md](../calculations/CoGe_Bulk/README.md).

Campaign records referenced below are in `archive/results/b20_campaign/CoGe_sym_native/dos_bands_20260911/`.

This describes the configured campaign, not a claim that it has completed. Job
IDs are in `submission.json`; convergence decisions are in `dos_gate.json` and
`gate.json`; final validation will be recorded in `summary.json` and
`validation.tsv`. The launcher is `archive/calculations/CoGe_Bulk/run_bulk_dos_bands_once.sh`.

**Initial submission:** arrays 7760 (non-SOC) and 7761 (SOC), each with elements
0 and 1, were submitted on 11 September 2026 on qcg0. Job 7762 evaluates their
DOS convergence after both arrays end. The live VASP headers confirmed version
6.4.3, six MPI ranks, one thread per rank, and three k-point groups. Twelve new
regression tests and seven retained recovery tests passed before submission;
static parsing/input checks also passed against existing completed outputs.
Slurm displays 12 allocated logical CPUs per task because each of the six
physical cores has two hardware threads; VASP runs only six MPI processes.

1. **Scope and starting densities.** Use the 623 archived eight-atom structures,
   `sym00000`–`sym00622`, and their validated metallic SCFs selected in
   `smearing_rerun_20260828/summary.tsv`. Preserve those SCFs unchanged. Retain
   VASP 6.4.3, the original Co/Ge PAW/PBE potentials, 600 eV cutoff, Accurate
   precision, and electronic tolerance 1e-8 eV. No geometry relaxation, GPU,
   [110]/[111] cells, or Wannier interpolation is included.

2. **Non-SOC density refresh.** Re-converge each selected non-SOC density with
   `LMAXMIX=4`, retaining `ISPIN=2`, `MAGMOM=4*1.0 4*0.0`, a Gamma-centered
   11³ mesh, `ISMEAR=-1`, `SIGMA=0.05 eV`, and `ICHARG=1`. Write a new CHGCAR
   in the new campaign tree. This restores the higher-angular-momentum PAW
   information needed by fixed-density post-processing; merely changing the
   DOS input cannot recover coefficients absent from an old CHGCAR. Check the
   energy change is at most 1 meV/atom and |magnetization| <1e-4 muB/cell. The
   existing SOC densities already used `LMAXMIX=4` and are retained.
   [VASP ICHARG guidance](https://vasp.at/wiki/ICHARG).

3. **DOS.** Use fixed density (`ICHARG=11`), Gamma-centered 15³ sampling,
   `ISMEAR=-5`, `ALGO=Normal`, `LREAL=.FALSE.`, and `LORBIT=11`. Use 48 bands
   per collinear spin channel, or 96 SOC spinor bands; require the top band
   everywhere to exceed EF+6 eV. Tetrahedron integration is appropriate for
   metallic bulk DOS as well as insulating DOS; it does not require replacing
   the selected Fermi–Dirac SCF density. Preserve all atom/orbital projections.
   Raw DOS includes all occupied bands and a one-eV margin around the requested
   -10 to +5 eV window, with spacing <=0.005 eV. `dos_relative.tsv` contains
   3001 Fermi-referenced total-DOS points. Its energy zero is the DOS Fermi
   energy, not an absolute energy of zero.
   [VASP smearing guidance](https://vasp.at/wiki/Smearing_technique).

4. **Accuracy gate.** On sym00000 (inverted) and sym00001 (regular), compare
   15³ against 21³ DOS for both spin treatments, using the same charge density.
   Require |Delta EF| <=0.010 eV and integrated |DOS15-DOS21| / integral(DOS21)
   <=0.05 over [-10,+5] eV, aligning each DOS to its own EF. These are campaign
   acceptance thresholds, not universal convergence guarantees. Failure stops
   the release of the remaining structures. This checks DOS sampling, not a
   new cutoff or SCF-mesh convergence study.

5. **Direct bands.** Once the DOS gate passes, run both representative band
   calculations; their successful validation releases the remaining batch.
   Use the same density, spin treatment, and band count as DOS, `ICHARG=11`,
   `ISMEAR=0`, `SIGMA=0.05`, and 60 points on each of seven segments:
   Gamma-X-M-Gamma-R-X | R-M-X1 (420 k-point records including repeated endpoints).
   Reciprocal coordinates: Gamma=(0,0,0), X=(0,1/2,0), M=(1/2,1/2,0),
   R=(1/2,1/2,1/2), X1=(1/2,0,0). The additional M-X1 segment belongs to the
   cP1 path for B20 symmetry. Use the matching 15³ DOS EF to interpret band
   eigenvalues, never the line-path Fermi estimate. SOC retains `LSORBIT=.TRUE.`,
   `MAGMOM=24*0.0`, and `SAXIS=0 0 1`. `LWRITE_SPN` and the Wannier90 interface
   are deliberately deferred; PROCAR spin projections are not a .spn file.
   [Path](https://raw.githubusercontent.com/giovannipizzi/seekpath/main/seekpath/hpkot/band_path_data/cP1/path.txt),
   [Fermi reference](https://vasp.at/wiki/EFERMI).

6. **Execution and provenance.** Six MPI ranks on six physical cores, KPAR=3,
   NCORE=1, one thread per rank, Intel MPI pinning, and node-local disk scratch.
   The representative DOS jobs reserve 16/40 GiB (non-SOC/SOC) on qcg0 to
   accommodate the 21³ comparison. Production DOS requests 12/16 GiB; bands
   request 16/24 GiB. These initial memory estimates were scaled from recorded
   rank-0 VASP wavefunction/projector memory, not new timing benchmarks.
   Production uses `general`, excluding `qcg-docker`, one-day limits, and shared
   nodes; the eligible physical-core capacity limits six-core jobs to 31.
   Per-element band dependencies permit finished DOS structures to advance.
   Preserve executable hashes, an immutable launcher/worker snapshot, exact
   generated inputs, seed hashes, MPI exit status, timings, and checksummed
   output exports. Licensed POTCAR contents are not included in the report.

The full accepted campaign contains 2492 main DOS/band calculations, 623
non-SOC density refreshes, and four additional 21³ DOS comparisons. The latter
checks and refreshes are additional VASP executions, not additional structures.

Commands: `checks` submits the representatives and their automatic gate;
`submit-rest` refuses to submit until both gates have passed; `status` reports
job/gate state; `validate` checks completed exports. Duplicate submission
records and claimed/partial stages are protected from automatic replacement.
