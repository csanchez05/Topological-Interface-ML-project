# CoGe bulk results validation — completed 11 September 2026

> Layout note (2026-10-02): run paths below are as they were during the campaign. The selected runs now sit directly in each `scf/`, `dos/` and `bandstructure/` folder; the other runs and the campaign scripts and records are under `archive/` at the same relative path. See [calculations/CoGe_Bulk/README.md](../calculations/CoGe_Bulk/README.md).

The campaign is complete. Final validation finished at **04:24:43 UTC**: **623 complete structure pairs, 1,246 validated selected SCFs, and zero unresolved results**. All 29 previously unfinished structures were recovered, requiring 74 successfully completed SCFs. The queue was empty at the final status check.

| Final results | Non-SOC | SOC |
|---|---:|---:|
| Validated selected SCFs | 623 | 623 |
| Probe classification | All metallic | All metallic |
| Selected settings | ISMEAR=-1, SIGMA=0.05 eV | ISMEAR=-1, SIGMA=0.05 eV |
| Mean TOTEN, eV/eight-atom cell | −47.765171888636 | −47.816577221685 |
| TOTEN spread, eV/cell | 0.00000112 | 0.00000108 |

Recovery preserved completed stages and all original outputs. VASP 6.4.3 ran with the same scientific settings and six MPI ranks/12 GiB per structure. Node-local scratch, fail-fast MPI handling, bounded transfers and checksum verification protected the new results. Eight slow unfinished workers were moved to idle qcg0 capacity while other qcg3-0 jobs continued; partial outputs were retained. The recovery record documents the initial validator correction and intentional job cancellations.

The final merged audit checked selected results and their -5 probes for normal OUTCAR/XML termination, explicit EDIFF convergence, effective numerical settings and mesh, complete finite EIGENVAL tables, consistent classifications/energies/final links, and complete CHGCAR density and PAW augmentation payloads. **All 121 previously inaccessible final charge files passed**, and the final all-structure audit reported no unresolved files.

These results support numerical consistency among the transformed structures. They do not replace cutoff, k-mesh, smearing-width or magnetic-state convergence studies. TOTEN is the finite-smearing free energy; XML termination was checked, not a full XML parse.

See the [final machine-readable summary](../archive/results/b20_campaign/CoGe_sym_native/smearing_rerun_20260828/summary.tsv), [validation statistics](../archive/results/b20_campaign/CoGe_sym_native/smearing_rerun_20260828/recovery_20260911/validation-job-7713/summary.json), [recovery record](scf_recovery_20260911.md), and [brief reproducibility guide](scf_reproducibility.md). The earlier dated validation tables remain archived as the pre-recovery audit.
