# DOS/band recovery — 12 September 2026

> Layout note (2026-10-02): run paths below are as they were during the campaign. The selected runs now sit directly in each `scf/`, `dos/` and `bandstructure/` folder; the other runs and the campaign scripts and records are under `archive/` at the same relative path. See [calculations/CoGe_Bulk/README.md](../calculations/CoGe_Bulk/README.md).

Both representative gates passed. Before recovery, 595/2492 main calculations
were recorded complete; 1897 main stages and 416 non-SOC density refreshes remained.

The input validator rejected 416 structures because its 2e-7 fractional-position
tolerance was tighter than CHGCAR's six-decimal output. All 1246 original seed
headers agree within 3.3e-7; the corrected tolerance is 5.1e-7. No coordinates or
densities were changed. The SOC no-output watchdog increased from 15 to 60 minutes:
qcg3-0 iterations were approaching 15 minutes and were being interrupted.

Submitted at 20:35 UTC: **9774**, 426 non-SOC workers; **9775**, 558 SOC workers.
Each resumes missing DOS then bands, with any required non-SOC refresh first.
Six MPI ranks, 16/24 GiB respectively, general partition, all four eligible nodes.
Completed stages are reused; scientific inputs and original frozen code remain intact.
Old claims were archived, not deleted; 20 blocked band jobs and finalizer 9712 were
superseded. Final checksum audit **9776** waits for both recovery arrays.

Eighteen DOS/recovery regression tests and seven retained SCF-recovery tests passed.
Exact scope, code hashes, submission commands and IDs are in
`archive/results/b20_campaign/CoGe_sym_native/dos_bands_20260911/recovery_20260912/`.
