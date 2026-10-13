# Fixed recovery: 11 September 2026

> Layout note (2026-10-02): run paths below are as they were during the campaign. The selected runs now sit directly in each `scf/`, `dos/` and `bandstructure/` folder; the other runs and the campaign scripts and records are under `archive/` at the same relative path. See [calculations/CoGe_Bulk/README.md](../calculations/CoGe_Bulk/README.md).

Records referenced below are in `archive/results/b20_campaign/CoGe_sym_native/smearing_rerun_20260828/recovery_20260911/`.

**Completed at 04:24:43 UTC.** All 29 recovery workers finished successfully in their latest attempts. The merged audit validated all 623 structure pairs (1,246 selected final SCFs and their probes), with zero unresolved results. Recovery required 74 successful SCFs; the final queue check was empty. Final evidence is in `validation-job-7713/summary.json`, `summary.tsv`, and the header-only `missing.tsv`.

Scope: the 29 structures listed in `submission.json`, not another full-campaign rerun. Resume from the earliest missing/failed method, preserving completed predecessors. Expected work was 74 SCFs if the remaining probes remain metallic. All original output directories are retained.

Physics is unchanged: VASP 6.4.3, six MPI ranks, KPAR=3, NCORE=1, the existing -5 classification and conditional -1/SIGMA=0.05 protocol. Each array element requests 12 GiB and one day in `general`, excluding `qcg-docker`; maximum concurrency is 29.

The interrupted August runs had reached EDIFF convergence before an Intel Fortran CLOSE error. The recovery writes VASP outputs on verified node-local `/tmp` (at least 10 GiB free), not directly on NFS. Each MPI step uses `--kill-on-bad-exit=1 --wait=120`; 15 minutes without output changes terminates the affected step. Transfers have 120-second deadlines, three attempts, and SHA-256 verification before final links are published. Local scratch is retained, including on transfer failure.

Validation checks normal OUTCAR/XML termination, explicit electronic convergence, effective settings, complete finite EIGENVAL tables, and every CHGCAR density/augmentation block. Summaries merge stages independently, including successful stages from originally failed workers. Per-structure audit deadlines produce an explicit unresolved list instead of an indefinitely blocked finalizer.

Initial recovery array 7683 exposed a validator bug before 20 workers launched VASP: tetrahedron integration weights were incorrectly constrained to 0–1. This check was removed, retaining finite-number/table validation and the original integer-distance classification. The other nine workers had already begun their required non-SOC -5 SCFs. Their successful numerical outputs are preserved and revalidated; the exact post-run parser exception establishes that their MPI launcher had already returned zero. The initially dependent summary job 7684 was cancelled so validation could follow the corrected recovery. This was not a change in scientific parameters. [VASP's explanation of tetrahedron weights](https://vasp.at/forum/viewtopic.php?t=18648).

`job-*/worker-*/records.json` records each selected stage, runtime, checks, seed provenance, node and scratch path. `validation-job-*/` contains the merged final audit and unresolved paths, if any. Historical submissions and failures remain available here; the most recent `submission.json` identifies the current array and dependent validator.

At approximately 02:52 UTC, an independent read-only check in a `qcg0` allocation rechecked all 121 selected final results whose CHGCAR access had timed out on the login node. All 121 passed numerical and full charge-payload validation; zero remained unresolved. Array 7713 was still processing SOC stages at that point. The final dependent validator is job 7714.

After all eight initial `qcg0` workers completed, eight slow unfinished SOC workers (49, 176, 192, 235, 251, 371, 378, 379) were moved from `qcg3-0` to the now-idle `qcg0` as array **7749**. Their original 7713 elements were intentionally cancelled; partial local outputs remain in the scratch paths recorded by those workers. Completed stages are reused, not rerun. The other thirteen `qcg3-0` workers continue. Resources and scientific settings are unchanged; this uses idle capacity to reduce the long SOC tail. Job 7714 now waits for both arrays 7713 and 7749. Exact migration scope is in `migration_submission.json`.
