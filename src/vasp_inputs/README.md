# src/vasp_inputs

The verified VASP input generator (CLAUDE.md section 7) and the runner that executes its folders.
Standard library only; run with plain `python3 -B`. Version tag in git: `vasp_inputs-v<VERSION>`.

| File | What it does |
| --- | --- |
| `scaffold_interface_tree.py` | Builds `calculations/CoGe_interface/<plane>/symNNNNN/<k>x/` (POSCAR copied from the GB catalogue, empty `convergence/` and `non-soc\|soc/{scf,dos,bandstructure}/` with POTCAR symlinks). Folder scaffold only; writes no INCAR/KPOINTS. `python3 -B src/vasp_inputs/scaffold_interface_tree.py [--planes 100 ...] [--dry-run]` |
| `make_vasp_inputs.py` | The generator. Recipes are data (`RECIPES`); writes INCAR (line 1 = version stamp), KPOINTS, a POSCAR copy and an absolute POTCAR symlink per stage folder |
| `vasp_task.py` | One array task: `prepare` (checks, claim, node-local scratch), `binary`, `sample` (peak RSS), `finish` (validate, copy back, `result.json`) |
| `run_vasp_array.slurm` | The Slurm array script; the VASP environment block is verbatim from CLAUDE.md section 2 |
| `submit_vasp.py` | Submits the pending folders of a task list as one array |
| `test_make_vasp_inputs.py`, `test_vasp_task.py`, `tests/golden/` | Tests (a)–(d) of CLAUDE.md 7.4 plus the runner tests |

The SCF trial tool of 2026-10-13 (`scf_trials.py`, `.slurm`, its tests) only ran against v0.2.0. It
moved to `archive/interface_100_convergence_20261013/v0.2.0/src/vasp_inputs/` on 2026-10-15 (also in
git at `b7e8237`); see "SCF trials" below.

## Recipes (v0.3.0)

| Family / stage | Target | Note |
| --- | --- | --- |
| `CoGe_100` / `scf-nonsoc` | `calculations/CoGe_Bulk/CoGe_100/symNNNNN/non-soc/scf` | Bulk anchor: the tags of the 623 finished runs. Base only, never written (every target holds a run). Test (b) checks it against sym00000/00311/00622 |
| `CoGe_100_GB` / `convergence-nonsoc` | `calculations/CoGe_interface/100/symNNNNN/<k>x/convergence/e<ENCUT>_k1x<N>x<N>/` | ENCUT 350/400/450/500/600 eV at Γ 1×9×9, and 1×N×N (N = 3/5/7/9/11) at 500 eV: 9 points. Anchor plus the overrides listed in `RECIPES`; test (b) checks that they are the whole difference |
| `CoGe_100_GB` / `moment-check` | `calculations/CoGe_interface/100/symNNNNN/<k>x/moment-check/e500_k1x9x9/` | One point: the sweep's e500_k1x9x9 spin-polarized (user, 2026-10-15), for sym00177 and sym00190. Test (b) checks it differs from that sweep point only by the tags below |

Convergence INCAR, changes from the bulk anchor: ENCUT per point, EDIFF 1E-6, ISMEAR 1 (SIGMA 0.05
kept), LCHARG .FALSE., ICHARG 2 (atomic start), KPAR 4 (1 for 1×3×3), NCORE 4, NBANDS 512 added,
LORBIT dropped. Run profile: 16 MPI ranks.

v0.3.0 (2026-10-15) makes the runs non-magnetic: ISPIN 1, MAGMOM and AMIX_MAG dropped (user: zero
moment for now). ISPIN 1 gives the same answer as ISPIN 2 with MAGMOM 0 at about half the work per
SCF step, and no run can drift into a moment. Full reasoning per tag:
`notes/gb_submission_plan_20261015.md`.

v0.2.0 added damped density mixing after NELM: AMIX 0.2, AMIX_MAG 0.8 (half VASP's defaults) and
AMIN 0.01 (VASP's advice for a lattice vector > 50 Å); BMIX/BMIX_MAG keep the default 1.0. With
v0.1.0's default mixing, sym00177 e500_k1x3x3 sloshed for 100 steps without converging. v0.3.0 keeps
AMIX 0.2 and AMIN 0.01.

`moment-check` (2026-10-15): is the small moment that the 10-13 trials found at 1×3×3 still there at
1×9×9, and on which atoms? It takes the sweep's e500_k1x9x9 INCAR and changes ISPIN to 2 and adds
MAGMOM `64*1.0 64*0.0`, AMIX_MAG 0.8 and BMIX_MAG 0.0001 (the trial `linmag` route, which converged
on both cells), plus LORBIT 11 for the per-atom moments in OUTCAR. Its E0 is compared with the
sweep's ISPIN 1 run of the same point. Its records go to their own folder
(`results/CoGe_interface/100/moment_check_8x/`): claims are named `<sym>_<k>x_<point>`, the same as
the sweep point's.

## Generate

```bash
python3 -B src/vasp_inputs/make_vasp_inputs.py --family CoGe_100_GB --stage convergence-nonsoc \
    --thickness 8x --sym sym00190 sym00174 sym00016 sym00177 \
    --write-tasks results/CoGe_interface/100/convergence_8x/tasks.txt [--dry-run]
```

- A target holding any VASP output (or `result.json`) is **refused**; identical inputs are **skipped**
  (only the git hash in the stamp may differ); different inputs or extra files **stop** the run.
- Every folder is classified before anything is written: if any folder has a problem, no folder is
  written and no task list is written (since v0.2.0).
- The POSCAR must be a byte copy of the catalogue POSCAR, scale 1.0, `Co Ge`, the recipe's counts.
- Real writes refuse while `src/vasp_inputs/` has uncommitted changes; the stamp carries the hash of
  the last commit touching it.

## Run

```bash
python3 -B src/vasp_inputs/submit_vasp.py --tasks results/CoGe_interface/100/convergence_8x/tasks.txt \
    [--only <path substring> ...] [--mem 96G] [--nodelist qcg3-0] [--throttle 8] [--order cost|list] [--dry-run]
```

- The array runs on **qcg3-0 only** (`--nodelist` defaults to qcg3-0; user, 2026-10-13). With 16 ranks
  per run, 8 runs fill its 128 physical cores, which is the default `--throttle 8`.
- **Set `--mem` from measured memory, per class of point.** Slurm here does not enforce `--mem` (it is
  a reservation only), so 8 heavy runs whose real use exceeds the node's ~716 GB free would
  overcommit qcg3-0 and can take every run on it down. A 1×9×9 ISPIN 2 run used 112–124 GB; a 1×3×3
  run uses 6–9 GB. The sweep's submission plan (pilot first, then three classes) is in
  `notes/gb_submission_plan_20261015.md`.

- The pending folders are submitted heaviest first by default (`--order cost`), ranked by the k-points
  of the full mesh × ENCUT^1.5, so the 1×11×11 and 600 eV runs start first and no long run starts last.
  `--order list` keeps the task-list order (added 2026-10-13).

- `submit_vasp.py` skips folders with output or with a claim, writes
  `<records>/submissions/<time>/{tasks.txt,submit.json}` and logs to `<records>/logs/%A_%a.out`.
  `<records>` defaults to the task list's folder.
- `prepare` refuses a folder whose inputs are not exactly what the current generator renders, checks
  SLURM_NTASKS and KPAR/NCORE against the recipe profile, claims the folder
  (`<records>/claims/<sym>_<k>x_<point>.json`, O_EXCL) and stages INCAR/KPOINTS/POSCAR into
  `mkdtemp` under `$TMPDIR` (else `/tmp`, at least 10 GiB free). POTCAR there is a symlink, never a copy.
- `finish` validates the run from OUTCAR/OSZICAR/vasprun.xml: exit code, VASP 6.4.3, finished, EDIFF
  reached in fewer than NELM steps, POTCAR TITELs, NIONS, NELECT, the INCAR tags echoed by VASP
  (including AMIX, AMIX_MAG, AMIN, BMIX and BMIX_MAG when the INCAR sets them; OUTCAR prints two
  decimals, so BMIX_MAG 0.0001 shows as 0.00), the k-mesh, the KPAR/NCORE decomposition and
  NBANDS. `result.json` records the mixing values VASP used.
  - Valid: copies OUTCAR, OSZICAR, EIGENVAL, IBZKPT, DOSCAR, vasprun.xml, vasp.out (plus CHGCAR if
    LCHARG, PROCAR if LORBIT) via `*.copying` + SHA-256 + rename, writes `result.json` last (energies,
    E_F, magnetization for ISPIN 2 runs, NKPTS, NBANDS, NPLWV, SCF steps, wall time, peak RSS, VASP warnings), removes
    the scratch and the claim.
  - Invalid: the stage folder is untouched, the scratch and claim stay, and
    `<records>/failed/<sym>_<k>x_<point>_<job>.json` says why. Delete the claim to allow a rerun.
- `vaspout.h5`, WAVECAR and CHG are never copied back.

## SCF trials (2026-10-13, tool archived)

v0.2.0's sym00177 `e500_k1x3x3` run (job 12919) stopped at NELM 100 without converging (the moment
swung from 64 to about 7 μB; rms(c) never below 1.3). Four mixing routes were tried on sym00177 and
sym00190 at `e500_k1x3x3` (array 12960, qcg3-0, 16 ranks, NELM 200). Records:
`results/CoGe_interface/100/convergence_8x/scf_trials_20261013/` (`trials.tsv`, `trials/*.json`, logs).

| Variant | Change from v0.2.0 | Result |
| --- | --- | --- |
| `linmag` | BMIX_MAG 0.0001 | converged on both (31, 42 steps) |
| `wiki` | BMIX 0.0001, BMIX_MAG 0.0001 | diverged on both, VASP crashed |
| `nmseed` | run1 ISPIN 1 from atoms; run2 ISPIN 2 from run1's CHGCAR | converged (run1 31, 35 steps; run2 79, 144) |
| `nmseed_linmag` | as `nmseed`, run2 with BMIX_MAG 0.0001 | converged (run1 31, 35; run2 31, 27) |

The non-magnetic run1 of the seeded trials is the v0.3.0 SCF (bar NELM and LCHARG): it is the test
run behind v0.3.0. What the magnetic runs found is summarized in `notes/gb_submission_plan_20261015.md`.

## Tests

`python3 -B -m unittest discover -s src/vasp_inputs -p 'test_*.py'` (login node, < 1 s). After a
deliberate recipe change: bump `VERSION`, run
`python3 -B src/vasp_inputs/test_make_vasp_inputs.py --update-goldens`, show the golden diff, commit
and tag (CLAUDE.md 7.5).
