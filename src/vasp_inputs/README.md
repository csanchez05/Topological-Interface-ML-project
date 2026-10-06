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

## Recipes (v0.2.0)

| Family / stage | Target | Note |
| --- | --- | --- |
| `CoGe_100` / `scf-nonsoc` | `calculations/CoGe_Bulk/CoGe_100/symNNNNN/non-soc/scf` | Bulk anchor: the tags of the 623 finished runs. Base only, never written (every target holds a run). Test (b) checks it against sym00000/00311/00622 |
| `CoGe_100_GB` / `convergence-nonsoc` | `calculations/CoGe_interface/100/symNNNNN/<k>x/convergence/e<ENCUT>_k1x<N>x<N>/` | ENCUT 350/400/450/500/600 eV at Γ 1×9×9, and 1×N×N (N = 3/5/7/9/11) at 500 eV: 9 points. Anchor plus the overrides listed in `RECIPES`; test (b) checks that they are the whole difference |

Convergence INCAR, changes from the bulk anchor: ENCUT per point, EDIFF 1E-6, ISMEAR 1 (SIGMA 0.05
kept), LCHARG .FALSE., ICHARG 2 (atomic start), KPAR 4 (1 for 1×3×3), NCORE 4, NBANDS 512 added,
LORBIT dropped. MAGMOM `64*1.0 64*0.0` follows the bulk rule. Run profile: 16 MPI ranks.

v0.2.0 adds damped density mixing after NELM: AMIX 0.2, AMIX_MAG 0.8 (half VASP's defaults) and
AMIN 0.01 (VASP's advice for a lattice vector > 50 Å); BMIX/BMIX_MAG keep the default 1.0. With
v0.1.0's default mixing, sym00177 e500_k1x3x3 sloshed for 100 steps without converging.

## Generate

```bash
python3 -B src/vasp_inputs/make_vasp_inputs.py --family CoGe_100_GB --stage convergence-nonsoc \
    --thickness 8x --sym sym00485 sym00174 sym00016 sym00177 \
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
    [--only <path substring> ...] [--mem 96G] [--nodelist qcg3-0] [--throttle 8] [--dry-run]
```

- `submit_vasp.py` skips folders with output or with a claim, writes
  `<records>/submissions/<time>/{tasks.txt,submit.json}` and logs to `<records>/logs/%A_%a.out`.
  `<records>` defaults to the task list's folder.
- `prepare` refuses a folder whose inputs are not exactly what the current generator renders, checks
  SLURM_NTASKS and KPAR/NCORE against the recipe profile, claims the folder
  (`<records>/claims/<sym>_<k>x_<point>.json`, O_EXCL) and stages INCAR/KPOINTS/POSCAR into
  `mkdtemp` under `$TMPDIR` (else `/tmp`, at least 10 GiB free). POTCAR there is a symlink, never a copy.
- `finish` validates the run from OUTCAR/OSZICAR/vasprun.xml: exit code, VASP 6.4.3, finished, EDIFF
  reached in fewer than NELM steps, POTCAR TITELs, NIONS, NELECT, the INCAR tags echoed by VASP
  (including AMIX, AMIX_MAG and AMIN when the INCAR sets them), the k-mesh, the KPAR/NCORE
  decomposition and NBANDS. `result.json` records the mixing values VASP used.
  - Valid: copies OUTCAR, OSZICAR, EIGENVAL, IBZKPT, DOSCAR, vasprun.xml, vasp.out (plus CHGCAR if
    LCHARG, PROCAR if LORBIT) via `*.copying` + SHA-256 + rename, writes `result.json` last (energies,
    E_F, magnetization, NKPTS, NBANDS, NPLWV, SCF steps, wall time, peak RSS, VASP warnings), removes
    the scratch and the claim.
  - Invalid: the stage folder is untouched, the scratch and claim stay, and
    `<records>/failed/<sym>_<k>x_<point>_<job>.json` says why. Delete the claim to allow a rerun.
- `vaspout.h5`, WAVECAR and CHG are never copied back.

## Tests

`python3 -B -m unittest discover -s src/vasp_inputs -p 'test_*.py'` (login node, < 1 s). After a
deliberate recipe change: bump `VERSION`, run
`python3 -B src/vasp_inputs/test_make_vasp_inputs.py --update-goldens`, show the golden diff, commit
and tag (CLAUDE.md 7.5).
