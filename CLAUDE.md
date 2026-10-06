# CLAUDE.md — ML_Interface_Project

B20 CoGe. Fixed-geometry VASP on the 1,541 transformed bulk cells (planes 100/110/111), a catalogue
of 1,541 grain-boundary (GB) sym folders × 9 thicknesses = **13,868 usable GB POSCARs**, and the SOAP →
KRR / PCA / UMAP search for unusual interface spin textures. Git (since 2026-10-05) tracks code only
(`.gitignore` excludes `calculations/ structures/ results/ archive/ plots/`); so far only `CLAUDE.md`
and `src/vasp_inputs/` are committed. Back up any file before editing it (originals go to `archive/`).

## 1. Working agreement

- I write and smoke-test code and run small test jobs (at most a few structures, e.g. one
  representative per recipe); **the user submits full Slurm runs** (whole planes or lists). Do not
  start KRR/PCA/VASP work or new analyses unasked; say so first if a task needs an extra script.
- When a plan is finished (and approved, if in plan mode), carry it out without waiting for another
  yes. State the scope up front: how many structures, which node, how many cores.
- Keep the safety steps: back up before editing, read-only checks before moving data, and stop and
  report on any unexplained guard failure.
- Explain in plain words and map every piece of work to a stage name or number (section 5).
- One condensed output file per run; anything the user may want to vary is a CLI flag (`--radii`).
- **Searches stay under `/home/calvinsanchez`.** Never grep/find from `/`, `/opt`, `/mnt` or other
  shared paths; if something lives outside the home directory, ask where it is.

## 2. Cluster rules (QCG, Slurm)

| Node | CPUs (physical) | RAM | Use |
|---|---|---|---|
| qcg3-0 | 256 (128), AMD EPYC 7T83 | 773 GB | preferred; sometimes DOWN, so check `sinfo` first |
| qcg0 | 96 (48), Xeon Gold 6248R | 773 GB | fallback |
| qcg1-1, qcg2-2 | 16 (8) each, i7-10700F | 32 GB | small jobs |
| qcg-docker | 8 | 8 GB | always `--exclude=qcg-docker` |

- One partition, `general`, no time limit and no GPUs. `sacct`/accounting is unavailable. Never
  change Slurm settings.
- **Login node `qcg-login` (4 cores, 3 GB, about 2 GB free): light work only.**
  - Allowed: editing, `ls`/`cat`/small `grep`, `sbatch`/`squeue`/`sinfo`, unit tests on synthetic
    data, `--dry-run`.
  - Anything that reads many outputs, needs more than about 1 GB, or runs longer than 1–2 min goes
    to Slurm: `sbatch`, or
    `srun -p general --exclude=qcg-docker -c 4 --mem=8G -t 00:30:00 bash -lc 'cd <root> && …'`.
  - Never run VASP, SOAP, KRR or plotting batches there. Anything that loads the GB vectors
    (`find_duplicates.py`, `krr_classify.py`, `select_gbs.py`: about 1.9 GB) goes to a compute node.
- **Batches:** many separate analyses become one whole-node job (qcg3-0, else qcg0) with an internal
  scheduler that keeps every core busy and uses the RAM (pattern: `src/ML_dscribe_scripts/rcut_scan_all.py`).
  Do not use many small array tasks for this.
- **VASP 6.4.3 (CPU)**, the only VASP on this cluster. Arrays use `--nodes=1 --cpus-per-task=1
  --hint=nomultithread --no-requeue --exclude=qcg-docker` and `%31`. Job environment, verbatim from
  the campaign:

  ```bash
  module purge; module load vasp/vasp.6.4.3-intel2024
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_STACKSIZE=512M OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
  export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=core I_MPI_PIN_RESPECT_CPUSET=1 I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=1
  ulimit -s unlimited
  srun --mpi=pmi2 --kill-on-bad-exit=1 --wait=120 --ntasks=$N --ntasks-per-node=$N --cpus-per-task=1 --cpu-bind=cores vasp_std > vasp.out 2>&1   # vasp_ncl for SOC
  ```
  Binaries: `/mnt/apps/QCGCodes/codes/VASP/vasp.6.4.3/bin/vasp_{std,ncl}`. Proven bulk profiles:

  | Cell | Ranks | KPAR / NCORE | Memory | Note |
  |---|---|---|---|---|
  | 100 (8 atoms) | 6 | 3 / 1 | 12 GiB | DOS/bands 12–40 GiB |
  | 110 (16) | 6 | 3 / 1 | 28 GiB | SOC needed 19.7 GiB, so 16 was too little |
  | 111 (48), non-SOC | 6 | 1 / 2 | 32 GiB | −5 / −1 took 10.3 h / 6.1 h |
  | 111 (48), SOC | 12 | 1 / 4 | 128 GiB | NBANDS 375; **unresolved**: not converged within 24 h |
- **Python environments:**
  - `source src/mlenv/env.sh`: SOAP and ML (DScribe, scikit-learn, UMAP); no pymatgen.
  - `src/dft_plots/.venv`: pymatgen, seekpath, spglib (plots; jobs stage it node-locally).
  - Campaign and standard tools: plain `python3 -B`.
- **Do not reuse** the per-structure `submit.sh` / `submit_vasp.sh` files in `archive/` and
  `structures/`. They are from another cluster: partition `snsm_itn19`, `apps/vasp/6.2.0`, `mpirun`.

## 3. Never overwrite finished runs

- A stage folder that holds **any VASP output** (OUTCAR, OSZICAR, vasprun.xml, CHGCAR, …) is off
  limits. Never write into it, rerun it in place, or "fix" its files. A rerun runs in scratch; the
  user decides whether and how it replaces the kept run (old run → `archive/`). Probe-only branches
  (46 CoGe_110 SOC, CoGe_111 sym00000 SOC) have no live `soc/scf/` folder: their probes are in `archive/`.
- VASP runs in node-local scratch (`mktemp -d` under `$TMPDIR`); results are copied into the stage
  folder only after validation. Failed scratch is kept. Stage folders hold one run's files directly:
  no `job-N/` subfolders, `final` links, templates, claim/complete JSONs, logs or scripts. Those go
  to scratch, `results/`, or `archive/`. The plot path resolver `src/dft_plots/coge_bulk/paths.py`
  relies on this layout.
- Generators: a target with any VASP output is **refused**; identical existing inputs are **skipped**;
  different existing inputs **stop and report**.
- The POTCAR master is `calculations/CoGe_Bulk/CoGe_100/sym00000/non-soc/scf/POTCAR`. **Never move or
  delete it**: every CoGe_110/111 POTCAR is an absolute link to it.
- ML outputs go to one folder per settings tag, so reruns never collide; use `--overwrite` only when
  the user asks.
- Superseded material moves to `archive/` at the same relative path. Nothing is deleted.

## 4. VASP defaults (bulk; proven by 4,779 finished stage folders) and why

N = atoms in the cell: 8 (100), 16 (110), 48 (111); half Co, half Ge.

| Setting | Value | Why |
|---|---|---|
| POTCAR | PAW_PBE `Co 02Aug2007`, `Ge 05Jan2001`, Co first; NELECT 52 / 104 / 312 (100 / 110 / 111) | Inherited. Header lines may be read (`grep -e TITEL -e ZVAL -e ENMAX POTCAR`); never `cat`, copy, export or print the rest. Generators make a symlink to the master. |
| ENCUT, PREC | 600 eV, Accurate | Retained from the earlier campaign; no convergence sweep |
| EDIFF | 1E-8 | Tight enough to compare symmetry-equivalent cells (TOTEN spread ~1e-6 eV/cell) |
| ALGO, LREAL, ISYM, IBRION, NSW | Normal, .FALSE., 2, −1, 0 | Fixed geometry, reciprocal-space projection |
| ISPIN, MAGMOM (non-SOC) | 2; `N/2*1.0 N/2*0.0` (Co 1 μB, Ge 0; POSCARs are Co then Ge) | Lets a moment form if one is stable. It collapses: \|m\| < 1e-4 μB/cell. Not a magnetic ground-state search. |
| SOC | `vasp_ncl`, LSORBIT .TRUE., SAXIS `0 0 1`, MAGMOM `3N*0.0`, no ISPIN | SAXIS: spin axis along cubic z (VASP's default, written explicitly). MAGMOM zero because the SOC run reads the non-SOC CHGCAR and inherits its magnetization from it. |
| LMAXMIX | 4 | Keeps the d-channel PAW density in CHGCAR for restarts and ICHARG 11. The first non-SOC runs used 2, which is why 623 "density refresh" runs exist. |
| ISMEAR, SIGMA (scf) | −5 probe, then −1 / 0.05 eV | Campaign rule: keep the −5 probe only if gap > 0.01 eV **and** every occupation is within 1e-6 of an integer; otherwise rerun −1 from the probe's CHGCAR. All 1,246+ probes were metallic. 0.05 eV is narrower than the original 0.1; no width study. |
| ICHARG, ISTART | non-SOC probe 2; SOC probe and −1 reruns 1; dos/bands 11; ISTART 0 | Each step seeds from the previous validated CHGCAR |
| NELM | 100 non-SOC, 120 SOC | Every kept run converged below it |
| LORBIT, BANDGAP | 11, KPOINT | LORBIT 11: lm-resolved PROCAR/DOSCAR for the orbital projections and the SOC spin texture. BANDGAP only adds reporting. |
| LWAVE, LCHARG | .FALSE.; .TRUE. in scf only | WAVECAR is never reused |
| KPOINTS (scf) | Γ-centred 11×11×11 (100), 8×8×11 (110), 7×8×5 (111) | 110/111 scaled from 11³ on the cubic cell; no convergence study |
| dos | ICHARG 11 from `../scf/CHGCAR`, ISMEAR −5, Γ 15×15×15, NBANDS 48 / 96 (100 non-SOC / SOC); EMIN = min(E_F−11, lowest−1), EMAX = E_F+6, NEDOS = ceil((EMAX−EMIN)/0.005)+1 | 15³ vs 21³ moved E_F by ≤ 2.6 meV; NBANDS puts the top band above E_F+6 eV |
| bandstructure | ICHARG 11, ISMEAR 0, NBANDS as dos; HPKOT cP1 line path Γ-X-M-Γ-R-X \| R-M-X₁, 60 points/segment | Energies referenced to the **dos** E_F (`energy_reference.json`), never the line-path E_F. Path valid for the 8-atom cubic cell only. |
| Not set in the bulk runs | LASPH (so off), GGA_COMPAT, ADDGRID, LDIPOL/IDIPOL | LASPH off was an **oversight**: finished bulk runs stay as they are, and LASPH for GB runs is an open question (below) |
| Not established | convergence in ENCUT, k-mesh, smearing width, magnetic ground state | Say so whenever results depend on them |

Bulk status (details in `calculations/CoGe_Bulk/README.md`):

| Plane | Cells | non-SOC scf | SOC scf | dos, bandstructure |
|---|---|---|---|---|
| CoGe_100 | 623 | 623 | 623 | 623 per branch |
| CoGe_110 | 543 | 543 | 497 (46 only reached the probe) | not run |
| CoGe_111 | 375 | sym00000 only | none | not run |

**GB / interface VASP settings.** Decided for the [100] GBs (user, 2026-10-05 and 2026-10-06; details
in `notes/interface_100_convergence_20261005.md`):
- Scope: the 66 [100] 8x cells of the Stage 2 list (`results/dscribe/total_sample_list.csv`). Static,
  no relaxation. Closest pairs: 61 of 66 have one < 2.0 Å, 33 < 1.5 Å, the worst 0.77 Å (bulk Co–Ge
  bond 2.40 Å). How production treats the overlapping cells is **open** (below).
- ISMEAR 1, SIGMA 0.05 eV for scf, dos and bands, with no −5 probe. EDIFF 1E-6 for the convergence
  sweep, 1E-8 for production. LASPH off, as in bulk. ALGO Normal, as in bulk.
- Damped mixing (v0.2.0, 2026-10-06): AMIN 0.01, AMIX 0.2, AMIX_MAG 0.8; BMIX/BMIX_MAG stay 1.0. With
  VASP's default mixing the 74 Å cell sloshed (sym00177, no close pair: 100 steps, no convergence).
- Convergence sweep first (recipe `CoGe_100_GB` / `convergence-nonsoc`, section 7), non-SOC, 16 ranks:
  - structures sym00485 (closest pair 1.49 Å, the median) and the clean cells sym00174 (2.02 Å),
    sym00016 (2.23 Å) and sym00177 (2.40 Å). sym00431 (0.77 Å, diverged) and sym00144 (1.14 Å)
    were dropped on 2026-10-06.
  - ENCUT 350–600 eV at Γ 1×9×9
  - Γ 1×N×N (N = 3–11) at 500 eV; kx stays 1 along the 74 Å axis
  - The references are 600 eV and N = 11, so the sweep shows convergence *towards* them, not of them.
- The production ENCUT and k-mesh come from the sweep; the user confirms them.

Everything else is **not decided. Ask; do not invent or copy bulk values silently.** Open questions:
- the overlapping production cells (61 of 66 have a pair < 2.0 Å): delete one atom of each close
  pair, relax, or leave them out. PAW is not valid at 0.77 Å, and that cell diverged.
- relaxation, if ever: ISIF, IBRION, EDIFFG, NSW, selective dynamics, ENCUT (Pulay stress)
- MAGMOM start and SAXIS for the GB SOC runs (bulk SAXIS `0 0 1` is an in-plane axis of the GB cell)
- NBANDS for production, dos and bands; the dos mesh; the band path (the bulk cP1 path is invalid here)
- Stage 2 E_GB needs a bulk ε at the same ISMEAR, ENCUT and k-density. The finished bulk runs
  (ISMEAR −1, 600 eV) are not that reference.
- the 110/111 GBs: all of the above, plus whether they too stay unrelaxed

The 5 template sets in `structures/CoGeGrainBoundaries/CoGe_100_GBs/Interface/` (sym00010 at 4x, 5x
and 6x; sym00013 at 8x; sym00008 at 11x: `INCAR.relax`, `INCAR.static`, `submit_vasp.sh`) are broken
and must not be reused:
- the nested MAGMOM `3*2*0.0` is invalid
- static ICHARG 11 has no CHGCAR to read
- ENCUT is 520, and the module is VASP 6.2.0

## 5. Stages: two separate schemes

**"Stage N" always means the SOAP guide (B).** VASP stages (A) are referred to by name only.

**A. VASP calculation stages.** These names are also the generator's `--stage` values.

| Name | Folder (in `symNNNNN/`) | Binary | Reads | Runs after |
|---|---|---|---|---|
| `scf-nonsoc` | `non-soc/scf/` | vasp_std | −5 probe from atomic density, then −1 rerun | — |
| `scf-soc` | `soc/scf/` | vasp_ncl | `../../non-soc/scf/CHGCAR`, then the same probe rule | `scf-nonsoc` |
| `dos-nonsoc`, `dos-soc` | `<branch>/dos/` | as branch | `../scf/CHGCAR` | scf of that branch |
| `bands-nonsoc`, `bands-soc` | `<branch>/bandstructure/` | as branch | `../scf/CHGCAR`; E_F from `../dos/` | dos of that branch |
| `convergence-nonsoc` (GB) | `<k>x/convergence/e<ENCUT>_k1x<N>x<N>/` | vasp_std | atomic density (ICHARG 2) | — |
| `relax` (GB) | not defined; the [100] GBs stay static | | | |

**B. SOAP guide "Unsupervised Discovery of Interface Spin Textures in B20 CoGe via SOAP"** (Dr. Gayles' report):

| Stage | Content | Status |
|---|---|---|
| 1 | 1.1 SOAP of pristine + 1,541 bulk cells · 1.2 bulk invariance check · 1.3 GB descriptors (13,868) · 1.4 KRR GB vector → plane and operation · 1.5 coinciding vectors / distinct GBs (then raise n_max, l_max) · r_cut scan 4–12 Å | Done at r_cut 6: plane 0.999, grain class 0.49 (baseline 0.19), 1,112 distinct GBs. Scan on all thicknesses: same conclusions. Code: `src/dscribe/CoGe/`, `src/ML_dscribe_scripts/` |
| 2 | KRR GB vector → GB energy, from VASP relaxations of FPS-chosen GBs; E_GB = (E_cell − n·ε_bulk) / (2A), two interfaces per cell | Selection done (`select_gbs.py`): six lists of 30 GBs (FPS and median, each in PCA, t-SNE and UMAP), in `plots/CoGe/dim-reduction/`. Their union, 162 8x POSCARs, is `results/dscribe/total_sample_list.csv`. VASP: [100] ENCUT/k sweep on 4 GBs, unrelaxed: v0.1.0 (2026-10-05) failed (sloshing; divergence at 0.77 Å), v0.2.0 with damped mixing and clean cells (2026-10-06) awaits its first test; production not started. |
| 3 | KRR GB vector → vectorized SOC spin texture (PROCAR σx/σy/σz) | Not started |
| 4 | PCA/UMAP anomaly map, bulk at the origin; isolated GBs are candidates | Not started |

Fixed choices:
- SOAP: r_cut 6 Å (then 8, 10), n_max 8, l_max 6, σ 0.5 Å, `gto`. Standard SOAP first (it cannot tell B20 enantiomers apart); covariant SOAP only if Stage 3 fails.
- GB vector: the mean of interface Co-centred vectors joined to the mean of interface Ge-centred vectors (interface radius 8 Å, both interfaces).
- The 4x–12x versions of one GB, and coinciding copies, **never** go to different train/test folds.

## 6. Layout and naming

```text
calculations/
  CoGe_Bulk/CoGe_<plane>/symNNNNN/<non-soc|soc>/<scf|dos|bandstructure>/   one kept run, files directly inside
  CoGe_interface/<100|110|111>/symNNNNN/<4..12>x/  POSCAR, convergence/, <non-soc|soc>/<scf|dos|bandstructure>/ (POTCAR links only; README there)
  descriptors/dscribe/{standard,covariant}/{global,both_interfaces}/<tag>/   SOAP descriptors (Stage 1),
                                           files only, one folder per setting (README there)
  ML_dscribe/CoGe/                         labels/, <tag>/<vector tag>/, rcut_scan_all/, rcut_scan_8x/, dim-reduction/
structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/   read-only source
  Bulk/symNNNNN/POSCAR,  Interface/symNNNNN/<4..12>x/POSCAR,  Interface/Information.txt
src/        code only, one README per folder; drivers never live in calculations/
results/    final tables and records only; every folder explained in results/README.md
plots/      figures
notes/      campaign notes, <topic>_YYYYMMDD.md
archive/    superseded material at the same relative path (archive/README.md); nothing reads from it
```

- **Names:**
  - `symNNNNN` is zero-padded to 5 digits; `CoGe_<plane>` in the bulk tree, `CoGe_<plane>_GBs` in the catalogue, bare `100/110/111` under `CoGe_interface/`.
  - Branches: `non-soc` in calculations, `non_soc` in plots.
  - Descriptor tags: `rc6.0_n8_l6_s0.5`. Thickness folders: `4x`…`12x`.
- **Words:**
  - **plane** = 100/110/111.
  - **family** = a recipe set on the generator CLI (`CoGe_110`, later e.g. `CoGe_110_GB`), but one GB sym folder (its 4x–12x set) in the ML code.
  - In `Information.txt`, "Num of Syms" is the number of operations that each *create* that interface, not the symmetries of the structure.
- **Source-structure traps:**
  - Double-scaled (scale 4.6294498444 with lattice vectors already in Å): `structures/…/Bulk/symNNNNN/POSCAR` on all three planes and `calculations/CoGe_Bulk/CoGe_{110,111}/symNNNNN/POSCAR`. The 110/111 run copies use scale 1.0; the CoGe_100 calculation POSCARs use unit vectors.
  - The `relax/POSCAR` files in the catalogue are malformed (comment and scale on one line).
  - `CoGe_100_GBs/Interface/sym00016/9x/POSCAR` is empty.
  - Each GB cell holds two interfaces (x = 0 and Lx/2).
  - 111 sym00119 and sym00175 are plain bulk.

## 7. Verified generators: how inputs get made

Regenerating inputs in a session and re-checking them by eye is the expensive loop: a verified
*prompt* must be re-checked on every run, but verified *code* stays verified. Review effort must
grow with the number of scripts, not with the number of structures (13,868 GB cells, 1,541 bulk).

**Current state:** `src/vasp_inputs/make_vasp_inputs.py` v0.2.0 (git tag `vasp_inputs-v0.2.0`) holds
the recipes in the registry below, and `vasp_task.py` / `run_vasp_array.slurm` / `submit_vasp.py` run
its folders (README there). Any other (family, stage) goes through rule 2 before production.

1. **Production inputs come only from a verified script** in `src/vasp_inputs/`. This covers INCAR,
   KPOINTS, POSCAR copies, POTCAR links, and the Slurm scripts and validators that go with them.
   Never write or patch them by hand or with one-off code, not even for one structure.
2. **No script yet for a (family, stage)?**
   1. Prototype it.
   2. Show the user the exact files for 1–3 representative structures, with a diff against the
      nearest verified recipe.
   3. The user confirms, or a small test run passes validation.
   4. Freeze it into the script **in the same session** and add it to the registry below.

   Production waits until the script exists.
3. **What a frozen script has:**
   - CLI: `python3 -B src/vasp_inputs/make_vasp_inputs.py --family CoGe_110 --stage scf-soc
     (--sym symNNNNN … | --all) [--dry-run]`
   - a `VERSION` constant
   - recipes as data (one entry per family and stage), not scattered conditionals
   - the refusal rules of section 3
   - a final count of written / identical / refused folders
   - stamps line 1 of every INCAR: `# make_vasp_inputs.py vX.Y.Z (git <short hash>) family=<F> stage=<S>`
   - refuses production writes while `src/vasp_inputs/` has uncommitted changes
4. **Tests** (`python3 -B -m unittest discover -s src/vasp_inputs`) must pass before every
   production use:
   - (a) Golden snapshots: the script's own output for 1–3 fixed structures per (family, stage),
     compared byte for byte with `src/vasp_inputs/tests/golden/`.
   - (b) For recipes that reproduce finished bulk runs: the parsed tags equal those of the
     `calculations/CoGe_Bulk` INCARs, ignoring order, comments and the stamp (EMIN to 1e-5), and the
     KPOINTS are byte-equal. Tags, not bytes, because the live INCARs exist in up to 3 line orders per
     stage. A deliberate change for new runs (e.g. LASPH on) is a new version whose test records
     the expected tag difference.
   - (c) Refusal tests on a temporary tree.
   - (d) POSCAR checks: scale fix, empty file, Co-then-Ge order, atom count.
5. **Changing a recipe = new version:**
   - Bump `VERSION` and regenerate the goldens.
   - Show the user the golden diff and get it re-verified.
   - Commit and tag it `vasp_inputs-vX.Y.Z`.

   Generated files are never edited; they are regenerated.
6. **Do not re-review a verified recipe's output per structure.** Per-structure checks are
   validators (code). If the output looks wrong, stop and report, then fix and re-verify the script,
   never the files.
7. **First build** (2026-10-05: git, `.gitignore`, `tags()` and the INCAR-writer pattern are done; the
   dos/bands pieces get copied when those recipes are built):
   - Run `git init` with a `.gitignore` that excludes `calculations/ structures/ results/ archive/ plots/`,
     every `src/*/.venv` and large binaries.
   - Copy (do not import) from `archive/calculations/CoGe_Bulk/`:
     - from `bulk_110_111_scf_once.py`: the INCAR parser `tags()` (:165-174), the scf INCAR writer
       `incar()` (:224-251) and `corrected()` (:192-199)
     - from `bulk_dos_bands_once.py`: the HPKOT `POINTS`/`SEGMENTS` (:43-46) and the dos/bands
       `prepare()` (:304-363)
   - CLI style: `add_path_arguments` in `src/dft_plots/coge_bulk/paths.py`.
   - The first candidates are the bulk stages; their finished runs are the goldens.

**Registry of verified recipes** (update in the same session that verifies one):

| Family | Stage | Version | Verified on | Golden | Note |
|---|---|---|---|---|---|
| CoGe_100 | scf-nonsoc | 0.1.0 | 2026-10-05, test (b): tags equal the live sym00000/00311/00622 INCARs, KPOINTS byte-equal | the live runs | bulk anchor; base only, every write refused |
| CoGe_100_GB | convergence-nonsoc | 0.2.0 | 2026-10-06, user chose the mixing tags and approved the golden diff to v0.1.0; tests (a)–(d) | `src/vasp_inputs/tests/golden/` (sym00177, sym00174 × e500_k1x9x9, e350_k1x9x9, e500_k1x3x3) | v0.1.0 + AMIX 0.2, AMIX_MAG 0.8, AMIN 0.01; a call with any problem writes nothing. v0.1.0 (2026-10-05) sloshed; its folders are in `archive/` |

Runs made before the generator carry no stamp. Their provenance is `result.json` / `validation.json`
in each stage folder and `archive/reorganization_20261002/stages.tsv`.

## 8. Code conventions

- Python with argparse (`description=__doc__`, `RawDescriptionHelpFormatter`), `--dry-run` /
  `--verify-only`, and `PROJECT_ROOT = Path(__file__).resolve().parents[N]`. Paths are project-relative.
- Tests use `unittest` on synthetic data, sit next to the code (`test_*.py`), and run with
  `python3 -B -m unittest discover -s <dir> -p 'test_*.py'`.
- No `__init__.py` under `src/dscribe/` (it would shadow the `dscribe` package). Nothing imports
  from `archive/`.
- Every `src/` folder has a README; update it in the same change.

## 9. Where to look

- `calculations/CoGe_Bulk/README.md`: bulk layout, stage files, POTCAR links, status.
- `notes/scf_reproducibility.md`, `notes/scf_110_111_20260914.md`, `notes/dos_bands_reproducibility.md`:
  the full VASP protocols and resource history.
- `src/dscribe/CoGe/README.md`, `src/ML_dscribe_scripts/README.md`: Stage 1 and 2 code, options, results.
- `src/dft_plots/README.md`: plotting.
- `results/README.md`, `archive/README.md`: what each record or archived item is.
