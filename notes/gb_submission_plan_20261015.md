# [100] GB VASP: the plan up to submission, and why each input is what it is (2026-10-15)

VASP stage `convergence-nonsoc` (scheme A), family `CoGe_100_GB`. It feeds SOAP guide **Stage 2**
(GB energies E_GB) and, through the production runs that follow, **Stage 3** (SOC spin textures).
The run log and older decisions are in `notes/interface_100_convergence_20261005.md`. This note is
the current plan.

## TL;DR

- **What runs first:** a convergence sweep of 36 static runs on 4 [100] GB cells. ENCUT goes from 350
  to 600 eV at Γ 1×9×9, and the k-mesh is Γ 1×N×N (N = 3, 5, 7, 9, 11) at 500 eV. The sweep picks the
  ENCUT and k-mesh for production on the 13 clean [100] cells.
- **New today: the inputs become non-magnetic (v0.3.0).** You chose a zero moment (MAGMOM = 0). I
  implemented it as ISPIN 1: same answer, about half the work per step. This exact SCF has already
  converged on the two hardest cells, in 31 and 35 steps. **It waits for your check of the input
  diff in section 5.2.**
- **How it runs:** on qcg3-0 only, 16 cores per run, up to 8 runs at once, heaviest first.
  - Two real sweep runs go first as a **memory pilot**. Eight heavy runs at once could exceed
    qcg3-0's memory, and Slurm here does not stop that. The `--mem 90G` I gave on 10-13 was wrong
    for the heavy points (section 7.2).
- **How long:** about 3 days for the sweep (2.5–4.5). About 2 days if the ENCUT points run at
  1×5×5 instead of 1×9×9 (your choice).
- **Something your student should see (section 4):** in both cells tested, the trials found a
  state with a small moment that is *lower* in energy than the zero-moment state, by 0.2–0.3
  meV/atom. Every route we tried reached that same state. The zero-moment plan is still fine for
  choosing ENCUT and k.
- **Moment check (added later on 10-15, your request):** sym00177 and sym00190 at e500_k1x9x9,
  spin-polarized with the trial `linmag` mixing and LORBIT 11, in their own `moment-check/` folders
  (section 8.1). About 1,600 core-h and 2–2.5 days, running beside the sweep.

## 1. What gets run

| Item | Value | Decided |
| --- | --- | --- |
| Cells | [100] 8x cells of the Stage 2 list (`results/dscribe/total_sample_list.csv`) whose closest atom pair is ≥ 1.8 Å: 13 of 66 | overlap limit, user 2026-10-06 |
| Geometry | static, as cut from the catalogue; no relaxation | user 2026-10-05 |
| Sweep cells | sym00190 (closest pair 1.80 Å), sym00174 (2.02), sym00016 (2.23), sym00177 (2.40). They span the clean range, from the cell nearest the limit to the bulk-like one | user 2026-10-06 |
| Cell | 74.07 × 4.63 × 4.63 Å, 64 Co + 64 Ge, 832 electrons, 16 bulk cells long, two interfaces (x = 0 and x = Lx/2) | catalogue |
| Sweep points | ENCUT 350/400/450/500/600 eV at Γ 1×9×9; Γ 1×N×N, N = 3/5/7/9/11, at 500 eV. The shared point e500_k1x9x9 runs once: 9 runs per cell, 36 in all | user 2026-10-05 |

Stage folders: `calculations/CoGe_interface/100/symNNNNN/8x/convergence/e<ENCUT>_k1x<N>x<N>/`.

## 2. How we got here

| Date | Version | What happened |
| --- | --- | --- |
| 10-05 | v0.1.0 | Bulk settings with VASP's default mixing. The 74 Å cell sloshed: sym00177 ran 100 steps without converging, E swinging −753 to −775 eV |
| 10-06 | v0.2.0 | Damped mixing (AMIX 0.2, AMIX_MAG 0.8, AMIN 0.01). The charge settled, but the test run 12919 (sym00177, 1×3×3) still failed: after 100 steps the moment had swung 64 → 68 → ~7 μB and rms(c) never fell below 1.3 |
| 10-13 | trials | 8 trial runs of 4 mixing routes on sym00177 and sym00190 (section 4) |
| 10-15 | v0.3.0 (pending your check) | Zero moment, your decision. ISPIN 1; MAGMOM and AMIX_MAG dropped |

## 3. What the sweep decides, and what it cannot

- **Rule (proposed 10-05; you confirm the values):** production uses the smallest ENCUT and N at
  which all 4 cells are within **1 meV/atom** of the reference and E_F is within **10 meV**. The
  reference points are 600 eV and N = 11. The energy compared is E0 (σ → 0).
- **Limits:**
  - The references (600 eV, N = 11) are not themselves shown to be converged. The sweep shows
    convergence *towards* them.
  - The ENCUT sweep runs at 1×9×9 and the k sweep at 500 eV, so a cross term is not measured.
  - 4 cells stand for 13.
  - The cells are unrelaxed.

## 4. The SCF trials (2026-10-13) and the moment question

Records: `results/CoGe_interface/100/convergence_8x/scf_trials_20261013/trials.tsv`. All trials ran
at `e500_k1x3x3`, 16 cores, NELM 200, ALGO Normal, ISMEAR 1 / SIGMA 0.05, on a full qcg3-0.

| Route | Change from v0.2.0 | sym00177 | sym00190 | Core-h, both |
| --- | --- | --- | --- | --- |
| `linmag` | BMIX_MAG 0.0001 | converged, 31 steps | converged, 42 steps | 269 |
| `wiki` | BMIX 0.0001 and BMIX_MAG 0.0001 | diverged, VASP crashed at step 65 | diverged, crashed at step 105 | — |
| `nmseed` | run 1: zero moment from atomic densities; run 2: magnetic from run 1's density | 31 + 79 steps | 35 + 144 steps | 509 |
| `nmseed_linmag` | as `nmseed`; run 2 with BMIX_MAG 0.0001 | 31 + 31 steps | 35 + 27 steps | 295 |

**What this means for v0.3.0:** run 1 of the seeded routes is the v0.3.0 SCF. It is non-magnetic,
starts from atomic densities and uses the same mixing. Only NELM and LCHARG differ, and neither
changes the SCF path. It converged in **31 steps (sym00177) and 35 steps (sym00190)**. These two
cells were the hardest cases so far. That counts as the small test run CLAUDE.md §7 asks for.

**What the magnetic runs found:**

| Cell | Zero moment, E0 (eV) | With moment, E0 (eV), 3 routes | Net moment | Difference |
| --- | --- | --- | --- | --- |
| sym00177 | −759.67006 | −759.69621 ± 0.00002 | 1.81 μB per cell | −26 meV per cell = −0.20 meV/atom |
| sym00190 | −749.58864 | −749.63216 ± 0.00001 | 1.57 μB per cell | −44 meV per cell = −0.34 meV/atom |

- **The state with a moment is well defined.** Three different routes reached the same energy to
  0.03 meV and the same moment to 0.005 μB. The sign of the moment in one sym00190 run is a global
  spin flip of the same state.
- **It lies below the zero-moment state in both cells.** A metastable state lies *above* the
  ground state. So of these two states, the zero-moment one is the higher one.
  - A different magnetic state lower still is not ruled out. Every magnetic run started from
    1 μB on each Co.
  - These numbers come from the coarsest mesh (1×3×3).
  - We don't know where the moment sits: the sweep INCAR does not ask VASP for per-atom moments.
- **Why the zero-moment sweep is still fine:**
  - The difference (0.2–0.3 meV/atom) is below the 1 meV/atom target of the sweep, so it should not
    change the ENCUT and k-mesh the sweep picks.
  - It matters later. For E_GB it is about 10–16 mJ/m² (two interfaces of 21.4 Å² each). For
    Stage 3, a moment at the interface changes the spin texture qualitatively.
- The question stays open, as CLAUDE.md §4 already lists ("magnetic ground state: not established").
  If it is ever needed, `linmag` (BMIX_MAG 0.0001) is the magnetic recipe that converged on both
  cells for the fewest core-hours.

## 5. Inputs (generator v0.3.0, pending your check)

Inputs come only from `src/vasp_inputs/make_vasp_inputs.py` (CLAUDE.md §7), never by hand. The
recipe is the finished bulk CoGe_100 scf INCAR plus a declared set of changes. A test checks that the
declared set is the whole difference.

### 5.1 INCAR (sym00177, e500_k1x9x9; other points differ only in ENCUT, and KPAR 1 at 1×3×3)

```text
# make_vasp_inputs.py v0.3.0 (git <hash>) family=CoGe_100_GB stage=convergence-nonsoc
SYSTEM = CoGe B20 [100] GB convergence
PREC = Accurate
ENCUT = 500
EDIFF = 1E-6
ALGO = Normal
NELM = 100
AMIX = 0.2
AMIN = 0.01
ISMEAR = 1
SIGMA = 0.05
IBRION = -1
NSW = 0
ISPIN = 1
LWAVE = .FALSE.
LCHARG = .FALSE.
BANDGAP = KPOINT
ISTART = 0
ICHARG = 2
ISYM = 2
KPAR = 4
NCORE = 4
NBANDS = 512
LREAL = .FALSE.
LMAXMIX = 4
```

| Tag | Value | Why | Source |
| --- | --- | --- | --- |
| PREC | Accurate | FFT grids large enough to avoid wrap-around (aliasing) errors. Needed when comparing energies at the meV level | bulk, 4,779 finished runs |
| ENCUT | 350–600 per point | The swept quantity. The POTCARs' ENMAX values are 268 eV (Co) and 174 eV (Ge), so 350 eV is 1.3× the larger, the usual minimum | sweep design |
| EDIFF | 1E-6 | The SCF stops when the energy changes by < 1e-6 eV per cell (8e-9 eV/atom). That is 10⁴× below the smallest difference of interest (0.1 meV/atom = 13 meV per cell). Production uses 1E-8 | user 10-05 |
| ALGO | Normal | Blocked Davidson: robust, as in bulk. ALGO Fast was turned down | user 10-06, 10-13 |
| NELM | 100 | Maximum SCF steps. The trials needed 31–35, so this leaves 3× headroom. A run that reaches 100 is marked failed rather than accepted | bulk |
| AMIX, AMIN | 0.2, 0.01 | Damped Kerker mixing (BMIX stays at VASP's 1.0, not written). The 74 Å axis allows long charge waves to slosh. For lattice vectors > 50 Å, VASP's own advice is AMIN 0.01. Default mixing sloshed in v0.1.0 | user 10-06 (v0.2.0); trial-tested 10-13 |
| ISMEAR, SIGMA | 1, 0.05 eV | Methfessel–Paxton smearing for metals; every bulk probe was metallic, so there is no ISMEAR −5 probe. Energies are compared as E0 (σ → 0) | user 10-05/06 |
| IBRION, NSW | −1, 0 | Static: no ionic steps | user 10-05 (no relaxation) |
| **ISPIN** | **1** | **v0.3.0.** Zero moment, as you asked. ISPIN 1 forces it exactly. ISPIN 2 with MAGMOM 0 would give the same answer but (a) costs about twice as much (measured: 6,700 vs 14,400 H·ψ products per step) and (b) could let rounding noise grow into the lower-energy moment at some points and not others, mixing two states in one sweep | user 10-15 (MAGMOM 0); ISPIN 1 is my implementation, for your check |
| LWAVE, LCHARG | .FALSE., .FALSE. | Sweep runs seed nothing; this saves about 100 MB per run. The production scf will write CHGCAR for dos, bands and SOC | design 10-05 |
| BANDGAP | KPOINT | Reporting only | bulk |
| ISTART, ICHARG | 0, 2 | Fresh start from atomic densities. Each point is independent, so there is no order constraint and no error is carried between points | design 10-05 |
| ISYM | 2 | Uses the cell's symmetry. Only inversion / time reversal remain here, so the mesh reduces to (N²+1)/2 k-points | bulk |
| KPAR, NCORE | 4, 4 (KPAR 1 at 1×3×3) | Parallel layout only; it does not change the answer beyond rounding. 16 ranks = 4 k-point groups × 4 ranks. 1×3×3 has only 5 k-points, so one group | design 10-05 |
| NBANDS | 512 | 832 electrons fill 416 bands, so 512 leaves 96 empty bands for smearing and Davidson. Fixed at every point, so every point uses the same band count. VASP's default (~480) changes with the parallel layout | design 10-05 |
| LREAL | .FALSE. | Projectors in reciprocal space: exact. VASP suggests real-space projection (LREAL Auto) for large cells because it is faster, but its error is around meV/atom, the size we are measuring | bulk |
| LMAXMIX | 4 | Mixes the d-channel PAW density (Co d). Kept as in bulk, so the production CHGCAR can be reused | bulk |
| *Dropped* | LORBIT, MAGMOM, AMIX_MAG | LORBIT: only energies are needed, and a PROCAR would be ~1 GB per run. MAGMOM and AMIX_MAG have no meaning with ISPIN 1 (v0.3.0) | 10-05; 10-15 |
| *Not set* | LASPH, dipole correction, ADDGRID | LASPH stays off as in bulk; this is an open question. A dipole correction does not apply: the cell is periodic with no vacuum | CLAUDE.md §4 |

### 5.2 The change from v0.2.0: please check this

This is the golden diff: the generator's output for the 6 fixed test cases (sym00177 and sym00174 ×
e500_k1x9x9, e350_k1x9x9, e500_k1x3x3). Every one of the 6 INCARs changes by exactly these lines, and
no KPOINTS changes:

```diff
-# make_vasp_inputs.py v0.2.0 (git GOLDEN) family=CoGe_100_GB stage=convergence-nonsoc
+# make_vasp_inputs.py v0.3.0 (git GOLDEN) family=CoGe_100_GB stage=convergence-nonsoc
-AMIX_MAG = 0.8
-ISPIN = 2
-MAGMOM = 64*1.0 64*0.0
+ISPIN = 1
```

All 39 generator and runner tests pass. The runner checks that VASP really ran with ISPIN 1, and a
test now covers that check.

### 5.3 KPOINTS

```text
CoGe [100] GB Gamma mesh
0
Gamma
1 9 9
0 0 0
```

| N | Irreducible k-points | In-plane spacing (Å⁻¹) |
| --- | --- | --- |
| 3 | 5 | 0.452 |
| 5 | 13 | 0.271 |
| 7 | 25 | 0.194 |
| 9 | 41 | 0.151 |
| 11 | 61 | 0.123 (= the bulk 11×11×11 density) |

- **kx = 1:** along the 74 Å axis one k-point already gives a spacing of 0.085 Å⁻¹, finer than the
  bulk mesh's 0.123 Å⁻¹. More k-points along x would add cost and no accuracy.
- **In plane (y, z):** these are the bulk 4.63 Å axes, so N = 11 matches the bulk density exactly.
- **Γ-centred, odd N:** the mesh contains Γ and is symmetric about it, as in bulk.
- **Cost:** the irreducible k-point count drives it, roughly ∝ N²/2, because these cells keep only
  inversion symmetry.

### 5.4 POSCAR and POTCAR

- **POSCAR:** a byte copy of `calculations/CoGe_interface/100/symNNNNN/8x/POSCAR`. The generator
  checks it against the catalogue: scale 1.0, Co then Ge, 64 + 64 atoms, Direct coordinates.
- **POTCAR:** a symlink to the campaign master, never a copy. It holds PAW_PBE Co 02Aug2007 (ZVAL 9)
  and Ge 05Jan2001 (ZVAL 4), so NELECT = 64·9 + 64·4 = 832. The runner checks both TITELs and
  NELECT in every OUTCAR.

## 6. How a run executes (no change)

- `submit_vasp.py` submits the pending folders as one Slurm array of `run_vasp_array.slurm`, with
  the environment block verbatim from CLAUDE.md §2.
- `vasp_task.py` stages each run into node-local scratch and runs VASP there, then validates it:
  - exit code and VASP 6.4.3
  - EDIFF reached in fewer than NELM steps
  - every INCAR tag echoed by VASP, the k-mesh, KPAR/NCORE and NBANDS
  - the POTCAR TITELs and NELECT
- **Valid runs:** only these are copied into the stage folder, with `result.json` written last
  (energies, E_F, steps, wall time, peak memory).
- **Failed runs:** the stage folder stays untouched, and the scratch and a failure record are kept.

## 7. Resources

### 7.1 Layout and time

- **Node and layout:** qcg3-0 only (user, 10-13), 16 ranks per run, so up to 8 runs fill its 128
  physical cores. Folders go heaviest first, ranked by k-points × ENCUT^1.5.
- **Speed is uneven when the node is full:**
  - With 8 runs sharing qcg3-0, each ran 2–3× slower per unit of work than the earlier run 12919. All ranks were on their own cores, so the 8 runs were competing for shared
    resources, most likely memory bandwidth.
  - Identical runs at the same time differed up to 2× in speed (3.6 h vs 7.5 h for the same
    sym00190 seed run).
- **Cost of the 36 runs at ~33 steps each:**
  - about **9,400 core-h** (7,500–13,400), or **about 3 days** (2.5–4.5) on qcg3-0
  - the longest single runs (1×11×11, and 600 eV at 1×9×9) take 1–2 days each
- **Faster option, only if you choose it:** run the four ENCUT points at 1×5×5 (13 k-points)
  instead of 1×9×9 (41).
  - The basis-set error hardly depends on the k-mesh, so the ENCUT answer should not change.
  - It cuts the cost by a third, to about 2 days.
- **Time limits per class:** 5 / 3 / 1 days (section 8). They allow 2× the normal estimate, so a
  slow run is not killed before it finishes. The script's default `--time` is 2 days, which a slow
  1×11×11 run could exceed.

### 7.2 Memory: why there is a pilot

- **Slurm here does not enforce `--mem`.** It is only a reservation (`ConstrainRAMSpace=no`). If the
  runs together use more than the node has, nothing stops them, and the node can kill all of them.
- **Measured so far:**

  | Run | Spin | k-points | Memory, all 16 ranks |
  | --- | --- | --- | --- |
  | sym00431 e500_k1x9x9 (10-05) | ISPIN 2 | 41 | 112 GiB (PSS) / 124 GB (RSS) |
  | trials e500_k1x3x3 | ISPIN 2 | 5 | 7.9–9.2 GB |
  | trials e500_k1x3x3, zero moment | ISPIN 1 | 5 | 6.3–6.7 GB |

- **Estimated for ISPIN 1:** the heavy points (1×11×11; 600 eV at 1×9×9) may need roughly
  60–130 GB each. Eight of them could ask for more than the ~716 GB free.
- **My error on 10-13:** `--mem 90G` for every run was chosen so that 8 runs fit on the node, not
  from these measurements. At the heavy points it would have overcommitted qcg3-0.
- **The fix:** two heavy runs start first as a pilot (they are real sweep points). I read their
  memory once they are running, and set `--mem` for each class of point from it.

## 8. Steps to submission

| # | Who | Step |
| --- | --- | --- |
| 0 | me | Done 10-15: v0.3.0 recipe, goldens and tests (39 pass). The trial tool is archived; it only worked with v0.2.0 |
| 1 | **you** | Check section 5: the diff in 5.2, and ISPIN 1 as the way to get MAGMOM = 0. Reply OK, or with changes |
| 2 | me | Commit `src/vasp_inputs/` and CLAUDE.md (new registry row; GB settings: ISPIN 1), then tag `vasp_inputs-v0.3.0` |
| 3 | me | Move the 36 v0.2.0 input folders and their records (`tasks.txt`, `submissions/`, `logs/`, `failed/`, `killed/`, `claims/`) to `archive/interface_100_convergence_20261013/v0.2.0/`, listed in `moves.tsv`, after a read-only check that they hold inputs only. Generate the 36 v0.3.0 folders: the dry run shows 36 to write, the write reports 36, and a rerun reports 36 identical. New `tasks.txt`. Also the 2 `moment-check` folders and their task list (section 8.1) |
| 4 | **you** | Memory pilot: the two heaviest points of sym00190 (command below), and the moment check (section 8.1) |
| 5 | me | 30–60 min after they start: read their memory and fill in `--mem` for classes A, B and C |
| 6 | **you** | Submit the other 34 as three arrays, heaviest class first; Slurm starts them in submission order (commands below) |
| 7 | me | Check the first finished `result.json` of each class (valid, steps, wall time, memory) and report anything odd |
| 8 | me | After all 36: `convergence.tsv` and figures, and a proposed production ENCUT and k-mesh. **You confirm** |

Commands (run from the project root; check `sinfo -n qcg3-0` first):

```bash
T=results/CoGe_interface/100/convergence_8x/tasks.txt

# step 4: pilot, 2 runs (real sweep points)
python3 -B src/vasp_inputs/submit_vasp.py --tasks $T \
    --only sym00190/8x/convergence/e500_k1x11x11 sym00190/8x/convergence/e600_k1x9x9 \
    --mem 160G --time 5-00:00:00

# step 6: the other 34 (running or finished folders are skipped automatically)
# A: 6 runs, 1x11x11 and 600 eV
python3 -B src/vasp_inputs/submit_vasp.py --tasks $T --only e500_k1x11x11 e600_k1x9x9 \
    --mem <A from pilot> --time 5-00:00:00
# B: 20 runs, 1x9x9 at 350-500 eV and 1x7x7
python3 -B src/vasp_inputs/submit_vasp.py --tasks $T \
    --only e350_k1x9x9 e400_k1x9x9 e450_k1x9x9 e500_k1x9x9 e500_k1x7x7 \
    --mem <B from pilot> --time 3-00:00:00
# C: 8 runs, 1x5x5 and 1x3x3
python3 -B src/vasp_inputs/submit_vasp.py --tasks $T --only e500_k1x5x5 e500_k1x3x3 \
    --mem <C from pilot> --time 1-00:00:00
```

Add `--dry-run` to any of them to see the folder list and the sbatch line without submitting.

### 8.1 Moment check (user, 2026-10-15)

- **Question:** is the small moment that the trials found at 1×3×3 still there at 1×9×9, on which
  atoms does it sit, and how far below the zero-moment state is it?
- **Stage `moment-check`** (same generator, v0.3.0): sym00177 and sym00190, 8x, one point,
  e500_k1x9x9, in `calculations/CoGe_interface/100/symNNNNN/8x/moment-check/e500_k1x9x9/`.
  Records: `results/CoGe_interface/100/moment_check_8x/`.
- **INCAR:** the sweep's e500_k1x9x9 INCAR with these changes (golden diff, CLAUDE.md §7 rule 2):

  ```diff
  -# make_vasp_inputs.py v0.3.0 (git GOLDEN) family=CoGe_100_GB stage=convergence-nonsoc
  +# make_vasp_inputs.py v0.3.0 (git GOLDEN) family=CoGe_100_GB stage=moment-check
   AMIX = 0.2
  +AMIX_MAG = 0.8
  +BMIX_MAG = 0.0001
   AMIN = 0.01
  -ISPIN = 1
  +ISPIN = 2
  +MAGMOM = 64*1.0 64*0.0
   LCHARG = .FALSE.
  +LORBIT = 11
  ```

  KPOINTS are byte-identical to the sweep point's. Compared with the INCAR of the `linmag` trials,
  the only differences are NELM 100 (the trials used 200 and needed 31 and 42 steps), KPAR 4 (more
  k-points) and LORBIT 11.
- **Cost:** at the trials' speed on a full node, about 690 core-h for sym00177 and 930 for
  sym00190, if they need as many steps as at 1×3×3 (31 and 42). The range is 1,400–2,800 core-h in
  all, and each run takes 1.5–4 days on 16 ranks. Two of the eight slots are busy for that
  time, so the sweep finishes about half a day later.
- **Memory:** 125 GB measured for ISPIN 2 at 41 k-points. `--mem 140G`.
- **Output:** LORBIT 11 adds a PROCAR and projections in vasprun.xml, roughly 1 GB per run.
- **When:** together with the pilot (step 4), once v0.3.0 is committed and the folders exist:

  ```bash
  python3 -B src/vasp_inputs/submit_vasp.py --tasks results/CoGe_interface/100/moment_check_8x/tasks.txt \
      --mem 140G --time 5-00:00:00
  ```

- **Report** (me, when they finish): the total moment, the per-atom moments from the OUTCAR
  `magnetization (x)` table (which atoms carry the moment, and how far from each interface), and
  E0 against the sweep's ISPIN 1 e500_k1x9x9 run of the same cell.

## 9. After the sweep

1. **Report:** `src/dft_plots/coge_interface/convergence_report.py` writes `convergence.tsv` and the
   figures in `plots/CoGe/interface/convergence/`, with ΔE in meV/atom against the references, and
   E_F and cost per point. You confirm the production ENCUT and k-mesh.
2. **Production on the 13 clean cells:** first `scf-nonsoc` at the chosen ENCUT and k-mesh, with
   EDIFF 1E-8 and LCHARG .TRUE. so that the CHGCAR is kept. Then `scf-soc`, `dos` and `bands`. Each
   is a new recipe, shown to you before production (CLAUDE.md §7 rule 2).
3. **Bulk reference for E_GB:** this needs bulk runs at the same settings (ISMEAR 1, the chosen
   ENCUT and k-density, non-magnetic). The finished bulk runs (ISMEAR −1, 600 eV) are not that
   reference.

## 10. Not decided yet

- **The moment:** zero for the sweep (section 4). The moment check (section 8.1) tests it on two cells.
- **LASPH:** off, as in bulk. Its being off in bulk was an oversight.
- **Production:** NBANDS for production, dos and bands; the dos mesh; the band path (the bulk path
  is invalid for this cell); MAGMOM and SAXIS for SOC.
- **Cells beyond the 13**, and the cells with a closest pair < 1.8 Å (not run).
- **Relaxation:** none so far, by decision.
