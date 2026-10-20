# [100] GB static VASP: ENCUT / k-mesh convergence (2026-10-05)

VASP stage `convergence-nonsoc` (scheme A) of family `CoGe_100_GB`. It feeds SOAP guide Stage 2 (GB
energies, here from **unrelaxed** cells) and Stage 3 (SOC spin textures), via the production
scf / dos / bands runs that follow.

## Scope

- The 66 [100] 8x cells in `results/dscribe/total_sample_list.csv` (the `CoGe_100_GBs` entries of the
  162-GB Stage 2 list). They share one box: 74.07 × 4.63 × 4.63 Å, 64 Co + 64 Ge, 832 electrons,
  16 bulk cells long, two interfaces per cell.
- Everything static: no relaxation (user decision).
- Closest pairs (`dmin`): 61 of 66 have a pair < 2.0 Å, 33 < 1.5 Å; the bulk Co–Ge bond is 2.40 Å.
  The first plan ran them as-is. Since 2026-10-06 the sweep uses clean cells, and how production
  treats the overlapping cells is decided separately (see the run log).

## Decisions (user)

| Item | Decision |
| --- | --- |
| Sweep structures | Since 2026-10-06 (overlap limit): sym00190 (dmin 1.80 Å), sym00174 (2.02 Å), sym00016 (2.23 Å), sym00177 (2.40 Å). Earlier sets had sym00431 (0.77 Å), sym00144 (1.14 Å) and sym00485 (1.49 Å) |
| Overlap limit | 2026-10-06: cells whose closest pair is < 1.8 Å are not run (13 of the 66 [100] cells remain) |
| Sweep design | 1-D, not a grid: ENCUT 350/400/450/500/600 eV at Γ 1×9×9; Γ 1×N×N, N = 3/5/7/9/11, at 500 eV. The shared point e500_k1x9x9 runs once: 9 runs per structure, 36 in all, non-SOC |
| Smearing | ISMEAR 1 (Methfessel–Paxton), SIGMA 0.05 eV, for scf, dos and bands; no −5 probe |
| EDIFF | 1E-6 for the sweep, 1E-8 for production |
| LASPH | off, as in bulk |
| kx | 1: the 74 Å axis needs no more k-points. The in-plane axes are the bulk 4.63 Å, so the in-plane mesh stays near the bulk density |
| ALGO | Normal, as in bulk (the user turned down ALGO Fast, 2026-10-06) |
| Mixing | Since v0.2.0 (2026-10-06): AMIN 0.01, AMIX 0.2, AMIX_MAG 0.8; BMIX/BMIX_MAG default 1.0 |

The user first asked for 8 × 8 points on every structure. That was estimated at 150–200k core-h, so
it was reduced to this design (about 2,000 core-h).

## Inputs

They are made only by `src/vasp_inputs/make_vasp_inputs.py`, v0.2.0 since 2026-10-06 (registry in
CLAUDE.md section 7). The recipe is the finished bulk CoGe_100 non-soc/scf INCAR with these changes:

| Tag | Bulk | GB sweep | Why |
| --- | --- | --- | --- |
| ENCUT | 600 | per point | swept |
| EDIFF | 1E-8 | 1E-6 | decision; energy differences of interest are ≥ 0.1 meV/atom = 13 meV/cell |
| ISMEAR | −1 | 1 | decision |
| LCHARG | .TRUE. | .FALSE. | sweep runs seed nothing |
| ICHARG | 1 | 2 | no CHGCAR to read; atomic start |
| KPAR / NCORE | 3 / 1 | 4 / 4 (KPAR 1 at 1×3×3) | 16 ranks; 1×3×3 may have fewer than 4 irreducible k-points |
| NBANDS | default | 512 | the same band count at every point (VASP's default is about 500) |
| LORBIT | 11 | dropped | energies only; avoids about 1 GB of PROCAR/vasprun per run |
| AMIX, AMIX_MAG, AMIN | default 0.4, 1.6, 0.1 | 0.2, 0.8, 0.01 (v0.2.0) | default mixing sloshed along the 74 Å axis (run log, 2026-10-06 09:45) |

Unchanged from bulk: PREC Accurate, ALGO Normal, NELM 100, SIGMA 0.05, ISPIN 2, MAGMOM
`64*1.0 64*0.0`, ISYM 2, LREAL .FALSE., LMAXMIX 4, LWAVE .FALSE., BANDGAP KPOINT, the same
POTCAR (Co 02Aug2007, Ge 05Jan2001).

## Running

- `run_vasp_array.slurm`: 16 ranks, `--hint=nomultithread`, environment block verbatim.
- `vasp_task.py` runs each folder in node-local scratch and validates it. Only a valid run is copied
  back, with `result.json` written last.
- Records are in `results/CoGe_interface/100/convergence_8x/`.
- Cost estimate (first plan; the run log has the measured times): about 500 core-h per structure, 2–7 h wall per run, uncertain by ×2–3. It is scaled
  from the 48-atom [111] non-SOC scf (6.1 h on 6 ranks).
- Two test runs come first: e500_k1x9x9 on sym00431 (worst overlap) and on sym00177. Their measured
  peak RSS sets `--mem` for the other 34.

## Limits

- The references are 600 eV and N = 11, so the sweep shows convergence *towards* them, not whether
  they are themselves converged.
- The ENCUT sweep runs at 1×9×9 and the k sweep at 500 eV. A cross term between the two is not
  measured.
- Unrelaxed cells with atoms as close as 0.77 Å have large forces and may need more SCF steps. They
  may also converge differently from relaxed GBs.
- Bulk ε for E_GB: the finished bulk runs used ISMEAR −1 and 600 eV. A matching bulk reference
  (ISMEAR 1, the chosen ENCUT and k-density) is still needed.

## Next

1. After the 36 runs: `src/dft_plots/coge_interface/convergence_report.py` writes `convergence.tsv`
   and figures in `plots/CoGe/interface/convergence/`. Proposed rule: the smallest ENCUT and N at
   which all 4 structures are within 1 meV/atom of the reference, with E_F within 10 meV. The user
   confirms.
2. Then the production recipes (scf-nonsoc, scf-soc, dos, bands) for the 66 cells, each verified per
   CLAUDE.md section 7 rule 2.

## Run log

- 2026-10-05 22:31: generator v0.1.0 (git cfd8ad9) wrote the 36 folders; a rerun reports 36 identical.
  Task list: `results/CoGe_interface/100/convergence_8x/tasks.txt`.
- 22:32: test array 12834 (`--mem 96G --nodelist qcg0`). Task 0, sym00431 e500_k1x9x9, started on
  qcg0 as job 12835. Task 1, sym00177, could not start: with `--hint=nomultithread`, 16 ranks take
  32 logical CPUs and qcg0 had 6 left.
- Measured at setup (sym00431, 1×9×9, 500 eV, 16 ranks):
  - NKPTS 41, NBANDS 512, NPLWV 699,840
  - VASP's estimate is 7.6 GB per rank; the 16 ranks use 112 GiB (summed PSS, not double-counted)
  - so the 96 GB estimate was about 2× too low
- **Slurm does not enforce memory here** (`ConstrainRAMSpace=no`, the job cgroup has
  `memory.max=max`). `--mem` is a scheduling reservation only, so it must be set from measurement:
  an overrun is not killed, it overcommits the node.
- 22:40: task 1 cancelled while still pending (no claim, nothing ran) and resubmitted alone as array
  12840 with `--mem 160G` and no node pin.
- VASP advice in vasp.out: "One of the lattice vectors is very long (>50 A), but AMIN is rather
  large... charge sloshing... try AMIN 0.01." This is not applied in v0.1.0. It is the first
  candidate fix if the SCF converges slowly.
- 2026-10-06 00:52, sym00431 after 3 SCF steps:
  - setup took 7 min; the steps took 2,810 / 2,571 / 2,308 s (about 43 min each)
  - these are the NELMDL non-self-consistent steps; dE went 7,283 → −6,922 → −929 eV
  - summed RSS is steady at 123.7 GB
  - Projection: 30–60 steps × 30–43 min ≈ 15–43 h per run, so the 36 runs need about 9k–25k
    core-h, not the planned ~2k.
  - Only one such run (16 cores, ≥ 160 GB reserved) fits on the cluster at a time today, so the
    sweep would take weeks.
  - NELM 100 at ~35 min per step exceeds the 2-day `--time`. A non-converging run would be killed
    by Slurm before `finish` runs, leaving no failure record (claim and scratch stay).
- 2026-10-06 ~01:10:
  - A v0.2.0 with ALGO Fast and AMIN 0.01 was proposed. The user asked why not keep ALGO Normal, and
    is fine with long runs. **The recipe stays v0.1.0 (ALGO Normal, as in bulk); nothing was
    changed.**
  - AMIN 0.01 stays in reserve, used only if sym00431 shows charge sloshing.
  - The queued sym00177 e500_k1x9x9 test (12840) was cancelled before it started.
  - The sym00177 test runs at e500_k1x3x3 instead (array 12886, `--mem 48G`): it is the only point
    that fits the ~50 GB still reservable on qcg3-0, so it can start now.
- 2026-10-06 01:25, the queue situation:
  - Planned whole-node jobs of another user (`--mem=0`, 20 h each): one on qcg3-0 from about today
    20:45, one on qcg0 at 10-07 22:32. The qcg0 one waits for sym00431's 2-day limit to end.
  - A 2-day job therefore cannot backfill. The 1×3×3 test was resubmitted with `--time 18:00:00`
    (array 12889) and started at once on qcg3-0.
  - Lesson for the remaining 34: set `--time` per point from the measured step time, so short
    points can backfill.
  - sym00431 (v0.1.0, ALGO Normal) must converge within its 2-day limit, roughly ≤ 65 steps at
    ~43 min each.
- 2026-10-06 01:36, sym00177 e500_k1x3x3 (12889, qcg3-0):
  - NKPTS 5, KPAR 1 (16 cores per k-point), NCORE 4
  - VASP's estimate is 0.34 GB for rank 0; the 16 ranks use 5 GiB (summed PSS), against 112 GiB at
    1×9×9
  - Memory grows steeply with the k-mesh, so `--mem` must be set per point from measurement.
- 2026-10-06 09:45, **sym00177 e500_k1x3x3 failed validation** ("electronic loop did not reach
  EDIFF"). VASP exited 0 after all NELM = 100 steps (8 h 20 min, 296 s per step). The stage folder
  is untouched; claim and scratch are kept (qcg3-0 `/tmp/coge-gb-sym00177-e500_k1x3x3-12889_0-9ejy6oy_`).
  - E oscillates between −753 and −775 eV from step 6 to step 100.
  - rms(c) grows from ~2 to 4–5.
  - The final magnetization is 58.1 μB (unconverged; bulk collapses to 0).
  - Mixing was the VASP default: IMIX 4 (Broyden), AMIX 0.4, BMIX 1.0, AMIX_MAG 1.6, BMIX_MAG 1.0,
    AMIN 0.10.
  - Diagnosis: charge (and moment) sloshing along the 74 Å axis, as VASP's AMIN advice warned. This
    cell has no close pair, so the overlaps are not the cause.
- 2026-10-06 11:56, **sym00431 e500_k1x9x9 is diverging**: after 14 steps (13.4 h, ~57 min per
  step), E jumps between −1,600 and −37,800 eV, where about −760 eV is expected.
  - vasp.out carries VASP's WARNING "The distance between some ions is very small … I HOPE YOU KNOW
    WHAT YOU ARE DOING!" (the 0.77 Å pair).
  - It cannot reach 100 steps within its 2-day limit.
- 2026-10-06 13:27: sym00431 e500_k1x9x9 (job 12835) **cancelled** by user decision after 14
  diverging steps. No failed record was written (the job was killed before `finish`); the claim and
  the qcg0 scratch `/tmp/coge-gb-sym00431-e500_k1x9x9-12834_0-lite6qzc` remain.
- User decisions (2026-10-06):
  - Test damped mixing as v0.2.0 = v0.1.0 + AMIN 0.01, AMIX 0.2, AMIX_MAG 0.8. ALGO Normal, ISMEAR 1
    and SIGMA 0.05 unchanged. One test first: sym00177 e500_k1x3x3, the run that failed.
  - Clean cells only: replace sym00431 and sym00144 with two cells whose closest pair is ≥ 2.0 Å; keep
    sym00485 and sym00177. The treatment of the overlapping production cells (delete one atom per
    close pair, relax, or leave out) is decided separately.
- ~14:00: the 36 v0.1.0 folders (inputs only) and their records (`tasks.txt`, `submissions/`,
  `logs/`, `claims/`, `failed/`) moved to `archive/` at the same relative paths, after a read-only check
  (36 folders, INCAR/KPOINTS/POSCAR/POTCAR only, all stamped v0.1.0 cfd8ad9). List:
  `archive/interface_100_convergence_20261006/moves.tsv`.
- Clean-cell choice: `archive/interface_100_convergence_20261006/selection/overlap_scan.py` (closest
  pair, and atoms whose count of unlike neighbours within 2.75 Å differs from bulk's 7) on the 66 cells;
  output `overlap_scan.tsv` there. Only 5 cells have dmin ≥ 2.0 Å:

  | sym | dmin (Å) | pair | atoms off bulk coordination | net coordination change |
  | --- | --- | --- | --- | --- |
  | sym00174 | 2.015 | Co–Co | 4 | −4 |
  | sym00035 | 2.092 | Co–Ge | 8 | −12 |
  | sym00016 | 2.230 | Co–Ge | 8 | −8 |
  | sym00015 | 2.315 | Ge–Ge | 16 | −16 |
  | sym00177 | 2.396 | Co–Co | 16 | −20 |

  Rule: with sym00177 (largest dmin) kept, take the smallest-dmin clean cell (sym00174) and the median
  one (sym00016). The three clean cells then span the clean range in dmin (2.02–2.40 Å) and in lost
  coordination (−4, −8, −20). sym00485 (1.49 Å, 8 atoms off, −6) stays as the overlapping case.
  The clean cells are 5 of 66: whether the sweep's answer carries over to the overlapping cells
  depends on how those are treated.
- Generator v0.2.0: the mixing tags; a call with any problem now writes nothing (v0.1.0 wrote the
  good folders and then stopped); `vasp_task.py` checks that VASP echoes the INCAR's AMIX, AMIX_MAG
  and AMIN and records the mixing in `result.json`. Goldens: sym00177 and sym00174 (sym00431 removed).
  37 tests pass. Originals in `archive/interface_100_convergence_20261006/before_edit/`.
- 2026-10-06 ~17:05: v0.2.0 committed and tagged (98d4495, `vasp_inputs-v0.2.0`) after the user
  approved the golden diff. The 36 folders were written (a rerun reports 36 identical); task list
  `results/CoGe_interface/100/convergence_8x/tasks.txt`.
- 17:07: test array 12907 (sym00177 e500_k1x3x3, `--mem 16G --time 09:00:00`) could not start: qcg0
  had been taken at 16:21 by a 20 h whole-node job (until 10-07 12:21), and qcg3-0 is booked for
  another from 20:45. Cancelled while pending (nothing ran, no claim). Resubmitted as array 12908
  with `--time 03:30:00`, which backfills on qcg3-0 at once. At the 296 s per step of v0.1.0 this
  covers about 42 SCF steps. If it needs more, Slurm kills it before `finish`: no failed record,
  the claim and scratch stay, and its OSZICAR still shows whether the sloshing is gone.
- 2026-10-06 17:16, the test is using the new mixing: VASP echoes AMIX 0.20, AMIX_MAG 0.80 and
  AMIN 0.01, and its "AMIN is rather large" advice is gone.
- 18:02, after 12 steps (about 4.5 min each), the damped run converges smoothly, where v0.1.0 kept
  swinging between −753 and −775 eV:

  | Step | E (eV) | dE (eV) | rms(c) |
  | --- | --- | --- | --- |
  | 9 | −757.99 | 6.4 | 3.08 |
  | 10 | −753.46 | 4.5 | 2.21 |
  | 11 | −752.18 | 1.3 | 1.85 |
  | 12 | −751.29 | 0.89 | 1.56 |

- 20:39:31: **12908 killed by Slurm at its 3.5 h limit** (an estimated ~45 steps), before `finish`: no record.
  Its claim was moved to `killed/sym00177_8x_e500_k1x3x3_12908_0.json`, and the stage folder holds
  inputs only. Steps 13–45 are in the qcg3-0 scratch `/tmp/coge-gb-sym00177-e500_k1x3x3-12908_0-n6ti2ptn`.
  It can't be read now: another user's whole-node job (12826, `--mem=0`) took qcg3-0 at 20:39:34, and
  ssh to the node is refused.
- The scheduler is `priority/basic` (submission order) with backfill. As of 20:43, 8 whole-node jobs of
  one other user are running or queued: qcg0 has 12825 until 10-07 12:21, then 12898 and 12900–12904;
  qcg3-0 has 12826 until 10-07 16:39, then 12827 until about 10-08 12:39. Any job submitted now waits
  behind them unless it fits a backfill gap.
- 20:43: the approved full-length test was resubmitted as array 12919 (`--mem 16G --time 09:00:00`,
  NELM 100 at ~4.5 min per step). It starts when qcg3-0 is next free.
- 2026-10-06 ~21:00, **user decision: leave out cells with extreme overlap; I set the limit.**
  - Survey: `src/dft_plots/coge_interface/overlap_survey.py` (job 12920 hit its 20 min limit on
    file-system latency, ~0.09 s per POSCAR; rerun with threaded reads in 7 min, qcg1-1).
    Output `results/CoGe_interface/overlap_survey.tsv`, all 13,869 GB POSCARs.
  - Bulk B20 shortest distances: Co–Ge 2.398, Co–Co 2.842, Ge–Ge 2.870 Å. A GB's closest pair is
    identical at every thickness (spread 0.000 Å in all 1,541 families).
  - Limit: **1.8 Å** (75% of the Co–Ge bond). No natural gap in the distribution; at 1.5 / 1.8 / 2.0 Å
    the [100] Stage 2 list keeps 33 / 13 / 5 of 66 cells, the [100] catalogue 307 / 114 / 31 of 623.
    [110] and [111] have far fewer clean cells (9 and 19 catalogue families at 1.8 Å).
  - The 13 [100] cells: sym00190 1.803, sym00317 1.842, sym00570 1.872, sym00571 1.890, sym00004
    1.894, sym00448 1.907, sym00008 1.960, sym00047 1.971, sym00174 2.015, sym00035 2.092, sym00016
    2.230, sym00015 2.315, sym00177 2.396 Å.
  - Sweep: sym00485 (1.49 Å) replaced by sym00190 (1.80 Å, the kept cell closest to the limit). Its 9
    v0.2.0 input folders and the old `tasks.txt` moved to `archive/overlap_limit_20261006/replaced/`
    (`moves.tsv` there); the 9 sym00190 folders written (27 others identical), new `tasks.txt`.
  - The queued test 12919 (sym00177 e500_k1x3x3) is unaffected.

- 2026-10-07 18:29: **12919 (v0.2.0, sym00177 e500_k1x3x3) failed**: NELM 100 reached after 5.9 h.
  The charge settled, but the moment swung 64 → 68 → ~7 μB and rms(c) never fell below 1.3.
- 2026-10-13: SCF trials (array 12960, qcg3-0): 4 mixing routes × sym00177, sym00190 at
  e500_k1x3x3. Records in `results/CoGe_interface/100/convergence_8x/scf_trials_20261013/`. GB VASP
  runs stay on qcg3-0 only (user).
- 2026-10-15: **user decision: zero moment for now (MAGMOM = 0)**. Implemented as generator v0.3.0
  (ISPIN 1; MAGMOM and AMIX_MAG dropped), which waits for the user's check of the golden diff. The
  trials found a moment-carrying state 0.2–0.3 meV/atom below the zero-moment state in both cells.
  The `--mem 90G` proposed on 10-13 would have overcommitted qcg3-0 at the heavy points; a memory
  pilot comes first. Plan, input reasoning and commands: `notes/gb_submission_plan_20261015.md`.
- 2026-10-15 (later): **user asked for a moment check**: new stage `moment-check` (generator
  v0.3.0, not yet committed). sym00177 and sym00190 at e500_k1x9x9, ISPIN 2 with the trial `linmag`
  mixing (AMIX_MAG 0.8, BMIX_MAG 0.0001) and LORBIT 11, in `8x/moment-check/`. It waits for the
  user's check of its golden diff. The runner now also checks BMIX and BMIX_MAG against the
  OUTCAR. Details: plan note section 8.1.
