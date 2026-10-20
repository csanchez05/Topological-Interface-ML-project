# Handoff: covariant SOAP becomes the main line of work (2026-10-13)

**Status: not handed off.** The user asked the same session to carry it out ("you do it"). Done in
session `d98ca6ff`: layout split, `--study covariant` smoke (job 13004), full run job 13016 (qcg0);
record in `archive/covariant_primary_20261013/README.md`, results in `results/covariant_soap/`.

Written at the end of session `d98ca6ff`. Transcript, if a detail is missing:
`~/.claude/projects/-home-calvinsanchez-ML-Interface-Project/d98ca6ff-a7c6-46b8-9679-8c9bec3e4b1b.jsonl`.
All of this is SOAP guide **Stage 1** (steps 1.4, 1.5, r_cut scan) and **Stage 2** (GB selection).
Start in plan mode: turn sections 3–5 into a plan, show it to the user, then carry it out.

## 1. What the user wants (2026-10-13)

> "I want SOAP covariant to be the PRIMARY focus here. Finish doing the same calcs we did for
> standard SOAP using Covariant SOAP."
>
> "I would like for the plots in plots/CoGe/dim-reduction/SOAP to be split into covariant, and soap,
> and then alongside those folders you can have the comparisons between them [...] each should have
> plots of similar stuff."
>
> "Also do this seperation for the ML-scripts" · "I also want the graphs of these things" · "I want
> this [...] PROPERLY Organized"

Decisions the user made:

| Question | Decision |
|---|---|
| ML scripts: what gets split? | **Outputs split, code shared.** The scripts gain `--descriptor-type covariant`, the way `ace` was added. Outputs go into `standard/`, `covariant/` and `comparison/` trees that match the plot folders |
| Covariant r_cut scan | **8x first**: r_cut 4, 5, 6, 8, 10, 12 on the 8x cells, then the user decides on all thicknesses |
| Job B (fifth study phase 2, thickness check, sixth study) | **Yes**, as the standard-vs-covariant comparison |
| Grain-class overview (figure + table of every method against its majority baseline) | **Yes** |
| Graphs | Covariant versions of every standard figure, plus the comparisons |
| No-boundary control (`noboundary_control.py`) | **Parked.** Code and 8 tests are kept; don't run it |
| rc8 σ0.3 bulk-check failures | **Keep both runs; don't investigate further.** Add one status line to `results/grain_class_rep_study/07_settings_grid/README.md`: "kept (user, 2026-10-13)". The quick diagnostic is in `archive/representation_study_20261005/diagnostics/rc8_s03_bulk_check/` |

## 2. Where things stand

**Covariant SOAP already has:**
- descriptors for all 13,868 GB cells and the bulk cells, at the reference setting
  `rc6.0_n8_l6_s0.5`, λ 0–2, interface and whole cell (`calculations/descriptors/dscribe/covariant/`);
- a passing bulk check: the λ=1/2 cell averages vanish in bulk (≤ 1e-6);
- 12 settings-grid runs at 8x (r_cut 5/6/8 × l_max 4/8 × σ 0.3/0.5, `_t8`);
- the standard-vs-covariant comparison, phase 1 (PCA/UMAP and a nearest-neighbour score), in
  `calculations/ML_dscribe/CoGe/comparison/representation_study/phase1/`, written up in
  `results/grain_class_rep_study/06_representation_response/`.

**Headline so far:** nearest neighbours share the grain class far more often with covariant SOAP.
Raw kNN grain-class accuracy is C 0.77 / D 0.78, against standard A 0.26 / B 0.39 and a majority
baseline of 0.24.
- A, B: standard interface, standard whole cell;
- C, D: covariant interface, covariant whole cell.

**Never run with covariant SOAP:**
- 1.4 KRR (plane, grain class, per plane, free-label regression);
- 1.5 distinct GBs;
- the r_cut scan;
- the Stage 2 selection lists;
- the descriptor study (sensitivity, seeds, thickness);
- the selection benchmark.

`ml_common.DESCRIPTOR_TYPES` is `("soap", "ace")` today.

**Standard-SOAP follow-ups already done** (the user wants them kept):
- grain class scored per plane;
- D0985 (the two plain-bulk families) dropped;
- free geometry labels;
- `notes/bulk_as_training_data_20261006.md`.

## 3. Target layout

```text
plots/CoGe/dim-reduction/SOAP/
  README.md        index: one line per folder and figure, each phrased as a question
  standard/        everything in SOAP/ today that comes from standard SOAP alone: plane100_dmin1.8/,
                   descriptor_settings/, maps_by_setting/, seed_stability/, thickness/, selection/,
                   free_labels/krr_r2.png, plus FPS/ and median/ (missing from disk, user's choice;
                   leave them missing)
  covariant/       the same subfolder names, made from covariant SOAP, plus krr/ and rcut_scan_8x/
                   (today's SOAP/covariant/vector_sizes.png moves to comparison/)
  comparison/      representation_response/, settings_grid/ (Job B), grain_class_overview/ (new),
                   vector_sizes.png, standard-vs-covariant pick overlap; the label-only figures
                   (free_labels/by_grain_class.png, edge_vs_mid.png)
calculations/ML_dscribe/CoGe/
  labels/          shared, descriptor-free (ACE reads it too): stays where it is
  standard/        today's rc6.0_n8_l6_s0.5/, rcut_scan_8x/, rcut_scan_all/, descriptor_sensitivity/,
                   selection_benchmark/, selection_benchmark_v2/, dim-reduction/
  covariant/       the same names, from covariant SOAP
  comparison/      representation_study/ (phase 1 now, Job B next), settings_grid/, grain_class_overview/
results/
  covariant_soap/  new: numbered studies mirroring the standard ones, each README opening with a TL;DR
  grain_class_rep_study/  stays (comparison and follow-ups); gets the overview as 08_grain_class_overview/
```

**Rules for the move:**
- Copy the ACE session's Phase A migration: `archive/reorganization_20261013/migrate_dimred.py` has
  plan, apply and verify steps, `moves.tsv` and hashes. Write a new migration on that model into
  `archive/reorganization_<date>/`.
- `ml_common.CENSUS` points into `rc6.0_n8_l6_s0.5/interface_r8_both_t4/distinct_gbs.csv`, and the
  ACE scripts read it. Update `ml_common` paths (`OUTPUT_ROOTS`, `PLOT_FOLDERS`, `CENSUS`) in the
  same change, and keep `test_descriptor_type.py` passing.
- Update every link: `results/grain_class_rep_study/`, `results/dscribe/descriptor_study.md`, the
  plot and script READMEs, and memory. Propose CLAUDE.md §6 edits; don't make them.
- **Don't move anything while a job reads these folders.** On 2026-10-13 the ACE ML job 12971 was
  running on qcg0 and reads `labels/` and the census. Check `squeue` and ask the ACE session's owner
  (the user).

## 4. Work, in order

1. **Reorganise** the plots and ML outputs (section 3), with backups and a verify step. Agent work
   only.
2. **Add `--descriptor-type covariant`** to `ml_common`, `krr_classify.py`, `find_duplicates.py`,
   `pooled_duplicates.py`, `compare_duplicates.py`, `select_gbs.py`, `selection_benchmark.py`,
   `descriptor_sensitivity.py` (seeds and thickness; the settings knobs come from the 12 grid runs,
   because covariant runs for n_max, weighting and processing don't exist), `rcut_scan_all.py` (8x)
   and `plot_descriptor_study.py` (writes into `standard/`, `covariant/` or `comparison/`).
   - Design to settle in the plan:
     - **Rows:** one row per interface (edge and mid; the user's rule is that these are never
       averaged for covariant SOAP). Both rows of a GB, its 4x–12x versions and coinciding copies
       always stay in the same fold.
     - **Blocks:** λ=1 polar and axial plus λ=2 tensor and pseudotensor (= C of study 06). λ=0 is
       added only as a variant.
     - **Size:** about 59k features per row, about 6.5 GB in float32 over all thicknesses. That is
       compute-node only.
     - **1.5 tolerance:** set it from the covariant thickness noise, as ACE did (the SOAP 1e-4 does
       not carry over).
   - Tests: unit tests on synthetic data; the SOAP default path must stay bit-identical (rerun one
     standard output and compare). Smoke test on 2 families per plane.
3. **Compute.** One whole-node job, driver pattern from `rcut_scan_all.py` / `descriptor_grid.py`.
   - Covariant r_cut scan at 8x: new runs at r_cut 4, 5, 8, 10, 12 (l_max 6). r_cut 6 exists.
     About 40 CPU-h.
   - Covariant 1.4, 1.5, selection lists, benchmark and seeds/thickness: some tens of CPU-h
     (estimate; measure in the smoke test).
   - Job B: about 30–60 CPU-h plus sketch-map about 46 CPU-h, about 100 GB peak. Command:
     `sbatch src/ML_dscribe_scripts/representation_study.slurm --part B`.
     - Before submitting, rerun `test_representation_study.py`, `test_plot_descriptor_study.py`,
       `test_descriptor_study.py` and `test_ml_scripts.py`, because the ACE session edited the
       driver and the plot script on 2026-10-13.
     - Add the overview section to its plot task first.
   - Nodes: GB VASP stays on qcg3-0 (memory rule), so prefer qcg0. Job A used
     `--nodelist=qcg0 --cpus-per-task=64 --mem=400G`. Show the job (structures, node, cores,
     memory); the user submits it or says "submit it".
4. **Figures.** A covariant twin of every standard figure, the comparison figures, and the
   grain-class overview. Title each figure with its question, add a note on how to read it, and
   look at every PNG.
   - Sources for the overview, to be re-read from their files:
     - Stage 1.4 KRR: pooled 0.488 (majority 0.181); per plane 0.628 / 0.558 / 0.153 (pooled
       model) and 0.563 / 0.578 / 0.153 (per-plane models);
     - phase 1 kNN, raw / PCA / UMAP for A–D;
     - selection benchmark v2 at n = 30 / 100;
     - the new covariant KRR;
     - Job B's other methods.
5. **Write-ups:**
   - `results/covariant_soap/` (new) and `08_grain_class_overview/`;
   - the READMEs of every touched folder (`src/ML_dscribe_scripts/README.md` has uncommitted edits
     by another session, so re-read it before editing);
   - the `results/README.md` row and memory;
   - proposed CLAUDE.md §5/§6 lines: covariant is now the main line, 1,111 distinct GBs without
     D0985, the new folders.

## 5. Time estimate (rough)

| Part | Agent work | Compute |
|---|---|---|
| Plan + reorganisation | 2–3 h | none |
| `--descriptor-type covariant` + tests + smoke | 4–6 h | small `srun` tests |
| Full covariant runs + Job B | under 1 h to set up | about 150–250 CPU-h: 3–5 h on 64 cores once a big node is free |
| Figures + write-ups | 3–4 h | plots in the same job |

About 10–15 h of agent work over 3–5 chats. Calendar time depends on the queue (both big nodes were
full on 2026-10-13) and on the user's OKs between steps: about 2–3 days if nodes are free, about a
week if not.

## 6. Rules that bit this work

- **Back up before editing:** `archive/representation_study_20261005/before_edit/` plus a
  `MANIFEST.tsv` row (a new archive folder for the reorganisation).
- **Compute nodes:**
  - the scratchpad is not visible on compute nodes, so diagnostics go under `archive/.../smoke/` or
    `diagnostics/`;
  - run with `src/mlenv/.venv/bin/python`, not plain `python3` (no numpy);
  - featomic needs a spawn Pool;
  - set `OMP_NUM_THREADS=1` and make the cwd a temp folder.
- **Login node:** never load descriptor arrays there. Loading the covariant npz OOM-killed it once.
- **Figures:** any figure not computed from the real structures must say so in its title and stays
  out of `results/` and `plots/CoGe/`.
