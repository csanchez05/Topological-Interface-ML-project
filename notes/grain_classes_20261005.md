# What the eight grain classes are, and whether they hold up (2026-10-05)

Question (user, 2026-10-05): "Did you reduce the entire grain boundary structure list to just 8
buckets? ... We are training / benchmarking the dscribe structures based on these grain classes so
it must hold up its truth."

Short answer: **the labels are correct, but they are not 8 buckets of GBs.** Every GB stays its own
sample (1,112 distinct GBs). The grain class is one property of each GB: how grain B is turned
relative to grain A. The number 8 is fixed by how the catalogue was generated, not by any reduction
in the ML code. Stage map: the labels are made in Stage 1.4 (`gb_labels.py`). They are the answer key
of the Stage 1.4 KRR test and of the selection benchmark in `results/dscribe/descriptor_study.md`.

## 1. What a catalogue GB is

- **Grain A** is the perfect B20 crystal.
- **Grain B** is the same crystal after one operation: a turn R (a rotation, possibly combined with
  a mirror or inversion), then a shift t.
- Each GB cell holds 2 interfaces (at x = 0 and Lx/2). At 8x, each grain is 8 bulk cells thick.

The generator's headers (`structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/Interface/Information.txt`):

| Plane | Applied operations | Discarded | Kept (= turns × 54 shifts) | Listed operations ("simple interfaces") | Unique interfaces |
|---|---|---|---|---|---|
| 100 | 3,456 | 864 | 2,592 = 48 × 54 | 2,580 | 623 |
| 110 | 3,456 | 2,592 | 864 = 16 × 54 | 860 | 543 |
| 111 | 3,456 | 3,024 | 432 = 8 × 54 | 430 | 375 |

- The listed operations use exactly 48, 16 and 8 different turns on 100, 110 and 111, and 54
  different shifts in all.
- These are the turns that fit grain B back into the plane's periodic box: all 48 cube operations
  for the cubic 100 cell, 16 for the √2×√2×1 110 cell and 8 for the √3×√2×√6 111 cell.
- So the catalogue never contains a general misorientation. Grain B is always turned by one of
  these box-keeping turns.

## 2. How the classes follow from that

- B20 CoGe (P2₁3) has only 12 rotations that leave the crystal unchanged (point group 23: 2-fold
  axes along the cube axes and 3-fold axes along the body diagonals). There is no 4-fold axis, no
  mirror and no inversion. spglib confirms this.
- Two turns that differ by one of these 12 build the same grain B, up to a shift. So the turns fall
  into groups of equivalent turns. Each group is one **grain class**.
  - Formally, the class is the right coset R·T, where T is that group of 12 rotations.
  - It is named after its member with the smallest rotation angle.
- Count: 48/12 = 4 on 100. The 16 turns of the 110 box share 4 rotations with T, so 16/4 = 4. The
  8 turns of the 111 box share only the identity with T, so 8/1 = 8.

| Class | Grain B is… | 100 | 110 | 111 | Distinct GBs |
|---|---|---|---|---|---|
| E | the same crystal, only shifted: no turn at all (a translation boundary) | 155 | 135 | 46 | 247 |
| I | the mirror-image crystal: the chirality flips across the GB | 156 | 136 | 47 | 265 |
| R90[100] | turned 90° about a cube axis, same chirality | 156 | 136 | 47 | 246 |
| I.R90[100] | turned 90° and mirrored | 156 | 136 | 47 | 238 |
| R60[111] | Σ3 twin: 180° about [111], whose smallest equivalent is 60° about [111] | – | – | 47 | 32 |
| I.R60[111] | that twin with flipped chirality | – | – | 47 | 26 |
| R70.53[1-10] | Σ3 twin: 180° about [11-2], whose smallest equivalent is 70.53° about [1-10] | – | – | 47 | 29 |
| I.R70.53[1-10] | that twin with flipped chirality | – | – | 47 | 29 |
| total | | 623 | 543 | 375 | 1,112 |

- **Why so few, and why they exist at all.** In a crystal with full cubic symmetry, all four 100/110
  classes would be one class, i.e. every 100/110 GB here would be a plain shift boundary. These
  classes exist only because B20 has lower symmetry: the four are the chirality × 90°-orientation
  domain states of B20.
- **Why four classes are 111-only.** A 180° turn about [111] or [11-2] is not a cube operation, so
  the twin classes fit only the 111 box. In a fully cubic crystal they would be one Σ3 class; B20
  splits them.
- **Swapping the grains does not change the class.** R and R⁻¹ always give the same class. On 111,
  the three rotations that keep [111] also leave every class unchanged. So the class does not
  depend on the choice of axes. On 100/110 this holds because T is normal in the cube group.

## 3. Evidence that the labels are true

| Check | Method | Result |
|---|---|---|
| Generator records | `gb_labels.py`: every operation listed for an interface rebuilds that interface's second-grain cell `Bulk/symNNNNN` | 623/623, 543/543, 375/375; every interface's operations fall in one class (the code stops otherwise) |
| Atoms only (this session) | Atoms in the middle of each grain of every 8x POSCAR (fractional x 0.125–0.375 and 0.625–0.875), matched against 96 candidate turns of the perfect crystal, tolerance 0.05 Å; no `Information.txt` | grain A = E in 1,541/1,541; grain B = stored label in 1,541/1,541; no family fails to match |
| Separate audit | `src/ML_dscribe_scripts/verify_grain_classes.py`, Slurm job 12823 (qcg3-0, 2026-10-05) → `results/dscribe/grain_class_verification.json` | passed: P2₁3 / 12 rotations; exactly 48/16/8 turns; the atom matching reproduces the labels; any two classes differ by ≥ 0.72 Å in atom positions under any shift |
| Duplicates | the 282 "same cell built twice" groups (`distinct_gbs.csv`) | each group carries one label |

- The atom-only check is in `archive/grain_class_20261005/structure_check/` (script,
  `grain_class_check.csv`, `.json`). It ran on a compute node in 445 s, using 115 MB.
- The separate audit, `verify_grain_classes.py` and the matching README edits (17:03–17:05 on
  2026-10-05) were **not** written by the Claude session that wrote this note. No `before_edit`
  backup of them is in `archive/`.

## 4. Why "8 buckets" is the wrong reading

- **The class ignores the shift.** Inside one class, GBs differ by how grain B is shifted against
  grain A, and by the plane.
  - On 100 each class has 39 shifts; on 110, 34. On 111, each class is a single turn combined with
    46–47 shifts (`gb_operations.csv`, first listed operation).
  - Example: `100/sym00001` and `100/sym00002` are both E, yet their closest Co–Ge distances are
    2.397 Å and 1.910 Å. They are different structures.
- **The ML code never merges GBs by class.** Stage 1.5 found 1,112 distinct GBs from the
  descriptors. The class enters only as the answer key in classification tests.
- **The counts are even only because of the construction.** Every turn was combined with the same
  set of shifts, which gives 155/156, 135/136 and 46/47.
- **Eight is not a law of CoGe.** A 36.87° turn about [001] (Σ5, cos 4/5) fits a 5×5×1 box but is
  in none of the eight classes. The catalogue simply did not build such GBs.

## 5. What this means for training and benchmarking

The labels are valid ground truth for one narrow question: which domain state grain B is in. They
ignore the shift, which GB energy (Stage 2) and spin texture (Stage 3) will also depend on. A good
grain-class score therefore says little about those targets, and a poor one does not rule them out.

Caveats:
- **E means no turn at all.** `111/sym00119` and `111/sym00175` are perfect bulk with no boundary,
  but they are labelled E. Their shift is a lattice translation of the cubic cell. Together they are
  one distinct GB: `distinct_id` D0985 in `distinct_gbs.csv`.
- **The four twin classes exist only on 111.** The pooled 8-class score therefore mixes a 4-class
  question (100/110) with an 8-class one (111). Scores per plane are cleaner.
- **What the Stage 1.4 KRR learns.** Source: `krr_confusions.csv` in
  `calculations/ML_dscribe/CoGe/rc6.0_n8_l6_s0.5/interface_r8_both_t4/`. Grouping is `distinct`
  (copies of one GB stay in one fold); predictions are for all 1,541 families. A "mirror class" is
  a class whose grain B is the mirror image of the true grain B, through a mirror plane that
  contains the GB normal and keeps the box. Each class has 2 mirror classes.

  | Plane | Correct (chance) | Errors that are a mirror class (if errors were random) | Correct or a mirror class |
  |---|---|---|---|
  | 100 | 0.563 (0.25) | 95% (67%) | 0.979 |
  | 110 | 0.578 (0.25) | 71% (67%) | 0.877 |
  | 111 | 0.139 (0.125) | 85% (29%) | 0.872 |

  - **On 100** the model finds the turn but cannot tell grain B from its mirror image. 87% of
    errors are the class that differs only by inversion, which on 100 is one of the two mirror
    classes.
  - **On 110** the errors spread about as they would at random.
  - **On 111** the model picks a mirror class for 73% of GBs and the correct class for only 14%.
    These pairs include the 440 "near-miss" pairs of the Stage 1.5 census: different structures
    whose SOAP vectors are only 1e-4 to 1e-3 apart. With repeats held out, the nearest training GB
    is often the mirror partner, so the answer is wrong.
  - So the grain-class score largely measures whether SOAP can see a mirror flip of grain B across
    the interface, and standard SOAP is weak at that.
  - **Why mirror-related classes in particular look alike to SOAP is not established.** Deep inside
    either grain, every class looks the same to SOAP, so the information is only at the interface;
    but that is true of every pair of classes.
  - **Relevance for Stage 3:** the I-type boundaries are where the crystal's chirality, and with it
    the sign of the DMI, flips.
- **Handedness, correcting an earlier claim.** The write-up first said handedness at chance was
  "expected, because standard SOAP cannot see handedness". That reasoning was wrong.
  - Standard SOAP cannot see the handedness of a whole crystal. This label compares grain B with
    grain A, and it does not change if the whole cell is inverted.
  - The result is real: 0.47–0.54 in the benchmark, and 0.55 against a 0.46 majority in the
    Stage 1 KRR.
  - Corrected on 2026-10-05 in `results/dscribe/descriptor_study.md` (section 4) and
    `plots/CoGe/dim-reduction/README.md`. The originals are in `archive/grain_class_20261005/before_edit/`.

## 6. Open points

Update 2026-10-06: the first two points are done (Job A 12895); results in
`results/grain_class_rep_study/`. The table in section 5 above comes from the per-plane models
of 2026-10-05 (with D0985).

- **Done: report grain class per plane, and drop D0985.** The pooled model scored per plane gives
  0.628 / 0.558 / 0.153; models trained on one plane give 0.563 / 0.578 / 0.153 (`distinct`, without
  D0985). See `results/grain_class_rep_study/02_grain_class_per_plane/`.
- **Done: free labels that depend on the shift** (`geometry_labels.py`: `d_min`, `n_close`,
  `cn_excess`, `cn_defect`). Plane and class explain ≤ 19% of their variance; KRR on the GB vector
  reaches R² 0.80–0.92. See `results/grain_class_rep_study/03_geometry_labels/`.
- **Still open: the 111 header count is unexplained.** `Information.txt` gives "Number of Crystal Symmetries: 2"
  on 111, while the audit finds only the identity shared between the 111 box turns and T. Possibly
  the generator counts a lattice translation (as in sym00119/sym00175), but this was not checked.
  The header values 12 and 4 on 100 and 110 match.
- **Correction to the chat answer of 2026-10-05.** It gave "46% mirror partner" on 111, counting one
  of the two mirror planes. With both it is 73%. It also explained the near-misses by SOAP's mirror
  invariance, which does not single out mirror pairs (see above).

## How to reproduce

- **Labels:** `src/ML_dscribe_scripts/gb_labels.py` → `calculations/ML_dscribe/CoGe/labels/`
  (`gb_labels.csv`, `gb_operations.csv`, `gb_labels.json`).
- **Audit:** see the `srun` command in `src/ML_dscribe_scripts/README.md` (section "Scope of the
  eight classes"). It needs a new `--out` path.
- **Atom-only check:**

  ```bash
  srun -p general --exclude=qcg-docker -c 1 --mem=2G -t 00:30:00 bash -lc \
    'cd ~/ML_Interface_Project && source src/mlenv/env.sh && \
     python -B archive/grain_class_20261005/structure_check/check_grain_classes.py <out_dir>'
  ```

  It runs from `archive/`, so it is a record, not live code.
- **Confusion shares:** computed from `krr_confusions.csv` with `gb_labels.crystal_rotations`,
  `to_cubic` and `grain_class` (mirror classes = `grain_class(m @ R)` for the det −1, trace 1
  operations among the plane's listed turns that keep the GB normal). This was a one-off on the
  login node; no script is kept.
