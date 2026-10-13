# Should the bulk B20 structures be training data? (2026-10-06)

Question (user, 2026-10-06): "do you think it would be a good idea to somehow include the bulk B20
structures as training data?"

Short answer: **not as extra rows; yes as one reference point; and worth testing as a control.**
Every descriptor we train on sees the 1,541 transformed bulk cells as a single structure. This note
is a recommendation only, and nothing was computed for it. Each number comes from the file or
README named next to it. Written 2026-10-07.

## 1. Why the 1,541 bulk cells count as one point

- **Standard SOAP.** Every transformed cell gives the pristine vector.
  - At the reference setting (`calculations/descriptors/dscribe/standard/global/rc6.0_n8_l6_s0.5/bulk_invariance.json`),
    the largest relative difference is 1.1e-13 on 100, 5.4e-9 on 110 and 9.3e-8 on 111.
  - Over the 31 standard runs that pass their bulk check, the largest is 3.4e-7
    (`rc5.0_n8_l8_s0.3`, 111). This comes from coordinate rounding in the catalogue.
  - The two `rc8.0_n8_l{4,8}_s0.3` runs fail their check; see
    [`results/grain_class_rep_study/07_settings_grid/`](../results/grain_class_rep_study/07_settings_grid/README.md).
- **Covariant SOAP.** In every bulk cell, the whole-cell average of λ = 1 and λ = 2 is zero, to
  within 9.4e-7 of one atom's vector length
  ([`05_covariant_lambda2`](../results/grain_class_rep_study/05_covariant_lambda2/README.md)).
- **The Stage 1.4/2 GB vector** of a bulk cell is therefore the pristine vector as well. That vector
  is the Co mean joined to the Ge mean over atoms within 8 Å of the interfaces.

**What copies of one point would do:**
- **KRR:** n identical rows act like one row with n times the weight. The fit is pulled through that
  point, and the choice of kernel width and ridge bends towards it.
- **Cross-validation:** copies that land in both the training and the test fold are predicted
  perfectly, so the scores come out too high. This is the rule that keeps the 4x–12x versions of a
  GB in one fold.
- **UMAP, t-SNE, FPS:** the copies form a dense blob that distorts the neighbour graph. Exact
  duplicates are already a known problem: Barnes-Hut t-SNE misplaces them (synthetic test in
  `archive/representation_study_20261005/smoke/diagnostics/SYNTHETIC_tsne_twins.txt`), which is why
  the fifth study uses exact t-SNE.
- **Labels that differ between copies:** plane, operation and the direction of a spin texture. The
  same input with different answers is pure label noise.

## 2. Use by stage (SOAP guide stages)

| Stage | What bulk would add | Recommendation |
|---|---|---|
| 1.4: grain class, plane | Nothing. Bulk has no boundary, so it has no grain class. The two plain-bulk catalogue families, 111/sym00119 and sym00175 (D0985), were dropped for this reason (`ml_common.NO_BOUNDARY`) | No |
| 2: GB energy | **One anchor row:** the pristine vector with E_GB = 0, which is exact by definition (E_cell = n·ε_bulk). The labelled set is tiny: 13 [100], 1 [110] and 1 [111] cells clear the 1.8 Å overlap limit (CLAUDE.md). **Plus a free baseline:** E_GB against distance from bulk in GB-vector space, to compare the KRR with | Yes, once the Stage 2 KRR exists. Report with and without the anchor |
| 3: SOC spin texture | Bulk textures exist: CoGe_100 has 623 SOC scf + dos + bands, CoGe_110 has 497 SOC scf. They could serve as a **baseline**, so the model learns GB minus bulk. Not as rows: the rotated copies have rotated textures but the same standard SOAP vector. Bulk bands also live in the bulk Brillouin zone (cP1 path), while the GB cell's zone is folded | Decide once the Stage 3 target is defined |
| 4: anomaly map | Bulk at the origin, as the guide says. **Already done:** `select_gbs.py` adds the pristine vector as one row, shifts each map so that row is the origin, never selects it, and starts FPS from the GB farthest from it (`farthest_points`) | Keep it. One row, never copies |

**Stage 2 caveats:**
- **ε_bulk has to use the GB settings:** ISMEAR 1, SIGMA 0.05, the production ENCUT and a matched
  k-density. Every E_GB needs this run anyway, and it is an open question in CLAUDE.md §4. The
  finished bulk runs (ISMEAR −1, 600 eV) are not that reference.
- **How much the anchor helps is unknown.** It depends on how far bulk sits from the GB cloud
  compared with the kernel width, and that hasn't been measured. It would matter most for the GBs
  that look nearly bulk-like.
- **The anchor stays in every training fold and is never scored.**

## 3. The one use that adds information now: a no-boundary control for covariant SOAP (proposal)

**Background:** in [`06_representation_response`](../results/grain_class_rep_study/06_representation_response/README.md),
covariant C and D sort GBs by grain class: raw kNN accuracy 0.77 and 0.78, against 0.26 for
standard A, with a majority baseline of 0.24. A sorts them by interface geometry instead.

**The question this control answers:** does that grain-class score come from the boundary, or only
from the direction grain B's lattice points in? Covariant SOAP sees orientation, and grain A's
orientation is fixed for each plane. So grain B's orientation alone already carries the turn R.

**A hint.** This is a hypothesis, not evidence. The table is copied from 06, Results 1 and 2:

| Plane | D vs C residual | Grain-class kNN, raw C / D | Majority |
|---|---|---|---|
| 100 | 0.39 | 0.84 / 0.86 | 0.27 |
| 110 | 0.25 | 0.91 / 0.89 | 0.26 |
| 111 | 3.6e-4 | 0.32 / 0.41 | 0.14 |

- **The pattern:**
  - On 100 and 110, the grain interior's covariant signal does not cancel, and grain class scores
    high.
  - On 111, it cancels, and grain class scores lower.
- **The pattern is confounded:** 111 has 8 classes, and every representation finds it hard (A
  scores 0.11 there).

**Proposed test (not run):**
- **Input:** `structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/Bulk/symNNNNN/POSCAR` is grain B on
  its own: the crystal after that family's operation. Mind the double-scaling trap (CLAUDE.md §6).
- **Method:** cut the same two 8 Å interface regions from the perfect rotated crystal. Compute C with
  the reference covariant settings, then score grain class with the same kNN as in 06.
  - Grain A on its own (the untransformed crystal) gives the same vector for every family of a
    plane. That is a check with zero information by construction.
- **How to read it:**
  - near 0.8: C/D's grain-class power is orientation;
  - near the majority: it comes from the boundary.
- **It also tests the open hypothesis in 06, Result 1:** grain-interior atoms that do not form whole
  sets of symmetry-related sites.
- **Cost:** one small new script plus a test. It covers 1,541 cells, about one ninth of the
  structures in the covariant reference run. **It waits for the user's OK.**

## 4. If the question is about the Hamiltonian models in `src/`

`src/mace` (MACE-H), `src/deeph-e3` and `src/uni-hamgnn` are a different case:
- For them, bulk is the usual first training set.
- Rotated copies add nothing to models that already respect rotations. Strained or rattled bulk
  cells would be the useful data.
- They train on local-orbital Hamiltonians, which the VASP runs don't provide directly. Their setup
  here has not been examined.

## 5. What was not done

- Nothing was computed for this note.
- There is no Stage 2 anchor code; the Stage 2 KRR doesn't exist yet.
- The covariant control has not been run.
- Job B stays unsubmitted.
