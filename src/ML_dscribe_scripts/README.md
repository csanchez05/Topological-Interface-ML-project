# Does the GB vector tell grain boundaries apart? (Stage 1, steps 1.4 and 1.5)

Analysis scripts that work on the SOAP vectors written by `src/dscribe/CoGe/`.
Environment: `source src/mlenv/env.sh`. Everything runs on one core in a few minutes, on a compute
node: loading the vectors takes about 1.9 GB, more than the login node has free (see `CLAUDE.md`).

- **1.4** Kernel ridge regression from the GB vector to the plane and to the operation that built the second grain.
- **1.5** Find GBs built with different operations that have the same vector, count the distinct GBs,
  and check from the POSCARs whether coinciding GBs are one structure built several times.

## Files

| File | What it does |
|---|---|
| `ml_common.py` | Shared code: loads one vector per GB through `soap_common.load_gb`, distances, grouping, writers |
| `gb_labels.py` | Reads the three `Information.txt` files and derives the labels |
| `krr_classify.py` | Step 1.4 |
| `find_duplicates.py` | Step 1.5 |
| `structure_match.py` | Atom-by-atom comparison of two GB cells, used by `find_duplicates.py` |
| `compare_duplicates.py` | Compares the step 1.5 result of two descriptor settings |
| `pooled_duplicates.py` | Step 1.5 on all thicknesses at once |
| `rcut_scan_all.py`, `rcut_scan_all.slurm` | r_cut scan on every structure, run in parallel on one node |
| `rcut_scan_summary.py` | Table and figures for either scan (the 8x-only `rcut_scan.slurm` is in `archive/src/ML_dscribe_scripts/`) |
| `test_ml_scripts.py` | Unit tests on synthetic data |
| `select_gbs.py`, `test_select_gbs.py` | Stage 2 GB selection in PCA, t-SNE and UMAP; sampling tests |
| `descriptor_grid.py`, `descriptor_grid.slurm` | Descriptor study: all SOAP runs, the covariant run and both analyses as one whole-node job |
| `descriptor_sensitivity.py` | Descriptor study part C: which SOAP settings change which GBs look alike; seed and thickness tests |
| `selection_benchmark.py` | Descriptor study part D: which way of choosing GBs teaches a model most from the fewest |
| `test_descriptor_study.py` | Unit tests of the three study scripts and the full-space selection helpers |
| `plot_descriptor_study.py`, `test_plot_descriptor_study.py` | Figures that explain the descriptor study, in `plots/CoGe/dim-reduction/<topic>/`; tests on synthetic study outputs |
| `geometry_labels.py`, `test_free_labels.py` | Free labels that depend on the shift (shortest distance, overlaps, coordination change) per interface, read from the POSCARs; tests on hand-built cells |
| `representation_study.py`, `test_representation_study.py` | Fifth study (standard/covariant × interface/whole cell, every map method) and sixth study (12 SOAP settings × 4 representations) |
| `representation_study.slurm` | Fifth and sixth studies as two whole-node jobs (`--part A`, `--part B`) through `descriptor_grid.py --study representation` |

## Stage 2: selecting GBs for calculation

```bash
# in a compute-node shell (see Running below)
source src/mlenv/env.sh
python -B src/ML_dscribe_scripts/select_gbs.py
python -B src/ML_dscribe_scripts/select_gbs.py --verify-only
```

Uses the 8x rows of `calculations/descriptors/dscribe/standard/both_interfaces/rc6.0_n8_l6_s0.5`
(interface radius 8, both interfaces; the run of 2 Oct 2026 read the identical 8x-only copy
`rc6.0_n8_l6_s0.5_t8`, now archived), and the existing family-level structure census to keep 1,112
distinct GBs. Pristine
bulk is included as an unselectable reference and placed at the origin of
each map. Features are not normalized or standardized. Sampling uses Euclidean
distances in each **two-dimensional embedding**. FPS starts farthest from bulk;
the central selection takes the actual member nearest each of 30 K-means
centroids. The output folder named `median` holds this centroid selection.
Selections can overlap between lists. Stochastic methods use seed 42 and one thread.

`plots/CoGe/dim-reduction/{FPS,median}/` contains `pca.txt`, `tsne.txt` and
`umap.txt`, each with exactly 30 project-relative POSCAR paths, and
`{method}_{colored,selected}.png`. Colored views use the eight established
`grain_class` labels from `labels/gb_labels.csv`, with consistent colors and
every class named in the legend. Selection views use bright red and gray;
`--selected-background grain_class` instead keeps class colors behind the red
selections. Every figure has a legend, including pristine bulk and selection
markers where applicable. Bulk is a gray
diamond. All figures are 1600 by 1200 pixels, with labeled axes and ticks but
no labels on individual structures. PCA axes include their explained variance;
t-SNE and UMAP axes identify the two embedding coordinates.

Coordinates, structure mappings, selected row indices, settings, input/output
hashes, and validation results are in `calculations/ML_dscribe/CoGe/dim-reduction/`.
Verification replays sampling from the saved coordinates without refitting
the embeddings. No new SOAP descriptors or DFT calculations are run.

Refitting does not give the kept t-SNE and UMAP lists back. Rerun on 3 Oct 2026 with the old code
on the old files and with the new code on the new files (bit-identical results on qcg1-1): on qcg1-1
(the same CPU as qcg2-2) and qcg3-0 the two PCA lists come back exactly, the t-SNE and UMAP lists do
not; on qcg0 the PCA k-means list differs as well. The maps agree between qcg1-1 and qcg3-0 but not
with qcg0 or the kept run. The kept run's PCA coordinates differ from the reruns by 1e-14, so it was
made on a machine with different floating-point rounding (its host is not recorded), and t-SNE and
UMAP, and k-means on a map, amplify such differences. The kept lists still pass `--verify-only`. The
descriptor study's seed test measures how much the picks depend on such details.

### Selecting within a subset (`--planes`, `--min-pair`, `--n-select`)

```bash
python -B src/ML_dscribe_scripts/select_gbs.py --planes 100 --min-pair 1.8 --dry-run   # counts only; login node is fine
python -B src/ML_dscribe_scripts/select_gbs.py --planes 100 --min-pair 1.8             # compute node
```

Used on 2026-10-07 to refill the [100] list under the 1.8 Å overlap limit (`CLAUDE.md` §4).
- **Filter.** Only the distinct GBs of `--planes` whose closest pair of atoms at 8x is ≥ `--min-pair` Å
  are kept. The closest pair comes from `results/CoGe_interface/overlap_survey.tsv` (`--overlap-survey`,
  rounded to 0.001 Å; see `src/dft_plots/coge_interface/`). The run stops if a structure class has
  families on both sides of the limit or spans planes, or if a family of those planes has no closest pair.
- **Maps.** Only the kept GBs and pristine bulk are embedded, so PCA, t-SNE and UMAP spread out that set
  rather than all 1,112 GBs. The rules are otherwise unchanged: seed 42, FPS from bulk, the member nearest
  each k-means centroid, `--n-select` per list (default 30).
- **Folders.** Without `--plot-dir`/`--out-dir` the outputs go to a tagged subfolder, for example
  `plots/CoGe/dim-reduction/plane100_dmin1.8/{FPS,median}/` and
  `calculations/ML_dscribe/CoGe/dim-reduction/plane100_dmin1.8/`.
- **What changes in the outputs.** The figures carry a title naming the subset, and their legends list
  only the grain classes present. `run.json` records the filter under `subset`.
- **Every run** also writes `sample_list.csv`, the union of the six lists in the layout of
  `results/dscribe/total_sample_list.csv`.
- **Guard.** A run stops if a selection already exists in its target folders; `--overwrite` replaces it.

## Running

From the project root, inside a compute-node shell (not on the login node):

```bash
srun -p general --exclude=qcg-docker -c 4 --mem=8G -t 01:00:00 --pty bash
source src/mlenv/env.sh
python -B src/ML_dscribe_scripts/gb_labels.py                      # label tables + checks, 10 s
python -B src/ML_dscribe_scripts/find_duplicates.py --structure-census --single-interfaces   # 1.5, 2 min
python -B src/ML_dscribe_scripts/krr_classify.py                   # 1.4, 1.5 min
python -B -m unittest discover -s src/ML_dscribe_scripts -p 'test_*.py'
```

Which vector is used (the same options on `krr_classify.py` and `find_duplicates.py`):

| Option | Default | Note |
|---|---|---|
| `--descriptor` | `rc6.0_n8_l6_s0.5` | a tag under `calculations/descriptors/dscribe/standard/`, or a folder |
| `--kind` | `interface` | or `nonbulk`, `global` |
| `--radius` | `8` | one of the stored radii; only for `interface` |
| `--planes` | `both` | or `edge`, `mid` (one of the two interfaces in the cell) |
| `--thickness` | `4` | one thickness for every GB, or `all` to pool them (`krr_classify.py`, `pooled_duplicates.py`) |

After raising n_max and l_max (step 1.5 of the guide):

```bash
N_MAX=10 L_MAX=8 sbatch src/dscribe/CoGe/run_soap.slurm
# then, in a compute-node shell as above:
python -B src/ML_dscribe_scripts/find_duplicates.py --descriptor rc6.0_n10_l8_s0.5
python -B src/ML_dscribe_scripts/compare_duplicates.py rc6.0_n8_l6_s0.5 rc6.0_n10_l8_s0.5
```

Loading the vectors goes through `soap_common.load_gb`, which reads only the arrays of the chosen
radius (about 0.2 GB stored, about 1 GB in memory while combining the planes, at the default settings).
Still a compute-node job.

## Outputs

One folder per descriptor and per choice of vector, so reruns never overwrite each other:

```
calculations/ML_dscribe/CoGe/
  labels/                                   gb_labels.csv, gb_operations.csv, gb_labels.json
  <descriptor tag>/<vector tag>/            e.g. rc6.0_n8_l6_s0.5/interface_r8_both_t4/
    krr_summary.csv, krr_summary.json       accuracies; settings and chosen hyperparameters
    krr_predictions.csv                     out-of-fold prediction for every GB
    krr_confusions.csv                      true x predicted counts
    krr_confused_pairs.csv                  label pairs the model mixes up, most mixed first
    krr_confusion_*.png                     plane and grain class, "distinct" grouping
    distinct_gbs.csv                        every GB with its distinct-GB number (use this later to drop repeats)
    duplicate_groups.csv                    one row per coinciding group, with operations and structure verdict
    duplicate_members.csv                   one row per GB in a group
    label_merges.csv                        operation labels that share a vector
    near_misses.csv                         closest pairs that do not coincide, structure-tested
    duplicates_summary.json                 counts, tolerance, sensitivity, structure-test totals
  comparisons/<A>__<B>/<vector tag>/        comparison_groups.csv, comparison_summary.json
```

## Labels

`Interface N` in `Information.txt` is folder `symNNNNN` (numbering starts at 0). Each interface lists
every operation, rotation part plus translation, that builds it. `gb_labels.py` parses all of them
(623 / 543 / 375 interfaces, 2580 / 860 / 430 operations) and checks the file against itself
(declared counts, summary list, header).

**Frame.** The operations are written for the fractional coordinates of *that plane's bulk cell*, not
in the cubic frame. For 100 the two are the same. For 110 and 111 the cell axes are

| Plane | x (interface normal) | y | z |
|---|---|---|---|
| 110 | [110] | [-110] | [001] |
| 111 | [111] | [-110] | [-1-12] |

so, for example, `2 [1-10]` in the 110 file is the two-fold axis along cubic [100]. Only operations
that map the cell onto itself were kept by the generator: all 48 for the cubic 100 cell, 16 for the
tetragonal 110 cell, 8 for the orthorhombic 111 cell. `gb_labels.py` converts every rotation to the
cubic frame before labelling, so one name means one rotation on all three planes. Names: `E`, `I`
(inversion), `R120[111]` (rotation by 120 degrees about [111]), `I.R180[001]` (inversion times that
rotation, i.e. a mirror).

Checked by the script on every run: the frames above reproduce the 110 and 111 reference cells from
the pristine cubic cell; and every listed operation, applied to the plane's bulk cell, reproduces
`Bulk/symNNNNN/POSCAR` within the source coordinate precision. So `Bulk/symNNNNN` is the second grain of `Interface/symNNNNN`, and
all operations listed for one interface make the same second grain.

**Label schemes** (columns of `gb_labels.csv`):

| Label | Meaning | Classes (100 / 110 / 111 / all) |
|---|---|---|
| `plane` | interface plane | 3 in all |
| `handedness` | `same` for a proper rotation, `opposite` for an improper one (B20 is chiral) | 2 / 2 / 2 / 2 |
| `grain_class` | second-grain orientation and relative handedness, modulo the 12 crystal rotations and ignoring translation | 4 / 4 / 8 / 8 |
| `first_op` | the first rotation listed | 16 / 16 / 8 / 22 |
| `rot_set` | the full set of listed rotations (members in column `rotations`) | 44 / 48 / 8 / 96 |

`grain_class` describes the second grain relative to the first crystal's fixed frame.
Two rotations give the same ideal second grain (up to a shift) exactly
when they differ by a rotation of the crystal, and every interface's operations fall in one class.
The class is named after its member with the smallest angle. On 100 and 110 there are four:
`E` (same orientation), `I` (inverted), `R90[100]`, `I.R90[100]`. These are the "four different
second grains" of the model-cell check. On 111 there are four more (`R60[111]`, `R70.53[1-10]` and
their inverted partners) because a two-fold axis of the 111 cell along [111] or [11-2] is not a
cubic operation. `first_op` and `rot_set` depend on which of several equivalent operations the
generator happened to list, so they are finer than anything physical; they are kept for reference.

**Scope of the eight classes:** this is the count within the catalogue's finite construction
operation set, not a universal count of CoGe orientations or grain-boundary structures, and not
eight clusters discovered by SOAP, PCA or UMAP. The retained sets contain 48, 16 and 8 distinct
orthogonal matrices on 100, 110 and 111. Their intersections with the crystal point group contain
12, 4 and 1 rotations, giving 48/12 = 4, 16/4 = 4 and 8/1 = 8 orientation/handedness classes.
The first two planes use the same four labels; 111 contains all eight.

The Stage 1 audit `verify_grain_classes.py` independently derives the crystal group from pristine
atomic coordinates using spglib and exhaustive lattice-isometry matching. It partitions the raw
matrices without importing `gb_labels`, independently groups all stored bulk cells by
species-preserving atom matching under translation, and checks both grain interiors in every 8x
boundary. Evidence is in `results/dscribe/grain_class_verification.json`. It also checks that all
eight classes remain distinct under symmetries of both grains and grain exchange. This does not
establish eight different energies, electronic structures or spin textures.

The relative translation, boundary plane and termination can distinguish GBs within a grain
class. For example, 100/sym00001 and 100/sym00002 both have label E, but their minimum Co–Ge
distances in the 8x boundary cells are 2.39749 and 1.90988 Å, respectively; they cannot be the
same atomic structure under a rigid motion. A 36.86990° rotation about [001], with cosine 4/5
and sine 3/5, gives a commensurate orientation outside all eight classes. Thus the catalogue
does not exhaust the possible orientations. `x_distance` and `total_distance` are copied from
the information files; their meaning was not checked.

To repeat the audit, use a new output path (existing files are refused):

```bash
srun -p general --exclude=qcg-docker --nodes=1 --ntasks=1 -c 1 --mem=2G -t 00:15:00 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  src/dft_plots/.venv/bin/python -B src/ML_dscribe_scripts/verify_grain_classes.py \
  --out results/dscribe/grain_class_verification_repeat.json
```

## Step 1.4: `krr_classify.py`

- Model: scikit-learn `KernelRidge` fitted to one 0/1 column per class; the predicted class is the
  column with the largest value. Kernel `exp(-gamma d^2 / median d^2)`.
- Hyperparameters (gamma, ridge strength, feature scaling) are chosen by 3-fold cross-validation
  inside each training set. Grid: 4 x 4 x 2, changeable on the command line.
- A k-nearest-neighbour classifier (k chosen the same way) is reported next to it.
- Data sets: all planes together, and each plane on its own.
- `--jobs N` runs the folds in N separate one-thread processes (default 1). The results do not
  depend on N. Worth it for `--thickness all`, where one fold takes minutes and about 8 GB.
- Two ways of leaving GBs out (5 folds each):
  - `family`: the guide's rule. With one thickness per GB this is plain cross-validation.
  - `distinct`: GBs whose vectors coincide (step 1.5) are also kept in one fold. Otherwise a test GB
    can have an identical copy in the training set, and the score measures look-up, not prediction.
    **Quote this one.**
- `krr_summary.csv` columns: `baseline_majority` (always predict the commonest class of the training
  set), `krr_accuracy`, `krr_fold_std`, `knn_accuracy`, `ceiling_from_coinciding_vectors` (coinciding
  vectors with different labels cap what any model can reach), accuracy split by whether a GB has a
  coinciding partner and whether a copy of its vector was in the training set, and for `first_op` the
  share of predictions that hit any of the listed operations.
- **Grain class per plane** (`krr_by_plane.csv`): a 100 or 110 GB can only be one of 4 grain classes
  (chance 1/4), a 111 GB one of 8 (chance 1/8), so the pooled score mixes three questions. This file
  scores the pooled model's out-of-fold predictions on each plane's GBs alone, next to the models
  trained on one plane: accuracy, balanced accuracy, chance, majority baseline, n, the share of
  impossible predictions (a 111-only class for a 100/110 GB) and, for grain class, the share of
  errors that are a mirror class (`mirror_share_of_errors`, next to `mirror_share_if_random`). On the stored rc6.0 4x record the pooled model scores 0.631 / 0.554 / 0.149 on 100 / 110 / 111.
- **D0985 is dropped** by default: `111/sym00119` and `111/sym00175` are plain bulk (`ml_common.NO_BOUNDARY`),
  so the pool is 1,111 distinct GBs (1,539 families). `--keep-no-boundary` keeps them. Library
  defaults in `ml_common.py` are unchanged, so `select_gbs.py`, `find_duplicates.py` and the scans
  still see 1,112.
- **`--regression [label …]`** (no names = all of `geometry_labels.TARGETS`: `d_min`,
  `log1p_n_close`, `cn_excess`, `cn_defect`) adds KRR regression with its own folds: settings by
  the inner folds' squared error; R² and MAE against predicting the training mean, pooled and per plane, in
  `krr_regression.csv`.
- **Refusal**: a folder that already holds `krr_summary.csv` is refused unless `--overwrite` is given.
  The rerun without D0985 writes `<tag>/interface_r8_both_t4_no_boundary/`; the 1,112-GB record in
  `interface_r8_both_t4/` stays.

## Step 1.5: `find_duplicates.py`

**Distance and tolerance.** Relative distance `|a - b| / mean(|a|, |b|)`; two vectors coincide at or
below `--tol` (default 1e-4), and groups are chains of coinciding pairs. Why 1e-4: the vector of one
GB changes by up to 9e-6 between thicknesses (measured by the script; coordinates are written to
3-6 decimals), float32 storage adds about 1e-7, so the tolerance sits ten times above the noise. The
summary reports the largest coinciding distance, the smallest other distance, and the count of
distinct GBs at tolerances from 1e-7 to 3e-3, so the choice can be judged from the data.

**Structure test.** For every group the 4x POSCARs are compared atom by atom (`structure_match.py`):

- *whole cell*: is there a rigid motion, proper or improper, that puts every atom of one cell on an
  atom of the same species in the other, and back? Tried: x to +x or -x, in-plane rotation by any
  multiple of 60 or 90 degrees, optional in-plane mirror, any shift. A motion is only accepted if it
  also maps each cell's in-plane periods onto real translations of the other.
- *interface slab*: the same for the atoms within `--slab` (12 A) of one interface plane, with the
  two planes placed on each other. This tells which of the two interfaces in each cell match.

Group verdicts: `same cell` (one structure built several times), `same interfaces` (cells differ but
hold the same interfaces), `different`. Pairs just above the tolerance (up to `--near-miss` times it)
get the same test and go to `near_misses.csv`.

**`--structure-census`** sorts all cells into rigid-motion classes from the POSCARs alone, without
SOAP, and compares that with the vector groups. This is the independent count of distinct GBs.

**`--single-interfaces`** pools the edge and mid vectors of every cell and counts distinct single
interfaces (vector-based only; not structure-tested).

## First results (rc6.0_n8_l6_s0.5, interface radius 8 A, both planes, 4x; 2 Oct 2026)

Distinct GBs:

| Plane | Families | Distinct | Coinciding groups | Families in groups |
|---    |-  -     -|-   -    -|---                |                 ---|
| 100   | 623      | 385      | 186               | 424 |
| 110   | 543      | 503      | 40                | 80 |
| 111   | 375      | 224      | 56                | 207 |
| all   | 1541     | 1112     | 282               | 711 |

- All 282 groups are `same cell`. The census without SOAP gives the same 1112 classes, GB for GB.
- Coinciding pairs differ by at most 2.5e-5; the closest other pair by 3.4e-4. The count is 1112 for
  any tolerance from 3e-5 to 3e-4.
- No group mixes two grain classes or two handednesses. 118 groups mix `first_op` and 100 mix `rot_set`.
- The 440 pairs between 1e-4 and 1e-3 are all on 111, all join two different grain classes of
  opposite handedness (`E` with `I.R70.53[1-10]`, `I` with `R70.53[1-10]`, `R90[100]` with
  `I.R60[111]`, `I.R90[100]` with `R60[111]`), and are different structures. These are the pairs to
  watch when n_max and l_max are raised.
- Single interfaces: 2024 distinct among 3082 (669 / 939 / 416).

KRR accuracy (majority baseline in brackets):

| Label       | Data| `family` | `distinct` |
|--          -|--  -|--         -|---|
| plane       | all | 0.999 (0.40)| 0.999 (0.40)|
| grain class | all | 0.70 (0.19) | 0.49 (0.19) |
| grain class | 100 | 0.81 (0.20) | 0.56 (0.19) |
| grain class | 110 | 0.60 (0.20) | 0.58 (0.20) |
| grain class | 111 | 0.63 (0.07) | 0.14 (0.05) |
| handedness  | all | 0.74 (0.49) | 0.55 (0.46) |
| first op    | all | 0.38 (0.06) | 0.24 (0.06) |

The `family` numbers are lifted by repeats: 611 test GBs had an identical vector in training and
all of those are classified correctly; the rest score 0.50 on grain class. On 100 the model mixes
classes that differ only by inversion (`E` with `I`, `R90[100]` with `I.R90[100]`); on 111 it mixes
the near-miss pairs listed above.

## r_cut scan on every structure (`rcut_scan_all.py`, `rcut_scan_all.slurm`)

```bash
sbatch src/ML_dscribe_scripts/rcut_scan_all.slurm             # whole node qcg0, all 96 CPUs
sbatch --nodelist=qcg3-0 --cpus-per-task=256 src/ML_dscribe_scripts/rcut_scan_all.slurm
python -B src/ML_dscribe_scripts/rcut_scan_all.py --dry-run   # list the tasks without running them
python -B src/ML_dscribe_scripts/rcut_scan_summary.py         # table + figures, after the job
```

For each cutoff (default 4 5 6 8 10 12 A; n_max, l_max and sigma unchanged) the scan computes SOAP
for all 13,868 structures (4x-12x) and then runs steps 1.5 and 1.4 on two GB vectors: interface
radius r_cut + 2 A (every disturbed atom) and a fixed 8 A (the same atoms at every cutoff). Each
vector is analysed two ways:

- **by thickness**: 4x, 5x, ... 12x separately (`find_duplicates.py`, `krr_classify.py --thickness T`);
- **pooled**: all thicknesses in one analysis (`pooled_duplicates.py`, `krr_classify.py --thickness all`).
  The `distinct` grouping keeps every thickness of a GB, and every copy of it, in one fold.

`rcut_scan_all.py` is the driver. It holds all tasks (6 SOAP runs, 11 pooled analyses, 99
one-thickness analyses), gives each a CPU cost, and starts a task as soon as its SOAP run is done
and enough CPUs are free, so the node stays full. Finished tasks are recognised by their output
files and skipped, so the job can be resubmitted after a cancel or a failure. A failed task does
not stop the others. Progress: `rcut_scan_all/scan_status.json`; one log per task in `rcut_scan_all/logs/`.

CPU use per task: a SOAP run uses 16 worker processes (`--descriptor-cpus`); a one-thickness
analysis uses 1 CPU (`--thickness-cpus`); a pooled analysis uses 6 (`--pooled-cpus`), passed on as
`krr_classify.py --jobs 6`, which runs the 40 folds of that analysis (4 data sets x 2 groupings x
5 folds) in separate one-thread processes. One fold of the 13,868-structure data set needs about
8 GB, so 11 pooled analyses x 6 processes need about 530 GB. On qcg3-0 (256 CPUs) add
`--pooled-cpus 8` after the script name; more than that would not fit in memory with all 11 running.

```
calculations/descriptors/dscribe/standard/{global,both_interfaces}/rc<r_cut>_n8_l6_s0.5/   descriptors
calculations/ML_dscribe/CoGe/rcut_scan_all/
  by_thickness/rc<r_cut>/interface_r<R>_both_t<T>/              one thickness
  pooled/rc<r_cut>/interface_r<R>_both_tall/                    all thicknesses: krr_*, pooled_*
  rcut_scan_summary.csv, rcut_scan_by_thickness.png, rcut_scan_pooled.png
```

Pooled outputs of step 1.5: `pooled_families.csv` (is each family thickness-consistent, and how far
apart its thicknesses are), `pooled_merges.csv` (different structures that share a vector group,
and at which thicknesses), `pooled_by_thickness.csv`, `pooled_summary.json`. "Different structure"
is taken from the `structure_class` column of the 4x census
(`rc6.0_n8_l6_s0.5/interface_r8_both_t4/distinct_gbs.csv`).

Thin cells at large cutoffs: in a 4x [100] cell the two interfaces are 18.5 A apart, so from
r_cut of about 8-10 A they see each other, and a radius above a quarter of the cell covers the
whole cell. Those analyses are run like the others; the grid figure marks them with `*`.

### 8x-only scan (`archive/src/ML_dscribe_scripts/rcut_scan.slurm`)

The first scan used one thickness (8x, 1,541 structures per cutoff). Its results are in
`calculations/ML_dscribe/CoGe/rcut_scan_8x/`
(`python -B src/ML_dscribe_scripts/rcut_scan_summary.py --scan-dir calculations/ML_dscribe/CoGe/rcut_scan_8x`).

| r_cut | Plane | Grain class | Distinct vectors (r_cut + 2 / fixed 8) | Closest different GBs | Farthest copies |
|---|---|---|---|---|---|
| 4 | 0.996 | 0.54 | 1112 / 1112 | 8.3e-4 | 5.5e-5 |
| 5 | 0.999 | 0.48 | 1112 / 1112 | 5.4e-4 | 3.4e-5 |
| 6 | 0.999 | 0.49 | 1112 / 1112 | 3.4e-4 | 2.6e-5 |
| 8 | 0.999 | 0.52 | 1112 / 1112 | 1.5e-4 | 6.1e-5 |
| 10 | 0.998 | 0.55 | 1109 / 1112 | 1.0e-4 | 9.4e-5 |
| 12 | 0.999 | 0.55 | 1106 / 1112 | 1.0e-4 | 9.9e-5 |

(Accuracy and distances are for the r_cut + 2 vector, `distinct` grouping, 8x only.)

## Descriptor study (`descriptor_grid.py`, `descriptor_sensitivity.py`, `selection_benchmark.py`, `plot_descriptor_study.py`)

Before Stage 2 picks GBs for DFT: does the choice of SOAP settings change which GBs look alike, and
which way of picking GBs teaches a model the most from the fewest labels? In plain words: change one
setting and see whether the GBs that were close together stay close together (part C); pretend only n
GBs may be labelled and see which way of choosing them learns fastest (part D).

- **Grid** (`descriptor_sensitivity.soap_runs()`): each run changes one setting away from the
  reference (r_cut 6, n_max 8, l_max 6, sigma 0.5, no weighting): r_cut 4–12 (the scan's runs),
  n_max 4–12, l_max 2–10, sigma 0.3–1.0, and the `pow` (Willatt) and `poly` (Caro) weightings at
  r_cut 6, 8 and 12. Plus region and processing choices that need no new run: radius 6/10, one
  plane, nonbulk atoms, global vector; one mean over all atoms, standardised, unit length.
- **Part C** compares every setting with its baseline on the 1,112 distinct GBs at 8x: rank
  correlation of all distances, nearest neighbours kept, feature reconstruction error both ways
  (GFRE), Procrustes distance of the PCA maps, overlap of 30 FPS picks, variance spread in PCA.
  Then the knobs are ranked. Thickness test: how much of PC1 follows the thickness for the global
  and the interface vector (all 13,868 structures). Seed test: t-SNE and UMAP with 10 seeds, and
  one map per setting against that seed-to-seed spread.
- **Part D** holds out 10 random test sets of 222 GBs and picks n = 5–100 GBs from the rest with:
  random, stratified (plane × grain class), FPS and k-means in the full space, GP variance
  (label-free Bayesian active learning), and `select_gbs.py`'s six map rules. KRR with fixed Stage 1
  settings and 1-NN learn grain class, plane and handedness (a control). Label-free coverage scores
  for every pick set and for the six existing lists. FPS and random curves for every grid setting.
- The covariant SOAP run (reference setting) is part of the same job; it is not used in C or D.
- **Figures**: the last task of the job runs `plot_descriptor_study.py`, which only reads the outputs
  above and writes one folder per question into `plots/CoGe/dim-reduction/` (`descriptor_settings/`,
  `maps_by_setting/`, `seed_stability/`, `thickness/`, `selection/`, `covariant/`; what each figure
  shows is in that folder's README). It leaves `select_gbs.py`'s `FPS/` and `median/` alone.
  Four of its figures reuse the analyses' own drawing code: `descriptor_sensitivity.draw_sensitivity`
  and `draw_seeds`, and `selection_benchmark.draw_coverage` and `draw_descriptors` return the figure
  unsaved. The plot script adds a title and reading note; the analyses' `plot_*` save their own copy
  as before. `cost.png` shows vector length only: the SOAP timings in `settings.json` are not
  comparable between runs.
  Figures not drawn from the full run on the real structures need `--title-prefix "SMOKE TEST (…)"` or
  `"SYNTHETIC DATA (…)"`, which starts every title with that text. The script refuses to write such
  figures, or figures drawn from inputs under `archive/`, into `plots/CoGe/` or `results/` (exit 2).
- **Results** (job 12808, 4 Oct 2026): `results/dscribe/descriptor_study.md`.

```bash
python -B src/ML_dscribe_scripts/descriptor_grid.py --dry-run --cpus 256   # task list, on the login node
sbatch src/ML_dscribe_scripts/descriptor_grid.slurm                         # the whole study, one node
```

Cost, from the smoke tests (`archive/descriptor_study_20261003/smoke/`): about 200 CPU-h in all
(17 SOAP runs at about 25 min on 16 CPUs each, covariant about 48 CPU-h, both analyses under 30 min),
about 1.5 h on qcg3-0 or 2–2.5 h on qcg0; about 22 GB of disk. The analyses' workers run one
thread each (`--jobs` workers use `--jobs` CPUs).

`descriptor_grid.py` uses `rcut_scan_all.py`'s scheduler: tasks start when CPUs and memory allow,
finished tasks are skipped on resubmission, a failed task only skips what needs it. Task logs and
`status.json`: `calculations/descriptors/dscribe/logs/grid/`. The two analyses can be rerun alone
(`--only` for a subset of settings, `--jobs` for worker processes; see each script's `--help`).
The figures can be redrawn alone (about 2 min, 0.3 GB, one CPU; a small compute node):

```bash
python -B src/ML_dscribe_scripts/plot_descriptor_study.py --dry-run         # which inputs exist
python -B src/ML_dscribe_scripts/plot_descriptor_study.py --only thickness selection
```

```
calculations/ML_dscribe/CoGe/descriptor_sensitivity/
  sensitivity.csv, knob_ranking.csv, seed_robustness.csv, thickness_test.csv, summary.json,
  maps.npz (PCA maps, t-SNE/UMAP maps and their picks), thickness_pcs.npz (PC1/PC2 of every structure),
  sensitivity.png, seed_robustness.png, pca_maps.png
calculations/ML_dscribe/CoGe/selection_benchmark/
  learning_curves.csv, coverage.csv, summary.csv, existing_lists.csv, picks.csv, summary.json,
  maps.npz (select_gbs.py's maps with families and labels),
  learning_curves.png, coverage.png, descriptor_efficiency.png
plots/CoGe/dim-reduction/
  descriptor_settings/, maps_by_setting/, seed_stability/, thickness/, selection/, covariant/
```

## Fifth and sixth studies (`representation_study.py`, `geometry_labels.py`, `representation_study.slurm`)

Still SOAP guide Stage 1, before Stage 2 trusts any map. **Fifth study:** keep the SOAP settings at the
reference and change only the *kind* of representation; how much does each dimensionality-reduction
method's map change? **Sixth study:** does that answer hold for other SOAP settings? In plain words:
give the same GBs to PCA, UMAP, t-SNE … in four forms, and see which methods redraw the map when the
form changes, more than their own randomness and more than the descriptors themselves changed.

| | interface (atoms within 8 Å of one interface) | whole cell (mean over every atom) |
|---|---|---|
| standard SOAP | **A** `standard/both_interfaces/<tag>` `X_r8[:, edge/mid]` | **B** `standard/global/<tag>` |
| covariant SOAP, λ = 1 and 2 | **C** `covariant/both_interfaces/<run>` | **D** `covariant/global/<run>` |

- **Rows:** one per interface, the 1,111 distinct GBs at 8x (Stage 1.5 census, D0985 dropped) × edge
  and mid = 2,222. B and D give both rows of a GB the same vector, so kNN, Jaccard, Spearman and
  trustworthiness leave out the other row of the same GB in every representation; Procrustes and ARI
  use all rows. No feature scaling. C0 (C plus λ = 0) is a one-off check in phase 1.
- **Methods:** PCA, UMAP (phase 1); kernel PCA, ICA, LDA (plane, grain class), PLS and PCovR (plane,
  grain class, geometry labels), metric MDS, Isomap, t-SNE, PaCMAP, diffusion maps, sketch-map
  (phase 2; sketch-map from `src/third_party/sketchmap`). Distance settings (kernel width, diffusion
  bandwidth, sketch-map σ) come from each representation's median pairwise distance. t-SNE uses the
  exact method: Barnes-Hut misplaces exactly duplicated rows, which B and D have (synthetic twins: map
  KL 5.0 against 0.2 at 56 rows, 1.4 against 0.7 at 1,000, 0.95 against 0.90 at 2,222;
  `archive/representation_study_20261005/smoke/diagnostics/SYNTHETIC_tsne_twins.txt`).
- **Comparisons** for each method, subset (pooled, 100, 110, 111) and each of the 6 pairs: Procrustes
  disparity, kNN Jaccard (k = 15, 50), Spearman of pairwise distances, ARI of 8 Ward clusters.
  **Noise floor:** the same scores between 10 seeds (stochastic methods) or 10 subsamples of 80 % of
  the GBs (deterministic ones); significant = above the floor's 95th percentile. **Input change:**
  Jaccard and Spearman on the raw vectors; **sensitivity** = map change / input change.
  Per map: kNN accuracy for plane, grain class (also per plane) and edge/mid, kNN R² for the geometry
  labels, trustworthiness, continuity, random-triplet accuracy. 2 × 2 effects (region, covariance,
  interaction): PERMANOVA on map-to-map changes, two-way ANOVA on the per-map variables. Sums of
  squares below 1e-12 of the total count as 0; with a zero residual (replicate maps identical, as for
  ICA in the smoke test) a term that explains something is significant and one that explains nothing
  has no p-value. Jaccard is NaN when k > rows − 3 (small subsets or their subsamples).
- **Method numbers:** t-SNE's KL divergence is KL(P‖Q) of the finished map with the dense P
  (`tsne_kl`), computed afresh rather than taken from sklearn (whose Barnes-Hut estimate came out
  negative for B and D in the smoke test). Diffusion maps: `spectral_gap_after` = the number of non-trivial eigenvalues
  before the largest gap among the first 10.
- **Phase 1** also runs the bulk test: for the pristine cell, the 3 plane references and the 1,541
  transformed bulk cells, the length of the cell average of each λ ≥ 1 block over one atom's length
  (expected near zero by the crystal's rotations); D against the count-weighted sum of C_edge and
  C_mid; B against A plus the bulk vector.
- **Thickness check** (`--phase thickness`): every method on all 9 thicknesses of 300 GBs (stratified
  by plane), and PCA on all 13,868 structures for A–D.
- **Sixth study** (`--study settings_grid`): PCA, metric MDS, UMAP, diffusion maps × 12 settings
  (r_cut 5/6/8 × l_max 4/8 × σ 0.3/0.5, n_max 8) × A–D × 5 replicates; PERMANOVA splits the changes
  into representation, settings, interaction and replicate noise. The reference is an anchor outside
  the split. Covariant grid runs are 8x only, λ ≥ 1 (`<tag>_t8`).
- **Geometry labels** (`geometry_labels.py`, all 13,868 structures, edge/mid/cell): `d_min` (shortest
  distance over every species pair, so Co–Co or Ge–Ge can set it),
  `n_close` (pairs < 1.5 Å per nm², modelled as `log1p_n_close`), `cn_excess`, `cn_defect`
  (coordination change against bulk per nm², cutoff 3.33 Å in the pristine shell gap). Checks: every
  bulk cell gives the pristine values; every family agrees across 4x–12x. Positions are compared to
  1e-3 Å (`POSITION_TOL`) because the catalogue gives y and z to 3–4 decimals (up to about 4e-4 Å
  off; the 111 bulk cells give d_min 2.3970 instead of 2.3975 Å).
  `krr_classify.py` and `selection_benchmark.py` read them from `--labels` (default
  `labels/geometry_labels.csv`); the driver passes `--labels-dir`, so the smoke test uses its own file.

```bash
source src/mlenv/env.sh
python -B src/ML_dscribe_scripts/descriptor_grid.py --study representation --part A --dry-run --cpus 56 --mem-gb 360
sbatch src/ML_dscribe_scripts/representation_study.slurm --part A   # then look at phase 1 before:
sbatch src/ML_dscribe_scripts/representation_study.slurm --part B
```

Job A: geometry labels → Stage 1.4 KRR without D0985 plus regression
(`rc6.0_n8_l6_s0.5/interface_r8_both_t4_no_boundary/`) → selection benchmark v2
(`selection_benchmark_v2/`, `--no-descriptors`) → covariant reference with λ = 2 (all thicknesses,
written as `rc6.0_n8_l6_s0.5_lam2`, checked against the λ ≤ 1 run, which it must reproduce bit for
bit) → 10 new standard and 12 covariant grid runs → phase 1 → figures (`selection/`, `free_labels/`,
`representation_response/`). Job B: phase 2, thickness check, sixth study, figures
(`representation_response/`, `settings_grid/`). Every task writes a new folder and none passes
`--overwrite`; figures that a rerun replaces move to `archive/representation_study_20261005/replaced_by_part<A|B>/`.
The plot task's done marker is `representation_response/phase<1|2>_done.txt`. Task logs:
`calculations/descriptors/dscribe/logs/representation/part<A|B>/`.

Cost (measured 2026-10-05 on one qcg1-1 core with 2,222 synthetic rows, `archive/representation_study_20261005/smoke/timing/SYNTHETIC_timing.txt`,
plus the earlier full runs):

| Piece | Time | Note |
|---|---|---|
| one 2,222-row map: PCA, kernel PCA, LDA, PLS, PCovR, Isomap, diffusion maps | 0.1–3 s | |
| ICA, MDS, PaCMAP, UMAP | 3–20 s (MDS main map 44 s: its extra variables) | UMAP's first call in a process 327 s, mostly numba compiling (hence the node-local cache) |
| t-SNE, exact method | 58 s per 2,222-row map on qcg2-2 (synthetic; 12 s at 1,000 rows); about 6 min at 5,400 rows (thickness check, scaled as rows²) | the main map runs it three times (perplexity 15, 30, 50); in all about 1.5 CPU-h of Job B |
| sketch-map | 2,108 s per 2,222-row map (13 `dimred` calls) | grows with rows²: about 3.5 h for a 5,400-row thickness map; ≈ 32 CPU-h in phase 2 + 14 CPU-h in the thickness check ≈ 46 CPU-h of Job B |
| covariant reference, λ ≤ 1 (earlier run) | 1,812 s on 38 workers | λ = 2 adds 16 blocks to the 19 |
| one standard SOAP setting, all thicknesses (earlier runs) | 1,600–2,100 s | |
| `check_covariant.py` on a smoke grid run | 88–94 s | |
| one map-pair comparison, 2,222 rows | 0.02 s on the same rows, 0.07 s for two 80 % subsamples; features 0.09–0.14 s per map | the sixth study's ≈ 135,000 pairs ≈ 2.6 CPU-h at most (`smoke/timing/SYNTHETIC_pair_timing.txt`) |
| first import of scipy/sklearn in a fresh worker | 1–2 min (137 s measured) | the venv is on NFS; paid once per worker pool, not per map |

Job A ≈ 100–170 CPU-h (3–5 h on 38+ CPUs, peak about 60 GB, about 29 GB of disk); Job B ≈ 75–105 CPU-h,
up to about 100 GB, its longest single task a 3.5 h sketch-map map.

```
calculations/ML_dscribe/CoGe/
  labels/geometry_labels.csv, geometry_labels.json
  rc6.0_n8_l6_s0.5/interface_r8_both_t4_no_boundary/   krr_summary.csv, krr_by_plane.csv, krr_regression.csv, …
  selection_benchmark_v2/                               as selection_benchmark/, plus grain_class@<plane> and r2/mae rows
  representation_study/phase1/, phase2/, thickness/     results.csv, maps.npz, summary.json
  settings_grid/                                        results.csv, maps.npz, summary.json
```

## Limits

- One thickness (4x) per GB. Nothing here has been run on relaxed structures.
- "Distinct" means distinct as a whole cell, which holds two interfaces. Two cells can share one
  interface and differ in the other; the single-interface count addresses that with vectors only.
- The structure test covers rigid motions that keep the interface normal along x and rotate in-plane
  by multiples of 60 or 90 degrees. That is every motion compatible with a rectangular, square or
  hexagonal in-plane lattice aligned with the cell axes; an in-plane lattice with other symmetry
  directions would not be covered. Position tolerance 0.02 A; it must stay below half the shortest
  same-species distance. That distance is only 0.07 A in the unrelaxed catalogue, so do not raise
  `--position-tol` above 0.035 (the script warns).
- The slab test does not allow a shift along the normal, so one interface with its dividing plane
  drawn at two different heights would count as two.
- A vector that is a mean over atoms can coincide for different structures in principle. None of
  the 282 groups is such a case at these settings.
- The clean agreement between vectors and structures holds for the default vector. With
  `--radius 4` or `--kind nonbulk`, some cells that are the same structure get vectors up to 1.8e-3
  apart (an atom falls just inside the cut in one copy and just outside in the other), which is
  further than the closest different structures. For those vectors no tolerance gives a clean count
  (1122 and 1119 distinct at 1e-4); use the census column `structure_class` instead.
- KRR: one kernel family and a small grid. The ceiling column bounds any model on these vectors
  from above; a low score means this model did not find the label, not that the vector lacks it.
- `first_op` and `rot_set` reflect the generator's listing, not physics; do not read much into them.
