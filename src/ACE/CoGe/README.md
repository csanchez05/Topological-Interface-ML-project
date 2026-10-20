# ACE descriptors for B20 CoGe (Stage 1, steps 1.1–1.3, for the ACE vs SOAP comparison)

Scripts that turn the bulk cells and the grain-boundary (GB) catalogue into ACE vectors with
featomic. They mirror `src/dscribe/CoGe/` (SOAP) file for file, and import its structure list,
POSCAR reader, plane assignment, pooling and writers, so both descriptors average exactly the same
atoms. Each run writes **one array file plus one index table**. Environment: `source src/mlenv/env.sh`.

## ACE vs SOAP in one paragraph

ACE (atomic cluster expansion, Drautz 2019) is a framework, not a single descriptor. It expands the
neighbour density of each atom once, A_nlm = Σ_j R_nl(r_ij) Y_lm(r̂_ij), and multiplies ν copies of
it, coupled to rotation invariants. ν counts the neighbours in each term, so ν = 1 is 2-body,
ν = 2 is 3-body and ν = 3 is 4-body. **SOAP's power spectrum is exactly the ν = 2 term** (checked:
`test_nu2_is_the_soap_power_spectrum`). ν = 3 adds 4-body terms, and among them the couplings with odd
l1 + l2 + l3. These are pseudoscalars that change sign under inversion, so they tell the two B20
enantiomers apart; SOAP cannot.

## Files

| File | Step | What it does |
|---|---|---|
| `ace_common.py` | – | Shared code: ACE settings and tag, featomic `Calculator` (per-atom blocks), loaders `load_bulk` / `load_gb` (block choice), `assemble` / `load_gb_vectors` / `load_bulk_vectors` (blocks joined, pristine-norm scaling) |
| `bulk/bulk_descriptor.py` | 1.1 | ACE for the pristine cell, the 3 orientation references and the 1,541 transformed bulk cells |
| `bulk/check_bulk_invariance.py` | 1.2 | Even blocks must equal pristine; the odd block must equal ±pristine, with the sign given by the handedness label. Exits 1 on any failure |
| `gb/gb_descriptor.py` | 1.3 | Global and interface-centred vectors for every GB |
| `gb/check_gb.py` | 1.3 | Checks on the GB output (thickness consistency, edge vs mid, close contacts, missing species, handedness report) |
| `run_ace.slurm` | – | Runs the four scripts in order on a compute node |
| `test_ace_common.py` | – | Unit tests: rotation, inversion, chunking, ν2 = featomic λ-SOAP, tags, round trip, scaling |
| `covariant/covariant_ace.py` | – | Covariant ACE (λ = 1, 2 blocks of ν1–ν3): `CovariantCalculator`, streaming averages, loaders `load_rows` / `bulk_averages` (pristine-atom scaling) |
| `covariant/covariant_descriptor.py` | 1.1, 1.3 | Covariant ACE for the bulk cells and a list of GB structures (`--structures`) |
| `covariant/check_covariant.py` | 1.2, 1.3 | Rotation, inversion, bulk averages ≈ 0, thickness consistency, coinciding copies |
| `covariant/test_covariant_ace.py` | – | Unit tests: rotation, inversion, chunking, ν2 = featomic λ-SOAP, averages, B20 bulk averages vanish, round trip |
| `ace_grid.py` | – | The ACE runs of the descriptor screens (27 one-knob settings at 8x, the reference at radii 6/8/10, covariant ACE), `--parallel` tasks at a time, with skip-if-done |
| `run_ace_grid.slurm` | – | `ace_grid.py` as one qcg0 job |
| `run_ace_screens.slurm` | – | The descriptor screens on the grid: `descriptor_sensitivity.py`, `selection_benchmark.py`, `plot_descriptor_study.py` with `--descriptor-type ace`, as one qcg0 job |

Do not add an `__init__.py` anywhere under `src/ACE/` (as for `src/dscribe/`).
`ace_common.py` imports the SOAP step scripts by file path under other module names
(`soap_gb_descriptor`, `soap_bulk_descriptor`), because the ACE scripts have the same file names.

## Running

Full run (about 13 CPU-h; job 12969 took 33 min on qcg0 with 96 workers, 13 of them spent on the worker
start that is now fixed, see Known points):

```bash
sbatch src/ACE/CoGe/run_ace.slurm
```

Other settings are passed through the environment (`R_CUT`, `N_MAX`, `L_MAX`, `SIGMA`, `SMOOTHING`,
`NU_MAX`, `N3`, `L3`, `WEIGHTING`, `RADII`, `OVERWRITE=1`), e.g.
`N3=6 L3=4 sbatch src/ACE/CoGe/run_ace.slurm` or `R_CUT=8 WEIGHTING=pow sbatch src/ACE/CoGe/run_ace.slurm`.

By hand, from the project root, in a compute-node shell (not on the login node, see `CLAUDE.md`):

```bash
srun -p general --exclude=qcg-docker -c 16 --mem=32G -t 01:00:00 --pty bash
source src/mlenv/env.sh
python -B src/ACE/CoGe/bulk/bulk_descriptor.py --workers 16
python -B src/ACE/CoGe/bulk/check_bulk_invariance.py
python -B src/ACE/CoGe/gb/gb_descriptor.py --workers 16
python -B src/ACE/CoGe/gb/check_gb.py
python -B -m unittest discover -s src/ACE/CoGe -p 'test_*.py'
```

`gb_descriptor.py --dry-run` prints the structure count, the vector widths, the output size and the
time per atom of the pristine cell (an upper bound: small cells are dominated by per-call overhead).
Filters as for SOAP: `--planes`, `--thickness`, `--limit N`; a filtered run gets its own folder
(`_p…`, `_t…`, `_lim…`), and so does a bulk run with `--planes` or `--limit`.
`--chunk` (default 128) sets how many atoms go through featomic per call: it bounds the memory and
does not change the numbers (tested). Skipped structures go to `skipped.csv`; only the empty
`CoGe_100_GBs/Interface/sym00016/9x/POSCAR` is expected.

## ACE settings

| Option | Default | Note |
|---|---|---|
| `--r-cut` | 6.0 Å | as SOAP |
| `--n-max`, `--l-max` | 8, 6 | radial and angular size of the expansion behind ν1 and ν2 (as SOAP) |
| `--sigma` | 0.5 Å | Gaussian width of each atom, as SOAP |
| `--density` | gaussian | `dirac` (textbook ACE, point atoms) is refused: featomic 0.6.7 cannot expand a Dirac density on the GTO basis, neither built in nor with its spline (NaN derivatives; found 2026-10-13, no run ever used it) |
| `--weighting` | none | `pow`: neighbour density weighted by w(r) = 1 / (1 + (r / r0)^m), `--weighting-r0` 3 Å, `--weighting-m` 4; featomic's Willatt2018 scaling, the same function as the SOAP study's DScribe `pow` (c = d = 1); tag suffix `_wpow-r3-c1-d1-m4`. DScribe's `poly` has no featomic counterpart |
| `--smoothing` | 0.5 Å | shifted-cosine cutoff smoothing (as the covariant run); other values add `_sm<w>` to the tag |
| `--nu-max` | 3 | 2 stops at the SOAP power spectrum (blocks ν1, ν2 only) |
| `--n3`, `--l3` | 4, 4 | a smaller expansion for ν3: the number of 4-body terms grows as n³·l³ |

GTO radial basis throughout. Folder name (tag): the SOAP tag of the same density plus the body-order
suffix, e.g. `rc6.0_n8_l6_s0.5_nu3-n4-l4` (`ace_common.ace_tag`). The first part means that its ν2 block
is the power spectrum of SOAP `rc6.0_n8_l6_s0.5`, computed by featomic instead of DScribe, so the
numbers are not the same as the DScribe vectors (different radial basis normalisation); the
information is.

## Blocks

| Block | Body order | Under inversion | Features per atom (default) | Entries per vector |
|---|---|---|---|---|
| `nu1` | 2 | unchanged | 16 (2 neighbour species × 8 radial) | 32 |
| `nu2` | 3 (SOAP power spectrum) | unchanged | 1,792 (7 l × 16 × 16) | 3,584 |
| `nu3_even` | 4 | unchanged | 7,168 | 14,336 |
| `nu3_odd` | 4 | **sign flips** | 4,096 | 8,192 |

The neighbour species is part of the properties, so the products mix Co and Ge neighbours. The centre
species is kept apart: every vector is `[Co-centred mean | Ge-centred mean]`, as for SOAP. featomic's
`DensityCorrelations` runs with `skip_redundant=True` (each distinct coupling once). Property labels are
stored with the arrays (`properties_<block>`, `property_names_<block>`); every structure is checked
against the labels of the pristine cell.

## Outputs

All descriptor files live in `calculations/descriptors/ACE/` (its `README.md`). One run fills two
folders with the same tag:

### `standard/global/<tag>/`
- `global_ace.npz`: `X_<block>` (N × 2P, float32), the mean over every atom of each GB cell; labels.
- `index.csv`: one row per GB structure (shared with `both_interfaces/`): plane, sym, thickness,
  `family`, atom counts, cell lengths, area, closest pair, `atom_norm_<block>` (mean length of one atom's
  vector), source.
- `bulk_ace.npz`: `X_<block>`, one row per bulk cell (float64); `bulk_index.csv`: kind (pristine /
  reference / transformed), plane, sym, cell, closest pair, `scale_corrected`,
  `within_species_spread_<block>`, `atom_norm_<block>`, source.
- `bulk_invariance.csv`, `bulk_invariance.json`: step 1.2 report (per block; odd sign vs label).
- `bulk_settings.json` (bulk run), `settings.json` (GB run, with the featomic hypers), `skipped.csv`.

### `standard/both_interfaces/<tag>/`
- `interface_ace.npz` (float32): `X_<block>_r8` (N × 2 planes (edge, mid) × 2P) for every radius,
  `n_r8` (N × 2 planes × 2 species, the same for every block), `radii`, `plane_names`, `species`, labels.
- `index.csv`, `settings.json`, `skipped.csv`; `checks.csv`, `checks.json` from `check_gb.py`.

No `X_nonbulk`: SOAP's nonbulk test compares each atom with pristine bulk, and through the odd block
every atom of a mirrored grain would count as disturbed.

### Loading

```python
import sys; sys.path.insert(0, "src/ACE/CoGe")
import ace_common as ac

blocks, index = ac.load_gb("rc6.0_n8_l6_s0.5_nu3-n4-l4", radius=8.0)            # {block: X}, both interfaces
X, index = ac.load_gb_vectors("rc6.0_n8_l6_s0.5_nu3-n4-l4", radius=8.0)        # all blocks joined
X_odd, _ = ac.load_gb_vectors(tag, radius=8.0, blocks=["nu3_odd"], scaling="none")
X_bulk, bulk_index = ac.load_bulk_vectors("rc6.0_n8_l6_s0.5_nu3-n4-l4")
```

`scaling="pristine"` (default) divides each block by the length of the pristine cell's vector of that
block, so the pristine cell has length 1 in every block and no body order dominates by size alone
(raw lengths differ by orders of magnitude between blocks). `scaling="none"` joins the raw blocks.

## Smoke test (2026-10-13, qcg3-0, 16 cores)

Record and outputs: `archive/ace_20261013/`. 10 bulk cells (pristine, 3 references, the first 2 cells
per plane), 54 GB structures (first 2 sym folders per plane, all thicknesses, 20,736 atoms):
- Bulk: even blocks equal pristine to ≤ 1.6e-7 and the odd block equals ±pristine to ≤ 9.5e-7; the sign
  matched the handedness label on 6 of 6 cells (3 `opposite`). The 111 deviations are the same for
  `ref_111` and both sym cells: they come from the four-decimal coordinates of the 111 cells, and sit
  just under the 1e-6 tolerance.
- GB: thickness consistency at radius 8 ≤ 6e-5 in every block; no slab misses a species. The global odd
  vector of the thickest cell, over pristine: +0.95 for the 3 `same` GBs, 0.000 for the 3 `opposite` ones
  (half of the cell is each enantiomer).
- Cost: 13 s for 20,736 atoms on 16 workers, about 10 CPU-ms per atom; peak memory 0.47 GB per process.

## Full run (2026-10-13, job 12969)

Default settings, the whole catalogue; outputs and numbers in `calculations/descriptors/ACE/README.md`.
Step 1.2 passes on all 1,545 bulk cells, and the odd-block sign matches the handedness label on all 1,541
transformed cells. The step 1.3 checks pass (thickness consistency <= 6.9e-5 in every block).

## Covariant ACE (`covariant/`, for the representation study)

The λ ≥ 1 companions of the invariant blocks, from the same two expansions (featomic `DensityCorrelations`
with selected keys λ = 1, 2): `nu1_{polar,tensor}`, `nu2_{polar,axial,tensor,pseudotensor}` and
`nu3_{polar,axial,tensor,pseudotensor}`. polar = λ 1 σ +1 (a vector, flips under inversion), axial = λ 1 σ −1
(a vector unchanged by inversion, like a spin), tensor = λ 2 σ +1, pseudotensor = λ 2 σ −1; under inversion a
block picks up σ(−1)^λ. ν1 has only σ = +1. They are the ACE counterpart of covariant SOAP
(`src/dscribe/CoGe/covariant/`): `nu2_*` equal featomic's λ-SOAP blocks (test `test_nu2_is_the_covariant_soap`).
λ = 1 components are stored as x, y, z, λ = 2 as m = −2 … 2.

Averaging as for the invariant blocks, `[Co mean | Ge mean]` per component. The atoms go through featomic
`--chunk` at a time and only running sums are kept. One vector holds 569,600 entries at the default
n3 = l3 = 4 (2.3 MB in float32), so the whole catalogue would take about 95 GB; `--structures` takes a CSV of
`family, thickness` pairs instead. The cell average of every λ ≥ 1 block vanishes in bulk (point group 23), so
the loaders scale each block by the mean length of one pristine atom's block vector (`atom_norm_<block>` of
the pristine row of `bulk_index.csv`).

```bash
python -B src/ACE/CoGe/covariant/covariant_descriptor.py --structures <list.csv> --workers 88
python -B src/ACE/CoGe/covariant/check_covariant.py --run-dir rc6.0_n8_l6_s0.5_nu3-n4-l4_lam1-2
python -B -m unittest discover -s src/ACE/CoGe/covariant -p 'test_*.py'
```

Smoke test (2026-10-13, qcg0, 6 workers; `archive/ace_20261013/smoke/covariant/`): 7 bulk cells and 6 GB
cells. Rotation errors ≤ 7e-14 and inversion errors ≤ 1e-14 on a 4x cell of each plane; bulk averages ≤ 1e-6
of one atom's vector in every block; thickness consistency ≤ 1.8e-3. Cost: about 18 CPU-ms per atom (the
invariant blocks: about 3 at the same settings), and about 1 GB per worker at the start.

**Worker memory grows: featomic leaks.** `DensityCorrelations.compute` (featomic 0.6.7) leaves the arrays it
hands to metatensor registered in `metatensor._data._array._KNOWN_ARRAY_WRAPPERS` after every call, and `gc`
does not free them (measured on qcg2-2, 2026-10-13: about 3,100 arrays and 60-160 MB per 8x structure for
the covariant blocks; about 1,400 small arrays per structure for the invariant blocks, whose workers reach
only 1.3-1.6 GB over an 8x run). In job 13012 the 13 covariant workers had grown to 24 GB each after about
150 structures (about 345 GB in all against the job's 160 GB; Slurm does not enforce `--mem` on this
cluster), so I stopped that step at 2,000 of 3,940 structures. `covariant_descriptor.py` now replaces each
worker after `--max-tasks-per-worker` GB structures (default 30, so at most about 5 GB per worker); the
bulk cells run first in a pool of their own that keeps its workers. Each replacement costs one start-up
(about 60 s). Replaced workers give bit-identical arrays (test of 2026-10-13, 38 structures and 10 bulk cells,
recycling every 2 and every 10 against never: `archive/ace_20261013/smoke/recycle/`).

## Settings grid (`ace_grid.py`, for the descriptor screens)

The ACE counterpart of the descriptor runs of `src/ML_dscribe_scripts/descriptor_grid.py`. One knob at a time
away from the reference `rc6.0_n8_l6_s0.5_nu3-n4-l4` (job 12969, all thicknesses): r_cut 4/5/8/10/12,
n_max 4/6/10/12, l_max 2/4/8/10 (ν1, ν2 expansion), σ 0.3/0.75/1.0, n3 2/3/5/6 and l3 2/3/5/6 (ν3 expansion),
pow weighting at r_cut 6, 8, 12. These runs cover the 1,541 8x structures only, named `<tag>_t8` (the screens
read the 1,112 distinct GBs at 8x; the thickness screen uses the full reference run). Further tasks:
- `region radii`: the reference setting at 8x with interface radii 6, 8 and 10 Å (`<tag>_t8`), for the
  region r6 and r10 vectors.
- `covariant`: covariant ACE at the reference setting on every GB at 8x plus the 300 GBs of the
  representation study's thickness check at all 9 thicknesses
  (`calculations/descriptors/ACE/covariant/structure_lists/t8-thick300.csv`, made with
  `representation_study.gb_table` and the same random draw, seed 11). Name `rc6.0_n8_l6_s0.5_nu3-n4-l4_lam1-2`.

Each setting gets bulk ACE, the bulk invariance check and GB ACE. Finished runs are skipped, the checks
always rerun, and a failed check does not stop the job (it exits 1 and lists it). Logs:
`calculations/descriptors/ACE/logs/grid/<task>.log` and `status.tsv` there.

`--parallel K` (Slurm: `PARALLEL`, default 3) runs K tasks at once with `--workers / K` workers each, the
covariant task first. Every step is a new Python process that imports numpy, featomic and metatensor
(about 840 modules) over NFS. On 2026-10-13, with qcg0 busy (four jobs reading and writing large files),
`python -X importtime` there showed about 0.2 s per module whatever its size (numpy about 20 s, the whole
stack about 60 s per process); on idle qcg2-2 a cold import took 43 s and a repeat 1 s. So on a busy node a
step spends minutes starting up (smoke steps on a few structures: 150-370 s; the full job's bulk steps,
about 12 s of compute: 505 s). With K > 1 that start-up overlaps the other tasks' compute. A node-local copy
of the venv does not help: copying its files pays the same per-file latency (`cp -a` moved 77 of 399 MB in
15 min).

```bash
python -B src/ACE/CoGe/ace_grid.py --dry-run          # tasks, state, commands
sbatch src/ACE/CoGe/run_ace_grid.slurm                 # all tasks; ONLY='covariant "n3 6"' for a subset
sbatch --cpus-per-task=40 --mem=160G src/ACE/CoGe/run_ace_grid.slurm   # when qcg0 is shared
```

Smoke test (2026-10-13, qcg0, `--limit 1`, `archive/ace_20261013/smoke/grid/`): every step runs. The bulk
check of n3 = 2 fails on the 111 cells by rounding: `ref_111` (pristine, only turned) differs from pristine by
1.4e-6 in `nu3_odd` (tolerance 1e-6); the 111 POSCARs have four-decimal coordinates (within-species spread
1.2e-4), and the smaller ν3 basis makes the odd block small enough for that noise to show. Such failures are
reported, not hidden by a looser tolerance.

`l3 = 2` fails its bulk check on every plane for a different reason: there the pristine `nu3_odd` block is
zero (length 3e-19; the rounded 111 cells 7e-9). Every B20 atom sits on a 3-fold axis, and up to l = 2 the
only C3-invariant harmonics are the m = 0 ones about that axis, so the density seen through l <= 2 is
axially symmetric and no inversion-odd invariant survives in bulk. In an `l3 = 2` GB vector the odd block
comes only from interface atoms, whose site symmetry is broken. The check's relative differences are then
over a zero vector and meaningless, and `ace_common.assemble` refuses to divide a block by a pristine length
below `VANISHING_NORM` (1e-6): such a run is loaded without `nu3_odd` (`--ace-blocks`) or unscaled.

## Descriptor screens (`run_ace_screens.slurm`)

The SOAP screens for ACE: `descriptor_sensitivity.py`, `selection_benchmark.py` and `plot_descriptor_study.py`
from `src/ML_dscribe_scripts/` with `--descriptor-type ace` (settings and figures in that folder's README).
They read the grid runs above, the reference run, the ACE KRR run with census folds
(`calculations/ML_ACE/CoGe/rc6.0_n8_l6_s0.5_nu3-n4-l4/interface_r8_both_t4_no_boundary`, for the benchmark's
KRR settings) and the ACE FPS/median lists in `plots/CoGe/dim-reduction/ACE/`. Not made for ACE:
`plane100_dmin1.8`, the settings grid and `vector_sizes` (user, 2026-10-13); `representation_response` needs
covariant ACE and waits for it.

```bash
sbatch --dependency=afterany:<grid job> src/ACE/CoGe/run_ace_screens.slurm   # 40 cores, after the grid
sbatch --cpus-per-task=96 --mem=400G src/ACE/CoGe/run_ace_screens.slurm     # whole qcg0 when it is free
```

Smoke test (2026-10-13, qcg2-2, `archive/ace_20261013/smoke/screens/`): 9 settings with 2 seeds, the benchmark
with 2 test sets and 2 draws, then every ACE section with the title prefix "SMOKE TEST".

## Known points

- `gb_descriptor.py` hands the expected property labels to its workers through a temporary file. Job 12969
  still passed them directly; at about 0.4 MB they overfilled the pipe that starts each spawned worker, so
  the main process started the 96 workers one at a time, each after its imports (about 8 s, 13 min in
  all). With the file the workers start together; the arrays of the 54-structure smoke test are
  bit-identical before and after the change.

- The odd block of a GB whose grains are mirror images can vanish by symmetry (e.g. `100/sym00000`, the
  inversion). `check_gb.py` therefore divides differences by the pristine block length, not by the
  vector itself.
- The data quirks of the catalogue (scale on Å vectors, the empty 9x POSCAR, unrelaxed close pairs) are
  handled by the shared SOAP code; see `src/dscribe/CoGe/README.md`.
