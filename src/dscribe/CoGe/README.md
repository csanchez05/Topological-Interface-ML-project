# SOAP descriptors for B20 CoGe (Stage 1, steps 1.1–1.3)

Scripts that turn the bulk cells and the grain-boundary (GB) catalogue into
SOAP vectors with DScribe. Each run writes **one array file plus one index
table**, not one file per structure. Environment: `source src/mlenv/env.sh`.

## Files

| File | Step | What it does |
|---|---|---|
| `soap_common.py` | – | Shared code: POSCAR reader, SOAP settings, pooling, writers, `load_bulk` / `load_gb` |
| `bulk/bulk_descriptor.py` | 1.1 | SOAP for the pristine cell and all 1,541 transformed bulk cells |
| `bulk/check_bulk_invariance.py` | 1.2 | Compares every bulk vector with pristine |
| `gb/gb_descriptor.py` | 1.3 | Global and interface-centred vectors for every GB |
| `gb/check_gb.py` | 1.3 | Checks on the GB output |
| `run_soap.slurm` | – | Runs the four scripts in order on a compute node |
| `test_soap_common.py` | – | Unit tests |
| `covariant/covariant_descriptor.py` | – | Covariant (λ-) SOAP with featomic, λ = 0, 1, 2 (`--blocks` chooses); fifth and sixth descriptor studies, kept for Stage 3 |
| `covariant/check_covariant.py` | – | Rotation, inversion, bulk and thickness checks of the covariant output; `--compare-with` checks a new run against an old one |
| `covariant/test_covariant.py` | – | Unit tests (rotation and inversion of a small random cell, λ = 1 and 2; the Gram-matrix shortcut) |
| `gb/global_descriptor.py` | – | Legacy reference script (old paths, max-pooling). Not used. |

Do not add an `__init__.py` anywhere under `src/dscribe/`: the folder shares
its name with the installed `dscribe` package.

## Running

Full run (about 12 core-minutes at the default settings):

```bash
sbatch src/dscribe/CoGe/run_soap.slurm
```

Other settings are passed through the environment:

```bash
R_CUT=8 N_MAX=10 L_MAX=8 RADII="4 6 8 10 12" sbatch src/dscribe/CoGe/run_soap.slurm
OVERWRITE=1 RADII="3 5 7 9" sbatch src/dscribe/CoGe/run_soap.slurm   # same SOAP settings, new radii
```

By hand, from the project root, in a compute-node shell (not on the login node, see `CLAUDE.md`):

```bash
srun -p general --exclude=qcg-docker -c 16 --mem=16G -t 01:00:00 --pty bash
source src/mlenv/env.sh
python -B src/dscribe/CoGe/bulk/bulk_descriptor.py
python -B src/dscribe/CoGe/bulk/check_bulk_invariance.py
python -B src/dscribe/CoGe/gb/gb_descriptor.py --radii 4 6 8 10 --workers 16
python -B src/dscribe/CoGe/gb/check_gb.py
python -B -m unittest discover -s src/dscribe/CoGe -p 'test_*.py'
python -B -m unittest discover -s src/dscribe/CoGe/covariant -p 'test_*.py'
```

The descriptor study's many settings are not run one by one: `src/ML_dscribe_scripts/descriptor_grid.py`
runs all of them (and the covariant run) as one whole-node job.

`gb_descriptor.py` filters: `--planes 100 110`, `--thickness 4` (one or more of
4–12), `--limit N` (first N sym folders per plane). A filtered run goes to its
own folder, so it cannot overwrite a full run. `--csv 8` also writes the
radius-8 vectors as a plain CSV.

## SOAP settings

| Option | Default | Note |
|---|---|---|
| `--r-cut` | 6.0 Å | guide: repeat at 8 and 10 Å |
| `--n-max` | 8 | |
| `--l-max` | 6 | |
| `--sigma` | 0.5 Å | DScribe's own default is 1.0 Å; the legacy script used 0.3 Å |
| `--rbf` | gto | |
| `--weighting` | none | JSON, passed to DScribe unchanged (DScribe gets a copy: it adds keys to the dict it is given) |

Each combination gets its own folder name, e.g. `rc6.0_n8_l6_s0.5`. A weighting is spelled out:
`{"function": "pow", "r0": 3, "c": 1, "d": 1, "m": 4}` gives `rc6.0_n8_l6_s0.5_wpow-r3-c1-d1-m4`.
A filtered GB run (`--planes`, `--thickness`, `--limit`) gets a suffix (`_t8`, `_lim2`).

## Outputs

All descriptor files live in `calculations/descriptors/dscribe/` (layout and tag rules in its
`README.md`). One SOAP run fills two folders with the same tag:

### `standard/global/<tag>/`
- `global_soap.npz`: `X` (N × 2F, float32), the mean over every atom of each GB cell; `species`.
- `index.csv`: one row per GB structure, same order as the arrays (shared with `both_interfaces/`).
  `family` (e.g. `100/sym00000`) is the group key that keeps the nine thicknesses of one GB together
  in train/test splits.
- `bulk_soap.npz`: `X`, one row per bulk cell (float64); `bulk_index.csv`: the cell behind each row:
  kind (pristine / reference / transformed), plane, sym, cell lengths, closest atom pair,
  `scale_corrected`, within-species spread, source path.
- `bulk_invariance.csv`, `bulk_invariance.json`: step 1.2 report.
- `bulk_settings.json` (bulk run), `settings.json` (GB run), `skipped.csv`.

### `standard/both_interfaces/<tag>/`
- `interface_soap.npz` (float32). One pair of arrays per radius, so loading one radius reads only those:

  | Array | Shape | Meaning |
  |---|---|---|
  | `X_r8` (one per radius: `X_r4`, `X_r6`, …) | N × 2 planes × 2F | mean over atoms within 8 Å of that plane (edge, mid) |
  | `n_r8` | N × 2 planes × 2 species | atoms behind each vector |
  | `radii` | radii | the radii, in Å |
  | `X_nonbulk` | N × 2 planes × 2F | mean over atoms whose SOAP differs from bulk (guide 2.4) |
  | `n_nonbulk` | N × 2 planes × 2 species | atoms behind each vector |
  | `n_nonbulk_within` | N × radii | disturbed atoms inside each radius |
  | `plane_names`, `species` | | `edge, mid`; `Co, Ge` |

- `index.csv`, `settings.json`, `skipped.csv`.
- `checks.csv`, `checks.json`: output of `check_gb.py`.

Every vector is `[Co-centred mean | Ge-centred mean]`, so its length is 2F
(F = SOAP features per atom; 952 at the default settings).

### `covariant/{global,both_interfaces}/<run>/`
Written by `covariant/covariant_descriptor.py` (featomic `EquivariantPowerSpectrum`), same averaging:

| Block | λ, σ | Behaviour | Stored as |
|---|---|---|---|
| `invariant` | 0, +1 | unchanged by rotations, like standard SOAP | `X_invariant`: N × 2P₀ |
| `polar` | 1, +1 | turns like a vector, flips sign under inversion | `X_polar`: N × 3 (x, y, z) × 2P₁ |
| `axial` | 1, −1 | turns like a vector, unchanged by inversion (as a spin does) | `X_axial`: N × 3 × 2P₁ |
| `tensor` | 2, +1 | turns like a symmetric traceless rank-2 tensor, unchanged by inversion | `X_tensor`: N × 5 (m = −2 … 2) × 2P₂ |
| `pseudotensor` | 2, −1 | turns like `tensor`, flips sign under inversion | `X_pseudotensor`: N × 5 × 2P₂ |

Under inversion a block is multiplied by σ·(−1)^λ. `--blocks` chooses the blocks (default all five);
`settings.json` lists the ones present, and `check_covariant.py` skips the tests that need a missing
block (the settings-grid runs `<tag>_t8` have only the four λ ≥ 1 blocks and only 8x).

`global/` holds `covariant_soap.npz` (whole-cell means), `bulk_covariant.npz`, `bulk_index.csv`,
`index.csv`, `settings.json`. `both_interfaces/` holds `covariant_soap.npz` with
`X_<block>_r8` (N × 2 planes × …), `n_r8` and `radii`, plus `index.csv`, `settings.json`, `checks.json`
and `coinciding_copies.csv` from `check_covariant.py`. Not used for Stage 2 selection: a GB turned in
space gives a different covariant vector. The fifth and sixth descriptor studies map it only to measure how much
the kind of representation changes the maps (`src/ML_dscribe_scripts/representation_study.py`).

Use the polar and axial blocks for each interface on its own (`edge`, `mid`), not their mean: the two
interfaces of a cell can carry opposite vectors that cancel in the mean (seen in `100/sym00001`).
Where symmetry forces a vector to zero, the stored value is rounding noise of the POSCAR coordinates
(up to about 1e-5 of one atom's vector); `check_covariant.py` counts vectors below `--zero-tol`
(1e-3 of one atom's vector) as zero. Cost: about 38 CPU-ms per atom, about 48 CPU-h for the catalogue.

`check_covariant.py`, what it tests:
- *rotation*: λ = 1 blocks must turn by the same 3 × 3 matrix as the cell. featomic has no Wigner-D
  matrices, so λ = 2 is tested without one: each atom's P × P Gram matrix VᵀV (V = its 5 × P block;
  the product sums over m, which a rotation leaves unchanged) must not change, computed through a QR factorisation of the stacked
  before/after components (same norms as the direct P × P product, but 2C × 2C matrices: the direct
  product needed about 31 GB for 30 structures at l_max 4 and was killed); and the fitted 5 × 5
  transformation must be orthogonal and the same for every atom.
- *inversion*: every block picks up σ·(−1)^λ.
- *bulk*: the cell average of each λ ≥ 1 block over one atom's length (zero by the crystal's rotations);
  the invariant block of every transformed bulk cell equals pristine.
- *thickness* consistency and coinciding copies, as before.
- `--compare-with <run>`: every array shared with that run (resolved under `covariant/`) must agree to
  `--compare-tol` (default 1e-6) and the atom counts must be identical. The λ = 2 reference
  (`rc6.0_n8_l6_s0.5_lam2`, fifth study) must reproduce the λ ≤ 1 run this way before it replaces it;
  the smoke test (10 sym folders per plane, all thicknesses) was bit-identical.

### Loading

```python
import sys; sys.path.insert(0, "src/dscribe/CoGe")
import soap_common as sc

X_bulk, bulk_index = sc.load_bulk("rc6.0_n8_l6_s0.5")
X, index = sc.load_gb("rc6.0_n8_l6_s0.5", radius=8.0)                 # both interfaces
X_mid, _ = sc.load_gb("rc6.0_n8_l6_s0.5", radius=8.0, planes="mid")   # one interface
X_glob, _ = sc.load_gb("rc6.0_n8_l6_s0.5", kind="global")
X_dist, _ = sc.load_gb("rc6.0_n8_l6_s0.5", kind="nonbulk")
```

A tag is looked up under `standard/`; a folder of either region also works
(`sc.run_dir("global", ".../both_interfaces/<tag>")` gives the matching global folder).

## How the interface-centred vector is defined

- Every GB cell is orthogonal with the interface normal along x. Grain 1 fills the first half of the cell
  and grain 2 the second half, so the cell holds **two interfaces**: `mid` at x = Lx/2 and `edge` at x = 0,
  which is the same plane as x = Lx through the periodic boundary.
- An atom belongs to the plane it is nearer to. It is included at radius R when its distance along x to
  that plane is at most R. Atoms on both sides of each plane are included, and the edge plane picks up
  atoms from both ends of the cell.
- `planes="both"` is the count-weighted mean of the two interfaces. The two are stored separately because
  they are not always the same interface.
- If R exceeds Lx/4, the two slabs meet and the vector becomes the global one. `check_gb.py` leaves such
  thicknesses out of the thickness-consistency test for that radius.
- `X_nonbulk` uses the guide's section 2.4 definition instead of a radius: atoms whose SOAP differs from
  pristine bulk by more than `--bulk-tol` (relative, default 1e-3). The disturbance fades smoothly towards
  the cutoff, so the atom count depends on that tolerance; the atoms near the threshold barely change the mean.

## Known data quirks

- The 110 and 111 bulk POSCARs carry the scale 4.6294 on lattice vectors already in Å. `read_structure`
  ignores the scale in that case and records `scale_corrected = 1`.
- 111 coordinates are written to four decimals, so atoms of one species differ from each other by about
  1e-4 (relative) inside those cells. The averaged vectors still agree with pristine to about 1e-7.
- `CoGe_100_GBs/Interface/sym00016/9x/POSCAR` is empty and is skipped.
- The GB structures are unrelaxed, and many contain atom pairs far closer than the 2.40 Å bulk bond.
  `min_pair_distance` in `gb_index.csv` records this for every structure.
