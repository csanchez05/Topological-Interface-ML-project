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

Do not add an `__init__.py` anywhere under `src/ACE/` (as for `src/dscribe/`).
`ace_common.py` imports the SOAP step scripts by file path under other module names
(`soap_gb_descriptor`, `soap_bulk_descriptor`), because the ACE scripts have the same file names.

## Running

Full run (user-submitted; about 13 CPU-h, about 15 min on the default 64 cores):

```bash
sbatch src/ACE/CoGe/run_ace.slurm
```

Other settings are passed through the environment (`R_CUT`, `N_MAX`, `L_MAX`, `SIGMA`, `SMOOTHING`,
`DENSITY`, `NU_MAX`, `N3`, `L3`, `RADII`, `OVERWRITE=1`), e.g.
`N3=6 L3=4 sbatch src/ACE/CoGe/run_ace.slurm` or `DENSITY=dirac sbatch src/ACE/CoGe/run_ace.slurm`.

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
| `--density` | gaussian | `dirac` gives textbook ACE (point atoms, no smearing; the tag then reads `_dirac`) |
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

## Known points

- The odd block of a GB whose grains are mirror images can vanish by symmetry (e.g. `100/sym00000`, the
  inversion). `check_gb.py` therefore divides differences by the pristine block length, not by the
  vector itself.
- The data quirks of the catalogue (scale on Å vectors, the empty 9x POSCAR, unrelaxed close pairs) are
  handled by the shared SOAP code; see `src/dscribe/CoGe/README.md`.
