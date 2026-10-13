# mlenv in ML_Interface_Project

Project-local CPU environment for the SOAP structural map and interface
spin-texture work on B20 CoGe grain boundaries. It provides
[DScribe](https://singroup.github.io/dscribe/) for SOAP descriptors,
[scikit-learn](https://scikit-learn.org/) for the regression and
decomposition models, and [umap-learn](https://umap-learn.readthedocs.io/)
for the second low-dimensional view. It is independent of the MACE-H,
DeepH-E3, Uni-HamGNN and plotting environments.

Python 3.10 (system `/usr/bin/python3`). Direct dependencies are in
`requirements.txt`; exact resolved versions are written to
`requirements-lock.txt` on the first install.

## Layout

- `env.sh`: Bash activation, local caches and thread settings.
- `setup.sh`: installation; reuses `requirements-lock.txt` when present.
- `requirements.txt`: direct dependencies (dscribe, ase, scikit-learn,
  umap-learn, numpy, scipy, pandas, matplotlib, featomic, skmatter, pacmap).
  featomic 0.6.7 and skmatter 0.3.2 were added on 2026-10-03 with
  `pip install featomic==0.6.7 skmatter==0.3.2`; this only added packages
  (featomic, metatensor-core, metatensor-operations, wigners, ctypes-dlpack,
  skmatter) and changed no existing version.
  pacmap 0.9.1 was added on 2026-10-05 with `pip install pacmap==0.9.1`
  (for the fifth study's PaCMAP maps); it added pacmap and faiss-cpu 1.15.1
  and changed no existing version.
- `requirements-lock.txt`: exact installed versions.
- `.venv/`: isolated Python environment.
- `.bootstrap/`, `.cache/`: pip bootstrap wheel, pip/numba/matplotlib caches.

## Install and activate

From the project root:

```bash
bash src/mlenv/setup.sh 2>&1 | tee src/mlenv/install.log
source src/mlenv/env.sh
```

`env.sh` sets `OMP_NUM_THREADS` (and the BLAS/numba equivalents) to
`$SLURM_CPUS_PER_TASK` inside a Slurm allocation, otherwise to 1. Set
`OMP_NUM_THREADS` before sourcing to override it.

## Main components

- SOAP: `dscribe.descriptors.SOAP` (takes `ase.Atoms`; read POSCARs with
  `ase.io.read`).
- Kernel ridge regression: `sklearn.kernel_ridge.KernelRidge`.
- Random forest: `sklearn.ensemble.RandomForestRegressor` /
  `RandomForestClassifier`.
- Gaussian process regression: `sklearn.gaussian_process.GaussianProcessRegressor`.
- PCA: `sklearn.decomposition.PCA`; UMAP: `umap.UMAP`; PaCMAP: `pacmap.PaCMAP`.
- Covariant (lambda-) SOAP: `featomic.clebsch_gordan.EquivariantPowerSpectrum`
  on top of `featomic.SphericalExpansion`; results are metatensor TensorMaps.
- Feature-space reconstruction error (GFRE): `skmatter.metrics`.
- Farthest-point sampling is not built into scikit-learn; implement it on the
  SOAP vectors (e.g. with `sklearn.metrics.pairwise_distances`).
- Grouped train/test splits (keeping all thicknesses of one GB together):
  `sklearn.model_selection.GroupKFold` / `GroupShuffleSplit`.

## Location for work

Descriptor scripts: `src/dscribe/CoGe/`; analysis scripts: `src/ML_dscribe_scripts/`.
Descriptors: `calculations/descriptors/dscribe/`; analyses: `calculations/ML_dscribe/CoGe/`.
