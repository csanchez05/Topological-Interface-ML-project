# Uni-HamGNN in ML_Interface_Project

Project-local CPU installation of [Uni-HamGNN](https://hamgnn.readthedocs.io/en/latest/user_guide/uni_hamgnn.html),
including its official pretrained model. The supporting HamGNN 2.3.0 source
is pinned to commit `64dba1dbf445ccfa2683f560059c3fbca0df624a` of
[QuantumLab-ZY/HamGNN](https://github.com/QuantumLab-ZY/HamGNN).

The environment uses Python 3.10, PyTorch 2.5.0+cpu, PyTorch Geometric 2.6.1,
matching CPU extension wheels, e3nn 0.5.0, and Lightning 2.5.x. NumPy 1.26.4,
SciPy 1.14.1, and pymatgen 2024.6.10 provide Python 3.10-compatible wheels.
The source includes compatibility handling for the published Uni-HamGNN 2.1 model.

## Layout

- `env.sh`: Bash activation, project paths, local caches, and thread settings.
- `setup.sh`: repeatable installation of the pinned source, packages, and model.
- `requirements.txt`: direct Python dependencies and compatibility constraints.
- `requirements-lock.txt`: exact resolved Python package versions.
- `setup.log`: installation output, appended on each installer invocation.
- `source/`: official HamGNN checkout, installed as an editable package.
- `.venv/`: independent Python virtual environment.
- `models/uni-hamgnn_2_1.pkl`: the official pretrained model.
- `.bootstrap/`, `.cache/`, `.tmp/`: downloads, dependency caches, and scratch files.
- `../../calculations/uni-hamgnn/`: future calculation inputs, configurations, and outputs.

## Install or reproduce

From the project root:

```bash
bash src/uni-hamgnn/setup.sh
```

The installer creates a virtual environment using `/usr/bin/python3.10` and
bootstraps pip from its official wheel because the system Python lacks
`ensurepip`. It installs the editable source and dependencies, reusing the lock
file when present. A source checkout at another revision is preserved and
requires deliberate correction before installation can continue.

Only `uni-hamgnn_2_1.pkl.zip` is downloaded from
[Zenodo record 17239078](https://zenodo.org/records/17239078), approximately
204 MB compressed. The archive's published MD5 is
`7f99cba551b4a0cbb84827cfe359cdac`. The model is extracted without being loaded
or evaluated, and subsequent installations keep the existing model file.

If the project moves, recreate `.venv` with the installer; virtual environments
and editable package registrations use absolute paths.

## Activate and prepare future runs

In Bash, from the project root:

```bash
source src/uni-hamgnn/env.sh
cd "$UNIHAMGNN_CALCULATIONS_DIR"
```

`UNIHAMGNN_DIR` points to the source checkout. `UNIHAMGNN_MODEL` points to the
pretrained model. All caches stay under this setup directory. Numerical
libraries use `SLURM_CPUS_PER_TASK` threads, or one outside an allocation.

For a future run, create `Input.yaml` in a calculation subdirectory with the
model path given by `UNIHAMGNN_MODEL`, prepared graph-file paths, and an output
directory under `calculations/uni-hamgnn`. YAML strings do not automatically
expand shell variables: use the actual model path in the configuration.

The upstream configuration keys are `model_pkl_path`, `non_soc_data_dir`,
`soc_data_dir`, `output_dir`, `device`, and `calculate_mae`. Despite their
names, the two data fields point to `graph_data.npz` files. Set `device: cpu`
for this installation and `calculate_mae: false` unless reference Hamiltonians
are available. SOC prediction needs both non-SOC and SOC graph files.

The upstream prediction command, to use only when a calculation is requested, is:

```bash
python "$UNIHAMGNN_DIR/Uni-HamGNN/Uni-HamiltonianPredictor.py" --config Input.yaml
```

HamGNN also installs the upstream `HamGNN2.0`, `band_cal`, `graph_data_gen`, and
`poscar2openmx` commands. The graph-preparation workflow additionally needs
external OpenMX preprocessing executables, which are outside this installation.
Use Slurm for future calculations on this cluster and keep their inputs,
job scripts, and results under the calculation directory.

## Installation scope

This setup includes Python software and the pretrained model. Julia, native
OpenMX builds, optional CUDA backends, and example datasets are excluded.
The bundled MACE and NequIP modules are supplied by the HamGNN source checkout.

The installer runs no tests, import checks, model loading, training, inference,
calculations, or scheduler submissions. The calculation directory is initially
empty.

## Upstream references

- [Uni-HamGNN documentation](https://hamgnn.readthedocs.io/en/latest/user_guide/uni_hamgnn.html)
- [Pinned source and Uni-HamGNN instructions](https://github.com/QuantumLab-ZY/HamGNN/tree/64dba1dbf445ccfa2683f560059c3fbca0df624a/Uni-HamGNN)
- [Pretrained model release](https://zenodo.org/records/17239078)
- [PyTorch Geometric wheels for PyTorch 2.5 CPU](https://data.pyg.org/whl/torch-2.5.0+cpu.html)
