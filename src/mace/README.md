# MACE-H in ML_Interface_Project

Project-local setup for **MACE-H 0.1.0**, the Hamiltonian prediction package
from [maurergroup/MACE-H](https://github.com/maurergroup/MACE-H), pinned to
commit `3d6248b715eef302a4fd674d52bcd69254596a51`.

The installation uses **Python 3.10**, **PyTorch 2.4.0+cpu**, **e3nn 0.5.1**,
matching CPU wheels for the PyTorch Geometric extensions, and **Julia 1.10.11**
for electronic-structure postprocessing.

## Layout

- `src/mace/env.sh`: environment activation and project-local cache paths.
- `src/mace/setup.sh`: reproducible installation from the pinned requirements.
- `src/mace/source/`: official MACE-H source, installed as the editable `maceh` package.
- `src/mace/.venv/`: isolated Python environment and dependencies.
- `src/mace/.cache/`: PyTorch, plotting and pip caches.
- `src/mace/.julia-runtime/`: local Julia executable and runtime libraries.
- `src/mace/.julia-depot/`: downloaded Julia packages and build artifacts.
- `src/mace/julia-project/`: Julia `Project.toml` and dependency `Manifest.toml`.
- `src/mace/.tmp/` and `.bootstrap/`: installation scratch files.
- `src/mace/requirements.txt`: top-level dependency versions.
- `src/mace/requirements-lock.txt`: exact installed dependency versions.
- `src/mace/setup.log`: installation output.
- `calculations/mace/`: location for future calculation inputs, runs and results.

## Install or reproduce

From the project root:

```bash
bash src/mace/setup.sh
```

The installer uses `requirements-lock.txt` when present. It creates a virtual
environment without changing system Python and bootstraps pip from a checksum
verified PyPI wheel because this cluster's Python lacks `ensurepip`. The source
revision is pinned in `setup.sh`; the requirements lock includes `-e ./source`
so the package remains linked to this local checkout.

The installer also downloads Julia, installs the dependencies used by the
upstream postprocessing/preprocessing scripts, and builds PyCall against the
local Python interpreter. Julia package versions are recorded in
`julia-project/Manifest.toml`. Downloads and package builds are installation
steps; the installer does not run tests, training or calculations.

Network access is needed for GitHub, PyPI, the PyTorch/PyG wheel hosts, and Julia's
download/package servers. Installation output is appended to `setup.log`.

The environment is pinned for Python 3.10 on Linux x86_64. If the project moves
to a different path or machine, recreate `.venv` with the installer; Python
virtual environments are not portable.

## Activate

In Bash, from the project root:

```bash
source src/mace/env.sh
cd "$MACE_CALCULATIONS_DIR"
```

Source `env.sh` to set the Python environment, Julia environment and local caches.
`MACEH_DIR` points to the source checkout. `JULIA_PROJECT` and `JULIA_DEPOT_PATH`
keep Julia's configuration and packages inside `src/mace`.
Numerical library thread counts use `SLURM_CPUS_PER_TASK`, or one outside an
allocation. Use Slurm for future calculations on this cluster.

Keep future calculation scripts, Slurm job files, inputs and outputs under
`calculations/mace/`, and run from that directory or its subdirectories.
After activation, use the upstream Python entry scripts with your own configs:
`$MACEH_DIR/deephe3-preprocess.py`, `$MACEH_DIR/deephe3-train.py`, and
`$MACEH_DIR/deephe3-eval.py`. The current band/DOS script is
`$MACEH_DIR/inference_tools/new_version/Band.py`. Upstream examples and default
configs are included in the source checkout. No datasets or trained
Hamiltonian models have been downloaded.

## Upstream references

- [Official MACE-H source and instructions](https://github.com/maurergroup/MACE-H)
- [MACE-H Python package metadata](https://github.com/maurergroup/MACE-H/blob/3d6248b715eef302a4fd674d52bcd69254596a51/pyproject.toml)
- [PyTorch Geometric CPU wheels](https://data.pyg.org/whl/torch-2.4.0+cpu.html)
- [PyJulia installation](https://pyjulia.readthedocs.io/en/latest/installation.html)
