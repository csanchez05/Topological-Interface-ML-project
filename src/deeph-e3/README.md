# DeepH-E3 in ML_Interface_Project

Local CPU installation of [Xiaoxun-Gong/DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3),
pinned to commit `036bf95b996052ef778c5c38e3c54c2ac0e2b591`.
This environment is independent of the MACE-H installation.

The core stack follows upstream's versions: **PyTorch 1.9.0+cpu**,
**PyTorch Geometric 1.7.2**, and **e3nn 0.3.5**. A local **Python 3.9.25**
interpreter provides compatibility with these packages. **Julia 1.5.4** and
its documented dependencies are included for the full upstream toolchain.

## Layout

- `env.sh`: Bash activation, local cache paths and numerical thread settings.
- `setup.sh`: installation from the pinned source and dependencies.
- `requirements.txt`: direct Python dependencies and compatibility constraints.
- `requirements-lock.txt`: all resolved Python dependencies, including wheel URLs.
- `setup.log`: installation output, appended on each installer invocation.
- `source/`: the official source checkout and its original entry scripts.
- `.python-runtime/` and `.venv/`: local Python interpreter and isolated environment.
- `.julia-runtime/` and `.julia-depot/`: local Julia interpreter, packages and artifacts.
- `julia-project/`: Julia project and resolved dependency manifest.
- `.bootstrap/`, `.cache/`, `.tmp/`: runtime archives, dependency caches and install scratch.
- `../../calculations/deeph-e3/`: all future calculation scripts, inputs and outputs.

The source repository has no pip installation metadata. The installer writes
`deeph_e3_source.pth` in the virtual environment so `deephe3` refers to the
local source checkout. Its revision is recorded in the installer and Python
lock file rather than as a fictitious PyPI package version.

## Install or reproduce

From the project root:

```bash
bash src/deeph-e3/setup.sh
```

The installer downloads checksum-verified Linux x86_64 Python and Julia
archives, creates the local environment, installs dependencies, registers the
source path, and installs the documented Julia packages. The Python lock file
and Julia manifest are reused when present. Numerical libraries default to
one thread outside a Slurm allocation.

Installation needs network access to GitHub, PyPI, the official PyTorch/PyG
wheel hosts, and Julia's download/package servers. No system Python packages
or MACE-H files are modified. If the project moves, recreate `.venv` with the
installer because virtual environments and the source-path registration use
absolute paths.

## Activate

In Bash, from the project root:

```bash
source src/deeph-e3/env.sh
cd "$DEEPHE3_CALCULATIONS_DIR"
```

`DEEPHE3_DIR` identifies the source checkout. `JULIA_PROJECT` and
`JULIA_DEPOT_PATH` select this installation's local Julia environment.
Activation sets its Julia executable on `PATH` after switching virtual
environments, so it can follow an active MACE-H environment.

The upstream entry scripts are:

- `$DEEPHE3_DIR/deephe3-preprocess.py`
- `$DEEPHE3_DIR/deephe3-train.py`
- `$DEEPHE3_DIR/deephe3-eval.py`
- `$DEEPHE3_DIR/deephe3-analyze.py`

Invoke these with the activated `python` and your own configuration file.
Default configuration references are in `source/deephe3/default_configs/`.
Keep run configurations, data, Slurm job files and all output paths under
`calculations/deeph-e3/`; run from that directory or one of its subdirectories.
Use Slurm for future calculations on this cluster. No custom run or submission
wrappers are added.

## Julia's role

Julia is a language for scientific and numerical computing. DeepH-E3's
training and Hamiltonian prediction use Python/PyTorch and can operate without
Julia when the input data is already prepared.

The supplied OpenMX converter (`source/deephe3/process_data_tools/openmx_get_data.jl`)
and sparse band-structure calculator (`source/inference_tools/sparse_calc.jl`)
use Julia. This installation includes Arpack, HDF5, ArgParse, JLD, JSON,
IterativeSolvers, DelimitedFiles and StaticArrays. DeepH-E3 launches Julia
directly, so no Python-Julia bridge is installed.

The upstream `inference_tools/deeph_band.sh` is an example containing the
author's absolute paths. For a future calculation, copy and adapt it under
the calculation directory; use `julia` from the activated environment and
the source location given by `DEEPHE3_DIR`.

## Installation scope

No datasets or trained models are downloaded. The installer performs no tests,
import checks, training, inference, calculations or Slurm submissions.
Successful installation does not represent runtime validation.

## Sources

- [DeepH-E3 requirements and usage](https://github.com/Xiaoxun-Gong/DeepH-E3#installation)
- [Python Build Standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20251031)
- [PyTorch Geometric wheels for PyTorch 1.9 CPU](https://data.pyg.org/whl/torch-1.9.0+cpu.html)
- [Official Julia release archives](https://julialang.org/downloads/oldreleases/)
