#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$MACE_SETUP_DIR"
exec > >(tee -a "$MACE_SETUP_DIR/setup.log") 2>&1
printf '\nMACE-H installation started: %s\n' "$(date -u +%FT%TZ)"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "This environment is pinned for Python 3.10"'
source_commit=3d6248b715eef302a4fd674d52bcd69254596a51
if [[ ! -d "$MACEH_DIR/.git" ]]; then
    git clone https://github.com/maurergroup/MACE-H.git "$MACEH_DIR"
    git -C "$MACEH_DIR" checkout --detach "$source_commit"
fi
if [[ "$(git -C "$MACEH_DIR" rev-parse HEAD)" != "$source_commit" ]]; then
    echo "MACE-H source must be at pinned commit $source_commit" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv --without-pip .venv
fi
# The cluster's system Python lacks ensurepip. Bootstrap from a verified PyPI wheel.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    if [[ ! -f .bootstrap/pip-25.3-py3-none-any.whl ]]; then
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/pip/25.3/json", timeout=60) as response:
    metadata = json.load(response)
wheel = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
with urllib.request.urlopen(wheel["url"], timeout=60) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
    raise RuntimeError("pip wheel checksum mismatch")
Path(".bootstrap").mkdir(exist_ok=True)
Path(".bootstrap", wheel["filename"]).write_bytes(data)
PY
    fi
    PYTHONPATH="$MACE_SETUP_DIR/.bootstrap/pip-25.3-py3-none-any.whl" \
        .venv/bin/python -m pip --isolated install --no-cache-dir \
        .bootstrap/pip-25.3-py3-none-any.whl
fi
.venv/bin/python -m pip --isolated install --cache-dir "$PIP_CACHE_DIR" \
    'setuptools==75.3.2' 'wheel==0.45.1'
requirements=requirements.txt
if [[ -f requirements-lock.txt ]]; then
    requirements=requirements-lock.txt
fi
.venv/bin/python -m pip --isolated install --no-compile --no-build-isolation \
    --cache-dir "$PIP_CACHE_DIR" -r "$requirements"
# Record the exact resolved environment without overwriting an existing lock.
if [[ ! -f requirements-lock.txt ]]; then
    {
        printf '%s\n' "# MACE-H source: $source_commit" \
            '# Python 3.10 / Linux x86_64 CPU environment' \
            '--extra-index-url https://download.pytorch.org/whl/cpu' \
            '--find-links https://data.pyg.org/whl/torch-2.4.0+cpu.html'
        .venv/bin/python -m pip freeze --all --exclude maceh
        printf '%s\n' '-e ./source'
    } > requirements-lock.txt
fi

# Install Julia and its package environment locally for MACE-H postprocessing.
if [[ ! -f .julia-runtime/.installed ]]; then
    python3 - <<'PY'
import hashlib
from pathlib import Path
import urllib.request

url = 'https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.11-linux-x86_64.tar.gz'
expected = 'fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf'
archive = Path('.bootstrap/julia-1.10.11-linux-x86_64.tar.gz')
archive.parent.mkdir(exist_ok=True)
if not archive.exists():
    partial = archive.with_suffix('.part')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(archive)
digest = hashlib.sha256()
with archive.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(chunk)
if digest.hexdigest() != expected:
    archive.unlink()
    raise RuntimeError('Julia archive checksum mismatch; rerun setup to download again')
PY
    mkdir -p .julia-runtime
    tar -xzf .bootstrap/julia-1.10.11-linux-x86_64.tar.gz \
        -C .julia-runtime --strip-components=1
    touch .julia-runtime/.installed
fi
mkdir -p "$JULIA_PROJECT" "$JULIA_DEPOT_PATH"
.julia-runtime/bin/julia --startup-file=no --project="$JULIA_PROJECT" -e '
    using Pkg
    if isfile(joinpath(ENV["JULIA_PROJECT"], "Project.toml"))
        Pkg.instantiate()
    else
        Pkg.add(["PyCall", "Distributed", "LinearAlgebra", "StaticArrays",
                 "HDF5", "JSON", "ArgParse", "DelimitedFiles", "Statistics",
                 "SparseArrays", "Arpack", "JLD"])
    end
    Pkg.build("PyCall")
'
printf 'MACE-H installed. Activate with: source %q/env.sh\n' "$MACE_SETUP_DIR"
