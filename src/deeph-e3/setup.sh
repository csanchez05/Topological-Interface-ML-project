#!/usr/bin/env bash
# Install only: no tests, training, inference, calculations, or scheduler jobs.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$DEEPHE3_SETUP_DIR"
exec > >(tee -a "$DEEPHE3_SETUP_DIR/setup.log") 2>&1
printf '\nDeepH-E3 installation started: %s\n' "$(date -u +%FT%TZ)"

source_commit=036bf95b996052ef778c5c38e3c54c2ac0e2b591
if [[ ! -d "$DEEPHE3_DIR/.git" ]]; then
    git clone https://github.com/Xiaoxun-Gong/DeepH-E3.git "$DEEPHE3_DIR"
fi
if [[ "$(git -C "$DEEPHE3_DIR" rev-parse HEAD)" != "$source_commit" ]]; then
    if [[ -n "$(git -C "$DEEPHE3_DIR" status --porcelain)" ]]; then
        echo 'Source checkout has changes; refusing to switch revisions.' >&2
        exit 1
    fi
    git -C "$DEEPHE3_DIR" checkout --detach "$source_commit"
fi

printf '\nInstalling local Python and Julia runtimes...\n'
python3 - <<'PY'
import hashlib
from pathlib import Path
import subprocess
import urllib.request

runtimes = [
    (
        '.python-runtime',
        'cpython-3.9.25+20251031-x86_64-unknown-linux-gnu-install_only.tar.gz',
        'https://github.com/astral-sh/python-build-standalone/releases/download/20251031/',
        '42834f61eb6df43432c3dd6ab9ca3fdf8c06d10a404ebdb53d6902e6b9570b08',
    ),
    (
        '.julia-runtime',
        'julia-1.5.4-linux-x86_64.tar.gz',
        'https://julialang-s3.julialang.org/bin/linux/x64/1.5/',
        '80dec351d1a593e8ad152636971a48d0c81bfcfab92c87f3604663616f1e8bc5',
    ),
]
bootstrap = Path('.bootstrap')
bootstrap.mkdir(exist_ok=True)
for directory, filename, base_url, expected in runtimes:
    destination = Path(directory)
    marker = destination / '.installed'
    if marker.exists():
        continue
    archive = bootstrap / filename
    if not archive.exists():
        print(f'Downloading {filename}', flush=True)
        partial = archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(base_url + filename, timeout=120) as response, partial.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        archive.unlink()
        raise RuntimeError(f'{filename}: checksum mismatch; rerun setup to download again')
    destination.mkdir(exist_ok=True)
    print(f'Extracting {filename}', flush=True)
    subprocess.run(['tar', '-xzf', str(archive), '-C', str(destination), '--strip-components=1'], check=True)
    marker.write_text(filename + '\nsha256=' + expected + '\n')
PY

printf '\nInstalling Python dependencies...\n'
if [[ ! -x .venv/bin/python ]]; then
    .python-runtime/bin/python3.9 -m venv .venv
fi
.venv/bin/python -m pip --isolated install --no-compile --cache-dir "$PIP_CACHE_DIR" \
    'pip==25.3' 'setuptools==59.5.0' 'wheel==0.45.1'
requirements=requirements.txt
if [[ -f requirements-lock.txt ]]; then
    requirements=requirements-lock.txt
fi
.venv/bin/python -m pip --isolated install --no-compile --no-build-isolation \
    --cache-dir "$PIP_CACHE_DIR" -r "$requirements"

# Upstream is a source checkout, not a pip-packaged project.
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import sysconfig

Path(sysconfig.get_path('purelib'), 'deeph_e3_source.pth').write_text(
    os.environ['DEEPHE3_DIR'] + '\n'
)
PY
if [[ ! -f requirements-lock.txt ]]; then
    {
        printf '%s\n' "# DeepH-E3 source: $source_commit" \
            '# CPython 3.9.25 / Linux x86_64 CPU environment' \
            '--extra-index-url https://download.pytorch.org/whl/cpu' \
            '--find-links https://data.pyg.org/whl/torch-1.9.0+cpu.html'
        .venv/bin/python -m pip freeze --all
    } > requirements-lock.txt.tmp
    mv requirements-lock.txt.tmp requirements-lock.txt
fi

printf '\nInstalling Julia dependencies...\n'
mkdir -p "$JULIA_PROJECT" "$JULIA_DEPOT_PATH"
.julia-runtime/bin/julia --startup-file=no --project="$JULIA_PROJECT" -e '
    using Pkg
    if isfile(joinpath(ENV["JULIA_PROJECT"], "Project.toml"))
        Pkg.instantiate()
    else
        Pkg.add(["Arpack", "HDF5", "ArgParse", "JLD", "JSON",
                 "IterativeSolvers", "DelimitedFiles", "StaticArrays"])
    end
'
printf '\nDeepH-E3 installed. Activate with: source %q/env.sh\n' "$DEEPHE3_SETUP_DIR"
