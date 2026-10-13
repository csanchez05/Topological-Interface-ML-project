#!/usr/bin/env bash
# Installation only: no tests, imports of HamGNN, model loads, or calculations.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$UNIHAMGNN_SETUP_DIR"
exec > >(tee -a "$UNIHAMGNN_SETUP_DIR/setup.log") 2>&1
printf '\nUni-HamGNN installation started: %s\n' "$(date -u +%FT%TZ)"

source_commit=64dba1dbf445ccfa2683f560059c3fbca0df624a
if [[ ! -d "$UNIHAMGNN_DIR/.git" ]]; then
    git init "$UNIHAMGNN_DIR"
    git -C "$UNIHAMGNN_DIR" remote add origin https://github.com/QuantumLab-ZY/HamGNN.git
fi
source_head="$(git -C "$UNIHAMGNN_DIR" rev-parse --verify HEAD 2>/dev/null || true)"
if [[ -z "$source_head" ]]; then
    git -C "$UNIHAMGNN_DIR" fetch --depth 1 origin "$source_commit"
    git -C "$UNIHAMGNN_DIR" checkout --detach "$source_commit"
elif [[ "$source_head" != "$source_commit" ]]; then
    echo "HamGNN source must be at pinned commit $source_commit; existing checkout preserved." >&2
    exit 1
fi

printf '\nInstalling the local Python environment...\n'
if [[ ! -x .venv/bin/python ]]; then
    /usr/bin/python3.10 -m venv --without-pip .venv
fi
# The system Python lacks ensurepip; bootstrap pip from its official wheel.
if [[ ! -d .venv/lib/python3.10/site-packages/pip ]]; then
    /usr/bin/python3.10 - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request

bootstrap = Path('.bootstrap')
bootstrap.mkdir(exist_ok=True)
with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=60) as response:
    metadata = json.load(response)
wheel = next(item for item in metadata['urls'] if item['filename'].endswith('.whl'))
with urllib.request.urlopen(wheel['url'], timeout=120) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != wheel['digests']['sha256']:
    raise RuntimeError('pip wheel download was corrupted')
archive = bootstrap / wheel['filename']
partial = archive.with_suffix('.part')
partial.write_bytes(data)
partial.replace(archive)
PY
    PYTHONPATH="$UNIHAMGNN_SETUP_DIR/.bootstrap/pip-25.3-py3-none-any.whl" \
        .venv/bin/python -m pip --isolated --disable-pip-version-check install \
        --no-compile --no-cache-dir .bootstrap/pip-25.3-py3-none-any.whl
fi
.venv/bin/python -m pip --isolated --disable-pip-version-check install \
    --no-compile --cache-dir "$PIP_CACHE_DIR" \
    'pip==25.3' 'setuptools==75.3.2' 'wheel==0.45.1'

requirements=requirements.txt
if [[ -f requirements-lock.txt ]]; then
    requirements=requirements-lock.txt
fi
.venv/bin/python -m pip --isolated --disable-pip-version-check install \
    --no-compile --no-build-isolation --cache-dir "$PIP_CACHE_DIR" -r "$requirements"
if [[ ! -f requirements-lock.txt ]]; then
    {
        printf '%s\n' "# HamGNN source: $source_commit" \
            '# Python 3.10 / Linux x86_64 CPU environment' \
            '--extra-index-url https://download.pytorch.org/whl/cpu' \
            '--find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html'
        .venv/bin/python -m pip --isolated --disable-pip-version-check freeze --all --exclude HamGNN
        printf '%s\n' '-e ./source'
    } > requirements-lock.txt.tmp
    mv requirements-lock.txt.tmp requirements-lock.txt
fi

printf '\nInstalling the official Uni-HamGNN pretrained model...\n'
.venv/bin/python -u - <<'PY'
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

destination = Path('models/uni-hamgnn_2_1.pkl')
if not destination.exists():
    bootstrap = Path('.bootstrap')
    bootstrap.mkdir(exist_ok=True)
    archive = bootstrap / 'uni-hamgnn_2_1.pkl.zip'
    url = 'https://zenodo.org/records/17239078/files/uni-hamgnn_2_1.pkl.zip?download=1'
    expected_md5 = '7f99cba551b4a0cbb84827cfe359cdac'
    if not archive.exists():
        partial = archive.with_suffix('.part')
        print('Downloading the pretrained model archive (204 MB)...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(archive)
    digest = hashlib.md5()
    with archive.open('rb') as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        archive.unlink()
        raise RuntimeError('Model archive download was corrupted; rerun setup to download again')
    with zipfile.ZipFile(archive) as package:
        members = [member for member in package.infolist()
                   if not member.is_dir()
                   and Path(member.filename).name == destination.name
                   and '__MACOSX' not in Path(member.filename).parts]
        if len(members) != 1:
            raise RuntimeError('Expected one uni-hamgnn_2_1.pkl file in the official archive')
        destination.parent.mkdir(exist_ok=True)
        partial = destination.with_suffix('.part')
        print('Extracting the pretrained model...', flush=True)
        with package.open(members[0]) as data, partial.open('wb') as output:
            shutil.copyfileobj(data, output, length=1024 * 1024)
        partial.replace(destination)
    print(f'Model installed: {destination}', flush=True)
else:
    print(f'Keeping existing model: {destination}', flush=True)
PY
printf '\nUni-HamGNN installed. Activate with: source %q/env.sh\n' "$UNIHAMGNN_SETUP_DIR"
