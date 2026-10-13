#!/usr/bin/env bash
set -euo pipefail
PLOT_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PLOT_SETUP_DIR"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request
with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=60) as response:
    metadata = json.load(response)
wheel = next(item for item in metadata['urls'] if item['filename'].endswith('.whl'))
with urllib.request.urlopen(wheel['url'], timeout=60) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == wheel['digests']['sha256']
Path('.bootstrap').mkdir(exist_ok=True)
Path('.bootstrap', wheel['filename']).write_bytes(data)
PY
    PYTHONPATH="$PLOT_SETUP_DIR/.bootstrap/pip-25.3-py3-none-any.whl" \
        .venv/bin/python -m pip --isolated install .bootstrap/pip-25.3-py3-none-any.whl
fi
PLOT_REQUIREMENTS=requirements.txt
if [[ -f requirements-lock.txt ]]; then
    PLOT_REQUIREMENTS=requirements-lock.txt
fi
.venv/bin/python -m pip --isolated install --no-compile -r "$PLOT_REQUIREMENTS"
if [[ ! -f requirements-lock.txt ]]; then
    .venv/bin/python -m pip freeze > requirements-lock.txt
fi
