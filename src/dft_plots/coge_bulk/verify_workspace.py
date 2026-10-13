"""Run the plot collection audit on the login node, where workspace reads are fast."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

project = Path(__file__).resolve().parents[3]
logs = project / 'results/CoGe_100_plots/logs'
environment = project / 'src/dft_plots/.venv'
record = {
    'started_at': time.time(),
    'reason': 'Compute-node NFS reads delay verification; use the same audit and pinned environment on the workspace host.',
    'command': [str(environment / 'bin/python'), '-B', '-u', '-m',
                'src.dft_plots.coge_bulk.plot_all', '--verify-all', '--dpi', '160'],
}
logs.mkdir(parents=True, exist_ok=True)
record_path = logs / 'workspace_verification.json'
record_path.write_text(json.dumps(record, indent=2) + '\n')
with tempfile.TemporaryDirectory(prefix='coge-plot-final-audit-') as scratch:
    env = dict(os.environ)
    env.update(MPLBACKEND='Agg', PYTHONNOUSERSITE='1',
               MPLCONFIGDIR=str(Path(scratch) / 'matplotlib'),
               XDG_CACHE_HOME=str(Path(scratch) / 'cache'),
               COGE_PLOT_SCRATCH=scratch, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VIRTUAL_ENV=str(environment),
               PATH=str(environment / 'bin') + os.pathsep + env.get('PATH', ''))
    with (logs / 'verify-workspace.out').open('w') as log:
        process = subprocess.Popen(record['command'], cwd=project, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        return_code = process.wait()
record.update(completed_at=time.time(), exit_code=return_code)
record_path.write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(return_code)
