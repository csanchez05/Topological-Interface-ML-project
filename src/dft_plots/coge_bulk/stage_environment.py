"""Stage the fixed plotting libraries once per node, avoiding NFS import stalls."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

project = Path(__file__).resolve().parents[3]
meta_path = project/'results/CoGe_100_plots/environment_archive.json'
meta = json.loads(meta_path.read_text())
base = Path('/tmp') / f'coge-plot-environment-{os.getuid()}'
base.mkdir(mode=0o700, exist_ok=True)
assert base.stat().st_uid == os.getuid() and not base.is_symlink()
target = base / meta['key']
with (base/(meta['key']+'.lock')).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not target.exists():
        print('Staging plotting environment on this node', file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix='build-',dir=base) as temporary:
            stage = Path(temporary)
            archive = stage/'environment.tar.gz'
            h = hashlib.sha256()
            # The archive sits next to its record; the recorded path may predate a move.
            with (meta_path.parent/Path(meta['path']).name).open('rb') as source, archive.open('wb') as destination:
                for data in iter(lambda: source.read(4*1024*1024), b''):
                    h.update(data)
                    destination.write(data)
            assert h.hexdigest() == meta['sha256'], 'Environment archive checksum mismatch'
            # This archive was generated from our installed environment, and
            # contains expected interpreter symlinks to /usr/bin/python3.
            with tarfile.open(archive,'r:gz') as tar:
                for member in tar:
                    p = Path(member.name)
                    assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='environment'
                tar.extractall(stage)
            env = stage/'environment'
            (env/'staged_archive.json').write_text(json.dumps(meta)+'\n')
            os.rename(env,target)
    assert json.loads((target/'staged_archive.json').read_text())['sha256'] == meta['sha256']
    assert (target/'bin/python').is_file()
print(target/'bin/python',flush=True)
