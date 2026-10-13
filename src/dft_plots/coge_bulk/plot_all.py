"""Render the requested CoGe_100 suite, one structure per resumable worker."""
from __future__ import annotations
import argparse
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from PIL import Image
from .. import b20
from .paths import PROJECT_ROOT, resolve_stage, band_reference_stage

AUDIT = PROJECT_ROOT / 'results/CoGe_100_plots'
OUTPUT = PROJECT_ROOT / 'plots/CoGe/bulk/100'
CALCULATIONS = PROJECT_ROOT / 'calculations/CoGe_Bulk/CoGe_100'
EXPECTED = {
    'non_soc': ['bands_nosoc.png', 'bands_dos_combined.png'] +
               [f'proj_{element}_{orbital}.png' for element in ('Co','Ge') for orbital in ('s','p','d')],
    'soc': ['bands_soc.png','bands_dos_combined.png'] + [f'soc_spin_{axis}.png' for axis in ('Sx','Sy','Sz')],
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_images(index, dpi=160):
    inventory = []
    for branch, names in EXPECTED.items():
        for name in names:
            path = OUTPUT/f'sym{index:05d}'/branch/name
            expected_size = (10*dpi,5*dpi) if name == 'bands_dos_combined.png' else (8*dpi,5*dpi)
            with Image.open(path) as image:
                image.load()
                if image.format != 'PNG' or image.size != expected_size:
                    raise ValueError(f'Unexpected format/size: {path}: {image.format} {image.size}')
                if all(low == high for low,high in image.convert('RGB').getextrema()):
                    raise ValueError(f'Blank image: {path}')
            inventory.append({'path':str(path.relative_to(PROJECT_ROOT)), 'width':expected_size[0],
                              'height':expected_size[1], 'bytes':path.stat().st_size,
                              'sha256':sha256(path)})
    return inventory


def sources(index, dpi):
    data = {'dpi':dpi, 'renderer_sha256':sha256(Path(b20.__file__)),
            'batch_sha256':sha256(Path(__file__)), 'branches':{}}
    for mode in ('non-soc','soc'):
        branch = CALCULATIONS/f'sym{index:05d}'/mode
        bands = resolve_stage(branch, 'bandstructure')
        dos = resolve_stage(branch, 'dos')
        if band_reference_stage(branch) != dos:
            raise ValueError('Band reference must be the matching DOS15 run')
        with (dos/'DOSCAR').open() as stream:
            header = [next(stream) for _ in range(6)]
        ef = float(header[5].split()[3])
        band_ref = json.loads((bands/'energy_reference.json').read_text())
        if abs(ef-band_ref['fermi_eV']) > 1e-6:
            raise ValueError('Band/DOS Fermi reference mismatch')
        data['branches'][mode] = {
            'bands':str(bands), 'dos':str(dos), 'fermi_eV':ef,
            'bands_result_sha256':sha256(bands/'result.json'),
            'dos_result_sha256':sha256(dos/'result.json'),
        }
    return data


def render(index, dpi):
    name = f'sym{index:05d}'
    report_path = AUDIT/'structures'/f'{name}.json'
    source = sources(index, dpi)
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('status') == 'complete' and previous.get('sources') == source:
            inventory = verify_images(index, dpi)
            if inventory == previous['images']:
                print(name + ': already complete and verified', flush=True)
                return
    started = time.time()
    report = {'structure':name, 'sources':source, 'status':'running', 'started_at':started,
              'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    atomic_json(report_path,report)
    try:
        for mode, output_mode in (('non-soc','non_soc'),('soc','soc')):
            branch = CALCULATIONS/name/mode
            destination = OUTPUT/name/output_mode
            cache = b20._RunCache()
            label = f'{name} {"SOC" if mode == "soc" else "non-SOC"}'
            print(name + ' ' + mode + ': bands and combined DOS', flush=True)
            b20._plot_bands(cache, branch, formula='CoGe', branch_label=label,
                            destination=destination/('bands_soc.png' if mode=='soc' else 'bands_nosoc.png'),
                            dpi=dpi, figure_size=(8,5))
            b20._plot_bands_dos(cache, branch, formula='CoGe', branch_label=label,
                                destination=destination/'bands_dos_combined.png',
                                dpi=dpi, figure_size=(10,5))
            print(name + ' ' + mode + ': projections', flush=True)
            function = b20._plot_soc_spin_projections if mode == 'soc' else b20._plot_orbital_projected_bands
            function(cache, branch, formula='CoGe', orientation=f'100 {name}',
                     destination=destination, dpi=dpi, figure_size=(8,5))
            del cache
            gc.collect()
        report.update(status='complete', images=verify_images(index,dpi), completed_at=time.time(),
                      elapsed_seconds=round(time.time()-started,2))
        atomic_json(report_path,report)
        print(f'{name}: 13 figures verified in {report["elapsed_seconds"]} seconds', flush=True)
    except Exception:
        report.update(status='failed', error=traceback.format_exc(), failed_at=time.time())
        atomic_json(report_path,report)
        raise


def verify_all(dpi):
    images, failures = [], []
    for index in range(623):
        name = f'sym{index:05d}'
        try:
            report = json.loads((AUDIT/'structures'/f'{name}.json').read_text())
            if report['status'] != 'complete' or report['sources'] != sources(index,dpi):
                raise ValueError('Missing or stale completion report')
            inventory = verify_images(index,dpi)
            if inventory != report['images']:
                raise ValueError('Image inventory changed')
            images.extend(inventory)
        except Exception as exc:
            failures.append({'structure':name,'error':str(exc)})
        if index % 50 == 0:
            print(f'Verified {index+1}/623 structures',flush=True)
    result = {'status':'complete' if not failures and len(images)==8099 else 'incomplete',
              'expected_images':8099,'verified_images':len(images), 'failures':failures,
              'verified_at':time.time(), 'total_image_bytes':sum(i['bytes'] for i in images)}
    atomic_json(AUDIT/'image_inventory.json',images)
    atomic_json(AUDIT/'summary.json',result)
    print(json.dumps(result,indent=2))
    if result['status'] != 'complete':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--index',type=int)
    group.add_argument('--verify-all',action='store_true')
    parser.add_argument('--dpi',type=int,default=160)
    args = parser.parse_args()
    if args.index is not None and not 0 <= args.index < 623:
        parser.error('--index must be between 0 and 622')
    if not 80 <= args.dpi <= 300:
        parser.error('--dpi must be between 80 and 300')
    b20.plt.rcParams['savefig.bbox'] = None
    b20.plt.rcParams['figure.max_open_warning'] = 0
    if args.verify_all:
        verify_all(args.dpi)
    else:
        lock_dir = AUDIT/'logs'/'locks'
        lock_dir.mkdir(parents=True,exist_ok=True)
        with (lock_dir/f'sym{args.index:05d}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            render(args.index,args.dpi)


if __name__ == '__main__':
    main()
