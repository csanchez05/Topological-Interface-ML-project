#!/usr/bin/env python3
"""Scaffold calculations/CoGe_interface/<plane>/symNNNNN/ from the GB catalogue.

Per thickness folder symNNNNN/<k>x/ (k = 4..12): copies POSCAR from
structures/CoGeGrainBoundaries/CoGe_<plane>_GBs/Interface/, makes an empty convergence/ folder
and non-soc|soc/{scf,dos,bandstructure}/ folders each holding an absolute POTCAR symlink to the
master calculations/CoGe_interface/POTCAR. Information.txt is copied to <plane>/.
The catalogue is only read. Existing identical files are skipped; different ones stop the run.
"""
import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAT = PROJECT_ROOT / "structures/CoGeGrainBoundaries"
DEST = PROJECT_ROOT / "calculations/CoGe_interface"
MASTER = DEST / "POTCAR"
PLANES = ("100", "110", "111")
THICK = [f"{k}x" for k in range(4, 13)]
STAGES = [f"{b}/{s}" for b in ("non-soc", "soc") for s in ("scf", "dos", "bandstructure")]
VASP_OUT = {"OUTCAR", "OSZICAR", "vasprun.xml", "CHGCAR", "WAVECAR", "PROCAR", "DOSCAR", "vasp.out"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", nargs="+", default=list(PLANES), choices=PLANES)
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    a = ap.parse_args()

    if not MASTER.is_file():
        sys.exit(f"STOP: master POTCAR missing: {MASTER}")
    n = dict(sym=0, poscar=0, poscar_skip=0, link=0, link_skip=0, dirs=0)
    empty, extra, problems = [], [], []

    def mk(p):
        if not p.is_dir():
            n["dirs"] += 1
            if not a.dry_run:
                p.mkdir(parents=True)

    for plane in a.planes:
        src = CAT / f"CoGe_{plane}_GBs/Interface"
        dst = DEST / plane
        if not src.is_dir():
            sys.exit(f"STOP: missing {src}")
        mk(dst)
        info = src / "Information.txt"
        if info.is_file() and not a.dry_run:
            t = dst / "Information.txt"
            if not t.exists():
                shutil.copy2(info, t)
            elif not filecmp.cmp(info, t, shallow=False):
                problems.append(f"{t} differs")
        for s in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("sym")):
            n["sym"] += 1
            t = dst / s.name
            if t.is_dir() and any(f.name in VASP_OUT for f in t.rglob("*")):
                problems.append(f"{t} holds VASP output")
                continue
            for k in THICK:
                f = s / k / "POSCAR"
                if not f.is_file():
                    problems.append(f"missing {f}")
                    continue
                if f.stat().st_size == 0:
                    empty.append(str(f.relative_to(PROJECT_ROOT)))
                extra += [str(x) for x in (s / k).iterdir() if x.name != "POSCAR"]
                tf = t / k / "POSCAR"
                mk(t / k / "convergence")
                for st in STAGES:
                    mk(t / k / st)
                    ln = t / k / st / "POTCAR"
                    if ln.is_symlink() and os.readlink(ln) == str(MASTER):
                        n["link_skip"] += 1
                    elif ln.exists() or ln.is_symlink():
                        problems.append(f"{ln} exists and differs")
                    else:
                        n["link"] += 1
                        if not a.dry_run:
                            ln.symlink_to(MASTER)
                if tf.exists():
                    if filecmp.cmp(f, tf, shallow=False):
                        n["poscar_skip"] += 1
                    else:
                        problems.append(f"{tf} differs from source")
                    continue
                mk(t / k)
                n["poscar"] += 1
                if not a.dry_run:
                    shutil.copy2(f, tf)

    print(("DRY RUN " if a.dry_run else "") + str(n))
    print("empty source POSCARs:", empty)
    print("extra files in source thickness folders:", len(extra), extra[:5])
    if problems:
        print("PROBLEMS:", *problems[:20], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
