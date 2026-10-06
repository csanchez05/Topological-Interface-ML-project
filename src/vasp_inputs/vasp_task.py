#!/usr/bin/env python3
"""Run one generated VASP stage folder in node-local scratch, validate it, copy it back.

Called by run_vasp_array.slurm, one array task per stage folder (standard library only):

  vasp_task.py prepare FOLDER --records DIR   checks, claims and stages the inputs; prints the scratch path
  vasp_task.py binary  FOLDER                 prints vasp_std or vasp_ncl (from the recipe)
  vasp_task.py sample  SCRATCH                background sampler: summed RSS of this job's VASP ranks
  vasp_task.py finish  FOLDER SCRATCH --exit-code N --records DIR
                                              validates; on success copies the outputs and result.json
                                              into FOLDER and removes the scratch

Rules (CLAUDE.md section 3):
- A folder that already holds VASP output is refused. Its inputs must be exactly what
  make_vasp_inputs.py renders now (only the stamp's git hash may differ).
- The run happens in a private mkdtemp under $TMPDIR (else /tmp). POTCAR there is a symlink to the
  master and is never copied.
- Only a validated run is copied back, file by file (*.copying, then SHA-256 check, then rename).
  result.json is written last.
- A failed run keeps its scratch and leaves the stage folder untouched. Its claim and a failure
  record stay in DIR (claims/, failed/). Delete the claim to allow a rerun.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402

PROJECT_ROOT = mvi.PROJECT_ROOT
VASP_VERSION = "vasp.6.4.3"
POTENTIALS = {"Co": ("PAW_PBE Co 02Aug2007", 9.0), "Ge": ("PAW_PBE Ge 05Jan2001", 4.0)}
EXPORT_ALWAYS = ("OUTCAR", "OSZICAR", "EIGENVAL", "IBZKPT", "DOSCAR", "vasprun.xml", "vasp.out")
MIN_FREE_GIB = 10
SAMPLE_SECONDS = 30
NUM = r"([-+]?[0-9.]+(?:[EeDd][-+]?\d+)?)"
MIXING = ("AMIX", "AMIX_MAG", "AMIN")  # OUTCAR "AMIX     =   0.40;   BMIX ...", "AMIN     =   0.10"


class TaskError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise TaskError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def last(pattern, text, flags=re.M):
    found = re.findall(pattern, text, flags)
    require(found, "missing in output: " + pattern)
    return found[-1]


def number(text):
    value = float(str(text).replace("D", "E").replace("d", "e"))
    require(math.isfinite(value), "non-finite number " + str(text))
    return value


# ---------------------------------------------------------------- folder -> recipe

def resolve_folder(folder):
    path = Path(folder)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    require(PROJECT_ROOT / "calculations/CoGe_interface" in path.parents,
            f"{path} is not under calculations/CoGe_interface")
    return path


def identify(folder):
    """Stamp, recipe and template fields (sym, thickness, point) of a generated stage folder."""
    first = (folder / "INCAR").read_text().split("\n", 1)[0]
    match = mvi.STAMP.match(first)
    require(match, f"{folder}/INCAR has no make_vasp_inputs.py stamp")
    version, git_hash, family, stage = match.groups()
    r = mvi.recipe(family, stage)
    rel = str(folder.relative_to(PROJECT_ROOT))
    pattern = re.escape(r["target"]).replace(r"\{sym\}", r"(?P<sym>sym\d{5})") \
        .replace(r"\{thickness\}", r"(?P<thickness>\d+x)").replace(r"\{point\}", r"(?P<point>[^/]+)")
    fields = re.fullmatch(pattern, rel)
    require(fields, f"{rel} does not match the recipe target {r['target']}")
    return {"version": version, "git": git_hash, "family": family, "stage": stage, "recipe": r,
            **fields.groupdict()}


def check_inputs_current(folder, info):
    """The folder's inputs equal what the recipe renders now (hash in the stamp aside)."""
    root = PROJECT_ROOT
    require(info["version"] == mvi.VERSION, f"INCAR made by v{info['version']}, code is v{mvi.VERSION}")
    target, files, potcar = mvi.plan_folder(info["family"], info["stage"], info["sym"], info["thickness"],
                                           info["point"], info["git"], root)
    require(target == folder, "folder/recipe mismatch")
    state = mvi.classify(folder, files, potcar)
    require(state == "identical", f"{folder}: inputs are not the generator's output ({state})")
    return potcar


def claim_path(records, info):
    return Path(records) / "claims" / f"{info['sym']}_{info['thickness']}_{info['point']}.json"


# ---------------------------------------------------------------- prepare / sample

def prepare(folder, records, environ=os.environ, scratch_base=None):
    folder = resolve_folder(folder)
    info = identify(folder)
    potcar = check_inputs_current(folder, info)
    r = info["recipe"]
    _, mesh = r["points"][info["point"]]
    ranks, kpar, ncore = mvi.check_profile(r, mesh)
    require(environ.get("SLURM_NTASKS") == str(ranks),
            f"recipe needs {ranks} ranks, allocation has SLURM_NTASKS={environ.get('SLURM_NTASKS')}")
    incar = mvi.tags((folder / "INCAR").read_text())
    require(incar.get("KPAR") == str(kpar) and incar.get("NCORE") == str(ncore), "INCAR KPAR/NCORE vs profile")

    claim = claim_path(records, info)
    claim.parent.mkdir(parents=True, exist_ok=True)
    job = f"{environ.get('SLURM_ARRAY_JOB_ID', environ.get('SLURM_JOB_ID', 'nojob'))}" \
          f"_{environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise TaskError(f"{claim} exists: running elsewhere, or failed before (delete it to rerun)")
    with os.fdopen(fd, "w") as stream:
        json.dump({"folder": str(folder.relative_to(PROJECT_ROOT)), "job": job,
                   "node": os.uname().nodename, "claimed_at": now()}, stream)

    try:
        base = scratch_base or environ.get("TMPDIR") or "/tmp"
        scratch = Path(tempfile.mkdtemp(prefix=f"coge-gb-{info['sym']}-{info['point']}-{job}-", dir=base))
        free = shutil.disk_usage(scratch).free / 1024**3
        require(free >= MIN_FREE_GIB, f"only {free:.1f} GiB free in {base}")
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            shutil.copyfile(folder / name, scratch / name)
        (scratch / "POTCAR").symlink_to(potcar)
        write_json(scratch / "prepare.json", {
            "folder": str(folder.relative_to(PROJECT_ROOT)), "scratch": str(scratch), "job": job,
            "node": os.uname().nodename, "ranks": ranks, "kpar": kpar, "ncore": ncore,
            "family": info["family"], "stage": info["stage"], "version": info["version"], "git": info["git"],
            "inputs_sha256": {n: digest(folder / n) for n in ("INCAR", "KPOINTS", "POSCAR")},
            "potcar": str(potcar), "claim": str(claim), "prepared_at": now(), "free_gib": round(free, 1)})
    except BaseException:
        claim.unlink()  # nothing ran; release the folder
        raise
    return scratch


def job_rss(job_id, proc=Path("/proc")):
    """Summed VmRSS (bytes) and count of VASP processes whose environment has SLURM_JOB_ID=job_id."""
    total, count = 0, 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() not in ("vasp_std", "vasp_ncl", "vasp_gam"):
                continue
            env = (entry / "environ").read_bytes().split(b"\0")
            if f"SLURM_JOB_ID={job_id}".encode() not in env:
                continue
            status = (entry / "status").read_text()
            total += int(re.search(r"^VmRSS:\s+(\d+) kB", status, re.M).group(1)) * 1024
            count += 1
        except (OSError, AttributeError):
            continue
    return total, count


def sample(scratch, environ=os.environ):
    scratch = Path(scratch)
    job_id = environ.get("SLURM_JOB_ID", "")
    state = {"peak_rss_bytes": 0, "max_ranks_seen": 0, "samples": 0, "interval_s": SAMPLE_SECONDS}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        rss, count = job_rss(job_id)
        state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss)
        state["max_ranks_seen"] = max(state["max_ranks_seen"], count)
        state["samples"] += count > 0
        write_json(scratch / "rss.json", state)
        time.sleep(SAMPLE_SECONDS)


# ---------------------------------------------------------------- validation

def poscar_counts(text):
    lines = text.splitlines()
    return dict(zip(lines[5].split(), (int(x) for x in lines[6].split())))


def warning_boxes(text):
    """[(title, text)] of VASP's boxed messages (ADVICE, WARNING, ERROR, ...)."""
    boxes = []
    for block in re.findall(r"^ -{20,}[ \t]*\n((?:\|.*\|[ \t]*\n)+) -{20,}", text, re.M):
        rows = [row.strip().strip("|").strip() for row in block.splitlines()]
        rows = [row for row in rows if row]
        if rows:
            title = re.sub(r"[-<>]", "", rows[0]).strip()
            boxes.append({"title": title, "text": " ".join(rows[1:])[:400]})
    return boxes


def validate(scratch, prepared, exit_code):
    """Checks the finished scratch run against its own inputs; returns the result record."""
    s = Path(scratch)
    require(exit_code == 0, f"srun/VASP exit code {exit_code}")
    for name, sha in prepared["inputs_sha256"].items():
        require(digest(s / name) == sha, f"{name} changed during the run")
    incar = mvi.tags((s / "INCAR").read_text())
    mesh = tuple(int(x) for x in (s / "KPOINTS").read_text().splitlines()[3].split())
    counts = poscar_counts((s / "POSCAR").read_text())
    out = (s / "OUTCAR").read_text(errors="replace")
    osz = (s / "OSZICAR").read_text(errors="replace")

    require(VASP_VERSION in out[:400], "not " + VASP_VERSION)
    require("General timing and accounting informations for this job" in out, "VASP did not finish")
    require("aborting loop because EDIFF is reached" in out, "electronic loop did not reach EDIFF")
    titles = list(dict.fromkeys(t.strip() for t in re.findall(r"^\s*TITEL\s*=\s*(.+)$", out, re.M)))
    require(titles == [POTENTIALS[e][0] for e in counts], f"POTCAR TITELs {titles}")
    nions = sum(counts.values())
    nelect = sum(n * POTENTIALS[e][1] for e, n in counts.items())
    require(int(last(r"NIONS\s*=\s*(\d+)", out)) == nions, "NIONS")
    require(abs(number(last(r"^\s*NELECT\s*=\s*" + NUM, out)) - nelect) < 1e-6, "NELECT")

    ints = ["NELM", "ICHARG", "ISTART", "ISYM", "NSW", "IBRION", "LMAXMIX", "ISPIN"]
    for key in ints:
        if key in incar:
            found = int(last(r"^\s*" + key + r"\s*=\s*(-?\d+)", out))
            require(found == int(incar[key]), f"OUTCAR {key} {found} != INCAR {incar[key]}")
    require(abs(number(last(r"^\s*ENCUT\s*=\s*" + NUM, out)) - float(incar["ENCUT"])) < 1e-6, "ENCUT")
    require(math.isclose(number(last(r"^\s*EDIFF\s*=\s*" + NUM, out)), float(incar["EDIFF"]), rel_tol=1e-9), "EDIFF")
    smear, sigma = last(r"^\s*ISMEAR\s*=\s*(-?\d+);\s*SIGMA\s*=\s*" + NUM, out)
    require(int(smear) == int(incar["ISMEAR"]) and abs(number(sigma) - float(incar["SIGMA"])) < 1e-9,
            "ISMEAR/SIGMA")
    mixing = {key: number(last(r"^\s*" + key + r"\s*=\s*" + NUM, out)) for key in MIXING}
    for key in MIXING:
        if key in incar:
            require(abs(mixing[key] - float(incar[key])) < 1e-9,
                    f"OUTCAR {key} {mixing[key]} != INCAR {incar[key]}")
    flags = {"LSORBIT": incar.get("LSORBIT", ".FALSE."), "LREAL": incar.get("LREAL", ".FALSE."),
             "LWAVE": incar.get("LWAVE", ".TRUE."), "LCHARG": incar.get("LCHARG", ".TRUE.")}
    for key, value in flags.items():
        want = "T" if value.upper().startswith(".T") else "F"
        require(last(r"^\s*" + key + r"\s*=\s*([TF])\b", out) == want, f"OUTCAR {key} != {want}")
    require(tuple(int(x) for x in last(r"generate k-points for:\s+(\d+)\s+(\d+)\s+(\d+)", out)) == mesh,
            "OUTCAR k-mesh")
    k_cores, k_groups = (int(x) for x in last(r"distrk:\s+each k-point on\s+(\d+) cores,\s+(\d+) groups", out))
    ncore = int(last(r"distr:\s+one band on NCORE=\s*(\d+) cores", out))
    require(k_groups == prepared["kpar"] and k_cores * k_groups == prepared["ranks"]
            and ncore == prepared["ncore"], "MPI decomposition (KPAR/NCORE/ranks)")
    nkpts, nbands = (int(x) for x in last(r"NKPTS\s*=\s*(\d+).*?NBANDS\s*=\s*(\d+)", out))
    if "NBANDS" in incar:
        require(nbands == int(incar["NBANDS"]), f"NBANDS {nbands} != {incar['NBANDS']}")
    require(k_groups <= nkpts, "KPAR exceeds NKPTS")

    steps = int(last(r"^\s*(?:DAV|RMM):\s*(\d+)", osz))
    require(0 < steps < int(incar.get("NELM", 60)), f"{steps} SCF steps (NELM {incar.get('NELM')})")
    require(re.search(r"^\s*1\s+F=", osz, re.M), "OSZICAR has no final F= line")
    with open(s / "vasprun.xml", "rb") as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 4096))
        require(b"</modeling>" in stream.read(), "vasprun.xml incomplete")

    toten = number(last(r"free\s+energy\s+TOTEN\s*=\s*" + NUM, out))
    rss = json.loads((s / "rss.json").read_text()) if (s / "rss.json").exists() else {}
    record = {
        "valid": True,
        "nions": nions, "nelect": nelect, "counts": counts, "encut_eV": float(incar["ENCUT"]),
        "mesh": list(mesh), "ismear": int(incar["ISMEAR"]), "sigma_eV": float(incar["SIGMA"]),
        "ediff_eV": float(incar["EDIFF"]), "mixing": mixing,
        "nkpts": nkpts, "nbands": nbands, "nplwv": int(last(r"NPLWV\s*=\s*(\d+)", out)),
        "scf_steps": steps,
        "toten_eV": toten, "toten_eV_atom": toten / nions,
        "e0_eV": number(last(r"energy\(sigma->0\)\s*=\s*" + NUM, out)),
        "eentro_eV": number(last(r"EENTRO\s*=\s*" + NUM, out)),
        "efermi_eV": number(last(r"E-fermi\s*:\s*" + NUM, out)),
        "elapsed_s": number(last(r"Elapsed time \(sec\):\s*" + NUM, out)),
        "rank0_max_memory_kb": number(last(r"Maximum memory used \(kb\):\s*" + NUM, out)),
        "peak_rss_bytes": rss.get("peak_rss_bytes"), "rss_samples": rss.get("samples"),
        "max_ranks_seen": rss.get("max_ranks_seen"),
        "warnings": warning_boxes((s / "vasp.out").read_text(errors="replace")),
    }
    if incar.get("ISPIN") == "2" and "LSORBIT" not in incar:
        record["magnetization_muB"] = number(last(r"^\s*1\s+F=.*mag=\s*" + NUM, osz))
    return record


def export_names(incar):
    names = list(EXPORT_ALWAYS)
    if incar.get("LCHARG", ".TRUE.").upper().startswith(".T"):
        names.append("CHGCAR")
    if "LORBIT" in incar:
        names.append("PROCAR")
    return names


def export(scratch, folder, names):
    """Copy names from scratch into folder: *.copying, SHA-256 check, rename. Returns {name: sha}."""
    present = {p.name for p in folder.iterdir()}
    require(not present & mvi.VASP_OUTPUT, f"{folder} gained VASP output meanwhile; not overwriting")
    hashes = {}
    for name in names:
        source = scratch / name
        require(source.is_file() and not source.is_symlink(), f"missing output {name}")
        sha = digest(source)
        temp = folder / (name + ".copying")
        shutil.copyfile(source, temp)
        require(digest(temp) == sha, f"copy checksum {name}")
        os.replace(temp, folder / name)
        hashes[name] = sha
    return hashes


def safe_remove_scratch(scratch, prepared):
    scratch = Path(scratch)
    require(str(scratch) == prepared["scratch"] and scratch.name.startswith("coge-gb-")
            and not scratch.is_symlink() and scratch.stat().st_uid == os.getuid(), "unsafe scratch removal")
    shutil.rmtree(scratch)


def finish(folder, scratch, exit_code, records):
    folder, scratch = resolve_folder(folder), Path(scratch)
    prepared = json.loads((scratch / "prepare.json").read_text())
    require(prepared["folder"] == str(folder.relative_to(PROJECT_ROOT)), "scratch belongs to another folder")
    info = identify(folder)
    claim = claim_path(records, info)
    try:
        record = validate(scratch, prepared, exit_code)
        incar = mvi.tags((scratch / "INCAR").read_text())
        record.update({k: prepared[k] for k in ("family", "stage", "version", "git", "job", "node",
                                                  "ranks", "kpar", "ncore", "inputs_sha256")})
        record.update(sym=info["sym"], thickness=info["thickness"], point=info["point"],
                      folder=prepared["folder"], validated_at=now())
        record["exported_sha256"] = export(scratch, folder, export_names(incar))
        write_json(folder / "result.json", record)
    except Exception as exc:  # keep scratch and claim; stage folder untouched (or partial copies only)
        failed = Path(records) / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        write_json(failed / f"{info['sym']}_{info['thickness']}_{info['point']}_{prepared['job']}.json",
                   {"folder": prepared["folder"], "scratch": str(scratch), "node": prepared["node"],
                    "job": prepared["job"], "exit_code": exit_code, "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": now()})
        print(f"FAILED {prepared['folder']}: {exc}; scratch kept at {scratch} on {prepared['node']}")
        return 1
    safe_remove_scratch(scratch, prepared)
    if claim.exists():
        claim.unlink()
    print(f"OK {prepared['folder']}: TOTEN {record['toten_eV']:.6f} eV, {record['scf_steps']} steps, "
          f"{record['elapsed_s']:.0f} s, NKPTS {record['nkpts']}")
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("folder")
    p.add_argument("--records", required=True)
    b = sub.add_parser("binary")
    b.add_argument("folder")
    s = sub.add_parser("sample")
    s.add_argument("scratch")
    f = sub.add_parser("finish")
    f.add_argument("folder")
    f.add_argument("scratch")
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--records", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(prepare(args.folder, args.records))
        elif args.command == "binary":
            print(identify(resolve_folder(args.folder))["recipe"]["binary"])
        elif args.command == "sample":
            sample(args.scratch)
        else:
            sys.exit(finish(args.folder, args.scratch, args.exit_code, args.records))
    except (TaskError, mvi.RecipeError) as exc:
        sys.exit(f"STOP: {exc}")


if __name__ == "__main__":
    main()
