#!/usr/bin/env python3
"""SCF trials for the [100] GB convergence sweep: test runs only, nothing is written to calculations/.

The v0.2.0 run sym00177 e500_k1x3x3 (job 12919) stopped at NELM 100 without converging. The moment
fell slowly from 64 to about 7 muB, and the charge residual rms(c) never dropped below 1.3. These
trials run four mixing routes on two structures at once, so a single batch decides the v0.3.0
recipe. ALGO, ISMEAR/SIGMA, ENCUT, k-mesh, EDIFF, ISPIN 2 and the 1 muB Co start all stay as in
v0.2.0 (user, 2026-10-13).

Each trial's INCARs are the v0.2.0 recipe, rendered by make_vasp_inputs.py, plus the named override
set in VARIANTS; line 1 is replaced by a trial stamp. A trial has one or two VASP runs in one
node-local scratch (run1/, run2/). A seeded run2 reads run1's CHGCAR and starts only if run1
reached EDIFF.

  scf_trials.py list                                 the 8 trials and each run's INCAR change from v0.2.0
  scf_trials.py submit --records DIR [--nodelist N] [--dry-run]
                                                     submits one array; submit.json keeps the code's SHA-256
  scf_trials.py prepare INDEX --records DIR          (array task) makes the scratch, prints its path
  scf_trials.py runs SCRATCH                         prints the run folders, in order
  scf_trials.py stage SCRATCH RUN                    before RUN: previous run converged? copy its CHGCAR
  scf_trials.py summarize SCRATCH --records DIR [--label L]
                                                     DIR/trials/<label>.json; a past v0.2.0 scratch works too
  scf_trials.py table --records DIR                  DIR/trials/*.json -> DIR/trials.tsv
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_vasp_inputs as mvi  # noqa: E402
import vasp_task as vt  # noqa: E402

PROJECT_ROOT = mvi.PROJECT_ROOT
FAMILY, STAGE, THICKNESS, POINT = "CoGe_100_GB", "convergence-nonsoc", "8x", "e500_k1x3x3"
BASE_VERSION = "0.2.0"
STRUCTURES = ("sym00177", "sym00190")  # no close pair (failed in 12919); 1.80 A, closest to the limit
COMMON = {"NELM": "200"}
SEED = {"set": {"ISPIN": "1", "LCHARG": ".TRUE."}, "drop": ("MAGMOM", "AMIX_MAG")}
VARIANTS = {
    # Kerker damping stays on the charge (74 A axis), linear mixing for the magnetization
    "linmag": ({"set": {"BMIX_MAG": "0.0001"}},),
    # the VASP-wiki recipe for hard magnetic systems: linear mixing for both
    "wiki": ({"set": {"BMIX": "0.0001", "BMIX_MAG": "0.0001"}},),
    # non-magnetic charge first, then ISPIN 2 from that CHGCAR with MAGMOM 64*1.0
    "nmseed": (SEED, {"set": {"ICHARG": "1"}}),
    "nmseed_linmag": (SEED, {"set": {"ICHARG": "1", "BMIX_MAG": "0.0001"}}),
}
INSERT_AFTER = {"BMIX": "AMIX", "BMIX_MAG": "AMIX_MAG"}
TRIALS = [(sym, variant) for sym in STRUCTURES for variant in VARIANTS]
RANKS = 16
SLURM_SCRIPT = Path(__file__).resolve().parent / "scf_trials.slurm"
CODE_FILES = ("scf_trials.py", "scf_trials.slurm", "make_vasp_inputs.py", "vasp_task.py")
NUM = vt.NUM
require = vt.require


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def trial_name(sym, variant):
    return f"{sym}_{variant}"


# ---------------------------------------------------------------- inputs

def base_files(sym, root=PROJECT_ROOT):
    """The v0.2.0 inputs of (sym, POINT): checked POSCAR, KPOINTS, INCAR and the POTCAR link target."""
    require(mvi.VERSION == BASE_VERSION, f"trials compare against v{BASE_VERSION}, generator is v{mvi.VERSION}")
    _, files, potcar = mvi.plan_folder(FAMILY, STAGE, sym, THICKNESS, POINT, "trial", root)
    return files, potcar


def apply(base_incar, run, name, index, total):
    """Base INCAR text -> trial INCAR text: COMMON and run['set'] applied, run['drop'] removed."""
    pairs = []
    for line in base_incar.splitlines()[1:]:
        key, value = (part.strip() for part in line.split("=", 1))
        pairs.append([key, value])
    changes = {**COMMON, **run.get("set", {})}
    for key in run.get("drop", ()):
        require(any(k == key for k, _ in pairs), f"drop of absent tag {key}")
        pairs = [p for p in pairs if p[0] != key]
    for key, value in changes.items():
        found = [p for p in pairs if p[0] == key]
        if found:
            found[0][1] = value
        else:
            after = INSERT_AFTER.get(key)
            keys = [k for k, _ in pairs]
            require(after in keys, f"no place for new tag {key}")
            pairs.insert(keys.index(after) + 1, [key, value])
    stamp = (f"# scf_trials.py trial={name} run={index}/{total} base=make_vasp_inputs.py v{BASE_VERSION} "
             f"family={FAMILY} stage={STAGE} point={POINT}")
    return "\n".join([stamp] + [f"{k} = {v}" for k, v in pairs]) + "\n"


def trial_incars(sym, variant, root=PROJECT_ROOT):
    files, _ = base_files(sym, root)
    runs = VARIANTS[variant]
    name = trial_name(sym, variant)
    return [apply(files["INCAR"], run, name, i + 1, len(runs)) for i, run in enumerate(runs)]


def incar_diff(base, trial):
    """{tag: (base value or None, trial value or None)} for the tags that differ."""
    a, b = mvi.tags(base), mvi.tags(trial)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


# ---------------------------------------------------------------- prepare / stage

def prepare(index, records, environ=os.environ, scratch_base=None, root=PROJECT_ROOT):
    require(0 <= index < len(TRIALS), f"trial index {index} not in 0..{len(TRIALS) - 1}")
    sym, variant = TRIALS[index]
    require(environ.get("SLURM_NTASKS") == str(RANKS), f"trials need {RANKS} ranks, "
            f"allocation has SLURM_NTASKS={environ.get('SLURM_NTASKS')}")
    files, potcar = base_files(sym, root)
    incars = trial_incars(sym, variant, root)
    job = f"{environ.get('SLURM_ARRAY_JOB_ID', environ.get('SLURM_JOB_ID', 'nojob'))}" \
          f"_{environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    base = scratch_base or environ.get("TMPDIR") or "/tmp"
    scratch = Path(tempfile.mkdtemp(prefix=f"coge-trial-{sym}-{variant}-{job}-", dir=base))
    free = shutil.disk_usage(scratch).free / 1024**3
    require(free >= vt.MIN_FREE_GIB, f"only {free:.1f} GiB free in {base}")
    shas = {}
    for i, incar in enumerate(incars, 1):
        run = scratch / f"run{i}"
        run.mkdir()
        (run / "INCAR").write_text(incar)
        (run / "KPOINTS").write_text(files["KPOINTS"])
        (run / "POSCAR").write_text(files["POSCAR"])
        (run / "POTCAR").symlink_to(potcar)
        shas[run.name] = {n: vt.digest(run / n) for n in ("INCAR", "KPOINTS", "POSCAR")}
    vt.write_json(scratch / "prepare.json", {
        "trial": trial_name(sym, variant), "index": index, "sym": sym, "variant": variant,
        "point": POINT, "base_version": BASE_VERSION, "runs": VARIANTS[variant], "common": COMMON,
        "job": job, "node": os.uname().nodename, "ranks": RANKS, "scratch": str(scratch),
        "records": str(records), "inputs_sha256": shas, "potcar": str(potcar), "prepared_at": now()})
    return scratch


def run_dirs(scratch):
    return sorted((p for p in Path(scratch).iterdir() if re.fullmatch(r"run\d+", p.name)),
                  key=lambda p: int(p.name[3:]))


def stage(scratch, run):
    """Before run k > 1: run k-1 must have reached EDIFF; its CHGCAR is copied in (seeded start)."""
    scratch = Path(scratch)
    k = int(re.fullmatch(r"run(\d+)", run).group(1))
    if k == 1:
        return "run1: atomic start"
    previous, target = scratch / f"run{k - 1}", scratch / run
    require((previous / "OUTCAR").is_file(), f"{previous.name} has no OUTCAR")
    parsed = parse_run(previous)
    require(parsed["converged"], f"{previous.name} did not reach EDIFF; {run} is not started")
    source = previous / "CHGCAR"
    require(source.is_file() and source.stat().st_size > 0, f"{previous.name}/CHGCAR missing or empty")
    sha = vt.digest(source)
    shutil.copyfile(source, target / "CHGCAR")
    require(vt.digest(target / "CHGCAR") == sha, "CHGCAR copy checksum")
    return f"{run}: seeded from {previous.name}/CHGCAR ({source.stat().st_size} bytes)"


# ---------------------------------------------------------------- parse / summarize

def parse_oszicar(text):
    steps = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("DAV:", "RMM:"):
            steps.append({"n": int(parts[1]), "E": vt.number(parts[2]), "dE": vt.number(parts[3]),
                          "rms": vt.number(parts[6]) if len(parts) > 6 else None,
                          "rmsc": vt.number(parts[7]) if len(parts) > 7 else None})
    final = re.search(r"^\s*\d+\s+F=\s*" + NUM + r".*?(?:mag=\s*" + NUM + r")?\s*$", text, re.M)
    return steps, final


def parse_run(run):
    """One VASP run folder -> record (works on finished, unconverged and killed runs)."""
    run = Path(run)
    out = (run / "OUTCAR").read_text(errors="replace") if (run / "OUTCAR").is_file() else ""
    osz = (run / "OSZICAR").read_text(errors="replace") if (run / "OSZICAR").is_file() else ""
    incar = mvi.tags((run / "INCAR").read_text())
    steps, final = parse_oszicar(osz)
    mags = [float(m) for m in re.findall(r"^\s*number of electron\s+\S+\s+magnetization\s+" + NUM, out, re.M)]

    def found(pattern, cast=vt.number):
        hits = re.findall(pattern, out, re.M)
        return cast(hits[-1]) if hits else None

    nelm = int(incar.get("NELM", 60))
    finished = "General timing and accounting informations for this job" in out
    converged = finished and "aborting loop because EDIFF is reached" in out and 0 < len(steps) < nelm
    rss = json.loads((run / "rss.json").read_text()) if (run / "rss.json").exists() else {}
    record = {
        "run": run.name, "incar": incar, "finished": finished, "converged": converged,
        "steps": len(steps), "nelm": nelm,
        "ispin_outcar": found(r"^\s*ISPIN\s*=\s*(\d+)", int),
        "icharg_outcar": found(r"^\s*ICHARG\s*=\s*(-?\d+)", int),
        "elapsed_s": found(r"Elapsed time \(sec\):\s*" + NUM),
        "toten_eV": found(r"free\s+energy\s+TOTEN\s*=\s*" + NUM),
        "e0_eV": found(r"energy\(sigma->0\)\s*=\s*" + NUM),
        "final_mag_muB": vt.number(final.group(2)) if final and final.group(2) else None,
        "initial_mag_muB": mags[0] if mags else None,
        "last_rmsc": next((s["rmsc"] for s in reversed(steps) if s["rmsc"] is not None), None),
        "peak_rss_bytes": rss.get("peak_rss_bytes"),
        "trace": {"E": [s["E"] for s in steps], "dE": [s["dE"] for s in steps],
                  "rmsc": [s["rmsc"] for s in steps], "mag": mags},
    }
    if incar.get("ICHARG") == "1" and incar.get("ISPIN", "2") == "2":
        # VASP takes the moment from MAGMOM when the CHGCAR it reads is non-magnetic
        expected = sum(float(n) * float(v) for n, v in re.findall(r"(\d+)\*([-\d.]+)", incar["MAGMOM"]))
        record["seed_check"] = {
            "icharg_1_used": record["icharg_outcar"] == 1,
            "initial_mag_muB": record["initial_mag_muB"], "magmom_total_muB": expected,
            "ok": record["icharg_outcar"] == 1 and record["initial_mag_muB"] is not None
            and abs(record["initial_mag_muB"] - expected) < 0.5 * expected}
    return record


def summarize(scratch, records, label=None):
    scratch = Path(scratch)
    prepared = json.loads((scratch / "prepare.json").read_text()) if (scratch / "prepare.json").exists() else {}
    runs = run_dirs(scratch) or [scratch]  # a v0.2.0 runner scratch is one run, files directly inside
    parsed = [parse_run(r) for r in runs if (r / "INCAR").is_file()]
    ranks = prepared.get("ranks", RANKS)
    last = parsed[-1] if parsed else {}
    spin_converged = bool(parsed) and all(r["converged"] for r in parsed) and \
        last.get("incar", {}).get("ISPIN", "2") == "2"
    seeds = [r["seed_check"]["ok"] for r in parsed if "seed_check" in r]
    label = label or prepared.get("trial")
    require(label, f"{scratch} has no prepare.json; give --label")
    record = {
        "label": label, "sym": prepared.get("sym"), "variant": prepared.get("variant"),
        "point": prepared.get("point", POINT), "job": prepared.get("job"), "node": prepared.get("node"),
        "scratch": str(scratch), "ranks": ranks, "runs": parsed,
        "converged": spin_converged and all(seeds),
        "core_hours": round(sum(r["elapsed_s"] or 0 for r in parsed) * ranks / 3600, 2),
        "summarized_at": now(),
    }
    if record["sym"] is None:
        match = re.search(r"coge-gb-(sym\d{5})-", scratch.name)
        record["sym"] = match.group(1) if match else None
    out = Path(records) / "trials"
    out.mkdir(parents=True, exist_ok=True)
    vt.write_json(out / f"{record['label']}.json", record)
    return record


COLUMNS = ("label", "sym", "variant", "converged", "core_hours", "nm_steps", "nm_converged", "steps",
           "elapsed_h", "toten_eV", "e_nm_eV", "final_mag_muB", "last_rmsc", "seed_ok", "peak_rss_GB", "node",
           "job")


def table_row(t):
    runs = t["runs"]
    spin = runs[-1] if runs else {}
    nm = runs[0] if len(runs) > 1 else None
    seed = spin.get("seed_check")
    cells = {
        "label": t["label"], "sym": t["sym"], "variant": t["variant"] or "v0.2.0", "converged": t["converged"],
        "core_hours": t["core_hours"], "nm_steps": nm["steps"] if nm else "",
        "nm_converged": nm["converged"] if nm else "", "steps": spin.get("steps"),
        "elapsed_h": round(spin["elapsed_s"] / 3600, 2) if spin.get("elapsed_s") else "",
        "toten_eV": spin.get("toten_eV"), "e_nm_eV": nm["toten_eV"] if nm else "",
        "final_mag_muB": spin.get("final_mag_muB"), "last_rmsc": spin.get("last_rmsc"),
        "seed_ok": seed["ok"] if seed else "",
        "peak_rss_GB": round(max(r["peak_rss_bytes"] or 0 for r in runs) / 1e9, 1) if runs else "",
        "node": t["node"], "job": t["job"]}
    return "\t".join("" if cells[c] is None else str(cells[c]) for c in COLUMNS)


def table(records):
    records = Path(records)
    trials = [json.loads(p.read_text()) for p in sorted((records / "trials").glob("*.json"))]
    require(trials, f"no trial records in {records / 'trials'}")
    text = "\t".join(COLUMNS) + "\n" + "".join(table_row(t) + "\n" for t in trials)
    (records / "trials.tsv").write_text(text)
    return text


# ---------------------------------------------------------------- list / submit

def listing(root=PROJECT_ROOT):
    lines = []
    for index, (sym, variant) in enumerate(TRIALS):
        files, _ = base_files(sym, root)
        lines.append(f"[{index}] {trial_name(sym, variant)}")
        for i, incar in enumerate(trial_incars(sym, variant, root), 1):
            diff = incar_diff(files["INCAR"], incar)
            lines.append(f"    run{i}: " + ", ".join(f"{k} {a or '-'} -> {b or '-'}" for k, (a, b) in diff.items()))
    return "\n".join(lines)


def submit(records, nodelist=None, mem="32G", time_limit="1-00:00:00", dry_run=False):
    records = Path(records)
    records = records if records.is_absolute() else PROJECT_ROOT / records
    print(listing())
    command = ["sbatch", "--parsable", f"--array=0-{len(TRIALS) - 1}%{len(TRIALS)}", f"--mem={mem}",
               f"--ntasks={RANKS}", f"--ntasks-per-node={RANKS}", f"--time={time_limit}",
               f"--output={records / 'logs'}/%A_%a.out"]
    if nodelist:
        command.append(f"--nodelist={nodelist}")
    command += [str(SLURM_SCRIPT), str(records)]
    print(" ".join(command))
    if dry_run:
        return None
    require(not (records / "submit.json").exists(), f"{records}/submit.json exists: trials already submitted")
    (records / "logs").mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    job = result.stdout.strip().split(";")[0]
    git = mvi.git("rev-parse", "--short", "HEAD").stdout.strip()
    status = mvi.git("status", "--porcelain", "--", "src/vasp_inputs").stdout
    vt.write_json(records / "submit.json", {
        "command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        "job_id": job if re.fullmatch(r"\d+", job) else None, "trials": [trial_name(*t) for t in TRIALS],
        "git_head": git, "src_vasp_inputs_uncommitted": status.splitlines(),
        "code_sha256": {n: vt.digest(here / n) for n in CODE_FILES}, "submitted_at": now(),
        "user": os.environ.get("USER")})
    require(result.returncode == 0, f"sbatch failed: {result.stderr.strip()}")
    print(f"submitted array job {job}; records in {records}")
    return job


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    s = sub.add_parser("submit")
    s.add_argument("--records", required=True)
    s.add_argument("--nodelist")
    s.add_argument("--mem", default="32G")
    s.add_argument("--time", default="1-00:00:00")
    s.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("index", type=int)
    p.add_argument("--records", required=True)
    r = sub.add_parser("runs")
    r.add_argument("scratch")
    g = sub.add_parser("stage")
    g.add_argument("scratch")
    g.add_argument("run")
    m = sub.add_parser("summarize")
    m.add_argument("scratch")
    m.add_argument("--records", required=True)
    m.add_argument("--label")
    t = sub.add_parser("table")
    t.add_argument("--records", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(listing())
        elif args.command == "submit":
            submit(args.records, args.nodelist, args.mem, args.time, args.dry_run)
        elif args.command == "prepare":
            print(prepare(args.index, args.records))
        elif args.command == "runs":
            print("\n".join(p.name for p in run_dirs(args.scratch)))
        elif args.command == "stage":
            print(stage(args.scratch, args.run))
        elif args.command == "summarize":
            t = summarize(args.scratch, args.records, args.label)
            print(f"{t['label']}: converged {t['converged']}, {t['core_hours']} core-h, "
                  + "; ".join(f"{r['run']} {r['steps']} steps (converged {r['converged']})" for r in t["runs"]))
        else:
            print(table(args.records), end="")
    except (vt.TaskError, mvi.RecipeError) as exc:
        sys.exit(f"STOP: {exc}")


if __name__ == "__main__":
    main()
