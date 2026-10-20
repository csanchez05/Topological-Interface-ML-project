#!/usr/bin/env python3
"""Verified VASP input generator (CLAUDE.md section 7), standard library only.

Recipes are data: one entry per (family, stage) in RECIPES. A derived recipe is
its base recipe plus an explicit override set, so the difference from the
finished bulk runs is visible in one place and is checked by the tests.

Registered recipes
  CoGe_100 / scf-nonsoc             bulk anchor: the tags of the 623 finished
                                    calculations/CoGe_Bulk/CoGe_100 non-soc/scf runs.
                                    Base only; its targets hold finished runs, so
                                    every write is refused.
  CoGe_100_GB / convergence-nonsoc  ENCUT and k-mesh sweep of the [100] GB cells:
                                    ENCUT 350-600 eV at Gamma 1x9x9 and 1xNxN
                                    (N = 3..11) at 500 eV, static, ISMEAR 1,
                                    damped density mixing for the 74 A cell
                                    (AMIN 0.01, AMIX 0.2; v0.2.0), non-magnetic
                                    (ISPIN 1, no MAGMOM; v0.3.0).
  CoGe_100_GB / moment-check        the sweep's e500_k1x9x9 point spin-polarized:
                                    ISPIN 2 from Co 1 muB, the `linmag` mixing of
                                    the 2026-10-13 trials (AMIX_MAG 0.8,
                                    BMIX_MAG 0.0001) and LORBIT 11 for per-atom
                                    moments; its own folder, moment-check/.

Each stage folder gets INCAR (line 1 is the version stamp), KPOINTS, a POSCAR
copy and an absolute POTCAR symlink. A folder holding any VASP output is
refused; identical inputs are skipped; different inputs stop the run, and
then nothing is written for any folder of that call.

Example
  python3 -B src/vasp_inputs/make_vasp_inputs.py --family CoGe_100_GB \\
      --stage convergence-nonsoc --thickness 8x \\
      --sym sym00485 sym00174 sym00016 sym00177 \\
      --write-tasks results/CoGe_interface/100/convergence_8x/tasks.txt --dry-run
"""
import argparse
import filecmp
import os
from pathlib import Path
import re
import subprocess
import sys

VERSION = "0.3.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
INTERFACE = PROJECT_ROOT / "calculations/CoGe_interface"
CATALOGUE = PROJECT_ROOT / "structures/CoGeGrainBoundaries"
VASP_OUTPUT = {"OUTCAR", "OSZICAR", "vasprun.xml", "CHGCAR", "CHG", "WAVECAR", "PROCAR",
               "DOSCAR", "EIGENVAL", "IBZKPT", "CONTCAR", "vasp.out", "vaspout.h5",
               "result.json"}
INPUTS = ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
STAMP = re.compile(r"^# make_vasp_inputs\.py v(\S+) \(git (\S+)\) family=(\S+) stage=(\S+)$")


class RecipeError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise RecipeError(message)


# ---------------------------------------------------------------- recipes (data)

# The live CoGe_100 non-soc/scf INCAR, tag for tag, in the line order of sym00000.
BULK_100_SCF_NONSOC = (
    ("SYSTEM", "CoGe B20 SCF"),
    ("PREC", "Accurate"),
    ("ENCUT", "600"),
    ("EDIFF", "1E-8"),
    ("ALGO", "Normal"),
    ("NELM", "100"),
    ("ISMEAR", "-1"),
    ("SIGMA", "0.05"),
    ("IBRION", "-1"),
    ("NSW", "0"),
    ("ISPIN", "2"),
    ("MAGMOM", "{magmom}"),
    ("LWAVE", ".FALSE."),
    ("LCHARG", ".TRUE."),
    ("LORBIT", "11"),
    ("BANDGAP", "KPOINT"),
    ("ISTART", "0"),
    ("ICHARG", "1"),
    ("ISYM", "2"),
    ("KPAR", "3"),
    ("NCORE", "1"),
    ("LREAL", ".FALSE."),
    ("LMAXMIX", "4"),
)

# ENCUT sweep at Gamma 1x9x9, k sweep at 500 eV; the shared point e500_k1x9x9 is one run.
GB_ENCUT_SWEEP = {"mesh": (1, 9, 9), "encut": (350, 400, 450, 500, 600)}
GB_K_SWEEP = {"encut": 500, "n": (3, 5, 7, 9, 11)}


def point_name(encut, mesh):
    return f"e{encut}_k{mesh[0]}x{mesh[1]}x{mesh[2]}"


def sweep_points(encut_sweep, k_sweep):
    points = {}
    for encut in encut_sweep["encut"]:
        points[point_name(encut, encut_sweep["mesh"])] = (encut, encut_sweep["mesh"])
    for n in k_sweep["n"]:
        mesh = (1, n, n)
        points[point_name(k_sweep["encut"], mesh)] = (k_sweep["encut"], mesh)
    return points


RECIPES = {
    ("CoGe_100", "scf-nonsoc"): {
        "incar": BULK_100_SCF_NONSOC,
        "kpoints_comment": "CoGe bulk Gamma mesh",
        "points": {"scf": (600, (11, 11, 11))},
        "counts": (4, 4),
        "profile": {"ranks": 6, "kpar": {"default": 3}, "ncore": 1},
        "binary": "vasp_std",
        "live": "calculations/CoGe_Bulk/CoGe_100/{sym}/non-soc/scf",  # finished runs (test b)
        "writable": False,
    },
    ("CoGe_100_GB", "convergence-nonsoc"): {
        "base": ("CoGe_100", "scf-nonsoc"),
        # Every change from the bulk anchor; test (b) checks this is the whole difference.
        "set": {
            "SYSTEM": "CoGe B20 [100] GB convergence",
            "ENCUT": "{encut}",       # per point (bulk 600)
            "EDIFF": "1E-6",          # sweeps only; production keeps 1E-8
            "ISMEAR": "1",            # Methfessel-Paxton, SIGMA 0.05 kept
            "LCHARG": ".FALSE.",      # sweep runs seed nothing
            "ICHARG": "2",            # atomic densities (bulk 1 read a CHGCAR)
            "ISPIN": "1",             # v0.3.0: zero moment (user, 2026-10-15); same answer as ISPIN 2 with
                                      # MAGMOM 0 at half the work, and no run can drift into a moment
            "KPAR": "{kpar}",
            "NCORE": "4",
        },
        "insert_after": {
            "NBANDS": ("NCORE", "512"),  # same band count at every point
            # v0.2.0: v0.1.0 sloshed (sym00177 e500_k1x3x3, 100 steps, E -753..-775 eV). VASP's
            # advice for a lattice vector > 50 A is AMIN 0.01; AMIX and AMIX_MAG are half the
            # defaults (0.4, 1.6). BMIX stays 1.0 (Kerker). The path changes, not the answer.
            # v0.3.0: AMIX_MAG dropped with the spin (ISPIN 1). This charge mixing converged
            # sym00177 and sym00190 non-magnetic in 31 and 35 steps (trials 12960, 2026-10-13).
            "AMIX": ("NELM", "0.2"),
            "AMIN": ("AMIX", "0.01"),
        },
        "drop": ("LORBIT", "MAGMOM"),  # energies only, no PROCAR; no moment (v0.3.0)
        "kpoints_comment": "CoGe [100] GB Gamma mesh",
        "points": sweep_points(GB_ENCUT_SWEEP, GB_K_SWEEP),
        "counts": (64, 64),
        "profile": {"ranks": 16, "kpar": {"default": 4, (1, 3, 3): 1}, "ncore": 4},
        "binary": "vasp_std",
        "poscar": "calculations/CoGe_interface/100/{sym}/{thickness}/POSCAR",
        "target": "calculations/CoGe_interface/100/{sym}/{thickness}/convergence/{point}",
        "potcar": "calculations/CoGe_interface/POTCAR",
        "catalogue": "structures/CoGeGrainBoundaries/CoGe_100_GBs/Interface/{sym}/{thickness}/POSCAR",
        "writable": True,
    },
    ("CoGe_100_GB", "moment-check"): {
        # Does the small moment of the 10-13 trials (1x3x3: 1.8 / 1.6 muB, 26 / 44 meV per cell below
        # zero moment) survive at 1x9x9, and on which atoms? The sweep's e500_k1x9x9 point spin-polarized
        # with the trial `linmag` mixing, plus LORBIT (user, 2026-10-15). E0 is compared with the
        # sweep's ISPIN 1 run of the same point.
        "base": ("CoGe_100_GB", "convergence-nonsoc"),
        "set": {"ISPIN": "2"},
        "insert_after": {
            "MAGMOM": ("ISPIN", "{magmom}"),     # Co 1 muB, Ge 0, as bulk and v0.2.0
            "LORBIT": ("LCHARG", "11"),          # per-atom moments in OUTCAR; lm-resolved PROCAR, as bulk
            "AMIX_MAG": ("AMIX", "0.8"),         # v0.2.0
            "BMIX_MAG": ("AMIX_MAG", "0.0001"),  # trial linmag: linear mixing of the moment, no Kerker
        },
        "drop": (),
        "kpoints_comment": "CoGe [100] GB Gamma mesh",
        "points": {"e500_k1x9x9": (500, (1, 9, 9))},
        "counts": (64, 64),
        "profile": {"ranks": 16, "kpar": {"default": 4}, "ncore": 4},
        "binary": "vasp_std",
        "poscar": "calculations/CoGe_interface/100/{sym}/{thickness}/POSCAR",
        "target": "calculations/CoGe_interface/100/{sym}/{thickness}/moment-check/{point}",
        "potcar": "calculations/CoGe_interface/POTCAR",
        "catalogue": "structures/CoGeGrainBoundaries/CoGe_100_GBs/Interface/{sym}/{thickness}/POSCAR",
        "writable": True,
    },
}


def recipe(family, stage):
    require((family, stage) in RECIPES, f"no verified recipe for family={family} stage={stage}")
    return RECIPES[(family, stage)]


def incar_template(family, stage):
    """Ordered (tag, value template) pairs of a recipe, its base expanded."""
    r = recipe(family, stage)
    if "base" not in r:
        return list(r["incar"])
    pairs = [(k, r["set"].get(k, v)) for k, v in incar_template(*r["base"]) if k not in r["drop"]]
    keys = [k for k, _ in pairs]
    require(set(r["set"]) <= set(keys), "override of a tag the base does not have")
    for tag, (after, value) in r.get("insert_after", {}).items():
        require(tag not in keys and after in keys, "bad insert " + tag)
        pairs.insert(keys.index(after) + 1, (tag, value))
        keys = [k for k, _ in pairs]
    return pairs


# ---------------------------------------------------------------- pieces

def tags(text):
    """INCAR text -> {TAG: value}; copied from bulk_110_111_scf_once.py:165-174."""
    result = {}
    for line in text.splitlines():
        for entry in re.split(r"[!#]", line, maxsplit=1)[0].split(";"):
            if "=" in entry:
                key, value = entry.split("=", 1)
                key = key.strip().upper()
                require(key not in result, "duplicate INCAR tag " + key)
                result[key] = value.strip()
    return result


def check_poscar(text, counts):
    """Raise RecipeError unless text is a Co-then-Ge POSCAR with scale 1.0 and these counts."""
    lines = text.splitlines()
    require(lines and text.strip(), "empty POSCAR")
    require(len(lines) >= 8, "truncated POSCAR header")
    try:
        scale = float(lines[1].split()[0])
    except (IndexError, ValueError):
        raise RecipeError("unreadable scale line (malformed POSCAR)")
    require(abs(scale - 4.6294498444) > 1e-6, "scale 4.6294498444: double-scaled catalogue POSCAR?")
    require(scale == 1.0, f"scale must be 1.0, got {scale}")
    for row in lines[2:5]:
        require(len(row.split()) == 3, "lattice vector needs 3 numbers")
        [float(x) for x in row.split()]
    require(lines[5].split() == ["Co", "Ge"], "species must be 'Co Ge' (Co first): " + lines[5].strip())
    try:
        found = tuple(int(x) for x in lines[6].split())
    except ValueError:
        raise RecipeError("unreadable counts line")
    require(found == tuple(counts), f"atom counts {found} != {tuple(counts)}")
    require(lines[7].strip().lower().startswith("d"), "Direct coordinates required")
    atoms = sum(counts)
    rows = [row.split() for row in lines[8:8 + atoms]]
    require(len(rows) == atoms and all(len(r) >= 3 for r in rows), f"need {atoms} coordinate rows")
    for r in rows:
        [float(x) for x in r[:3]]
    return {"atoms": atoms, "counts": tuple(counts)}


def kpar_for(r, mesh):
    return r["profile"]["kpar"].get(tuple(mesh), r["profile"]["kpar"]["default"])


def check_profile(r, mesh):
    ranks, kpar, ncore = r["profile"]["ranks"], kpar_for(r, mesh), r["profile"]["ncore"]
    require(ranks % kpar == 0 and (ranks // kpar) % ncore == 0, "ranks/KPAR/NCORE do not divide")
    return ranks, kpar, ncore


def render_incar(family, stage, point, counts, git_hash):
    r = recipe(family, stage)
    encut, mesh = r["points"][point]
    _, kpar, _ = check_profile(r, mesh)
    values = {"encut": str(encut), "kpar": str(kpar),
              "magmom": f"{counts[0]}*1.0 {counts[1]}*0.0"}
    lines = [f"# make_vasp_inputs.py v{VERSION} (git {git_hash}) family={family} stage={stage}"]
    for key, template in incar_template(family, stage):
        lines.append(f"{key} = {template.format(**values)}")
    return "\n".join(lines) + "\n"


def render_kpoints(family, stage, point):
    r = recipe(family, stage)
    _, mesh = r["points"][point]
    return f"{r['kpoints_comment']}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"


def mask_stamp(text):
    """The INCAR text with the git hash of its stamp line replaced by '*'."""
    first, newline, rest = text.partition("\n")
    match = STAMP.match(first)
    if match:
        first = first.replace(f"(git {match.group(2)})", "(git *)")
    return first + newline + rest


def same_incar(old, new):
    """Equal apart from the git hash in the stamp line (same VERSION, family, stage)."""
    return mask_stamp(old) == mask_stamp(new)


# ---------------------------------------------------------------- git guard

def git(*args):
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True)


def code_hash():
    """Short hash of the last commit touching src/vasp_inputs; refuses uncommitted code."""
    rel = os.path.relpath(CODE_DIR, PROJECT_ROOT)
    status = git("status", "--porcelain", "--", rel)
    require(status.returncode == 0, "git is not usable here: " + status.stderr.strip())
    require(not status.stdout.strip(), "src/vasp_inputs has uncommitted changes; commit before writing:\n"
            + status.stdout)
    log = git("log", "-1", "--format=%h", "--", rel)
    require(log.returncode == 0 and log.stdout.strip(), "src/vasp_inputs has no commit")
    return log.stdout.strip()


# ---------------------------------------------------------------- one folder

def plan_folder(family, stage, sym, thickness, point, git_hash, root=PROJECT_ROOT):
    """Return (target, files) with files = {name: text} plus the POTCAR link target."""
    r = recipe(family, stage)
    require(re.fullmatch(r"sym\d{5}", sym), "sym must look like symNNNNN: " + sym)
    require(point in r["points"], f"unknown point {point}; known: {', '.join(r['points'])}")
    fields = {"sym": sym, "thickness": thickness, "point": point}
    poscar_path = root / r["poscar"].format(**fields)
    require(poscar_path.is_file(), f"missing POSCAR {poscar_path}")
    poscar = poscar_path.read_text()
    try:
        check_poscar(poscar, r["counts"])
    except RecipeError as exc:
        raise RecipeError(f"{poscar_path}: {exc}")
    if r["catalogue"]:
        source = root / r["catalogue"].format(**fields)
        require(source.is_file() and filecmp.cmp(source, poscar_path, shallow=False),
                f"{poscar_path} is not a byte copy of the catalogue {source}")
    target = root / r["target"].format(**fields)
    files = {"INCAR": render_incar(family, stage, point, r["counts"], git_hash),
             "KPOINTS": render_kpoints(family, stage, point),
             "POSCAR": poscar}
    return target, files, root / r["potcar"]


def classify(target, files, potcar):
    """'refused' | 'identical' | 'different: why' | 'new'."""
    if not target.exists():
        return "new"
    names = {p.name for p in target.iterdir()}
    if names & VASP_OUTPUT:
        return "refused"
    extra = names - set(INPUTS)
    if extra:
        return f"different: unexpected file(s) {sorted(extra)}"
    for name in INPUTS:
        path = target / name
        if not os.path.lexists(path):
            return f"different: {name} missing"
    if not (target / "POTCAR").is_symlink() or os.readlink(target / "POTCAR") != str(potcar):
        return "different: POTCAR is not the expected symlink"
    if not same_incar((target / "INCAR").read_text(), files["INCAR"]):
        return "different: INCAR"
    for name in ("KPOINTS", "POSCAR"):
        if (target / name).read_text() != files[name]:
            return f"different: {name}"
    return "identical"


def write_folder(target, files, potcar):
    target.mkdir(parents=True, exist_ok=False)
    for name, text in files.items():
        (target / name).write_text(text)
    (target / "POTCAR").symlink_to(potcar)


def generate(family, stage, syms, thickness, points, dry_run, git_hash, root=PROJECT_ROOT):
    """Classify every folder first; write the new ones only if no folder has a problem."""
    r = recipe(family, stage)
    points = list(points or r["points"])
    counts = {"written": 0, "identical": 0, "refused": 0, "new (dry-run)": 0}
    problems, planned, refused = [], [], []
    require(r["writable"], f"{family}/{stage} is a base recipe; its targets hold finished runs")
    require((root / r["potcar"]).is_file(), f"missing POTCAR master {root / r['potcar']}")
    for sym in syms:
        for point in points:
            try:
                target, files, potcar = plan_folder(family, stage, sym, thickness, point, git_hash, root)
            except RecipeError as exc:
                problems.append(f"{sym} {point}: {exc}")
                continue
            state = classify(target, files, potcar)
            if state == "refused":
                counts["refused"] += 1
                refused.append(target)
            elif state.startswith("different"):
                problems.append(f"{target}: {state}")
            else:
                planned.append((state, target, files, potcar))
    new = sum(state == "new" for state, *_ in planned)
    counts["identical"] = len(planned) - new
    if problems:
        counts["not written"] = new
        return counts, problems, [], refused
    ready = []
    for state, target, files, potcar in planned:
        if state == "new" and not dry_run:
            write_folder(target, files, potcar)
        ready.append(target)
    counts["new (dry-run)" if dry_run else "written"] = new
    return counts, problems, ready, refused


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, help="e.g. CoGe_100_GB")
    parser.add_argument("--stage", required=True, help="e.g. convergence-nonsoc")
    parser.add_argument("--sym", nargs="+", required=True, help="symNNNNN folders")
    parser.add_argument("--thickness", default="8x", help="GB thickness folder (default 8x)")
    parser.add_argument("--points", nargs="+", help="subset of sweep points (default: all of the recipe)")
    parser.add_argument("--write-tasks", type=Path,
                        help="write the ready folders (project-relative, one per line) to this file")
    parser.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = parser.parse_args(argv)

    try:
        git_hash = "dryrun" if args.dry_run else code_hash()
        counts, problems, ready, refused = generate(args.family, args.stage, args.sym, args.thickness,
                                                    args.points, args.dry_run, git_hash)
    except RecipeError as exc:
        sys.exit(f"STOP: {exc}")
    print(f"make_vasp_inputs.py v{VERSION} family={args.family} stage={args.stage} "
          f"({len(args.sym)} sym x {len(args.points or recipe(args.family, args.stage)['points'])} points)")
    print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    for target in refused:
        print(f"  refused (holds VASP output): {os.path.relpath(target, PROJECT_ROOT)}")
    if problems:
        print(f"STOP: {len(problems)} problem(s):")
        for line in problems[:20]:
            print("  " + line)
        sys.exit(1)
    if args.write_tasks and not args.dry_run:
        text = "".join(os.path.relpath(t, PROJECT_ROOT) + "\n" for t in ready)
        path = args.write_tasks if args.write_tasks.is_absolute() else PROJECT_ROOT / args.write_tasks
        if path.exists() and path.read_text() != text:
            sys.exit(f"STOP: {path} exists with different content; move it away first")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"  task list: {os.path.relpath(path, PROJECT_ROOT)} ({len(ready)} folders)")


if __name__ == "__main__":
    main()
