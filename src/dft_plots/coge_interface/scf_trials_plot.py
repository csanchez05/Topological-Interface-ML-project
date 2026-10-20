#!/usr/bin/env python3
"""Plot the 2026-10-13 SCF trials of the [100] GB sweep: which mixing route converges, and how fast?

Reads the per-trial records written by src/vasp_inputs/scf_trials.py (RECORDS/trials/*.json) and
draws one figure with two rows and two columns:
  rows     charge residual rms(c) (log scale) and total moment (muB), per SCF step
  columns  sym00177 and sym00190
  x axis   wall-clock hours on 16 ranks; each run's steps are spread evenly over its elapsed time
One line per variant. The ISPIN 1 seed run of the nmseed variants is drawn dashed. The v0.2.0 run
(job 12919) is drawn in gray for comparison. An end dot means the run reached EDIFF; an end cross
means it stopped without converging.

  src/dft_plots/.venv/bin/python -B src/dft_plots/coge_interface/scf_trials_plot.py \\
      --records results/CoGe_interface/100/convergence_8x/scf_trials_20261013 [--out PNG] [--overwrite]
  ... scf_trials_plot.py --smoke OUTDIR     synthetic records; the figure is titled SYNTHETIC
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("linmag", "wiki", "nmseed", "nmseed_linmag")
COLORS = {"linmag": "#2a78d6", "wiki": "#eb6834", "nmseed": "#1baf7a", "nmseed_linmag": "#eda100",
          None: "#8f8e89"}  # reference palette slots 1-4 (validated, light); gray = the v0.2.0 run
LABELS = {"linmag": "linmag: BMIX_MAG 0.0001", "wiki": "wiki: BMIX, BMIX_MAG 0.0001",
          "nmseed": "nmseed: ISPIN 1 first, then ICHARG 1", "nmseed_linmag": "nmseed_linmag: both",
          None: "v0.2.0 (job 12919, NELM 100)"}
SHORT = {"linmag": "linmag", "wiki": "wiki", "nmseed": "nmseed", "nmseed_linmag": "nmseed_linmag",
         None: "v0.2.0"}
STRUCTURES = {"sym00177": "sym00177 (closest pair 2.40 Å)", "sym00190": "sym00190 (closest pair 1.80 Å)"}
SURFACE, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5"


def step_hours(run, fallback):
    """Hours per SCF step of one run: elapsed / steps, else the fallback for its ISPIN."""
    if run.get("elapsed_s") and run.get("steps"):
        return run["elapsed_s"] / run["steps"] / 3600
    return fallback[run["incar"].get("ISPIN", "2")]


def fallbacks(trials):
    """Median hours per step by ISPIN over the runs that have timing (for killed runs)."""
    per = {"1": [], "2": []}
    for t in trials:
        for r in t["runs"]:
            if r.get("elapsed_s") and r.get("steps"):
                per[r["incar"].get("ISPIN", "2")].append(r["elapsed_s"] / r["steps"] / 3600)
    med = {k: sorted(v)[len(v) // 2] if v else None for k, v in per.items()}
    med["2"] = med["2"] or 211 / 3600  # 12919: 211 s per step on 16 ranks
    med["1"] = med["1"] or med["2"] / 2
    return med


def series(trial, fallback):
    """[(segment dict)] per run: x/rmsc, x/mag, dashed (non-magnetic), end state."""
    start, out = 0.0, []
    for run in trial["runs"]:
        dt = step_hours(run, fallback)
        n = run["steps"]
        x = [start + (i + 1) * dt for i in range(n)]
        pts = [(xi, v) for xi, v in zip(x, run["trace"]["rmsc"]) if v is not None]
        mags = run["trace"]["mag"]
        mx = [start + n * dt * i / max(len(mags) - 1, 1) for i in range(len(mags))]
        out.append({"rmsc": pts, "mag": list(zip(mx, mags)), "dashed": run["incar"].get("ISPIN") == "1",
                    "converged": run["converged"], "end": start + n * dt, "run": run["run"]})
        start += n * dt
    return out


def synthetic_trials(seed=7):
    """Made-up records with the shape of scf_trials.py output (for --smoke only)."""
    rnd = random.Random(seed)

    def run(name, ispin, icharg, steps, converged, mag0):
        rmsc = [None] * 4 + [3 * math.exp(-k / (steps / 6 if converged else 1e9)) * (1 + 0.3 * rnd.random())
                             for k in range(steps - 4)]
        mags = [] if ispin == 1 else [mag0 * math.exp(-k / 15) + 0.5 for k in range(steps + 2)]
        return {"run": name, "incar": {"ISPIN": str(ispin), "ICHARG": str(icharg)}, "steps": steps,
                "converged": converged, "elapsed_s": steps * (105 if ispin == 1 else 211),
                "trace": {"rmsc": rmsc, "mag": mags}}

    trials = []
    for sym in STRUCTURES:
        trials.append({"label": f"{sym}_linmag", "sym": sym, "variant": "linmag",
                       "runs": [run("run1", 2, 2, 90, True, 64)]})
        trials.append({"label": f"{sym}_wiki", "sym": sym, "variant": "wiki",
                       "runs": [run("run1", 2, 2, 200, False, 64)]})
        for v in ("nmseed", "nmseed_linmag"):
            trials.append({"label": f"{sym}_{v}", "sym": sym, "variant": v,
                           "runs": [run("run1", 1, 2, 40, True, 0), run("run2", 2, 1, 60, True, 64)]})
    trials.append({"label": "sym00177_v0.2.0_12919", "sym": "sym00177", "variant": None,
                   "runs": [run("run1", 2, 2, 100, False, 64)]})
    return trials


def label_positions(ends, lo, hi, gap):
    """Spread label y positions (axis units, log10 for rms(c)) so none is closer than gap."""
    order = sorted(range(len(ends)), key=lambda i: ends[i])
    pos = [None] * len(ends)
    last = -math.inf
    for i in order:
        pos[i] = max(min(max(ends[i], lo), hi), last + gap)
        last = pos[i]
    return pos


def draw(trials, out, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fallback = fallbacks(trials)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex="col", facecolor=SURFACE)
    xmax = 0.0
    for col, sym in enumerate(STRUCTURES):
        top, bottom = axes[0][col], axes[1][col]
        ends = []
        for t in sorted((t for t in trials if t["sym"] == sym),
                        key=lambda t: -1 if t["variant"] is None else VARIANTS.index(t["variant"])):
            color = COLORS[t["variant"]]
            width = 1.4 if t["variant"] is None else 2.0
            segs = series(t, fallback)
            for seg in segs:
                if seg["rmsc"]:
                    xs, ys = zip(*seg["rmsc"])
                    top.plot(xs, ys, color=color, lw=width if not seg["dashed"] else 1.4,
                             ls=(0, (4, 2)) if seg["dashed"] else "-", solid_capstyle="round", zorder=3)
                if seg["mag"]:
                    xs, ys = zip(*seg["mag"])
                    bottom.plot(xs, ys, color=color, lw=width, solid_capstyle="round", zorder=3)
                xmax = max(xmax, seg["end"])
            last = segs[-1]
            if last["rmsc"]:
                x, y = last["rmsc"][-1]
                top.plot([x], [y], marker="o" if last["converged"] else "X", ms=8, color=color,
                         mec=SURFACE, mew=1.5, zorder=4)
                ends.append((x, y, SHORT[t["variant"]]))
            if len(segs) > 1 and segs[0]["rmsc"]:  # seed -> magnetic run boundary
                x, y = segs[0]["rmsc"][-1]
                top.plot([x], [y], marker="o", ms=6, mfc=SURFACE, mec=color, mew=1.5, zorder=4)
        top.set_yscale("log")
        top.set_title(STRUCTURES[sym], loc="left", fontsize=10, color=INK, pad=6)
        bottom.set_xlabel("wall-clock hours on 16 ranks (qcg3-0)")
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=RULE, lw=0.6, alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        if ends:  # direct labels at the line ends, text in ink (identity is the end marker beside it)
            lo, hi = top.get_ylim()
            logs = label_positions([math.log10(y) for _, y, _ in ends], math.log10(lo), math.log10(hi),
                                   (math.log10(hi) - math.log10(lo)) / 16)
            for (x, y, text), ly in zip(ends, logs):
                moved = abs(ly - math.log10(y)) > 1e-9
                top.annotate(text, (x, y), xytext=(9, 10 ** ly), textcoords=("offset points", "data"),
                             va="center", fontsize=8, color=INK2, zorder=5,
                             arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkA": 1,
                                         "shrinkB": 5} if moved else None)
    for row in axes:
        for ax in row:
            ax.set_xlim(0, xmax * 1.18 if xmax else 1)
    axes[0][0].set_ylabel("charge residual rms(c), log scale")
    axes[1][0].set_ylabel("total moment (μB)")

    handles = [Line2D([], [], color=COLORS[v], lw=2, label=LABELS[v]) for v in VARIANTS]
    handles.append(Line2D([], [], color=COLORS[None], lw=1.4, label=LABELS[None]))
    handles += [Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="ISPIN 1 seed run (no moment)"),
                Line2D([], [], color=INK2, lw=0, marker="o", ms=7, label="reached EDIFF 1E-6"),
                Line2D([], [], color=INK2, lw=0, marker="X", ms=7, label="stopped unconverged")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.935), ncol=4, frameon=False,
               fontsize=8.5, handlelength=2.6, columnspacing=1.6)
    title = "Which SCF route converges the [100] GB cell, and how fast?"
    if synthetic:
        title = "SYNTHETIC TEST DATA (smoke test, not results) - " + title
    fig.text(0.06, 0.985, title, fontsize=12.5, weight="bold", va="top", color=INK)
    fig.text(0.06, 0.955, "e500_k1x3x3 (5 k-points), ALGO Normal, ISMEAR 1 / SIGMA 0.05, ISPIN 2 from "
             "MAGMOM 1 μB per Co, NELM 200. Steps are spread evenly over each run's wall time.",
             fontsize=9, va="top", color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.08, hspace=0.12, wspace=0.14)
    fig.savefig(out, dpi=150, facecolor=SURFACE,
                metadata={"Title": title, "Software": "src/dft_plots/coge_interface/scf_trials_plot.py"})
    plt.close(fig)
    return out


def load(records):
    paths = sorted((Path(records) / "trials").glob("*.json"))
    if not paths:
        sys.exit(f"STOP: no trial records in {records}/trials")
    trials = [json.loads(p.read_text()) for p in paths]
    unknown = {t["sym"] for t in trials} - set(STRUCTURES)
    if unknown:
        sys.exit(f"STOP: records for unexpected structures {sorted(unknown)}")
    return trials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, help="trial records folder (holds trials/*.json)")
    parser.add_argument("--out", type=Path, help="PNG path (default RECORDS/trace.png)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing PNG")
    parser.add_argument("--smoke", type=Path, help="write a SYNTHETIC figure into this folder instead")
    args = parser.parse_args(argv)
    if args.smoke:
        args.smoke.mkdir(parents=True, exist_ok=True)
        print(draw(synthetic_trials(), args.smoke / "SYNTHETIC_trace.png", synthetic=True))
        return
    if not args.records:
        parser.error("--records or --smoke is required")
    records = args.records if args.records.is_absolute() else PROJECT_ROOT / args.records
    out = args.out or records / "trace.png"
    if out.exists() and not args.overwrite:
        sys.exit(f"STOP: {out} exists (use --overwrite)")
    print(draw(load(records), out))


if __name__ == "__main__":
    main()
