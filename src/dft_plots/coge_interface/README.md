# src/dft_plots/coge_interface

Analysis of the CoGe grain-boundary (GB) cells. Uses the `src/dft_plots/.venv` Python (numpy).

| File | What it does |
| --- | --- |
| `overlap_survey.py` | Closest pair of atoms in every GB POSCAR of the catalogue (13,869 files, all planes and thicknesses), and the number of pairs below each threshold. Writes `results/CoGe_interface/overlap_survey.tsv` and prints, per plane and threshold, how many GB families clear it in the catalogue and in the Stage 2 list |
| `test_overlap_survey.py` | Unit tests on synthetic cells |

## Overlap limit

Set 2026-10-06 from this survey, at the user's request to leave out cells with extreme overlap:
**GB cells whose closest pair is below 1.8 Å are not run.**

- 1.8 Å is 75% of the shortest bulk bond, Co–Ge 2.40 Å. (In bulk B20, Co–Co is 2.84 Å and Ge–Ge
  2.87 Å; in elemental hcp Co and diamond Ge the nearest neighbours are 2.50 and 2.45 Å.)
- The closest pair of a GB is the same at every thickness (4x–12x), so the limit applies per GB family.
- The catalogue has no natural gap in the closest-pair distribution, so the line is a judgment call.
  Counts at 8x (families in the catalogue | in the Stage 2 list):

  | Plane | All | ≥ 1.5 Å | ≥ 1.8 Å | ≥ 2.0 Å |
  | --- | --- | --- | --- | --- |
  | 100 | 623 \| 66 | 307 \| 33 | 114 \| 13 | 31 \| 5 |
  | 110 | 543 \| 63 | 106 \| 17 | 9 \| 1 | 1 \| 0 |
  | 111 | 375 \| 33 | 81 \| 8 | 19 \| 1 | 14 \| 1 |

## Run

```bash
src/dft_plots/.venv/bin/python -B src/dft_plots/coge_interface/overlap_survey.py --dry-run   # login node: count only
srun -p general --exclude=qcg-docker -n 1 -c 4 --mem=4G -t 00:45:00 bash -lc \
    'cd <project root> && src/dft_plots/.venv/bin/python -B src/dft_plots/coge_interface/overlap_survey.py'
```

Options: `--planes`, `--thicknesses`, `--thresholds` (Å; each must be below half the shortest cell
edge, 2.31 Å, where the per-axis minimum image is exact), `--workers` (file-reading threads; each read
costs ~0.09 s of file-system latency), `--out`, `--overwrite`. A full run takes about 7 minutes.

Tests: `src/dft_plots/.venv/bin/python -B -m unittest discover -s src/dft_plots/coge_interface -p 'test_*.py'`.

## TSV columns

`plane`, `sym`, `thickness`, `status` (`ok`, `empty`, `missing`, …), `atoms`, `dmin_A` (closest pair),
`pair` (its species), `dmin_over_bulk_pair` (dmin over the shortest bulk distance of that species pair),
`pairs_lt_<t>A` (pairs closer than each threshold), `in_stage2_list` (in
`results/dscribe/total_sample_list.csv`).
