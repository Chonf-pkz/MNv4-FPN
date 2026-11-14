#!/usr/bin/env python3
"""Mean and SD over the 5 held-out folds of one or more target runs (compare recipes here, not on val-85)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Segmentation first; runs trained before 29/09 have no IoU/BF1/MAPE/R² in their summaries and print "-".
METRICS = ("symptom_mean_dice", "symptom_mean_iou", "symptom_mean_bf1", "symptom_micro_dice",
           "host_mean_dice_pseudo", "burden_mae_pp_vs_pseudo", "burden_mape_pct_vs_pseudo", "burden_r2_vs_pseudo",
           "disease_macro_f1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path, help=".../target/<run_name>_seed<seed>")
    args = parser.parse_args()
    print(f"{'run':40s} " + " ".join(f"{m[:22]:>24s}" for m in METRICS))
    for run in args.run_dirs:
        summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run.glob("fold_[0-9]/heldout_summary.json"))]
        cells = []
        for metric in METRICS:
            values = [s[metric] for s in summaries if s.get(metric) is not None]
            cells.append(f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else "-")
        print(f"{run.name + f' ({len(summaries)} folds)':40s} " + " ".join(f"{c:>24s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
