#!/usr/bin/env python3
"""Run both compact examples and create the paper-ready outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cosmologyfredipy.analysis import (
    analyse_planck,
    analyse_planck_sensitivity,
    analyse_synthetic,
    make_summary,
    write_summary,
)
from cosmologyfredipy.plotting import (
    plot_operator,
    plot_planck,
    plot_synthetic,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=ROOT / "data/synthetic_class.npz",
    )
    parser.add_argument(
        "--planck",
        type=Path,
        default=ROOT / "data/planck_pr3_tt.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="directory for summary.json (default: results)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "paper/figures",
        help="directory for the three PDF figures (default: paper/figures)",
    )
    args = parser.parse_args()

    synthetic = analyse_synthetic(args.synthetic)
    planck = analyse_planck(args.planck)
    planck_sensitivity = analyse_planck_sensitivity(args.planck, baseline=planck)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_operator(synthetic, args.figure_dir / "fredholm_operator.pdf")
    plot_synthetic(synthetic, args.figure_dir / "synthetic_reconstruction.pdf")
    plot_planck(planck, args.figure_dir / "planck_reconstruction.pdf")
    text = write_summary(
        args.output_dir / "summary.json",
        make_summary(synthetic, planck, planck_sensitivity),
    )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
