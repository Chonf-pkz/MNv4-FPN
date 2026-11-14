#!/usr/bin/env python3
"""Parameterized Kaggle runner for one seed and one or all paper baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tomato_burden import baselines, config as config_io  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--seed", type=int, choices=baselines.SEEDS, required=True)
    parser.add_argument("--arch", choices=(*baselines.ARCHS, "all"), default="all")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = config_io.load(args.config)
    if args.preflight:
        baselines.command_preflight(config)
    selected = baselines.ARCHS if args.arch == "all" else (args.arch,)
    for arch in selected:
        baselines.command_run(config, arch, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

