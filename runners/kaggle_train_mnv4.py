#!/usr/bin/env python3
"""Parameterized Kaggle runner for the paper's MNv4-FPN and MNv4-FPN (MH) runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tomato_burden import config as config_io  # noqa: E402
from tomato_burden import training  # noqa: E402


def validate_training_cohort(records: list, variant: str) -> dict:
    return training.validate_target_cohort(records, variant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--seed", type=int, choices=(20260924, 20260925, 20260926), required=True)
    parser.add_argument("--variant", choices=("standard", "manual-host"), default="standard")
    parser.add_argument("--manual-host-manifest", type=Path)
    parser.add_argument("--stage", choices=("source", "target", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.variant == "manual-host" and args.manual_host_manifest is None:
        parser.error("--manual-host-manifest is required for --variant manual-host")
    config = config_io.load(args.config)
    overrides = {
        "seed": args.seed,
        "run_name": "mnv4_fpn_manual_host" if args.variant == "manual-host" else "mnv4_fpn",
    }
    if args.manual_host_manifest:
        overrides["paths.pseudo_host_manifest"] = str(args.manual_host_manifest)
    config = config_io.with_overrides(config, overrides)
    cohort = validate_training_cohort(training.tomato_records(config, use_cache=False), args.variant)
    host_manifest = Path(config["paths"]["pseudo_host_manifest"])
    config = config_io.with_overrides(config, {
        "run_metadata.variant": args.variant,
        "run_metadata.cohort": cohort,
        "run_metadata.host_manifest_sha256": training.sha256_file(host_manifest),
    })
    generated = Path(config["paths"]["output_root"]) / "run_configs" / f"{config['run_name']}_{args.seed}.json"
    config_io.write(generated, config)
    print(json.dumps({"preflight": "ok", "variant": args.variant, **cohort}, indent=2), flush=True)

    if args.stage in ("source", "all"):
        training.command_source(config, generated.resolve(), args.resume)
    if args.stage in ("target", "all"):
        training.command_target(config, generated.resolve(), "all", args.seed, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

