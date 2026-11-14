#!/usr/bin/env python3
"""Validate an author-provided private-artifact bundle before a paper rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tomato_burden.artifacts import validate_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", nargs="?", type=Path, default=Path("configs/private_artifacts.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-placeholder-hashes", action="store_true")
    args = parser.parse_args()
    report = validate_contract(args.contract, args.root, not args.allow_placeholder_hashes)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

