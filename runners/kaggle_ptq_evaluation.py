#!/usr/bin/env python3
"""Run the frozen static-PTQ sequence on Kaggle or another Linux host."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: str, config: Path, bundle: Path, sources: Path | None = None, extra: list[str] | None = None) -> None:
    argv = [sys.executable, "-m", "tomato_burden.paper_pipeline", command, "--config", str(config), "--bundle", str(bundle)]
    if sources is not None:
        argv.extend(("--sources", str(sources)))
    argv.extend(extra or [])
    env = os.environ.copy()
    source_root = str(ROOT / "src")
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(argv, check=True, env=env, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true", help="open the declared test set after lock verification")
    args = parser.parse_args()
    for command in ("export", "p3", "select", "int8", "validate", "bench", "lock"):
        run(command, args.config, args.bundle, args.sources)
    if args.include_test:
        run("test", args.config, args.bundle, args.sources, extra=["--confirm-test-use"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

