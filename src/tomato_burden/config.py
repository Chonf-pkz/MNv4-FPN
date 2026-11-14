"""Configuration loading, validation, and command-line overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "protocol_final.json"


def load(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_paper_recipe(config)
    return config


def set_dotted(config: dict, dotted: str, value: Any) -> None:
    cursor = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def with_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    updated = copy.deepcopy(config)
    for dotted, value in overrides.items():
        set_dotted(updated, dotted, value)
    validate_paper_recipe(updated)
    return updated


def validate_paper_recipe(config: dict) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    def get_dotted(dotted: str) -> Any:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(dotted)
            value = value[part]
        return value

    for dotted, expected in protocol["frozen_config"].items():
        try:
            actual = get_dotted(dotted)
        except KeyError:
            errors.append(f"missing {dotted}")
            continue
        if actual != expected:
            errors.append(f"{dotted}={actual!r}, expected {expected!r}")
    if config.get("seed") not in protocol["seeds"]:
        errors.append(f"seed={config.get('seed')!r}, expected one of {protocol['seeds']}")
    if errors:
        raise ValueError("Invalid paper configuration: " + "; ".join(errors))


def write(path: Path, config: dict) -> None:
    validate_paper_recipe(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

