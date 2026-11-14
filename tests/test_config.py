from __future__ import annotations

import json
from pathlib import Path

import pytest

from tomato_burden import config


ROOT = Path(__file__).resolve().parents[1]


def paper_config() -> dict:
    return json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))


def test_checked_config_is_valid() -> None:
    config.validate_paper_recipe(paper_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("loss.disease_weight", 1.0),
        ("target_stage.recalibrate_bn", True),
        ("target_stage.epochs", 119),
        ("target_stage.batch_size", 4),
        ("input_size", 640),
        ("amp", False),
        ("cache.max_side", 512),
        ("focal_classes", ["tomato_early_blight"]),
    ],
)
def test_paper_critical_overrides_are_rejected(path: str, value) -> None:
    recipe = paper_config()
    config.set_dotted(recipe, path, value)
    with pytest.raises(ValueError, match="Invalid paper configuration"):
        config.validate_paper_recipe(recipe)


def test_allowed_path_override_does_not_mutate_input() -> None:
    recipe = paper_config()
    changed = config.with_overrides(recipe, {"paths.dataset_root": "/tmp/dataset"})
    assert changed["paths"]["dataset_root"] == "/tmp/dataset"
    assert recipe["paths"]["dataset_root"] != changed["paths"]["dataset_root"]
