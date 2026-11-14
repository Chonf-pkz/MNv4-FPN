from __future__ import annotations

import json
from pathlib import Path

from tomato_burden.paper_pipeline import registry


ROOT = Path(__file__).resolve().parents[1]


def test_registry_matches_the_frozen_model_counts() -> None:
    models = registry()
    assert sum(row["fmt"] == "fp32" for row in models.values()) == 24
    assert sum(row["fmt"] == "int8" for row in models.values()) == 18
    assert all("qat" not in label.lower() for label in models)


def test_model_source_example_covers_every_fp32_export() -> None:
    sources = json.loads((ROOT / "configs" / "model_sources.example.json").read_text(encoding="utf-8"))
    expected = {label for label, row in registry().items() if row["fmt"] == "fp32"}
    assert set(sources) == expected
    assert all(isinstance(entry, dict) for entry in sources.values())
    assert all({"checkpoint", "sha256", "seed", "network", "experiment", "source_run"} <= entry.keys()
               for entry in sources.values())
    for label, entry in sources.items():
        if label.startswith("p1_host_"):
            assert entry["experiment"] == "manual-host"
            assert entry["data_attestation"]["host_supervised_records"] == 595
            assert entry["data_attestation"]["manual_host_masks"] == 396


def test_checkpoint_loads_use_restricted_weights_only_mode() -> None:
    for path in (ROOT / "src" / "tomato_burden").glob("*.py"):
        assert "weights_only=False" not in path.read_text(encoding="utf-8")
