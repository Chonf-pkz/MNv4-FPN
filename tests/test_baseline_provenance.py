from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tomato_burden import baselines


ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[SimpleNamespace]:
    records = []
    for index in range(595):
        active = index < 199
        records.append(SimpleNamespace(
            sample_id=f"train-{index}", split="train", host_weight=1.0 if active else 0.0,
            host="mask.png" if active else None,
            extra={"host_status": "high_confidence" if active else ""},
        ))
    records.extend(
        SimpleNamespace(sample_id=f"val-{index}", split="validation", host_weight=0.0, host=None, extra={})
        for index in range(85)
    )
    return records


def test_baseline_recipe_embeds_verified_data_provenance() -> None:
    session = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))
    builder = getattr(baselines, "attested_recipe", None)
    assert builder is not None

    config = builder(session, "unet_r34", 20260924, _records(), "A" * 64)

    assert config["run_metadata"] == {
        "variant": "baseline",
        "cohort": {
            "train_records": 595,
            "validation_records": 85,
            "host_supervised_records": 199,
            "sam2_host_masks": 199,
            "manual_host_masks": 0,
        },
        "host_manifest_sha256": "A" * 64,
    }
