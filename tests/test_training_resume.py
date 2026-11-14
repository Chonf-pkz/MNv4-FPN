from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tomato_burden import training


def test_completed_checkpoint_must_match_requested_run_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_ema.pt"
    old_config = {
        "run_metadata": {
            "variant": "standard",
            "cohort": {"train_records": 595, "host_supervised_records": 199, "manual_host_masks": 0},
            "host_manifest_sha256": "A" * 64,
        }
    }
    torch.save({"config": old_config, "seed": 20260926, "fold": "all", "ema_state": {"x": torch.ones(1)}}, checkpoint)
    requested = {
        "run_metadata": {
            "variant": "manual-host",
            "cohort": {"train_records": 595, "host_supervised_records": 595, "manual_host_masks": 396},
            "host_manifest_sha256": "B" * 64,
        }
    }

    validator = getattr(training, "validate_completed_target", None)
    assert validator is not None
    with pytest.raises(RuntimeError, match="does not match the requested config"):
        validator(checkpoint, requested, 20260926, "all")


def test_completed_checkpoint_with_exact_identity_can_be_reused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_ema.pt"
    config = {"run_metadata": {"variant": "standard"}}
    torch.save({"config": config, "seed": 20260924, "fold": "all", "ema_state": {"x": torch.ones(1)}}, checkpoint)

    validator = getattr(training, "validate_completed_target", None)
    assert validator is not None
    validator(checkpoint, config, 20260924, "all")


def test_completed_source_checkpoint_must_match_requested_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source_ema_final.pt"
    torch.save({
        "config": {"model": {"backbone": "old"}}, "seed": 20260924,
        "ema_state": {"x": torch.ones(1)},
    }, checkpoint)
    validator = getattr(training, "validate_completed_source", None)
    assert validator is not None
    with pytest.raises(RuntimeError, match="does not match the requested config"):
        validator(checkpoint, {"model": {"backbone": "new"}}, 20260924)
