from __future__ import annotations

from types import SimpleNamespace

import pytest

from runners import kaggle_train_mnv4


def _records(host_count: int, manual_count: int = 0) -> list[SimpleNamespace]:
    rows = []
    for index in range(595):
        active = index < host_count
        rows.append(SimpleNamespace(
            sample_id=f"train-{index}", split="train", host_weight=1.0 if active else 0.0,
            host="mask.png" if active else None,
            extra={"host_status": "manual_reviewed" if index < manual_count else "high_confidence"},
        ))
    rows.extend(
        SimpleNamespace(sample_id=f"val-{index}", split="validation", host_weight=0.0, host=None, extra={})
        for index in range(85)
    )
    return rows


def test_standard_runner_accepts_199_host_supervised_train_records() -> None:
    validator = getattr(kaggle_train_mnv4, "validate_training_cohort", None)
    assert validator is not None
    assert validator(_records(199), "standard")["host_supervised_records"] == 199


def test_manual_host_runner_requires_all_595_train_records_to_have_hosts() -> None:
    validator = getattr(kaggle_train_mnv4, "validate_training_cohort", None)
    assert validator is not None
    summary = validator(_records(595, manual_count=396), "manual-host")
    assert summary["host_supervised_records"] == 595
    assert summary["manual_host_masks"] == 396
    with pytest.raises(RuntimeError, match="host-supervised"):
        validator(_records(199), "manual-host")
    with pytest.raises(RuntimeError, match="manual host masks"):
        validator(_records(595, manual_count=395), "manual-host")


def test_manual_host_runner_rejects_unknown_nonmanual_status() -> None:
    records = _records(595, manual_count=396)
    records[500].extra["host_status"] = "unknown"
    with pytest.raises(RuntimeError, match="SAM2"):
        kaggle_train_mnv4.validate_training_cohort(records, "manual-host")
