from __future__ import annotations

from types import SimpleNamespace

import pytest

from tomato_burden import training


def _source_records(train: int = 6828, validation: int = 752) -> list[SimpleNamespace]:
    records = [SimpleNamespace(sample_id=f"source-train-{index}", split="train") for index in range(train)]
    records.extend(
        SimpleNamespace(sample_id=f"source-validation-{index}", split="validation")
        for index in range(validation)
    )
    return records


def test_source_cohort_requires_frozen_train_and_validation_counts() -> None:
    summary = training.validate_source_cohort(_source_records())
    assert summary["records"] == 7580
    assert summary["train_records"] == 6828
    assert summary["monitor_records"] == 752

    with pytest.raises(RuntimeError, match="7,580"):
        training.validate_source_cohort(_source_records(validation=751))

    with pytest.raises(RuntimeError, match="6,828 train and 752 validation"):
        training.validate_source_cohort(_source_records(train=6827, validation=753))


def test_source_cohort_rejects_duplicate_ids_and_missing_monitor_split() -> None:
    duplicate = _source_records()
    duplicate[-1].sample_id = duplicate[0].sample_id
    with pytest.raises(RuntimeError, match="unique"):
        training.validate_source_cohort(duplicate)

    no_monitor = _source_records()
    for record in no_monitor:
        record.split = "train"
    with pytest.raises(RuntimeError, match="6,828 train and 752 validation"):
        training.validate_source_cohort(no_monitor)
