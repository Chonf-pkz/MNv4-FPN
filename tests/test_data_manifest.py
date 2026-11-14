from __future__ import annotations

from pathlib import Path

from tomato_burden import data


def test_host_status_is_preserved_for_training_provenance(tmp_path: Path) -> None:
    tomato = tmp_path / "tomato.csv"
    host = tmp_path / "host.csv"
    data.write_csv(tomato, [{
        "sample_id": "a", "hard_validation": "pass", "model_use": "train",
        "experimental_split": "train", "class_name": "disease", "image_relpath": "a.jpg",
        "symptom_mask_relpath": "a.png", "duplicate_group": "a",
    }])
    data.write_csv(host, [{
        "sample_id": "a", "host_weight": "1", "burden_weight": "1",
        "pseudo_host_mask_relpath": "host.png", "burden_policy": "active",
        "host_status": "manual_reviewed",
    }])

    records = data.load_tomato_records(tomato, host, tmp_path, tmp_path)

    assert records[0].extra["host_status"] == "manual_reviewed"
