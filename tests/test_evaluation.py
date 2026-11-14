from __future__ import annotations

from types import SimpleNamespace

from tomato_burden.evaluation import stratified, summarize_rows


def test_stratified_selection_is_balanced_and_reproducible() -> None:
    records = [
        SimpleNamespace(sample_id=f"{name}-{index}", class_name=name)
        for name in ("a", "b", "c")
        for index in range(4)
    ]
    first = stratified(records, 7, 123)
    second = stratified(records, 7, 123)
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    counts = {name: sum(row.class_name == name for row in first) for name in ("a", "b", "c")}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_summary_keeps_micro_and_macro_dice_distinct() -> None:
    rows = [
        {
            "sample_id": "a",
            "focal": True,
            "symptom_dice": 1.0,
            "symptom_iou": 1.0,
            "symptom_bf1": 1.0,
            "symptom_tp": 10,
            "symptom_pred_pixels": 10,
            "symptom_true_pixels": 10,
        },
        {
            "sample_id": "b",
            "focal": False,
            "symptom_dice": 0.0,
            "symptom_iou": 0.0,
            "symptom_bf1": 0.0,
            "symptom_tp": 0,
            "symptom_pred_pixels": 1,
            "symptom_true_pixels": 1,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["symptom_mean_dice"] == 0.5
    assert summary["symptom_micro_dice"] == 20 / 22
    assert summary["symptom_mean_dice_focal"] == 1.0
