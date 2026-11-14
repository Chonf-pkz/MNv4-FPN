from __future__ import annotations

import pytest

from tomato_burden.statistics import cluster_bootstrap_delta


def rows(values: list[float]) -> list[dict]:
    return [
        {"sample_id": f"s{index}", "duplicate_group": f"g{index // 2}", "metric": value, "keep": True}
        for index, value in enumerate(values)
    ]


def test_cluster_bootstrap_is_paired_and_deterministic() -> None:
    candidate = rows([2.0, 3.0, 6.0, 8.0])
    comparator = rows([1.0, 1.0, 4.0, 5.0])
    first = cluster_bootstrap_delta(candidate, comparator, "metric", lambda row: row["keep"], 2000, 17)
    second = cluster_bootstrap_delta(candidate, comparator, "metric", lambda row: row["keep"], 2000, 17)
    assert first == second
    assert first["records"] == 4
    assert first["clusters"] == 2
    assert first["delta"] == pytest.approx(2.0)


def test_cluster_bootstrap_rejects_unpaired_rows() -> None:
    with pytest.raises(RuntimeError, match="Unpaired"):
        cluster_bootstrap_delta(rows([1.0]), rows([1.0, 2.0]), "metric", lambda _: True, 10, 1)
