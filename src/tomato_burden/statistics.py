"""Registered cluster-bootstrap utilities used by the paper analysis."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap_delta(
    candidate: list[dict],
    comparator: list[dict],
    endpoint: str,
    population,
    resamples: int,
    seed: int,
) -> dict:
    """Paired bootstrap over near-duplicate clusters."""
    candidate_map = {row["sample_id"]: row for row in candidate if population(row)}
    comparator_map = {row["sample_id"]: row for row in comparator if population(row)}
    if set(candidate_map) != set(comparator_map):
        raise RuntimeError(f"Unpaired bootstrap inputs for {endpoint}")
    ids = sorted(candidate_map)
    if not ids:
        raise ValueError(f"No records selected for {endpoint}")
    deltas = np.asarray(
        [float(candidate_map[item][endpoint]) - float(comparator_map[item][endpoint]) for item in ids]
    )
    groups = [candidate_map[item]["duplicate_group"] for item in ids]
    names = sorted(set(groups))
    sums = np.asarray([deltas[[group == name for group in groups]].sum() for name in names])
    counts = np.asarray([sum(group == name for group in groups) for name in names])
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        sampled = rng.integers(0, len(names), size=(stop - start, len(names)))
        estimates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "records": len(ids),
        "clusters": len(names),
        "candidate_mean": float(np.mean([float(candidate_map[item][endpoint]) for item in ids])),
        "comparator_mean": float(np.mean([float(comparator_map[item][endpoint]) for item in ids])),
        "delta": float(deltas.mean()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "probability_below_zero": float(np.mean(estimates < 0.0)),
    }

