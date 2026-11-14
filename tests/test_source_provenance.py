from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tomato_burden import paper_pipeline
from tomato_burden.artifacts import sha256_file


def _complete_registry(tmp_path: Path) -> tuple[Path, dict]:
    sources: dict[str, dict] = {}
    for label, model in paper_pipeline.registry().items():
        if model["fmt"] != "fp32":
            continue
        checkpoint = tmp_path / "checkpoints" / f"{label}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(label.encode("utf-8"))
        if label.startswith("p1_host_"):
            experiment = "manual-host"
            host_supervised = 595
            manual_masks = 396
        elif label.startswith("p1_nohost_"):
            experiment = "standard"
            host_supervised = 199
            manual_masks = 0
        elif label.startswith("p2_"):
            experiment = "baseline"
            host_supervised = 199
            manual_masks = 0
        else:
            experiment = "supplementary"
            host_supervised = None
            manual_masks = None
        entry = {
            "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(checkpoint),
            "seed": model["seed"],
            "network": model["net"],
            "experiment": experiment,
            "source_run": {"platform": "local", "id": f"test/{label}", "version": 1},
        }
        if host_supervised is not None:
            entry["data_attestation"] = {
                "train_records": 595,
                "host_supervised_records": host_supervised,
                "manual_host_masks": manual_masks,
                "host_manifest_sha256": "A" * 64,
            }
        sources[label] = entry
    path = tmp_path / "model_sources.json"
    path.write_text(json.dumps(sources), encoding="utf-8")
    return path, sources


def test_structured_source_registry_accepts_exact_checkpoint_identity(tmp_path: Path) -> None:
    loader = getattr(paper_pipeline, "load_model_sources", None)
    assert loader is not None, "paper pipeline must validate structured model provenance"
    path, sources = _complete_registry(tmp_path)

    loaded = loader(path, root=tmp_path)

    assert set(loaded) == set(sources)
    assert loaded["p1_host_s26"]["checkpoint"].is_file()
    assert loaded["p1_host_s26"]["experiment"] == "manual-host"


def test_source_registry_rejects_manual_host_run_with_standard_host_coverage(tmp_path: Path) -> None:
    loader = getattr(paper_pipeline, "load_model_sources", None)
    assert loader is not None, "paper pipeline must validate structured model provenance"
    path, sources = _complete_registry(tmp_path)
    sources["p1_host_s26"]["data_attestation"]["host_supervised_records"] = 199
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(RuntimeError, match="host_supervised_records"):
        loader(path, root=tmp_path)


def test_source_registry_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    loader = getattr(paper_pipeline, "load_model_sources", None)
    assert loader is not None, "paper pipeline must validate structured model provenance"
    path, sources = _complete_registry(tmp_path)
    sources["p1_host_s26"]["sha256"] = "0" * 64
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        loader(path, root=tmp_path)


def test_lock_hashes_structured_source_checkpoints(tmp_path: Path) -> None:
    path, sources = _complete_registry(tmp_path)

    hashes = paper_pipeline.source_checkpoint_hashes(path, root=tmp_path)

    assert hashes["p1_host_s26"] == sources["p1_host_s26"]["sha256"]


def test_checkpoint_metadata_must_match_registered_seed_and_network() -> None:
    validator = getattr(paper_pipeline, "validate_checkpoint_identity", None)
    assert validator is not None, "export must compare checkpoint metadata with the source registry"
    source = {
        "seed": 20260926,
        "network": "ours",
        "experiment": "manual-host",
        "data_attestation": {
            "train_records": 595,
            "host_supervised_records": 595,
            "manual_host_masks": 396,
            "host_manifest_sha256": "A" * 64,
        },
    }
    good = {
        "seed": 20260926,
        "arch": "ours",
        "host_manifest": "/kaggle/working/host_train_v1/pseudo_host_manifest.csv",
        "run_metadata": None,
    }

    validator("p1_host_s26", source, good)
    with pytest.raises(RuntimeError, match="seed"):
        validator("p1_host_s26", source, {**good, "seed": 20260924})
    with pytest.raises(RuntimeError, match="network"):
        validator("p1_host_s26", source, {**good, "arch": "segformer_b0"})


def test_manual_host_checkpoint_rejects_superseded_standard_manifest() -> None:
    source = {
        "seed": 20260926,
        "network": "ours",
        "experiment": "manual-host",
        "data_attestation": {
            "train_records": 595,
            "host_supervised_records": 595,
            "manual_host_masks": 396,
            "host_manifest_sha256": "A" * 64,
        },
    }
    broken_notebook_checkpoint = {
        "seed": 20260926,
        "arch": "ours",
        "host_manifest": "/kaggle/input/phase3/pseudo_host_manifest.csv",
        "run_metadata": None,
    }

    with pytest.raises(RuntimeError, match="manual-host provenance"):
        paper_pipeline.validate_checkpoint_identity("p1_host_s26", source, broken_notebook_checkpoint)


def test_embedded_run_metadata_must_match_registered_manifest_hash() -> None:
    source = {
        "seed": 20260926,
        "network": "ours",
        "experiment": "manual-host",
        "data_attestation": {
            "train_records": 595,
            "host_supervised_records": 595,
            "manual_host_masks": 396,
            "host_manifest_sha256": "A" * 64,
        },
    }
    checkpoint = {
        "seed": 20260926,
        "arch": "ours",
        "host_manifest": "artifacts/private/manifests/manual_host_manifest.csv",
        "run_metadata": {
            "variant": "manual-host",
            "cohort": {"train_records": 595, "host_supervised_records": 595, "manual_host_masks": 396},
            "host_manifest_sha256": "B" * 64,
        },
    }

    with pytest.raises(RuntimeError, match="host_manifest_sha256"):
        paper_pipeline.validate_checkpoint_identity("p1_host_s26", source, checkpoint)


def test_source_registry_rejects_placeholder_run_identity(tmp_path: Path) -> None:
    path, sources = _complete_registry(tmp_path)
    sources["p1_host_s26"]["source_run"]["id"] = "REPLACE_WITH_FINAL_RUN_ID"
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(RuntimeError, match="placeholder"):
        paper_pipeline.load_model_sources(path, root=tmp_path)


def test_existing_export_must_match_source_provenance(tmp_path: Path) -> None:
    validator = getattr(paper_pipeline, "validate_export_cache", None)
    assert validator is not None, "resumed exports must validate their source checkpoint"
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    meta = tmp_path / "model.json"
    source = {"sha256": "A" * 64, "seed": 20260926, "network": "ours"}
    meta.write_text(json.dumps({
        "label": "p1_host_s26",
        "onnx_sha256": sha256_file(onnx),
        "seed": 20260926,
        "arch": "ours",
        "source_provenance": {"sha256": "A" * 64},
    }), encoding="utf-8")

    validator("p1_host_s26", onnx, meta, source)
    document = json.loads(meta.read_text(encoding="utf-8"))
    document["source_provenance"]["sha256"] = "B" * 64
    meta.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checkpoint"):
        validator("p1_host_s26", onnx, meta, source)


def test_pretest_lock_cannot_omit_model_sources() -> None:
    with pytest.raises(RuntimeError, match="--sources is required"):
        paper_pipeline.lock_state(SimpleNamespace(sources=None), session=None)
