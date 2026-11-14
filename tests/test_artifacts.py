from __future__ import annotations

import json
from pathlib import Path

from tomato_burden.artifacts import sha256_directory, sha256_file, validate_contract


def test_private_artifact_contract_checks_count_and_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "records.csv"
    manifest.write_text("sample_id\na\nb\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "records",
                        "path": "records.csv",
                        "kind": "csv",
                        "expected_records": 2,
                        "sha256": sha256_file(manifest),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = validate_contract(contract, root=tmp_path)
    assert result["ok"]
    assert result["artifacts"][0]["records"] == 2


def test_private_artifact_contract_reports_placeholder(tmp_path: Path) -> None:
    manifest = tmp_path / "records.csv"
    manifest.write_text("sample_id\na\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "records",
                        "path": "records.csv",
                        "kind": "csv",
                        "expected_records": 1,
                        "sha256": "REPLACE_WITH_AUTHOR_PROVIDED_SHA256",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = validate_contract(contract, root=tmp_path)
    assert not result["ok"]
    assert "placeholder SHA-256" in result["errors"][0]


def test_private_artifact_contract_validates_referenced_files_and_tree(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    model = checkpoints / "model.pt"
    model.write_bytes(b"safe-test-checkpoint")
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps({"model": {"checkpoint": "checkpoints/model.pt", "sha256": sha256_file(model)}}),
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"artifacts": [
            {"name": "tree", "path": "checkpoints", "kind": "directory", "expected_files": 1,
             "sha256": sha256_directory(checkpoints)},
            {"name": "sources", "path": "sources.json", "kind": "json", "expected_records": 1,
             "path_values_are_files": "checkpoint", "sha256": sha256_file(sources)},
        ]}),
        encoding="utf-8",
    )
    result = validate_contract(contract, root=tmp_path)
    assert result["ok"]
    assert result["artifacts"][0]["files"] == 1
    assert result["artifacts"][1]["referenced_files"]["model"] == sha256_file(model)
