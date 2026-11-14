"""Validation for author-provided private artifact bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PLACEHOLDER = "REPLACE_WITH_AUTHOR_PROVIDED_SHA256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest().upper()


def record_count(path: Path, kind: str) -> int | None:
    if kind == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if kind == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return len(value) if isinstance(value, (list, dict)) else None
    return None


def validate_contract(contract_path: Path, root: Path | None = None, require_hashes: bool = True) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root = root or contract_path.parents[1]
    results, errors = [], []
    for spec in contract["artifacts"]:
        path = root / spec["path"]
        item = {"name": spec["name"], "path": str(path), "exists": path.exists()}
        if not path.exists():
            errors.append(f"missing {spec['name']}: {path}")
            results.append(item)
            continue
        actual_hash = sha256_directory(path) if path.is_dir() else sha256_file(path)
        expected_hash = spec.get("sha256", PLACEHOLDER).upper()
        item["sha256"] = actual_hash
        if expected_hash == PLACEHOLDER:
            if require_hashes:
                errors.append(f"placeholder SHA-256 for {spec['name']}")
        elif actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch for {spec['name']}")
        expected_records = spec.get("expected_records")
        if expected_records is not None and path.is_file():
            actual_records = record_count(path, spec["kind"])
            item["records"] = actual_records
            if actual_records != int(expected_records):
                errors.append(
                    f"record count mismatch for {spec['name']}: {actual_records} != {expected_records}"
                )
        expected_files = spec.get("expected_files")
        if expected_files is not None and path.is_dir():
            actual_files = sum(1 for item_path in path.rglob("*") if item_path.is_file())
            item["files"] = actual_files
            if actual_files != int(expected_files):
                errors.append(f"file count mismatch for {spec['name']}: {actual_files} != {expected_files}")
        path_value_spec = spec.get("path_values_are_files")
        if path_value_spec and path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                errors.append(f"{spec['name']} must be a JSON object of labeled file references")
            else:
                referenced = {}
                for label, value in values.items():
                    if path_value_spec is True:
                        relative = value
                        declared_hash = None
                    elif isinstance(path_value_spec, str) and isinstance(value, dict):
                        relative = value.get(path_value_spec)
                        declared_hash = value.get("sha256")
                    else:
                        relative = None
                        declared_hash = None
                    if not isinstance(relative, str) or not relative:
                        errors.append(f"invalid referenced file entry {label} in {spec['name']}")
                        continue
                    referenced_path = root / relative
                    if not referenced_path.is_file():
                        errors.append(f"missing referenced file {label}: {referenced_path}")
                    else:
                        actual_referenced_hash = sha256_file(referenced_path)
                        referenced[label] = actual_referenced_hash
                        if declared_hash and declared_hash.upper() != actual_referenced_hash:
                            errors.append(f"SHA-256 mismatch for referenced file {label}")
                item["referenced_files"] = referenced
        results.append(item)
    return {"ok": not errors, "errors": errors, "artifacts": results}

