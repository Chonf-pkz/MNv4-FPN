#!/usr/bin/env python3
"""Local, blinded host-organ mask correction tool for PlantSeg evaluation data.

The tool deliberately keeps model predictions out of the annotation interface.
It shows only the source RGB image, the released PlantSeg symptom reference, and
the locked SAM2 pseudo-host proposal. Corrected masks are written to new output
directories; pseudo-host masks are never modified.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import shutil
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "host_train_annotation_config.json"
UI_PATH = ROOT / "host_gold_annotation_ui.html"

QUEUE_FIELDS = [
    "annotation_item_id",
    "queue_position",
    "sample_id",
    "repeat_index",
    "is_repeat",
    "class_name",
    "experimental_split",
    "image_relpath",
    "symptom_mask_relpath",
    "pseudo_host_mask_relpath",
    "output_mask_relpath",
]

ANNOTATION_FIELDS = [
    "annotation_item_id",
    "review_status",
    "decision",
    "quality_flags",
    "notes",
    "annotation_seconds",
    "annotator_id",
    "completed_at_utc",
    "mask_sha256",
    "mask_pixels",
    "pseudo_difference_pixels",
    "symptom_outside_host_pixels",
]

APPROVED_STATUS = "approved"
PENDING_STATUS = "pending"
UNCERTAIN_STATUS = "uncertain"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def project_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes project root: {relative}") from exc
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def png_data_url(array: np.ndarray) -> str:
    image = Image.fromarray(array.astype(np.uint8), mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_data_url(path: Path) -> str:
    """Return an EXIF-normalized RGB preview aligned to the locked masks."""
    with Image.open(path) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        stream = io.BytesIO()
        normalized.save(stream, format="JPEG", quality=94, subsampling=0)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L")) > 0


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).size


def decode_mask_data_url(value: str) -> np.ndarray:
    if not value.startswith("data:image/png;base64,"):
        raise ValueError("Mask must be a PNG data URL")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as opened:
        return np.asarray(opened.convert("L")) >= 128


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        temporary, format="PNG", optimize=True
    )
    temporary.replace(path)


def opaque_item_id(seed: int, sample_id: str, repeat_index: int) -> str:
    token = f"{seed}|{sample_id}|{repeat_index}".encode("utf-8")
    return "hg_" + hashlib.sha256(token).hexdigest()[:16]


def select_repeats(
    rows: list[dict[str, str]], per_class: int, seed: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    selected: list[dict[str, str]] = []
    for class_name in sorted(grouped):
        ranked = sorted(
            grouped[class_name],
            key=lambda row: hashlib.sha256(
                f"repeat|{seed}|{row['sample_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_class:
            raise RuntimeError(
                f"Class {class_name} has {len(ranked)} records; needs {per_class}"
            )
        selected.extend(ranked[:per_class])
    return selected


def build_queue(config: dict[str, Any], rebuild: bool = False) -> dict[str, Any]:
    source_path = project_path(config["source_manifest"])
    queue_path = project_path(config["queue_manifest"])
    annotation_path = project_path(config["annotation_manifest"])
    lock_path = project_path(config["queue_lock"])

    if queue_path.exists() and not rebuild:
        return verify_queue(config)
    if rebuild and annotation_path.exists():
        progress = read_csv(annotation_path)
        if any(row.get("review_status") != PENDING_STATUS for row in progress):
            raise RuntimeError(
                "Refusing to rebuild a queue with saved annotation decisions"
            )

    source_rows = read_csv(source_path)
    allowed_splits = set(config["evaluation_splits"])
    evaluation_rows = [
        row for row in source_rows if row["experimental_split"] in allowed_splits
    ]
    sample_ids = [row["sample_id"] for row in evaluation_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Evaluation source contains duplicate sample_id values")

    split_counts = Counter(row["experimental_split"] for row in evaluation_rows)
    class_counts = Counter(row["class_name"] for row in evaluation_rows)
    expected = {
        "unique": int(config["expected_unique_records"]),
        "validation": int(config["expected_validation_records"]),
        "test": int(config["expected_test_records"]),
        "classes": int(config["expected_classes"]),
    }
    actual = {
        "unique": len(evaluation_rows),
        "validation": split_counts.get("validation", 0),
        "test": split_counts.get("test", 0),
        "classes": len(class_counts),
    }
    if actual != expected:
        raise RuntimeError(f"Evaluation count mismatch: expected={expected}, actual={actual}")

    for row in evaluation_rows:
        for field in (
            "image_relpath",
            "symptom_mask_relpath",
            "pseudo_host_mask_relpath",
        ):
            path = project_path(row[field])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {field} for {row['sample_id']}: {path}")
        image_size = load_image_size(project_path(row["image_relpath"]))
        symptom_size = load_image_size(project_path(row["symptom_mask_relpath"]))
        pseudo_size = load_image_size(project_path(row["pseudo_host_mask_relpath"]))
        if image_size != symptom_size or image_size != pseudo_size:
            raise RuntimeError(
                f"Shape mismatch for {row['sample_id']}: "
                f"image={image_size}, symptom={symptom_size}, pseudo={pseudo_size}"
            )

    seed = int(config["queue_seed"])
    rng = random.Random(seed)
    primary_rows = list(evaluation_rows)
    rng.shuffle(primary_rows)
    repeated_sources = select_repeats(
        evaluation_rows, int(config["blind_repeats_per_class"]), seed
    )
    rng.shuffle(repeated_sources)

    primary_dir = str(config["primary_output_dir"]).rstrip("/\\")
    repeat_dir = str(config["repeat_output_dir"]).rstrip("/\\")
    queue_rows: list[dict[str, Any]] = []
    for source in primary_rows:
        queue_rows.append(
            {
                "annotation_item_id": opaque_item_id(seed, source["sample_id"], 0),
                "sample_id": source["sample_id"],
                "repeat_index": 0,
                "is_repeat": "false",
                "class_name": source["class_name"],
                "experimental_split": source["experimental_split"],
                "image_relpath": source["image_relpath"],
                "symptom_mask_relpath": source["symptom_mask_relpath"],
                "pseudo_host_mask_relpath": source["pseudo_host_mask_relpath"],
                "output_mask_relpath": f"{primary_dir}/{source['sample_id']}.png",
            }
        )
    for source in repeated_sources:
        item_id = opaque_item_id(seed, source["sample_id"], 1)
        queue_rows.append(
            {
                "annotation_item_id": item_id,
                "sample_id": source["sample_id"],
                "repeat_index": 1,
                "is_repeat": "true",
                "class_name": source["class_name"],
                "experimental_split": source["experimental_split"],
                "image_relpath": source["image_relpath"],
                "symptom_mask_relpath": source["symptom_mask_relpath"],
                "pseudo_host_mask_relpath": source["pseudo_host_mask_relpath"],
                "output_mask_relpath": f"{repeat_dir}/{item_id}.png",
            }
        )
    for position, row in enumerate(queue_rows, start=1):
        row["queue_position"] = position

    expected_repeats = int(config["expected_classes"]) * int(
        config["blind_repeats_per_class"]
    )
    if len(queue_rows) != expected["unique"] + expected_repeats:
        raise RuntimeError("Constructed queue has an unexpected length")

    project_path(config["primary_output_dir"]).mkdir(parents=True, exist_ok=True)
    project_path(config["repeat_output_dir"]).mkdir(parents=True, exist_ok=True)
    project_path(config["revision_output_dir"]).mkdir(parents=True, exist_ok=True)
    atomic_write_csv(queue_path, queue_rows, QUEUE_FIELDS)

    annotations = [
        {
            "annotation_item_id": row["annotation_item_id"],
            "review_status": PENDING_STATUS,
            "decision": "",
            "quality_flags": "",
            "notes": "",
            "annotation_seconds": "0.000",
            "annotator_id": config["annotator_id"],
            "completed_at_utc": "",
            "mask_sha256": "",
            "mask_pixels": "",
            "pseudo_difference_pixels": "",
            "symptom_outside_host_pixels": "",
        }
        for row in queue_rows
    ]
    atomic_write_csv(annotation_path, annotations, ANNOTATION_FIELDS)

    lock = {
        "protocol": config["protocol"],
        "created_at_utc": utc_now(),
        "status": "HOST_GOLD_QUEUE_LOCKED",
        "test_model_outputs_opened": False,
        "unique_records": len(evaluation_rows),
        "blind_repeat_records": expected_repeats,
        "total_annotation_items": len(queue_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "repeats_per_class": dict(
            sorted(Counter(row["class_name"] for row in repeated_sources).items())
        ),
        "artifacts": {},
    }
    for relative in (
        config["config_path"],
        config["annotation_script"],
        config["annotation_ui"],
        config["source_manifest"],
        config["queue_manifest"],
    ):
        artifact_path = project_path(relative)
        lock["artifacts"][relative] = {
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        }
    atomic_write_json(lock_path, lock)
    return verify_queue(config)


def verify_queue(config: dict[str, Any]) -> dict[str, Any]:
    source_path = project_path(config["source_manifest"])
    queue_path = project_path(config["queue_manifest"])
    annotation_path = project_path(config["annotation_manifest"])
    lock_path = project_path(config["queue_lock"])
    for path in (source_path, queue_path, annotation_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required annotation artifact is missing: {path}")
    lock = load_json(lock_path)
    if lock.get("protocol") != config["protocol"]:
        raise RuntimeError(
            f"Queue protocol mismatch: lock={lock.get('protocol')}, "
            f"config={config['protocol']}"
        )
    for relative, expected in lock["artifacts"].items():
        path = project_path(relative)
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Locked size mismatch: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Locked SHA-256 mismatch: {relative}")
    queue = read_csv(queue_path)
    annotations = read_csv(annotation_path)
    if len(queue) != int(lock["total_annotation_items"]):
        raise RuntimeError("Queue length differs from queue lock")
    if [row["annotation_item_id"] for row in queue] != [
        row["annotation_item_id"] for row in annotations
    ]:
        raise RuntimeError("Annotation manifest is not aligned with the locked queue")
    return {
        "status": "PASS",
        "queue_records": len(queue),
        "unique_records": int(lock["unique_records"]),
        "blind_repeats": int(lock["blind_repeat_records"]),
        "queue_sha256": lock["artifacts"][config["queue_manifest"]]["sha256"],
    }


def migrate_v1_to_v2_if_needed(config: dict[str, Any]) -> bool:
    """Preserve all decisions while enabling audited symptom-label exceptions."""
    lock_path = project_path(config["queue_lock"])
    if not lock_path.is_file():
        return False
    old_lock = load_json(lock_path)
    old_protocol = old_lock.get("protocol")
    if old_protocol == config["protocol"]:
        return False
    if old_protocol != config.get("migrates_from_protocol"):
        raise RuntimeError(
            f"Unsupported host-gold protocol migration: {old_protocol} -> "
            f"{config['protocol']}"
        )

    queue_path = project_path(config["queue_manifest"])
    annotation_path = project_path(config["annotation_manifest"])
    source_path = project_path(config["source_manifest"])
    queue = read_csv(queue_path)
    annotations = read_csv(annotation_path)
    if [row["annotation_item_id"] for row in queue] != [
        row["annotation_item_id"] for row in annotations
    ]:
        raise RuntimeError("Cannot migrate: annotation manifest is not queue-aligned")

    old_queue_expected = old_lock.get("artifacts", {}).get(config["queue_manifest"])
    old_source_expected = old_lock.get("artifacts", {}).get(config["source_manifest"])
    for path, expected, label in (
        (queue_path, old_queue_expected, "queue"),
        (source_path, old_source_expected, "source manifest"),
    ):
        if not expected:
            raise RuntimeError(f"Cannot migrate: v1 lock does not contain {label}")
        if path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Cannot migrate: locked {label} changed")

    row_by_id = {row["annotation_item_id"]: row for row in queue}
    for annotation in annotations:
        item_id = annotation["annotation_item_id"]
        outside = ""
        if annotation.get("review_status") == APPROVED_STATUS:
            queue_row = row_by_id[item_id]
            output_path = project_path(queue_row["output_mask_relpath"])
            if not output_path.is_file():
                raise RuntimeError(f"Cannot migrate: approved mask missing for {item_id}")
            mask = load_binary_mask(output_path)
            symptom = load_binary_mask(project_path(queue_row["symptom_mask_relpath"]))
            outside = str(int(np.logical_and(symptom, np.logical_not(mask)).sum()))
        annotation["symptom_outside_host_pixels"] = outside
    atomic_write_csv(annotation_path, annotations, ANNOTATION_FIELDS)

    primary = [row for row in queue if row["is_repeat"] == "false"]
    repeated = [row for row in queue if row["is_repeat"] == "true"]
    split_counts = Counter(row["experimental_split"] for row in primary)
    class_counts = Counter(row["class_name"] for row in primary)
    new_lock = {
        "protocol": config["protocol"],
        "created_at_utc": utc_now(),
        "status": "HOST_GOLD_QUEUE_LOCKED",
        "migrated_from_protocol": old_protocol,
        "migrated_from_lock_sha256": sha256_file(lock_path),
        "migration_preserved_annotation_status_counts": dict(
            sorted(Counter(row["review_status"] for row in annotations).items())
        ),
        "test_model_outputs_opened": False,
        "unique_records": len(primary),
        "blind_repeat_records": len(repeated),
        "total_annotation_items": len(queue),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "repeats_per_class": dict(
            sorted(Counter(row["class_name"] for row in repeated).items())
        ),
        "symptom_label_exception_policy": {
            "allowed": True,
            "required_flag": "symptom_label_inconsistency",
            "required_note": True,
            "symptom_mask_is_never_modified": True,
        },
        "artifacts": {},
    }
    for relative in (
        config["config_path"],
        config["annotation_script"],
        config["annotation_ui"],
        config["source_manifest"],
        config["queue_manifest"],
    ):
        artifact_path = project_path(relative)
        new_lock["artifacts"][relative] = {
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        }
    atomic_write_json(lock_path, new_lock)
    event_path = project_path(config["event_log"])
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "protocol": config["protocol"],
                    "event": "protocol_migrated",
                    "from_protocol": old_protocol,
                    "approved_records_preserved": sum(
                        row["review_status"] == APPROVED_STATUS for row in annotations
                    ),
                    "at_utc": utc_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return True


class AnnotationStore:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.queue_path = project_path(config["queue_manifest"])
        self.annotation_path = project_path(config["annotation_manifest"])
        self.event_log_path = project_path(config["event_log"])
        self.revision_dir = project_path(config["revision_output_dir"])
        self.lock = threading.RLock()
        self.queue = read_csv(self.queue_path)
        self._reload_annotations()

    def _reload_annotations(self) -> None:
        rows = read_csv(self.annotation_path)
        self.annotations = {row["annotation_item_id"]: row for row in rows}

    def _ordered_annotations(self) -> list[dict[str, Any]]:
        return [self.annotations[row["annotation_item_id"]] for row in self.queue]

    def summary(self) -> dict[str, Any]:
        counts = Counter(row["review_status"] for row in self.annotations.values())
        return {
            "total": len(self.queue),
            "approved": counts.get(APPROVED_STATUS, 0),
            "uncertain": counts.get(UNCERTAIN_STATUS, 0),
            "pending": counts.get(PENDING_STATUS, 0),
            "remaining": len(self.queue) - counts.get(APPROVED_STATUS, 0),
        }

    def first_pending_position(self) -> int | None:
        for index, row in enumerate(self.queue, start=1):
            if self.annotations[row["annotation_item_id"]]["review_status"] == PENDING_STATUS:
                return index
        for index, row in enumerate(self.queue, start=1):
            if self.annotations[row["annotation_item_id"]]["review_status"] == UNCERTAIN_STATUS:
                return index
        return None

    def public_item(self, position: int) -> dict[str, Any]:
        if position < 1 or position > len(self.queue):
            raise IndexError("Queue position outside valid range")
        row = self.queue[position - 1]
        annotation = self.annotations[row["annotation_item_id"]]
        image_path = project_path(row["image_relpath"])
        symptom_path = project_path(row["symptom_mask_relpath"])
        pseudo_path = project_path(row["pseudo_host_mask_relpath"])
        output_path = project_path(row["output_mask_relpath"])

        symptom = load_binary_mask(symptom_path)
        pseudo = load_binary_mask(pseudo_path)
        if symptom.shape != pseudo.shape:
            raise RuntimeError("Locked symptom and pseudo masks have different shapes")
        initial_repair = np.logical_and(symptom, np.logical_not(pseudo))
        initial = np.logical_or(pseudo, symptom)
        if output_path.is_file():
            saved = load_binary_mask(output_path)
            if saved.shape != symptom.shape:
                raise RuntimeError("Saved host-gold mask shape is invalid")
            initial = saved

        width, height = load_image_size(image_path)
        if initial.shape != (height, width):
            raise RuntimeError("Image and mask dimensions do not match")
        return {
            "protocol": self.config["protocol"],
            "annotation_item_id": row["annotation_item_id"],
            "position": position,
            "total": len(self.queue),
            "width": width,
            "height": height,
            "image": image_data_url(image_path),
            "symptom_mask": png_data_url(symptom.astype(np.uint8) * 255),
            "pseudo_mask": png_data_url(pseudo.astype(np.uint8) * 255),
            "editable_mask": png_data_url(initial.astype(np.uint8) * 255),
            "initial_symptom_union_pixels": int(initial_repair.sum()),
            "review_status": annotation["review_status"],
            "decision": annotation["decision"],
            "quality_flags": [
                value for value in annotation["quality_flags"].split("|") if value
            ],
            "notes": annotation["notes"],
            "summary": self.summary(),
        }

    def _write_annotations(self) -> None:
        atomic_write_csv(
            self.annotation_path, self._ordered_annotations(), ANNOTATION_FIELDS
        )

    def _append_event(self, payload: dict[str, Any]) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_mask(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("annotation_item_id", ""))
        row_by_id = {row["annotation_item_id"]: row for row in self.queue}
        if item_id not in row_by_id:
            raise ValueError("Unknown annotation item")
        row = row_by_id[item_id]
        mask = decode_mask_data_url(str(payload.get("mask_png", "")))
        symptom = load_binary_mask(project_path(row["symptom_mask_relpath"]))
        pseudo = load_binary_mask(project_path(row["pseudo_host_mask_relpath"]))
        if mask.shape != symptom.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match source shape {symptom.shape}"
            )
        flags = payload.get("quality_flags") or []
        if not isinstance(flags, list):
            raise ValueError("quality_flags must be a list")
        cleaned_flags = sorted(
            {
                str(value).strip()
                for value in flags
                if str(value).strip() and "|" not in str(value)
            }
        )
        notes = str(payload.get("notes") or "").strip()[:2000]
        missing_symptom = int(np.logical_and(symptom, np.logical_not(mask)).sum())
        if missing_symptom:
            if "symptom_label_inconsistency" not in cleaned_flags:
                raise ValueError(
                    f"Host mask excludes {missing_symptom} symptom pixels. If those "
                    "red pixels are truly outside every affected host organ, select "
                    "'Nghi ngờ nhãn triệu chứng' and add a note. Do not paint background "
                    "as host."
                )
            if not notes:
                raise ValueError(
                    "A note is required when symptom pixels are outside host_gold."
                )
        if int(mask.sum()) == 0:
            raise ValueError("Host mask cannot be empty")

        output_path = project_path(row["output_mask_relpath"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if output_path.is_file():
            revision = self.revision_dir / f"{item_id}_{timestamp}.png"
            revision.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, revision)
        save_binary_mask(output_path, mask)

        differences = int(np.logical_xor(mask, pseudo).sum())
        decision = "accepted_pseudo" if differences == 0 else "corrected"
        annotation = self.annotations[item_id]
        previous_seconds = float(annotation.get("annotation_seconds") or 0)
        elapsed = max(0.0, min(float(payload.get("elapsed_seconds") or 0), 86400.0))
        annotation.update(
            {
                "review_status": APPROVED_STATUS,
                "decision": decision,
                "quality_flags": "|".join(cleaned_flags) or "none",
                "notes": notes,
                "annotation_seconds": f"{previous_seconds + elapsed:.3f}",
                "annotator_id": self.config["annotator_id"],
                "completed_at_utc": utc_now(),
                "mask_sha256": sha256_file(output_path),
                "mask_pixels": str(int(mask.sum())),
                "pseudo_difference_pixels": str(differences),
                "symptom_outside_host_pixels": str(missing_symptom),
            }
        )
        self._write_annotations()
        self._append_event(
            {
                "protocol": self.config["protocol"],
                "event": "mask_saved",
                "annotation_item_id": item_id,
                "decision": decision,
                "quality_flags": cleaned_flags or ["none"],
                "symptom_outside_host_pixels": missing_symptom,
                "mask_sha256": annotation["mask_sha256"],
                "at_utc": annotation["completed_at_utc"],
            }
        )
        return {
            "status": "SAVED",
            "decision": decision,
            "summary": self.summary(),
            "next_position": self.first_pending_position(),
        }

    def mark_uncertain(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("annotation_item_id", ""))
        if item_id not in self.annotations:
            raise ValueError("Unknown annotation item")
        flags = payload.get("quality_flags") or []
        cleaned_flags = sorted(
            {
                str(value).strip()
                for value in flags
                if str(value).strip() and "|" not in str(value)
            }
        )
        if not cleaned_flags:
            cleaned_flags = ["uncertain_host_boundary"]
        annotation = self.annotations[item_id]
        previous_seconds = float(annotation.get("annotation_seconds") or 0)
        elapsed = max(0.0, min(float(payload.get("elapsed_seconds") or 0), 86400.0))
        annotation.update(
            {
                "review_status": UNCERTAIN_STATUS,
                "decision": "needs_resolution",
                "quality_flags": "|".join(cleaned_flags),
                "notes": str(payload.get("notes") or "").strip()[:2000],
                "annotation_seconds": f"{previous_seconds + elapsed:.3f}",
                "annotator_id": self.config["annotator_id"],
                "completed_at_utc": utc_now(),
                "mask_sha256": "",
                "mask_pixels": "",
                "pseudo_difference_pixels": "",
                "symptom_outside_host_pixels": "",
            }
        )
        self._write_annotations()
        self._append_event(
            {
                "protocol": self.config["protocol"],
                "event": "marked_uncertain",
                "annotation_item_id": item_id,
                "quality_flags": cleaned_flags,
                "at_utc": annotation["completed_at_utc"],
            }
        )
        return {
            "status": "UNCERTAIN",
            "summary": self.summary(),
            "next_position": self.first_pending_position(),
        }


def validation_payload(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    verify_queue(config)
    queue = read_csv(project_path(config["queue_manifest"]))
    annotations = {
        row["annotation_item_id"]: row
        for row in read_csv(project_path(config["annotation_manifest"]))
    }
    failures: list[str] = []
    primary_masks: dict[str, np.ndarray] = {}
    repeat_scores: list[dict[str, Any]] = []
    status_counts = Counter(row["review_status"] for row in annotations.values())
    decision_counts = Counter(row["decision"] for row in annotations.values())
    flag_counts: Counter[str] = Counter()
    total_seconds = 0.0
    artifacts: dict[str, dict[str, Any]] = {}
    symptom_exception_records: list[dict[str, Any]] = []

    for annotation in annotations.values():
        total_seconds += float(annotation.get("annotation_seconds") or 0)
        for flag in annotation.get("quality_flags", "").split("|"):
            if flag:
                flag_counts[flag] += 1

    for row in queue:
        item_id = row["annotation_item_id"]
        annotation = annotations[item_id]
        if annotation["review_status"] != APPROVED_STATUS:
            failures.append(f"{item_id}: status={annotation['review_status']}")
            continue
        output_path = project_path(row["output_mask_relpath"])
        if not output_path.is_file():
            failures.append(f"{item_id}: approved mask is missing")
            continue
        try:
            mask = load_binary_mask(output_path)
            symptom = load_binary_mask(project_path(row["symptom_mask_relpath"]))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{item_id}: cannot read mask ({exc})")
            continue
        if mask.shape != symptom.shape:
            failures.append(f"{item_id}: shape mismatch")
            continue
        missing = int(np.logical_and(symptom, np.logical_not(mask)).sum())
        flags = {
            value
            for value in annotation.get("quality_flags", "").split("|")
            if value and value != "none"
        }
        recorded_missing = int(annotation.get("symptom_outside_host_pixels") or 0)
        if recorded_missing != missing:
            failures.append(
                f"{item_id}: symptom-outside-host count differs from manifest "
                f"({recorded_missing} != {missing})"
            )
        if missing:
            if "symptom_label_inconsistency" not in flags:
                failures.append(
                    f"{item_id}: excludes {missing} symptom pixels without the "
                    "required symptom-label-inconsistency flag"
                )
            if not annotation.get("notes", "").strip():
                failures.append(
                    f"{item_id}: symptom-label exception has no explanatory note"
                )
            symptom_exception_records.append(
                {
                    "annotation_item_id": item_id,
                    "sample_id": row["sample_id"],
                    "experimental_split": row["experimental_split"],
                    "class_name": row["class_name"],
                    "symptom_outside_host_pixels": missing,
                    "is_repeat": row["is_repeat"] == "true",
                }
            )
        actual_sha = sha256_file(output_path)
        if actual_sha != annotation["mask_sha256"]:
            failures.append(f"{item_id}: saved SHA-256 differs from manifest")
        relative = output_path.relative_to(ROOT).as_posix()
        artifacts[relative] = {
            "bytes": output_path.stat().st_size,
            "sha256": actual_sha,
        }
        if row["is_repeat"] == "false":
            primary_masks[row["sample_id"]] = mask

    for row in queue:
        if row["is_repeat"] != "true":
            continue
        annotation = annotations[row["annotation_item_id"]]
        if annotation["review_status"] != APPROVED_STATUS:
            continue
        repeat_path = project_path(row["output_mask_relpath"])
        primary = primary_masks.get(row["sample_id"])
        if primary is None or not repeat_path.is_file():
            continue
        repeated = load_binary_mask(repeat_path)
        intersection = int(np.logical_and(primary, repeated).sum())
        denominator = int(primary.sum()) + int(repeated.sum())
        dice = 1.0 if denominator == 0 else (2.0 * intersection) / denominator
        union = int(np.logical_or(primary, repeated).sum())
        iou = 1.0 if union == 0 else intersection / union
        repeat_scores.append(
            {
                "annotation_item_id": row["annotation_item_id"],
                "class_name": row["class_name"],
                "dice": dice,
                "iou": iou,
            }
        )

    complete = not failures and len(queue) == status_counts.get(APPROVED_STATUS, 0)
    report = {
        "protocol": config["protocol"],
        "generated_at_utc": utc_now(),
        "status": "PASS" if complete else "INCOMPLETE",
        "unique_evaluation_records": int(config["expected_unique_records"]),
        "blind_repeat_records": int(config["expected_classes"])
        * int(config["blind_repeats_per_class"]),
        "total_annotation_items": len(queue),
        "status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "symptom_label_exceptions": {
            "records": len(symptom_exception_records),
            "pixels": sum(
                row["symptom_outside_host_pixels"]
                for row in symptom_exception_records
            ),
            "items": symptom_exception_records,
            "policy": "outside-host symptom pixels remain immutable in symptom_gold; "
            "they are excluded from the host-normalized burden numerator by "
            "intersection with host_gold and reported as a QC stratum",
        },
        "annotation_seconds_total": total_seconds,
        "intra_annotator_repeatability": {
            "completed_pairs": len(repeat_scores),
            "mean_dice": float(np.mean([row["dice"] for row in repeat_scores]))
            if repeat_scores
            else None,
            "median_dice": float(np.median([row["dice"] for row in repeat_scores]))
            if repeat_scores
            else None,
            "mean_iou": float(np.mean([row["iou"] for row in repeat_scores]))
            if repeat_scores
            else None,
            "pairs": repeat_scores,
        },
        "failure_count": len(failures),
        "failures": failures[:100],
        "test_model_outputs_opened_for_annotation": False,
        "host_masks_are_evaluation_references_only": True,
    }
    report_path = project_path(config["validation_report"])
    atomic_write_json(report_path, report)

    if complete:
        fixed_artifacts = {
            config["queue_manifest"]: {
                "bytes": project_path(config["queue_manifest"]).stat().st_size,
                "sha256": sha256_file(project_path(config["queue_manifest"])),
            },
            config["queue_lock"]: {
                "bytes": project_path(config["queue_lock"]).stat().st_size,
                "sha256": sha256_file(project_path(config["queue_lock"])),
            },
            config["annotation_manifest"]: {
                "bytes": project_path(config["annotation_manifest"]).stat().st_size,
                "sha256": sha256_file(project_path(config["annotation_manifest"])),
            },
            config["validation_report"]: {
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            },
        }
        fixed_artifacts.update(dict(sorted(artifacts.items())))
        completion_lock = {
            "protocol": config["protocol"],
            "created_at_utc": utc_now(),
            "status": "HOST_GOLD_COMPLETE",
            "test_model_outputs_opened_for_annotation": False,
            "artifacts": fixed_artifacts,
        }
        atomic_write_json(project_path(config["completion_lock"]), completion_lock)
    return report, complete


class AnnotationHandler(BaseHTTPRequestHandler):
    store: AnnotationStore
    ui_bytes: bytes

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[host-gold-ui] {self.address_string()} - {fmt % args}")

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"status": "ERROR", "message": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_bytes(self.ui_bytes, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/summary":
                self._send_json(self.store.summary())
                return
            if parsed.path == "/api/item":
                query = parse_qs(parsed.query)
                if query.get("pending", [""])[0] == "1":
                    position = self.store.first_pending_position()
                    if position is None:
                        self._send_json(
                            {"status": "COMPLETE", "summary": self.store.summary()}
                        )
                        return
                else:
                    position = int(query.get("position", ["1"])[0])
                self._send_json(self.store.public_item(position))
                return
            if parsed.path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
                return
            self._error("Not found", HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - server diagnostic path
            self._error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with self.store.lock:
                if parsed.path == "/api/save":
                    result = self.store.save_mask(payload)
                elif parsed.path == "/api/uncertain":
                    result = self.store.mark_uncertain(payload)
                else:
                    self._error("Not found", HTTPStatus.NOT_FOUND)
                    return
            self._send_json(result)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - server diagnostic path
            self._error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rebuild-queue", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    migrated = migrate_v1_to_v2_if_needed(config)
    if migrated:
        print(
            "Migrated host-gold protocol v1 -> v2 without resetting annotation progress."
        )
    queue_status = build_queue(config, rebuild=args.rebuild_queue)
    if args.initialize_only:
        print(json.dumps(queue_status, indent=2, ensure_ascii=False))
        return 0

    store = AnnotationStore(config)
    if args.status:
        print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
        return 0
    if args.validate_only:
        report, complete = validation_payload(config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if complete else 2

    if not UI_PATH.is_file():
        raise FileNotFoundError(f"Annotation UI is missing: {UI_PATH}")
    AnnotationHandler.store = store
    AnnotationHandler.ui_bytes = UI_PATH.read_bytes()
    host = str(config.get("bind_host", "127.0.0.1"))
    port = int(args.port or config.get("port", 7861))
    server = ThreadingHTTPServer((host, port), AnnotationHandler)
    url = f"http://{host}:{port}"
    print(f"Host-gold annotation UI: {url}")
    print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
    print("Press Ctrl+C to stop; progress is saved after every decision.")
    if bool(config.get("open_browser", True)) and not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping host-gold annotation server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
