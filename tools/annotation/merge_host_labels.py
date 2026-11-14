#!/usr/bin/env python3
"""Merge approved train host masks from the annotation packages into one zip for Kaggle.

usage (from this folder):
    python merge_host_labels.py --manifest pseudo_host_manifest.csv host_train_A host_train_B
    -> host_train_v1.zip  (upload to Drive, paste the link into HOST_LABELS_URL in cell 1)

Only items with review_status = approved are used; unfinished or uncertain items keep their
old status (no host label). Every other row of the frozen pseudo-host manifest is copied unchanged.
Weights follow the paper policy: host_weight = 1.0 for a manual mask; burden_weight = 1.0
for focal-lesion classes and 0.0 for the two virus classes (exploratory_diffuse_symptom).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "host_train_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="frozen pseudo-host manifest to update")
    parser.add_argument("packages", nargs="+", type=Path, help="host_train_A host_train_B ...")
    parser.add_argument("--out", type=Path, default=HERE / f"{PREFIX}.zip")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        manifest = list(reader)
    by_id = {row["sample_id"]: row for row in manifest}

    approved: dict[str, tuple[bytes, dict[str, str], str]] = {}
    status = Counter()
    seconds = Counter()
    for package in args.packages:
        folder = package / "host_train"
        queue = read_csv(folder / "host_train_queue.csv")
        annotations = {row["annotation_item_id"]: row for row in read_csv(folder / "host_train_manifest.csv")}
        for item in queue:
            if item["is_repeat"] == "true":
                continue
            annotation = annotations[item["annotation_item_id"]]
            status[(package.name, annotation["review_status"])] += 1
            if annotation["review_status"] != "approved":
                continue
            payload = (package / item["output_mask_relpath"]).read_bytes()
            if sha256_bytes(payload) != annotation["mask_sha256"].upper():
                raise RuntimeError(f"{package.name}: mask changed after saving: {item['sample_id']}")
            sample_id = item["sample_id"]
            if sample_id in approved:
                raise RuntimeError(f"{sample_id} was annotated in two packages")
            row = by_id[sample_id]
            if row["experimental_split"] != "train":
                raise RuntimeError(f"Only train images may receive new host labels: {sample_id}")
            approved[sample_id] = (payload, annotation, package.name)
            seconds[package.name] += float(annotation.get("annotation_seconds") or 0)

    classes = Counter()
    for sample_id, (payload, annotation, package_name) in approved.items():
        row = by_id[sample_id]
        focal = row["burden_policy"] == "primary_focal_lesion"
        row.update({
            "pseudo_host_mask_relpath": f"{PREFIX}/masks/{sample_id}.png",
            "pseudo_host_sha256": sha256_bytes(payload),
            "host_weight": "1.00",
            "burden_weight": "1.00" if focal else "0.00",
            "host_status": "manual_train_v1",
            "host_failure_reason": "",
            "provider": f"manual:{annotation['annotator_id']}",
            "model_id": f"proposal=v2 held-out fold model; package={package_name}",
            "generated_at_utc": annotation["completed_at_utc"],
        })
        classes[row["class_name"]] += 1

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manual_train_masks": len(approved),
        "per_class": dict(sorted(classes.items())),
        "review_status_per_package": {f"{p}:{s}": n for (p, s), n in sorted(status.items())},
        "annotation_hours_per_package": {p: round(s / 3600, 2) for p, s in sorted(seconds.items())},
        "train_images_with_host_label": sum(1 for r in manifest if r["experimental_split"] == "train" and float(r["host_weight"]) > 0),
    }
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{PREFIX}/pseudo_host_manifest.csv", buffer.getvalue())
        archive.writestr(f"{PREFIX}/summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
        for sample_id, (payload, _, _) in sorted(approved.items()):
            archive.writestr(f"{PREFIX}/masks/{sample_id}.png", payload)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
