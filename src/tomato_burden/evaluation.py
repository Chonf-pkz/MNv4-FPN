"""Paper-aligned record selection, preprocessing, scoring, and score I/O."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from . import data
from .models import PlantSegV2
from .training import (
    boundary_f1_np,
    burden_metrics,
    burden_np,
    dice_np,
    iou_np,
    to_canvas,
    tomato_records,
    write_json,
)


class SegmentationOnly(nn.Module):
    """Deployment contract: lesion and host logits at input resolution."""

    def __init__(self, full: PlantSegV2) -> None:
        super().__init__()
        self.backbone, self.decoder = full.backbone, full.decoder
        self.symptom_head, self.host_head = full.symptom_head, full.host_head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.decoder(self.backbone(images))
        size = images.shape[-2:]
        symptom = F.interpolate(self.symptom_head(decoded), size=size, mode="bilinear", align_corners=False)
        host = F.interpolate(self.host_head(decoded), size=size, mode="bilinear", align_corners=False)
        return symptom, host


def letterbox_input(image: Image.Image, size: int) -> tuple[np.ndarray, dict]:
    geometry = data.letterbox_geometry(*image.size, size)
    boxed = data.letterbox_pil(image, size, Image.Resampling.BILINEAR)
    tensor = data.normalize(torch.from_numpy(np.array(boxed)).permute(2, 0, 1).contiguous())[None]
    return tensor.numpy(), geometry


def stratified(records: list, count: int, seed: int) -> list:
    grouped = defaultdict(list)
    for record in sorted(records, key=lambda row: row.sample_id):
        grouped[record.class_name].append(record)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    selected, index = [], {name: 0 for name in grouped}
    while len(selected) < count:
        progressed = False
        for name in sorted(grouped):
            if index[name] < len(grouped[name]) and len(selected) < count:
                selected.append(grouped[name][index[name]])
                index[name] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _validate_cohort(records: list, split: str, expected: int) -> list:
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} {split} records, found {len(records)}")
    identifiers = [row.sample_id for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Duplicate sample_id values found in the {split} cohort")
    wrong = sorted({row.split for row in records if row.split != split})
    if wrong:
        raise RuntimeError(f"Unexpected split labels in {split} cohort: {wrong}")
    return records


def train_records(recipe: dict, expected: int = 595) -> list:
    records = [row for row in tomato_records(recipe) if row.split == "train"]
    return _validate_cohort(records, "train", expected)


def validation_records(recipe: dict, expected: int = 85) -> list:
    records = [row for row in tomato_records(recipe, use_cache=False) if row.split == "validation"]
    return _validate_cohort(records, "validation", expected)


def test_records(recipe: dict, expected: int = 176) -> list:
    paths = recipe["paths"]
    records = data.load_tomato_records(
        Path(paths["tomato_manifest"]),
        Path(paths["pseudo_host_manifest"]),
        Path(paths["dataset_root"]),
        Path(paths["input_root"]),
        None,
    )
    return _validate_cohort([row for row in records if row.split == "test"], "test", expected)


def gold_hosts(recipe: dict) -> dict[str, Path]:
    paths = recipe["paths"]
    rows = data.read_csv(Path(paths["host_gold_manifest"]))
    if len(rows) != 261 or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Expected 261 unique gold host rows, found {len(rows)} rows")
    hosts = {
        row["sample_id"]: Path(paths["host_gold_root"]) / row["final_mask_relpath"]
        for row in rows
    }
    return hosts


def score_records(
    session,
    records: list,
    gold: dict,
    focal: set,
    size: int,
    reference: dict | None = None,
    limit: int = 0,
) -> tuple[list[dict], dict]:
    """Score in the locked 384-pixel letterbox space."""
    eval_size = 384
    rows, maps = [], {}
    for record in records[: limit or None]:
        image, symptom, _ = data.load_record_arrays(record)
        inputs, geometry = letterbox_input(image, size)
        symptom_logit, host_logit = session.run(None, {"image": inputs})
        symptom_map = to_canvas(symptom_logit[0, 0], geometry, eval_size) >= 0.0
        host_map = to_canvas(host_logit[0, 0], geometry, eval_size) >= 0.0
        symptom_true = (
            np.asarray(
                data.letterbox_pil(
                    Image.fromarray(symptom * 255), eval_size, Image.Resampling.NEAREST
                )
            )
            != 0
        )
        row = {
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "focal": record.class_name in focal,
            "duplicate_group": record.duplicate_group or record.sample_id,
            "symptom_dice": dice_np(symptom_map, symptom_true),
            "symptom_iou": iou_np(symptom_map, symptom_true),
            "symptom_bf1": boundary_f1_np(symptom_map, symptom_true),
            "symptom_tp": int(np.logical_and(symptom_map, symptom_true).sum()),
            "symptom_pred_pixels": int(symptom_map.sum()),
            "symptom_true_pixels": int(symptom_true.sum()),
        }
        if record.sample_id in gold:
            with Image.open(gold[record.sample_id]) as opened:
                host_true = (
                    np.asarray(
                        data.letterbox_pil(opened.convert("L"), eval_size, Image.Resampling.NEAREST)
                    )
                    != 0
                )
            gold_burden = burden_np(symptom_true, host_true)
            e2e_burden = burden_np(symptom_map, host_map)
            gold_host_burden = burden_np(symptom_map, host_true)
            row.update(
                {
                    "host_dice_gold": dice_np(host_map, host_true),
                    "host_iou_gold": iou_np(host_map, host_true),
                    "gold_burden_pp": gold_burden,
                    "e2e_burden_pp": e2e_burden,
                    "gold_host_burden_pred_pp": gold_host_burden,
                    "e2e_abs_error_pp": abs(e2e_burden - gold_burden),
                    "e2e_signed_error_pp": e2e_burden - gold_burden,
                    "gold_host_abs_error_pp": abs(gold_host_burden - gold_burden),
                    "gold_host_signed_error_pp": gold_host_burden - gold_burden,
                }
            )
        if reference is not None:
            ref = reference["rows"][record.sample_id]
            ref_symptom, ref_host = (
                np.unpackbits(mask)[: eval_size * eval_size]
                .reshape(eval_size, eval_size)
                .astype(bool)
                for mask in reference["maps"][record.sample_id]
            )
            row.update(
                {
                    "symptom_dice_vs_fp32": dice_np(symptom_map, ref_symptom),
                    "host_dice_vs_fp32": dice_np(host_map, ref_host),
                    "symptom_dice_delta_vs_fp32": row["symptom_dice"] - ref["symptom_dice"],
                }
            )
            if "e2e_burden_pp" in row and "e2e_burden_pp" in ref:
                row["drift_e2e_pp"] = abs(row["e2e_burden_pp"] - ref["e2e_burden_pp"])
                row["drift_gold_host_pp"] = abs(
                    row["gold_host_burden_pred_pp"] - ref["gold_host_burden_pred_pp"]
                )
        rows.append(row)
        maps[record.sample_id] = (np.packbits(symptom_map), np.packbits(host_map))
    return rows, maps


def summarize_rows(rows: list[dict]) -> dict:
    focal_rows = [row for row in rows if row["focal"]]

    def mean(key, subset=rows):
        values = [float(row[key]) for row in subset if key in row]
        return float(np.mean(values)) if values else None

    def largest(key):
        values = [float(row[key]) for row in rows if key in row]
        return float(np.max(values)) if values else None

    def burden(key, subset):
        scored = [row for row in subset if key in row]
        return burden_metrics(
            [row[key] for row in scored], [row["gold_burden_pp"] for row in scored]
        )

    details = {
        "gold_host_all": burden("gold_host_burden_pred_pp", rows),
        "gold_host_focal": burden("gold_host_burden_pred_pp", focal_rows),
        "e2e_all": burden("e2e_burden_pp", rows),
        "e2e_focal": burden("e2e_burden_pp", focal_rows),
    }
    summary = {
        "records": len(rows),
        "focal_records": len(focal_rows),
        "symptom_mean_dice": mean("symptom_dice"),
        "symptom_mean_iou": mean("symptom_iou"),
        "symptom_mean_bf1": mean("symptom_bf1"),
        "symptom_micro_dice": 2 * sum(row["symptom_tp"] for row in rows)
        / max(1, sum(row["symptom_pred_pixels"] + row["symptom_true_pixels"] for row in rows)),
        "symptom_mean_dice_focal": mean("symptom_dice", focal_rows),
        "host_mean_dice_gold": mean("host_dice_gold"),
        "host_mean_iou_gold": mean("host_iou_gold"),
        "drift_gold_host_mean_pp": mean("drift_gold_host_pp"),
        "drift_gold_host_max_pp": largest("drift_gold_host_pp"),
        "drift_e2e_mean_pp": mean("drift_e2e_pp"),
        "drift_e2e_max_pp": largest("drift_e2e_pp"),
        "drift_gold_host_mean_pp_focal": mean("drift_gold_host_pp", focal_rows),
        "symptom_dice_vs_fp32_mean": mean("symptom_dice_vs_fp32"),
        "host_dice_vs_fp32_mean": mean("host_dice_vs_fp32"),
        "burden_details": details,
    }
    for name, values in details.items():
        prefix = "gold_host_burden" if name.startswith("gold_host") else "e2e_burden"
        scope = name.rsplit("_", 1)[1]
        summary.update(
            {
                f"{prefix}_mae_pp_{scope}": values.get("mae_pp"),
                f"{prefix}_mape_pct_{scope}": values.get("mape_pct"),
                f"{prefix}_r2_{scope}": values.get("r2"),
            }
        )
    return summary


def save_scores(folder: Path, name: str, rows: list[dict], maps: dict, summary: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"{name}_rows.json", rows)
    data.write_csv(folder / f"{name}_rows.csv", rows, sorted({key for row in rows for key in row}))
    identifiers = sorted(maps)
    np.savez_compressed(
        folder / f"{name}_maps.npz",
        sample_id=np.array(identifiers),
        symptom=np.stack([maps[item][0] for item in identifiers]),
        host=np.stack([maps[item][1] for item in identifiers]),
    )
    write_json(folder / f"{name}_summary.json", summary)


def load_reference(folder: Path, name: str) -> dict:
    rows_path = folder / f"{name}_rows.json"
    maps_path = folder / f"{name}_maps.npz"
    summary_path = folder / f"{name}_summary.json"
    if not rows_path.is_file() or not maps_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"FP32 reference missing: {rows_path}")
    rows = {row["sample_id"]: row for row in json.loads(rows_path.read_text(encoding="utf-8"))}
    with np.load(maps_path, allow_pickle=False) as archive:
        maps = {
            str(identifier): (symptom, host)
            for identifier, symptom, host in zip(
                archive["sample_id"], archive["symptom"], archive["host"]
            )
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"rows": rows, "maps": maps, "model_sha256": summary.get("model_sha256")}

