"""Records, cache, cross-validation folds and paired augmentation for the FP32 v2 recipe."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

REVIEW_SEG_WEIGHT = {"keep": 1.0, "fixed": 1.0, "downweight": 0.3, "exclude_seg": 0.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Record:
    sample_id: str
    source: str  # "tomato" or "non_tomato"
    split: str
    class_name: str
    image: Path
    symptom: Path
    host: Path | None = None
    host_weight: float = 0.0
    burden_weight: float = 0.0
    seg_weight: float = 1.0
    duplicate_group: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        # A reviewer-fixed mask is cached separately so the original label stays available
        # for held-out scoring.
        return f"{self.sample_id}__fixed" if self.extra.get("fixed") else self.sample_id


# ----------------------------------------------------------------------------- records

def load_tomato_records(
    manifest: Path,
    pseudo_host_manifest: Path,
    dataset_root: Path,
    input_root: Path,
    review_csv: Path | None = None,
) -> list[Record]:
    """Phase-1 target rows merged with Phase-2 pseudo-host weights and the train label review."""
    pseudo = {row["sample_id"]: row for row in read_csv(pseudo_host_manifest)}
    review = {row["sample_id"]: row for row in read_csv(review_csv)} if review_csv else {}
    records = []
    for row in read_csv(manifest):
        if row["hard_validation"] != "pass" or row["model_use"] == "none":
            continue
        host_row = pseudo[row["sample_id"]]
        # Pseudo-host masks are shipped for train only; validation/test use host_gold at evaluation.
        host_weight = float(host_row["host_weight"]) if row["experimental_split"] == "train" else 0.0
        record = Record(
            sample_id=row["sample_id"],
            source="tomato",
            split=row["experimental_split"],
            class_name=row["class_name"],
            image=dataset_root / row["image_relpath"],
            symptom=dataset_root / row["symptom_mask_relpath"],
            host=input_root / host_row["pseudo_host_mask_relpath"] if host_weight > 0 else None,
            host_weight=host_weight,
            burden_weight=float(host_row["burden_weight"]) if host_weight > 0 else 0.0,
            duplicate_group=row["duplicate_group"],
            extra={
                "burden_policy": host_row["burden_policy"],
                "host_status": host_row.get("host_status", ""),
            },
        )
        decision = review.get(row["sample_id"])
        if decision is not None:
            if record.split != "train":
                raise RuntimeError(f"Label review may only touch train records: {record.sample_id}")
            action = decision["decision"].strip() or "keep"  # unreviewed rows stay as they are
            if action == "exclude":
                continue
            if action not in REVIEW_SEG_WEIGHT:
                raise ValueError(f"Unknown review decision {action!r} for {record.sample_id}")
            record.seg_weight = REVIEW_SEG_WEIGHT[action]
            if decision.get("fixed_symptom_mask", "").strip():
                fixed = Path(decision["fixed_symptom_mask"].strip())
                record.symptom = fixed if fixed.is_absolute() else dataset_root / fixed
                record.extra["fixed"] = True
            if decision.get("corrected_class", "").strip():
                record.class_name = decision["corrected_class"].strip()
        records.append(record)
    return records


def load_source_records(pretraining_manifest: Path, dataset_root: Path) -> list[Record]:
    """Leakage-audited non-tomato PlantSeg rows (use_for_pretraining == yes)."""
    records = []
    for row in read_csv(pretraining_manifest):
        if row["source_dataset"] != "plantseg_non_tomato" or row["use_for_pretraining"] != "yes":
            continue
        records.append(Record(
            sample_id=row["sample_id"],
            source="non_tomato",
            split="train" if row["pretrain_split"] == "pretrain_train" else "validation",
            class_name=row["class_name"],
            image=dataset_root / row["image_relpath"],
            symptom=dataset_root / row["symptom_mask_relpath"],
        ))
    return records


# ----------------------------------------------------------------------------- cache

def _open_image_matching_mask(image_path: Path, mask_size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as opened:
        opened.load()
        transposed = ImageOps.exif_transpose(opened).convert("RGB")
        if transposed.size == mask_size:
            return transposed
        raw = opened.convert("RGB")
    if raw.size == mask_size:
        return raw
    raise RuntimeError(f"Image/mask size mismatch: {image_path} {transposed.size} vs {mask_size}")


def _cache_one(job: tuple) -> dict:
    record_id, image_path, symptom_path, host_path, out_root, max_side = job
    out_root = Path(out_root)
    with Image.open(symptom_path) as opened:
        symptom = (np.asarray(opened.convert("L")) != 0).astype(np.uint8) * 255
    height, width = symptom.shape
    image = _open_image_matching_mask(Path(image_path), (width, height))
    scale = min(1.0, max_side / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    outputs = {"sample_id": record_id, "orig_width": width, "orig_height": height}
    image_out = out_root / "images" / f"{record_id}.jpg"
    image.resize(size, Image.Resampling.BILINEAR).save(image_out, quality=95)
    outputs["image"] = str(image_out)
    symptom_out = out_root / "symptom" / f"{record_id}.png"
    Image.fromarray(symptom).resize(size, Image.Resampling.NEAREST).save(symptom_out)
    outputs["symptom"] = str(symptom_out)
    outputs["host"] = ""
    if host_path:
        with Image.open(host_path) as opened:
            host = opened.convert("L")
            if host.size != (width, height):
                raise RuntimeError(f"Host/symptom size mismatch: {host_path}")
            host_array = (np.asarray(host) != 0).astype(np.uint8) * 255
        host_out = out_root / "host" / f"{record_id}.png"
        Image.fromarray(host_array).resize(size, Image.Resampling.NEAREST).save(host_out)
        outputs["host"] = str(host_out)
    return outputs


def build_cache(records: list[Record], cache_root: Path, max_side: int, workers: int) -> Path:
    """Resize every train image once (long side <= max_side) so data loading is not the bottleneck."""
    for sub in ("images", "symptom", "host"):
        (cache_root / sub).mkdir(parents=True, exist_ok=True)
    jobs = [
        (r.cache_key, str(r.image), str(r.symptom), str(r.host) if r.host else "", str(cache_root), max_side)
        for r in {r.cache_key: r for r in records}.values()
    ]
    rows = []
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            for index, row in enumerate(pool.imap_unordered(_cache_one, jobs, chunksize=16), start=1):
                rows.append(row)
                if index % 500 == 0:
                    print(f"Cached {index}/{len(jobs)}", flush=True)
    else:
        rows = [_cache_one(job) for job in jobs]
    index_path = cache_root / "cache_index.csv"
    write_csv(index_path, rows, ["sample_id", "image", "symptom", "host", "orig_width", "orig_height"])
    print(f"Cache complete: {len(rows)} records -> {index_path}", flush=True)
    return index_path


def apply_cache(records: list[Record], cache_index: Path) -> list[Record]:
    """Point records at the cached files. Rebuild the cache after fixing any mask."""
    cached = {row["sample_id"]: row for row in read_csv(cache_index)}
    for record in records:
        row = cached.get(record.cache_key)
        if row is None:
            raise RuntimeError(f"Record missing from cache (rerun `cache`): {record.cache_key}")
        record.image = Path(row["image"])
        record.symptom = Path(row["symptom"])
        if record.host is not None:
            if not row["host"]:
                raise RuntimeError(f"Cache has no host mask for {record.sample_id}; rerun `cache`")
            record.host = Path(row["host"])
        record.extra["cached"] = True
    return records


# ----------------------------------------------------------------------------- folds

def make_folds(records: list[Record], n_splits: int, seed: int) -> dict[str, int]:
    """Stratified (class) and grouped (duplicate_group) folds over train records only."""
    from sklearn.model_selection import StratifiedGroupKFold

    train = sorted((r for r in records if r.split == "train"), key=lambda r: r.sample_id)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [r.class_name for r in train]
    groups = [r.duplicate_group or r.sample_id for r in train]
    assignment = {}
    for fold, (_, held_out) in enumerate(splitter.split(train, labels, groups)):
        for index in held_out:
            assignment[train[index].sample_id] = fold
    return assignment


def summarize_folds(records: list[Record], assignment: dict[str, int]) -> dict:
    counts = Counter((assignment[r.sample_id], r.class_name) for r in records if r.sample_id in assignment)
    folds = sorted({fold for fold, _ in counts})
    classes = sorted({name for _, name in counts})
    return {f"fold_{fold}": {name: counts[(fold, name)] for name in classes} for fold in folds}


# ----------------------------------------------------------------------------- transforms

def build_train_transform(size: int, aug: dict) -> v2.Compose:
    return v2.Compose([
        v2.ScaleJitter(target_size=(size, size), scale_range=tuple(aug["scale_range"]), antialias=True),
        v2.RandomCrop(size, pad_if_needed=True, fill=0),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(aug["vflip_probability"]),
        v2.RandomApply([v2.RandomRotation(aug["rotation_degrees"], fill=0)], p=aug["rotation_probability"]),
        v2.ColorJitter(**aug["color_jitter"]),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=aug["blur_probability"]),
    ])


def letterbox_geometry(width: int, height: int, size: int) -> dict:
    scale = size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return {
        "size": size,
        "width": width,
        "height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "left": (size - resized_width) // 2,
        "top": (size - resized_height) // 2,
    }


def letterbox_pil(image: Image.Image, size: int, resample: int) -> Image.Image:
    """Identical to Phase 3/9 resize_and_pad, so v2 numbers stay comparable."""
    geometry = letterbox_geometry(*image.size, size)
    resized = image.resize((geometry["resized_width"], geometry["resized_height"]), resample=resample)
    canvas = Image.new(image.mode, (size, size), color=(0, 0, 0) if image.mode == "RGB" else 0)
    canvas.paste(resized, (geometry["left"], geometry["top"]))
    return canvas


def normalize(image_uint8: torch.Tensor) -> torch.Tensor:
    return (image_uint8.float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _load_mask(path: Path | None, size: tuple[int, int] | None = None) -> np.ndarray:
    if path is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    with Image.open(path) as opened:
        mask = opened.convert("L")
        if size is not None and mask.size != size:
            raise RuntimeError(f"Mask size {mask.size} != image size {size}: {path}")
        return (np.asarray(mask) != 0).astype(np.uint8)


def load_record_arrays(record: Record) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """RGB image plus binary symptom/host masks, all in the same (EXIF-normalized) geometry."""
    symptom = _load_mask(record.symptom)
    size = (symptom.shape[1], symptom.shape[0])
    if record.extra.get("cached"):
        with Image.open(record.image) as opened:
            image = opened.convert("RGB")
        if image.size != size:
            raise RuntimeError(f"Cached image/mask size mismatch: {record.sample_id}")
    else:
        image = _open_image_matching_mask(record.image, size)
    host = _load_mask(record.host, size)
    return image, symptom, host


class SegmentationDataset(Dataset):
    """Tomato or non-tomato records; train mode applies paired augmentation."""

    def __init__(
        self,
        records: list[Record],
        class_to_idx: dict[str, int],
        size: int,
        training: bool,
        aug: dict | None = None,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.size = size
        self.training = training
        self.transform = build_train_transform(size, aug) if training else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image, symptom, host = load_record_arrays(record)
        if self.training:
            image_tv = tv_tensors.Image(torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous())
            masks = tv_tensors.Mask(torch.from_numpy(np.stack([symptom, host])))
            image_tv, masks = self.transform(image_tv, masks)
            image_tensor = normalize(image_tv)
            symptom_tensor = masks[0:1].float()
            host_tensor = masks[1:2].float()
            geometry = letterbox_geometry(*image.size, self.size)
        else:
            geometry = letterbox_geometry(*image.size, self.size)
            boxed = letterbox_pil(image, self.size, Image.Resampling.BILINEAR)
            image_tensor = normalize(torch.from_numpy(np.array(boxed)).permute(2, 0, 1).contiguous())
            symptom_tensor = torch.from_numpy(
                np.asarray(letterbox_pil(Image.fromarray(symptom * 255), self.size, Image.Resampling.NEAREST)) != 0
            ).float()[None]
            host_tensor = torch.from_numpy(
                np.asarray(letterbox_pil(Image.fromarray(host * 255), self.size, Image.Resampling.NEAREST)) != 0
            ).float()[None]
        host_active = record.host is not None and record.host_weight > 0
        burden_target = (symptom_tensor * host_tensor).sum() / (host_tensor.sum() + 1e-6)
        lesion_in_crop = bool(symptom_tensor.sum() > 0)
        return {
            "image": image_tensor,
            "symptom": symptom_tensor,
            "host": host_tensor,
            "disease": torch.tensor(self.class_to_idx.get(record.class_name, -100), dtype=torch.long),
            "disease_weight": torch.tensor(1.0 if lesion_in_crop else 0.25),
            "seg_weight": torch.tensor(float(record.seg_weight)),
            "host_weight": torch.tensor(float(record.host_weight) if host_active else 0.0),
            "burden_weight": torch.tensor(float(record.burden_weight) if host_active else 0.0),
            "burden_target": burden_target.float(),
            "sample_id": record.sample_id,
            "left": geometry["left"],
            "top": geometry["top"],
            "resized_width": geometry["resized_width"],
            "resized_height": geometry["resized_height"],
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def inverse_sqrt_weights(records: list[Record], class_to_idx: dict[str, int]) -> torch.Tensor:
    counts = Counter(r.class_name for r in records)
    weights = torch.tensor([1.0 / math.sqrt(max(1, counts[name])) for name in class_to_idx])
    return weights / weights.mean()
