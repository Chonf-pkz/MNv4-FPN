#!/usr/bin/env python3
"""Final FP32 recipe: MobileNetV4-Conv-Medium + FPN-lite, five-fold CV, co-training, and EMA.

Subcommands (run in this order, see docs/REPRODUCIBILITY.md):
  cache     resize tomato + non-tomato records once (long side <= cache.max_side)
  folds     stratified/grouped 5-fold split of the 595 reviewed train records
  source    non-tomato PlantSeg stage (symptom + 107-class disease), fixed epochs
  target    tomato stage for one fold (--fold 0..4) or on all train records (--fold all)
  evaluate  final check on the 85 validation images in the locked 384-pixel metric space

The target test split is never loaded by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import config as config_io
from . import data
from .losses import source_symptom_loss, target_losses
from .models import PlantSegV2, load_source_stage

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads(config_io.PROTOCOL_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- utilities

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, required: set[str]) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must be a mapping: {path}")
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Checkpoint is missing {sorted(missing)}: {path}")
    return checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def provenance(config_path: Path) -> dict:
    files = [config_path, HERE / "data.py", HERE / "losses.py", HERE / "models.py", Path(__file__).resolve()]
    return {path.name: sha256_file(path) for path in files}


def validate_completed_target(checkpoint_path: Path, config: dict, seed: int, fold: str) -> None:
    checkpoint = load_checkpoint(checkpoint_path, {"config", "seed", "fold", "ema_state"})
    if checkpoint["seed"] != seed or str(checkpoint["fold"]) != str(fold) or checkpoint["config"] != config:
        raise RuntimeError(
            f"Existing checkpoint {checkpoint_path} does not match the requested config, seed, or fold"
        )


def validate_completed_source(checkpoint_path: Path, config: dict, seed: int) -> None:
    checkpoint = load_checkpoint(checkpoint_path, {"config", "seed", "ema_state"})
    if checkpoint["seed"] != seed or checkpoint["config"] != config:
        raise RuntimeError(f"Existing source checkpoint {checkpoint_path} does not match the requested config or seed")


def validate_target_cohort(records: list[data.Record], variant: str) -> dict:
    if variant not in ("standard", "manual-host", "baseline"):
        raise ValueError(f"Unknown target variant: {variant}")
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    train_ids = [record.sample_id for record in train]
    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Reviewed train sample IDs must be unique")
    active_hosts = [record for record in train if record.host_weight > 0 and record.host is not None]
    manual_hosts = [
        record for record in active_hosts
        if str(record.extra.get("host_status", "")).strip().lower().startswith("manual")
    ]
    sam2_statuses = {"high_confidence", "medium_confidence"}
    sam2_hosts = [
        record for record in active_hosts
        if str(record.extra.get("host_status", "")).strip().lower() in sam2_statuses
    ]
    expected_train = int(PROTOCOL["cohort"]["train"])
    expected_validation = int(PROTOCOL["cohort"]["validation"])
    expected_sam2 = int(PROTOCOL["cohort"]["sam2_training_host_masks"])
    expected_manual = int(PROTOCOL["cohort"]["manual_training_host_masks_mh_only"]) if variant == "manual-host" else 0
    expected_hosts = expected_sam2 + expected_manual
    if len(train) != expected_train:
        raise RuntimeError(f"Expected {expected_train} reviewed train records, found {len(train)}")
    if len(validation) != expected_validation:
        raise RuntimeError(f"Expected {expected_validation} validation records, found {len(validation)}")
    if len(active_hosts) != expected_hosts:
        raise RuntimeError(f"{variant} requires {expected_hosts} host-supervised train records, found {len(active_hosts)}")
    if len(manual_hosts) != expected_manual:
        raise RuntimeError(f"{variant} requires {expected_manual} manual host masks, found {len(manual_hosts)}")
    if len(sam2_hosts) != expected_sam2:
        raise RuntimeError(f"{variant} requires {expected_sam2} accepted SAM2 host masks, found {len(sam2_hosts)}")
    if len(manual_hosts) + len(sam2_hosts) != len(active_hosts):
        raise RuntimeError(f"{variant} has active host masks with an unrecognized status")
    return {
        "train_records": len(train),
        "validation_records": len(validation),
        "host_supervised_records": len(active_hosts),
        "sam2_host_masks": len(sam2_hosts),
        "manual_host_masks": len(manual_hosts),
    }


def validate_source_cohort(records: list[data.Record]) -> dict:
    cohort = PROTOCOL["cohort"]
    expected = int(cohort["non_tomato_pretraining"])
    expected_train = int(cohort["non_tomato_pretraining_train"])
    expected_validation = int(cohort["non_tomato_pretraining_validation"])
    ids = [record.sample_id for record in records]
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected:,} active non-tomato pretraining records, found {len(records):,}")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Active non-tomato pretraining sample IDs must be unique")
    train = sum(record.split == "train" for record in records)
    monitor = sum(record.split == "validation" for record in records)
    if train + monitor != expected:
        raise RuntimeError("Source cohort may contain only train and validation records")
    if train != expected_train or monitor != expected_validation:
        raise RuntimeError(
            "Source cohort requires "
            f"{expected_train:,} train and {expected_validation:,} validation records; "
            f"found {train:,} and {monitor:,}"
        )
    return {"records": expected, "train_records": train, "monitor_records": monitor}


def make_loader(dataset, batch_size: int, training: bool, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=data.seed_worker,
        generator=generator,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for ema_value, value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def build_optimizer(model: nn.Module, opt: dict) -> torch.optim.Optimizer:
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "backbone" if name.startswith("backbone.") else "head"
        decay = "no_decay" if parameter.ndim <= 1 else "decay"
        groups[f"{part}_{decay}"].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": groups["backbone_decay"], "lr": opt["backbone_lr"], "weight_decay": opt["weight_decay"]},
            {"params": groups["backbone_no_decay"], "lr": opt["backbone_lr"], "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": opt["head_lr"], "weight_decay": opt["weight_decay"]},
            {"params": groups["head_no_decay"], "lr": opt["head_lr"], "weight_decay": 0.0},
        ]
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_ratio: float):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> None:
    """Recompute BN running stats on tomato images only (co-training mixed the domains)."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in norms]
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        model(batch["image"].to(device, non_blocking=True))
    for module, momentum in zip(norms, saved):
        module.momentum = momentum
    model.eval()


# ----------------------------------------------------------------------------- metrics

def macro_f1(confusion: np.ndarray) -> tuple[float, list[float]]:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    denominator = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(scores.mean()), scores.tolist()


def dice_np(prediction: np.ndarray, truth: np.ndarray) -> float:
    total = prediction.sum() + truth.sum()
    return 1.0 if total == 0 else float(2.0 * np.logical_and(prediction, truth).sum() / total)


def burden_np(symptom: np.ndarray, host: np.ndarray) -> float:
    denominator = host.sum()
    return 0.0 if denominator == 0 else float(100.0 * np.logical_and(symptom, host).sum() / denominator)


def burden_metrics(predicted: list[float], target: list[float]) -> dict:
    """Burden agreement: MAE / RMSE / bias in percentage points, MAPE in %, R².

    MAPE is undefined for a 0 pp target, so those images are left out of it (`n_mape` = images used).
    R² is 1 - SS_res / SS_tot against the identity line (sklearn `r2_score`), not the squared
    correlation, so a model that is consistently too high or too low is penalised.
    """
    p, t = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if p.size == 0:
        return {}
    error = p - t
    positive = t > 0
    total = float(np.sum((t - t.mean()) ** 2))
    return {
        "n": int(p.size),
        "mae_pp": float(np.abs(error).mean()),
        "rmse_pp": float(np.sqrt(np.mean(error ** 2))),
        "bias_pp": float(error.mean()),
        "mape_pct": float(np.mean(np.abs(error[positive]) / t[positive]) * 100) if positive.any() else None,
        "n_mape": int(positive.sum()),
        "r2": float(1.0 - np.sum(error ** 2) / total) if total > 0 else None,
    }


def iou_np(prediction: np.ndarray, truth: np.ndarray) -> float:
    union = np.logical_or(prediction, truth).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, truth).sum() / union)


def boundary_f1_np(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Phase-9 definition: 8-neighbour boundary, disk tolerance of 0.75% of the image diagonal."""
    from scipy import ndimage

    radius = max(1, int(round(math.hypot(*prediction.shape) * 0.0075)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (yy * yy + xx * xx) <= radius * radius
    eight = np.ones((3, 3), dtype=bool)
    pb = prediction & ~ndimage.binary_erosion(prediction, structure=eight, border_value=0)
    tb = truth & ~ndimage.binary_erosion(truth, structure=eight, border_value=0)
    if pb.sum() == 0 and tb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or tb.sum() == 0:
        return 0.0
    precision = (pb & ndimage.binary_dilation(tb, structure=disk)).sum() / pb.sum()
    recall = (tb & ndimage.binary_dilation(pb, structure=disk)).sum() / tb.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


@torch.no_grad()
def predict_heldout(model, loader, device, amp: bool, keep_maps: int = 0) -> dict:
    """Held-out predictions in the training letterbox space (OOF / validation monitor)."""
    model.eval()
    rows, maps = [], defaultdict(list)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            disease, symptom, host = model(images)
        disease_probability = torch.softmax(disease.float(), 1).cpu().numpy()
        symptom_probability = torch.sigmoid(symptom.float())
        host_probability = torch.sigmoid(host.float())
        for i, sample_id in enumerate(batch["sample_id"]):
            s_prob = symptom_probability[i, 0].cpu().numpy()
            h_prob = host_probability[i, 0].cpu().numpy()
            s_true = batch["symptom"][i, 0].numpy() > 0.5
            h_true = batch["host"][i, 0].numpy() > 0.5
            s_pred, h_pred = s_prob >= 0.5, h_prob >= 0.5
            has_host = float(batch["host_weight"][i]) > 0
            rows.append({
                "sample_id": sample_id,
                "disease_true": int(batch["disease"][i]),
                "disease_pred": int(disease_probability[i].argmax()),
                "disease_prob_true": float(disease_probability[i, int(batch["disease"][i])]),
                "symptom_dice": dice_np(s_pred, s_true),
                "symptom_iou": iou_np(s_pred, s_true),
                "symptom_bf1": boundary_f1_np(s_pred, s_true),
                "symptom_tp": int(np.logical_and(s_pred, s_true).sum()),
                "symptom_pred_pixels": int(s_pred.sum()),
                "symptom_true_pixels": int(s_true.sum()),
                "host_dice": dice_np(h_pred, h_true) if has_host else "",
                "burden_true_pp": burden_np(s_true, h_true) if float(batch["burden_weight"][i]) > 0 else "",
                "burden_pred_pp": burden_np(s_pred, h_pred) if float(batch["burden_weight"][i]) > 0 else "",
            })
            if keep_maps:
                small = (keep_maps, keep_maps)
                maps["symptom_probability"].append(
                    np.asarray(Image.fromarray((s_prob * 255).astype(np.uint8)).resize(small, Image.Resampling.BILINEAR))
                )
                maps["symptom_label"].append(
                    np.asarray(Image.fromarray(s_true.astype(np.uint8) * 255).resize(small, Image.Resampling.NEAREST))
                )
                image = batch["image"][i] * data.IMAGENET_STD + data.IMAGENET_MEAN
                image = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                maps["image"].append(np.asarray(Image.fromarray(image).resize(small, Image.Resampling.BILINEAR)))
                maps["disease_probability"].append(disease_probability[i])
                maps["sample_id"].append(sample_id)
    return {"rows": rows, "maps": maps}


def summarize(rows: list[dict], num_classes: int) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for row in rows:
        confusion[row["disease_true"], row["disease_pred"]] += 1
    f1, per_class = macro_f1(confusion)
    tp = sum(r["symptom_tp"] for r in rows)
    micro = 2 * tp / max(1, sum(r["symptom_pred_pixels"] + r["symptom_true_pixels"] for r in rows))
    host = [r["host_dice"] for r in rows if r["host_dice"] != ""]
    pairs = [(r["burden_pred_pp"], r["burden_true_pp"]) for r in rows if r["burden_true_pp"] != ""]
    burden = burden_metrics([p for p, _ in pairs], [t for _, t in pairs])
    return {
        "records": len(rows),
        "disease_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "disease_macro_f1": f1,
        "disease_per_class_f1": per_class,
        "symptom_mean_dice": float(np.mean([r["symptom_dice"] for r in rows])),
        "symptom_micro_dice": float(micro),
        "symptom_mean_iou": float(np.mean([r["symptom_iou"] for r in rows])),
        "symptom_mean_bf1": float(np.mean([r["symptom_bf1"] for r in rows])),
        "host_mean_dice_pseudo": float(np.mean(host)) if host else None,
        "burden_mae_pp_vs_pseudo": burden.get("mae_pp"),
        "burden_mape_pct_vs_pseudo": burden.get("mape_pct"),
        "burden_r2_vs_pseudo": burden.get("r2"),
        "confusion": confusion.tolist(),
    }


# ----------------------------------------------------------------------------- stages

def source_dir(config: dict) -> Path:
    """One source-stage folder per backbone, so a backbone ablation never loads foreign weights."""
    return Path(config["paths"]["output_root"]) / f"source_{config['model']['backbone'].split('.')[0]}"


def load_config(path: Path) -> dict:
    return config_io.load(Path(path))


def tomato_records(config: dict, use_cache: bool = True, use_review: bool = True) -> list[data.Record]:
    paths = config["paths"]
    review = Path(paths["review_csv"]) if use_review and paths.get("review_csv") else None
    records = data.load_tomato_records(
        Path(paths["tomato_manifest"]), Path(paths["pseudo_host_manifest"]),
        Path(paths["dataset_root"]), Path(paths["input_root"]), review,
    )
    records = [r for r in records if r.split in ("train", "validation")]  # test is never loaded
    if use_cache:
        data.apply_cache(records, Path(paths["cache_root"]) / "cache_index.csv")
    return records


def source_records(config: dict, use_cache: bool = True) -> list[data.Record]:
    paths = config["paths"]
    records = data.load_source_records(Path(paths["pretraining_manifest"]), Path(paths["dataset_root"]))
    if use_cache:
        data.apply_cache(records, Path(paths["cache_root"]) / "cache_index.csv")
    return records


def command_cache(config: dict) -> None:
    # Original labels for every record, plus reviewer-fixed masks under their own cache key.
    records = tomato_records(config, use_cache=False, use_review=False)
    records += [r for r in tomato_records(config, use_cache=False) if r.extra.get("fixed")]
    if config["cotraining"]["enabled"] or config["source_stage"]["enabled"]:
        records += source_records(config, use_cache=False)
    cache = config["cache"]
    data.build_cache(records, Path(config["paths"]["cache_root"]), int(cache["max_side"]), int(cache["workers"]))


def command_folds(config: dict) -> None:
    records = tomato_records(config, use_cache=False, use_review=False)
    cv = config["cross_validation"]
    assignment = data.make_folds(records, int(cv["folds"]), int(cv["seed"]))
    path = Path(config["paths"]["output_root"]) / "folds.csv"
    if path.is_file():
        existing = {row["sample_id"]: int(row["fold"]) for row in data.read_csv(path)}
        if existing != assignment:
            raise RuntimeError(f"{path} already exists with a different split; restore it instead of regenerating")
        print("Folds already exist and match:", path)
        return
    data.write_csv(path, [{"sample_id": k, "fold": v} for k, v in sorted(assignment.items())], ["sample_id", "fold"])
    print(json.dumps(data.summarize_folds(records, assignment), indent=2))
    print("Folds written:", path)


def command_source(config: dict, config_path: Path, resume: bool) -> None:
    stage = config["source_stage"]
    seed = int(config["seed"])
    out = source_dir(config)
    checkpoint_path = out / "source_ema_final.pt"
    if checkpoint_path.is_file():
        validate_completed_source(checkpoint_path, config, seed)
        print("Source stage already complete and provenance matches, skipping:", checkpoint_path)
        return
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"]) and device.type == "cuda"
    seed_everything(seed)
    records = source_records(config)
    validate_source_cohort(records)
    train = [r for r in records if r.split == "train"]
    monitor = [r for r in records if r.split == "validation"]
    classes = sorted({r.class_name for r in train})
    class_to_idx = {name: i for i, name in enumerate(classes)}
    size = int(config["input_size"])
    workers = int(config["num_workers"])
    train_loader = make_loader(
        data.SegmentationDataset(train, class_to_idx, size, True, config["augmentation"]),
        int(stage["batch_size"]), True, workers, seed,
    )
    monitor_loader = make_loader(
        data.SegmentationDataset(monitor, class_to_idx, size, False), int(stage["batch_size"]), False, workers, seed
    )
    model = PlantSegV2(
        num_classes=7, backbone=config["model"]["backbone"], pretrained=bool(config["model"].get("pretrained", True)),
        dropout=float(config["model"]["dropout"]), symptom_guided_pooling=bool(config["model"]["symptom_guided_pooling"]),
        source_classes=len(classes),
    ).to(device)
    out.mkdir(parents=True, exist_ok=True)
    epochs = int(stage["epochs"])
    steps = len(train_loader)
    optimizer = build_optimizer(model, stage)
    scheduler = build_scheduler(optimizer, epochs * steps, int(stage["warmup_epochs"]) * steps, float(stage["min_lr_ratio"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = ModelEma(model, float(config["ema_decay"]))
    class_weights = data.inverse_sqrt_weights(train, class_to_idx).to(device)
    start, history = 1, []
    if resume and (out / "last.pt").is_file():
        state = load_checkpoint(out / "last.pt", {
            "model_state", "ema_state", "optimizer_state", "scheduler_state", "scaler_state", "epoch", "history"
        })
        model.load_state_dict(state["model_state"]); ema.module.load_state_dict(state["ema_state"])
        optimizer.load_state_dict(state["optimizer_state"]); scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"]); start, history = state["epoch"] + 1, state["history"]
    write_json(out / "run_config.json", {"config": config, "provenance": provenance(config_path), "classes": classes})
    step = (start - 1) * steps
    for epoch in range(start, epochs + 1):
        model.train()
        totals, tic = Counter(), time.time()
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                disease, symptom = model.forward_source(batch["image"])
            ce = F.cross_entropy(disease.float(), batch["disease"], weight=class_weights,
                                 label_smoothing=float(config["label_noise"]["label_smoothing"]))
            seg = source_symptom_loss(symptom, batch, config)
            loss = float(stage["disease_weight"]) * ce + seg
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); scheduler.step()
            step += 1
            ema.update(model, step)
            totals["loss"] += float(loss.detach()); totals["ce"] += float(ce.detach()); totals["seg"] += float(seg.detach()); totals["n"] += 1
        metrics = {"epoch": epoch, "lr_head": optimizer.param_groups[2]["lr"], "seconds": round(time.time() - tic, 1),
                   **{k: totals[k] / totals["n"] for k in ("loss", "ce", "seg")}}
        if epoch % int(stage["monitor_every"]) == 0 or epoch == epochs:
            metrics.update(monitor_source(ema.module, monitor_loader, device, amp, len(classes)))
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        torch.save({"model_state": model.state_dict(), "ema_state": ema.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(), "epoch": epoch, "history": history}, out / "last.pt")
    torch.save({"ema_state": ema.module.state_dict(), "epoch": epochs, "classes": classes, "config": config,
                "seed": seed, "created_at_utc": utc_now()},
               out / "source_ema_final.pt")
    data.write_csv(out / "history.csv", history, sorted({k for row in history for k in row}))
    (out / "last.pt").unlink(missing_ok=True)
    print("Source stage complete:", out / "source_ema_final.pt")


@torch.no_grad()
def monitor_source(model, loader, device, amp, num_classes) -> dict:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    dices, tp, pixels = [], 0, 0
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            disease, symptom = model.forward_source(batch["image"].to(device))
        for true, pred in zip(batch["disease"].tolist(), disease.argmax(1).cpu().tolist()):
            if true >= 0:  # a monitor class absent from source-train is skipped
                confusion[true, pred] += 1
        prediction = (torch.sigmoid(symptom.float()) >= 0.5).cpu().numpy()
        truth = batch["symptom"].numpy() > 0.5
        for p, t in zip(prediction, truth):
            dices.append(dice_np(p, t)); tp += int(np.logical_and(p, t).sum()); pixels += int(p.sum() + t.sum())
    return {"monitor_macro_f1": macro_f1(confusion)[0], "monitor_mean_dice": float(np.mean(dices)),
            "monitor_micro_dice": 2 * tp / max(1, pixels)}


def command_target(config: dict, config_path: Path, fold: str, seed: int, resume: bool) -> None:
    run_name = f"{config['run_name']}_seed{seed}"
    out = Path(config["paths"]["output_root"]) / "target" / run_name / f"fold_{fold}"
    checkpoint_path = out / "final_ema.pt"
    if checkpoint_path.is_file():
        validate_completed_target(checkpoint_path, config, seed, fold)
        print("Fold already complete and provenance matches, skipping:", out)
        return
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"]) and device.type == "cuda"
    seed_everything(seed)
    records = tomato_records(config)  # reviewed labels: used for TRAINING only
    original = tomato_records(config, use_review=False)  # original labels: used for SCORING
    train_all = [r for r in records if r.split == "train"]
    classes = sorted({r.class_name for r in original if r.split == "train"})
    class_to_idx = {name: i for i, name in enumerate(classes)}
    if fold == "all":
        train = train_all
        heldout = [r for r in original if r.split == "validation"]
    else:
        assignment = {row["sample_id"]: int(row["fold"]) for row in data.read_csv(Path(config["paths"]["output_root"]) / "folds.csv")}
        train = [r for r in train_all if assignment[r.sample_id] != int(fold)]
        # Held-out images keep their released labels (and excluded images stay in), so a
        # reviewed recipe is never scored on an easier subset than the baseline.
        heldout = [r for r in original if r.split == "train" and assignment[r.sample_id] == int(fold)]
    tcfg = config["target_stage"]
    size = int(config["input_size"])
    workers = int(config["num_workers"])
    train_loader = make_loader(
        data.SegmentationDataset(train, class_to_idx, size, True, config["augmentation"]),
        int(tcfg["batch_size"]), True, workers, seed,
    )
    heldout_loader = make_loader(
        data.SegmentationDataset(heldout, class_to_idx, size, False), int(tcfg["batch_size"]), False, workers, seed
    )
    bn_loader = make_loader(
        data.SegmentationDataset(train, class_to_idx, size, False), int(tcfg["batch_size"]), True, workers, seed + 7
    )
    cotrain = config["cotraining"]
    source_iter = None
    if cotrain["enabled"]:
        all_source_records = source_records(config)
        validate_source_cohort(all_source_records)
        source_train = [r for r in all_source_records if r.split == "train"]
        source_loader = make_loader(
            data.SegmentationDataset(source_train, {}, size, True, config["augmentation"]),
            int(cotrain["batch_size"]), True, max(1, workers // 2) if workers else 0, seed + 1,
        )
        source_iter = data.infinite(source_loader)

    model = PlantSegV2(
        num_classes=len(classes), backbone=config["model"]["backbone"],
        pretrained=bool(config["model"].get("pretrained", True)) and not config["source_stage"]["enabled"],
        dropout=float(config["model"]["dropout"]), symptom_guided_pooling=bool(config["model"]["symptom_guided_pooling"]),
    )
    init_info = {"initialization": "imagenet_timm"}
    source_checkpoint = source_dir(config) / "source_ema_final.pt"
    if config["source_stage"]["enabled"]:
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"Run the source stage first: {source_checkpoint}")
        init_info = load_source_stage(model, str(source_checkpoint))
    model = model.to(device)

    out.mkdir(parents=True, exist_ok=True)
    epochs = int(tcfg["epochs"])
    steps = len(train_loader)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, epochs * steps, int(tcfg["warmup_epochs"]) * steps, float(tcfg["min_lr_ratio"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = ModelEma(model, float(config["ema_decay"]))
    class_weights = data.inverse_sqrt_weights(train, class_to_idx).to(device)
    noise = config["label_noise"]
    start, history = 1, []
    if resume and (out / "last.pt").is_file():
        state = load_checkpoint(out / "last.pt", {
            "model_state", "ema_state", "optimizer_state", "scheduler_state", "scaler_state", "epoch", "history"
        })
        model.load_state_dict(state["model_state"]); ema.module.load_state_dict(state["ema_state"])
        optimizer.load_state_dict(state["optimizer_state"]); scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"]); start, history = state["epoch"] + 1, state["history"]
    write_json(out / "run_config.json", {
        "config": config, "fold": fold, "seed": seed, "classes": classes, "train_records": len(train),
        "heldout_records": len(heldout), "initialization": init_info, "provenance": provenance(config_path),
        "test_records_loaded": 0, "created_at_utc": utc_now(),
    })
    step = (start - 1) * steps
    bootstrap_start = int(round(float(noise["bootstrap_start_fraction"]) * epochs))
    for epoch in range(start, epochs + 1):
        model.train()
        totals, tic = Counter(), time.time()
        beta = float(noise["bootstrap_beta"]) if epoch > bootstrap_start else 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch["image"])
            soft = None
            if beta > 0:
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                    soft = torch.sigmoid(ema.module(batch["image"])[1].float())
            loss, parts = target_losses(outputs, batch, config, class_weights, soft, beta)
            if source_iter is not None:
                source_batch = to_device(next(source_iter), device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    source_symptom = model.forward_symptom(source_batch["image"])
                source_loss = source_symptom_loss(source_symptom, source_batch, config)
                loss = loss + float(cotrain["symptom_weight"]) * source_loss
                parts["source_symptom"] = source_loss.detach()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); scheduler.step()
            step += 1
            ema.update(model, step)
            totals["loss"] += float(loss.detach()); totals["n"] += 1
            for key, value in parts.items():
                totals[key] += float(value)
        metrics = {"epoch": epoch, "lr_head": optimizer.param_groups[2]["lr"], "bootstrap_beta": beta,
                   "seconds": round(time.time() - tic, 1),
                   **{k: totals[k] / totals["n"] for k in totals if k != "n"}}
        if epoch % int(tcfg["monitor_every"]) == 0:
            # Monitoring only: never used to pick a checkpoint (fixed-epoch protocol).
            monitor = summarize(predict_heldout(ema.module, heldout_loader, device, amp)["rows"], len(classes))
            metrics.update({f"heldout_{k}": v for k, v in monitor.items() if not isinstance(v, list)})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        torch.save({"model_state": model.state_dict(), "ema_state": ema.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(), "epoch": epoch, "history": history}, out / "last.pt")

    final = ema.module
    if bool(tcfg["recalibrate_bn"]):
        recalibrate_batchnorm(final, bn_loader, device, int(tcfg["recalibrate_bn_batches"]))
    prediction = predict_heldout(final, heldout_loader, device, amp, keep_maps=int(config["oof_map_size"]))
    summary = summarize(prediction["rows"], len(classes))
    data.write_csv(out / "heldout_per_record.csv", prediction["rows"])
    write_json(out / "heldout_summary.json", {"fold": fold, "seed": seed, "classes": classes, **summary})
    if fold != "all":
        maps = prediction["maps"]
        np.savez_compressed(
            out / "oof_predictions.npz",
            sample_id=np.array(maps["sample_id"]), image=np.stack(maps["image"]),
            symptom_probability=np.stack(maps["symptom_probability"]), symptom_label=np.stack(maps["symptom_label"]),
            disease_probability=np.stack(maps["disease_probability"]),
            disease_label=np.array([row["disease_true"] for row in prediction["rows"]]), classes=np.array(classes),
        )
    torch.save({"ema_state": final.state_dict(), "classes": classes, "config": config, "fold": fold, "seed": seed,
                "epoch": epochs, "created_at_utc": utc_now()}, out / "final_ema.pt")
    (out / "last.pt").unlink(missing_ok=True)
    data.write_csv(out / "history.csv", history, sorted({k for row in history for k in row}))
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion"}, indent=2))


# ----------------------------------------------------------------------------- final check

def to_canvas(logits: np.ndarray, source: dict, target_size: int) -> np.ndarray:
    """Move a map from one letterbox size to another using the exact Phase-9 geometry."""
    content = logits[source["top"]:source["top"] + source["resized_height"],
                     source["left"]:source["left"] + source["resized_width"]]
    goal = data.letterbox_geometry(source["width"], source["height"], target_size)
    resized = F.interpolate(torch.from_numpy(content)[None, None], size=(goal["resized_height"], goal["resized_width"]),
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    canvas = np.full((target_size, target_size), -20.0, dtype=np.float32)
    canvas[goal["top"]:goal["top"] + goal["resized_height"], goal["left"]:goal["left"] + goal["resized_width"]] = resized
    return canvas


@torch.no_grad()
def command_evaluate(config: dict, checkpoint_path: Path, tta: bool, sizes: list[int] | None = None) -> None:
    """85 validation images, original files, metrics in the Phase-9 384 letterbox space.

    `sizes` = inference input sizes (default: the training size). With several sizes the logits are
    averaged on the 384 canvas (multi-scale TTA); choose sizes on train CV, never on these 85 images.
    """
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, {"classes", "ema_state"})
    classes = checkpoint["classes"]
    model = PlantSegV2(num_classes=len(classes), backbone=config["model"]["backbone"], pretrained=False,
                       symptom_guided_pooling=bool(config["model"]["symptom_guided_pooling"]))
    model.load_state_dict(checkpoint["ema_state"])
    model.to(device).eval()
    paths = config["paths"]
    gold = {}
    if paths.get("host_gold_manifest"):
        for row in data.read_csv(Path(paths["host_gold_manifest"])):
            gold[row["sample_id"]] = Path(paths["host_gold_root"]) / row["final_mask_relpath"]
    focal = set(config["focal_classes"])
    sizes, eval_size = sizes or [int(config["input_size"])], 384
    records = [r for r in tomato_records(config, use_cache=False) if r.split == "validation"]
    rows = []
    for record in records:
        image, symptom, _ = data.load_record_arrays(record)
        disease, s_canvas, h_canvas = 0.0, [], []
        for size in sizes:
            geometry = data.letterbox_geometry(*image.size, size)
            boxed = data.letterbox_pil(image, size, Image.Resampling.BILINEAR)
            tensor = data.normalize(torch.from_numpy(np.array(boxed)).permute(2, 0, 1).contiguous())[None].to(device)
            d_logit, s_logit, h_logit = model(tensor)
            if tta:
                d2, s2, h2 = model(torch.flip(tensor, dims=[3]))
                d_logit = (d_logit + d2) / 2
                s_logit = (s_logit + torch.flip(s2, dims=[3])) / 2
                h_logit = (h_logit + torch.flip(h2, dims=[3])) / 2
            disease = disease + d_logit / len(sizes)
            s_canvas.append(to_canvas(s_logit[0, 0].float().cpu().numpy(), geometry, eval_size))
            h_canvas.append(to_canvas(h_logit[0, 0].float().cpu().numpy(), geometry, eval_size))
        s_map = np.mean(s_canvas, axis=0) >= 0.0
        h_map = np.mean(h_canvas, axis=0) >= 0.0
        s_true = np.asarray(data.letterbox_pil(Image.fromarray(symptom * 255), eval_size, Image.Resampling.NEAREST)) != 0
        row = {"sample_id": record.sample_id, "class_name": record.class_name, "focal": record.class_name in focal,
               "disease_true": classes.index(record.class_name), "disease_pred": int(disease.argmax(1)),
               "symptom_dice": dice_np(s_map, s_true), "symptom_iou": iou_np(s_map, s_true),
               "symptom_bf1": boundary_f1_np(s_map, s_true), "symptom_tp": int(np.logical_and(s_map, s_true).sum()),
               "symptom_pred_pixels": int(s_map.sum()), "symptom_true_pixels": int(s_true.sum())}
        if record.sample_id in gold:
            with Image.open(gold[record.sample_id]) as opened:
                h_true = np.asarray(data.letterbox_pil(opened.convert("L"), eval_size, Image.Resampling.NEAREST)) != 0
            gold_burden = burden_np(s_true, h_true)
            e2e_burden, gold_host_burden = burden_np(s_map, h_map), burden_np(s_map, h_true)
            row.update({"host_dice_gold": dice_np(h_map, h_true), "host_iou_gold": iou_np(h_map, h_true),
                        "gold_burden_pp": gold_burden,
                        "e2e_burden_pp": e2e_burden, "gold_host_burden_pred_pp": gold_host_burden,
                        "e2e_abs_error_pp": abs(e2e_burden - gold_burden),
                        "e2e_signed_error_pp": e2e_burden - gold_burden,
                        "gold_host_abs_error_pp": abs(gold_host_burden - gold_burden)})
        rows.append(row)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for row in rows:
        confusion[row["disease_true"], row["disease_pred"]] += 1
    f1, per_class = macro_f1(confusion)

    def mean(key, subset):
        values = [r[key] for r in subset if key in r]
        return float(np.mean(values)) if values else None

    def burden(key, subset):
        scored = [r for r in subset if key in r]
        return burden_metrics([r[key] for r in scored], [r["gold_burden_pp"] for r in scored])

    focal_rows = [r for r in rows if r["focal"]]
    e2e_all, e2e_focal, gold_host = burden("e2e_burden_pp", rows), burden("e2e_burden_pp", focal_rows), burden("gold_host_burden_pred_pp", rows)
    report = {
        "checkpoint": str(checkpoint_path), "tta_hflip": tta, "inference_sizes": sizes, "records": len(rows),
        "evaluation_space": "letterbox_384",
        "disease_accuracy": float(np.trace(confusion) / len(rows)), "disease_macro_f1": f1,
        "disease_per_class_f1": dict(zip(classes, per_class)), "confusion": confusion.tolist(),
        "symptom_mean_dice": mean("symptom_dice", rows),
        "symptom_micro_dice": 2 * sum(r["symptom_tp"] for r in rows)
        / max(1, sum(r["symptom_pred_pixels"] + r["symptom_true_pixels"] for r in rows)),
        "symptom_mean_iou": mean("symptom_iou", rows),
        "symptom_mean_bf1": mean("symptom_bf1", rows),
        "host_mean_dice_gold": mean("host_dice_gold", rows),
        "host_mean_iou_gold": mean("host_iou_gold", rows),
        "e2e_burden_mae_pp_all": mean("e2e_abs_error_pp", rows), "e2e_burden_mae_pp_focal": mean("e2e_abs_error_pp", focal_rows),
        "e2e_burden_bias_pp_all": mean("e2e_signed_error_pp", rows),
        "e2e_burden_mape_pct_all": e2e_all.get("mape_pct"), "e2e_burden_r2_all": e2e_all.get("r2"),
        "e2e_burden_mape_pct_focal": e2e_focal.get("mape_pct"), "e2e_burden_r2_focal": e2e_focal.get("r2"),
        "gold_host_burden_mae_pp_all": mean("gold_host_abs_error_pp", rows),
        "gold_host_burden_mape_pct_all": gold_host.get("mape_pct"), "gold_host_burden_r2_all": gold_host.get("r2"),
        "burden_details": {"e2e_all": e2e_all, "e2e_focal": e2e_focal, "gold_host_all": gold_host},
        "phase3_fp32_reference": {"disease_macro_f1": 0.7406, "symptom_micro_dice": 0.7082, "symptom_mean_dice": 0.6495,
                                  "symptom_mean_iou": 0.5121, "symptom_mean_bf1": 0.4519,
                                  "host_mean_dice_gold": 0.7898, "host_mean_iou_gold": 0.6760,
                                  "e2e_burden_mae_pp_all": 16.70, "e2e_burden_mape_pct_all": 58.87, "e2e_burden_r2_all": 0.4631,
                                  "e2e_burden_mae_pp_focal": 14.47, "e2e_burden_mape_pct_focal": 60.30, "e2e_burden_r2_focal": 0.5057,
                                  "gold_host_burden_mae_pp_all": 14.41, "gold_host_burden_mape_pct_all": 52.50,
                                  "gold_host_burden_r2_all": 0.5900},
    }
    suffix = "" if sizes == [int(config["input_size"])] else "_s" + "+".join(map(str, sizes))
    out = checkpoint_path.parent / f"validation85_report{suffix}{'_tta' if tta else ''}.json"
    write_json(out, report)
    data.write_csv(out.with_suffix(".csv"), rows, sorted({k for r in rows for k in r}))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cache", "folds", "source", "target", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--sizes", default="", help="evaluate only: inference sizes, e.g. 640 or 512,768")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "cache":
        command_cache(config)
    elif args.command == "folds":
        command_folds(config)
    elif args.command == "source":
        command_source(config, args.config.resolve(), args.resume)
    elif args.command == "target":
        if args.fold != "all" and args.fold not in {str(i) for i in range(int(config["cross_validation"]["folds"]))}:
            raise ValueError("--fold must be 0..K-1 or all")
        command_target(config, args.config.resolve(), args.fold, args.seed or int(config["seed"]), args.resume)
    else:
        if args.checkpoint is None:
            raise ValueError("evaluate needs --checkpoint")
        sizes = [int(v) for v in args.sizes.split(",") if v.strip()] or None
        command_evaluate(config, args.checkpoint.resolve(), args.tta, sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
