"""Noise-tolerant multitask losses for the FP32 v2 recipe.

Label-noise handling, all switchable from the config:
- boundary band: pixels within `radius` of a mask edge get weight `band_weight` (lesion
  edges are the least reliable part of PlantSeg polygons and of SAM2 pseudo-hosts);
- per-image seg_weight from the train label review (keep / downweight / exclude_seg);
- label smoothing on the disease head;
- optional soft bootstrapping of symptom targets toward the EMA model after a warm-up
  (a simplified form of early-learning correction, cf. ADELE, CVPR 2022).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(mask)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def pixel_weights(mask: torch.Tensor, radius: int, band_weight: float) -> torch.Tensor:
    return 1.0 - boundary_band(mask, radius) * (1.0 - band_weight)


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    bce_fraction: float,
    dice_fraction: float,
) -> torch.Tensor:
    """Per-sample weighted BCE + soft Dice; +1 smoothing keeps lesion-free crops finite."""
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    bce = (bce * weights).sum(dims) / weights.sum(dims).clamp_min(1.0)
    probability = torch.sigmoid(logits.float()) * weights
    weighted_target = target * weights
    intersection = (probability * weighted_target).sum(dims)
    denominator = probability.sum(dims) + weighted_target.sum(dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce_fraction * bce + dice_fraction * dice


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum()
    if float(total.detach()) <= 0:
        return values.sum() * 0.0
    return (values * weights).sum() / total


def soft_burden(symptom_logits: torch.Tensor, host_logits: torch.Tensor) -> torch.Tensor:
    symptom = torch.sigmoid(symptom_logits.float())
    host = torch.sigmoid(host_logits.float())
    return (symptom * host).sum((1, 2, 3)) / (host.sum((1, 2, 3)) + 1e-6)


def target_losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict,
    config: dict,
    class_weights: torch.Tensor,
    bootstrap_probability: torch.Tensor | None = None,
    bootstrap_beta: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    loss_cfg = config["loss"]
    noise = config["label_noise"]
    disease_logits, symptom_logits, host_logits = outputs

    ce = F.cross_entropy(
        disease_logits.float(), batch["disease"], weight=class_weights,
        label_smoothing=float(noise["label_smoothing"]), reduction="none",
    )
    disease = weighted_mean(ce, batch["disease_weight"])

    symptom_target = batch["symptom"]
    if bootstrap_probability is not None and bootstrap_beta > 0:
        symptom_target = (1.0 - bootstrap_beta) * symptom_target + bootstrap_beta * bootstrap_probability
    symptom_pixels = pixel_weights(batch["symptom"], int(noise["boundary_radius"]), float(noise["boundary_band_weight"]))
    symptom_each = segmentation_loss(
        symptom_logits, symptom_target, symptom_pixels,
        float(loss_cfg["bce_fraction"]), float(loss_cfg["dice_fraction"]),
    )
    symptom = weighted_mean(symptom_each, batch["seg_weight"])

    host_pixels = pixel_weights(batch["host"], int(noise["host_boundary_radius"]), float(noise["boundary_band_weight"]))
    host_each = segmentation_loss(
        host_logits, batch["host"], host_pixels,
        float(loss_cfg["bce_fraction"]), float(loss_cfg["dice_fraction"]),
    )
    host = weighted_mean(host_each, batch["host_weight"])

    predicted_burden = soft_burden(symptom_logits, host_logits)
    burden_each = F.smooth_l1_loss(predicted_burden, batch["burden_target"], reduction="none")
    burden = weighted_mean(burden_each, batch["burden_weight"] * batch["seg_weight"])

    symptom_probability = torch.sigmoid(symptom_logits.float())
    host_probability = torch.sigmoid(host_logits.float())
    outside = (symptom_probability * (1.0 - host_probability)).sum((1, 2, 3)) / (
        symptom_probability.sum((1, 2, 3)) + 1e-6
    )
    containment = weighted_mean(outside, batch["host_weight"])

    total = (
        float(loss_cfg["disease_weight"]) * disease
        + float(loss_cfg["symptom_weight"]) * symptom
        + float(loss_cfg["host_weight"]) * host
        + float(loss_cfg["burden_weight"]) * burden
        + float(loss_cfg["containment_weight"]) * containment
    )
    return total, {
        "disease": disease.detach(), "symptom": symptom.detach(), "host": host.detach(),
        "burden": burden.detach(), "containment": containment.detach(),
    }


def source_symptom_loss(symptom_logits: torch.Tensor, batch: dict, config: dict) -> torch.Tensor:
    noise = config["label_noise"]
    weights = pixel_weights(batch["symptom"], int(noise["boundary_radius"]), float(noise["boundary_band_weight"]))
    return segmentation_loss(
        symptom_logits, batch["symptom"], weights,
        float(config["loss"]["bce_fraction"]), float(config["loss"]["dice_fraction"]),
    ).mean()
