"""MobileNetV4-Conv-Medium + FPN-lite decoder with disease, symptom and host outputs.

The model keeps the paper output contract while using a compact, ONNX-friendly operator
set so post-training static quantization remains predictable.
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_BACKBONE = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"


def replace_batchnorm_act(module: nn.Module) -> int:
    """Swap timm BatchNormAct2d for nn.BatchNorm2d (+ activation) modules.

    Export is easiest to inspect when normalization layers use plain BatchNorm2d; timm's
    BatchNormAct2d is otherwise traced as a functional batch_norm. Converting before FP32
    training also keeps checkpoint keys stable across environments.
    """
    from timm.layers import BatchNormAct2d

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, BatchNormAct2d):
            bn = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            bn.load_state_dict(child.state_dict(), strict=True)
            layers: list[nn.Module] = [bn]
            if not isinstance(child.drop, nn.Identity):
                layers.append(child.drop)
            if not isinstance(child.act, nn.Identity):
                layers.append(child.act)
            setattr(module, name, bn if len(layers) == 1 else nn.Sequential(*layers))
            replaced += 1
        else:
            replaced += replace_batchnorm_act(child)
    return replaced


def conv_bn_relu(in_channels: int, out_channels: int, kernel: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def depthwise_separable(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        conv_bn_relu(in_channels, in_channels, 3, groups=in_channels),
        conv_bn_relu(in_channels, out_channels, 1),
    )


class FPNLiteDecoder(nn.Module):
    """Top-down pyramid from stride 32 to stride 4 with a global-context branch."""

    def __init__(self, in_channels: list[int], pyramid_channels: int = 96, out_channels: int = 64) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(conv_bn_relu(channels, pyramid_channels) for channels in in_channels)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[-1], pyramid_channels, 1),
            nn.ReLU(inplace=True),
        )
        # Smoothing after the stride-16 and stride-8 merges; stride 4 is handled by `fuse`.
        self.smooth = nn.ModuleList(depthwise_separable(pyramid_channels, pyramid_channels) for _ in range(2))
        self.fuse = depthwise_separable(pyramid_channels, out_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pyramid = self.lateral[-1](features[-1]) + self.context(features[-1])
        for level, smooth in zip((2, 1), self.smooth):
            skip = features[level]
            pyramid = self.lateral[level](skip) + F.interpolate(
                pyramid, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid = smooth(pyramid)
        skip = features[0]
        pyramid = self.lateral[0](skip) + F.interpolate(
            pyramid, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(pyramid)


class MaskHead(nn.Module):
    def __init__(self, in_channels: int, prior_logit: float) -> None:
        super().__init__()
        self.block = depthwise_separable(in_channels, in_channels)
        self.classifier = nn.Conv2d(in_channels, 1, 1)
        nn.init.constant_(self.classifier.bias, prior_logit)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block(features))


class PlantSegV2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.2,
        symptom_guided_pooling: bool = True,
        source_classes: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        replace_batchnorm_act(self.backbone)
        channels = self.backbone.feature_info.channels()
        reductions = self.backbone.feature_info.reduction()
        if reductions != [4, 8, 16, 32]:
            raise RuntimeError(f"Backbone must expose strides 4/8/16/32, got {reductions}")
        self.decoder = FPNLiteDecoder(channels)
        self.symptom_head = MaskHead(64, prior_logit=-1.5)
        self.host_head = MaskHead(64, prior_logit=0.0)
        self.symptom_guided_pooling = symptom_guided_pooling
        pooled = channels[-1] * (2 if symptom_guided_pooling else 1)
        self.disease_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(pooled, num_classes))
        # Only used by the non-tomato source stage; ignored by the target model contract.
        self.source_disease_head = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], source_classes))
            if source_classes
            else None
        )

    def _shared(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        decoded = self.decoder(features)
        return features[-1], self.symptom_head(decoded), decoded

    def _pool(self, top: torch.Tensor, symptom_low: torch.Tensor) -> torch.Tensor:
        pooled = top.mean(dim=(2, 3))
        if not self.symptom_guided_pooling:
            return pooled
        # Average the deepest features inside the predicted lesion area. The mask is
        # detached so the disease loss cannot reshape the symptom prediction.
        weight = F.adaptive_avg_pool2d(torch.sigmoid(symptom_low.detach()), top.shape[-2:])
        lesion = (top * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-6)
        return torch.cat([pooled, lesion], dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = images.shape[-2:]
        top, symptom_low, decoded = self._shared(images)
        host_low = self.host_head(decoded)
        disease = self.disease_head(self._pool(top, symptom_low))
        symptom = F.interpolate(symptom_low, size=size, mode="bilinear", align_corners=False)
        host = F.interpolate(host_low, size=size, mode="bilinear", align_corners=False)
        return disease, symptom, host

    def forward_symptom(self, images: torch.Tensor) -> torch.Tensor:
        """Symptom logits only (co-training batches from non-tomato PlantSeg)."""
        _, symptom_low, _ = self._shared(images)
        return F.interpolate(symptom_low, size=images.shape[-2:], mode="bilinear", align_corners=False)

    def forward_source(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source_disease_head is None:
            raise RuntimeError("Model was built without a source disease head")
        size = images.shape[-2:]
        top, symptom_low, _ = self._shared(images)
        disease = self.source_disease_head(top.mean(dim=(2, 3)))
        return disease, F.interpolate(symptom_low, size=size, mode="bilinear", align_corners=False)


def load_source_stage(model: PlantSegV2, checkpoint_path: str) -> dict:
    """Initialise a target model from the non-tomato source stage (EMA weights).

    Backbone, decoder and symptom head are copied; the host head starts as a copy of
    the symptom head (same policy as Phase 3); the 7-class disease head starts fresh.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("ema_state"), dict):
        raise RuntimeError(f"Invalid source checkpoint schema: {checkpoint_path}")
    state = checkpoint["ema_state"]
    wanted = {
        key: value
        for key, value in state.items()
        if key.startswith(("backbone.", "decoder.", "symptom_head."))
    }
    result = model.load_state_dict(wanted, strict=False)
    unexpected = [key for key in result.unexpected_keys]
    missing = [key for key in result.missing_keys if key.startswith(("backbone.", "decoder.", "symptom_head."))]
    if unexpected or missing:
        raise RuntimeError(f"Source checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.host_head.load_state_dict(model.symptom_head.state_dict())
    nn.init.constant_(model.host_head.classifier.bias, 0.0)
    return {"source_checkpoint": checkpoint_path, "source_epoch": checkpoint.get("epoch")}
