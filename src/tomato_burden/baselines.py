#!/usr/bin/env python3
"""Paper baselines trained and scored with the same target-stage protocol as MNv4-FPN.

Every network goes through the shared target stage and validation scorer; only the network is swapped. Recipe for
all five: ImageNet initialisation, no non-tomato source stage, no co-training, reviewed train labels,
frozen pseudo-host masks, the paper loss and label-noise settings, AdamW (backbone lr 2e-4, head
lr 6e-4), 5 warm-up epochs, cosine to 1%, EMA 0.999, 120 epochs, batch 8, train 512, evaluate 640.

Networks output two logit maps (symptom, host) at input resolution. Their last layer starts from the
same prior as our heads (symptom bias -1.5, host bias 0). The disease output is a constant zero tensor
(the disease loss weight is 0 for every model).

  preflight  build every network with its ImageNet weights, one training step and one 640 forward pass
  run        --arch A --seed S: train on the 595 reviewed images, score the 85 validation images
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import data
from . import training as base
from .losses import target_losses
from .models import PlantSegV2

HERE = Path(__file__).resolve().parent
ARCHS = ("unet_r34", "deeplabv3p_r50", "segformer_b0", "lraspp_mnv3", "ours_imagenet")
SEEDS = (20260924, 20260925, 20260926)
PRIOR = (-1.5, 0.0)  # symptom, host: the initial biases of MaskHead in models.py
SMP_VERSION = "0.5.0"


def set_prior(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.bias.copy_(torch.tensor(PRIOR, dtype=conv.bias.dtype))


class SMPNet(nn.Module):
    """segmentation_models_pytorch model split into backbone (encoder) and head parts for the optimizer groups."""

    def __init__(self, net: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone, self.decoder, self.segmentation_head = net.encoder, net.decoder, net.segmentation_head
        self.num_classes = num_classes
        set_prior(self.segmentation_head[0])

    def forward(self, images: torch.Tensor):
        logits = self.segmentation_head(self.decoder(self.backbone(images)))
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return images.new_zeros(images.shape[0], self.num_classes), logits[:, 0:1], logits[:, 1:2]


class LRASPPNet(nn.Module):
    """torchvision LR-ASPP MobileNetV3-Large (the Phase 3 architecture) with the same two-map contract."""

    def __init__(self, net: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone, self.classifier = net.backbone, net.classifier
        self.num_classes = num_classes
        set_prior(self.classifier.high_classifier)
        with torch.no_grad():
            self.classifier.low_classifier.bias.zero_()

    def forward(self, images: torch.Tensor):
        logits = self.classifier(self.backbone(images))
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return images.new_zeros(images.shape[0], self.num_classes), logits[:, 0:1], logits[:, 1:2]


def build_network(arch: str, num_classes: int, pretrained: bool, ours_kwargs: dict | None = None) -> nn.Module:
    if arch == "ours_imagenet":
        kwargs = dict(ours_kwargs or {})
        kwargs.update(num_classes=num_classes, pretrained=pretrained)
        return PlantSegV2(**kwargs)
    if arch == "lraspp_mnv3":
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        return LRASPPNet(lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=2), num_classes)
    import segmentation_models_pytorch as smp

    if smp.__version__ != SMP_VERSION:
        raise RuntimeError(f"segmentation_models_pytorch {smp.__version__}, protocol uses {SMP_VERSION}")
    builder, encoder = {"unet_r34": (smp.Unet, "resnet34"), "deeplabv3p_r50": (smp.DeepLabV3Plus, "resnet50"),
                        "segformer_b0": (smp.Segformer, "mit_b0")}[arch]
    net = builder(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=2)
    return SMPNet(net, num_classes)


def recipe(session: dict, arch: str, seed: int) -> dict:
    """The session config (paths, review CSV) with the fixed P2 recipe on top."""
    if arch not in ARCHS or seed not in SEEDS:
        raise ValueError(f"arch must be one of {ARCHS}, seed one of {SEEDS}")
    config = copy.deepcopy(session)
    config["run_name"] = f"p2_{arch}"
    config["seed"] = seed
    config["model"]["arch"] = arch
    config["model"]["pretrained"] = True
    config["source_stage"]["enabled"] = False
    config["cotraining"]["enabled"] = False
    config["loss"]["disease_weight"] = 0.0
    config["target_stage"]["recalibrate_bn"] = False
    if not config["paths"].get("review_csv"):
        raise ValueError("P2 trains on the reviewed labels: review_csv is empty in the session config")
    if not config["paths"].get("pseudo_host_manifest"):
        raise ValueError("P2 requires the frozen pseudo-host manifest used by the no-host recipe")
    return config


def attested_recipe(session: dict, arch: str, seed: int, records: list, host_manifest_sha256: str) -> dict:
    config = recipe(session, arch, seed)
    config["run_metadata"] = {
        "variant": "baseline",
        "cohort": base.validate_target_cohort(records, "baseline"),
        "host_manifest_sha256": host_manifest_sha256,
    }
    return config


def install_factory(arch: str) -> None:
    """Point the shared training pipeline at the selected architecture."""

    def factory(num_classes, backbone=None, pretrained=True, dropout=0.2, symptom_guided_pooling=True, **extra):
        ours = {"backbone": backbone, "dropout": dropout, "symptom_guided_pooling": symptom_guided_pooling, **extra}
        return build_network(arch, num_classes, pretrained, ours)

    base.PlantSegV2 = factory


def versions() -> dict:
    import platform

    import timm
    import torchvision

    out = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "timm": timm.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import segmentation_models_pytorch as smp
        out["segmentation_models_pytorch"] = smp.__version__
    except ImportError:
        out["segmentation_models_pytorch"] = None
    return out


def command_preflight(session: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    config = recipe(session, ARCHS[0], SEEDS[0])
    records = [r for r in base.tomato_records(config) if r.split == "train"]
    classes = sorted({r.class_name for r in records})
    class_to_idx = {name: i for i, name in enumerate(classes)}
    dataset = data.SegmentationDataset(records[:4], class_to_idx, int(config["input_size"]), True, config["augmentation"])
    batch = base.to_device(next(iter(base.make_loader(dataset, 4, True, 0, 1))), device)
    weights = torch.ones(len(classes), device=device)
    report = {}
    for arch in ARCHS:
        tic = time.time()
        model = build_network(arch, len(classes), True, {"backbone": config["model"]["backbone"],
                                                         "dropout": config["model"]["dropout"],
                                                         "symptom_guided_pooling": config["model"]["symptom_guided_pooling"]})
        model = model.to(device).train()
        optimizer = base.build_optimizer(model, recipe(session, arch, SEEDS[0])["target_stage"])
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(batch["image"])
        loss, parts = target_losses(outputs, batch, config, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        groups = [len(g["params"]) for g in optimizer.param_groups]
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
            _, symptom, host = model(torch.zeros(1, 3, 640, 640, device=device))
        ok = (finite and groups[0] > 0 and groups[2] > 0 and tuple(symptom.shape) == (1, 1, 640, 640)
              and tuple(host.shape) == (1, 1, 640, 640) and bool(torch.isfinite(loss)))
        report[arch] = {"ok": ok, "loss": round(float(loss.detach()), 4), "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                        "param_groups": groups, "seconds": round(time.time() - tic, 1)}
        print(arch, json.dumps(report[arch]), flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"versions": versions(), "status": "PREFLIGHT_PASS" if all(r["ok"] for r in report.values()) else "PREFLIGHT_FAIL"}, indent=1))
    if not all(r["ok"] for r in report.values()):
        raise RuntimeError("Baseline preflight failed")


def command_run(session: dict, arch: str, seed: int) -> None:
    initial = recipe(session, arch, seed)
    host_manifest = Path(initial["paths"]["pseudo_host_manifest"])
    records = base.tomato_records(initial, use_cache=False)
    config = attested_recipe(session, arch, seed, records, base.sha256_file(host_manifest))
    out = Path(config["paths"]["output_root"]) / "target" / f"{config['run_name']}_seed{seed}" / "fold_all"
    config_path = Path(config["paths"]["output_root"]) / "p2_configs" / f"{arch}_s{seed}.json"
    base.write_json(config_path, config)
    install_factory(arch)
    tic = time.time()
    base.command_target(config, config_path, "all", seed, resume=True)
    checkpoint = out / "final_ema.pt"
    base.command_evaluate(config, checkpoint, False, None)
    base.command_evaluate(config, checkpoint, False, [640])
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("ema_state"), dict):
        raise RuntimeError(f"Invalid baseline checkpoint schema: {checkpoint}")
    state = loaded["ema_state"]
    info = {"arch": arch, "seed": seed, "protocol": "TOMATO-BURDEN-PTQ-20261015-v1", "parameters": int(sum(v.numel() for k, v in state.items()
                                                                                               if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))),
            "train_baseline_sha256": base.sha256_file(Path(__file__).resolve()), "versions": versions(),
            "session_minutes": round((time.time() - tic) / 60, 1), "created_at_utc": base.utc_now()}
    base.write_json(out / "p2_run_info.json", info)
    print(json.dumps(info, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True, help="session config (paths) written by the notebook")
    parser.add_argument("--arch", choices=ARCHS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    session = base.load_config(args.config)
    if args.command == "preflight":
        command_preflight(session)
    else:
        if args.arch is None or args.seed is None:
            raise ValueError("run needs --arch and --seed")
        command_run(session, args.arch, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
