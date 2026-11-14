#!/usr/bin/env python3
"""Paper evaluation: ONNX export, static PTQ selection, INT8 scoring, lock, and analysis.

  export     FP32 ONNX of every trained model (640, opset 17) + parity vs PyTorch
  p3         ORT static PTQ with MinMax / Percentile 99.999 / Entropy on the six seed-20260924 models, scored on val
  select     pick the P3 method (pooled smallest Dice drop; < 0.002 apart -> smaller gold-host drift)
  int8       quantize the 18 P2 models with the selected method; list operators left in floating point
  validate   score every model on the 85 validation images (FP32 first, INT8 with drift against its FP32 model)
  bench      size of every model; latency of the seed-20260924 model of each network (FP32 and INT8)
  lock       pretest_lock.json: hashes of every model, the code, the protocols and the validation tables
  test       P4, once: verifies the lock, writes the test-attempt marker, scores the 176 test images, P1/P2 analyses
  report     markdown table (mean ± sd over seeds) of the validation or test results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import baselines as tb
from . import data, evaluation, runtime, statistics
from .models import PlantSegV2
from .training import load_config, sha256_file, utc_now, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "configs" / "protocol_final.json"
PROTOCOL_DATA = json.loads(PROTOCOL.read_text(encoding="utf-8"))
SIZE = int(PROTOCOL_DATA["training"]["evaluation_size"])
OPSET = int(PROTOCOL_DATA["ptq"]["opset"])
CALIB_COUNT = int(PROTOCOL_DATA["ptq"]["calibration_images"])
CALIB_SEED = int(PROTOCOL_DATA["ptq"]["calibration_seed"])
CALIB_CHUNK = int(PROTOCOL_DATA["ptq"]["calibration_chunk"])
METHODS = {"minmax": ("MinMax", {}), "percentile": ("Percentile", {"CalibPercentile": 99.999}), "entropy": ("Entropy", {})}
SEEDS = {20260924: "s24", 20260925: "s25", 20260926: "s26"}
BOOT_RESAMPLES = int(PROTOCOL_DATA["evaluation"]["bootstrap_resamples"])
BOOT_SEED = int(PROTOCOL_DATA["evaluation"]["bootstrap_seed"])
CODE_FILES = (
    "paper_pipeline.py", "baselines.py", "training.py", "data.py", "losses.py", "models.py",
    "evaluation.py", "runtime.py", "statistics.py", "config.py", "artifacts.py", "__init__.py",
)


# ----------------------------------------------------------------------------- registry

def registry() -> dict:
    """Every model of the paper: label -> role, network, seed, format."""
    models = {}
    for seed, tag in SEEDS.items():
        models[f"p1_nohost_{tag}"] = {"role": "P1 no_host = P2 ours", "net": "ours", "seed": seed, "fmt": "fp32"}
        models[f"p1_host_{tag}"] = {"role": "P1 host_v1", "net": "ours", "seed": seed, "fmt": "fp32"}
        for arch in tb.ARCHS:
            models[f"p2_{arch}_{tag}"] = {"role": "P2 baseline", "net": arch, "seed": seed, "fmt": "fp32"}
    for label, seed in (("supp_luot8_s24_py312", 20260924), ("supp_nohost_s26_py312", 20260926),
                        ("supp_nohost_s26_rerun_py312", 20260926)):
        models[label] = {"role": "supplementary (old image)", "net": "ours", "seed": seed, "fmt": "fp32"}
    for seed, tag in SEEDS.items():
        for base in [f"p1_nohost_{tag}"] + [f"p2_{arch}_{tag}" for arch in tb.ARCHS]:
            models[f"{base}__int8"] = {"role": "P2 INT8", "net": models[base]["net"], "seed": seed, "fmt": "int8", "fp32_ref": base}
    return models


P3_LABELS = ["p1_nohost_s24"] + [f"p2_{arch}_s24" for arch in tb.ARCHS]


def _source_experiment(label: str) -> str:
    if label.startswith("p1_host_"):
        return "manual-host"
    if label.startswith("p1_nohost_"):
        return "standard"
    if label.startswith("p2_"):
        return "baseline"
    return "supplementary"


def load_model_sources(source_registry: Path, root: Path | None = None) -> dict:
    """Validate and resolve the exact checkpoint provenance used by the paper.

    A plain label-to-path mapping cannot distinguish the corrected manual-host
    run from an older checkpoint trained with the standard host manifest.  The
    structured registry therefore locks both the bytes and the experiment/data
    identity before any ONNX export is allowed.
    """
    source_registry = Path(source_registry)
    root = Path(root) if root is not None else ROOT
    sources = json.loads(source_registry.read_text(encoding="utf-8"))
    expected = {label: info for label, info in registry().items() if info["fmt"] == "fp32"}
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise RuntimeError("Model source registry does not exactly match the 24 FP32 exports")

    resolved = {}
    for label, model in expected.items():
        entry = sources[label]
        if not isinstance(entry, dict):
            raise RuntimeError(f"Model source {label} must be a structured provenance object, not a path string")
        required = {"checkpoint", "sha256", "seed", "network", "experiment", "source_run"}
        missing = required - entry.keys()
        if missing:
            raise RuntimeError(f"Model source {label} is missing fields: {sorted(missing)}")

        experiment = _source_experiment(label)
        identity = {
            "seed": int(model["seed"]),
            "network": model["net"],
            "experiment": experiment,
        }
        for key, expected_value in identity.items():
            if entry[key] != expected_value:
                raise RuntimeError(
                    f"Model source {label} has {key}={entry[key]!r}, expected {expected_value!r}"
                )

        source_run = entry["source_run"]
        if not isinstance(source_run, dict) or not {"platform", "id", "version"} <= source_run.keys():
            raise RuntimeError(f"Model source {label} needs source_run.platform, source_run.id, and source_run.version")
        platform = str(source_run["platform"]).strip()
        run_id = str(source_run["id"]).strip()
        if not platform or not run_id:
            raise RuntimeError(f"Model source {label} has an empty source_run identity")
        if platform.upper().startswith("REPLACE_") or run_id.upper().startswith("REPLACE_"):
            raise RuntimeError(f"Model source {label} still has a source_run placeholder")
        if not isinstance(source_run["version"], int) or source_run["version"] < 1:
            raise RuntimeError(f"Model source {label} has an invalid source_run.version")

        if experiment != "supplementary":
            attestation = entry.get("data_attestation")
            if not isinstance(attestation, dict):
                raise RuntimeError(f"Model source {label} needs a data_attestation")
            expected_host = 595 if experiment == "manual-host" else 199
            expected_manual = 396 if experiment == "manual-host" else 0
            expected_counts = {
                "train_records": int(PROTOCOL_DATA["cohort"]["train"]),
                "host_supervised_records": expected_host,
                "manual_host_masks": expected_manual,
            }
            for key, expected_value in expected_counts.items():
                if attestation.get(key) != expected_value:
                    raise RuntimeError(
                        f"Model source {label} has data_attestation.{key}={attestation.get(key)!r}, "
                        f"expected {expected_value!r}"
                    )
            manifest_hash = str(attestation.get("host_manifest_sha256", "")).upper()
            if len(manifest_hash) != 64 or any(char not in "0123456789ABCDEF" for char in manifest_hash):
                raise RuntimeError(f"Model source {label} has an invalid host_manifest_sha256")

        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if not checkpoint.is_file():
            raise RuntimeError(f"Source checkpoint is missing for {label}: {checkpoint}")
        expected_hash = str(entry["sha256"]).upper()
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for source checkpoint {label}")
        resolved[label] = {**entry, "sha256": expected_hash, "checkpoint": checkpoint.resolve()}
    return resolved


def model_path(bundle: Path, label: str, method: str | None = None) -> Path:
    if method:
        return bundle / "p3" / method / f"{label}.onnx"
    fmt = registry()[label]["fmt"]
    return bundle / fmt / f"{label}.onnx"


def results_dir(bundle: Path) -> Path:
    return bundle / "results"


# ----------------------------------------------------------------------------- data

class Session:
    """Session config + records, with the decoded images cached across models."""

    def __init__(self, config_path: Path, cache_images: bool = False) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_config(config_path)
        self.focal = set(self.config["focal_classes"])
        self.gold = evaluation.gold_hosts(self.config)
        self._cache: dict[str, tuple] = {}
        original = data.load_record_arrays

        def cached(record):
            key = (record.sample_id, str(record.image), str(record.symptom))
            if key not in self._cache:
                self._cache[key] = original(record)
            return self._cache[key]

        if cache_images:  # saves decoding time when many models are scored; costs a few GB of RAM
            data.load_record_arrays = cached

    def validation(self) -> list:
        records = evaluation.validation_records(self.config, int(PROTOCOL_DATA["cohort"]["validation"]))
        self._check_gold_and_focal(records, "validation")
        return records

    def test(self) -> list:
        records = evaluation.test_records(self.config, int(PROTOCOL_DATA["cohort"]["test"]))
        self._check_gold_and_focal(records, "test")
        return records

    def _check_gold_and_focal(self, records: list, split: str) -> None:
        missing = sorted(row.sample_id for row in records if row.sample_id not in self.gold)
        if missing:
            raise RuntimeError(f"Gold host masks missing for {split}: {missing[:6]}")
        expected = int(PROTOCOL_DATA["cohort"][f"focal_{split}"])
        actual = sum(row.class_name in self.focal for row in records)
        if actual != expected:
            raise RuntimeError(f"Expected {expected} focal {split} records, found {actual}")

    def calibration(self) -> list:
        records = evaluation.train_records(self.config, expected=595)
        return evaluation.stratified(records, CALIB_COUNT, CALIB_SEED)


# ----------------------------------------------------------------------------- export

class TwoMaps(nn.Module):
    """P2 network -> (symptom_logits, host_logits) at input resolution."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor):
        _, symptom, host = self.net(images)
        return symptom, host


def load_network(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must be a mapping: {checkpoint_path}")
    missing = {"config", "classes", "ema_state"} - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Checkpoint is missing required fields {sorted(missing)}: {checkpoint_path}")
    if not isinstance(checkpoint["ema_state"], dict) or not checkpoint["ema_state"]:
        raise RuntimeError(f"Checkpoint has an invalid ema_state: {checkpoint_path}")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise RuntimeError(f"Checkpoint has an invalid config: {checkpoint_path}")
    arch = config["model"].get("arch", "ours")
    classes = len(checkpoint["classes"])
    ours = {"backbone": config["model"]["backbone"], "symptom_guided_pooling": bool(config["model"]["symptom_guided_pooling"])}
    if arch in ("ours", "ours_imagenet"):
        full = PlantSegV2(num_classes=classes, pretrained=False, **ours)
        full.load_state_dict(checkpoint["ema_state"])
        wrapped = evaluation.SegmentationOnly(full)
    else:
        net = tb.build_network(arch, classes, False, ours)
        net.load_state_dict(checkpoint["ema_state"])
        wrapped = TwoMaps(net)
    parameters = int(sum(v.numel() for k, v in checkpoint["ema_state"].items()
                         if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))))
    return wrapped.eval(), {
        "arch": arch,
        "parameters": parameters,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "host_manifest": config.get("paths", {}).get("pseudo_host_manifest"),
        "run_metadata": config.get("run_metadata"),
    }


def validate_checkpoint_identity(label: str, source: dict, checkpoint_meta: dict) -> None:
    if checkpoint_meta.get("seed") != source["seed"]:
        raise RuntimeError(
            f"Checkpoint seed mismatch for {label}: {checkpoint_meta.get('seed')!r} != {source['seed']!r}"
        )
    if checkpoint_meta.get("arch") != source["network"]:
        raise RuntimeError(
            f"Checkpoint network mismatch for {label}: {checkpoint_meta.get('arch')!r} != {source['network']!r}"
        )
    experiment = source.get("experiment")
    if experiment == "supplementary":
        return
    attestation = source["data_attestation"]
    run_metadata = checkpoint_meta.get("run_metadata")
    if isinstance(run_metadata, dict):
        expected = {
            "variant": experiment,
            "host_manifest_sha256": attestation["host_manifest_sha256"],
        }
        for key, expected_value in expected.items():
            if run_metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"Checkpoint {key} mismatch for {label}: {run_metadata.get(key)!r} != {expected_value!r}"
                )
        cohort = run_metadata.get("cohort")
        if not isinstance(cohort, dict):
            raise RuntimeError(f"Checkpoint run_metadata.cohort is missing for {label}")
        for key in ("train_records", "host_supervised_records", "manual_host_masks"):
            if cohort.get(key) != attestation[key]:
                raise RuntimeError(
                    f"Checkpoint cohort.{key} mismatch for {label}: {cohort.get(key)!r} != {attestation[key]!r}"
                )
        return

    # Historical checkpoints predate embedded run_metadata. Their immutable
    # config still records the manifest path. The final manual-host runs used
    # host_train_v*/pseudo_host_manifest.csv; the superseded run did not.
    manifest = str(checkpoint_meta.get("host_manifest") or "").replace("\\", "/").lower()
    manual_marker = "/host_train_v" in manifest or "manual_host" in Path(manifest).name
    if experiment == "manual-host" and not manual_marker:
        raise RuntimeError(
            f"Checkpoint manual-host provenance mismatch for {label}: embedded manifest is {manifest!r}"
        )
    if experiment in ("standard", "baseline") and manual_marker:
        raise RuntimeError(
            f"Checkpoint standard-host provenance mismatch for {label}: embedded manifest is {manifest!r}"
        )


def validate_export_cache(label: str, target: Path, meta_path: Path, source: dict) -> dict:
    if not target.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Incomplete export cache for {label}; expected both ONNX and metadata")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "label": label,
        "onnx_sha256": sha256_file(target),
        "seed": source["seed"],
        "arch": source["network"],
    }
    mismatched = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
    recorded_source = meta.get("source_provenance")
    if not isinstance(recorded_source, dict) or recorded_source.get("sha256") != source["sha256"]:
        mismatched["source checkpoint"] = (
            recorded_source.get("sha256") if isinstance(recorded_source, dict) else None,
            source["sha256"],
        )
    if mismatched:
        raise RuntimeError(f"Stale or inconsistent export cache for {label}: {mismatched}")
    return meta


def command_export(args, session: Session) -> None:
    if not args.sources:
        raise RuntimeError("--sources is required for checkpoint provenance validation")
    sources = load_model_sources(Path(args.sources))
    bundle = Path(args.bundle)
    probe = [evaluation.letterbox_input(data.load_record_arrays(r)[0], SIZE)[0] for r in session.validation()[:2]]
    for label, info in registry().items():
        if info["fmt"] == "int8" or (args.only and label not in args.only):
            continue
        target = model_path(bundle, label)
        meta_path = bundle / "meta" / f"{label}.json"
        source_entry = sources[label]
        if target.exists() or meta_path.exists():
            validate_export_cache(label, target, meta_path, source_entry)
            continue
        source = source_entry["checkpoint"]
        target.parent.mkdir(parents=True, exist_ok=True)
        model, meta = load_network(source)
        validate_checkpoint_identity(label, source_entry, meta)
        runtime.export_onnx(model, target, SIZE, OPSET)
        onnx_session = runtime.ort_session(target, exact=False)
        worst = 0.0
        with torch.no_grad():
            for x in probe:
                reference = [tensor.numpy() for tensor in model(torch.from_numpy(x))]
                produced = onnx_session.run(None, {"image": x})
                worst = max(worst, *(float(np.abs(a - b).max()) for a, b in zip(reference, produced)))
        provenance = {**source_entry, "checkpoint": str(source)}
        meta.update({"label": label, "source": str(source), "source_provenance": provenance,
                     "parity_max_abs_logit": worst})
        meta.update({"onnx_sha256": sha256_file(target), "onnx_bytes": target.stat().st_size, **info})
        write_json(meta_path, meta)
        print(f"[export] {label}: {target.stat().st_size / 1e6:.1f} MB | parity {meta.get('parity_max_abs_logit', '-')}", flush=True)


# ----------------------------------------------------------------------------- PTQ

class CalibrationReader:
    """ORT CalibrationDataReader over pre-letterboxed images, with the range interface used by CalibStridedMinMax."""

    def __init__(self, inputs: list[np.ndarray]) -> None:
        self.inputs, self.start, self.stop, self.position = inputs, 0, len(inputs), 0

    def __len__(self) -> int:
        return len(self.inputs)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.start, self.stop, self.position = start_index, min(end_index, len(self.inputs)), start_index

    def get_next(self):
        if self.position >= self.stop:
            return None
        item = {"image": self.inputs[self.position]}
        self.position += 1
        return item

    def rewind(self) -> None:
        self.position = self.start


def quantize(fp32: Path, target: Path, method: str, inputs: list[np.ndarray]) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    name, extra = METHODS[method]
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared = target.with_suffix(".pre.onnx")
    tic = time.time()
    try:
        quant_pre_process(str(fp32), str(prepared))
        pre = "quant_pre_process"
    except Exception as error:  # noqa: BLE001 - symbolic shape inference fails on a few graphs
        quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
        pre = f"quant_pre_process(skip_symbolic_shape) after: {str(error)[:120]}"
    quantize_static(str(prepared), str(target), CalibrationReader(inputs), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=getattr(CalibrationMethod, name),
                    extra_options={"CalibStridedMinMax": CALIB_CHUNK, **extra})
    prepared.unlink(missing_ok=True)
    fused = runtime.fused_op_counts(target)
    floating = {op: n for op, n in fused.items()
                if not op.startswith("QLinear") and op not in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")}
    return {"method": method, "ort_method": name, "options": extra, "chunk": CALIB_CHUNK, "preprocess": pre,
            "seconds": round(time.time() - tic, 1), "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "fused_ops": fused, "floating_point_ops": floating, "QLinearConv": fused.get("QLinearConv", 0)}


def calibration_inputs(session: Session) -> tuple[list[np.ndarray], list[str]]:
    records = session.calibration()
    return [evaluation.letterbox_input(data.load_record_arrays(r)[0], SIZE)[0] for r in records], [r.sample_id for r in records]


# ----------------------------------------------------------------------------- scoring

def sha256_json(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def record_ids_sha256(records: list) -> str:
    return sha256_json([row.sample_id for row in records])


def validate_score_cache(folder: Path, name: str, path: Path, records: list, reference: dict | None) -> dict:
    summary_path = folder / f"{name}_summary.json"
    rows_path = folder / f"{name}_rows.json"
    csv_path = folder / f"{name}_rows.csv"
    maps_path = folder / f"{name}_maps.npz"
    missing = [p.name for p in (summary_path, rows_path, csv_path, maps_path) if not p.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete score cache for {name}: missing {missing}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ids = [row.sample_id for row in records]
    expected = {
        "model_sha256": sha256_file(path),
        "record_ids_sha256": record_ids_sha256(records),
        "records": len(records),
        "scoring_code_sha256": sha256_json(code_hashes()),
        "reference_model_sha256": reference.get("model_sha256") if reference else None,
    }
    mismatched = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    row_ids = [row.get("sample_id") for row in rows]
    if row_ids != expected_ids or len(set(row_ids)) != len(row_ids):
        mismatched["row_sample_ids"] = (row_ids[:3], expected_ids[:3])
    with np.load(maps_path, allow_pickle=False) as archive:
        map_ids = [str(item) for item in archive["sample_id"]]
    if map_ids != sorted(expected_ids):
        mismatched["map_sample_ids"] = (map_ids[:3], sorted(expected_ids)[:3])
    if mismatched:
        raise RuntimeError(f"Stale or inconsistent score cache for {name}: {mismatched}")
    return summary


def score(session: Session, path: Path, records: list, folder: Path, name: str, reference: dict | None) -> dict:
    done = folder / f"{name}_summary.json"
    if done.is_file():
        return validate_score_cache(folder, name, path, records, reference)
    rows, maps = evaluation.score_records(runtime.ort_session(path), records, session.gold, session.focal, SIZE, reference)
    summary = evaluation.summarize_rows(rows)
    summary.update({
        "label": name,
        "model_sha256": sha256_file(path),
        "records": len(rows),
        "record_ids_sha256": record_ids_sha256(records),
        "scoring_code_sha256": sha256_json(code_hashes()),
        "reference_model_sha256": reference.get("model_sha256") if reference else None,
    })
    evaluation.save_scores(folder, name, rows, maps, summary)
    return summary


def reference_of(folder: Path, name: str) -> dict:
    return evaluation.load_reference(folder, name)


def command_p3(args, session: Session) -> None:
    bundle = Path(args.bundle)
    inputs, ids = calibration_inputs(session)
    write_json(results_dir(bundle) / "p3_calibration_ids.json", ids)
    validation = session.validation()
    val_dir = results_dir(bundle) / "val"
    for label in [l for l in (args.only or P3_LABELS) if l in P3_LABELS]:  # --only also sets the order
        score(session, model_path(bundle, label), validation, val_dir / "fp32", label, None)
        reference = reference_of(val_dir / "fp32", label)
        for method in [m for m in METHODS if not args.methods or m in args.methods]:
            target = model_path(bundle, label, method)
            report_path = bundle / "p3" / method / f"{label}.json"
            if not report_path.is_file():
                report = quantize(model_path(bundle, label), target, method, inputs)
                write_json(report_path, report)
            summary = score(session, target, validation, val_dir / f"p3_{method}", label, reference)
            fp32 = json.loads((val_dir / "fp32" / f"{label}_summary.json").read_text(encoding="utf-8"))
            print(f"[p3] {label:24s} {method:10s} Dice {summary['symptom_mean_dice']:.4f} "
                  f"(FP32 {fp32['symptom_mean_dice']:.4f}) | drift GH {summary['drift_gold_host_mean_pp']:.2f} pp", flush=True)


def command_select(args, session: Session) -> None:
    bundle = Path(args.bundle)
    val_dir = results_dir(bundle) / "val"
    table = {}
    for method in METHODS:
        drops, drifts, per_model = [], [], {}
        for label in P3_LABELS:
            fp32 = json.loads((val_dir / "fp32" / f"{label}_summary.json").read_text(encoding="utf-8"))
            int8 = json.loads((val_dir / f"p3_{method}" / f"{label}_summary.json").read_text(encoding="utf-8"))
            drop = fp32["symptom_mean_dice"] - int8["symptom_mean_dice"]
            drops.append(drop); drifts.append(int8["drift_gold_host_mean_pp"])
            per_model[label] = {"dice_drop": drop, "gold_host_drift_pp": int8["drift_gold_host_mean_pp"]}
        table[method] = {"mean_dice_drop": float(np.mean(drops)), "mean_gold_host_drift_pp": float(np.mean(drifts)),
                         "models": per_model}
    ranked = sorted(table, key=lambda m: table[m]["mean_dice_drop"])
    best = ranked[0]
    if len(ranked) > 1 and table[ranked[1]]["mean_dice_drop"] - table[best]["mean_dice_drop"] < 0.002:
        tied = [m for m in ranked if table[m]["mean_dice_drop"] - table[best]["mean_dice_drop"] < 0.002]
        best = min(tied, key=lambda m: table[m]["mean_gold_host_drift_pp"])
    report = {"rule": "smallest pooled mean Dice drop over the six seed-20260924 models; < 0.002 apart -> smaller gold-host drift",
              "methods": table, "selected": best}
    write_json(results_dir(bundle) / "p3_report.json", report)
    for method in ranked:
        print(f"[select] {method:10s} mean Dice drop {table[method]['mean_dice_drop']:.4f} | "
              f"mean GH drift {table[method]['mean_gold_host_drift_pp']:.2f} pp")
    print("[select] SELECTED:", best)


def command_int8(args, session: Session) -> None:
    bundle = Path(args.bundle)
    selected = json.loads((results_dir(bundle) / "p3_report.json").read_text(encoding="utf-8"))["selected"]
    inputs = None
    for label, info in registry().items():
        if info["fmt"] != "int8" or (args.only and label not in args.only):
            continue
        target, meta_path = model_path(bundle, label), bundle / "meta" / f"{label}.json"
        if target.is_file() and meta_path.is_file():
            continue
        base = info["fp32_ref"]
        candidate = model_path(bundle, base, selected)
        if candidate.is_file():  # seed-20260924 models were already quantized in P3
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, target)
            report = json.loads((bundle / "p3" / selected / f"{base}.json").read_text(encoding="utf-8"))
        else:
            inputs = inputs or calibration_inputs(session)[0]
            report = quantize(model_path(bundle, base), target, selected, inputs)
        write_json(meta_path, {"label": label, **info, **report, "onnx_sha256": sha256_file(target)})
        print(f"[int8] {label}: {report['bytes'] / 1e6:.1f} MB | QLinearConv {report['QLinearConv']} | "
              f"float ops {report['floating_point_ops']}", flush=True)


def command_validate(args, session: Session) -> None:
    bundle = Path(args.bundle)
    validation = session.validation()
    val_dir = results_dir(bundle) / "val"
    models = registry()
    for fmt in ("fp32", "int8"):
        for label, info in models.items():
            if info["fmt"] != fmt or (args.only and label not in args.only):
                continue
            reference = reference_of(val_dir / "fp32", info["fp32_ref"]) if fmt != "fp32" else None
            summary = score(session, model_path(bundle, label), validation, val_dir / fmt, label, reference)
            print(f"[val] {label:32s} Dice {summary['symptom_mean_dice']:.4f} | host {summary['host_mean_dice_gold']:.4f} | "
                  f"GH {summary['gold_host_burden_mae_pp_all']:.2f} | e2e {summary['e2e_burden_mae_pp_all']:.2f}", flush=True)


def command_bench(args, session: Session) -> None:
    bundle = Path(args.bundle)
    labels = (["p1_nohost_s24", "p1_host_s24"] + [f"p2_{arch}_s24" for arch in tb.ARCHS]
              + ["p1_nohost_s24__int8"] + [f"p2_{arch}_s24__int8" for arch in tb.ARCHS])
    inputs = [evaluation.letterbox_input(data.load_record_arrays(r)[0], SIZE)[0] for r in session.validation()[:16]]
    sizes = {label: model_path(bundle, label).stat().st_size for label in registry() if model_path(bundle, label).is_file()}
    results = {label: {"bytes": sizes[label], "rounds": {"1": [], "4": []}} for label in labels}
    rounds, warmup, iterations = 3, 5, 30
    for round_index in range(rounds):
        shift = round_index * max(1, len(labels) // rounds)
        for label in labels[shift:] + labels[:shift]:
            for threads in (1, 4):
                results[label]["rounds"][str(threads)].append(
                    runtime.benchmark(model_path(bundle, label), inputs, threads, warmup, iterations))
            print(f"[bench] round {round_index + 1} {label}", flush=True)
    for label, result in results.items():
        result["latency_ms"] = {t: float(np.median([r["median_ms"] for r in rs])) for t, rs in result["rounds"].items()}
        print(f"[bench] {label:28s} {result['bytes'] / 1e6:6.1f} MB | 1 thread {result['latency_ms']['1']:.0f} ms | "
              f"4 threads {result['latency_ms']['4']:.0f} ms")
    write_json(results_dir(bundle) / "bench_report.json",
               {"created_at_utc": utc_now(), "environment": runtime.environment(), "rounds": rounds, "warmup": warmup,
                "iterations": iterations, "size": SIZE, "models": results, "all_model_bytes": sizes})


# ----------------------------------------------------------------------------- lock + test

def code_hashes() -> dict:
    hashes = {name: sha256_file(HERE / name) for name in CODE_FILES}
    hashes[PROTOCOL.relative_to(ROOT).as_posix()] = sha256_file(PROTOCOL)
    return hashes


def model_hashes(bundle: Path) -> dict:
    return {label: sha256_file(model_path(bundle, label)) for label in registry()}


def manifest_hashes(session: Session) -> dict:
    keys = ("tomato_manifest", "pseudo_host_manifest", "pretraining_manifest", "review_csv", "host_gold_manifest")
    paths = {key: Path(session.config["paths"][key]) for key in keys}
    missing = {key: str(path) for key, path in paths.items() if not path.is_file()}
    if missing:
        raise RuntimeError(f"Manifest files are missing: {missing}")
    return {key: sha256_file(path) for key, path in paths.items()}


def tree_file_hashes(root: Path) -> dict:
    if not root.is_dir():
        raise RuntimeError(f"Required bundle directory is missing: {root}")
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def source_checkpoint_hashes(source_registry: Path, root: Path | None = None) -> dict:
    sources = load_model_sources(source_registry, root=root)
    return {label: entry["sha256"] for label, entry in sources.items()}


def lock_state(args, session: Session) -> dict:
    if not args.sources:
        raise RuntimeError("--sources is required for the pre-test provenance lock")
    bundle = Path(args.bundle)
    val_dir = results_dir(bundle) / "val"
    validation = session.validation()
    for label, info in registry().items():
        reference = reference_of(val_dir / "fp32", info["fp32_ref"]) if info["fmt"] == "int8" else None
        validate_score_cache(val_dir / info["fmt"], label, model_path(bundle, label), validation, reference)
    p3_report = json.loads((results_dir(bundle) / "p3_report.json").read_text(encoding="utf-8"))
    if p3_report.get("selected") != PROTOCOL_DATA["ptq"]["selected"].split()[0].lower():
        raise RuntimeError("P3 selection does not match the frozen protocol")
    for method in METHODS:
        for label in P3_LABELS:
            reference = reference_of(val_dir / "fp32", label)
            validate_score_cache(
                val_dir / f"p3_{method}", label, model_path(bundle, label, method), validation, reference
            )
    source_path = Path(args.sources)
    if not source_path.is_file():
        raise RuntimeError(f"Model source registry is missing: {source_path}")
    source_hash = sha256_file(source_path)
    checkpoint_hashes = source_checkpoint_hashes(source_path)
    tables = {
        path.relative_to(bundle).as_posix(): sha256_file(path)
        for path in sorted(val_dir.rglob("*_summary.json"))
    }
    config_text_hash = sha256_file(session.config_path)
    return {
        "protocol": PROTOCOL_DATA["protocol"],
        "models": model_hashes(bundle),
        "code_sha256": code_hashes(),
        "config_file_sha256": config_text_hash,
        "effective_config_sha256": sha256_json(session.config),
        "manifests_sha256": manifest_hashes(session),
        "model_sources_sha256": source_hash,
        "source_checkpoints_sha256": checkpoint_hashes,
        "validation_record_ids_sha256": record_ids_sha256(validation),
        "p3_report_sha256": sha256_file(results_dir(bundle) / "p3_report.json"),
        "p3_candidates_sha256": tree_file_hashes(bundle / "p3"),
        "validation_summaries_sha256": tables,
        "environment": runtime.environment(),
    }


def verify_lock(args, session: Session) -> dict:
    lock_path = results_dir(Path(args.bundle)) / "pretest_lock.json"
    if not lock_path.is_file():
        raise RuntimeError(f"Pre-test lock is missing: {lock_path}")
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = document.get("locked_state")
    if not isinstance(locked, dict):
        raise RuntimeError("Invalid pre-test lock schema")
    current = lock_state(args, session)
    differences = {key: (locked.get(key), value) for key, value in current.items() if locked.get(key) != value}
    if differences:
        raise RuntimeError(f"Pre-test lock mismatch in fields: {sorted(differences)}")
    return document


def command_lock(args, session: Session) -> None:
    bundle = Path(args.bundle)
    missing = [label for label in registry() if not model_path(bundle, label).is_file()]
    if missing:
        raise RuntimeError(f"Not every model is exported and validated: {missing[:6]}")
    state = lock_state(args, session)
    lock = {"schema_version": 2, "locked_state": state, "metadata": {"created_at_utc": utc_now()}}
    lock_path = results_dir(bundle) / "pretest_lock.json"
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        previous_state = previous.get("locked_state", {})
        differences = {key for key, value in state.items() if previous_state.get(key) != value}
        if differences:
            raise RuntimeError(f"A lock already exists but differs in: {sorted(differences)}")
        print("Lock already exists and matches.")
        return
    write_json(lock_path, lock)
    print(f"Locked {len(state['models'])} models, {len(state['validation_summaries_sha256'])} validation summaries -> {lock_path}")


def seed_mean_rows(rows_by_label: dict, labels: list[str], endpoints: list[str]) -> list[dict]:
    first = rows_by_label[labels[0]]
    merged = []
    for sample_id, row in first.items():
        entry = {k: row[k] for k in ("sample_id", "class_name", "focal", "duplicate_group")}
        for endpoint in endpoints:
            entry[endpoint] = float(np.mean([float(rows_by_label[l][sample_id][endpoint]) for l in labels]))
        merged.append(entry)
    return merged


def p_value(result: dict) -> float:
    below = result["probability_below_zero"]
    return float(min(1.0, 2 * min(below, 1 - below)))


def holm(results: list[dict]) -> None:
    order = sorted(range(len(results)), key=lambda i: results[i]["p_value"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(results) - rank) * results[index]["p_value"]))
        results[index]["p_holm"] = running


def analyses(rows_by_label: dict, summaries: dict) -> dict:
    endpoints = ["symptom_dice", "host_dice_gold", "e2e_abs_error_pp", "gold_host_abs_error_pp"]
    tags = list(SEEDS.values())
    nohost, host = [f"p1_nohost_{t}" for t in tags], [f"p1_host_{t}" for t in tags]
    focal = lambda r: bool(r["focal"])  # noqa: E731
    every = lambda r: True  # noqa: E731
    boot = lambda a, b, e, pop: statistics.cluster_bootstrap_delta(a, b, e, pop, BOOT_RESAMPLES, BOOT_SEED)  # noqa: E731
    # P1 primary (registered): e2e absolute burden error, focal images, host_v1 - no_host.
    host_rows, nohost_rows = seed_mean_rows(rows_by_label, host, endpoints), seed_mean_rows(rows_by_label, nohost, endpoints)
    primary = boot(host_rows, nohost_rows, "e2e_abs_error_pp", focal)
    dice_host = float(np.mean([r["symptom_dice"] for r in host_rows]))
    dice_nohost = float(np.mean([r["symptom_dice"] for r in nohost_rows]))
    p1 = {"primary": primary, "dice_host_v1": dice_host, "dice_no_host": dice_nohost,
          "supported": bool(primary["ci95_upper"] < 0 and dice_host >= dice_nohost - 0.01),
          "secondary": {f"{e}|{name}": boot(host_rows, nohost_rows, e, pop)
                        for e in endpoints for name, pop in (("all", every), ("focal", focal))}}
    # P2: ours (no_host) vs each baseline, FP32 and INT8, Holm over the 5 baselines per endpoint.
    p2 = {}
    for fmt, suffix in (("fp32", ""), ("int8", "__int8")):
        ours = seed_mean_rows(rows_by_label, [f"{l}{suffix}" for l in nohost], endpoints)
        block = {}
        for endpoint in ("symptom_dice", "gold_host_abs_error_pp"):
            for population_name, population in (("all", every), ("focal", focal)):
                results = []
                for arch in tb.ARCHS:
                    other = seed_mean_rows(rows_by_label, [f"p2_{arch}_{t}{suffix}" for t in tags], endpoints)
                    result = boot(ours, other, endpoint, population)
                    result.update({"baseline": arch, "p_value": p_value(result)})
                    results.append(result)
                holm(results)
                block[f"{endpoint}|{population_name}"] = results
        p2[fmt] = block
    # Supplementary: same seed, other software image or plain rerun.
    pairs = {"luot8_vs_new_image_s24": ("supp_luot8_s24_py312", "p1_nohost_s24"),
             "old_vs_new_image_s26": ("supp_nohost_s26_py312", "p1_nohost_s26"),
             "rerun_same_image_s26": ("supp_nohost_s26_rerun_py312", "supp_nohost_s26_py312")}
    supplementary = {}
    for name, (a, b) in pairs.items():
        supplementary[name] = {e: float(np.mean([abs(float(rows_by_label[a][i][e]) - float(rows_by_label[b][i][e]))
                                                 for i in rows_by_label[a]])) for e in endpoints}
    return {"P1": p1, "P2": p2, "supplementary_mean_abs_per_image_difference": supplementary}


def command_test(args, session: Session) -> None:
    bundle = Path(args.bundle)
    lock_path = results_dir(bundle) / "pretest_lock.json"
    lock = verify_lock(args, session)
    dry = args.dry_run_on_validation
    test_dir = results_dir(bundle) / ("dryrun_on_validation" if dry else "test")
    marker = test_dir / "test_attempt.json"
    if not (args.confirm_test_use or dry):
        print("Lock verified. The test set is NOT opened without --confirm-test-use.")
        return
    if dry:
        records = session.validation()  # rehearsal of the whole test path; no test record is read
    else:
        if marker.is_file():
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            current_lock_hash = sha256_file(lock_path)
            if marker_data.get("pretest_lock_sha256") != current_lock_hash:
                raise RuntimeError("Existing test-attempt marker belongs to a different pre-test lock")
            print("Test already opened: technical resume of the missing models only.")
        else:
            protocol_id = json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol"]
            write_json(marker, {"opened_at_utc": utc_now(), "pretest_lock_sha256": sha256_file(lock_path),
                                "declared": f"third and final declared use of the 176 test images ({protocol_id})"})
        records = session.test()
        if len(records) != 176:
            raise RuntimeError(f"Expected 176 test records, found {len(records)}")
    rows_by_label, summaries = {}, {}
    for fmt in ("fp32", "int8"):
        for label, info in registry().items():
            if info["fmt"] != fmt:
                continue
            reference = reference_of(test_dir / "fp32", info["fp32_ref"]) if fmt != "fp32" else None
            summaries[label] = score(session, model_path(bundle, label), records, test_dir / fmt, label, reference)
            rows = json.loads((test_dir / fmt / f"{label}_rows.json").read_text(encoding="utf-8"))
            rows_by_label[label] = {r["sample_id"]: r for r in rows}
            print(f"[test] {label:32s} Dice {summaries[label]['symptom_mean_dice']:.4f} | "
                  f"e2e {summaries[label]['e2e_burden_mae_pp_all']:.2f}", flush=True)
    report = analyses(rows_by_label, summaries)
    write_json(test_dir / "analysis_report.json", report)
    p = report["P1"]["primary"]
    print(f"\nP1 (host_v1 - no_host, e2e error, focal): {p['delta']:+.2f} pp [{p['ci95_lower']:+.2f}; {p['ci95_upper']:+.2f}] "
          f"-> {'SUPPORTED' if report['P1']['supported'] else 'NOT supported'}")



# ----------------------------------------------------------------------------- tables

FAMILIES = [("ours, full recipe (no_host)", "p1_nohost_{t}"), ("ours + host labels (host_v1)", "p1_host_{t}"),
            ("ours, ImageNet only", "p2_ours_imagenet_{t}"), ("SegFormer-B0", "p2_segformer_b0_{t}"),
            ("U-Net ResNet-34", "p2_unet_r34_{t}"), ("DeepLabV3+ ResNet-50", "p2_deeplabv3p_r50_{t}"),
            ("LR-ASPP MobileNetV3", "p2_lraspp_mnv3_{t}")]
TABLE = [("symptom_mean_dice", "Dice", 3), ("symptom_mean_bf1", "BF1", 3), ("host_mean_dice_gold", "host Dice", 3),
         ("gold_host_burden_mae_pp_all", "GH MAE", 2), ("e2e_burden_mae_pp_all", "e2e MAE", 2),
         ("e2e_burden_mae_pp_focal", "e2e MAE focal", 2), ("drift_gold_host_mean_pp", "GH drift", 2)]


def command_report(args, session: Session) -> None:
    bundle = Path(args.bundle)
    base = results_dir(bundle) / args.split

    def load(fmt: str, label: str):
        path = base / fmt / f"{label}_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def cell(values: list, digits: int) -> str:
        values = [v for v in values if v is not None]
        if not values:
            return "-"
        text = f"{np.mean(values):.{digits}f}"
        return text + (f" ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else "")

    lines = [f"# {args.split}: {len(list(base.rglob('*_summary.json')))} scored models", "",
             "| model | format | runs | " + " | ".join(name for _, name, _ in TABLE) + " |",
             "|---|---|---|" + "---|" * len(TABLE)]
    rows = []
    for fmt, suffix in (("fp32", ""), ("int8", "__int8")):
        for name, pattern in FAMILIES:
            if fmt == "int8" and "host_v1" in name:
                continue
            found = [s for s in (load(fmt, pattern.format(t=t) + suffix) for t in SEEDS.values()) if s]
            if found:
                rows.append({"model": name, "format": fmt, "runs": len(found),
                             **{key: [s.get(key) for s in found] for key, _, _ in TABLE}})
    for label in ("supp_luot8_s24_py312", "supp_nohost_s26_py312", "supp_nohost_s26_rerun_py312"):
        found = load("fp32", label)
        if found:
            rows.append({"model": label, "format": "fp32 (old image)", "runs": 1, **{key: [found.get(key)] for key, _, _ in TABLE}})
    for row in rows:
        lines.append(f"| {row['model']} | {row['format']} | {row['runs']} | "
                     + " | ".join(cell(row[key], digits) for key, _, digits in TABLE) + " |")
    text = "\n".join(lines) + "\n"
    (results_dir(bundle) / f"{args.split}_tables.md").write_text(text, encoding="utf-8")
    write_json(results_dir(bundle) / f"{args.split}_tables.json", rows)
    print(text)


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "p3", "select", "int8", "validate", "bench", "lock", "test", "report"))
    parser.add_argument("--config", type=Path, required=True, help="session config with the data paths")
    parser.add_argument("--bundle", type=Path, required=True, help="folder with the ONNX models and results")
    parser.add_argument("--sources", type=Path, help="export: JSON label -> checkpoint or INT8 ONNX path")
    parser.add_argument("--only", nargs="*", default=[], help="restrict to these labels")
    parser.add_argument("--methods", nargs="*", default=[], help="p3: restrict to these methods")
    parser.add_argument("--confirm-test-use", action="store_true", help="test: open the 176 test images (once)")
    parser.add_argument("--dry-run-on-validation", action="store_true", help="test: rehearse every step on the val images")
    parser.add_argument("--cache-images", action="store_true", help="keep decoded images in RAM across models")
    parser.add_argument("--split", choices=("val", "test", "dryrun_on_validation"), default="val",
                        help="report: which results to tabulate")
    args = parser.parse_args()
    if args.command in ("export", "lock", "test") and args.sources is None:
        parser.error(f"{args.command} requires --sources for checkpoint provenance validation")
    session = Session(args.config, args.cache_images)
    {"export": command_export, "p3": command_p3, "select": command_select, "int8": command_int8,
     "validate": command_validate, "bench": command_bench, "lock": command_lock, "test": command_test,
     "report": command_report}[args.command](args, session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
