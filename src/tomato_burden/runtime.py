"""ONNX export, graph inspection, environment capture, and CPU benchmarking."""

from __future__ import annotations

import inspect
import os
import platform
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUTPUT_NAMES = ["symptom_logits", "host_logits"]


def cpu_info() -> dict:
    name, flags = platform.processor(), ""
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        name = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), name)
        flags = next((line.split(":", 1)[1] for line in lines if line.startswith("flags")), "")
    except OSError:
        pass
    vnni = ("avx512_vnni" in flags or "avx_vnni" in flags) if flags else None
    return {"processor": name, "vnni": vnni, "logical_cpus": os.cpu_count()}


def environment() -> dict:
    import onnx
    import onnxruntime
    import timm

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **cpu_info(),
    }


def export_onnx(model: nn.Module, path: Path, size: int, opset: int = 17) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "opset_version": opset,
        "input_names": ["image"],
        "output_names": OUTPUT_NAMES,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(model.cpu().eval(), (torch.zeros(1, 3, size, size),), str(path), **kwargs)
    onnx.checker.check_model(str(path))


def ort_session(
    path: Path,
    threads: int = 0,
    level: str = "all",
    exact: bool = True,
    optimized_path: Path | None = None,
):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[level]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    if exact:
        options.add_session_config_entry("session.x64quantprecision", "1")
    if optimized_path is not None:
        options.optimized_model_filepath = str(optimized_path)
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def op_counts(path: Path) -> dict[str, int]:
    import onnx

    return dict(Counter(node.op_type for node in onnx.load(str(path)).graph.node).most_common())


def fused_op_counts(path: Path) -> dict[str, int]:
    """Return operators after ONNX Runtime graph optimization."""
    with tempfile.TemporaryDirectory() as folder:
        optimized = Path(folder) / "optimized.onnx"
        ort_session(path, threads=1, level="extended", exact=False, optimized_path=optimized)
        return op_counts(optimized)


def benchmark(path: Path, inputs: list[np.ndarray], threads: int, warmup: int, iterations: int) -> dict:
    if not inputs:
        raise ValueError("At least one benchmark input is required")
    session = ort_session(path, threads=threads, exact=False)
    for index in range(warmup):
        session.run(None, {"image": inputs[index % len(inputs)]})
    times = []
    for index in range(iterations):
        start = time.perf_counter()
        session.run(None, {"image": inputs[index % len(inputs)]})
        times.append((time.perf_counter() - start) * 1000)
    median = float(np.median(times))
    return {
        "threads": threads,
        "median_ms": median,
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "fps": 1000.0 / median,
    }

