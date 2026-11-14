# Quantization-Friendly Lesion and Host Segmentation

Official research code for tomato-lesion segmentation, host-leaf segmentation, disease-burden estimation, and ONNX Runtime static INT8 post-training quantization (PTQ).

> Reproducibility level: **partial**. The implementation and frozen protocol are included. Exact numerical reproduction additionally requires the prepared manifests, reviewed labels, host masks, checkpoints, and per-image locked outputs listed in [`configs/private_artifacts.example.json`](configs/private_artifacts.example.json). Those artifacts are not distributed in this repository.

This release follows the final paper pipeline only:

1. Train MobileNetV4-FPN and the comparison networks in FP32.
2. Export the lesion and host heads to ONNX at 640×640.
3. Compare MinMax, entropy, and 99.999th-percentile static PTQ on validation data.
4. Quantize every comparison model with the selected calibration method.
5. Score FP32 and INT8 models with the same 384×384 metric canvas.
6. Lock models, code, protocol, and validation summaries before the declared test evaluation.

No quantization-aware training method is part of the final paper. Historical QAT experiments are preserved only in the Git tag `experiment-original-2026` and are intentionally absent from this release tree.

## Repository layout

```text
configs/          frozen paper recipe, protocol, and private-artifact contract
src/tomato_burden reusable training, evaluation, PTQ, and statistics package
scripts/          command-line entry points
runners/          parameterized Kaggle/Linux launchers replacing run-specific notebooks
tools/annotation/ local host-mask annotation utility
tests/            data-independent checks
docs/             data, Kaggle, and reproducibility documentation
```

## Installation

Python 3.13 was used for the final runs. Create an isolated environment, install the package, and verify the CLI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
tomato-train --help
tomato-paper --help
```

The exact core versions recorded by the protocol are in [`requirements-paper.txt`](requirements-paper.txt). Kaggle already supplies CUDA-enabled PyTorch; do not replace its PyTorch installation unless you are deliberately recreating the recorded software image.

## Train and evaluate

First obtain the author-provided artifact bundle, copy the example contract, insert the supplied SHA-256 values, and validate it:

```bash
cp configs/private_artifacts.example.json configs/private_artifacts.json
python scripts/validate_private_artifacts.py configs/private_artifacts.json
```

Then adjust only local paths in `configs/paper.json`. The paper-critical recipe fields are validated by code.

```bash
# Shared cache and folds
tomato-train cache  --config configs/paper.json
tomato-train folds  --config configs/paper.json

# Source pretraining and target training
python runners/kaggle_train_mnv4.py --seed 20260924 --variant standard --stage all
python runners/kaggle_train_baselines.py --seed 20260924 --arch all --preflight

# Export, PTQ selection, validation, benchmark, and pre-test lock
python runners/kaggle_ptq_evaluation.py \
  --config configs/paper.json \
  --sources artifacts/private/model_sources.json \
  --bundle artifacts/paper_bundle
```

The runner deliberately does not open the test set unless `--include-test` is supplied. The model-source registry records the checkpoint SHA-256, seed, architecture, experiment variant, host-supervision counts, host-manifest hash, and source-run identity; export stops if any of these fields conflict with the frozen protocol. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete boundary between public and private artifacts, [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the checkpoint chain of custody, and [`docs/KAGGLE.md`](docs/KAGGLE.md) for Kaggle commands.

## Citation

The author list and manuscript title are recorded in [`CITATION.cff`](CITATION.cff). Add the final venue and DOI after publication.

## License

Code is released under the [Apache License 2.0](LICENSE). Dataset images, annotations, trained weights, and third-party model weights remain subject to their original licenses and are not relicensed by this repository.
