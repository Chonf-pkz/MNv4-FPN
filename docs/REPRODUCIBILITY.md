# Reproducibility statement

## What this repository reproduces without private artifacts

- The final FP32-to-static-INT8 PTQ implementation and its frozen hyperparameters.
- Configuration validation, metric implementations, clustered bootstrap statistics, and model registry.
- Data-independent unit tests and command-line interfaces.

## What requires the author artifact bundle

Exact training, per-image scoring, bootstrap recomputation, model export, latency measurement, and checksum verification require the artifacts described by `configs/private_artifacts.example.json`. In particular, the release does not contain dataset copies, reviewed-label decisions, generated host masks, model checkpoints, ONNX files, or per-image predictions.

This is a framework reproduction package. Running the full workflow requires the prepared data and weights described above. Manuscript figures, tables, and their aggregate snapshot are excluded.

The reported runs used seeded data loading but were not guaranteed bit-for-bit deterministic across CUDA kernels or software images. The historical cache files and SHA-256 values of every third-party ImageNet initialization were not retained. The three-seed summaries and the supplementary rerun quantify some of this variation; the release does not claim identical floating-point bits on a new machine.

## Integrity controls

1. `configs/protocol_final.json` freezes cohort sizes, seeds, image sizes, PTQ candidates, selection rule, and bootstrap settings.
2. `tomato_burden.config.validate_paper_recipe` rejects changes to paper-critical training fields.
3. `scripts/validate_private_artifacts.py` checks artifact presence, record counts, and SHA-256 values.
4. The structured `model_sources.json` binds every checkpoint to a seed, network, experiment variant, source run, cohort attestation, host-manifest hash, and checkpoint hash before export.
5. `tomato-paper lock` hashes all exported models, source checkpoints, evaluation code, protocol, manifests, and validation summaries before test access.
6. `tomato-paper test` refuses to score test images unless the lock matches and explicit confirmation is supplied.

## Recommended verification order

```bash
python -m pip install -e .[test]
pytest
python scripts/validate_private_artifacts.py configs/private_artifacts.json
```

With the private bundle available, run the training launchers for all three seeds, create `artifacts/private/model_sources.json` from the structured example, replace every placeholder, and run the PTQ evaluation runner. A legacy label-to-path mapping is intentionally rejected because it cannot distinguish superseded and final training runs. Use `--include-test` only for an authorized reproduction of the declared test session.

## Historical experiments

The pre-refactor repository, including exploratory and quantization-aware-training code that is outside the final paper, is preserved at Git tag `experiment-original-2026`. It is not part of the paper reproduction workflow and should not be mixed with the final protocol.
