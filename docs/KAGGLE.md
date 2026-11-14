# Kaggle execution

The three scripts in `runners/` replace the run-specific notebooks used during development. They accept explicit seed and architecture arguments, making the public workflow easier to audit.

On Kaggle, add the repository and the private artifact bundle as inputs, enable a Tesla T4 GPU for training, and set paths in a copied `configs/paper.json`. The final target-stage software image reported Python 3.13.15, PyTorch 2.11.0+cu128, and timm 1.0.29.

```bash
python runners/kaggle_train_mnv4.py \
  --config configs/paper.json --seed 20260924 --variant standard --stage all

python runners/kaggle_train_mnv4.py \
  --config configs/paper.json --seed 20260924 --variant manual-host \
  --manual-host-manifest artifacts/private/manifests/manual_host_manifest.csv --stage target

python runners/kaggle_train_baselines.py \
  --config configs/paper.json --seed 20260924 --arch all --preflight
```

Run the shared source stage once with seed `20260924`. For seeds `20260925` and `20260926`, reuse that source checkpoint and launch the MNv4 runner with `--stage target`; repeat each baseline command with the new seed. Once every FP32 checkpoint is present, fill `model_sources.json` from `configs/model_sources.example.json` and run:

```bash
python runners/kaggle_ptq_evaluation.py \
  --config configs/paper.json \
  --sources artifacts/private/model_sources.json \
  --bundle artifacts/paper_bundle
```

That command stops after the pre-test lock. Supplying `--include-test` opens and scores the test records; it should be used only when the reproduction protocol permits a test evaluation.

## Verified seed-26 manual-host provenance

The final seed-26 manual-host source is [khanhlnht/seed-26-host, version 1](https://www.kaggle.com/code/khanhlnht/seed-26-host). Its executed preflight found 595 reviewed train records with host supervision (199 accepted SAM2 masks plus 396 manually corrected masks). Its 640-pixel validation lesion Dice was `0.7042755211857831`, exactly matching `val.host_v1.fp32.symptom_mean_dice.runs[2]` in the checked-in aggregate snapshot.

Do **not** use `khanhlnht/seed-26-c-nh-n` versions 2--4. Those superseded runs declared a host-label download URL but did not extract it or replace `paths.pseudo_host_manifest`; they therefore trained from the standard host manifest. The structured registry and the runner's cohort preflight were added specifically to make that failure stop before training/export.

The repository does not invent source-run IDs for runs that were not independently verified during this audit. Replace every `REPLACE_WITH_FINAL_RUN_ID` and every hash placeholder in the example registry from the retained Kaggle version and private bundle before release.
