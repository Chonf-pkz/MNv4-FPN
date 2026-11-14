# Model provenance

Every FP32 checkpoint used by the paper is declared in `artifacts/private/model_sources.json`, copied from `configs/model_sources.example.json`. Each entry binds six kinds of evidence:

- the exact checkpoint path and SHA-256;
- the registered seed and network family;
- the experiment identity (`standard`, `manual-host`, `baseline`, or `supplementary`);
- the immutable source-run identity and version;
- the reviewed train-cohort and host-supervision counts;
- the SHA-256 of the host manifest used by that run.

The export command rejects legacy string-only mappings, missing files, hash mismatches, wrong seeds or networks, wrong experiment labels, and cohort counts that conflict with `configs/protocol_final.json`. The pre-test lock then incorporates the source-registry file and all source-checkpoint hashes.

For new runs, `runners/kaggle_train_mnv4.py` performs a data preflight before creating a training config. The standard variant requires 595 reviewed train records, of which 199 have active host supervision. The manual-host variant requires all 595 train records to have active host supervision; its registered decomposition is 199 accepted SAM2 masks plus 396 manually corrected masks. The generated config records the variant, observed counts, and host-manifest hash, and that config is embedded in `final_ema.pt`.

This mechanism establishes an auditable chain of custody; it does not make private artifacts public. Exact numerical reproduction still requires the author bundle described in `configs/private_artifacts.example.json`.
