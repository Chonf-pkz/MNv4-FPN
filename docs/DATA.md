# Data and artifact contract

The frozen modelling cohort contains 595 training, 85 validation, and 176 test images. The focal test subset contains 152 images. Gold host masks cover all 85 validation and 176 test images. The non-tomato source manifest contains 7,580 active records: 6,828 training images and 752 validation/monitoring images.

Training uses reviewed lesion labels and generated host proposals. The manual-host comparison substitutes 396 manually corrected training masks; the main recipe uses 199 accepted SAM2 host proposals where available. Exact inclusion rules and counts are frozen in `configs/protocol_final.json`.

## Expected private layout

```text
artifacts/private/
  manifests/
    tomato_manifest.csv
    label_review_decisions.csv
    pseudo_host_manifest.csv
    manual_host_manifest.csv
    host_gold_final_manifest.csv
    pretraining_manifest.csv
  model_sources.json
  locked_outputs/
  ... image and mask roots referenced by the manifests ...
```

Paths inside manifests may be relative to the private bundle. Do not commit private images, masks, checkpoints, credentials, or signed download URLs.

`model_sources.json` is not a plain path list. Each entry must follow `configs/model_sources.example.json` and include the checkpoint and host-manifest SHA-256 values, seed, network, experiment variant, source-run identity, and cohort attestation. The paper pipeline validates this metadata before ONNX export.

## Dataset licensing

This repository does not redistribute source datasets. Users must obtain each source dataset under its own terms and construct the prepared manifests consistently with the author bundle. The Apache-2.0 license applies only to repository code and documentation.
