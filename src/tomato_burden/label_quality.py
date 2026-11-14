#!/usr/bin/env python3
"""Rank TRAIN images whose labels disagree most with out-of-fold (OOF) predictions.

Input: the fold_0..fold_4 folders of one target run (each holds oof_predictions.npz).
Output: label_review_candidates.csv (with an empty `decision` column to fill in) and
review_panels/*.jpg (image | label | OOF prediction | disagreement).

A low score means "the model trained WITHOUT this image disagrees with its label".
That is either a label error or a genuinely hard image; a person decides which.
Only train images are scored, so validation/test ground truth is never edited.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_oof(run_dir: Path) -> dict:
    parts = [np.load(path, allow_pickle=False) for path in sorted(run_dir.glob("fold_*/oof_predictions.npz"))]
    if not parts:
        raise FileNotFoundError(f"No fold_*/oof_predictions.npz under {run_dir}")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in
              ("sample_id", "image", "symptom_probability", "symptom_label", "disease_probability", "disease_label")}
    merged["classes"] = parts[0]["classes"]
    if len(set(merged["sample_id"].tolist())) != len(merged["sample_id"]):
        raise RuntimeError("A sample appears in more than one fold")
    return merged


def cleanlab_scores(label: np.ndarray, probability: np.ndarray) -> np.ndarray | None:
    try:
        from cleanlab.segmentation.rank import get_label_quality_scores
    except ImportError:
        return None
    pred_probs = np.stack([1.0 - probability, probability], axis=1).astype(np.float32)
    scores, _ = get_label_quality_scores(label.astype(np.int64), pred_probs, method="softmin", verbose=False)
    return scores


def panel(image: np.ndarray, label: np.ndarray, probability: np.ndarray, title: str) -> Image.Image:
    size = image.shape[0]
    base = image.astype(np.float32)

    def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        out = base.copy()
        out[mask] = 0.45 * out[mask] + 0.55 * np.array(color, dtype=np.float32)
        return out.astype(np.uint8)

    prediction = probability >= 0.5
    disagreement = base.copy() * 0.35
    disagreement[label & (probability < 0.1)] = (255, 60, 60)   # labelled lesion, model confidently says no
    disagreement[~label & (probability > 0.9)] = (60, 140, 255)  # model confidently sees a lesion that is unlabelled
    tiles = [image, overlay(label, (0, 255, 0)), overlay(prediction, (255, 0, 255)), disagreement.astype(np.uint8)]
    canvas = Image.new("RGB", (4 * size, size + 18), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(Image.fromarray(tile), (index * size, 18))
    ImageDraw.Draw(canvas).text((4, 3), title + "   [image | label | OOF pred | red=missed blue=extra]", fill="black")
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help=".../target/<run_name>_seed<seed>")
    parser.add_argument("--top-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run_dir / "label_review"
    (out / "review_panels").mkdir(parents=True, exist_ok=True)
    oof = load_oof(args.run_dir)
    classes = [str(name) for name in oof["classes"]]
    probability = oof["symptom_probability"].astype(np.float32) / 255.0
    label = oof["symptom_label"] > 127
    dims = (1, 2)
    soft_dice = (2 * (probability * label).sum(dims) + 1.0) / (probability.sum(dims) + label.sum(dims) + 1.0)
    label_area = np.maximum(label.sum(dims), 1)
    missed = (label & (probability < 0.1)).sum(dims) / label_area
    extra = (~label & (probability > 0.9)).sum(dims) / label_area
    disease = oof["disease_probability"]
    truth = oof["disease_label"]
    p_true = disease[np.arange(len(truth)), truth]
    predicted = disease.argmax(1)
    class_suspect = (p_true < 0.1) & (disease.max(1) > 0.7)
    clean = cleanlab_scores(label, probability)

    order = np.argsort(soft_dice)
    top = set(order[: int(round(args.top_fraction * len(order)))].tolist()) | set(np.flatnonzero(class_suspect).tolist())
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "sample_id": str(oof["sample_id"][index]),
            "review_candidate": index in top,
            "soft_dice_oof": round(float(soft_dice[index]), 4),
            "missed_fraction": round(float(missed[index]), 4),
            "extra_fraction": round(float(extra[index]), 4),
            "cleanlab_softmin": "" if clean is None else round(float(clean[index]), 4),
            "disease_label": classes[truth[index]],
            "disease_oof_pred": classes[predicted[index]],
            "p_label_class": round(float(p_true[index]), 4),
            "class_suspect": bool(class_suspect[index]),
            "decision": "", "corrected_class": "", "fixed_symptom_mask": "", "note": "",
        })
        if index in top:
            title = (f"#{rank} {oof['sample_id'][index]} dice={soft_dice[index]:.2f} "
                     f"label={classes[truth[index]][7:]} pred={classes[predicted[index]][7:]}")
            panel(oof["image"][index], label[index], probability[index], title).save(
                out / "review_panels" / f"rank_{rank:03d}_{oof['sample_id'][index]}.jpg", quality=90
            )
    with (out / "label_review_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} train images; {len(top)} review candidates -> {out}")
    print(f"Class-label suspects: {int(class_suspect.sum())}; cleanlab: {'used' if clean is not None else 'not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
