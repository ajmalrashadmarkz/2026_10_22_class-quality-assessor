"""Calibration script for the class assessor.

Runs the assessment pipeline over a labelled sample set, reports per-file
predictions and the Spearman rank correlation between true and predicted
labels, and derives suggested score thresholds from per-class mean scores.
"""

import csv
import json
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from class_assessor import pipeline

ORDER = ["Poor", "Average", "Good", "Excellent"]


def main(samples_dir: str, labels_csv: str) -> None:
    """Evaluate the pipeline on labelled samples and suggest thresholds."""
    rows: list[dict[str, str]] = []
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    pairs: list[tuple[str, float]] = []
    for r in rows:
        video = Path(samples_dir) / r["file"]
        assessment = pipeline.run(video)
        pairs.append((r["label"], assessment.overall_score))
        print(
            f"{r['file']}: true={r['label']} "
            f"predicted={assessment.label} score={assessment.overall_score:.2f}"
        )

    true_ranks = [ORDER.index(label) for label, _score in pairs]
    pred_scores = [score for _label, score in pairs]
    if len(pairs) >= 2:
        corr, pvalue = spearmanr(true_ranks, pred_scores)
        print(f"\nSpearman rank correlation: {corr:.3f} (p={pvalue:.3f})")
    else:
        print("\nNot enough samples to compute Spearman correlation.")

    per_class: dict[str, list[float]] = {label: [] for label in ORDER}
    for label, score in pairs:
        per_class[label].append(score)

    class_means = [
        mean(per_class[label]) for label in ORDER if per_class[label]
    ]
    print("\nMean score per class:")
    for label in ORDER:
        if per_class[label]:
            print(f"  {label}: {mean(per_class[label]):.2f}")

    mids = [
        (class_means[i] + class_means[i + 1]) / 2.0
        for i in range(len(class_means) - 1)
    ]

    if len(mids) == 3:
        calibration_path = (
            Path(__file__).resolve().parent.parent
            / "class_assessor"
            / "calibration.json"
        )
        calibration = {
            "thresholds": {
                "Poor": 0.0,
                "Average": round(mids[0], 2),
                "Good": round(mids[1], 2),
                "Excellent": round(mids[2], 2),
            },
            "confidence_margin": 0.25,
        }
        with calibration_path.open("w", encoding="utf-8") as fh:
            json.dump(calibration, fh, indent=2)
            fh.write("\n")
        print(f"\nUpdated calibration written to {calibration_path}")
    else:
        print(
            "\nSkipped calibration update: need samples for all four classes "
            "to derive three midpoints."
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
