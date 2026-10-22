"""Paired comparison of two prediction sets measured on identical rows.

The point of this script is that comparisons must respect the pairing. Two runs
scored on the same validation rows have correlated errors, so:

  * the noise on the DELTA is not the noise on either estimate - it is measured by
    resampling rows and recomputing both F1s on the same resample;
  * the right significance test is McNemar on discordant pairs, not a Poisson
    model on a marginal count.

Reports, per subset (all rows, then each language): both F1s, the delta, a paired
bootstrap CI for the delta, and an exact McNemar p-value. Also prints both full
confusion matrices, because a change that trades one error cell for another can
look fine on a two-cell check.

Usage:
    python work/compare_pair.py --probs-a A.npy --probs-b B.npy --fold 0
    python work/compare_pair.py --probs-a A.npy --probs-b B.npy --fold all
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
LANGUAGES = ["pt", "fr", "it", "en"]
BOOTSTRAP_DRAWS = 2000


def macro_f1(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Macro-F1 from counts, equivalent to sklearn with zero_division=0.

    Written locally because this runs 2,000 times inside the bootstrap, and because
    it keeps the estimate independent of sklearn's zero-division typing.
    """
    scores = []
    for class_id in range(len(LABELS)):
        true_positive = float(np.sum((prediction == class_id) & (reference == class_id)))
        false_positive = float(np.sum((prediction == class_id) & (reference != class_id)))
        false_negative = float(np.sum((prediction != class_id) & (reference == class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(max(only_a, only_b), discordant + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def paired_bootstrap(
    reference: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float]:
    """Resample rows; return (delta std, 2.5th pct, 97.5th pct) of F1_b - F1_a."""
    generator = np.random.default_rng(seed)
    count = len(reference)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pick = generator.integers(0, count, size=count)
        deltas[draw] = macro_f1(reference[pick], prediction_b[pick]) - macro_f1(
            reference[pick], prediction_a[pick]
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.std()), float(low), float(high)


def confusion(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true_id, pred_id in zip(reference, prediction, strict=True):
        matrix[true_id, pred_id] += 1
    return matrix


def describe(name: str, mask: np.ndarray, reference: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> None:
    ref, pa, pb = reference[mask], pred_a[mask], pred_b[mask]
    if len(ref) < 30:
        print(f"{name:6s} n={len(ref):5d}  (too few rows to compare)")
        return
    f1_a = macro_f1(ref, pa)
    f1_b = macro_f1(ref, pb)
    delta_std, low, high = paired_bootstrap(ref, pa, pb, BOOTSTRAP_DRAWS, seed=7)
    correct_a, correct_b = pa == ref, pb == ref
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    p_value = mcnemar_exact(only_a, only_b)
    crosses_zero = low <= 0.0 <= high
    print(
        f"{name:6s} n={len(ref):5d}  A={f1_a:.4f}  B={f1_b:.4f}  "
        f"delta={f1_b - f1_a:+.4f}  paired CI=[{low:+.4f},{high:+.4f}]  "
        f"McNemar {only_a}/{only_b} p={p_value:.4f}{'  (CI spans 0)' if crosses_zero else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs-a", type=Path, required=True)
    parser.add_argument("--probs-b", type=Path, required=True)
    parser.add_argument("--fold", default="0", help="fold index, or 'all' for every covered row")
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--folds", type=Path, default=Path("kernels/rmit-sgs-ae-do-toc-c1/out/folds.csv"))
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    folds = pd.read_csv(args.folds)
    frame = train.merge(folds, on="id", how="left")

    reference = np.array([LABEL_TO_ID[name] for name in frame.label])
    pred_a = np.load(args.probs_a).argmax(axis=1)
    pred_b = np.load(args.probs_b).argmax(axis=1)
    covered = np.load(args.probs_a).sum(axis=1) > 0
    if args.fold == "all":
        rows = covered
    else:
        rows = (frame.fold == int(args.fold)).to_numpy() & covered

    print(f"A={args.label_a}  B={args.label_b}  fold={args.fold}  rows={int(rows.sum())}")
    describe("ALL", rows, reference, pred_a, pred_b)
    langs = frame.language.to_numpy()
    for language in LANGUAGES:
        describe(language, rows & (langs == language), reference, pred_a, pred_b)

    print("\nconfusion for A (rows=true, cols=pred):")
    print(
        pd.DataFrame(
            confusion(reference[rows], pred_a[rows]),
            index=pd.Index(LABELS),
            columns=pd.Index(LABELS),
        ).to_string()
    )
    print("\nconfusion for B:")
    print(
        pd.DataFrame(
            confusion(reference[rows], pred_b[rows]),
            index=pd.Index(LABELS),
            columns=pd.Index(LABELS),
        ).to_string()
    )

    benign, jailbreak = LABEL_TO_ID["BENIGN"], LABEL_TO_ID["JAILBREAK"]
    pair_rows = rows & ((reference == benign) | (reference == jailbreak))
    describe("B|J", pair_rows, reference, pred_a, pred_b)
    for label, pred in ((args.label_a, pred_a), (args.label_b, pred_b)):
        b_to_j = int(((reference == benign) & (pred == jailbreak) & pair_rows).sum())
        j_to_b = int(((reference == jailbreak) & (pred == benign) & pair_rows).sum())
        print(f"  {label}: B->J {b_to_j:3d}   J->B {j_to_b:3d}")


if __name__ == "__main__":
    main()
