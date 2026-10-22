"""Merge sharded fold artifacts and produce submission.csv.

Usage:
    python work/make_submission.py --probs-dir kernels/rmit-sgs-ae-do-toc-c1/out --tags f0_1_2 f3_4
    python work/make_submission.py --probs-dir kernels/rmit-sgs-ae-do-toc-c1/out --tags all

Does four things:
  1. Merges per-shard OOF matrices (disjoint folds) and fold-averaged test probabilities.
  2. Reports the merged OOF macro-F1 plus per-class, per-language and BENIGN<->JAILBREAK detail.
  3. Fits per-class multipliers HONESTLY: tuned on 4 folds, scored on the held-out 5th.
  4. Writes submission.csv with the same pre-flight assertions the teammate notebook uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}


def load_shard(probs_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
    oof = np.load(probs_dir / f"oof_probs_{tag}.npy")
    test = np.load(probs_dir / f"test_probs_{tag}.npy")
    with open(probs_dir / f"metrics_{tag}.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return oof, test, metrics


def merge(tags: list[str], probs_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    oof_total: np.ndarray | None = None
    test_accumulator: np.ndarray | None = None
    fold_counts: list[int] = []

    for tag in tags:
        oof, test, metrics = load_shard(probs_dir, tag)
        n_folds = max(len(metrics.get("folds_run", [])), 1)
        fold_counts.append(n_folds)
        if oof_total is None:
            oof_total = np.zeros_like(oof)
            test_accumulator = np.zeros_like(test)
        assert test_accumulator is not None
        covered = oof.sum(axis=1) > 0
        overlap = covered & (oof_total.sum(axis=1) > 0)
        if overlap.any():
            raise ValueError(f"shard '{tag}' overlaps previous shards on {int(overlap.sum())} rows")
        oof_total[covered] = oof[covered]
        test_accumulator += test * n_folds
        print(f"shard {tag}: folds={metrics.get('folds_run')} covered_rows={int(covered.sum())} "
              f"shard_macro_f1={metrics.get('oof_macro_f1', float('nan')):.4f}")

    assert test_accumulator is not None and oof_total is not None
    return oof_total, test_accumulator / sum(fold_counts)


def tune_multipliers(probs: np.ndarray, reference: np.ndarray, rounds: int = 3) -> tuple[np.ndarray, float]:
    """Coordinate ascent over per-class multipliers to maximise macro-F1."""
    grid = np.round(np.arange(0.6, 2.51, 0.1), 2)
    multipliers = np.ones(probs.shape[1])
    best = f1_score(reference, (probs * multipliers).argmax(axis=1), average="macro")
    for _ in range(rounds):
        improved = False
        for class_id in range(probs.shape[1]):
            for candidate in grid:
                trial = multipliers.copy()
                trial[class_id] = candidate
                score = f1_score(reference, (probs * trial).argmax(axis=1), average="macro")
                if score > best + 1e-6:
                    best, multipliers, improved = score, trial, True
        if not improved:
            break
    return multipliers, best


def honest_tuning_gain(
    probs: np.ndarray, reference: np.ndarray, fold_ids: np.ndarray
) -> tuple[float, float]:
    """Tune on folds!=k, score on fold k. Returns (baseline, tuned) held-out means."""
    base_scores, tuned_scores = [], []
    for fold in sorted(set(fold_ids.tolist())):
        is_held_out = fold_ids == fold
        train_mask = ~is_held_out
        multipliers, _ = tune_multipliers(probs[train_mask], reference[train_mask])
        base_scores.append(f1_score(reference[is_held_out], probs[is_held_out].argmax(axis=1), average="macro"))
        tuned = (probs[is_held_out] * multipliers).argmax(axis=1)
        tuned_scores.append(f1_score(reference[is_held_out], tuned, average="macro"))
    return float(np.mean(base_scores)), float(np.mean(tuned_scores))


def per_class_f1(reference: np.ndarray, prediction: np.ndarray, class_id: int) -> float:
    """F1 of a single class, written explicitly to avoid averaged-call ambiguity."""
    true_positive = float(np.sum((prediction == class_id) & (reference == class_id)))
    false_positive = float(np.sum((prediction == class_id) & (reference != class_id)))
    false_negative = float(np.sum((prediction != class_id) & (reference == class_id)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator > 0 else 0.0


def report(oof: np.ndarray, frame: pd.DataFrame) -> None:
    reference = np.array([LABEL_TO_ID[name] for name in frame.label])
    prediction = oof.argmax(axis=1)
    covered = oof.sum(axis=1) > 0

    print(f"\nOOF macro-F1 (all covered rows): {f1_score(reference[covered], prediction[covered], average='macro'):.4f}")

    covered_reference = reference[covered]
    covered_prediction = prediction[covered]
    per_class = {
        name: per_class_f1(covered_reference, covered_prediction, class_id)
        for name, class_id in LABEL_TO_ID.items()
    }
    print("per-class F1:", {name: round(score, 4) for name, score in per_class.items()})

    print("\nper-language macro-F1:")
    languages = frame.language.to_numpy()
    for language in sorted(set(languages.tolist())):
        mask = covered & (languages == language)
        print(f"  {language}: {f1_score(reference[mask], prediction[mask], average='macro'):.4f} (n={int(mask.sum())})")

    matrix = pd.crosstab(
        pd.Categorical([LABELS[i] for i in reference[covered]], categories=LABELS),
        pd.Categorical([LABELS[i] for i in prediction[covered]], categories=LABELS),
    )
    matrix.index.name = "true"
    matrix.columns.name = "pred"
    print("\nconfusion:\n", matrix.to_string())
    benign, jailbreak = LABEL_TO_ID["BENIGN"], LABEL_TO_ID["JAILBREAK"]
    cross = int(((reference == benign) & (prediction == jailbreak) & covered).sum())
    cross += int(((reference == jailbreak) & (prediction == benign) & covered).sum())
    total_errors = int(((reference != prediction) & covered).sum())
    print(f"\nBENIGN<->JAILBREAK errors: {cross} of {total_errors} total errors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--out", type=Path, default=Path("work/submission.csv"))
    args = parser.parse_args()

    train_frame = pd.read_csv(args.data_dir / "train.csv")
    test_frame = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    folds_path = args.probs_dir / "folds.csv"
    oof, test = merge(args.tags, args.probs_dir)
    report(oof, train_frame)

    reference = np.array([LABEL_TO_ID[name] for name in train_frame.label])
    covered = oof.sum(axis=1) > 0
    if folds_path.exists():
        fold_ids = pd.read_csv(folds_path).fold.to_numpy()
        baseline, tuned = honest_tuning_gain(oof[covered], reference[covered], fold_ids[covered])
        print(f"\nhonest (4-fold-tune / held-out-score): {baseline:.4f} -> {tuned:.4f} "
              f"(gain {tuned - baseline:+.4f})")
        use_tuning = tuned > baseline + 0.002
    else:
        print("\nno folds.csv alongside the probabilities - skipping honest tuning check")
        use_tuning = True

    multipliers, full_fit = tune_multipliers(oof[covered], reference[covered])
    print(f"multipliers fitted on all OOF: {multipliers.round(3).tolist()} (in-sample {full_fit:.4f})")
    if not use_tuning:
        print("tuning gain did not survive held-out scoring -> using raw argmax")
        multipliers = np.ones(3)

    prediction = (test * multipliers).argmax(axis=1)
    submission = sample.copy()
    id_column, label_column = submission.columns[0], submission.columns[1]
    submission[label_column] = [LABELS[index] for index in prediction]

    assert len(submission) == 6000, f"expected 6000 rows, got {len(submission)}"
    assert set(submission[label_column]) <= set(LABELS), "unexpected label"
    assert list(submission[id_column]) == list(test_frame[test_frame.columns[0]]), "id order mismatch"
    assert submission[label_column].isna().sum() == 0, "missing labels"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("predicted distribution:")
    print(submission[label_column].value_counts(normalize=True).round(4).to_string())
    print("\ntrain distribution for comparison:")
    print(train_frame.label.value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
