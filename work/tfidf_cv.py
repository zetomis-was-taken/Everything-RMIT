"""Honest 5-fold CPU floor for Challenge 1, matching the folds the GPU runs use.

Reports macro-F1 plus the decomposition that matters (per-class, per-language,
BENIGN<->JAILBREAK error share), and ablates the language prefix on the cheap model
so we know whether that idea has any signal before spending GPU time on it.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

DATA = "data"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
SEED = 42
N_FOLDS = 5


def per_class_f1(reference: np.ndarray, prediction: np.ndarray, class_id: int) -> float:
    true_positive = float(np.sum((prediction == class_id) & (reference == class_id)))
    false_positive = float(np.sum((prediction == class_id) & (reference != class_id)))
    false_negative = float(np.sum((prediction != class_id) & (reference == class_id)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator > 0 else 0.0


def texts_with_prefix(frame: pd.DataFrame, use_prefix: bool) -> list[str]:
    texts = frame["text"].astype(str).tolist()
    if not use_prefix:
        return texts
    languages = frame["language"].astype(str).tolist()
    return [f"[{language}] {text}" for language, text in zip(languages, texts, strict=True)]


def run_cv(frame: pd.DataFrame, use_prefix: bool) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([LABEL_TO_ID[name] for name in frame.label])
    oof = np.zeros((len(frame), len(LABELS)), dtype=np.float32)
    texts = texts_with_prefix(frame, use_prefix)

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_index, valid_index) in enumerate(splitter.split(frame, frame.label)):
        train_texts = [texts[i] for i in train_index]
        valid_texts = [texts[i] for i in valid_index]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
            lowercase=True,
        )
        train_matrix = cast(csr_matrix, vectorizer.fit_transform(train_texts))
        valid_matrix = cast(csr_matrix, vectorizer.transform(valid_texts))

        classifier: Any = LinearSVC(C=0.5, class_weight="balanced")
        classifier.fit(train_matrix, reference[train_index])
        decision = classifier.decision_function(valid_matrix)
        shifted = decision - decision.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        oof[valid_index] = probabilities / probabilities.sum(axis=1, keepdims=True)
        print(f"  fold {fold}: macro_f1={f1_score(reference[valid_index], oof[valid_index].argmax(1), average='macro'):.4f}")

    return oof, reference


def report(name: str, oof: np.ndarray, reference: np.ndarray, frame: pd.DataFrame) -> float:
    prediction = oof.argmax(axis=1)
    macro = float(f1_score(reference, prediction, average="macro"))
    print(f"\n=== {name}: OOF macro-F1 = {macro:.4f} ===")
    per_class = {label: round(per_class_f1(reference, prediction, LABEL_TO_ID[label]), 4) for label in LABELS}
    print("per-class F1:", per_class)

    matrix = pd.crosstab(
        pd.Categorical([LABELS[i] for i in reference], categories=LABELS),
        pd.Categorical([LABELS[i] for i in prediction], categories=LABELS),
    )
    matrix.index.name = "true"
    matrix.columns.name = "pred"
    print("OOF confusion (every row predicted by a model that never saw it):")
    print(matrix.to_string())

    languages = frame["language"].to_numpy()
    for language in sorted(set(languages.tolist())):
        mask = languages == language
        print(f"  {language}: {f1_score(reference[mask], prediction[mask], average='macro'):.4f} (n={int(mask.sum())})")

    benign, jailbreak = LABEL_TO_ID["BENIGN"], LABEL_TO_ID["JAILBREAK"]
    cross = int(((reference == benign) & (prediction == jailbreak)).sum())
    cross += int(((reference == jailbreak) & (prediction == benign)).sum())
    total_errors = int((reference != prediction).sum())
    print(f"  BENIGN<->JAILBREAK errors: {cross} of {total_errors} ({cross / max(total_errors, 1):.1%})")
    return macro


def main() -> None:
    frame = pd.read_csv(f"{DATA}/train.csv")
    results: dict[str, float] = {}

    print("baseline (raw text):")
    oof_plain, reference = run_cv(frame, use_prefix=False)
    results["raw text"] = report("raw text", oof_plain, reference, frame)

    print("\nwith language prefix:")
    oof_prefixed, _ = run_cv(frame, use_prefix=True)
    results["with [lang] prefix"] = report("with [lang] prefix", oof_prefixed, reference, frame)

    delta = results["with [lang] prefix"] - results["raw text"]
    print(f"\nlanguage-prefix delta on the cheap model: {delta:+.4f}")
    print("verdict:", "worth an arm on the transformer" if delta > 0.003 else "no signal - skip on GPU")

    try:
        np.save(f"{DATA}/../oof_probs_tfidf.npy", oof_plain)
        print("saved OOF probabilities for reuse: work/oof_probs_tfidf.npy")
    except OSError as error:
        print(f"could not save OOF probabilities: {error}")


if __name__ == "__main__":
    main()
