"""CPU-only TF-IDF baseline: establishes a local macro-F1 floor for Challenge 1."""

from __future__ import annotations

import time
from typing import Any, cast

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

DATA = "data"


def vectorize(train_text: Any, valid_text: Any) -> tuple[csr_matrix, csr_matrix, int]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=300_000,
        sublinear_tf=True,
        lowercase=True,
    )
    train_matrix = cast(csr_matrix, vectorizer.fit_transform(train_text))
    valid_matrix = cast(csr_matrix, vectorizer.transform(valid_text))
    return train_matrix, valid_matrix, len(vectorizer.vocabulary_)


def main() -> None:
    train = pd.read_csv(f"{DATA}/train.csv")
    split = train_test_split(train, test_size=0.15, stratify=train.label, random_state=42)
    train_frame = cast(pd.DataFrame, split[0])
    valid_frame = cast(pd.DataFrame, split[1])
    train_label = train_frame.label
    valid_label = valid_frame.label

    started = time.time()
    train_matrix, valid_matrix, n_features = vectorize(train_frame.text, valid_frame.text)
    print(f"vectorized: n_rows={len(train_frame)} n_features={n_features} in {time.time() - started:.1f}s")

    classifiers: list[tuple[str, Any]] = [
        ("logreg", LogisticRegression(C=4, max_iter=2000, class_weight="balanced", n_jobs=-1)),
        ("linearsvc", LinearSVC(C=0.5, class_weight="balanced")),
    ]
    for name, classifier in classifiers:
        started = time.time()
        classifier.fit(train_matrix, train_label)
        predicted = classifier.predict(valid_matrix)
        macro_f1 = f1_score(valid_label, predicted, average="macro")
        accuracy = float((predicted == valid_label).mean())
        print(f"{name}: macro_f1={macro_f1:.4f} acc={accuracy:.4f} in {time.time() - started:.1f}s")
        print(classification_report(valid_label, predicted, digits=3))


if __name__ == "__main__":
    main()
