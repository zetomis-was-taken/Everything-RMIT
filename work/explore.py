"""Step 1 - data understanding for Challenge 1 (no GPU, no submissions).

Goals:
  1. What separates the three labels empirically (length, discriminative cues).
  2. Whether the corpus is machine-translated parallel data (cross-language duplicates).
  3. Where a cheap lexical model fails, i.e. which boundary is genuinely semantic.
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

DATA = "data"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def length_by_label(train: pd.DataFrame) -> None:
    section("1. Text length and token-ish size by label")
    train = train.assign(chars=train.text.str.len(), words=train.text.str.split().str.len())
    stats = train.groupby("label")[["chars", "words"]].describe(percentiles=[0.5, 0.9, 0.99])
    print(stats.round(1).T.to_string())


def samples(train: pd.DataFrame) -> None:
    section("2. Samples per label (one per language)")
    for label in LABELS:
        print(f"\n### {label}")
        subset = cast(pd.DataFrame, train[train.label == label])
        languages_in_split = subset["language"].to_numpy()
        texts = subset["text"]
        for language in ["pt", "fr", "it", "en"]:
            positions = np.flatnonzero(languages_in_split == language)
            if positions.size == 0:
                continue
            text = re.sub(r"\s+", " ", str(texts.iloc[int(positions[0])]))
            print(f"  [{language}] {text[:260]}...")


def discriminative_cues(train: pd.DataFrame, top_k: int = 18) -> None:
    section("3. Discriminative word cues per label (log-odds with prior smoothing)")
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=20, max_features=60_000,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )
    matrix = cast(csr_matrix, vectorizer.fit_transform(train.text))
    terms = np.array(vectorizer.get_feature_names_out())
    sparse_binary = cast(Any, matrix > 0)
    binary = cast(csr_matrix, sparse_binary.astype(np.float32))

    for label in LABELS:
        mask = (train.label == label).to_numpy()
        in_class = np.asarray(cast(Any, binary[mask]).sum(axis=0)).ravel()
        out_class = np.asarray(cast(Any, binary[~mask]).sum(axis=0)).ravel()
        # smoothed log-odds ratio
        score = np.log((in_class + 1) / (mask.sum() + 2)) - np.log((out_class + 1) / ((~mask).sum() + 2))
        order = np.argsort(-score)[:top_k]
        print(f"\n{label}: " + ", ".join(f"{terms[i]}({in_class[i]:.0f})" for i in order))


def near_duplicates(train: pd.DataFrame) -> None:
    section("4. Is the corpus machine-translated parallel data? (cross-language neighbours)")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=200_000, sublinear_tf=True
    )
    matrix = cast(csr_matrix, vectorizer.fit_transform(train.text.str.slice(0, 400)))
    norms = np.sqrt(np.asarray(cast(Any, matrix.multiply(matrix)).sum(axis=1))).ravel()
    normed = cast(csr_matrix, matrix.multiply(1.0 / np.maximum(norms, 1e-9)[:, None])).tocsr()

    sample_idx = np.random.default_rng(0).choice(len(train), 400, replace=False)
    similarity = np.asarray((normed[sample_idx] @ normed.T).todense())
    languages = train.language.to_numpy()
    labels = train.label.to_numpy()

    same_lang, cross_lang, best_sim = 0, 0, []
    for row, idx in enumerate(sample_idx):
        scores = similarity[row].copy()
        scores[idx] = -1.0
        nearest = int(np.argmax(scores))
        best_sim.append(scores[nearest])
        if languages[nearest] == languages[idx]:
            same_lang += 1
        else:
            cross_lang += 1
    best_sim_arr = np.array(best_sim)
    print(f"nearest neighbour is same language: {same_lang}/400, cross language: {cross_lang}/400")
    print(f"cosine to nearest neighbour: p50={np.percentile(best_sim_arr, 50):.3f}"
          f" p90={np.percentile(best_sim_arr, 90):.3f} max={best_sim_arr.max():.3f}")

    # do cross-language neighbours share the label?
    agree = []
    for row, idx in enumerate(sample_idx):
        scores = similarity[row].copy()
        scores[idx] = -1.0
        other = languages != languages[idx]
        scores[~other] = -1.0
        nearest = int(np.argmax(scores))
        agree.append(labels[nearest] == labels[idx])
    print(f"cross-language nearest neighbour shares the label: {np.mean(agree):.3f}")


def confusion_analysis(train: pd.DataFrame) -> None:
    section("5. Where does the cheap lexical model fail? (hold-out confusion)")
    split = train_test_split(train, test_size=0.15, stratify=train.label, random_state=42)
    train_frame = cast(pd.DataFrame, split[0])
    valid_frame = cast(pd.DataFrame, split[1])
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True
    )
    train_matrix = cast(csr_matrix, vectorizer.fit_transform(train_frame.text))
    valid_matrix = cast(csr_matrix, vectorizer.transform(valid_frame.text))
    model: Any = LinearSVC(C=0.5, class_weight="balanced").fit(train_matrix, train_frame.label)
    predicted = model.predict(valid_matrix)

    print(classification_report(valid_frame.label, predicted, digits=3))
    matrix = confusion_matrix(valid_frame.label, predicted, labels=LABELS)
    frame = pd.DataFrame(
        matrix,
        index=pd.Index([f"true_{name}" for name in LABELS]),
        columns=pd.Index([f"pred_{name}" for name in LABELS]),
    )
    print(frame.to_string())

    print("\nper-language macro-F1:")
    for language in ["pt", "fr", "it", "en"]:
        mask = (valid_frame.language == language).to_numpy()
        score = f1_score(valid_frame.label[mask], predicted[mask], average="macro")
        print(f"  {language}: {score:.4f}  (n={mask.sum()})")

    print("\nper-language mean text length (train):")
    print(train.groupby("language").text.apply(lambda s: float(s.str.len().mean())).round(0).to_string())


def main() -> None:
    train = pd.read_csv(f"{DATA}/train.csv")
    length_by_label(train)
    samples(train)
    discriminative_cues(train)
    near_duplicates(train)
    confusion_analysis(train)


if __name__ == "__main__":
    main()
