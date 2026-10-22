"""Probe 2: verify model attachment, competition paths and token-length budget."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def list_inputs() -> None:
    for pattern in ("/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/models/*/*"):
        print(f"--- {pattern} ---")
        for path in sorted(glob.glob(pattern)):
            print(" ", path)


def token_stats(train_csv: str, tokenizer_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train = pd.read_csv(train_csv)
    counts = np.array([
        len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        for text in train.text.tolist()
    ])
    print("token length percentiles:", np.percentile(counts, [50, 75, 90, 95, 98, 99, 99.9]).round(1))
    print("max:", counts.max())
    for max_len in (128, 192, 256, 320, 384, 512):
        print(f"  max_len={max_len}: coverage={(counts <= max_len).mean():.4f}")


def main() -> None:
    list_inputs()
    train_csv = "/kaggle/input/competitions/rmit-hackathon-2026/train.csv"
    print("cwd:", os.getcwd())
    print("train_csv exists:", os.path.exists(train_csv))
    candidates = glob.glob("/kaggle/input/models/*/*/transformers/default/*")
    print("model candidates:", candidates)
    if candidates:
        token_stats(train_csv, candidates[0])


if __name__ == "__main__":
    main()
